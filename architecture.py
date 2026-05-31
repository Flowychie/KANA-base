"""Neural network architecture: encoders, permutation-invariant aggregator, and prediction head."""

import jax.numpy as jnp
from flax import linen as nn

from .config import Config


class SigmaEncoder(nn.Module):
    """1-D CNN encoder for sigma profiles."""

    cfg: Config

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = nn.Conv(features=32, kernel_size=(5,), padding="SAME")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        x = nn.Conv(features=64, kernel_size=(3,), padding="SAME")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        x = nn.Conv(features=128, kernel_size=(3,), padding="SAME")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # Global average pooling over the sigma dimension
        x = jnp.mean(x, axis=1)
        x = nn.Dense(self.cfg.LATENT_A)(x)
        x = nn.relu(x)
        return x


class ScalarEncoder(nn.Module):
    """MLP encoder for scalar molecular descriptors."""

    cfg: Config

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = nn.Dense(128)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.cfg.DROPOUT_RATE, deterministic=not training)(x)

        x = nn.Dense(128)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.cfg.DROPOUT_RATE, deterministic=not training)(x)

        x = nn.Dense(self.cfg.LATENT_B)(x)
        x = nn.relu(x)
        return x


class ComponentEncoder(nn.Module):
    """Fuses sigma and scalar embeddings into a single component vector z_i."""

    cfg: Config

    @nn.compact
    def __call__(
        self, sigma: jnp.ndarray, scalar: jnp.ndarray, training: bool = True
    ) -> jnp.ndarray:
        z_a = SigmaEncoder(cfg=self.cfg)(sigma, training=training)
        z_b = ScalarEncoder(cfg=self.cfg)(scalar, training=training)
        return jnp.concatenate([z_a, z_b], axis=-1)


class MoleFractionWeightedAggregator(nn.Module):
    """True permutation-invariant aggregation.

    z_mixture = Σ(x_i * z_i) / Σ(x_i)

    Composition is fused BEFORE aggregation. The downstream MLP never sees raw x_i.
    Swapping component order: z_i and x_i swap together → weighted sum unchanged.
    """

    cfg: Config
    output_dim: int = 128

    @nn.compact
    def __call__(
        self, z_components: jnp.ndarray, x: jnp.ndarray, mask: jnp.ndarray
    ) -> jnp.ndarray:
        mask_f = mask.astype(jnp.float32)
        x_masked = x * mask_f  # (batch, max_n)

        # Weighted sum: Σ (x_i * z_i)
        weights = x_masked[:, :, None]  # (batch, max_n, 1)
        z_weighted = z_components * weights
        z_sum = jnp.sum(z_weighted, axis=1)

        # Normalize by total mole fraction
        x_total = jnp.sum(x_masked, axis=1, keepdims=True)
        z_pooled = z_sum / jnp.maximum(x_total, 1e-10)

        # ρ: MLP projection
        h = nn.Dense(128)(z_pooled)
        h = nn.relu(h)
        h = nn.Dense(128)(h)
        h = nn.relu(h)
        return nn.Dense(self.output_dim)(h)


class GE_PredictionHead(nn.Module):
    """Predicts g^E / (R*T) from mixture embedding and temperature.

    Accepts only z_mixture and T. Composition is already encoded inside z_mixture
    via the weighted aggregator. The pairwise boundary constraint ensures
    g^E vanishes at pure-component limits for any mixture order.
    """

    cfg: Config

    @nn.compact
    def __call__(
        self, z_mixture: jnp.ndarray, T: jnp.ndarray, x: jnp.ndarray, mask: jnp.ndarray
    ) -> jnp.ndarray:
        # Temperature features
        T_norm = T / self.cfg.T_REF
        T_feats = jnp.stack(
            [
                T_norm,
                1.0 / jnp.maximum(T_norm, 0.1),
                jnp.log(jnp.maximum(T_norm, 0.1)),
            ],
            axis=-1,
        )

        # HANYA z_mixture + T. No raw composition here.
        features = jnp.concatenate([z_mixture, T_feats], axis=-1)

        h = nn.Dense(self.cfg.GE_HEAD_HIDDEN)(features)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)
        h = nn.Dropout(rate=self.cfg.DROPOUT_RATE, deterministic=True)(h)

        h = nn.Dense(256)(h)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)
        h = nn.Dropout(rate=self.cfg.DROPOUT_RATE, deterministic=True)(h)

        h = nn.Dense(64)(h)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)

        gE_raw = nn.Dense(1)(h).squeeze(-1)  # (batch,)

        # =====================================================================
        # PAIRWISE BOUNDARY CONSTRAINT (Thermodynamic Way)
        # =====================================================================
        # Σ_{i<j} x_i * x_j  instead of  Π x_i
        # Binary: x1*x2. Quaternary: x1*x2 + x1*x3 + ... + x3*x4
        # Pure component → sum = 0. Scale is O(1) regardless of n.
        # =====================================================================
        mask_f = mask.astype(jnp.float32)
        x_masked = x * mask_f
        max_n = x.shape[-1]

        x_i = x_masked[:, :, None]  # (batch, max_n, 1)
        x_j = x_masked[:, None, :]  # (batch, 1, max_n)
        pairwise = x_i * x_j  # (batch, max_n, max_n)

        # Upper triangular: i < j
        triu = jnp.triu(jnp.ones((max_n, max_n)), k=1)[None, ...]
        pairwise_sum = jnp.sum(pairwise * triu, axis=(1, 2))  # (batch,)

        gE_RT = gE_raw * pairwise_sum
        return gE_RT


class HardConstrainedCINN(nn.Module):
    """Complete hard-constrained chemical-informed neural network."""

    cfg: Config
    use_binary_projection: bool = False  # Reserved for future binary-specific head

    @nn.compact
    def __call__(
        self,
        sigma_profiles: jnp.ndarray,
        scalar_features: jnp.ndarray,
        mask: jnp.ndarray,
        T: jnp.ndarray,
        x: jnp.ndarray,
        training: bool = True,
    ) -> jnp.ndarray:
        batch_size, max_n = sigma_profiles.shape[:2]

        # Part 1: Shared component encoding
        sigmas_flat = sigma_profiles.reshape(
            -1, self.cfg.SIGMA_DIM, self.cfg.SIGMA_CHANNELS
        )
        scalars_flat = scalar_features.reshape(-1, self.cfg.SCALAR_DIM)
        z_flat = ComponentEncoder(cfg=self.cfg)(
            sigmas_flat, scalars_flat, training=training
        )
        z_components = z_flat.reshape(batch_size, max_n, self.cfg.LATENT_Z)

        # Part 2: Mole-fraction weighted aggregation (TRUE permutation invariance)
        z_mixture = MoleFractionWeightedAggregator(cfg=self.cfg, output_dim=128)(
            z_components, x, mask
        )

        # Part 3: g^E prediction
        gE_RT = GE_PredictionHead(cfg=self.cfg)(z_mixture, T, x, mask)

        return gE_RT
