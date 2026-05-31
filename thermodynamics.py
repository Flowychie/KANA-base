"""Thermodynamic engine: hard constraints, consistency relations, and property calculations."""

import jax
import jax.numpy as jnp
from jax import vmap, grad

from .config import Config


class ThermodynamicEngine:
    """Computes thermodynamic properties and enforces physical consistency via automatic differentiation."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.R = cfg.R_GAS

    def predict_gE(
        self,
        params: dict,
        apply_fn: callable,
        sigmas: jnp.ndarray,
        scalars: jnp.ndarray,
        mask: jnp.ndarray,
        T: jnp.ndarray,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Predict excess Gibbs energy g^E (J/mol)."""
        gE_RT = apply_fn(
            {"params": params}, sigmas, scalars, mask, T, x, training=False
        )
        return gE_RT * self.R * T

    def compute_ln_gamma(
        self,
        params: dict,
        apply_fn: callable,
        sigmas: jnp.ndarray,
        scalars: jnp.ndarray,
        mask: jnp.ndarray,
        T: jnp.ndarray,
        n: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute ln(γ_i) via automatic differentiation of n·g^E/RT."""
        def G_excess_over_RT(n_vec, T_scalar, mask_vec, sigma_vec, scalar_vec):
            n_tot = jnp.sum(n_vec * mask_vec)
            x_vec = (n_vec * mask_vec) / jnp.maximum(n_tot, 1e-10)
            gE_RT = apply_fn(
                {"params": params},
                sigma_vec[None, ...],
                scalar_vec[None, ...],
                mask_vec[None, ...],
                jnp.array([T_scalar]),
                x_vec[None, ...],
                training=False,
            )[0]
            return n_tot * gE_RT

        batch_grad = vmap(
            lambda ni, Ti, mi, si, ci: grad(G_excess_over_RT)(ni, Ti, mi, si, ci),
            in_axes=(0, 0, 0, 0, 0),
        )
        return batch_grad(n, T, mask, sigmas, scalars)

    def compute_hE(
        self,
        params: dict,
        apply_fn: callable,
        sigmas: jnp.ndarray,
        scalars: jnp.ndarray,
        mask: jnp.ndarray,
        T: jnp.ndarray,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute excess enthalpy h^E via the Gibbs-Helmholtz relation."""
        def gE_over_T_scalar(T_scalar, x_vec, mask_vec, sigma_vec, scalar_vec):
            gE = self.predict_gE(
                params,
                apply_fn,
                sigma_vec[None, ...],
                scalar_vec[None, ...],
                mask_vec[None, ...],
                jnp.array([T_scalar]),
                x_vec[None, ...],
            )[0]
            return gE / T_scalar

        batch_grad = vmap(
            lambda Ti, xi, mi, si, ci: -Ti**2
            * grad(gE_over_T_scalar)(Ti, xi, mi, si, ci),
            in_axes=(0, 0, 0, 0, 0),
        )
        return batch_grad(T, x, mask, sigmas, scalars)

    def compute_sE(
        self,
        params: dict,
        apply_fn: callable,
        sigmas: jnp.ndarray,
        scalars: jnp.ndarray,
        mask: jnp.ndarray,
        T: jnp.ndarray,
        x: jnp.ndarray,
        n: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute excess entropy s^E from g^E and h^E."""
        gE = self.predict_gE(params, apply_fn, sigmas, scalars, mask, T, x)
        hE = self.compute_hE(params, apply_fn, sigmas, scalars, mask, T, x)
        return (hE - gE) / T

    def gibbs_duhem_residual(
        self,
        ln_gamma: jnp.ndarray,
        x: jnp.ndarray,
        mask: jnp.ndarray,
        gE_RT: jnp.ndarray,
    ) -> jnp.ndarray:
        """Gibbs-Duhem consistency residual: Σ x_i ln(γ_i) − g^E/RT."""
        x_masked = x * mask.astype(jnp.float32)
        return jnp.sum(x_masked * ln_gamma, axis=-1) - gE_RT
