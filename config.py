"""Global configuration and hyperparameters for the KANA CINN."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Immutable configuration container for the Hard-Constrained CINN."""

    # Sigma profile dimensions
    SIGMA_DIM: int = 51
    SIGMA_CHANNELS: int = 1

    # Scalar molecular descriptor dimensions
    SCALAR_DIM: int = 10

    # Latent space dimensions
    LATENT_A: int = 128
    LATENT_B: int = 128
    LATENT_Z: int = LATENT_A + LATENT_B  # Concatenated component embedding

    # Mixture constraints
    MAX_COMPONENTS: int = 5
    AGGREGATOR: str = "molefrac_weighted"

    # Thermodynamic reference constants
    T_REF: float = 500.0
    R_GAS: float = 8.314

    # Network architecture
    GE_HEAD_HIDDEN: int = 512
    DROPOUT_RATE: float = 0.15

    # Optimization
    LR: float = 1e-3
    LR_SCHEDULE: bool = True

    # Reproducibility
    SEED: int = 42

    # Multi-objective loss weights
    W_LN_GAMMA: float = 1.0
    W_GAMMA_INF: float = 0.5
    W_HE: float = 0.3
    W_SE: float = 0.3
    # W_BOUNDARY: float = 0.0
