"""Configuration dataclass and preset variant definitions for KANA."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Config:
    SIGMA_DIM: int = 51
    SIGMA_CHANNELS: int = 1
    SCALAR_DIM: int = 10
    LATENT_A: int = 256
    LATENT_B: int = 256
    LATENT_Z: int = LATENT_A + LATENT_B
    MAX_COMPONENTS: int = 10
    AGGREGATOR: str = 'molefrac_weighted'
    T_REF: float = 500.0
    R_GAS: float = 8.314
    GE_HEAD_HIDDEN: int = 1024
    LR: float = 1e-3
    LR_SCHEDULE: bool = True
    DROPOUT_RATE: float = 0.15
    SEED: int = 42
    W_LN_GAMMA: float = 1.0
    W_GAMMA_INF: float = 0.5
    W_HE: float = 0.3
    W_SE: float = 0.3
    W_BOUNDARY: float = 0.1


# 2x2 grid of presets: latent_size x learning_rate
PRESETS: Dict[str, Config] = {
    '1': Config(LATENT_A=256, LATENT_B=256, LATENT_Z=512,  LR=1e-3),
    '2': Config(LATENT_A=512, LATENT_B=512, LATENT_Z=1024, LR=1e-3),
    '3': Config(LATENT_A=256, LATENT_B=256, LATENT_Z=512,  LR=1e-4),
    '4': Config(LATENT_A=512, LATENT_B=512, LATENT_Z=1024, LR=1e-4),
}
