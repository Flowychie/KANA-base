"""
KANA: Hard-Constrained Chemical-Informed Neural Network
=======================================================

Multi-component excess Gibbs energy prediction with automatic satisfaction of
Gibbs-Duhem, Gibbs-Helmholtz, and pure-component boundary constraints.

Usage:
    python -m kana.main --csv-path data/dataset.csv --preset 1

    from kana import Config, HardConstrainedCINN, ThermodynamicEngine, train
    from kana.config import PRESETS

    cfg = PRESETS['1']
    state, engine, model, val_data = train(cfg, 'data/dataset.csv', 'outputs')
"""

from .config import Config, PRESETS
from .architecture import HardConstrainedCINN
from .thermodynamics import ThermodynamicEngine
from .main import train

__all__ = [
    'Config', 'PRESETS',
    'HardConstrainedCINN', 'ThermodynamicEngine',
    'train',
]
