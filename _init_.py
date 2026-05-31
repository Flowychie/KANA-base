"""KANA: Chemical-Informed Neural Network with Hard Thermodynamic Constraints."""

from .config import Config
from .architecture import (
    HardConstrainedCINN,
    ComponentEncoder,
    SigmaEncoder,
    ScalarEncoder,
    MoleFractionWeightedAggregator,
    GE_PredictionHead,
)
from .thermodynamics import ThermodynamicEngine
from .dataset import ThermoDataset, load_raw_data, standardize_data
from .train import (
    create_train_state,
    loss_fn,
    train_step,
    eval_step,
    evaluate_model,
    generate_thermo_verification_report,
    export_parity_data,
    main,
)

__version__ = "0.1.0"

__all__ = [
    "Config",
    "HardConstrainedCINN",
    "ComponentEncoder",
    "SigmaEncoder",
    "ScalarEncoder",
    "MoleFractionWeightedAggregator",
    "GE_PredictionHead",
    "ThermodynamicEngine",
    "ThermoDataset",
    "load_raw_data",
    "standardize_data",
    "create_train_state",
    "loss_fn",
    "train_step",
    "eval_step",
    "evaluate_model",
    "generate_thermo_verification_report",
    "export_parity_data",
    "main",
]
