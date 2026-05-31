"""Data loading, standardization, and batching utilities."""

import numpy as np
import jax.numpy as jnp
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import Config


class ThermoDataset:
    """Lightweight dataset wrapper for thermodynamic data."""

    def __init__(self, data_dict: dict, cfg: Config):
        self.data = data_dict
        self.cfg = cfg
        self.N = len(data_dict["T"])

    def get_batch(self, indices: np.ndarray) -> dict:
        """Return a dictionary of JAX arrays for the given indices."""
        return {k: jnp.array(v[indices]) for k, v in self.data.items()}


def load_raw_data(csv_path: str, cfg: Config) -> dict:
    """Load raw CSV data into numpy arrays without standardization.

    Expected CSV columns:
        n_components, T_K,
        sigma_{i}, HOMO_{i}, LUMO_{i}, Dipole_{i},
        M0_{i}, M1_{i}, M2_{i}, M3_{i}, M4_{i},
        x_{i}, n_{i}, ln_gamma_{i}
    for i = 0 .. MAX_COMPONENTS-1.
    """
    df = pd.read_csv(csv_path)
    N = len(df)
    max_n = cfg.MAX_COMPONENTS

    sigmas = np.zeros((N, max_n, cfg.SIGMA_DIM, 1), dtype="float32")
    scalars = np.zeros((N, max_n, cfg.SCALAR_DIM), dtype="float32")
    mask = np.zeros((N, max_n), dtype=bool)
    x = np.zeros((N, max_n), dtype="float32")
    n = np.zeros((N, max_n), dtype="float32")
    ln_gamma = np.zeros((N, max_n), dtype="float32")

    for idx, row in df.iterrows():
        n_comp = int(row["n_components"])
        mask[idx, :n_comp] = True
        for i in range(n_comp):
            sigmas[idx, i, :, 0] = np.fromstring(row[f"sigma_{i}"], sep=",")
            scalars[idx, i] = np.array([
                row[f"HOMO_{i}"], row[f"LUMO_{i}"], row[f"Dipole_{i}"],
                row[f"M0_{i}"], row[f"M1_{i}"], row[f"M2_{i}"],
                row[f"M3_{i}"], row[f"M4_{i}"], 0.0, 0.0
            ])
            x[idx, i] = row[f"x_{i}"]
            n[idx, i] = row[f"n_{i}"]
            ln_gamma[idx, i] = row[f"ln_gamma_{i}"]

    data = {
        "sigma_profiles": sigmas,
        "scalar_features": scalars,
        "mask": mask,
        "T": df["T_K"].values.astype("float32"),
        "x": x,
        "n": n,
        "ln_gamma_target": ln_gamma,
    }
    return data


def standardize_data(data: dict, train_indices: np.ndarray, cfg: Config) -> dict:
    """Fit standardizers on train data only, then transform the full dataset.

    FIX 1: Prevents data leakage by fitting StandardScaler exclusively on
    the training split.
    """
    scalars = data["scalar_features"]
    mask = data["mask"]
    N, max_n = scalars.shape[:2]

    scalars_2d = scalars.reshape(-1, cfg.SCALAR_DIM)
    mask_1d = mask.reshape(-1)

    # --- Scalar standardization ---
    scaler = StandardScaler()
    train_mask_flat = mask[train_indices].reshape(-1)
    train_scalars_flat = scalars[train_indices].reshape(-1, cfg.SCALAR_DIM)[
        train_mask_flat
    ]
    scaler.fit(train_scalars_flat)

    scalars_2d[mask_1d] = scaler.transform(scalars_2d[mask_1d])
    data["scalar_features"] = scalars_2d.reshape(N, max_n, cfg.SCALAR_DIM)

    # --- Sigma profile standardization ---
    sigmas = data["sigma_profiles"]
    sigmas_2d = sigmas.reshape(-1, cfg.SIGMA_DIM)
    scaler_sigma = StandardScaler()
    train_sigmas_flat = sigmas[train_indices].reshape(-1, cfg.SIGMA_DIM)[
        train_mask_flat
    ]
    scaler_sigma.fit(train_sigmas_flat)
    sigmas_2d[mask_1d] = scaler_sigma.transform(sigmas_2d[mask_1d])
    data["sigma_profiles"] = sigmas_2d.reshape(N, max_n, cfg.SIGMA_DIM, 1)

    return data
