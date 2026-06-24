"""Data loading and standardization pipeline."""

import pickle
import numpy as np
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler

from .config import Config


def load_raw_data(csv_path, cfg: Config):
    import pandas as pd

    df = pd.read_csv(csv_path, low_memory=False)
    N = len(df)
    max_n = cfg.MAX_COMPONENTS

    sigmas = np.zeros((N, max_n, cfg.SIGMA_DIM, 1), dtype='float32')
    scalars = np.zeros((N, max_n, cfg.SCALAR_DIM), dtype='float32')
    masks = np.zeros((N, max_n), dtype=bool)
    x = np.zeros((N, max_n), dtype='float32')
    n = np.zeros((N, max_n), dtype='float32')
    ln_gamma = np.zeros((N, max_n), dtype='float32')
    solute_mask = np.zeros((N, max_n), dtype=bool)

    n_comp_arr = df['n_components'].values.astype(int)
    for i in range(max_n):
        masks[:, i] = n_comp_arr > i
    row_idx = np.arange(N)
    solute_mask[row_idx, n_comp_arr - 1] = True

    for i in range(max_n):
        comp_exists = n_comp_arr > i
        if not comp_exists.any():
            continue

        sigma_strings = df.loc[comp_exists, f'sigma_{i}'].values
        sigmas_i = np.array(
            [np.fromstring(s, sep=',') for s in sigma_strings], dtype='float32'
        )
        sigmas[comp_exists, i, :, 0] = sigmas_i

        scalar_cols = [
            f'HOMO_{i}', f'LUMO_{i}', f'Dipole_{i}',
            f'M0_{i}', f'M1_{i}', f'M2_{i}', f'M3_{i}', f'M4_{i}',
        ]
        scalars[comp_exists, i, :8] = df.loc[comp_exists, scalar_cols].values.astype('float32')
        x[comp_exists, i] = df.loc[comp_exists, f'x_{i}'].values.astype('float32')
        n[comp_exists, i] = df.loc[comp_exists, f'n_{i}'].values.astype('float32')
        ln_gamma[comp_exists, i] = df.loc[comp_exists, f'ln_gamma_{i}'].values.astype('float32')

    data = {
        'sigma_profiles': sigmas,
        'scalar_features': scalars,
        'mask': masks,
        'T': df['T_K'].values.astype('float32'),
        'x': x,
        'n': n,
        'ln_gamma_target': ln_gamma,
        'solute_mask': solute_mask,
    }
    return data


def standardize_data(data, train_indices, cfg: Config):
    scalars = data['scalar_features']
    masks = data['mask']
    N, max_n = scalars.shape[:2]

    scalars_2d = scalars.reshape(-1, cfg.SCALAR_DIM)
    mask_1d = masks.reshape(-1)

    train_row_mask = np.zeros(N, dtype=bool)
    train_row_mask[train_indices] = True
    train_mask_flat = (train_row_mask[:, None] & masks).reshape(-1)

    scaler_scalar = StandardScaler()
    scaler_scalar.fit(scalars_2d[train_mask_flat])
    scalars_2d[mask_1d] = scaler_scalar.transform(scalars_2d[mask_1d])
    data['scalar_features'] = scalars_2d.reshape(N, max_n, cfg.SCALAR_DIM)

    sigmas = data['sigma_profiles']
    sigmas_2d = sigmas.reshape(-1, cfg.SIGMA_DIM)
    scaler_sigma = StandardScaler()
    scaler_sigma.fit(sigmas_2d[train_mask_flat])
    sigmas_2d[mask_1d] = scaler_sigma.transform(sigmas_2d[mask_1d])
    data['sigma_profiles'] = sigmas_2d.reshape(N, max_n, cfg.SIGMA_DIM, 1)

    return data, scaler_scalar, scaler_sigma


class ThermoDataset:
    def __init__(self, data_dict, cfg: Config):
        self.data = data_dict
        self.cfg = cfg
        self.N = len(data_dict['T'])

    def get_batch(self, indices):
        return {k: jnp.array(v[indices]) for k, v in self.data.items()}


def save_scalers(scaler_scalar, scaler_sigma, path):
    scalers = {'scaler_scalar': scaler_scalar, 'scaler_sigma': scaler_sigma}
    with open(path, 'wb') as f:
        pickle.dump(scalers, f)


def load_scalers(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
