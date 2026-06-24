"""Unified main entry point with CLI for training KANA model variants."""

import argparse
import pickle
import numpy as np
import jax
from jax import random

from .config import Config, PRESETS
from .architecture import HardConstrainedCINN
from .thermodynamics import ThermodynamicEngine
from .dataset import load_raw_data, standardize_data, ThermoDataset, save_scalers
from .train import (
    create_train_state, train_step, evaluate_model,
    generate_thermo_verification_report,
)
from flax.training import checkpoints


def train(cfg: Config, csv_path: str, output_dir: str,
          batch_size: int = 1024, n_epochs: int = 1001,
          patience: int = 25):
    rng = random.PRNGKey(cfg.SEED)

    model = HardConstrainedCINN(cfg=cfg)
    engine = ThermodynamicEngine(cfg)

    raw_data = load_raw_data(csv_path, cfg)

    n_total = len(raw_data['T'])
    n_train = int(0.8 * n_total)
    rng_np = np.random.default_rng(cfg.SEED)
    indices = rng_np.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    raw_data, scaler_scalar, scaler_sigma = standardize_data(raw_data, train_idx, cfg)

    train_data = ThermoDataset({k: v[train_idx] for k, v in raw_data.items()}, cfg)
    val_data = ThermoDataset({k: v[val_idx] for k, v in raw_data.items()}, cfg)

    steps_per_epoch = int(np.ceil(train_data.N / batch_size))
    total_steps = n_epochs * steps_per_epoch

    state = create_train_state(model, cfg, rng, total_steps)

    patience_counter = 0
    best_val_mae = float('inf')
    best_state = state

    print("=" * 70)
    print("KANA AI — Hard-Constrained Chemical-Informed Neural Network")
    print(f"  Preset: LATENT_A={cfg.LATENT_A}, LATENT_B={cfg.LATENT_B}, LR={cfg.LR}")
    print("=" * 70)
    print(f"Training samples: {train_data.N}")
    print(f"Validation samples: {val_data.N}")
    print(f"Epochs: {n_epochs}")
    print("=" * 70)

    for epoch in range(n_epochs):
        perm = np.random.permutation(train_data.N)
        epoch_loss = 0.0

        for i in range(steps_per_epoch):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            batch = train_data.get_batch(idx)
            dropout_rng = random.PRNGKey(epoch * 1000 + i)
            state, metrics = train_step(state, batch, cfg, dropout_rng, engine)
            epoch_loss += float(metrics['loss'])

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            val_metrics = evaluate_model(state, val_data, cfg, engine, batch_size)
            current_val_mae = val_metrics['mae_ln_gamma']

            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {epoch_loss / steps_per_epoch:.4f} | "
                f"Val MAE(ln gamma): {current_val_mae:.4f} | "
                f"GD Residual: {val_metrics['gibbs_duhem_residual']:.2e}"
            )

            if current_val_mae < best_val_mae:
                best_val_mae = current_val_mae
                patience_counter = 0
                best_state = state
                print(f"   >> New best model! Val MAE: {best_val_mae:.4f}")
                checkpoints.save_checkpoint(
                    f'{output_dir}/best_checkpoint', state, step=epoch,
                    keep=1, overwrite=True,
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}. Best MAE: {best_val_mae:.4f}")
                    break

    print("\nTRAINING COMPLETE")

    save_scalers(scaler_scalar, scaler_sigma, f'{output_dir}/scalers.pkl')
    print(f"Saved: {output_dir}/scalers.pkl")

    best_state = checkpoints.restore_checkpoint(
        ckpt_dir=f'{output_dir}/best_checkpoint', target=state,
    )

    test_batch = val_data.get_batch(np.arange(min(10, val_data.N)))
    verif = generate_thermo_verification_report(best_state, model, engine, test_batch)

    print("Generating validation predictions for parity plots...")
    y_pred_list = []
    y_exp_list = []
    order_list = []

    eval_batch_size = 1024
    for start_idx in range(0, val_data.N, eval_batch_size):
        end_idx = min(start_idx + eval_batch_size, val_data.N)
        batch_idx = np.arange(start_idx, end_idx)
        batch = val_data.get_batch(batch_idx)

        pred_batch = engine.compute_ln_gamma(
            best_state.params, model.apply,
            batch['sigma_profiles'], batch['scalar_features'],
            batch['mask'], batch['T'], batch['n'],
        )

        mask_batch = batch['mask']
        exp_batch = batch['ln_gamma_target']
        pred_np = np.array(pred_batch)
        order_batch = np.sum(mask_batch, axis=1)

        solute_indices = (order_batch - 1).astype(int)
        row_idx = np.arange(len(order_batch))
        y_pred_solute = pred_np[row_idx, solute_indices]
        y_exp_solute = exp_batch[row_idx, solute_indices]
        y_pred_list.append(y_pred_solute)
        y_exp_list.append(y_exp_solute)
        order_list.append(order_batch)

    y_pred_flat = np.concatenate(y_pred_list)
    y_exp_flat = np.concatenate(y_exp_list)
    order_flat = np.concatenate(order_list)

    combined_data = np.column_stack((y_exp_flat, y_pred_flat, order_flat))
    np.savetxt(
        f"{output_dir}/Figure3_Parity_Data.csv", combined_data,
        delimiter=",", header="y_exp,y_pred,order", comments="",
    )
    print(f"Exported: {output_dir}/Figure3_Parity_Data.csv")

    return state, engine, model, val_data


def main():
    parser = argparse.ArgumentParser(
        description="KANA: Hard-Constrained Chemical-Informed Neural Network",
    )
    parser.add_argument('--csv-path', type=str, required=True,
                        help='Path to merged CSV dataset')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory for checkpoints and outputs')
    parser.add_argument('--preset', type=str, default='1',
                        choices=['1', '2', '3', '4'],
                        help='Config preset (1-4): 1=256/lr1e-3, 2=512/lr1e-3, 3=256/lr1e-4, 4=512/lr1e-4')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=1001)
    parser.add_argument('--patience', type=int, default=25)

    cli_overrides = parser.add_argument_group('Config overrides')
    cli_overrides.add_argument('--latent-a', type=int, default=None)
    cli_overrides.add_argument('--latent-b', type=int, default=None)
    cli_overrides.add_argument('--lr', type=float, default=None)
    cli_overrides.add_argument('--dropout', type=float, default=None)
    cli_overrides.add_argument('--seed', type=int, default=None)
    cli_overrides.add_argument('--ge-head-hidden', type=int, default=None)

    args = parser.parse_args()

    cfg = PRESETS[args.preset]

    if args.latent_a is not None:
        cfg = Config(**{**cfg.__dict__, 'LATENT_A': args.latent_a, 'LATENT_Z': args.latent_a + cfg.LATENT_B})
    if args.latent_b is not None:
        cfg = Config(**{**cfg.__dict__, 'LATENT_B': args.latent_b, 'LATENT_Z': cfg.LATENT_A + args.latent_b})
    if args.lr is not None:
        cfg = Config(**{**cfg.__dict__, 'LR': args.lr})
    if args.dropout is not None:
        cfg = Config(**{**cfg.__dict__, 'DROPOUT_RATE': args.dropout})
    if args.seed is not None:
        cfg = Config(**{**cfg.__dict__, 'SEED': args.seed})
    if args.ge_head_hidden is not None:
        cfg = Config(**{**cfg.__dict__, 'GE_HEAD_HIDDEN': args.ge_head_hidden})

    train(
        cfg=cfg, csv_path=args.csv_path, output_dir=args.output_dir,
        batch_size=args.batch_size, n_epochs=args.epochs,
        patience=args.patience,
    )


if __name__ == '__main__':
    main()
