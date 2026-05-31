"""Training loop, evaluation, checkpointing, and parity-data export."""

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
from flax.training import train_state, checkpoints
import optax

from .config import Config
from .architecture import HardConstrainedCINN
from .thermodynamics import ThermodynamicEngine
from .dataset import ThermoDataset, load_raw_data, standardize_data


def create_train_state(model: HardConstrainedCINN, cfg: Config, rng: jax.Array):
    """Initialize model parameters and optimizer state."""
    dummy_sigmas = jnp.ones((1, cfg.MAX_COMPONENTS, cfg.SIGMA_DIM, 1))
    dummy_scalars = jnp.ones((1, cfg.MAX_COMPONENTS, cfg.SCALAR_DIM))
    dummy_mask = jnp.ones((1, cfg.MAX_COMPONENTS), dtype=bool)
    dummy_T = jnp.ones((1,)) * 298.15
    dummy_x = jnp.ones((1, cfg.MAX_COMPONENTS)) / cfg.MAX_COMPONENTS

    variables = model.init(
        rng, dummy_sigmas, dummy_scalars, dummy_mask, dummy_T, dummy_x, training=False
    )
    params = variables["params"]

    if cfg.LR_SCHEDULE:
        schedule = optax.exponential_decay(
            cfg.LR, transition_steps=1000, decay_rate=0.9
        )
        tx = optax.adam(schedule)
    else:
        tx = optax.adam(cfg.LR)

    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )


def loss_fn(
    params: dict,
    batch: dict,
    cfg: Config,
    model_apply: callable,
    engine: ThermodynamicEngine,
    dropout_rng: jax.Array,
):
    """Compute weighted multi-objective loss and diagnostics."""
    sigmas = batch["sigma_profiles"]
    scalars = batch["scalar_features"]
    mask = batch["mask"]
    T = batch["T"]
    x = batch["x"]
    n = batch["n"]

    gE_RT_pred = model_apply(
        {"params": params},
        sigmas,
        scalars,
        mask,
        T,
        x,
        training=True,
        rngs={"dropout": dropout_rng},
    )

    ln_gamma_pred = engine.compute_ln_gamma(
        params, model_apply, sigmas, scalars, mask, T, n
    )
    hE_pred = engine.compute_hE(
        params, model_apply, sigmas, scalars, mask, T, x
    )
    sE_pred = engine.compute_sE(
        params, model_apply, sigmas, scalars, mask, T, x, n
    )

    mask_f = mask.astype(jnp.float32)
    ln_gamma_target = batch["ln_gamma_target"]

    # Primary loss: ln(gamma) MSE
    loss = cfg.W_LN_GAMMA * jnp.sum(
        mask_f * (ln_gamma_pred - ln_gamma_target) ** 2
    )
    loss = loss / jnp.maximum(jnp.sum(mask_f), 1.0)

    # Auxiliary losses
    if "hE_target" in batch:
        loss += cfg.W_HE * jnp.mean((hE_pred - batch["hE_target"]) ** 2)
    if "sE_target" in batch:
        loss += cfg.W_SE * jnp.mean((sE_pred - batch["sE_target"]) ** 2)

    metrics = {
        "loss": loss,
        "mae_ln_gamma": jnp.sum(mask_f * jnp.abs(ln_gamma_pred - ln_gamma_target))
        / jnp.maximum(jnp.sum(mask_f), 1.0),
        "gibbs_duhem_residual": jnp.mean(
            jnp.abs(engine.gibbs_duhem_residual(ln_gamma_pred, x, mask, gE_RT_pred))
        ),
    }
    return loss, metrics


@partial(jit, static_argnames=("cfg", "engine"))
def train_step(
    state: train_state.TrainState,
    batch: dict,
    cfg: Config,
    dropout_rng: jax.Array,
    engine: ThermodynamicEngine,
):
    """Single JIT-compiled training step."""
    def _loss(params):
        return loss_fn(params, batch, cfg, state.apply_fn, engine, dropout_rng)

    (loss, metrics), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


@partial(jit, static_argnames=("cfg", "engine"))
def eval_step(
    state: train_state.TrainState,
    batch: dict,
    cfg: Config,
    engine: ThermodynamicEngine,
):
    """Single JIT-compiled evaluation step."""
    dummy_rng = random.PRNGKey(0)
    _, metrics = loss_fn(state.params, batch, cfg, state.apply_fn, engine, dummy_rng)
    return metrics


def evaluate_model(
    state: train_state.TrainState,
    dataset: ThermoDataset,
    cfg: Config,
    engine: ThermodynamicEngine,
    batch_size: int = 512,
) -> dict:
    """Evaluate model on a full dataset in batches."""
    n_batches = int(np.ceil(dataset.N / batch_size))
    all_metrics = []
    for i in range(n_batches):
        idx = np.arange(i * batch_size, min((i + 1) * batch_size, dataset.N))
        batch = dataset.get_batch(idx)
        metrics = eval_step(state, batch, cfg, engine)
        all_metrics.append(metrics)

    aggregated = {}
    keys = all_metrics[0].keys()
    for k in keys:
        aggregated[k] = float(np.mean([float(m[k]) for m in all_metrics]))
    return aggregated


def generate_thermo_verification_report(
    state: train_state.TrainState,
    model: HardConstrainedCINN,
    engine: ThermodynamicEngine,
    test_batch: dict,
) -> dict:
    """Print and return a hard-constraint verification report."""
    print("\n" + "=" * 70)
    print("THERMODYNAMIC VERIFICATION REPORT (DROPOUT OFF)")
    print("=" * 70)

    # 1. Gibbs-Duhem
    ln_gamma = engine.compute_ln_gamma(
        state.params,
        model.apply,
        test_batch["sigma_profiles"],
        test_batch["scalar_features"],
        test_batch["mask"],
        test_batch["T"],
        test_batch["n"],
    )

    gE_RT_test = model.apply(
        {"params": state.params},
        test_batch["sigma_profiles"],
        test_batch["scalar_features"],
        test_batch["mask"],
        test_batch["T"],
        test_batch["x"],
        training=False,
    )

    gd_res = engine.gibbs_duhem_residual(
        ln_gamma, test_batch["x"], test_batch["mask"], gE_RT_test
    )
    print(
        f"[1] Gibbs-Duhem residual: {jnp.max(jnp.abs(gd_res)):.2e} (target: < 1e-6)"
    )

    # 2. Pure component boundary
    pure_x = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    pure_mask = jnp.array([[True, False, False, False, False]])
    pure_sigmas = test_batch["sigma_profiles"][0:1]
    pure_scalars = test_batch["scalar_features"][0:1]
    pure_T = test_batch["T"][0:1]
    gE_pure = model.apply(
        {"params": state.params},
        pure_sigmas,
        pure_scalars,
        pure_mask,
        pure_T,
        pure_x,
        training=False,
    )
    print(f"[2] g^E at pure component: {float(gE_pure[0]):.6f} (target: ≈ 0)")

    # 3. Permutation invariance
    perm = jnp.array([1, 0, 2, 3, 4])
    sigmas_perm = test_batch["sigma_profiles"].at[0].set(
        test_batch["sigma_profiles"][0, perm]
    )
    scalars_perm = test_batch["scalar_features"].at[0].set(
        test_batch["scalar_features"][0, perm]
    )
    n_perm = test_batch["n"].at[0].set(test_batch["n"][0, perm])

    ln_gamma_perm = engine.compute_ln_gamma(
        state.params,
        model.apply,
        sigmas_perm,
        scalars_perm,
        test_batch["mask"],
        test_batch["T"],
        n_perm,
    )
    perm_diff = jnp.abs(ln_gamma[0, 0] - ln_gamma_perm[0, 1])
    print(
        f"[3] Permutation invariance (swap 0↔1): {perm_diff:.2e} (target: < 1e-5)"
    )

    # 4. Gibbs-Helmholtz Consistency
    hE_pure = engine.compute_hE(
        state.params,
        model.apply,
        test_batch["sigma_profiles"],
        test_batch["scalar_features"],
        test_batch["mask"],
        test_batch["T"],
        test_batch["x"],
    )
    sE_pure = engine.compute_sE(
        state.params,
        model.apply,
        test_batch["sigma_profiles"],
        test_batch["scalar_features"],
        test_batch["mask"],
        test_batch["T"],
        test_batch["x"],
        test_batch["n"],
    )
    gE_recon = hE_pure - test_batch["T"] * sE_pure
    gE_actual = gE_RT_test * engine.R * test_batch["T"]

    gh_err = jnp.max(jnp.abs(gE_actual - gE_recon))
    print(f"[4] Gibbs-Helmholtz error: {gh_err:.2e} (target: < 1e-4)")
    print("=" * 70)

    return {
        "gibbs_duhem_max": float(jnp.max(jnp.abs(gd_res))),
        "pure_boundary_gE": float(gE_pure[0]),
        "permutation_err": float(perm_diff),
        "gibbs_helmholtz_err": float(gh_err),
    }


def export_parity_data(
    state: train_state.TrainState,
    model: HardConstrainedCINN,
    engine: ThermodynamicEngine,
    val_data: ThermoDataset,
    output_path: str,
    batch_size: int = 1024,
):
    """Export validation predictions for external parity-plot generation."""
    y_pred_list = []
    y_exp_list = []
    order_list = []

    for start_idx in range(0, val_data.N, batch_size):
        end_idx = min(start_idx + batch_size, val_data.N)
        indices = np.arange(start_idx, end_idx)
        batch = val_data.get_batch(indices)

        pred_batch = engine.compute_ln_gamma(
            state.params,
            model.apply,
            batch["sigma_profiles"],
            batch["scalar_features"],
            batch["mask"],
            batch["T"],
            batch["n"],
        )

        mask_batch = batch["mask"]
        exp_batch = batch["ln_gamma_target"]
        order_batch = np.sum(mask_batch, axis=1)
        order_broadcast = np.repeat(order_batch, mask_batch.shape[1])

        mask_flat = mask_batch.flatten()
        y_pred_list.append(np.array(pred_batch).flatten()[mask_flat])
        y_exp_list.append(np.array(exp_batch).flatten()[mask_flat])
        order_list.append(order_broadcast[mask_flat])

    y_pred_flat = np.concatenate(y_pred_list)
    y_exp_flat = np.concatenate(y_exp_list)
    order_flat = np.concatenate(order_list)

    combined_data = np.column_stack((y_exp_flat, y_pred_flat, order_flat))
    np.savetxt(
        output_path,
        combined_data,
        delimiter=",",
        header="y_exp,y_pred,order",
        comments="",
    )
    print(f"SUCCESS: Exported parity data to '{output_path}'")


def main(
    csv_path: str = "/kaggle/input/datasets/flowychie/kana-dataset/merged_KANA_dataset.csv",
    output_dir: str = "/kaggle/working",
    batch_size: int = 512,
    n_epochs: int = 201,
):
    """End-to-end training and evaluation pipeline."""
    cfg = Config()
    rng = random.PRNGKey(cfg.SEED)

    model = HardConstrainedCINN(cfg=cfg)
    state = create_train_state(model, cfg, rng)
    engine = ThermodynamicEngine(cfg)

    # Load raw data
    raw_data = load_raw_data(csv_path, cfg)

    # Split train/val
    n_total = len(raw_data["T"])
    n_train = int(0.8 * n_total)
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # FIX 1: Standardize AFTER split, fit on train only
    raw_data = standardize_data(raw_data, train_idx, cfg)

    train_data = ThermoDataset({k: v[train_idx] for k, v in raw_data.items()}, cfg)
    val_data = ThermoDataset({k: v[val_idx] for k, v in raw_data.items()}, cfg)

    n_batches = int(np.ceil(train_data.N / batch_size))

    print("=" * 70)
    print("KANA AI — Hard-Constrained CINN")
    print("=" * 70)
    print(f"Training samples:   {train_data.N}")
    print(f"Validation samples: {val_data.N}")
    print(f"Epochs:             {n_epochs}")
    print(f"Batch size:         {batch_size}")
    print("=" * 70)

    best_val_mae = float("inf")
    best_ckpt_dir = f"{output_dir}/best_cinn_checkpoint"

    for epoch in range(n_epochs):
        perm = np.random.permutation(train_data.N)
        epoch_loss = 0.0

        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            batch = train_data.get_batch(idx)
            dropout_rng = random.PRNGKey(epoch * 1000 + i)
            state, metrics = train_step(state, batch, cfg, dropout_rng, engine)
            epoch_loss += float(metrics["loss"])

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            val_metrics = evaluate_model(state, val_data, cfg, engine, batch_size)
            current_val_mae = val_metrics["mae_ln_gamma"]

            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {epoch_loss / n_batches:.4f} | "
                f"Val MAE(lnγ): {current_val_mae:.4f} | "
                f"GD Residual: {val_metrics['gibbs_duhem_residual']:.2e}"
            )

            if current_val_mae < best_val_mae:
                best_val_mae = current_val_mae
                print(
                    f"   >> 🔥 New best model! Val MAE: {best_val_mae:.4f}. Saving..."
                )
                checkpoints.save_checkpoint(
                    best_ckpt_dir,
                    state,
                    step=epoch,
                    keep=1,
                    overwrite=True,
                )

    # Verification report
    test_batch = val_data.get_batch(np.arange(10))
    verif = generate_thermo_verification_report(state, model, engine, test_batch)

    # Restore best model and export parity data
    print("\nRestoring best model for final parity export...")
    best_state = checkpoints.restore_checkpoint(ckpt_dir=best_ckpt_dir, target=state)
    parity_path = f"{output_dir}/Figure3_Parity_Data.csv"
    export_parity_data(best_state, model, engine, val_data, parity_path)

    print("\nTRAINING COMPLETE")
    return state, engine, model, val_data


if __name__ == "__main__":
    state, engine, model, val_data = main()
