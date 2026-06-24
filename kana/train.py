"""Training loop, evaluation, and thermodynamic verification."""

import jax
import jax.numpy as jnp
from jax import random, jit
from flax.training import train_state, checkpoints
import optax
import numpy as np
from functools import partial

from .config import Config
from .architecture import HardConstrainedCINN
from .thermodynamics import ThermodynamicEngine


def create_train_state(model, cfg: Config, rng, total_steps):
    dummy_sigmas = jnp.ones((1, cfg.MAX_COMPONENTS, cfg.SIGMA_DIM, 1))
    dummy_scalars = jnp.ones((1, cfg.MAX_COMPONENTS, cfg.SCALAR_DIM))
    dummy_mask = jnp.ones((1, cfg.MAX_COMPONENTS), dtype=bool)
    dummy_T = jnp.ones((1,)) * 298.15
    dummy_x = jnp.ones((1, cfg.MAX_COMPONENTS)) / cfg.MAX_COMPONENTS

    variables = model.init(
        rng, dummy_sigmas, dummy_scalars, dummy_mask, dummy_T, dummy_x, training=False
    )
    params = variables['params']

    if cfg.LR_SCHEDULE:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.LR,
            warmup_steps=total_steps // 20,
            decay_steps=total_steps,
            end_value=cfg.LR * 0.01,
        )
        tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    else:
        tx = optax.adam(cfg.LR)

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def loss_fn(params, batch, cfg: Config, model_apply, engine, dropout_rng):
    sigmas = batch['sigma_profiles']
    scalars = batch['scalar_features']
    masks = batch['mask']
    T = batch['T']
    x = batch['x']
    n = batch['n']

    gE_RT_pred = model_apply(
        {'params': params}, sigmas, scalars, masks, T, x,
        training=True, rngs={'dropout': dropout_rng},
    )

    ln_gamma_pred = engine.compute_ln_gamma(params, model_apply, sigmas, scalars, masks, T, n)

    solute_mask_f = batch['solute_mask'].astype(jnp.float32)
    ln_gamma_target = batch['ln_gamma_target']
    loss = cfg.W_LN_GAMMA * jnp.sum(solute_mask_f * (ln_gamma_pred - ln_gamma_target) ** 2)
    loss = loss / jnp.maximum(jnp.sum(solute_mask_f), 1.0)

    if 'hE_target' in batch:
        hE_pred = engine.compute_hE(params, model_apply, sigmas, scalars, masks, T, x)
        loss += cfg.W_HE * jnp.mean((hE_pred - batch['hE_target']) ** 2)
    if 'sE_target' in batch:
        sE_pred = engine.compute_sE(params, model_apply, sigmas, scalars, masks, T, x, n)
        loss += cfg.W_SE * jnp.mean((sE_pred - batch['sE_target']) ** 2)
    if cfg.W_BOUNDARY > 0.0:
        batch_size = sigmas.shape[0]
        max_n = sigmas.shape[1]

        pure_mask = jnp.zeros((batch_size, max_n), dtype=bool)
        pure_mask = pure_mask.at[:, 0].set(True)
        pure_x = jnp.zeros((batch_size, max_n))
        pure_x = pure_x.at[:, 0].set(1.0)

        gE_pure = model_apply(
            {'params': params}, sigmas, scalars, pure_mask, T, pure_x,
            training=True, rngs={'dropout': dropout_rng},
        )
        boundary_loss = jnp.mean(gE_pure ** 2)
        loss += cfg.W_BOUNDARY * boundary_loss

    metrics = {
        'loss': loss,
        'mae_ln_gamma': (
            jnp.sum(solute_mask_f * jnp.abs(ln_gamma_pred - ln_gamma_target))
            / jnp.maximum(jnp.sum(solute_mask_f), 1.0)
        ),
        'gibbs_duhem_residual': jnp.mean(
            jnp.abs(engine.gibbs_duhem_residual(ln_gamma_pred, x, masks, gE_RT_pred))
        ),
    }
    return loss, metrics


@partial(jit, static_argnames=('cfg', 'engine'))
def train_step(state, batch, cfg: Config, dropout_rng, engine):
    def _loss(params):
        return loss_fn(params, batch, cfg, state.apply_fn, engine, dropout_rng)
    (loss, metrics), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


@partial(jit, static_argnames=('cfg', 'engine'))
def eval_step(state, batch, cfg: Config, engine):
    dummy_rng = random.PRNGKey(0)
    _, metrics = loss_fn(state.params, batch, cfg, state.apply_fn, engine, dummy_rng)
    return metrics


def evaluate_model(state, dataset, cfg: Config, engine, batch_size=1024):
    steps_per_epoch = int(np.ceil(dataset.N / batch_size))
    all_metrics = []
    for i in range(steps_per_epoch):
        idx = np.arange(i * batch_size, min((i + 1) * batch_size, dataset.N))
        batch = dataset.get_batch(idx)
        metrics = eval_step(state, batch, cfg, engine)
        all_metrics.append(metrics)

    aggregated = {}
    for k in all_metrics[0].keys():
        aggregated[k] = float(np.mean([float(m[k]) for m in all_metrics]))
    return aggregated


def generate_thermo_verification_report(best_state, model, engine, test_batch):
    print("\n" + "=" * 70)
    print("THERMODYNAMIC VERIFICATION REPORT (DROPOUT OFF)")
    print("=" * 70)

    ln_gamma = engine.compute_ln_gamma(
        best_state.params, model.apply,
        test_batch['sigma_profiles'], test_batch['scalar_features'],
        test_batch['mask'], test_batch['T'], test_batch['n'],
    )

    gE_RT_test = model.apply(
        {'params': best_state.params},
        test_batch['sigma_profiles'], test_batch['scalar_features'],
        test_batch['mask'], test_batch['T'], test_batch['x'],
        training=False,
    )

    gd_res = engine.gibbs_duhem_residual(ln_gamma, test_batch['x'], test_batch['mask'], gE_RT_test)
    print(f"[1] Gibbs-Duhem residual: {jnp.max(jnp.abs(gd_res)):.2e} (target: < 1e-6)")

    max_n = test_batch['sigma_profiles'].shape[1]
    pure_x = jnp.zeros((1, max_n))
    pure_x = pure_x.at[0, 0].set(1.0)
    pure_mask = jnp.zeros((1, max_n), dtype=bool)
    pure_mask = pure_mask.at[0, 0].set(True)

    pure_sigmas = test_batch['sigma_profiles'][0:1]
    pure_scalars = test_batch['scalar_features'][0:1]
    pure_T = test_batch['T'][0:1]

    gE_pure = model.apply(
        {'params': best_state.params}, pure_sigmas, pure_scalars,
        pure_mask, pure_T, pure_x, training=False,
    )
    print(f"[2] g^E at pure component: {float(gE_pure[0]):.6f} (target: ~0)")

    perm = jnp.array([1, 0] + list(range(2, max_n)))
    sigmas_perm = test_batch['sigma_profiles'][:, perm, :, :]
    scalars_perm = test_batch['scalar_features'][:, perm, :]
    n_perm = test_batch['n'][:, perm]
    mask_perm = test_batch['mask'][:, perm]

    ln_gamma_perm = engine.compute_ln_gamma(
        best_state.params, model.apply, sigmas_perm, scalars_perm,
        mask_perm, test_batch['T'], n_perm,
    )
    perm_diff = jnp.abs(ln_gamma[0, 0] - ln_gamma_perm[0, 1])
    print(f"[3] Permutation invariance (swap 0<->1): {perm_diff:.2e} (target: < 1e-5)")

    hE_pure = engine.compute_hE(
        best_state.params, model.apply,
        test_batch['sigma_profiles'], test_batch['scalar_features'],
        test_batch['mask'], test_batch['T'], test_batch['x'],
    )
    sE_pure = engine.compute_sE(
        best_state.params, model.apply,
        test_batch['sigma_profiles'], test_batch['scalar_features'],
        test_batch['mask'], test_batch['T'], test_batch['x'], test_batch['n'],
    )
    gE_recon = hE_pure - test_batch['T'] * sE_pure
    gE_actual = gE_RT_test * engine.R * test_batch['T']
    gh_err = jnp.max(jnp.abs(gE_actual - gE_recon))
    print(f"[4] Gibbs-Helmholtz error: {gh_err:.2e} (target: < 1e-3)")
    print("=" * 70)

    return {
        'gibbs_duhem_max': float(jnp.max(jnp.abs(gd_res))),
        'pure_boundary_gE': float(gE_pure[0]),
        'permutation_err': float(perm_diff),
        'gibbs_helmholtz_err': float(gh_err),
    }
