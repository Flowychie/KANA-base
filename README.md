# KANA: Hard-Constrained Chemical-Informed Neural Network

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4+-orange.svg)](https://github.com/google/jax)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **Foundational MVP for multi-component thermodynamic property prediction with automatic satisfaction of Gibbs-Duhem, Gibbs-Helmholtz, and pure-component boundary constraints.**

---

## Scientific Breakthrough

Traditional neural-network models for excess Gibbs energy ($g^E$) are typically validated only on binary mixtures and fail to generalize when scaled to ternary or quaternary systems. **KANA** (Chemical-Informed Neural Network) introduces a **hard-constrained architecture** that:

1. **Scales from binary to quaternary (and beyond)** via a pairwise multiplicative boundary condition $\sum_{i<j} x_i x_j$ that naturally vanishes at pure-component limits regardless of mixture order.
2. **Enforces the Gibbs-Duhem relation exactly** by computing $\ln\gamma_i$ through automatic differentiation of $n g^E\!/\!RT$ rather than predicting activity coefficients directly.
3. **Satisfies the Gibbs-Helmholtz relation** by constructing $h^E$ and $s^E$ from temperature derivatives of the same $g^E$ head, guaranteeing thermodynamic consistency.
4. **Maintains strict permutation invariance** through a mole-fraction weighted aggregator $\sum_i x_i z_i / \sum_i x_i$, ensuring that component ordering does not affect mixture predictions.

This repository contains the **minimum viable product (MVP)** architecture and training pipeline. It is intended as a reproducible foundation for journal submission and community extension.

---

## Architecture Overview

```
Component i
    ├─ Sigma Profile  →  SigmaEncoder  ─┐
    └─ Scalar Feats   →  ScalarEncoder ─┘
                                        ↓
                              z_i = [z_a ‖ z_b]
                                        ↓
                         ┌────────────────────────┐
                         │  MoleFractionWeighted  │
                         │     Aggregator         │
                         │  z_mix = Σ x_i z_i / Σx_i │
                         └────────────────────────┘
                                        ↓
                              GE_PredictionHead
                         g^E/RT = raw(z_mix, T) × Σ_{i<j} x_i x_j
                                        ↓
                         ThermodynamicEngine (autodiff)
                         ├─ ln γ_i  = ∂(n g^E/RT)/∂n_i
                         ├─ h^E     = -T² ∂(g^E/T)/∂T
                         └─ s^E     = (h^E - g^E)/T
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/kana-cinn.git
cd kana-cinn
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note on JAX:** The `requirements.txt` installs the CPU-only version of JAX. For GPU or TPU support, follow the [official JAX installation guide](https://github.com/google/jax#installation) to install the wheel matching your CUDA/cuDNN version.

---

## Quick Start

### Training from a CSV dataset

```python
from kana import main

state, engine, model, val_data = main(
    csv_path="data/merged_KANA_dataset.csv",
    output_dir="outputs",
    batch_size=512,
    n_epochs=201
)
```

The script will:
- Load and standardize data (leakage-free: fit on train, transform all).
- Train with a composite loss on $\ln\gamma$, $h^E$, and $s^E$.
- Track the best validation MAE and save checkpoints.
- Print a **thermodynamic verification report** verifying:
  - Gibbs-Duhem residual $< 10^{-6}$
  - Pure-component boundary $g^E \approx 0$
  - Permutation invariance $< 10^{-5}$
  - Gibbs-Helmholtz consistency $< 10^{-4}$

### Using individual modules

```python
from kana import Config, HardConstrainedCINN, ThermodynamicEngine, create_train_state
import jax

cfg = Config()
model = HardConstrainedCINN(cfg=cfg)
state = create_train_state(model, cfg, jax.random.PRNGKey(42))
engine = ThermodynamicEngine(cfg)

# ... load your batch ...
gE = engine.predict_gE(state.params, model.apply, sigmas, scalars, mask, T, x)
```

---

## Repository Structure

```
kana-cinn/
├── README.md
├── requirements.txt
├── kana/
│   ├── __init__.py          # Public API
│   ├── config.py            # Hyperparameters & constants
│   ├── architecture.py      # Flax modules (encoders, aggregator, head)
│   ├── thermodynamics.py    # ThermodynamicEngine & consistency relations
│   ├── dataset.py           # Data loading & standardization
│   └── train.py             # Training loop, evaluation, checkpointing
└── examples/
    └── train_from_csv.py    # End-to-end example
```

---

## Thermodynamic Constraints

| Constraint | Enforcement Mechanism | Mathematical Form |
|---|---|---|
| **Gibbs-Duhem** | Auto-diff of $n g^E\!/\!RT$ w.r.t. $n_i$ | $\sum_i x_i \ln\gamma_i = g^E/RT$ |
| **Pure-component boundary** | Pairwise multiplicative mask | $g^E \propto \sum_{i<j} x_i x_j \to 0$ when $x_k \to 1$ |
| **Gibbs-Helmholtz** | Temperature auto-diff of unified $g^E$ head | $h^E = -T^2 \frac{\partial(g^E/T)}{\partial T}$ |
| **Permutation invariance** | Mole-fraction weighted aggregation | $\sum_i x_i z_i$ is symmetric under index permutation |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{oktaviani2026kana,
  title={KANA: A Hard-Constrained Chemical-Informed Neural Network for Multi-Component Excess Gibbs Energy Prediction},
  author={Oktaviani, Patricia Yolanda},
  journal={Computer Physics Communications},
  year={2026},
  publisher={Elsevier}
}
```

---

## License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**.  
You may obtain a copy of the License at: https://www.gnu.org/licenses/gpl-3.0

See the `LICENSE` file in the repository root for the full license text.

---

## Acknowledgements

This work was developed as a foundational architecture for thermodynamic deep learning. The author gratefully acknowledges the open-source JAX/Flax ecosystem for enabling differentiable scientific computing at scale.
