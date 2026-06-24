[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4+-orange.svg)](https://github.com/google/jax)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# KANA: Hard-Constrained Chemical-Informed Neural Network

> Multi-component excess Gibbs energy prediction with automatic satisfaction of
> Gibbs-Duhem, Gibbs-Helmholtz, and pure-component boundary constraints.

## Scientific Breakthrough

Traditional neural-network models for excess Gibbs energy ($g^E$) are typically validated only on binary mixtures and fail to generalize when scaled to ternary or quaternary systems. **KANA** (Chemical-Informed Neural Network) introduces a **hard-constrained architecture** that:

1. **Scales from binary to quaternary (and beyond)** via a pairwise multiplicative boundary condition $\sum_{i<j} x_i x_j$ that naturally vanishes at pure-component limits regardless of mixture order.
2. **Enforces the Gibbs-Duhem relation exactly** by computing $\ln\gamma_i$ through automatic differentiation of $n g^E/RT$ rather than predicting activity coefficients directly.
3. **Satisfies the Gibbs-Helmholtz relation** by constructing $h^E$ and $s^E$ from temperature derivatives of the same $g^E$ head, guaranteeing thermodynamic consistency.
4. **Maintains strict permutation invariance** through a mole-fraction weighted aggregator $\sum_i x_i z_i / \sum_i x_i$, ensuring that component ordering does not affect mixture predictions.

## Architecture Overview

```
Component i
    +-- Sigma Profile  -->  SigmaEncoder  --+
    +-- Scalar Feats   -->  ScalarEncoder --+
                                            v
                              z_i = [z_a || z_b]
                                            v
                         +-----------------------+
                         | MoleFractionWeighted  |
                         |     Aggregator        |
                         | z_mix = sum x_i z_i   |
                         +-----------------------+
                                            v
                              GE_PredictionHead
                         gE/RT = raw(z_mix, T) * sum_{i<j} x_i x_j
                                            v
                         ThermodynamicEngine (autodiff)
                         +-- ln gamma_i = d(n gE/RT)/dn_i
                         +-- hE = -T^2 * d(gE/T)/dT
                         +-- sE = (hE - gE)/T
```

## Installation

```bash
git clone https://github.com/Flowychie/KANA.git
cd KANA
pip install -r requirements.txt
```

> **Note on JAX:** `requirements.txt` installs CPU-only JAX. For GPU/TPU support, follow the [official JAX installation guide](https://github.com/google/jax#installation).

## Quick Start

### Training with a config preset

```bash
# Preset 1: LATENT=256, LR=1e-3
python -m kana.main --csv-path data/merged_KANA_dataset.csv --output-dir outputs --preset 1

# Preset 2: LATENT=512, LR=1e-3
python -m kana.main --csv-path data/merged_KANA_dataset.csv --output-dir outputs --preset 2

# Preset 3: LATENT=256, LR=1e-4
python -m kana.main --csv-path data/merged_KANA_dataset.csv --output-dir outputs --preset 3

# Preset 4: LATENT=512, LR=1e-4
python -m kana.main --csv-path data/merged_KANA_dataset.csv --output-dir outputs --preset 4

# Custom overrides
python -m kana.main --csv-path data/dataset.csv --preset 1 --lr 5e-4 --dropout 0.2
```

### Using the Python API

```python
from kana import Config, HardConstrainedCINN, ThermodynamicEngine, train
from kana.config import PRESETS

cfg = PRESETS['1']
state, engine, model, val_data = train(cfg, 'data/dataset.csv', 'outputs')

# Access predictions
gE = engine.predict_gE(state.params, model.apply, sigmas, scalars, mask, T, x)
ln_gamma = engine.compute_ln_gamma(state.params, model.apply, sigmas, scalars, mask, T, n)
```

### Generating figures and tables

```bash
cd scripts
python generate_figure1.py   # Parity plot
python generate_figure2.py   # Training curves
python generate_figure3.py   # Residual plot
python generate_figure4.py   # Thermodynamic constraints
python generate_figure5.py   # Error boxplot
python generate_figure6.py   # Error histogram

python generate_table1.py    # Results summary
python generate_table2.py    # Hyperparameters
python generate_table3.py    # Comparison table
python generate_table4.py    # Efficiency table
```

## Config Presets

| Preset | LATENT_A | LATENT_B | Learning Rate |
|--------|----------|----------|---------------|
| 1      | 256      | 256      | 1e-3          |
| 2      | 512      | 512      | 1e-3          |
| 3      | 256      | 256      | 1e-4          |
| 4      | 512      | 512      | 1e-4          |

## Repository Structure

```
KANA/
+-- kana/
|   +-- __init__.py          # Public API
|   +-- config.py            # Config dataclass + 4 presets
|   +-- architecture.py      # Flax modules (encoders, aggregator, head)
|   +-- thermodynamics.py    # ThermodynamicEngine & constraints
|   +-- dataset.py           # Data loading & standardization
|   +-- train.py             # Training loop, evaluation, verification
|   +-- main.py              # CLI entry point
+-- scripts/
|   +-- generate_figure1.py  # Parity plot
|   +-- generate_figure2.py  # Training curves
|   +-- generate_figure3.py  # Residual plot
|   +-- generate_figure4.py  # Thermo constraint metrics
|   +-- generate_figure5.py  # Error boxplot
|   +-- generate_figure6.py  # Error histogram
|   +-- generate_table1.py   # Results summary
|   +-- generate_table2.py   # Hyperparameter configs
|   +-- generate_table3.py   # Model comparison
|   +-- generate_table4.py   # Efficiency comparison
+-- data/                    # Input datasets and parity data CSVs
+-- results/                 # Training run logs
+-- README.md
+-- requirements.txt
+-- LICENSE
```

## Thermodynamic Constraints

| Constraint | Enforcement Mechanism | Mathematical Form |
|---|---|---|
| **Gibbs-Duhem** | Auto-diff of $n g^E/RT$ w.r.t. $n_i$ | $\sum_i x_i \ln\gamma_i = g^E/RT$ |
| **Pure-component boundary** | Pairwise multiplicative mask | $g^E \propto \sum_{i<j} x_i x_j \to 0$ when $x_k \to 1$ |
| **Gibbs-Helmholtz** | Temperature auto-diff of unified $g^E$ head | $h^E = -T^2 \frac{\partial(g^E/T)}{\partial T}$ |
| **Permutation invariance** | Mole-fraction weighted aggregation | $\sum_i x_i z_i$ is symmetric under permutation |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{oktaviani2026kana,
  title={KANA: A Hard-Constrained Chemical-Informed Neural Network
         for Multi-Component Excess Gibbs Energy Prediction},
  author={Oktaviani, Patricia Yolanda},
  journal={Computer Physics Communications},
  year={2026},
  publisher={Elsevier}
}
```

## License

GNU General Public License v3.0 (GPLv3). See [LICENSE](LICENSE).

## Acknowledgements

This work was developed as a foundational architecture for thermodynamic deep learning. The authors gratefully acknowledge the open-source JAX/Flax ecosystem for enabling differentiable scientific computing at scale.
