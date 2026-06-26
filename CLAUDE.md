# CLAUDE.md

Research codebase for counting the **linear regions** of ReLU MLPs over the course of
training, while varying the optimizer (Adam vs. SGD), loss, architecture, and task.

## Setup & running

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`).

```bash
uv sync                                  # install deps into .venv
uv run run-experiment --task mnist --optimizer adam --loss ce   # console script
uv run python run_experiment.py --help   # equivalent; see all flags
```

`run_experiment.py` runs exactly one experiment. There is no batch/sweep runner — drive
multiple configs from the shell. Device is auto-detected (`cuda` → `mps` → `cpu`); override
with `--device`.

**Python is pinned to 3.11** (`.python-version`, `requires-python = ">=3.11,<3.12"`). This is
required by `auto_LiRPA` (the verification dependency), whose current release supports 3.11
only. The original training/region code runs on 3.10+; the pin is purely for the verification
experiment below.

## Layout

| File | Role |
|------|------|
| `run_experiment.py` | CLI entry point (also `run-experiment` script). Parses args, builds model/optimizer/loss, runs training, writes JSON. |
| `models.py` | `MLP(input_dim, output_dim, hidden_width, num_hidden_layers)`. ReLU between hidden layers; depth 0 = single `Linear` (no ReLUs). |
| `datasets.py` | `get_dataset(task, train_fraction, seed, noise)` → `(train_ds, test_ds, task_info)`. |
| `trainer.py` | `train(...)` loop: loss/accuracy/timing per epoch, optional linear-region counting. |
| `linear_regions.py` | `count_regions(...)` dispatcher + the three counting methods. |
| `analysis.ipynb` | Loads result JSONs and plots task data + training/region curves. |
| `run_verification.py` | CLI entry point (`run-verification`) for the verifiability experiment. Trains matched Adam/SGD pairs, runs all verification tiers, writes JSON to `results_verification/`. |
| `verification.py` | IBP toolkit (`count_unstable_neurons`, `certified_radius`, …) + auto_LiRPA CROWN tier (`certified_robust_crown`, `certified_radius_crown`) + `lipschitz_product`. |
| `complete_verify.py` | α,β-CROWN complete-verifier integration (ONNX + VNNLIB export, `abcrown` CLI subprocess); guarded — skips if the CLI is absent. |
| `analysis_verification.py` | Loads `results_verification/*.json` and produces the Phase 1–3 figures. |

## Tasks (`datasets.py`)

- **Synthetic, low-D** (carry an `input_range`, so exact `grid` counting works):
  `simple_classification` (2-D, 2 Gaussian blobs), `simple_regression` (1-D, `y = sin(x)`),
  `bullseye` (2-D concentric circles).
- **Benchmarks** (no `input_range`): `mnist` (784-D), `cifar10` (3072-D). Auto-downloaded to
  `./data/`.

`task_info` carries `type` (`classification`/`regression`), `input_dim`, `output_dim`, and
`input_range` (synthetic only). `--noise` and `--train_fraction` apply to any task.

## Linear-region counting (`linear_regions.py`)

A linear region = a unique binary ReLU activation pattern across all hidden units. Counting is
off by default; enable with `--count_regions_every N` (also counts at epoch 0, pre-training).
Pass one or more `--region_method`:

- **`grid`** — exact unique-pattern count over a uniform grid. 1-D/2-D only (needs `input_range`).
- **`pairwise`** — approximation: average unique patterns sampled along lines between random
  training-point pairs. Any dimension. Returns a float.
- **`local`** — approximation: average unique patterns from orthonormal probes around random
  anchor points, scaled by `--local_scale` × the data's RMS per-feature std. Any dimension.
  Returns a float.
- **`auto`** (default) — `grid` if `input_range` exists, else `pairwise`.

Depth-0 models always report 1 region.

## Key flags (`run_experiment.py`)

- Architecture: `--width` (128), `--depth` (2 hidden layers).
- Optimizer/loss: `--optimizer` (adam/sgd), `--loss` (ce/mse). CE on a regression task falls
  back to MSE with a warning; MSE on classification one-hot-encodes targets (`trainer.py`).
- Hyperparameters: `--lr` (default 1e-3 Adam / 1e-2 SGD), `--batch_size` (64), `--b1`
  (Adam β₁ **or** SGD momentum), `--b2` (Adam β₂ only), `--epochs` (50), `--seed` (42).

## Results

Each run writes one JSON to `results/` named
`{task}_{optimizer}_{loss}_w{width}_d{depth}_{timestamp}.json`, containing `config`,
`task_info`, `num_params`, `results` (per-epoch history + optional `linear_regions`), and
`timestamp`. `results/` is created on demand; committed `results_mnist/` and `results_cifar/`
hold prior runs. `analysis.ipynb` loads via a glob pattern (e.g. `results_mnist/*.json`).

## Verifiability experiment (`run_verification.py`)

Tests the claim (see `applications.md` idea #1) that Adam's lower local linear-region density
makes networks **cheaper and stronger to formally verify**. Trains matched Adam/SGD pairs from
**identical initialization per seed** (only the optimizer differs), then measures three tiers
over an L-∞ ε grid:

1. **IBP** (`verification.py`, dependency-free) — `count_unstable_neurons` (the *bridge metric*:
   ReLUs whose pre-activation straddles 0 over the ε-ball = local region density in verifier
   units), plus a sound `certified_radius`.
2. **CROWN / α-CROWN** (`verification.py` via `auto_LiRPA`) — tighter sound certificates and
   radii (`certified_robust_crown`, `certified_radius_crown`).
3. **α,β-CROWN complete** (`complete_verify.py`) — true `status` / `verification_time` /
   `num_branchings`. **Optional**: requires the external `abcrown` CLI; auto-skips otherwise.

Soundness ordering (useful as a run sanity check): `radius_IBP ≤ radius_CROWN ≤ radius_complete`.

```bash
# Fast smoke test (complete tier auto-skips):
uv run python run_verification.py --task bullseye --seeds 2 --epochs 5 --n_points 10
# Full run with complete verifier (needs abcrown installed):
uv run python run_verification.py --task mnist --seeds 5 --epochs 15 --complete
uv run python analysis_verification.py --save figs        # Phase 1–3 plots
```

Output JSON → `results_verification/`, one record per (seed, optimizer) with `per_eps`
(unstable counts + IBP/CROWN verified flags), `ibp_radius` / `crown_radius`, `lipschitz`
(weight-scale control), `local_region_count`, and optional `complete`. Key controls:
**matched initialization**, **accuracy matching** (compare only within an overlapping-accuracy
band — the script prints both optimizers' accuracy), and the **Lipschitz** column to separate a
region-count effect from a pure weight-rescaling effect.

**Enabling the complete tier (α,β-CROWN):** it is a separate tool, not a pip dependency. Install
[alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN), then point the
integration at it:

```bash
export ABCROWN_PATH=/path/to/alpha-beta-CROWN/complete_verifier/abcrown.py
export ABCROWN_PYTHON=/path/to/that/env/bin/python   # optional; defaults to current interp
```

`complete_verify.py` self-tests its dependency-free parts (ONNX export, VNNLIB, output parsing)
via `uv run python complete_verify.py`.
