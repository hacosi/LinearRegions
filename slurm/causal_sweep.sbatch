#!/bin/bash
# Causal-mediation sweep (run_causal_sweep.py) on MNIST + CIFAR-10, MIT Engaging.
#
# One array task per (task, seed): index 0-4 = MNIST seeds 0-4, 5-9 = CIFAR seeds 0-4.
# Each task runs the full 4-knob population for its seed (both optimizers) and writes
# its own JSON (flushed per model, so a preempted task resumes cleanly on resubmit).
#
#   sbatch slurm/causal_sweep.sbatch
#   # when done, pool + plot per task:
#   uv run python analysis_causal.py results_causal_mnist/*.json   --save figs_mnist
#   uv run python analysis_causal.py results_causal_cifar10/*.json --save figs_cifar
#
#SBATCH --job-name=causal-regions
#SBATCH --array=0-9
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/causal_%A_%a.out
#SBATCH --error=logs/causal_%A_%a.err
# ---- EDIT for your Engaging allocation ----
#SBATCH --partition=mit_normal        # e.g. mit_normal, sched_mit_hill, newnodes
# #SBATCH --account=<your_account>
# ---- For faster training on a GPU node, uncomment: ----
# #SBATCH --gres=gpu:1
# #SBATCH --partition=mit_normal_gpu

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# ---- Environment (pick one) ----
# A) uv is available on the node:
#      RUN="uv run python"
# B) prebuilt venv (recommended on clusters): run `uv sync` (or build a py3.11 venv
#    and `pip install -e .`) once on a login node, then:
#      module load anaconda3/2023.07        # or any python/3.11 module
#      source .venv/bin/activate; RUN="python"
RUN="${RUN:-uv run python}"

# ---- Map array index -> (task, seed) ----
TASKS=(mnist cifar10)
i=${SLURM_ARRAY_TASK_ID}
task=${TASKS[$((i / 5))]}
seed=$((i % 5))

# ---- Sweep config (edit as desired) ----
WIDTH=128
DEPTH=2
EPOCHS=20
TRAIN_FRACTION=0.5        # 1.0 for the full training set if the time budget allows

echo "[$(date)] array $i -> task=$task seed=$seed"
$RUN run_causal_sweep.py \
    --task "$task" --seed0 "$seed" --seeds 1 \
    --width "$WIDTH" --depth "$DEPTH" --epochs "$EPOCHS" \
    --train_fraction "$TRAIN_FRACTION" --device auto \
    --out_dir "results_causal_${task}"
echo "[$(date)] done array $i"
