#!/usr/bin/env python3
"""Quick sanity check: exact linear-region count over training, Adam vs SGD.

Bullseye is 2-D, so `count_regions(..., method='grid')` gives the EXACT number of
linear regions (unique ReLU activation patterns over a dense grid) -- no
approximation, unlike the local/pairwise estimators we must use on MNIST.

Trains matched Adam/SGD pairs from identical init per seed and plots the exact
region count (and test accuracy) vs epoch, mean +/- SEM over seeds.

    uv run python sanity_bullseye_regions.py
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models import MLP
from datasets import get_dataset
from trainer import train

SEEDS = 5
EPOCHS = 60
WIDTH = 64
DEPTH = 2
COLORS = {"adam": "C0", "sgd": "C1"}


def run_one(init_state, task_info, train_ds, test_ds, opt_name, seed, device):
    model = MLP(input_dim=task_info["input_dim"], output_dim=task_info["output_dim"],
                hidden_width=WIDTH, num_hidden_layers=DEPTH)
    model.load_state_dict(copy.deepcopy(init_state))
    model = model.to(device)
    if opt_name == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    torch.manual_seed(seed)  # matched batch order
    history = train(
        model=model, optimizer=optim, criterion=nn.CrossEntropyLoss(),
        train_dataset=train_ds, test_dataset=test_ds, task_info=task_info,
        loss_type="ce", batch_size=64, epochs=EPOCHS, device=device,
        count_regions_every=1, region_methods=["grid"], grid_points=20000, seed=seed,
    )
    return history


def main():
    device = torch.device("cpu")  # tiny task; keeps grid counting simple
    train_ds, test_ds, task_info = get_dataset("bullseye", 1.0, 0)
    print(f"bullseye: input_range={task_info.get('input_range')} "
          f"(exact grid counting enabled)")

    curves = {"adam": {"regions": [], "acc": []}, "sgd": {"regions": [], "acc": []}}
    epochs_axis = None
    for s in range(SEEDS):
        torch.manual_seed(s)
        base = MLP(input_dim=task_info["input_dim"], output_dim=task_info["output_dim"],
                   hidden_width=WIDTH, num_hidden_layers=DEPTH)
        init_state = copy.deepcopy(base.state_dict())
        for opt_name in ("adam", "sgd"):
            h = run_one(init_state, task_info, train_ds, test_ds, opt_name, s, device)
            curves[opt_name]["regions"].append(h["linear_regions"]["counts"]["grid"])
            curves[opt_name]["acc"].append([1.0] + h["test_accuracy"])  # epoch0 has no acc
            epochs_axis = h["linear_regions"]["epochs"]
            print(f"seed {s} {opt_name}: final regions="
                  f"{h['linear_regions']['counts']['grid'][-1]}  "
                  f"test_acc={h['test_accuracy'][-1]:.3f}")

    def mean_sem(rows):
        a = np.array(rows, dtype=float)
        return a.mean(0), (a.std(0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0 * a.mean(0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for opt in ("adam", "sgd"):
        m, e = mean_sem(curves[opt]["regions"])
        ax1.plot(epochs_axis, m, color=COLORS[opt], label=opt.upper())
        ax1.fill_between(epochs_axis, m - e, m + e, color=COLORS[opt], alpha=0.2)
        ma, ea = mean_sem(curves[opt]["acc"])
        ax2.plot(range(len(ma)), ma, color=COLORS[opt], label=opt.upper())
        ax2.fill_between(range(len(ma)), ma - ea, ma + ea, color=COLORS[opt], alpha=0.2)

    ax1.set_xlabel("epoch"); ax1.set_ylabel("exact linear regions (grid)")
    ax1.set_title(f"Exact region count vs training (bullseye, w{WIDTH} d{DEPTH}, "
                  f"{SEEDS} seeds)")
    ax1.legend()
    ax2.set_xlabel("epoch"); ax2.set_ylabel("test accuracy")
    ax2.set_title("Test accuracy vs training")
    ax2.legend()
    fig.tight_layout()

    import os
    os.makedirs("figs", exist_ok=True)
    out = "figs/bullseye_region_sanity.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")

    # headline numbers
    fa = mean_sem(curves["adam"]["regions"])[0][-1]
    fs = mean_sem(curves["sgd"]["regions"])[0][-1]
    print(f"final exact regions:  Adam={fa:.1f}  SGD={fs:.1f}  "
          f"({'Adam fewer' if fa < fs else 'SGD fewer'})")


if __name__ == "__main__":
    main()
