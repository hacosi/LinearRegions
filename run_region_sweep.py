#!/usr/bin/env python3
"""Region-tracking sweep: exact linear regions over training, Adam vs SGD.

Sweeps architecture (depth x width) x seed on a low-D task where regions can be
counted EXACTLY (`count_regions(method="grid")`, 1-D/2-D only).  For each cell it
trains a matched Adam/SGD pair from identical initialization and records the
exact region-count trajectory (per epoch) plus test accuracy, so we can see how
the "Adam carves fewer regions" effect scales with depth and width.

    uv run python run_region_sweep.py --depths 1 2 3 4 --widths 16 32 64 128 --seeds 5
    uv run python analysis_region_sweep.py --save figs

One JSON is written to results_regions/ and rewritten after every (depth, width,
seed) cell, so an interrupted run keeps completed cells and can be re-invoked on
the remaining subset.
"""

import argparse
import copy
import json
import os
from datetime import datetime

import torch
import torch.nn as nn

from models import MLP
from datasets import get_dataset
from trainer import train


def parse_args():
    p = argparse.ArgumentParser(description="Exact region tracking across depth/width/seed")
    p.add_argument("--task", default="bullseye",
                   choices=["simple_classification", "bullseye", "simple_regression"],
                   help="Low-D task (needs input_range for exact grid counting)")
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--widths", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--seeds", type=int, default=5, help="seeds 0..seeds-1")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_adam", type=float, default=1e-3)
    p.add_argument("--lr_sgd", type=float, default=1e-2)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--grid_points", type=int, default=20000)
    p.add_argument("--device", default="cpu", help="cpu recommended (tiny nets, grid on CPU)")
    p.add_argument("--out_dir", default="results_regions")
    return p.parse_args()


def train_tracked(init_state, task_info, train_ds, test_ds, opt_name, args, seed, device):
    """Train one model from a fixed init, tracking exact regions every epoch."""
    model = MLP(input_dim=task_info["input_dim"], output_dim=task_info["output_dim"],
                hidden_width=args.width, num_hidden_layers=args.depth)
    model.load_state_dict(copy.deepcopy(init_state))
    model = model.to(device)
    if opt_name == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr_adam)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=args.lr_sgd, momentum=args.momentum)
    is_cls = task_info["type"] == "classification"
    criterion = nn.CrossEntropyLoss() if is_cls else nn.MSELoss()
    torch.manual_seed(seed)  # matched batch order
    h = train(
        model=model, optimizer=optim, criterion=criterion,
        train_dataset=train_ds, test_dataset=test_ds, task_info=task_info,
        loss_type="ce" if is_cls else "mse", batch_size=args.batch_size,
        epochs=args.epochs, device=device,
        count_regions_every=1, region_methods=["grid"],
        grid_points=args.grid_points, seed=seed,
    )
    acc_key = "test_accuracy" if is_cls else "test_loss"
    return {
        "region_epochs": h["linear_regions"]["epochs"],
        "regions": h["linear_regions"]["counts"]["grid"],
        "test_metric": h[acc_key],
        "final_regions": h["linear_regions"]["counts"]["grid"][-1],
        "final_test_metric": h[acc_key][-1],
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Region sweep | task={args.task} depths={args.depths} widths={args.widths} "
          f"seeds={args.seeds} device={device}")

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(args.out_dir, f"regions_{args.task}_{ts}.json")

    cells = []  # one record per (depth, width, seed, optimizer)

    def flush():
        with open(fname, "w") as f:
            json.dump({"config": vars(args), "cells": cells,
                       "timestamp": datetime.now().isoformat()}, f, indent=2)

    for depth in args.depths:
        for width in args.widths:
            args.depth, args.width = depth, width  # consumed by train_tracked
            for seed in range(args.seeds):
                # matched init per (arch, seed); data shared across optimizers
                train_ds, test_ds, task_info = get_dataset(args.task, 1.0, seed)
                torch.manual_seed(seed)
                base = MLP(input_dim=task_info["input_dim"],
                           output_dim=task_info["output_dim"],
                           hidden_width=width, num_hidden_layers=depth)
                init_state = copy.deepcopy(base.state_dict())
                for opt_name in ("adam", "sgd"):
                    res = train_tracked(init_state, task_info, train_ds, test_ds,
                                        opt_name, args, seed, device)
                    cells.append({"depth": depth, "width": width, "seed": seed,
                                  "optimizer": opt_name, **res})
                    flush()
                a = [c for c in cells if c["depth"] == depth and c["width"] == width
                     and c["seed"] == seed]
                fa = next(c["final_regions"] for c in a if c["optimizer"] == "adam")
                fs = next(c["final_regions"] for c in a if c["optimizer"] == "sgd")
                print(f"  d{depth} w{width} s{seed}: final regions "
                      f"adam={fa} sgd={fs} (gap={fs - fa:+d})")

    print(f"\nSaved {fname}")


if __name__ == "__main__":
    main()
