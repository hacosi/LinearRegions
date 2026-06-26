#!/usr/bin/env python3
"""Run a single ML experiment.

Examples
--------
# Adam on MNIST with default hyperparameters
python run_experiment.py --task mnist --optimizer adam --loss ce

# SGD on the bullseye task, narrow & deep network, 50 % of training data
python run_experiment.py --task bullseye --optimizer sgd --loss ce \
    --width 64 --depth 4 --lr 0.01 --b1 0.9 --train_fraction 0.5

# Regression with MSE
python run_experiment.py --task simple_regression --optimizer adam --loss mse \
    --width 256 --depth 3 --epochs 200
"""

import argparse
import json
import os
import time
from datetime import datetime

import torch
import torch.nn as nn

from models import MLP
from datasets import get_dataset
from trainer import train


def parse_args():
    p = argparse.ArgumentParser(description="ML Experiment Runner")

    # What to run
    p.add_argument("--task", required=True,
                   choices=["simple_classification", "simple_regression",
                            "bullseye", "mnist", "cifar10"])
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--loss", default="ce", choices=["mse", "ce"])

    # Architecture
    p.add_argument("--width", type=int, default=128,
                   help="Hidden-layer width")
    p.add_argument("--depth", type=int, default=2,
                   help="Number of hidden layers")

    # Hyperparameters
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (default: 0.001 for Adam, 0.01 for SGD)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--b1", type=float, default=0.9,
                   help="Beta-1 (Adam) / momentum (SGD)")
    p.add_argument("--b2", type=float, default=0.999,
                   help="Beta-2 (Adam only; ignored for SGD)")

    # Data
    p.add_argument("--noise", type=float, default=0.0,
                   help="Stddev of Gaussian noise added to inputs (1-D/2-D tasks only)")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--train_fraction", type=float, default=1.0,
                   help="Fraction of training set to use, in (0, 1]")

    # Linear region counting
    p.add_argument("--count_regions_every", type=int, default=0,
                   help="Count linear regions every N epochs (0 = disabled)")
    p.add_argument("--region_method", default=["auto"], nargs="+",
                   choices=["auto", "grid", "pairwise", "local"],
                   help="One or more counting methods (e.g. --region_method pairwise local)")
    p.add_argument("--grid_points", type=int, default=10000,
                   help="Grid points for grid method")
    p.add_argument("--n_pairs", type=int, default=100,
                   help="Number of point pairs for pairwise method")
    p.add_argument("--n_line_samples", type=int, default=100,
                   help="Samples per line for pairwise method")
    p.add_argument("--n_anchors", type=int, default=100,
                   help="Number of anchor points for local method")
    p.add_argument("--n_directions", type=int, default=100,
                   help="Orthonormal directions per anchor for local method")
    p.add_argument("--local_scale", type=float, default=1,
                   help="Probe distance as a fraction of the RMS per-feature std of the training data")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto",
                   help="cpu | cuda | mps | auto")

    return p.parse_args()


def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main():
    args = parse_args()
    if args.lr is None:
        args.lr = 1e-3 if args.optimizer == "adam" else 1e-2
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    # ---- data ----
    train_ds, test_ds, task_info = get_dataset(
        args.task, args.train_fraction, args.seed, noise=args.noise,
    )
    print(f"Task: {args.task}  |  train={len(train_ds)}  test={len(test_ds)}")

    # ---- model ----
    model = MLP(
        input_dim=task_info["input_dim"],
        output_dim=task_info["output_dim"],
        hidden_width=args.width,
        num_hidden_layers=args.depth,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,}")

    # ---- optimizer ----
    if args.optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(args.b1, args.b2))
    else:
        optim = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.b1)

    # ---- loss ----
    is_cls = task_info["type"] == "classification"
    loss_type = args.loss

    if loss_type == "ce" and not is_cls:
        print("Warning: CE loss is not applicable to regression tasks. "
              "Falling back to MSE.")
        loss_type = "mse"

    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    # ---- train ----
    results = train(
        model=model,
        optimizer=optim,
        criterion=criterion,
        train_dataset=train_ds,
        test_dataset=test_ds,
        task_info=task_info,
        loss_type=loss_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
        count_regions_every=args.count_regions_every,
        region_methods=args.region_method,
        grid_points=args.grid_points,
        n_pairs=args.n_pairs,
        n_line_samples=args.n_line_samples,
        n_anchors=args.n_anchors,
        n_directions=args.n_directions,
        local_scale=args.local_scale,
        seed=args.seed,
    )

    # ---- save ----
    config = vars(args)
    config["device"] = str(device)
    config["loss"] = loss_type  # reflect any fallback

    output = {
        "config": config,
        "task_info": task_info,
        "num_params": num_params,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = (f"results/{args.task}_{args.optimizer}_{loss_type}_"
             f"w{args.width}_d{args.depth}_{ts}.json")

    with open(fname, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {fname}")


if __name__ == "__main__":
    main()
