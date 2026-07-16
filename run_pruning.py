#!/usr/bin/env python3
"""Optimizer -> linear regions -> compressibility experiment.

Trains matched Adam/SGD MLP pairs (identical initialization per seed, only the
optimizer differs) and measures, for each model, how far it can be compressed
before accuracy drops:

  * pruning frontiers -- test accuracy vs sparsity for several criteria
                         (magnitude, per-layer magnitude, sensitivity, SynFlow,
                         structured), each run on both the raw weights and a
                         spectral-norm-*balanced* copy (the scale control);
                         one-shot for all, plus fine-tuned for magnitude;
  * quantization frontier -- test accuracy across a weight-bit-width grid;
  * max_sparsity / min_bits_at_drop -- compressibility summaries at a fixed
                         tolerated accuracy drop, per (arm, method);
  * Lipschitz product (weight-scale control) and the repo's own local region
    count, for the mediation analysis.

The hypothesis (applications.md, idea #3): Adam models, with fewer realized
linear regions / fewer effective degrees of freedom, prune to higher sparsity
and tolerate fewer bits than SGD models at matched accuracy.  Phase 1 found the
weight-scale confound hid any such effect under global magnitude pruning; the
scale-invariant criteria and the balanced arm are here to isolate it.

Results are written as one JSON to results_pruning/.  Use analysis_pruning.py
to produce the frontier figures.

Examples
--------
# Fast smoke test on the 2-D bullseye task:
python run_pruning.py --task bullseye --seeds 2 --epochs 5 --finetune_epochs 2

# Full MNIST run:
python run_pruning.py --task mnist --seeds 5 --epochs 15
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
from linear_regions import count_regions
import pruning as P


def parse_args():
    p = argparse.ArgumentParser(description="Optimizer vs. compressibility experiment")

    p.add_argument("--task", default="mnist",
                   choices=["simple_classification", "bullseye", "mnist", "cifar10"],
                   help="Classification task (regression tasks are not supported here)")
    p.add_argument("--seeds", type=int, default=5,
                   help="Number of matched Adam/SGD pairs (one per seed)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--lr_adam", type=float, default=1e-3)
    p.add_argument("--lr_sgd", type=float, default=1e-2)
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--train_fraction", type=float, default=1.0)

    p.add_argument("--sparsities", type=float, nargs="+",
                   default=[0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99],
                   help="Pruning sparsity grid (neuron fraction for 'structured')")
    p.add_argument("--methods", nargs="+",
                   default=["magnitude", "magnitude_local", "sensitivity",
                            "synflow", "structured"],
                   choices=["magnitude", "magnitude_local", "sensitivity",
                            "synflow", "structured"],
                   help="Pruning criteria to sweep")
    p.add_argument("--finetune_methods", nargs="*", default=["magnitude"],
                   help="Methods to also fine-tune (others are one-shot only)")
    p.add_argument("--sensitivity_batches", type=int, default=10,
                   help="Training batches for the sensitivity gradient estimate")
    p.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4, 3, 2],
                   help="Weight quantization bit-widths")
    p.add_argument("--finetune_epochs", type=int, default=5,
                   help="Epochs of post-pruning fine-tuning (same optimizer/lr)")
    p.add_argument("--acc_drop", type=float, default=0.01,
                   help="Tolerated accuracy drop for max_sparsity/min_bits summaries")
    p.add_argument("--skip_quant", action="store_true",
                   help="Skip the quantization frontier")
    p.add_argument("--skip_finetune", action="store_true",
                   help="Skip the fine-tuned pruning frontier (one-shot only)")
    p.add_argument("--skip_balance", action="store_true",
                   help="Skip the spectral-norm-balanced arm (run unbalanced only)")

    p.add_argument("--seed0", type=int, default=0,
                   help="First seed (seeds are seed0..seed0+N-1)")
    p.add_argument("--device", default="auto", help="cpu | cuda | mps | auto")
    p.add_argument("--out_dir", default="results_pruning")

    return p.parse_args()


def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def make_optimizer(model, optimizer_name, args):
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr_adam)
    return torch.optim.SGD(model.parameters(), lr=args.lr_sgd, momentum=args.momentum)


def train_one(init_state, args, task_info, train_ds, test_ds, optimizer_name,
              device, seed):
    """Train one model from a fixed initialization with the given optimizer."""
    model = MLP(input_dim=task_info["input_dim"], output_dim=task_info["output_dim"],
                hidden_width=args.width, num_hidden_layers=args.depth)
    model.load_state_dict(copy.deepcopy(init_state))
    model = model.to(device)

    optim = make_optimizer(model, optimizer_name, args)

    # Reseed so both optimizers see the same batch ordering (matched control).
    torch.manual_seed(seed)
    history = train(
        model=model, optimizer=optim, criterion=nn.CrossEntropyLoss(),
        train_dataset=train_ds, test_dataset=test_ds, task_info=task_info,
        loss_type="ce", batch_size=args.batch_size, epochs=args.epochs,
        device=device, count_regions_every=0,
    )
    return model, history


def _max_sparsity_at_drop(frontier, floor, prefer_finetuned):
    """Highest sparsity whose accuracy stays >= floor (fine-tuned if available)."""
    def acc(e):
        if prefer_finetuned and e["acc_finetuned"] is not None:
            return e["acc_finetuned"]
        return e["acc_oneshot"]
    ok = [e["sparsity"] for e in frontier if acc(e) >= floor]
    return max(ok) if ok else 0.0


def _min_bits_at_drop(quant_frontier, floor):
    ok = [e["bits"] for e in quant_frontier if e["acc"] >= floor]
    return min(ok) if ok else None


def _prune_frontier(model, arm_state, method, train_ds, test_ds, task_info,
                    args, device, optimizer_name, ctx):
    """One method's accuracy-vs-sparsity frontier from a fixed arm state."""
    do_finetune = (not args.skip_finetune and args.finetune_epochs > 0
                   and method in args.finetune_methods)
    frontier = []
    for s in args.sparsities:
        model.load_state_dict(copy.deepcopy(arm_state))
        masks = P.build_mask(model, method, s, ctx)
        P.apply_mask_(model, masks)
        nz, tot = P.count_nonzero(model)
        acc_oneshot = P.evaluate(model, test_ds, task_info, device)

        acc_finetuned = None
        if do_finetune:
            optim = make_optimizer(model, optimizer_name, args)
            P.finetune_with_masks(model, masks, optim, train_ds, task_info,
                                  device, args.finetune_epochs, args.batch_size)
            acc_finetuned = P.evaluate(model, test_ds, task_info, device)

        frontier.append({
            "sparsity": s,
            "realized_sparsity": 1 - nz / tot,
            "acc_oneshot": acc_oneshot,
            "acc_finetuned": acc_finetuned,
        })
    return frontier


def prune_model(model, train_ds, test_ds, task_info, args, device, optimizer_name):
    """Run every pruning criterion (x balanced/unbalanced) + quantization.

    Returns a nested result dict:
      prune_frontiers[arm][method] -> list of per-sparsity entries
      max_sparsity[arm][method]    -> summary scalar
    """
    clean_state = copy.deepcopy(model.state_dict())
    dense_acc = P.evaluate(model, test_ds, task_info, device)
    floor = dense_acc - args.acc_drop
    ctx = {
        "train_ds": train_ds, "task_info": task_info, "device": device,
        "input_dim": task_info["input_dim"],
        "n_batches": args.sensitivity_batches, "batch_size": args.batch_size,
    }

    arms = ["unbalanced"] if args.skip_balance else ["unbalanced", "balanced"]
    prune_frontiers, max_sparsity, dense_acc_arm = {}, {}, {}
    for arm in arms:
        # Balancing is function-preserving, so dense accuracy should match.
        model.load_state_dict(copy.deepcopy(clean_state))
        if arm == "balanced":
            P.balance_spectral_norms_(model)
        arm_state = copy.deepcopy(model.state_dict())
        dense_acc_arm[arm] = P.evaluate(model, test_ds, task_info, device)

        prune_frontiers[arm], max_sparsity[arm] = {}, {}
        for method in args.methods:
            fr = _prune_frontier(model, arm_state, method, train_ds, test_ds,
                                 task_info, args, device, optimizer_name, ctx)
            prune_frontiers[arm][method] = fr
            prefer_ft = method in args.finetune_methods
            max_sparsity[arm][method] = _max_sparsity_at_drop(fr, floor, prefer_ft)

    # ---- quantization frontier (unbalanced arm only) ----
    quant_frontier = []
    if not args.skip_quant:
        for b in args.bits:
            model.load_state_dict(copy.deepcopy(clean_state))
            P.quantize_weights_(model, b)
            quant_frontier.append({
                "bits": b,
                "acc": P.evaluate(model, test_ds, task_info, device),
            })

    model.load_state_dict(clean_state)  # restore dense weights
    return {
        "dense_acc": dense_acc,
        "dense_acc_arm": dense_acc_arm,
        "prune_frontiers": prune_frontiers,
        "quant_frontier": quant_frontier,
        "max_sparsity": max_sparsity,
        "min_bits_at_drop": _min_bits_at_drop(quant_frontier, floor),
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device} | task: {args.task} | seeds: {args.seeds}")

    # ---- data (shared across all seeds/optimizers) ----
    train_ds, test_ds, task_info = get_dataset(args.task, args.train_fraction, args.seed0)
    if task_info["type"] != "classification":
        raise SystemExit("Pruning experiment requires a classification task.")

    # Output file is created up front and rewritten after every model, so an
    # interrupted run keeps all completed (seed, optimizer) records.
    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(
        args.out_dir,
        f"prune_{args.task}_w{args.width}_d{args.depth}_s{args.seeds}_{ts}.json")

    models_out = []

    def flush():
        output = {
            "config": {**vars(args), "device": str(device)},
            "task_info": task_info,
            "models": models_out,
            "timestamp": datetime.now().isoformat(),
        }
        with open(fname, "w") as f:
            json.dump(output, f, indent=2)

    for s in range(args.seed0, args.seed0 + args.seeds):
        # Identical initialization for the matched pair.
        torch.manual_seed(s)
        base = MLP(input_dim=task_info["input_dim"], output_dim=task_info["output_dim"],
                   hidden_width=args.width, num_hidden_layers=args.depth)
        init_state = copy.deepcopy(base.state_dict())
        num_params = sum(p.numel() for p in base.parameters())

        for opt_name in ("adam", "sgd"):
            print(f"\n=== seed {s} | {opt_name} ===")
            model, history = train_one(init_state, args, task_info, train_ds,
                                       test_ds, opt_name, device, s)

            pres = prune_model(model, train_ds, test_ds, task_info, args,
                               device, opt_name)

            # Repo's own local region count + weight-scale control.
            local_regions = count_regions(
                model.to(device), task_info, train_ds, device, method="local",
                n_anchors=100, n_directions=50, local_scale=1.0, seed=s)
            lipschitz = P.lipschitz_product(model.to("cpu"))

            rec = {
                "seed": s, "optimizer": opt_name,
                "train_acc": history["train_accuracy"][-1],
                "test_acc": history["test_accuracy"][-1],
                "num_params": num_params,
                "lipschitz": lipschitz,
                "local_region_count": local_regions,
                **pres,
            }
            models_out.append(rec)
            flush()  # persist after every model

            ms = pres["max_sparsity"]["unbalanced"]
            summ = "  ".join(f"{m}={ms[m]:.2f}" for m in args.methods)
            print(f"  test_acc={rec['test_acc']:.4f} lipschitz={lipschitz:.3g} "
                  f"local_regions={local_regions:.1f}")
            print(f"  max_sparsity@{args.acc_drop} [unbalanced]: {summ}")

    print(f"\nResults saved to {fname}")
    _print_summary(models_out, args)


def _print_summary(models_out, args):
    """Print the Adam-vs-SGD headline comparison."""
    def agg(opt, key_fn):
        vals = [key_fn(m) for m in models_out if m["optimizer"] == opt]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n" + "=" * 66)
    print("SUMMARY (mean over seeds)              Adam        SGD")
    print("-" * 66)
    print(f"  test accuracy                  {agg('adam', lambda m: m['test_acc']):>10.4f}  "
          f"{agg('sgd', lambda m: m['test_acc']):>10.4f}")
    print(f"  lipschitz product              {agg('adam', lambda m: m['lipschitz']):>10.3g}  "
          f"{agg('sgd', lambda m: m['lipschitz']):>10.3g}")
    print(f"  local region count             {agg('adam', lambda m: m['local_region_count']):>10.1f}  "
          f"{agg('sgd', lambda m: m['local_region_count']):>10.1f}")

    print("-" * 66)
    print("  max sparsity @drop (higher = more prunable):")
    arms = ["unbalanced"] if args.skip_balance else ["unbalanced", "balanced"]
    for arm in arms:
        for method in args.methods:
            a = agg('adam', lambda m: m["max_sparsity"][arm][method])
            sg = agg('sgd', lambda m: m["max_sparsity"][arm][method])
            print(f"    {method:<16}[{arm:<10}] {a:>10.2f}  {sg:>10.2f}")
    print(f"  min bits @drop                 {agg('adam', lambda m: m['min_bits_at_drop']):>10.2f}  "
          f"{agg('sgd', lambda m: m['min_bits_at_drop']):>10.2f}")
    print("=" * 66)
    print("Hypothesis: at matched accuracy, Adam >= SGD on max sparsity.\n"
          "Scale-invariant criteria (sensitivity/synflow) and the balanced arm\n"
          "should reveal the region effect if magnitude's scale-sensitivity hid it.")


if __name__ == "__main__":
    main()
