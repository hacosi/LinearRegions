#!/usr/bin/env python3
"""Causal-mediation study: does region count *cause* the downstream costs?

Builds a population of bullseye models spanning a wide range of linear-region
counts by intervening on FOUR knobs other than the optimizer -- weight decay,
input noise, learning rate, label noise -- in a one-knob-at-a-time design around
a baseline.  For each model it records the region count (exact grid + local +
pairwise), the weight-scale control (Lipschitz), and two downstream outcomes:

  * compressibility  -- one-shot pruning-frontier AUC, scale-invariant (SynFlow,
    balanced-magnitude) and scale-sensitive (raw magnitude);
  * verifiability    -- mean unstable neurons and mean IBP certified radius.

analysis_causal.py then shows that outcomes track region count regardless of the
optimizer (both fall on one curve) and that region count survives conditioning on
Lipschitz -- the causal, not merely correlational, claim.

    uv run python run_causal_sweep.py            # full local sweep (~140 models)
    uv run python analysis_causal.py --save figs

One JSON is written to results_causal/ and flushed after every model, so an
interrupted run keeps completed records and can be re-invoked on any subset.
"""

import argparse
import copy
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from models import MLP
from datasets import get_dataset
from trainer import train
from linear_regions import count_regions
import pruning as P
import verification as V


def parse_args():
    p = argparse.ArgumentParser(description="Causal mediation study on bullseye")
    p.add_argument("--task", default="bullseye")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed0", type=int, default=0,
                   help="First seed (seeds are seed0..seed0+seeds-1); for SLURM array tasks")
    p.add_argument("--train_fraction", type=float, default=1.0,
                   help="Subsample training set (use <1 to keep MNIST/CIFAR tractable)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_adam", type=float, default=1e-3)
    p.add_argument("--lr_sgd", type=float, default=1e-2)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--optimizers", nargs="+", default=["adam", "sgd"])
    # intervention knob grids (baseline value first)
    p.add_argument("--weight_decays", type=float, nargs="+",
                   default=[0.0, 1e-4, 1e-3, 1e-2, 5e-2])
    p.add_argument("--noises", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    p.add_argument("--lr_mults", type=float, nargs="+", default=[1.0, 0.3, 3.0, 10.0])
    p.add_argument("--label_noises", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    # measurement grids
    p.add_argument("--sparsities", type=float, nargs="+",
                   default=[0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99])
    p.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    p.add_argument("--eps_max", type=float, default=1.0)
    p.add_argument("--n_verify", type=int, default=30)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_dir", default="results_causal")
    return p.parse_args()


def _materialize(train_ds):
    """Stack a (possibly torchvision/Subset) dataset into (X, y) tensors."""
    Xs = torch.stack([train_ds[i][0].view(-1).float() for i in range(len(train_ds))])
    ys = torch.tensor([int(train_ds[i][1]) for i in range(len(train_ds))])
    return Xs, ys


def apply_label_noise(train_ds, frac, seed, num_classes):
    """TensorDataset with `frac` of labels reassigned to a different random class."""
    Xs, ys = _materialize(train_ds)
    if frac > 0:
        rng = np.random.default_rng(1000 + seed)
        k = int(round(frac * len(ys)))
        idx = rng.choice(len(ys), size=k, replace=False)
        if num_classes == 2:
            ys[idx] = 1 - ys[idx]
        else:                                       # reassign to a different class
            offset = torch.tensor(rng.integers(1, num_classes, size=k))
            ys[idx] = (ys[idx] + offset) % num_classes
    return TensorDataset(Xs, ys)


def apply_input_noise(train_ds, sigma, seed):
    """TensorDataset with i.i.d. Gaussian(0, sigma) added to the (train) inputs.

    Applied uniformly across tasks (torchvision datasets ignore get_dataset's
    `noise` argument, so the input-noise knob is handled here instead).
    """
    Xs, ys = _materialize(train_ds)
    if sigma > 0:
        rng = np.random.default_rng(2000 + seed)
        Xs = Xs + torch.from_numpy(rng.normal(0, sigma, size=Xs.shape).astype("float32"))
    return TensorDataset(Xs, ys)


def settings_grid(args):
    """One-knob-at-a-time settings around the baseline (first value of each knob)."""
    base = dict(weight_decay=args.weight_decays[0], noise=args.noises[0],
                lr_mult=args.lr_mults[0], label_noise=args.label_noises[0])
    out = [("baseline", base.copy())]
    for knob, values in (("weight_decay", args.weight_decays), ("noise", args.noises),
                         ("lr_mult", args.lr_mults), ("label_noise", args.label_noises)):
        for v in values[1:]:                       # skip baseline value
            s = base.copy(); s[knob] = v
            out.append((knob, s))
    return out


def train_model(setting, opt_name, seed, args, device):
    train_ds, test_ds, task_info = get_dataset(args.task, args.train_fraction, seed)
    if setting["noise"] > 0:
        train_ds = apply_input_noise(train_ds, setting["noise"], seed)
    if setting["label_noise"] > 0:
        train_ds = apply_label_noise(train_ds, setting["label_noise"], seed,
                                     task_info.get("num_classes", task_info["output_dim"]))

    torch.manual_seed(seed)
    model = MLP(input_dim=task_info["input_dim"], output_dim=task_info["output_dim"],
                hidden_width=args.width, num_hidden_layers=args.depth).to(device)
    base_lr = args.lr_adam if opt_name == "adam" else args.lr_sgd
    lr = base_lr * setting["lr_mult"]
    if opt_name == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=setting["weight_decay"])
    else:
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum,
                                weight_decay=setting["weight_decay"])
    torch.manual_seed(seed)
    h = train(model=model, optimizer=optim, criterion=nn.CrossEntropyLoss(),
              train_dataset=train_ds, test_dataset=test_ds, task_info=task_info,
              loss_type="ce", batch_size=args.batch_size, epochs=args.epochs,
              device=device, count_regions_every=0)
    return model, train_ds, test_ds, task_info, h["test_accuracy"][-1]


def frontier_auc(model, clean_state, method, args, test_ds, task_info, device, ctx,
                 balanced=False):
    """Mean-normalized AUC of one-shot accuracy over the sparsity grid."""
    accs = []
    for s in args.sparsities:
        model.load_state_dict(copy.deepcopy(clean_state))
        if balanced:
            P.balance_spectral_norms_(model)
        masks = P.build_mask(model, method, s, ctx)
        P.apply_mask_(model, masks)
        accs.append(P.evaluate(model, test_ds, task_info, device))
    model.load_state_dict(clean_state)
    sp = np.array(args.sparsities)
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy 2.x rename
    auc = float(trapezoid(accs, sp) / (sp[-1] - sp[0]))
    return auc, accs


def verify_model(model, test_ds, task_info, args):
    """Mean unstable neurons (per eps) and mean IBP certified radius."""
    model = model.to("cpu").eval()
    # correctly classified test points
    pts = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, y = test_ds[i]
            xf = x.view(-1).float()
            if model(xf.unsqueeze(0)).argmax(1).item() == int(y):
                pts.append((xf, int(y)))
            if len(pts) >= args.n_verify:
                break
    if not pts:
        return {"unstable_mean": {}, "ibp_radius_mean": None, "n_verify": 0}
    unstable = {}
    for eps in args.eps:
        vals = [V.count_unstable_neurons(model, x, eps)[0] for x, _ in pts]
        unstable[str(eps)] = float(np.mean(vals))
    radii = [V.certified_radius(model, x, y, eps_max=args.eps_max) for x, y in pts]
    return {"unstable_mean": unstable, "ibp_radius_mean": float(np.mean(radii)),
            "n_verify": len(pts)}


def main():
    args = parse_args()
    device = torch.device(args.device)
    settings = settings_grid(args)
    seed_range = range(args.seed0, args.seed0 + args.seeds)
    total = len(settings) * len(args.optimizers) * args.seeds
    print(f"Causal sweep | {len(settings)} settings x {args.optimizers} x seeds "
          f"{list(seed_range)} = {total} models | device={device}")

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(args.out_dir,
                         f"causal_{args.task}_w{args.width}_d{args.depth}"
                         f"_s{args.seed0}-{args.seed0 + args.seeds - 1}_{ts}.json")
    records = []

    def flush():
        with open(fname, "w") as f:
            json.dump({"config": vars(args), "records": records,
                       "timestamp": datetime.now().isoformat()}, f, indent=2)

    done = 0
    for knob, setting in settings:
        for opt_name in args.optimizers:
            for seed in seed_range:
                model, train_ds, test_ds, task_info, test_acc = train_model(
                    setting, opt_name, seed, args, device)
                clean_state = copy.deepcopy(model.state_dict())
                ctx = {"train_ds": train_ds, "task_info": task_info, "device": device,
                       "input_dim": task_info["input_dim"], "n_batches": 10,
                       "batch_size": args.batch_size}

                regions = {
                    m: count_regions(model.to(device), task_info, train_ds, device,
                                     method=m, grid_points=20000, n_anchors=100,
                                     n_directions=50, local_scale=1.0, seed=seed)
                    for m in ("grid", "local", "pairwise")
                }
                mag_auc, _ = frontier_auc(model, clean_state, "magnitude", args,
                                          test_ds, task_info, device, ctx)
                bal_auc, _ = frontier_auc(model, clean_state, "magnitude", args,
                                          test_ds, task_info, device, ctx, balanced=True)
                syn_auc, _ = frontier_auc(model, clean_state, "synflow", args,
                                          test_ds, task_info, device, ctx)
                lipschitz = P.lipschitz_product(model.to("cpu"))
                verif = verify_model(model, test_ds, task_info, args)
                model.load_state_dict(clean_state)

                records.append({
                    "knob": knob, "setting": setting, "optimizer": opt_name, "seed": seed,
                    "test_acc": test_acc, "lipschitz": lipschitz,
                    "regions": regions,
                    "compress_auc": {"magnitude": mag_auc, "balanced": bal_auc,
                                     "synflow": syn_auc},
                    "verify": verif,
                })
                flush()
                done += 1
                print(f"  [{done}/{total}] {knob:<12} {opt_name} s{seed}: "
                      f"grid={regions['grid']} local={regions['local']:.1f} "
                      f"lip={lipschitz:.2g} synAUC={syn_auc:.3f} "
                      f"unstable@{args.eps[0]}={verif['unstable_mean'].get(str(args.eps[0])):.1f}")

    print(f"\nSaved {fname}")


if __name__ == "__main__":
    main()
