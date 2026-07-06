#!/usr/bin/env python3
"""Optimizer -> linear regions -> verifiability experiment (Phases 1-3).

Trains matched Adam/SGD MLP pairs (identical initialization per seed, only the
optimizer differs) and measures, for each model, over an L-inf eps grid:

  * unstable-neuron count per eps-ball     -- the "bridge" metric (Phase 1),
                                              local linear-region density in the
                                              units a verifier branches on;
  * IBP + CROWN certified robustness        -- sound, incomplete (Phase 2);
  * IBP + CROWN certified radius;
  * alpha,beta-CROWN complete verification  -- status / time / branchings
                                              (Phase 2, if --complete and the
                                              abcrown CLI is installed);
  * Lipschitz product (weight-scale control) and the repo's own local region
    count, for the Phase-3 mediation analysis.

Results are written as one JSON to results_verification/.  Use
analysis_verification.py to produce the Phase 1-3 figures.

Examples
--------
# Fast smoke test on the 2-D bullseye task (complete tier auto-skips):
python run_verification.py --task bullseye --seeds 2 --epochs 5 --n_points 10

# Full MNIST run with the complete verifier:
python run_verification.py --task mnist --seeds 5 --epochs 15 --complete
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
import verification as V
import complete_verify as CV


def parse_args():
    p = argparse.ArgumentParser(description="Optimizer vs. verifiability experiment")

    p.add_argument("--task", default="mnist",
                   choices=["simple_classification", "bullseye", "mnist", "cifar10"],
                   help="Classification task (regression tasks are not verifiable here)")
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

    p.add_argument("--n_points", type=int, default=50,
                   help="Correctly-classified test points to verify per model")
    p.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2],
                   help="L-inf radii (in normalized input space) for the sweep")
    p.add_argument("--eps_max", type=float, default=0.5,
                   help="Upper bound for certified-radius binary search")
    p.add_argument("--crown_method", default="CROWN-Optimized",
                   choices=["CROWN", "CROWN-Optimized"],
                   help="auto_LiRPA method for the per-eps CROWN certificate")

    p.add_argument("--complete", action="store_true",
                   help="Also run alpha,beta-CROWN complete verification")
    p.add_argument("--complete_eps", type=float, default=None,
                   help="eps for the complete tier (default: median of --eps)")
    p.add_argument("--complete_timeout", type=int, default=120)
    p.add_argument("--complete_n_points", type=int, default=10,
                   help="Points for the (expensive) complete tier; capped at --n_points")

    p.add_argument("--seed0", type=int, default=0, help="First seed (seeds are seed0..seed0+N-1)")
    p.add_argument("--device", default="auto", help="cpu | cuda | mps | auto")
    p.add_argument("--out_dir", default="results_verification")

    return p.parse_args()


def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def train_one(init_state, args, task_info, train_ds, test_ds, optimizer_name,
              device, seed):
    """Train one model from a fixed initialization with the given optimizer."""
    model = MLP(input_dim=task_info["input_dim"], output_dim=task_info["output_dim"],
                hidden_width=args.width, num_hidden_layers=args.depth)
    model.load_state_dict(copy.deepcopy(init_state))
    model = model.to(device)

    if optimizer_name == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr_adam)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=args.lr_sgd,
                                momentum=args.momentum)

    # Reseed so both optimizers see the same batch ordering (matched control).
    torch.manual_seed(seed)
    history = train(
        model=model, optimizer=optim, criterion=nn.CrossEntropyLoss(),
        train_dataset=train_ds, test_dataset=test_ds, task_info=task_info,
        loss_type="ce", batch_size=args.batch_size, epochs=args.epochs,
        device=device, count_regions_every=0,
    )
    return model, history


def pick_correct_points(model, test_ds, n_points, device):
    """Return indices of the first n_points test inputs the model gets right."""
    model.eval()
    idxs = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, y = test_ds[i]
            pred = model(x.unsqueeze(0).to(device)).argmax(1).item()
            if pred == int(y):
                idxs.append(i)
                if len(idxs) >= n_points:
                    break
    return idxs


def verify_model(model, test_ds, point_idxs, args, num_classes, input_dim,
                 complete_eps):
    """Run all verification tiers for one model. Returns a result dict."""
    model = model.to("cpu").eval()
    bounded = V.build_bounded_model(model, input_dim, device="cpu")

    # Per-point pre-extract flattened inputs and labels.
    pts = [(test_ds[i][0].view(-1).float(), int(test_ds[i][1])) for i in point_idxs]

    # ---- per-eps metrics (bridge metric + verified rates) ----
    per_eps = []
    for eps in args.eps:
        unstable, ibp_ok, crown_ok = [], [], []
        for x, y in pts:
            unstable.append(V.count_unstable_neurons(model, x, eps)[0])
            ibp_ok.append(V.certified_robust_ibp(model, x, y, eps))
            crown_ok.append(V.certified_robust_crown(
                model, x, y, eps, num_classes, method=args.crown_method,
                bounded_model=bounded, input_dim=input_dim))
        n = len(pts)
        per_eps.append({
            "eps": eps,
            "unstable": unstable,
            "unstable_mean": sum(unstable) / n,
            "ibp_verified": ibp_ok,
            "ibp_verified_rate": sum(ibp_ok) / n,
            "crown_verified": crown_ok,
            "crown_verified_rate": sum(crown_ok) / n,
        })

    # ---- certified radii (one number per point) ----
    ibp_radius, crown_radius = [], []
    for x, y in pts:
        ibp_radius.append(V.certified_radius(model, x, y, eps_max=args.eps_max))
        crown_radius.append(V.certified_radius_crown(
            model, x, y, num_classes, eps_max=args.eps_max, method="CROWN",
            bounded_model=bounded, input_dim=input_dim))

    # ---- complete tier (optional, capped at complete_n_points) ----
    complete = None
    if args.complete:
        complete = []
        for idx, (x, y) in list(zip(point_idxs, pts))[:args.complete_n_points]:
            res = CV.run_complete(model, x, y, complete_eps, num_classes,
                                  input_dim=input_dim, timeout=args.complete_timeout)
            res["point"] = idx
            res["eps"] = complete_eps
            complete.append(res)

    n = len(pts)
    return {
        "points": point_idxs,
        "n_points": n,
        "per_eps": per_eps,
        "ibp_radius": ibp_radius,
        "ibp_radius_mean": sum(ibp_radius) / n if n else None,
        "crown_radius": crown_radius,
        "crown_radius_mean": sum(crown_radius) / n if n else None,
        "complete": complete,
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    complete_eps = args.complete_eps if args.complete_eps is not None \
        else sorted(args.eps)[len(args.eps) // 2]
    print(f"Device: {device} | task: {args.task} | seeds: {args.seeds}")
    if args.complete:
        print(f"Complete tier: abcrown available = {CV.is_available()} "
              f"(eps={complete_eps})")

    # ---- data (shared across all seeds/optimizers) ----
    train_ds, test_ds, task_info = get_dataset(args.task, args.train_fraction, args.seed0)
    if task_info["type"] != "classification":
        raise SystemExit("Verification experiment requires a classification task.")
    num_classes = task_info.get("num_classes", task_info["output_dim"])
    input_dim = task_info["input_dim"]

    # Output file is created up front and rewritten after every model, so an
    # interrupted run keeps all completed (seed, optimizer) records.
    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(
        args.out_dir,
        f"verify_{args.task}_w{args.width}_d{args.depth}_s{args.seeds}_{ts}.json")

    models_out = []

    def flush():
        output = {
            "config": {**vars(args), "device": str(device), "complete_eps": complete_eps},
            "task_info": task_info,
            "models": models_out,
            "timestamp": datetime.now().isoformat(),
        }
        with open(fname, "w") as f:
            json.dump(output, f, indent=2)

    for s in range(args.seed0, args.seed0 + args.seeds):
        # Identical initialization for the matched pair.
        torch.manual_seed(s)
        base = MLP(input_dim=input_dim, output_dim=task_info["output_dim"],
                   hidden_width=args.width, num_hidden_layers=args.depth)
        init_state = copy.deepcopy(base.state_dict())
        num_params = sum(p.numel() for p in base.parameters())

        for opt_name in ("adam", "sgd"):
            print(f"\n=== seed {s} | {opt_name} ===")
            model, history = train_one(init_state, args, task_info, train_ds,
                                       test_ds, opt_name, device, s)

            point_idxs = pick_correct_points(model, test_ds, args.n_points, device)
            vres = verify_model(model, test_ds, point_idxs, args, num_classes,
                                input_dim, complete_eps)

            # Repo's own local region count + weight-scale control.
            local_regions = count_regions(
                model.to(device), task_info, train_ds, device, method="local",
                n_anchors=100, n_directions=50, local_scale=1.0, seed=s)
            lipschitz = V.lipschitz_product(model.to("cpu"))

            rec = {
                "seed": s, "optimizer": opt_name,
                "train_acc": history["train_accuracy"][-1],
                "test_acc": history["test_accuracy"][-1],
                "num_params": num_params,
                "lipschitz": lipschitz,
                "local_region_count": local_regions,
                **vres,
            }
            models_out.append(rec)
            flush()  # persist after every model
            cstat = ""
            if vres["complete"]:
                nb = [c.get("num_branchings") for c in vres["complete"]
                      if c.get("num_branchings") is not None]
                vr = sum(c["status"] == "verified" for c in vres["complete"])
                cstat = (f" complete[verified={vr}/{len(vres['complete'])} "
                         f"mean_branch={sum(nb)/len(nb):.1f}]" if nb else
                         f" complete[verified={vr}/{len(vres['complete'])}]")
            print(f"  test_acc={rec['test_acc']:.4f} lipschitz={lipschitz:.3g} "
                  f"local_regions={local_regions:.1f} "
                  f"unstable@{args.eps[0]}={vres['per_eps'][0]['unstable_mean']:.1f} "
                  f"crown_radius_mean={vres['crown_radius_mean']:.4f}{cstat}")

    print(f"\nResults saved to {fname}")
    _print_summary(models_out, args.eps)


def _print_summary(models_out, eps_grid):
    """Print the Adam-vs-SGD headline comparison."""
    def agg(opt, key_fn):
        vals = [key_fn(m) for m in models_out if m["optimizer"] == opt]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n" + "=" * 60)
    print("SUMMARY (mean over seeds)        Adam        SGD")
    print("-" * 60)
    print(f"  test accuracy            {agg('adam', lambda m: m['test_acc']):>10.4f}  "
          f"{agg('sgd', lambda m: m['test_acc']):>10.4f}")
    print(f"  lipschitz product        {agg('adam', lambda m: m['lipschitz']):>10.3g}  "
          f"{agg('sgd', lambda m: m['lipschitz']):>10.3g}")
    print(f"  local region count       {agg('adam', lambda m: m['local_region_count']):>10.1f}  "
          f"{agg('sgd', lambda m: m['local_region_count']):>10.1f}")
    for i, e in enumerate(eps_grid):
        a = agg('adam', lambda m: m['per_eps'][i]['unstable_mean'])
        s = agg('sgd', lambda m: m['per_eps'][i]['unstable_mean'])
        print(f"  unstable neurons @{e:<5}   {a:>10.1f}  {s:>10.1f}")
    print(f"  CROWN certified radius   {agg('adam', lambda m: m['crown_radius_mean']):>10.4f}  "
          f"{agg('sgd', lambda m: m['crown_radius_mean']):>10.4f}")
    print("=" * 60)
    print("Hypothesis: Adam < SGD on unstable neurons (at matched accuracy),\n"
          "and Adam >= SGD on certified radius.")


if __name__ == "__main__":
    main()
