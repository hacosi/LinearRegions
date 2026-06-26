#!/usr/bin/env python3
"""Phase 1-3 figures for the optimizer -> verifiability experiment.

Loads results_verification/*.json (written by run_verification.py) and produces:

  Phase 1 (bridge):    local region density (mean unstable neurons) vs eps,
                       Adam vs SGD.
  Phase 2 (payoff):    CROWN verified rate and certified radius vs eps, plus
                       complete-tier verification time / branchings if present.
  Phase 3 (mechanism): per-point verification difficulty vs unstable-neuron
                       count, both optimizers on one curve -> the mediation
                       scatter; annotated with the weight-scale (Lipschitz)
                       control so the effect can be read independent of scale.

Usage
-----
python analysis_verification.py                       # latest results file
python analysis_verification.py results_verification/verify_*.json
python analysis_verification.py --save figs           # write PNGs to figs/
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

COLORS = {"adam": "C0", "sgd": "C1"}


def load(paths):
    if not paths:
        paths = sorted(glob.glob("results_verification/*.json"))
    if not paths:
        raise SystemExit("No results_verification/*.json files found.")
    runs = [json.load(open(p)) for p in paths]
    # Flatten all model records, tagging each with its eps grid.
    models = []
    for r in runs:
        for m in r["models"]:
            m["_eps_grid"] = r["config"]["eps"]
            models.append(m)
    print(f"Loaded {len(paths)} file(s), {len(models)} model records "
          f"({sum(m['optimizer']=='adam' for m in models)} Adam / "
          f"{sum(m['optimizer']=='sgd' for m in models)} SGD)")
    return models


def _by_opt(models, opt):
    return [m for m in models if m["optimizer"] == opt]


def _mean_sem(rows):
    """Column-wise mean and standard error of a list of equal-length lists."""
    a = np.array(rows, dtype=float)
    mean = a.mean(axis=0)
    sem = a.std(axis=0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.zeros_like(mean)
    return mean, sem


def plot_phase1(models, ax):
    """Bridge metric: mean unstable neurons vs eps, Adam vs SGD."""
    eps = models[0]["_eps_grid"]
    for opt in ("adam", "sgd"):
        ms = _by_opt(models, opt)
        rows = [[pe["unstable_mean"] for pe in m["per_eps"]] for m in ms]
        mean, sem = _mean_sem(rows)
        ax.errorbar(eps, mean, yerr=sem, marker="o", color=COLORS[opt],
                    capsize=3, label=opt.upper())
    ax.set_xlabel(r"perturbation radius $\epsilon$")
    ax.set_ylabel("mean unstable neurons / point")
    ax.set_title("Phase 1: local region density (bridge metric)")
    ax.legend()


def plot_phase2(models, ax_rate, ax_radius):
    """Payoff: CROWN verified rate vs eps, and certified radius distribution."""
    eps = models[0]["_eps_grid"]
    for opt in ("adam", "sgd"):
        ms = _by_opt(models, opt)
        rows = [[pe["crown_verified_rate"] for pe in m["per_eps"]] for m in ms]
        mean, sem = _mean_sem(rows)
        ax_rate.errorbar(eps, mean, yerr=sem, marker="o", color=COLORS[opt],
                         capsize=3, label=opt.upper())
    ax_rate.set_xlabel(r"perturbation radius $\epsilon$")
    ax_rate.set_ylabel("CROWN verified rate")
    ax_rate.set_title("Phase 2: certified robustness")
    ax_rate.legend()

    # Certified-radius distribution (pooled over points & seeds).
    data = []
    for opt in ("adam", "sgd"):
        radii = [r for m in _by_opt(models, opt) for r in m["crown_radius"]]
        data.append(radii)
    parts = ax_radius.violinplot(data, showmeans=True)
    for pc, opt in zip(parts["bodies"], ("adam", "sgd")):
        pc.set_facecolor(COLORS[opt])
        pc.set_alpha(0.5)
    ax_radius.set_xticks([1, 2])
    ax_radius.set_xticklabels(["ADAM", "SGD"])
    ax_radius.set_ylabel("CROWN certified radius")
    ax_radius.set_title("Phase 2: certified radius")


def plot_phase3(models, ax):
    """Mediation scatter: verification difficulty vs unstable-neuron count.

    Per (point, eps): x = unstable neurons, y = whether CROWN failed to certify
    (a 0/1 difficulty proxy). We bin x and plot the failure rate so both
    optimizers fall on one curve -> difficulty is explained by region density,
    not the optimizer label. If the complete tier ran, we instead use the real
    branch count as y.
    """
    use_branch = any(m.get("complete") for m in models)

    xs_all, ys_all, cols = [], [], []
    for opt in ("adam", "sgd"):
        xs, ys = [], []
        for m in _by_opt(models, opt):
            if use_branch and m.get("complete"):
                # branchings at complete_eps vs unstable at the nearest eps
                comp = {c["point"]: c for c in m["complete"]}
                ceps = m["complete"][0]["eps"]
                # unstable counts at the eps closest to complete_eps
                gi = int(np.argmin([abs(pe["eps"] - ceps) for pe in m["per_eps"]]))
                un = m["per_eps"][gi]["unstable"]
                for pt_i, pt in enumerate(m["points"]):
                    c = comp.get(pt)
                    if c and c.get("num_branchings") is not None:
                        xs.append(un[pt_i]); ys.append(c["num_branchings"])
            else:
                for pe in m["per_eps"]:
                    for u, ok in zip(pe["unstable"], pe["crown_verified"]):
                        xs.append(u); ys.append(0 if ok else 1)
        ax.scatter(xs, ys, s=8, alpha=0.25, color=COLORS[opt], label=opt.upper())
        xs_all += xs; ys_all += ys; cols += [opt] * len(xs)

    # One shared trend curve over binned x (the "same curve" claim).
    xs_all = np.array(xs_all); ys_all = np.array(ys_all)
    if len(xs_all) > 5:
        bins = np.linspace(xs_all.min(), xs_all.max(), 8)
        idx = np.digitize(xs_all, bins)
        bx, by = [], []
        for b in range(1, len(bins) + 1):
            sel = idx == b
            if sel.sum() >= 3:
                bx.append(xs_all[sel].mean()); by.append(ys_all[sel].mean())
        ax.plot(bx, by, "k-", lw=2, label="shared trend")

    ax.set_xlabel("unstable neurons over $\\epsilon$-ball")
    ax.set_ylabel("# BaB branchings" if use_branch else "CROWN fail rate")
    ax.set_title("Phase 3: difficulty is set by region density")
    ax.legend()


def print_control_table(models):
    """Weight-scale control: the effect should survive conditioning on Lipschitz."""
    print("\nWeight-scale control (mean over seeds):")
    print(f"{'optimizer':<10}{'test_acc':>10}{'lipschitz':>12}"
          f"{'local_reg':>11}{'crown_radius':>14}")
    for opt in ("adam", "sgd"):
        ms = _by_opt(models, opt)
        f = lambda k: np.mean([m[k] for m in ms])
        print(f"{opt.upper():<10}{f('test_acc'):>10.4f}{f('lipschitz'):>12.3g}"
              f"{f('local_region_count'):>11.1f}{f('crown_radius_mean'):>14.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="result JSON files (default: latest in dir)")
    ap.add_argument("--save", default=None, help="directory to write PNGs instead of showing")
    args = ap.parse_args()

    models = load(args.paths)
    print_control_table(models)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plot_phase1(models, axes[0, 0])
    plot_phase2(models, axes[0, 1], axes[1, 0])
    plot_phase3(models, axes[1, 1])
    fig.tight_layout()

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        out = os.path.join(args.save, "verification_phases.png")
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
