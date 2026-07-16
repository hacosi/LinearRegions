#!/usr/bin/env python3
"""Figures for the optimizer -> linear regions -> compressibility experiment.

Loads results_pruning/*.json (written by run_pruning.py) and produces:

  1. Pruning frontiers:  small-multiples grid, one panel per criterion
                         (magnitude, per-layer magnitude, sensitivity, SynFlow,
                         structured), test accuracy vs sparsity, Adam vs SGD,
                         one-shot (dashed) + fine-tuned (solid) where present.
  2. Magnitude A/B:      the same magnitude frontier unbalanced vs spectral-norm
                         -balanced -- the direct read on the weight-scale confound.
  3. Mechanism scatter:  max_sparsity_at_drop vs local region count, for a
                         scale-invariant criterion (sensitivity) contrasted with
                         scale-sensitive magnitude; marker size ~ Lipschitz.

Usage
-----
python analysis_pruning.py                       # latest results file
python analysis_pruning.py results_pruning/prune_*.json
python analysis_pruning.py --save figs           # write PNGs to figs/
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

COLORS = {"adam": "C0", "sgd": "C1"}


def _shim(m):
    """Upgrade a Phase-1 record (flat prune_frontier) to the nested schema."""
    if "prune_frontiers" not in m:
        fr = m.get("prune_frontier", [])
        m["prune_frontiers"] = {"unbalanced": {"magnitude": fr}}
        ms = m.get("max_sparsity_at_drop")
        m["max_sparsity"] = {"unbalanced": {"magnitude": ms}} if ms is not None else {}
    return m


def load(paths):
    if not paths:
        paths = sorted(glob.glob("results_pruning/*.json"))
    if not paths:
        raise SystemExit("No results_pruning/*.json files found.")
    runs = [json.load(open(p)) for p in paths]
    # Dedup by (seed, optimizer), keeping the last occurrence -- lets a run be
    # completed across several files (e.g. --seed0 resumes) without double-counting.
    uniq = {}
    for r in runs:
        for m in r["models"]:
            uniq[(m["seed"], m["optimizer"])] = _shim(m)
    models = list(uniq.values())
    print(f"Loaded {len(paths)} file(s), {len(models)} model records "
          f"({sum(m['optimizer']=='adam' for m in models)} Adam / "
          f"{sum(m['optimizer']=='sgd' for m in models)} SGD)")
    return models


def _by_opt(models, opt):
    return [m for m in models if m["optimizer"] == opt]


def _methods(models):
    """Union of criteria across records (ordered by first appearance).

    Robust to mixing Phase-1 (magnitude-only) files with newer multi-method ones.
    """
    seen = []
    for m in models:
        for method in m["prune_frontiers"].get("unbalanced", {}):
            if method not in seen:
                seen.append(method)
    return seen


def _mean_sem(rows):
    a = np.array(rows, dtype=float)
    mean = np.nanmean(a, axis=0)
    sem = (np.nanstd(a, axis=0, ddof=1) / np.sqrt(len(a))
           if len(a) > 1 else np.zeros_like(mean))
    return mean, sem


def _frontier_curve(ax, models, arm, method, key, ls, label_suffix):
    """Plot mean +/- SEM of one accuracy key over the sparsity grid."""
    drew = False
    for opt in ("adam", "sgd"):
        rows, xs = [], None
        for m in _by_opt(models, opt):
            fr = m["prune_frontiers"].get(arm, {}).get(method)
            if not fr:
                continue
            xs = [e["sparsity"] for e in fr]
            rows.append([e[key] if e[key] is not None else np.nan for e in fr])
        if not rows or all(np.all(np.isnan(r)) for r in rows):
            continue
        mean, sem = _mean_sem(rows)
        ax.errorbar(xs, mean, yerr=sem, marker="o", ls=ls, color=COLORS[opt],
                    capsize=3, label=f"{opt.upper()} {label_suffix}")
        drew = True
    return drew


def plot_frontier_grid(models, fig):
    """One panel per criterion: accuracy vs sparsity, Adam vs SGD."""
    methods = _methods(models)
    axes = fig.subplots(1, len(methods), sharey=True, squeeze=False)[0]
    for ax, method in zip(axes, methods):
        _frontier_curve(ax, models, "unbalanced", method, "acc_oneshot", "--", "one-shot")
        _frontier_curve(ax, models, "unbalanced", method, "acc_finetuned", "-", "fine-tuned")
        ax.set_xlabel("sparsity")
        ax.set_title(method)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("test accuracy")


def plot_magnitude_ab(models, ax):
    """Magnitude frontier: unbalanced vs spectral-norm-balanced (confound A/B)."""
    have_bal = any("balanced" in m["prune_frontiers"] for m in models)
    _frontier_curve(ax, models, "unbalanced", "magnitude", "acc_oneshot", "--", "unbal")
    if have_bal:
        _frontier_curve(ax, models, "balanced", "magnitude", "acc_oneshot", "-", "balanced")
    ax.set_xlabel("sparsity")
    ax.set_ylabel("test accuracy (one-shot)")
    ax.set_title("Magnitude pruning: raw vs scale-balanced")
    ax.legend(fontsize=7)


def plot_mechanism(models, ax):
    """max_sparsity vs local region count: scale-sensitive vs scale-invariant."""
    def scatter(method, marker, arm="unbalanced"):
        for opt in ("adam", "sgd"):
            ms = [m for m in _by_opt(models, opt)
                  if m.get("max_sparsity", {}).get(arm, {}).get(method) is not None]
            if not ms:
                continue
            x = [m["local_region_count"] for m in ms]
            y = [m["max_sparsity"][arm][method] for m in ms]
            lips = np.array([m["lipschitz"] for m in ms])
            sizes = 40 + 160 * (lips - lips.min()) / (lips.max() - lips.min() + 1e-12)
            ax.scatter(x, y, s=sizes, alpha=0.6, color=COLORS[opt], marker=marker,
                       edgecolor="k", linewidth=0.5,
                       label=f"{opt.upper()} {method}")
    scatter("magnitude", "o")
    if any("sensitivity" in m["prune_frontiers"].get("unbalanced", {}) for m in models):
        scatter("sensitivity", "^")
    ax.set_xlabel("local region count")
    ax.set_ylabel("max sparsity @ accuracy drop")
    ax.set_title("Mechanism: compressibility vs region density\n"
                 "(o = magnitude, ^ = sensitivity; size ~ Lipschitz)")
    ax.legend(fontsize=7)


def print_control_table(models):
    """Weight-scale control: max sparsity per criterion/arm, Adam vs SGD."""
    methods = _methods(models)
    arms = list(models[0]["prune_frontiers"].keys())
    print("\nmax sparsity @drop (mean over seeds):")
    print(f"{'criterion':<16}{'arm':<12}{'ADAM':>8}{'SGD':>8}")
    for arm in arms:
        for method in methods:
            def mean(opt):
                v = [m["max_sparsity"][arm][method] for m in _by_opt(models, opt)
                     if m.get("max_sparsity", {}).get(arm, {}).get(method) is not None]
                return np.mean(v) if v else float("nan")
            print(f"{method:<16}{arm:<12}{mean('adam'):>8.2f}{mean('sgd'):>8.2f}")
    print("\ncontrols (mean over seeds):")
    for opt in ("adam", "sgd"):
        ms = _by_opt(models, opt)
        f = lambda k: np.mean([m[k] for m in ms])
        print(f"  {opt.upper():<6} test_acc={f('test_acc'):.4f} "
              f"lipschitz={f('lipschitz'):.3g} local_reg={f('local_region_count'):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="result JSON files (default: latest in dir)")
    ap.add_argument("--save", default=None, help="directory to write PNGs instead of showing")
    args = ap.parse_args()

    models = load(args.paths)
    print_control_table(models)

    n_methods = len(_methods(models))
    fig = plt.figure(figsize=(max(16, 3.2 * n_methods), 10), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1])
    plot_frontier_grid(models, top)
    ab_ax, mech_ax = bottom.subplots(1, 2)
    plot_magnitude_ab(models, ab_ax)
    plot_mechanism(models, mech_ax)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        out = os.path.join(args.save, "pruning_frontiers.png")
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
