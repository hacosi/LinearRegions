#!/usr/bin/env python3
"""Figures for the exact-region-tracking sweep (run_region_sweep.py).

Loads results_regions/*.json and produces:

  1. Region-vs-epoch small-multiples grid (rows=depth, cols=width), Adam vs SGD,
     mean +/- SEM over seeds.
  2. Final region count vs width (one line per depth), Adam vs SGD.
  3. Final region count vs depth (one line per width), Adam vs SGD.
  4. SGD - Adam final-region gap heatmap over depth x width.
  5. Final test accuracy Adam vs SGD across the grid (the gap is not an accuracy
     artifact if these overlap).

    uv python analysis_region_sweep.py --save figs
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
        paths = sorted(glob.glob("results_regions/*.json"))
    if not paths:
        raise SystemExit("No results_regions/*.json files found.")
    # dedup by (depth,width,seed,optimizer), last wins (resume across files)
    uniq = {}
    for p in paths:
        for c in json.load(open(p))["cells"]:
            uniq[(c["depth"], c["width"], c["seed"], c["optimizer"])] = c
    cells = list(uniq.values())
    depths = sorted({c["depth"] for c in cells})
    widths = sorted({c["width"] for c in cells})
    print(f"Loaded {len(cells)} cells | depths={depths} widths={widths} "
          f"seeds={sorted({c['seed'] for c in cells})}")
    return cells, depths, widths


def _sel(cells, opt, depth=None, width=None):
    return [c for c in cells if c["optimizer"] == opt
            and (depth is None or c["depth"] == depth)
            and (width is None or c["width"] == width)]


def _mean_sem(rows):
    a = np.array(rows, dtype=float)
    m = a.mean(0)
    s = a.std(0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.zeros_like(m)
    return m, s


def plot_traj_grid(cells, depths, widths, save):
    fig, axes = plt.subplots(len(depths), len(widths), figsize=(3.2 * len(widths),
                             2.6 * len(depths)), squeeze=False, sharex=True)
    for i, d in enumerate(depths):
        for j, w in enumerate(widths):
            ax = axes[i][j]
            for opt in ("adam", "sgd"):
                sel = _sel(cells, opt, d, w)
                if not sel:
                    continue
                ep = sel[0]["region_epochs"]
                m, s = _mean_sem([c["regions"] for c in sel])
                ax.plot(ep, m, color=COLORS[opt], label=opt.upper())
                ax.fill_between(ep, m - s, m + s, color=COLORS[opt], alpha=0.2)
            if i == 0:
                ax.set_title(f"width {w}")
            if j == 0:
                ax.set_ylabel(f"depth {d}\nregions")
            if i == len(depths) - 1:
                ax.set_xlabel("epoch")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Exact linear regions vs training (mean +/- SEM over seeds)")
    fig.tight_layout()
    _save_or_show(fig, save, "regions_trajectory_grid.png")


def plot_final_vs_arch(cells, depths, widths, save):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # vs width, line per depth
    for d in depths:
        for opt in ("adam", "sgd"):
            ys, es = [], []
            for w in widths:
                sel = _sel(cells, opt, d, w)
                m, s = _mean_sem([[c["final_regions"]] for c in sel]) if sel else (np.array([np.nan]), np.array([0]))
                ys.append(m[0]); es.append(s[0])
            ls = "-" if opt == "adam" else "--"
            ax1.errorbar(widths, ys, yerr=es, marker="o", ls=ls, color=f"C{depths.index(d)}",
                         capsize=3, label=f"d{d} {opt.upper()}")
    ax1.set_xlabel("width"); ax1.set_ylabel("final regions"); ax1.set_xscale("log", base=2)
    ax1.set_title("Final regions vs width (solid=Adam, dashed=SGD)")
    ax1.legend(fontsize=6, ncol=2)
    # vs depth, line per width
    for w in widths:
        for opt in ("adam", "sgd"):
            ys, es = [], []
            for d in depths:
                sel = _sel(cells, opt, d, w)
                m, s = _mean_sem([[c["final_regions"]] for c in sel]) if sel else (np.array([np.nan]), np.array([0]))
                ys.append(m[0]); es.append(s[0])
            ls = "-" if opt == "adam" else "--"
            ax2.errorbar(depths, ys, yerr=es, marker="o", ls=ls, color=f"C{widths.index(w)}",
                         capsize=3, label=f"w{w} {opt.upper()}")
    ax2.set_xlabel("depth"); ax2.set_ylabel("final regions")
    ax2.set_title("Final regions vs depth (solid=Adam, dashed=SGD)")
    ax2.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    _save_or_show(fig, save, "regions_final_vs_arch.png")


def plot_gap_and_acc(cells, depths, widths, save):
    gap = np.full((len(depths), len(widths)), np.nan)
    acc_a = np.full((len(depths), len(widths)), np.nan)
    acc_s = np.full((len(depths), len(widths)), np.nan)
    for i, d in enumerate(depths):
        for j, w in enumerate(widths):
            a, s = _sel(cells, "adam", d, w), _sel(cells, "sgd", d, w)
            if a and s:
                gap[i, j] = np.mean([c["final_regions"] for c in s]) - \
                            np.mean([c["final_regions"] for c in a])
                acc_a[i, j] = np.mean([c["final_test_metric"] for c in a])
                acc_s[i, j] = np.mean([c["final_test_metric"] for c in s])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    im = ax1.imshow(gap, cmap="RdBu", aspect="auto",
                    vmin=-np.nanmax(np.abs(gap)), vmax=np.nanmax(np.abs(gap)))
    ax1.set_xticks(range(len(widths))); ax1.set_xticklabels(widths)
    ax1.set_yticks(range(len(depths))); ax1.set_yticklabels(depths)
    ax1.set_xlabel("width"); ax1.set_ylabel("depth")
    ax1.set_title("SGD - Adam final regions (blue = Adam fewer)")
    for i in range(len(depths)):
        for j in range(len(widths)):
            if not np.isnan(gap[i, j]):
                ax1.text(j, i, f"{gap[i, j]:+.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax1)

    # accuracy parity scatter
    ax2.scatter(acc_a.flatten(), acc_s.flatten(), c="C2")
    lim = [np.nanmin([acc_a, acc_s]), np.nanmax([acc_a, acc_s])]
    ax2.plot(lim, lim, "k--", alpha=0.5)
    ax2.set_xlabel("Adam final test metric"); ax2.set_ylabel("SGD final test metric")
    ax2.set_title("Accuracy parity (points on y=x -> gap is not an accuracy artifact)")
    fig.tight_layout()
    _save_or_show(fig, save, "regions_gap_and_accuracy.png")


def _save_or_show(fig, save, name):
    if save:
        os.makedirs(save, exist_ok=True)
        out = os.path.join(save, name)
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    cells, depths, widths = load(args.paths)
    plot_traj_grid(cells, depths, widths, args.save)
    plot_final_vs_arch(cells, depths, widths, args.save)
    plot_gap_and_acc(cells, depths, widths, args.save)


if __name__ == "__main__":
    main()
