#!/usr/bin/env python3
"""Causal-mediation analysis for run_causal_sweep.py.

Tests whether linear-region count *causes* the downstream costs (compressibility,
verifiability), rather than merely correlating with the optimizer.  Pools
results_causal/*.json and produces:

  1. Dose-response: each knob -> region count and -> Lipschitz (knobs really move
     regions; shows how region/scale co-vary).
  2. Mediation scatter (money plot): outcome vs region count, ALL models pooled,
     colored by optimizer, sized by Lipschitz -> Adam & SGD on ONE shared curve
     means the optimizer acts only through region count.
  3. Partial regression: outcome ~ region + lipschitz (standardized); region must
     stay significant controlling for Lipschitz (independent causal signal).
  4. Matched-region-bin test: within region-count bins, is there a residual
     Adam-vs-SGD gap?  A vanishing gap => region count fully mediates.
  5. Which region metric (grid / local / pairwise) predicts best.

    uv run python analysis_causal.py --save figs
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

COLORS = {"adam": "C0", "sgd": "C1"}

# Outcomes: (label, extractor, "up/down" = does MORE compressible/verifiable mean higher?)
OUTCOMES = {
    "synflow_auc": ("SynFlow compress AUC (scale-inv)", lambda r: r["compress_auc"]["synflow"]),
    "balanced_auc": ("balanced-magnitude AUC (scale-inv)", lambda r: r["compress_auc"]["balanced"]),
    "magnitude_auc": ("magnitude AUC (scale-sensitive)", lambda r: r["compress_auc"]["magnitude"]),
    "unstable": ("mean unstable neurons (verif difficulty)",
                 lambda r: np.mean(list(r["verify"]["unstable_mean"].values()))
                 if r["verify"]["unstable_mean"] else np.nan),
    "ibp_radius": ("IBP certified radius", lambda r: r["verify"]["ibp_radius_mean"]),
}
REGION_METRICS = ["grid", "local", "pairwise"]


def load(paths):
    if not paths:
        paths = sorted(glob.glob("results_causal/*.json"))
    if not paths:
        raise SystemExit("No results_causal/*.json files found.")
    uniq = {}
    for p in paths:
        for r in json.load(open(p))["records"]:
            uniq[(r["knob"], tuple(sorted(r["setting"].items())), r["optimizer"], r["seed"])] = r
    recs = list(uniq.values())
    print(f"Loaded {len(recs)} models "
          f"({sum(r['optimizer']=='adam' for r in recs)} Adam / "
          f"{sum(r['optimizer']=='sgd' for r in recs)} SGD)")
    return recs


def _region(r, metric):
    v = r["regions"].get(metric)
    return np.nan if v is None else v  # grid is None on MNIST/CIFAR (no input_range)


def _available_metrics(recs):
    """Region metrics that have at least one non-NaN value (grid absent off low-D)."""
    return [m for m in REGION_METRICS
            if any(not np.isnan(_region(r, m)) for r in recs)]


def _standardize(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-12)


def partial_ols(y, X):
    """Least-squares standardized coefficients for y ~ [1, X columns]."""
    y = _standardize(y)
    Xs = np.column_stack([_standardize(c) for c in X] + [np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    yhat = Xs @ beta
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return beta[:-1], r2  # drop intercept


def plot_dose_response(recs, ax_reg, ax_lip, metric):
    knobs = ["weight_decay", "noise", "lr_mult", "label_noise"]
    for knob in knobs:
        sel = [r for r in recs if r["knob"] in (knob, "baseline")]
        xs = sorted({r["setting"][knob] for r in sel})
        reg = [np.nanmean([_region(r, metric) for r in sel if r["setting"][knob] == v]) for v in xs]
        lip = [np.mean([r["lipschitz"] for r in sel if r["setting"][knob] == v]) for v in xs]
        # normalize x to [0,1] rank so knobs share an axis
        rank = np.linspace(0, 1, len(xs))
        ax_reg.plot(rank, reg, marker="o", label=knob)
        ax_lip.plot(rank, lip, marker="o", label=knob)
    ax_reg.set_xlabel("knob strength (ranked)"); ax_reg.set_ylabel(f"{metric} region count")
    ax_reg.set_title("Dose-response: knob -> region count"); ax_reg.legend(fontsize=7)
    ax_lip.set_xlabel("knob strength (ranked)"); ax_lip.set_ylabel("Lipschitz product")
    ax_lip.set_title("Dose-response: knob -> weight scale"); ax_lip.legend(fontsize=7)


def plot_mediation(recs, outcome_key, region_metric, ax):
    label, extract = OUTCOMES[outcome_key]
    xs_all, ys_all = [], []
    for opt in ("adam", "sgd"):
        rs = [r for r in recs if r["optimizer"] == opt]
        x = np.array([_region(r, region_metric) for r in rs], float)
        y = np.array([extract(r) for r in rs], float)
        lips = np.array([r["lipschitz"] for r in rs], float)
        good = ~(np.isnan(x) | np.isnan(y))
        x, y, lips = x[good], y[good], lips[good]
        sizes = 30 + 150 * (lips - lips.min()) / (lips.max() - lips.min() + 1e-12)
        ax.scatter(x, y, s=sizes, alpha=0.6, color=COLORS[opt], edgecolor="k",
                   linewidth=0.4, label=opt.upper())
        xs_all += list(x); ys_all += list(y)
    # shared trend (both optimizers): if they lie on one curve, optimizer acts via regions
    xs_all, ys_all = np.array(xs_all), np.array(ys_all)
    if len(xs_all) > 3:
        order = np.argsort(xs_all)
        b1, b0 = np.polyfit(xs_all, ys_all, 1)
        xx = np.array([xs_all.min(), xs_all.max()])
        ax.plot(xx, b0 + b1 * xx, "k-", lw=2, label="shared trend")
        r = np.corrcoef(xs_all, ys_all)[0, 1]
        ax.text(0.03, 0.97, f"r={r:.2f}", transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xlabel(f"{region_metric} region count"); ax.set_ylabel(label)
    ax.set_title(f"{label}\nvs {region_metric} regions")
    ax.legend(fontsize=7)


def matched_bin_test(recs, outcome_key, region_metric, n_bins=4):
    """Within region-count bins, residual Adam-SGD gap in the outcome."""
    _, extract = OUTCOMES[outcome_key]
    x = np.array([_region(r, region_metric) for r in recs], float)
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    print(f"\nMatched-region-bin test  outcome={outcome_key}  metric={region_metric}")
    print(f"  {'region bin':<22}{'Adam':>9}{'SGD':>9}{'gap':>9}{'n':>6}")
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        inb = [r for r in recs if lo <= _region(r, region_metric) <= hi]
        a = [extract(r) for r in inb if r["optimizer"] == "adam"]
        s = [extract(r) for r in inb if r["optimizer"] == "sgd"]
        a = [v for v in a if not np.isnan(v)]; s = [v for v in s if not np.isnan(v)]
        if a and s:
            ma, ms = np.mean(a), np.mean(s)
            print(f"  [{lo:7.0f},{hi:7.0f}]  {ma:>9.3f}{ms:>9.3f}{ma - ms:>+9.3f}{len(inb):>6}")


def print_partial_regression(recs):
    print("\nPartial regression  outcome ~ region + lipschitz  (standardized beta, R^2)")
    print(f"  {'outcome':<26}{'metric':<9}{'b_region':>10}{'b_lip':>9}{'R2':>7}")
    for ok in OUTCOMES:
        _, extract = OUTCOMES[ok]
        y = np.array([extract(r) for r in recs], float)
        lip = np.array([r["lipschitz"] for r in recs], float)
        for m in REGION_METRICS:
            reg = np.array([_region(r, m) for r in recs], float)
            good = ~(np.isnan(y) | np.isnan(reg) | np.isnan(lip))
            if good.sum() < 5 or np.std(reg[good]) < 1e-9:
                continue
            (b_reg, b_lip), r2 = partial_ols(y[good], [reg[good], lip[good]])
            print(f"  {ok:<26}{m:<9}{b_reg:>10.3f}{b_lip:>9.3f}{r2:>7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--save", default=None)
    ap.add_argument("--metric", default=None, choices=REGION_METRICS,
                    help="region metric for the mediation/bin plots (default: grid if present, "
                         "else pairwise)")
    args = ap.parse_args()
    recs = load(args.paths)

    avail = _available_metrics(recs)
    if args.metric is None:
        args.metric = "grid" if "grid" in avail else ("pairwise" if "pairwise" in avail else avail[0])
    print(f"Region metrics available: {avail} | using '{args.metric}' for scatter/bins")

    print_partial_regression(recs)
    for ok in ("synflow_auc", "unstable"):
        matched_bin_test(recs, ok, args.metric)

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    top, bot = fig.subfigures(2, 1)
    a1, a2 = top.subplots(1, 2)
    plot_dose_response(recs, a1, a2, args.metric)
    axes = bot.subplots(1, 3)
    plot_mediation(recs, "synflow_auc", args.metric, axes[0])
    plot_mediation(recs, "magnitude_auc", args.metric, axes[1])
    plot_mediation(recs, "unstable", args.metric, axes[2])

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        out = os.path.join(args.save, "causal_mediation.png")
        fig.savefig(out, dpi=150)
        print(f"\nSaved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
