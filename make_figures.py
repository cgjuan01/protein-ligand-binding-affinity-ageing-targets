#!/usr/bin/env python3
"""
make_figures.py -- publication figure for the cathepsin rigor analysis
======================================================================

Reads the CSVs written by run_pipeline.py (rigor_results_*.csv, rigor_audit_*.csv)
for any endpoints present (Ki, IC50) and produces a 4-panel figure:

  A  Leakage audit: exact duplicates + near-duplicate fraction per split
     (the core result -- random splitting leaks; group splits do not).
  B  Prospective performance: R2 per split, mean +/- SD over seeds, by endpoint.
  C  Calibration: conformal coverage vs nominal per split.
  D  Applicability domain: interval width, near vs novel test compounds.

Usage:  python make_figures.py            # auto-detects endpoints in the folder
Output: rigor_figure.png (300 dpi) and rigor_figure.pdf (vector, for the paper)
"""
import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "figure.dpi": 120,
})
SPLIT_ORDER = ["random", "scaffold", "cluster", "time"]
COL = {"Ki": "#1f6f8b", "IC50": "#c8553d"}

def load(prefix):
    out = {}
    for f in sorted(glob.glob(f"{prefix}_*.csv")):
        ep = os.path.basename(f).replace(prefix + "_", "").replace(".csv", "")
        out[ep] = pd.read_csv(f)
    return out

def main():
    import argparse, datetime
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="output basename (no extension). "
                         "Default: rigor_figure_<YYYYMMDD-HHMM>")
    ap.add_argument("--name", default=None,
                    help="alias for --out; e.g. --name rigor_figure2")
    args = ap.parse_args()
    base = args.out or args.name
    if base is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        base = f"rigor_figure_{stamp}"
    base = base.replace(".png", "").replace(".pdf", "")  # tolerate an extension

    results = load("rigor_results")
    audits = load("rigor_audit")
    if not results:
        raise SystemExit("No rigor_results_*.csv found -- run run_pipeline.py first.")
    endpoints = list(results.keys())
    feature = "IC50" if "IC50" in endpoints else endpoints[-1]

    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    fig.suptitle("Cathepsin binding-affinity model: split rigor, leakage, and calibration",
                 fontsize=12, fontweight="bold", y=0.99)

    # ---- Panel A: leakage audit (use first endpoint's audit, they're similar) ----
    a = ax[0, 0]
    ad = audits[feature].set_index("split").reindex(SPLIT_ORDER)
    x = np.arange(len(SPLIT_ORDER))
    a.bar(x - 0.2, ad["frac_nn90"], 0.4, label="test cpds with near-dup in train (NN>=0.9)",
          color="#c8553d")
    a.bar(x + 0.2, ad["scaffold_straddle"], 0.4, label="scaffold straddle", color="#8aa1b1")
    for xi, ex in zip(x, ad["exact_dupes"]):
        a.text(xi - 0.2, ad["frac_nn90"].iloc[xi] + 0.02, f"{int(ex)}\ndupes",
               ha="center", va="bottom", fontsize=6.5)
    a.set_xticks(x); a.set_xticklabels(SPLIT_ORDER)
    a.yaxis.set_major_formatter(PercentFormatter(1.0))
    a.set_ylim(0, 1.15); a.set_ylabel("fraction of test set")
    a.set_title(f"A  Leakage audit ({feature})")
    a.legend(fontsize=6.5, loc="upper right", framealpha=0.9)

    # ---- Panel B: R2 per split, mean+/-SD, by endpoint ----
    b = ax[0, 1]
    width = 0.8 / max(1, len(endpoints))
    for i, ep in enumerate(endpoints):
        g = results[ep][results[ep]["split"].isin(SPLIT_ORDER)].groupby("split")["r2"]
        m = g.mean().reindex(SPLIT_ORDER); sd = g.std().reindex(SPLIT_ORDER).fillna(0)
        xb = np.arange(len(SPLIT_ORDER)) + (i - (len(endpoints)-1)/2) * width
        b.bar(xb, m, width, yerr=sd, capsize=3, label=ep, color=COL.get(ep, "#666"))
    b.set_xticks(np.arange(len(SPLIT_ORDER))); b.set_xticklabels(SPLIT_ORDER)
    b.set_ylabel("R$^2$ (held-out)"); b.set_title("B  Prospective performance (mean +/- SD)")
    b.legend(title="endpoint", fontsize=8); b.axhline(0, color="k", lw=0.5)

    # ---- Panel C: calibration -- normalized vs Mondrian on featured endpoint ----
    c = ax[1, 0]
    rf = results[feature]
    rf = rf[rf["split"].isin(SPLIT_ORDER)]
    xn = np.arange(len(SPLIT_ORDER))
    gn = rf.groupby("split")["coverage"]
    mn = gn.mean().reindex(SPLIT_ORDER); sn = gn.std().reindex(SPLIT_ORDER).fillna(0)
    c.errorbar(xn, mn, yerr=sn, marker="o", capsize=3, color="#c8553d",
               label="normalized")
    if "coverage_mondrian" in rf.columns:
        gm = rf.groupby("split")["coverage_mondrian"]
        mm = gm.mean().reindex(SPLIT_ORDER); sm = gm.std().reindex(SPLIT_ORDER).fillna(0)
        c.errorbar(xn, mm, yerr=sm, marker="s", capsize=3, ls="--", color="#1f6f8b",
                   label="Mondrian")
    c.axhline(0.90, ls=":", color="k", lw=1, label="nominal 0.90")
    c.set_xticks(xn); c.set_xticklabels(SPLIT_ORDER)
    c.set_ylim(0.85, 1.01); c.set_ylabel("conformal coverage")
    c.set_title(f"C  Calibration ({feature}): Mondrian -> nominal")
    c.legend(fontsize=8)

    # ---- Panel D: applicability domain (width near vs novel) ----
    d = ax[1, 1]
    ep = feature
    g = results[ep][results[ep]["split"].isin(SPLIT_ORDER)].groupby("split")
    near = g["width_near"].mean().reindex(SPLIT_ORDER)
    novel = g["width_novel"].mean().reindex(SPLIT_ORDER)
    xd = np.arange(len(SPLIT_ORDER))
    d.bar(xd - 0.2, near, 0.4, label="near train (NN>=0.7)", color="#4c9f70")
    d.bar(xd + 0.2, novel, 0.4, label="novel (NN<0.5)", color="#d4a373")
    d.set_xticks(xd); d.set_xticklabels(SPLIT_ORDER)
    d.set_ylabel("mean interval width (log units)")
    d.set_title(f"D  Applicability domain ({ep}): width tracks novelty")
    d.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    print(f"[figure] wrote {base}.png (300 dpi) and {base}.pdf")

if __name__ == "__main__":
    main()
