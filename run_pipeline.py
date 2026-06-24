#!/usr/bin/env python3
"""
run_pipeline.py -- publication-grade rigor analysis on real ChEMBL data
Multi-seed variance, leakage-safe LOGO, CSV output for figures.
  python run_pipeline.py --data cathepsins_IC50.parquet --fp cathepsins_IC50_fp.npz --seeds 10
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import affinity_rigor as ar


def physchem_block(df):
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen
    except ImportError:
        return np.zeros((len(df), 0), dtype=np.float32)
    feats = []
    for smi in df["std_smiles"]:
        m = Chem.MolFromSmiles(smi)
        feats.append([Descriptors.MolWt(m), Crippen.MolLogP(m),
                      Descriptors.NumHAcceptors(m), Descriptors.NumHDonors(m),
                      Descriptors.TPSA(m), Descriptors.NumRotatableBonds(m)])
    return np.asarray(feats, dtype=np.float32)


def make_split(kind, *, n, scaf, clusters, years, test_frac, seed):
    if kind == "random":
        return ar.random_split(n, test_frac, seed=seed)
    if kind == "scaffold":
        return ar.grouped_split(scaf, test_frac, seed=seed, largest_to_train=False)
    if kind == "cluster":
        return ar.grouped_split(clusters, test_frac, seed=seed, largest_to_train=False)
    if kind == "time":
        return ar.time_split(years, test_frac)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fp", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.data).reset_index(drop=True)
    npz = np.load(args.fp)
    fp_bit, fp_count = npz["bits"], npz["counts"]
    phys = physchem_block(df)
    X = np.hstack([fp_count, phys]) if phys.shape[1] else fp_count
    y = df["pchembl"].to_numpy()
    keys = df["inchikey"].to_numpy()
    scaf = df["scaffold"].fillna("").to_numpy()
    years = df["year"].fillna(df["year"].median()).to_numpy()
    genes = df["gene"].to_numpy()
    n = len(df)
    endpoint = str(df["endpoint"].iloc[0])
    report_path = args.report or f"rigor_report_{endpoint}.txt"

    lines = []
    def log(s=""):
        print(s); lines.append(s)

    log("=" * 92)
    log(f"CATHEPSIN BINDING-AFFINITY RIGOR REPORT  ({endpoint})")
    log("=" * 92)
    log(f"compounds={n}  endpoint={endpoint}  scaffolds={df['scaffold'].nunique()}  "
        f"features={X.shape[1]}  seeds={args.seeds}")
    log("per-gene n:\n" + df.groupby("gene").size().to_string())
    log("")

    clusters = ar.butina_clusters(fp_bit, cutoff=0.65)

    log("LEAKAGE AUDIT  (reference split, seed=1)")
    log("-" * 92)
    audit_rows = []
    for kind in ("random", "scaffold", "cluster", "time"):
        tr, te = make_split(kind, n=n, scaf=scaf, clusters=clusters, years=years,
                            test_frac=args.test_frac, seed=1)
        rep = ar.audit_leakage(kind, tr, te, fp_bit, keys, scaf)
        log(rep.render())
        audit_rows.append(dict(endpoint=endpoint, split=kind, n_train=rep.n_train,
                               n_test=rep.n_test, exact_dupes=rep.exact_dupe_keys,
                               nn_median=rep.nn_tanimoto_median,
                               frac_nn90=rep.frac_test_nn_ge_0_9,
                               scaffold_straddle=rep.scaffold_straddle_frac))

    log(f"\nPERFORMANCE + NORMALISED SPLIT-CONFORMAL  "
        f"(alpha={args.alpha} -> coverage {1-args.alpha:.2f}; mean+/-SD over {args.seeds} seeds)")
    log("-" * 92)
    result_rows = []
    summary = {}
    for kind in ("random", "scaffold", "cluster", "time"):
        per_seed = []
        for s in range(args.seeds):
            tr, te = make_split(kind, n=n, scaf=scaf, clusters=clusters, years=years,
                                test_frac=args.test_frac, seed=s + 1)
            if len(te) < 10 or len(tr) < 40:
                continue
            r = ar.evaluate_split(kind, X, y, fp_bit, tr, te, alpha=args.alpha,
                                  seed=s + 1, conformal="normalized")
            rm = ar.evaluate_split(kind, X, y, fp_bit, tr, te, alpha=args.alpha,
                                   seed=s + 1, conformal="mondrian")
            per_seed.append((r, rm))
            result_rows.append(dict(endpoint=endpoint, split=kind, seed=s + 1,
                                    rmse=r.rmse, mae=r.mae, r2=r.r2, spearman=r.spearman,
                                    coverage=r.coverage, width=r.mean_width,
                                    width_near=r.width_low_novel, width_novel=r.width_high_novel,
                                    coverage_mondrian=rm.coverage, width_mondrian=rm.mean_width))
            if kind == "time":
                break
        if not per_seed:
            log(f"  [{kind:>8}] skipped (too few compounds)"); continue
        norm = [p[0] for p in per_seed]; mond = [p[1] for p in per_seed]
        def ms(rs, attr):
            v = np.array([getattr(r, attr) for r in rs], float)
            return np.nanmean(v), np.nanstd(v)
        r2m, r2s = ms(norm, "r2"); rmm, rms = ms(norm, "rmse")
        cvm, cvs = ms(norm, "coverage"); wmm, wms = ms(norm, "mean_width")
        cvM, _ = ms(mond, "coverage"); wmM, _ = ms(mond, "mean_width")
        summary[kind] = dict(r2=r2m, r2_sd=r2s, rmse=rmm, coverage=cvm, cov_sd=cvs,
                             width=wmm, cov_mond=cvM, width_mond=wmM)
        log(f"  [{kind:>8}] R2={r2m:.3f}+/-{r2s:.3f}  RMSE={rmm:.3f}+/-{rms:.3f}  "
            f"coverage={cvm:.2f}+/-{cvs:.2f}  width={wmm:.2f}")
        log(f"  {'':>10} conformal: normalized cov={cvm:.2f} w={wmm:.2f}  |  "
            f"mondrian cov={cvM:.2f} w={wmM:.2f}")

    log(f"\nLEAVE-CTSF-OUT (LOGO), leakage-safe")
    log("-" * 92)
    ctsf = np.where(genes == "CTSF")[0]
    rest = np.where(genes != "CTSF")[0]
    train_keys = set(keys[i] for i in rest)
    ctsf_distinct = np.array([i for i in ctsf if keys[i] not in train_keys], dtype=int)
    n_dup = len(ctsf) - len(ctsf_distinct)
    log(f"  CTSF compounds: {len(ctsf)} total; {n_dup} are exact duplicates of training "
        f"compounds (pan-cathepsin ligands).")
    log(f"  Chemically distinct CTSF compounds for an honest held-out test: {len(ctsf_distinct)}")
    if len(ctsf_distinct) >= 10:
        r = ar.evaluate_split("LOGO-CTSF", X, y, fp_bit, rest, ctsf_distinct,
                              alpha=args.alpha, seed=42)
        log(f"  LOGO-CTSF (distinct only): R2={r.r2:.3f} rho={r.spearman:.3f} "
            f"coverage={r.coverage:.2f}  (n_test={len(ctsf_distinct)})")
        log(f"  Honest transfer estimate -- no train/test compound overlap.")
        result_rows.append(dict(endpoint=endpoint, split="LOGO-CTSF-distinct", seed=42,
                                rmse=r.rmse, mae=r.mae, r2=r.r2, spearman=r.spearman,
                                coverage=r.coverage, width=r.mean_width,
                                width_near=r.width_low_novel, width_novel=r.width_high_novel))
    else:
        log(f"  -> Too few chemically distinct CTSF ligands (<10) for an honest LOGO test.")
        log(f"     HONEST FINDING: CTSF's ChEMBL inhibitors are almost entirely pan-cathepsin")
        log(f"     compounds already in the family. The model shows cathepsin POCKET-CLASS")
        log(f"     tractability; it cannot make a CTSF-specific claim. Reporting the naive")
        log(f"     LOGO R2 would be reporting leakage (all {len(ctsf)} have a train twin).")

    log("\nINTERPRETATION")
    log("-" * 92)
    if "random" in summary and "scaffold" in summary:
        gap = summary["random"]["r2"] - summary["scaffold"]["r2"]
        gap_sd = np.hypot(summary["random"]["r2_sd"], summary["scaffold"]["r2_sd"])
        log(f"  Generalisation gap (random - scaffold R2) = {gap:+.3f} +/- {gap_sd:.3f}")
        if abs(gap) <= gap_sd:
            log(f"  Gap is within its own SD -> NOT significant; random and scaffold are")
            log(f"  statistically indistinguishable for this set.")
        elif gap > 0:
            log(f"  Random over-estimates prospective skill by ~{gap:.2f} R2 beyond noise.")
        else:
            log(f"  Scaffold scored above random beyond noise (favourable scaffold draw).")
        prosp = {k: summary[k]["r2"] for k in ("scaffold", "cluster", "time") if k in summary}
        worst = min(prosp, key=prosp.get)
        log(f"  Lead with the most conservative prospective estimate: {worst} "
            f"R2={summary[worst]['r2']:.3f}+/-{summary[worst]['r2_sd']:.3f}.")
        nominal = 1 - args.alpha
        over = [k for k in ("scaffold", "cluster", "time")
                if k in summary and summary[k]["coverage"] >= nominal + 0.03]
        if over:
            # honest, empirical comparison: did Mondrian move coverage toward nominal?
            improved = [k for k in over
                        if abs(summary[k]["cov_mond"] - nominal) < abs(summary[k]["coverage"] - nominal)
                        and summary[k]["cov_mond"] >= nominal - 0.02]
            log(f"  Normalised conformal OVER-covers on {', '.join(over)} (conservative under")
            log(f"  shift). Mondrian (difficulty-binned) comparison:")
            for k in over:
                log(f"    {k:>8}: normalized cov={summary[k]['coverage']:.2f} w={summary[k]['width']:.2f}"
                    f"  ->  mondrian cov={summary[k]['cov_mond']:.2f} w={summary[k]['width_mond']:.2f}")
            if improved:
                log(f"  Mondrian moves coverage toward nominal with tighter intervals on "
                    f"{', '.join(improved)} -> prefer it there.")
            else:
                log(f"  Mondrian does NOT improve on this data; the over-coverage is honest")
                log(f"  conservatism under distribution shift, not a miscalibration to 'fix'.")
    log("\n  CAVEAT (mechanistic): INHIBITOR data; manuscript MR direction is higher CTSF ->")
    log("  longevity (activation/stabilisation). Characterises pocket tractability, NOT the")
    log("  protective direction, and is not ageing validation.")
    log("=" * 92)

    pd.DataFrame(result_rows).to_csv(f"rigor_results_{endpoint}.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(f"rigor_audit_{endpoint}.csv", index=False)
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"\n[written] {report_path}")
    log(f"[written] rigor_results_{endpoint}.csv  +  rigor_audit_{endpoint}.csv")


if __name__ == "__main__":
    main()
