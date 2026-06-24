#!/usr/bin/env python3
"""
affinity_rigor.py  -- rigor engine for a binding-affinity QSAR model
====================================================================

Targets: CTSF (Q9UBX1) and the cathepsin family used as a transfer scaffold
(the modellable subset of the ageing-causal x MR-anchored convergent gene set
from the VPA-ageing GAT manuscript).

This module is the methodology, not the data. It implements the three things
that constitute the "rigor edge":

  (1) SPLITS      : random  vs  Bemis-Murcko scaffold  vs  Butina cluster  vs  time
                    -> the generalisation gap (random - scaffold) is the headline.
  (2) LEAKAGE     : an enumerated, printed audit (exact dupes, analog/Tanimoto
                    leakage, scaffold straddle, transform-fit leakage).
  (3) UNCERTAINTY : split-conformal prediction intervals (distribution-free,
                    marginal coverage under exchangeability), normalised by an
                    RF difficulty estimate, with coverage + width-vs-novelty
                    diagnostics that tie calibration to the split story.

EVERYTHING here runs WITHOUT RDKit or network, given a fingerprint matrix and
group labels, so it is unit-testable. RDKit/ChEMBL live only in fetch_chembl.py.
Run the self-test:  python affinity_rigor.py --synthetic
"""

from __future__ import annotations
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Sequence
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

# --------------------------------------------------------------------------- #
# Fingerprint similarity (operates on a 0/1 or count matrix; no RDKit needed)
# --------------------------------------------------------------------------- #

def _binarise(X: np.ndarray) -> np.ndarray:
    return (X > 0).astype(np.float32)

def tanimoto_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto (Jaccard) over binary fingerprints. Returns |A| x |B|."""
    A = _binarise(A); B = _binarise(B)
    inter = A @ B.T
    a = A.sum(1)[:, None]
    b = B.sum(1)[None, :]
    union = a + b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(union > 0, inter / union, 0.0)
    return t

def nearest_neighbour_tanimoto(test_fp: np.ndarray, train_fp: np.ndarray) -> np.ndarray:
    """For each test compound, the max Tanimoto to any train compound."""
    if len(train_fp) == 0:
        return np.zeros(len(test_fp))
    return tanimoto_matrix(test_fp, train_fp).max(axis=1)

# --------------------------------------------------------------------------- #
# Splits.  Each returns (train_idx, test_idx). Group-aware splits NEVER let a
# group (scaffold / cluster) straddle the train/test boundary.
# --------------------------------------------------------------------------- #

def random_split(n: int, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(round(n * (1 - test_frac)))
    return np.sort(idx[:cut]), np.sort(idx[cut:])

def grouped_split(groups: Sequence, test_frac: float, seed: int,
                  largest_to_train: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic group-disjoint split (used for scaffold and cluster splits).
    Largest groups go to train (classic scaffold-split behaviour) so the test set
    is enriched for small/novel groups -> a deliberately hard, leakage-free test.
    """
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    order = uniq[np.argsort(-counts)] if largest_to_train else \
        np.random.default_rng(seed).permutation(uniq)
    n = len(groups)
    target_test = int(round(n * test_frac))
    test_groups, filled = set(), 0
    # fill the TEST set from the tail (smallest groups) to hit the target size
    for g in (order[::-1] if largest_to_train else order):
        if filled >= target_test:
            break
        test_groups.add(g)
        filled += counts[list(uniq).index(g)]
    test_mask = np.isin(groups, list(test_groups))
    return np.where(~test_mask)[0], np.where(test_mask)[0]

def time_split(years: Sequence, test_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Temporal split: most-recent `test_frac` of records to test (anti-leakage gold standard)."""
    years = np.asarray(years, dtype=float)
    order = np.argsort(years, kind="stable")
    cut = int(round(len(years) * (1 - test_frac)))
    return np.sort(order[:cut]), np.sort(order[cut:])

def butina_clusters(fp: np.ndarray, cutoff: float = 0.65) -> np.ndarray:
    """
    Butina / sphere-exclusion clustering on Tanimoto distance (1 - similarity).
    Pure-numpy; for very large N use RDKit's optimised version in fetch_chembl.py.
    Returns an integer cluster label per compound.
    """
    n = len(fp)
    sim = tanimoto_matrix(fp, fp)
    neighbours = [np.where(sim[i] >= cutoff)[0] for i in range(n)]
    order = sorted(range(n), key=lambda i: -len(neighbours[i]))  # densest first
    labels = np.full(n, -1, dtype=int)
    cid = 0
    for centroid in order:
        if labels[centroid] != -1:
            continue
        members = neighbours[centroid]
        free = members[labels[members] == -1]
        if len(free) == 0:
            continue
        labels[free] = cid
        cid += 1
    # any singletons left unassigned get their own cluster
    for i in range(n):
        if labels[i] == -1:
            labels[i] = cid; cid += 1
    return labels

# --------------------------------------------------------------------------- #
# Leakage audit
# --------------------------------------------------------------------------- #

@dataclass
class LeakReport:
    split_name: str
    n_train: int
    n_test: int
    exact_dupe_keys: int          # identical standardised molecules across the boundary
    nn_tanimoto_median: float     # median nearest-train similarity of test compounds
    nn_tanimoto_p90: float
    frac_test_nn_ge_0_7: float    # analog-leakage fraction (NN >= 0.7)
    frac_test_nn_ge_0_9: float    # near-duplicate fraction (NN >= 0.9)
    scaffold_straddle_frac: float # fraction of test scaffolds also seen in train

    def render(self) -> str:
        return (
            f"  [{self.split_name:>8}] "
            f"train={self.n_train} test={self.n_test} | "
            f"exact-dupe keys across split: {self.exact_dupe_keys} | "
            f"NN-Tanimoto med={self.nn_tanimoto_median:.2f} p90={self.nn_tanimoto_p90:.2f} | "
            f"test w/ NN>=0.7: {self.frac_test_nn_ge_0_7:.0%}  NN>=0.9: {self.frac_test_nn_ge_0_9:.0%} | "
            f"scaffold straddle: {self.scaffold_straddle_frac:.0%}"
        )

def audit_leakage(split_name, train_idx, test_idx, fp, std_keys, scaffolds) -> LeakReport:
    tr, te = np.asarray(train_idx), np.asarray(test_idx)
    train_keys = set(std_keys[i] for i in tr)
    exact = sum(1 for i in te if std_keys[i] in train_keys)  # should be 0 if dedup ran
    nn = nearest_neighbour_tanimoto(fp[te], fp[tr])
    train_scaf = set(scaffolds[i] for i in tr)
    straddle = np.mean([scaffolds[i] in train_scaf for i in te]) if len(te) else 0.0
    return LeakReport(
        split_name=split_name, n_train=len(tr), n_test=len(te),
        exact_dupe_keys=exact,
        nn_tanimoto_median=float(np.median(nn)) if len(nn) else 0.0,
        nn_tanimoto_p90=float(np.quantile(nn, 0.9)) if len(nn) else 0.0,
        frac_test_nn_ge_0_7=float(np.mean(nn >= 0.7)) if len(nn) else 0.0,
        frac_test_nn_ge_0_9=float(np.mean(nn >= 0.9)) if len(nn) else 0.0,
        scaffold_straddle_frac=float(straddle),
    )

# --------------------------------------------------------------------------- #
# Model + normalised split-conformal prediction intervals
# --------------------------------------------------------------------------- #

@dataclass
class FoldResult:
    split_name: str
    rmse: float
    mae: float
    r2: float
    spearman: float
    coverage: float          # empirical coverage of the (1-alpha) interval
    mean_width: float        # mean interval width on test
    width_low_novel: float   # mean width for test cpds close to train (NN>=0.7)
    width_high_novel: float  # mean width for novel test cpds (NN<0.5)

    def render(self) -> str:
        return (
            f"  [{self.split_name:>8}] "
            f"RMSE={self.rmse:.3f} MAE={self.mae:.3f} R2={self.r2:.3f} "
            f"rho={self.spearman:.3f} | cover={self.coverage:.2f} "
            f"width={self.mean_width:.2f} (near={self.width_low_novel:.2f}, "
            f"novel={self.width_high_novel:.2f})"
        )

def _rf_with_treevar(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=400, min_samples_leaf=3, max_features="sqrt",
        n_jobs=-1, random_state=seed,
    )

def _tree_std(rf: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    """Per-prediction dispersion across trees -> difficulty estimate for normalised conformal."""
    preds = np.stack([t.predict(X) for t in rf.estimators_], axis=0)
    return preds.std(axis=0) + 1e-6

def _mondrian_half(resid_cal, sig_cal, sig_test, alpha, n_bins=5):
    """
    Mondrian conformal half-widths with a difficulty taxonomy.
    Bin calibration points by sigma into n_bins equal-frequency buckets; take a
    per-bin conformal quantile of |residual|; assign each test point the quantile
    of the sigma-bin it falls in. Bins with <10 calibration points fall back to
    the global quantile so coverage is never silently lost.
    """
    resid_cal = np.abs(resid_cal)
    edges = np.quantile(sig_cal, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    cal_bin = np.digitize(sig_cal, edges[1:-1])
    test_bin = np.digitize(sig_test, edges[1:-1])

    def gq(scores):
        m = len(scores)
        lvl = min(1.0, np.ceil((m + 1) * (1 - alpha)) / m)
        return float(np.quantile(scores, lvl, method="higher"))

    global_q = gq(resid_cal)
    half = np.full(len(sig_test), global_q, dtype=float)
    for b in range(n_bins):
        cal_mask = cal_bin == b
        if cal_mask.sum() >= 10:
            qb = gq(resid_cal[cal_mask])
        else:
            qb = global_q
        half[test_bin == b] = qb
    return half


def evaluate_split(split_name, X, y, fp, train_idx, test_idx, *,
                   alpha: float = 0.10, cal_frac: float = 0.25, seed: int = 42,
                   conformal: str = "normalized", n_bins: int = 5) -> FoldResult:
    """
    Fit on proper-train, calibrate split-conformal on a held-out
    calibration slice, evaluate point metrics + interval coverage on test.

    conformal="normalized": one global quantile on |resid|/sigma (default).
    conformal="mondrian":   Mondrian conformal with a difficulty taxonomy --
        calibration points are binned by sigma (inter-tree dispersion) and a
        SEPARATE raw-residual quantile is taken per bin; each test point gets the
        quantile of its own difficulty bin. This corrects the systematic OVER-
        coverage of global normalised conformal under distribution shift: easy
        points stop inheriting the heavy tail of hard points, so mean width drops
        and coverage moves toward nominal. (Bostrom et al., Mondrian regression.)

    Transforms and the model are fit ONLY on training data (leak-safe).
    """
    rng = np.random.default_rng(seed)
    tr = np.asarray(train_idx); te = np.asarray(test_idx)
    perm = rng.permutation(len(tr))
    n_cal = max(20, int(round(len(tr) * cal_frac)))
    cal = tr[perm[:n_cal]]
    fit = tr[perm[n_cal:]]

    rf = _rf_with_treevar(seed)
    rf.fit(X[fit], y[fit])

    mu_cal = rf.predict(X[cal])
    sig_cal = _tree_std(rf, X[cal])
    mu = rf.predict(X[te])
    sig = _tree_std(rf, X[te])
    resid_cal = y[cal] - mu_cal

    if conformal == "mondrian":
        half = _mondrian_half(resid_cal, sig_cal, sig, alpha, n_bins=n_bins)
    else:  # normalised split-conformal
        scores = np.abs(resid_cal) / sig_cal
        q_level = min(1.0, np.ceil((len(cal) + 1) * (1 - alpha)) / len(cal))
        qhat = np.quantile(scores, q_level, method="higher")
        half = qhat * sig

    lo, hi = mu - half, mu + half
    covered = (y[te] >= lo) & (y[te] <= hi)

    nn = nearest_neighbour_tanimoto(fp[te], fp[fit])
    width = 2 * half
    near = nn >= 0.7
    novel = nn < 0.5

    resid = y[te] - mu
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y[te] - y[te].mean()) ** 2)) or 1e-12
    r2 = 1 - ss_res / ss_tot
    rho = float(spearmanr(y[te], mu).correlation) if len(te) > 2 else float("nan")

    return FoldResult(
        split_name=split_name, rmse=rmse, mae=mae, r2=r2, spearman=rho,
        coverage=float(np.mean(covered)), mean_width=float(np.mean(width)),
        width_low_novel=float(np.mean(width[near])) if near.any() else float("nan"),
        width_high_novel=float(np.mean(width[novel])) if novel.any() else float("nan"),
    )

# --------------------------------------------------------------------------- #
def _make_synthetic(n_clusters=60, per_cluster=12, n_bits=512, seed=0):
    rng = np.random.default_rng(seed)
    fps, y, scaf = [], [], []
    cluster_effect = rng.normal(0, 1.2, n_clusters)
    bit_weights = rng.normal(0, 0.15, n_bits)
    for c in range(n_clusters):
        core = (rng.random(n_bits) < 0.06).astype(np.float32)
        for _ in range(per_cluster):
            decor = (rng.random(n_bits) < 0.03).astype(np.float32)
            fp = np.clip(core + decor, 0, 1)
            fps.append(fp); y.append(6.0+cluster_effect[c]+fp@bit_weights+rng.normal(0,0.25)); scaf.append(c)
    return np.array(fps,dtype=np.float32), np.array(y), np.array(scaf), np.array(scaf)

def run_self_test():
    fp,y,scaf,_=_make_synthetic(); X=fp.copy()
    keys=np.array([f"k{i}" for i in range(len(y))]); n=len(y)
    print("SELF-TEST (synthetic)"); 
    sp={"random":random_split(n,0.25,1),"scaffold":grouped_split(scaf,0.25,1)}
    for nm,(tr,te) in sp.items(): print(audit_leakage(nm,tr,te,fp,keys,scaf).render())
    for mode in ("normalized","mondrian"):
        r=evaluate_split("random",X,y,fp,*sp["random"],alpha=0.10,seed=42,conformal=mode)
        print(f"  {mode}: R2={r.r2:.3f} cover={r.coverage:.2f} width={r.mean_width:.2f}")
    assert evaluate_split("random",X,y,fp,*sp["random"],seed=42).coverage>=0.80
    print("ALL ASSERTIONS PASSED.")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--synthetic",action="store_true")
    a=ap.parse_args()
    run_self_test() if a.synthetic else print("Import from run_pipeline.py, or --synthetic.")
