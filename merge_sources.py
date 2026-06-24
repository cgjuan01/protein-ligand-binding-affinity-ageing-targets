#!/usr/bin/env python3
"""
merge_sources.py -- harmonise ChEMBL + BindingDB into one de-duplicated dataset
===============================================================================

Takes the per-source parquet+npz outputs (ChEMBL, BindingDB) and produces a
single harmonised dataset for run_pipeline.py. The non-trivial step: BindingDB
re-distributes ChEMBL, so the same molecule-target pair appears in both sources.
We de-duplicate ACROSS sources by standardised InChIKey (the same key both
fetchers compute), aggregate to the median pAffinity, track provenance, and
report the overlap. Fingerprints are identical for identical InChIKeys, so we
carry them over without re-running RDKit.

Usage:
  python merge_sources.py --inputs cathepsins_Ki.parquet bindingdb_Ki.parquet \
                          --out merged_Ki.parquet
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def load_pair(parquet_path):
    df = pd.read_parquet(parquet_path).reset_index(drop=True)
    npz = np.load(parquet_path.replace(".parquet", "_fp.npz"))
    if "source" not in df.columns:
        df["source"] = "ChEMBL"
    return df, npz["bits"], npz["counts"]


def merge(inputs):
    frames, key2bit, key2cnt = [], {}, {}
    for p in inputs:
        df, bits, cnts = load_pair(p)
        frames.append(df)
        for i, k in enumerate(df["inchikey"].to_numpy()):
            if k not in key2bit:                      # identical key -> identical fp
                key2bit[k] = bits[i]; key2cnt[k] = cnts[i]
    allrows = pd.concat(frames, ignore_index=True)

    # provenance BEFORE aggregation: which sources hold each (inchikey, gene, endpoint)
    prov = (allrows.groupby(["inchikey", "gene", "endpoint"])["source"]
            .agg(lambda s: ",".join(sorted(set(s)))).rename("sources"))

    agg = (allrows.groupby(["inchikey", "gene", "endpoint"], as_index=False)
           .agg(std_smiles=("std_smiles", "first"),
                pchembl=("pchembl", "median"),
                scaffold=("scaffold", "first"),
                year=("year", "min"),
                target_chembl_id=("target_chembl_id", "first")))
    agg = agg.merge(prov, on=["inchikey", "gene", "endpoint"])
    agg["source"] = agg["sources"]

    # rebuild fingerprint matrices in the merged row order
    bits = np.vstack([key2bit[k] for k in agg["inchikey"]])
    cnts = np.vstack([key2cnt[k] for k in agg["inchikey"]])

    # overlap report
    n = len(agg)
    both = int((agg["sources"].str.contains(",")).sum())
    only = {s: int((agg["sources"] == s).sum())
            for s in sorted(set(",".join(agg["sources"]).split(",")))}
    print("=" * 70)
    print("CROSS-SOURCE HARMONISATION")
    print("=" * 70)
    print(f"  merged unique compound-target pairs: {n}")
    for s, c in only.items():
        print(f"    only in {s}: {c}")
    print(f"    in >1 source (de-duplicated): {both}")
    raw_total = sum(len(pd.read_parquet(p)) for p in inputs)
    print(f"  naive union would have been {raw_total} rows; harmonisation removed "
          f"{raw_total - n} duplicates ({(raw_total-n)/raw_total:.0%}).")
    print("=" * 70)
    return agg, bits, cnts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="per-source parquet files (ChEMBL, BindingDB)")
    ap.add_argument("--out", default="merged.parquet")
    args = ap.parse_args()
    agg, bits, cnts = merge(args.inputs)
    agg.to_parquet(args.out, index=False)
    np.savez_compressed(args.out.replace(".parquet", "_fp.npz"), bits=bits, counts=cnts)
    print(f"[merge] wrote {args.out} ({len(agg)} rows) + _fp.npz")
    print(f"[merge] feed to: python run_pipeline.py --data {args.out} "
          f"--fp {args.out.replace('.parquet','_fp.npz')} --seeds 10")


if __name__ == "__main__":
    main()
