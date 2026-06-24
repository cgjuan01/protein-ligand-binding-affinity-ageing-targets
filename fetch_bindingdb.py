#!/usr/bin/env python3
"""
fetch_bindingdb.py -- BindingDB data layer + cross-source harmonisation
=======================================================================

Adds BindingDB as a second affinity source and harmonises it with ChEMBL. The
harmonisation is the point: BindingDB ingests ChEMBL, so a naive union double-
counts. We standardise both sources to the SAME InChIKey and de-duplicate across
sources, then report the overlap -- the data-centric step that matters.

Get BindingDB: https://www.bindingdb.org/  ->  Download  ->  "BindingDB_All" TSV
(tab-separated). The full file is large; this reads it in chunks and keeps only
the cathepsin rows for the chosen endpoint.

Harmonisation rules (mirroring fetch_chembl, so the two are comparable):
  - filter by UniProt accession (the FAMILY dict)
  - keep only EXACT measurements (a bare number; values like ">10000" or "<0.5"
    carry an inequality relation and are dropped, matching ChEMBL relation='=')
  - convert nM to pAffinity:  pAff = 9 - log10(value_nM)  [= -log10(Molar)]
  - standardise SMILES -> InChIKey with the SAME RDKit pipeline as ChEMBL
  - aggregate replicate measurements to the median pAffinity per compound

Usage:
  python fetch_bindingdb.py --tsv BindingDB_All.tsv --endpoint Ki --out bindingdb_Ki.parquet
"""

from __future__ import annotations
import argparse, sys
import numpy as np
import pandas as pd

# reuse the cathepsin accessions + RDKit standardisation/featurisation from ChEMBL layer
try:
    from fetch_chembl import FAMILY, standardise, featurise
except ImportError:
    # fallback accessions if run standalone; standardise/featurise still required for main()
    FAMILY = {"CTSF": "Q9UBX1", "CTSD": "P07339", "CTSB": "P07858", "CTSL": "P07711",
              "CTSK": "P43235", "CTSS": "P25774", "CTSV": "O60911"}

ENDPOINT_COL = {"Ki": "Ki (nM)", "IC50": "IC50 (nM)", "Kd": "Kd (nM)"}


def _is_exact(v):
    """True iff the BindingDB value is a bare number (no >, <, ~ inequality)."""
    if v is None:
        return False
    s = str(v).strip()
    if not s or s[0] in "<>~=":
        return False
    try:
        float(s); return True
    except ValueError:
        return False


def _nm_to_paffinity(nm):
    nm = float(nm)
    if nm <= 0:
        return np.nan
    return 9.0 - np.log10(nm)   # -log10(nm * 1e-9 M)


def _col(cols, *cands):
    low = {c.lower().strip(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


def parse_bindingdb(tsv_path, endpoint, accessions, chunksize=100_000):
    """
    Stream the BindingDB TSV, keep cathepsin rows with an exact value for the
    chosen endpoint, convert to pAffinity. RDKit-free and chunked (the full file
    is multi-GB). Returns a raw dataframe in the ChEMBL-fetch schema + 'source'.
    """
    acc2gene = {a: g for g, a in accessions.items()}
    want_acc = set(acc2gene)
    val_col_name = ENDPOINT_COL[endpoint]
    frames = []
    reader = pd.read_csv(tsv_path, sep="\t", dtype=str, chunksize=chunksize,
                         on_bad_lines="skip", quoting=3, low_memory=False)
    seen_cols = None
    scanned = 0; kept = 0
    print(f"[bdb] scanning {tsv_path} for {endpoint} cathepsin rows "
          f"(chunks of {chunksize:,})...", flush=True)
    for ci, chunk in enumerate(reader):
        scanned += len(chunk)
        print(f"[bdb]   chunk {ci+1}: scanned {scanned:,} rows, kept {kept} so far",
              flush=True)
        if seen_cols is None:
            seen_cols = list(chunk.columns)
            c_smiles = _col(seen_cols, "Ligand SMILES", "SMILES")
            c_val = _col(seen_cols, val_col_name)
            # a cathepsin may be ANY chain of a complex -> scan every SwissProt Primary ID col
            acc_cols = [c for c in seen_cols
                        if c.startswith("UniProt (SwissProt) Primary ID of Target Chain")]
            c_mol = _col(seen_cols, "BindingDB MonomerID", "BindingDB Ligand Name",
                         "Ligand InChI Key")
            if not (c_smiles and c_val and acc_cols):
                sys.exit(f"[bdb] required columns missing. Need Ligand SMILES, "
                         f"'{val_col_name}', and a UniProt Primary ID column. "
                         f"Got e.g. {seen_cols[:6]}...")
        # row matches if ANY chain's primary accession is a cathepsin; take the first match
        acc_block = chunk[acc_cols]
        matched_acc = acc_block.where(acc_block.isin(want_acc)).bfill(axis=1).iloc[:, 0]
        keep = matched_acc.notna() & chunk[c_val].map(_is_exact)
        sub = chunk[keep]
        if sub.empty:
            continue
        macc = matched_acc[keep]
        frames.append(pd.DataFrame({
            "gene": macc.map(acc2gene).values,
            "target_chembl_id": ("BDB:" + macc.astype(str)).values,
            "mol_id": (sub[c_mol] if c_mol else "BDB_NA"),
            "smiles": sub[c_smiles].values,
            "pchembl": sub[c_val].map(_nm_to_paffinity).values,
            "endpoint": endpoint,
            "year": np.nan,
            "assay_id": "BDB",
            "source": "BindingDB",
        }))
        kept += int(keep.sum())
    if not frames:
        sys.exit(f"[bdb] no exact {endpoint} cathepsin rows found in {tsv_path}")
    out = pd.concat(frames, ignore_index=True).dropna(subset=["smiles", "pchembl"])
    print(f"[bdb] parsed {len(out)} exact {endpoint} cathepsin measurements")
    print("[bdb] per-gene (raw):\n" + out.groupby("gene").size().to_string())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="BindingDB_All.tsv path")
    ap.add_argument("--endpoint", default="Ki", choices=["Ki", "IC50", "Kd"])
    ap.add_argument("--out", default="bindingdb.parquet")
    ap.add_argument("--n-bits", type=int, default=2048)
    args = ap.parse_args()

    raw = parse_bindingdb(args.tsv, args.endpoint, FAMILY)
    # SAME standardisation + featurisation as ChEMBL, so InChIKeys are comparable
    df = standardise(raw)                 # parent/desalt/neutralise, InChIKey, median-aggregate
    df["source"] = "BindingDB"
    df, fp_count, fp_bit = featurise(df, n_bits=args.n_bits)
    df.to_parquet(args.out, index=False)
    np.savez_compressed(args.out.replace(".parquet", "_fp.npz"),
                        counts=fp_count, bits=fp_bit)
    print(f"[bdb] wrote {args.out} ({len(df)} unique compound-target pairs) + _fp.npz")


if __name__ == "__main__":
    main()
