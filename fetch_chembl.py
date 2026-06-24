#!/usr/bin/env python3
"""
fetch_chembl.py -- data layer (REQUIRES network + RDKit; run on your machine)
=============================================================================

Pulls binding bioactivities for CTSF and the cathepsin family from ChEMBL,
standardises molecules, deduplicates, aggregates replicate measurements, and
writes a clean table that run_pipeline.py consumes.

Why the cathepsin FAMILY and not CTSF alone:
  CTSF (Q9UBX1) has too few ligands to train/validate a model on its own. The
  family (CTSB/CTSL/CTSK/CTSS/CTSV) is data-rich and shares the papain-fold
  cysteine-protease active site, so it is a defensible transfer scaffold. The
  rigorous test is then: hold CTSF out entirely (LOGO -- leave-one-gene-out)
  and see whether a family model generalises to it. fetch keeps the gene label
  so run_pipeline can do exactly that.

Endpoint hygiene (prevents label leakage/noise):
  - keep standard_type in {IC50, Ki, Kd}; keep standard_relation '='
  - require a curated pChEMBL value (already -log10 M, assay-normalised)
  - model ONE endpoint at a time (default Ki) OR carry endpoint as a covariate;
    do NOT silently pool IC50+Ki+Kd into one regression target.

Standardisation (prevents 'same molecule on both sides' leakage):
  parent -> desalt -> neutralise -> canonical tautomer(optional) -> InChIKey.
  Records collapsing to the same InChIKey+target+endpoint are aggregated to the
  MEDIAN pChEMBL BEFORE any split, so a compound can never straddle train/test.

Usage:
  pip install rdkit chembl_webresource_client pandas pyarrow
  python fetch_chembl.py --endpoint Ki --out cathepsins_Ki.parquet
"""

from __future__ import annotations
import argparse, sys, time
import numpy as np
import pandas as pd

# Cathepsin family by UniProt accession (resolve targets robustly, not by ChEMBL ID).
FAMILY = {
    "CTSF": "Q9UBX1",   # cathepsin F  -- the manuscript target (held out for LOGO)
    "CTSD": "P07339",   # cathepsin D  -- aspartyl; included, flagged (different fold)
    "CTSB": "P07858",   # cathepsin B
    "CTSL": "P07711",   # cathepsin L
    "CTSK": "P43235",   # cathepsin K  -- most ligand-rich
    "CTSS": "P25774",   # cathepsin S
    "CTSV": "O60911",   # cathepsin V (CTSL2)
}

def _require(mod):
    try:
        return __import__(mod)
    except ImportError:
        sys.exit(f"[fetch] missing '{mod}'. Run: pip install rdkit chembl_webresource_client pandas pyarrow")

def get_new_client(max_tries: int = 8, base_delay: float = 4.0):
    """
    Load chembl_webresource_client.new_client with retry + exponential backoff.

    The client fetches its schema from /spore AT IMPORT TIME, so an EBI 500 (which
    commonly lags behind the website coming back up) raises during import, not
    during our query. We therefore retry the IMPORT, clearing the half-initialised
    modules from sys.modules between attempts so the next try starts clean.
    """
    import importlib
    _require("chembl_webresource_client")
    last = None
    for attempt in range(1, max_tries + 1):
        try:
            for name in list(sys.modules):
                if name.startswith("chembl_webresource_client"):
                    del sys.modules[name]
            mod = importlib.import_module("chembl_webresource_client.new_client")
            print(f"[fetch] ChEMBL API reachable (attempt {attempt}).")
            return mod.new_client
        except Exception as e:  # noqa: BLE001 -- EBI raises bare Exception on 500
            last = e
            msg = str(e).split("\n")[0][:120]
            delay = base_delay * (2 ** (attempt - 1))
            print(f"[fetch] API not ready (attempt {attempt}/{max_tries}): {msg}")
            if attempt < max_tries:
                print(f"[fetch]   waiting {delay:.0f}s before retry...")
                time.sleep(delay)
    sys.exit(f"[fetch] ChEMBL API still failing after {max_tries} tries.\n"
             f"        Last error: {str(last).splitlines()[0]}\n"
             f"        Either retry later, or use the API-free path: --from-csv DIR\n"
             f"        (download per-gene activity CSVs from https://www.ebi.ac.uk/chembl/).")

def _retry_call(fn, what: str, max_tries: int = 6, base_delay: float = 3.0):
    """
    Run fn() with retry+backoff. The ChEMBL client is lazy: the HTTP request
    fires when the queryset is materialised (list/iterate), so a 500 raises HERE,
    not at .filter(). Wrapping materialisation is what makes the pull survivable.
    """
    last = None
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            delay = base_delay * (2 ** (attempt - 1))
            print(f"[fetch]   {what}: server error (attempt {attempt}/{max_tries}), "
                  f"waiting {delay:.0f}s...")
            if attempt < max_tries:
                time.sleep(delay)
    raise RuntimeError(f"{what}: failed after {max_tries} retries: "
                       f"{str(last).splitlines()[0]}")

def resolve_targets(accessions: dict[str, str]) -> dict[str, list[str]]:
    """gene -> [ChEMBL target ids] via UniProt accession of a target component."""
    new_client = get_new_client()
    tclient = new_client.target
    out = {}
    for gene, acc in accessions.items():
        def pull():
            q = tclient.filter(target_components__accession=acc,
                               target_type="SINGLE PROTEIN").only(["target_chembl_id"])
            return sorted({h["target_chembl_id"] for h in q})  # materialises -> HTTP
        ids = _retry_call(pull, f"resolve {gene}")
        out[gene] = ids
        print(f"[fetch] {gene} ({acc}) -> {ids}")
        time.sleep(0.2)
    return out

def fetch_activities(target_map: dict[str, list[str]], endpoint: str) -> pd.DataFrame:
    new_client = get_new_client()
    act = new_client.activity
    rows = []
    for gene, tids in target_map.items():
        for tid in tids:
            def pull():
                q = act.filter(target_chembl_id=tid,
                               standard_type=endpoint,
                               standard_relation="=",
                               pchembl_value__isnull=False).only([
                    "molecule_chembl_id", "canonical_smiles", "pchembl_value",
                    "standard_type", "document_year", "assay_chembl_id"])
                return list(q)  # materialises -> paginated HTTP calls happen here
            records = _retry_call(pull, f"activities {gene}/{tid}")
            n = 0
            for r in records:
                if not r.get("canonical_smiles"):
                    continue
                rows.append({
                    "gene": gene, "target_chembl_id": tid,
                    "mol_id": r["molecule_chembl_id"],
                    "smiles": r["canonical_smiles"],
                    "pchembl": float(r["pchembl_value"]),
                    "endpoint": r["standard_type"],
                    "year": r.get("document_year"),
                    "assay_id": r.get("assay_chembl_id"),
                }); n += 1
            print(f"[fetch] {gene}/{tid}: {n} {endpoint} records")
            time.sleep(0.2)
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit(f"[fetch] no {endpoint} records found -- try --endpoint IC50")
    return df

def from_sqlite(db_path: str, endpoint: str, assay_type: str = "B") -> pd.DataFrame:
    """
    API-FREE, ONE-DOWNLOAD path. Query the ChEMBL SQLite dump directly.

    Get the dump (any recent release) from:
      https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/
      -> chembl_XX_sqlite.tar.gz  (expand to chembl_XX.db)

    Filters by UniProt accession (your FAMILY dict), so no target-ID resolution
    and no network. Joins, per the standard ChEMBL schema:
      component_sequences.accession  (UniProt)
        -> target_components -> target_dictionary (tid, chembl_id)
        -> assays (tid, assay_type)
        -> activities (standard_type/relation/value, pchembl_value, molregno, doc_id)
        -> compound_structures (canonical_smiles)
        -> docs (year)
    assay_type 'B' = binding assays (the right class for Ki/Kd/IC50 affinity).
    """
    import sqlite3
    con = sqlite3.connect(db_path)
    sql = """
        SELECT td.chembl_id            AS target_chembl_id,
               md.chembl_id            AS mol_id,
               cs.canonical_smiles     AS smiles,
               act.pchembl_value       AS pchembl,
               act.standard_type       AS endpoint,
               d.year                  AS year,
               a.chembl_id             AS assay_id
        FROM component_sequences  csq
        JOIN target_components    tc  ON tc.component_id = csq.component_id
        JOIN target_dictionary    td  ON td.tid         = tc.tid
        JOIN assays               a   ON a.tid          = td.tid
        JOIN activities           act ON act.assay_id   = a.assay_id
        JOIN compound_structures  cs  ON cs.molregno    = act.molregno
        JOIN molecule_dictionary  md  ON md.molregno    = act.molregno
        LEFT JOIN docs            d   ON d.doc_id       = act.doc_id
        WHERE csq.accession   = ?
          AND act.standard_type = ?
          AND act.standard_relation = '='
          AND act.pchembl_value IS NOT NULL
          AND a.assay_type    = ?
          AND cs.canonical_smiles IS NOT NULL
    """
    frames = []
    for gene, acc in FAMILY.items():
        d = pd.read_sql_query(sql, con, params=(acc, endpoint, assay_type))
        if d.empty:
            print(f"[fetch] {gene} ({acc}): 0 {endpoint} binding rows"); continue
        d.insert(0, "gene", gene)
        d["pchembl"] = pd.to_numeric(d["pchembl"], errors="coerce")
        frames.append(d)
        print(f"[fetch] {gene} ({acc}): {len(d)} {endpoint} binding rows")
    con.close()
    if not frames:
        sys.exit(f"[fetch] no {endpoint} binding rows for any cathepsin -- try --endpoint IC50")
    out = pd.concat(frames, ignore_index=True).dropna(subset=["smiles", "pchembl"])
    print(f"[fetch] total: {len(out)} rows across {out['gene'].nunique()} genes")
    return out

def from_csv(csv_dir: str, endpoint: str) -> pd.DataFrame:
    """
    API-FREE path. Read per-gene activity CSVs exported from the ChEMBL web
    interface (https://www.ebi.ac.uk/chembl/), one file per gene named <GENE>.csv
    (e.g. CTSF.csv, CTSK.csv). The gene label is taken from the file name, so no
    target resolution and no /spore call is needed -- fully reproducible.

    How to export (per target): search the gene/UniProt on the ChEMBL site, open
    the target, go to Activities, filter Standard Type to your endpoint, and use
    the download (CSV) button. The interface emits ';'-separated CSVs with columns
    like 'Smiles', 'Standard Type', 'Standard Relation', 'pChEMBL Value',
    'Document Year', 'Assay ChEMBL ID', 'Molecule ChEMBL ID'.
    """
    import glob, os
    # tolerant column lookup (interface header names vary slightly by release)
    def col(df, *cands):
        low = {c.lower().strip(): c for c in df.columns}
        for c in cands:
            if c.lower() in low:
                return low[c.lower()]
        return None
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not files:
        sys.exit(f"[fetch] no *.csv files in {csv_dir} (name them <GENE>.csv, e.g. CTSF.csv)")
    frames = []
    for path in files:
        gene = os.path.splitext(os.path.basename(path))[0].upper()
        # ChEMBL exports are ';'-separated; fall back to auto-detect
        try:
            d = pd.read_csv(path, sep=";", dtype=str)
            if d.shape[1] == 1:
                d = pd.read_csv(path, sep=None, engine="python", dtype=str)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"[fetch] could not read {path}: {e}")
        c_smiles = col(d, "Smiles", "Canonical Smiles", "canonical_smiles")
        c_type   = col(d, "Standard Type", "standard_type")
        c_rel    = col(d, "Standard Relation", "standard_relation")
        c_pchembl= col(d, "pChEMBL Value", "pchembl_value")
        c_year   = col(d, "Document Year", "document_year")
        c_assay  = col(d, "Assay ChEMBL ID", "assay_chembl_id")
        c_mol    = col(d, "Molecule ChEMBL ID", "molecule_chembl_id")
        c_target = col(d, "Target ChEMBL ID", "target_chembl_id")
        if not (c_smiles and c_type and c_pchembl):
            sys.exit(f"[fetch] {path}: missing Smiles/Standard Type/pChEMBL columns; "
                     f"got {list(d.columns)[:8]}...")
        d = d[d[c_type].str.strip().str.upper() == endpoint.upper()]
        if c_rel is not None:
            rel = d[c_rel].astype(str).str.strip().str.strip("'\"")
            d = d[rel.isin(["=", ""])]               # keep '=' (blank in some exports)
        d = d[d[c_pchembl].notna() & (d[c_pchembl].astype(str).str.strip() != "")]
        if d.empty:
            print(f"[fetch] {gene}: 0 usable {endpoint} rows after filtering"); continue
        frames.append(pd.DataFrame({
            "gene": gene,
            "target_chembl_id": d[c_target] if c_target else "NA",
            "mol_id": d[c_mol] if c_mol else "NA",
            "smiles": d[c_smiles],
            "pchembl": pd.to_numeric(d[c_pchembl], errors="coerce"),
            "endpoint": endpoint,
            "year": pd.to_numeric(d[c_year], errors="coerce") if c_year else np.nan,
            "assay_id": d[c_assay] if c_assay else "NA",
        }))
        print(f"[fetch] {gene}: {len(frames[-1])} {endpoint} rows from {os.path.basename(path)}")
    out = pd.concat(frames, ignore_index=True).dropna(subset=["smiles", "pchembl"])
    if out.empty:
        sys.exit(f"[fetch] no usable {endpoint} rows across CSVs")
    return out

# ---- RDKit standardisation ------------------------------------------------- #

def _standardiser():
    _require("rdkit")
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    lfc = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()
    te = rdMolStandardize.TautomerEnumerator()
    def std(smiles, canonical_tautomer=False):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None, None
        m = rdMolStandardize.Cleanup(m)
        m = lfc.choose(m)                 # desalt / largest fragment (parent)
        m = uncharger.uncharge(m)         # neutralise
        if canonical_tautomer:
            m = te.Canonicalize(m)        # optional, slow
        try:
            ik = Chem.MolToInchiKey(m)
        except Exception:
            return None, None
        return Chem.MolToSmiles(m), ik
    return std

def standardise(df: pd.DataFrame, canonical_tautomer=False) -> pd.DataFrame:
    std = _standardiser()
    keys, smis = [], []
    for smi in df["smiles"]:
        s, ik = std(smi, canonical_tautomer)
        smis.append(s); keys.append(ik)
    df = df.assign(std_smiles=smis, inchikey=keys).dropna(subset=["inchikey"])
    before = len(df)
    # aggregate replicate measurements -> median pChEMBL per (compound,gene,endpoint)
    agg = (df.groupby(["inchikey", "gene", "endpoint"], as_index=False)
             .agg(std_smiles=("std_smiles", "first"),
                  pchembl=("pchembl", "median"),
                  n_meas=("pchembl", "size"),
                  year=("year", "min"),               # earliest doc year for time-split
                  target_chembl_id=("target_chembl_id", "first")))
    print(f"[fetch] standardised: {before} records -> {len(agg)} unique compound-target pairs")
    # within a gene, a molecule now appears once -> cannot straddle a split
    dup = agg.duplicated(["inchikey", "gene"]).sum()
    assert dup == 0, f"{dup} residual duplicate compound-gene rows after aggregation"
    return agg

# ---- Morgan fingerprints + Murcko scaffolds -------------------------------- #

def featurise(df: pd.DataFrame, radius=2, n_bits=2048) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Returns (df+scaffold, count-fingerprint matrix, bit-fingerprint matrix)."""
    _require("rdkit")
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    counts, bits, scaffolds = [], [], []
    for smi in df["std_smiles"]:
        m = Chem.MolFromSmiles(smi)
        counts.append(np.array(gen.GetCountFingerprint(m).ToList(), dtype=np.float32))
        bits.append(np.array(gen.GetFingerprint(m), dtype=np.float32))
        try:
            scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        except Exception:
            scaffolds.append("")
    df = df.assign(scaffold=scaffolds)
    return df, np.vstack(counts), np.vstack(bits)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="Ki", choices=["Ki", "IC50", "Kd"])
    ap.add_argument("--canonical-tautomer", action="store_true")
    ap.add_argument("--out", default="cathepsins.parquet")
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--from-csv", default=None,
                    help="API-free: directory of per-gene <GENE>.csv exports "
                         "from the ChEMBL web interface (skips /spore entirely)")
    ap.add_argument("--from-sqlite", default=None,
                    help="API-free: path to ChEMBL SQLite dump (chembl_XX.db); "
                         "pulls all cathepsins by UniProt in one command")
    args = ap.parse_args()

    if args.from_sqlite:
        df = from_sqlite(args.from_sqlite, args.endpoint)
    elif args.from_csv:
        df = from_csv(args.from_csv, args.endpoint)
    else:
        target_map = resolve_targets(FAMILY)
        df = fetch_activities(target_map, args.endpoint)
    df = standardise(df, canonical_tautomer=args.canonical_tautomer)
    df, fp_count, fp_bit = featurise(df, n_bits=args.n_bits)

    df.to_parquet(args.out, index=False)
    np.savez_compressed(args.out.replace(".parquet", "_fp.npz"),
                        counts=fp_count, bits=fp_bit)
    print(f"[fetch] wrote {args.out}  ({len(df)} rows) and fingerprints .npz")
    print("[fetch] per-gene counts:\n", df.groupby("gene").size().to_string())

if __name__ == "__main__":
    main()
