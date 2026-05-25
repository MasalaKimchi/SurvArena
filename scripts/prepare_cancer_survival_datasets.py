from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


XENA_COHORTS = {
    "tcga_luad_xena": {
        "prefix": "TCGA-LUAD",
        "name": "TCGA Lung Adenocarcinoma",
    },
    "tcga_kirc_xena": {
        "prefix": "TCGA-KIRC",
        "name": "TCGA Kidney Renal Clear Cell Carcinoma",
    },
}

METABRIC_GENES = {
    "MKI67": 4288,
    "ESR1": 2099,
    "PGR": 5241,
    "ERBB2": 2064,
    "TP53": 7157,
    "GATA3": 2625,
    "BRCA1": 672,
    "BRCA2": 675,
    "FOXA1": 3169,
    "MYC": 4609,
    "CCND1": 595,
    "BCL2": 596,
    "EGFR": 1956,
    "PTEN": 5728,
    "PIK3CA": 5290,
    "AKT1": 207,
    "CDH1": 999,
    "MUC1": 4582,
    "AURKA": 6790,
    "BIRC5": 332,
    "CHEK2": 11200,
    "ATM": 472,
    "RB1": 5925,
    "MDM2": 4193,
    "VEGFA": 7422,
    "MMP9": 4318,
    "CXCL8": 3576,
    "IL6": 3569,
    "STAT3": 6774,
    "CD274": 29126,
}


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, path.open("wb") as handle:
        handle.write(response.read())


def _read_cbioportal_patient_clinical(study_id: str, cache_path: Path | None = None) -> pd.DataFrame:
    if cache_path is not None and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        url = f"https://www.cbioportal.org/api/studies/{study_id}/clinical-data?clinicalDataType=PATIENT&projection=DETAILED"
        with urllib.request.urlopen(url) as response:
            payload = json.load(response)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)
    rows = [
        {
            "patient_id": item["patientId"],
            "attribute": item["clinicalAttributeId"],
            "value": item.get("value"),
        }
        for item in payload
    ]
    clinical_long = pd.DataFrame(rows)
    return clinical_long.pivot(index="patient_id", columns="attribute", values="value").reset_index()


def _fetch_cbioportal_molecular_profile(
    *,
    molecular_profile_id: str,
    sample_list_id: str,
    entrez_gene_ids: list[int],
) -> pd.DataFrame:
    url = f"https://www.cbioportal.org/api/molecular-profiles/{molecular_profile_id}/molecular-data/fetch?projection=DETAILED"
    request = urllib.request.Request(
        url,
        data=json.dumps({"sampleListId": sample_list_id, "entrezGeneIds": entrez_gene_ids}).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        payload = json.load(response)
    rows = [
        {
            "sample_id": item["sampleId"],
            "patient_id": item["patientId"],
            item["gene"]["hugoGeneSymbol"]: item.get("value"),
        }
        for item in payload
    ]
    if not rows:
        raise ValueError(f"cBioPortal returned no molecular data for {molecular_profile_id}.")
    frame = pd.DataFrame(rows)
    value_columns = sorted(set(frame.columns) - {"sample_id", "patient_id"})
    pieces = []
    for column in value_columns:
        piece = frame.loc[frame[column].notna(), ["sample_id", "patient_id", column]]
        pieces.append(piece.set_index(["sample_id", "patient_id"]))
    return pd.concat(pieces, axis=1).reset_index()


def _normalize_event_status(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map(
        {
            "1": 1,
            "1:deceased": 1,
            "deceased": 1,
            "dead": 1,
            "0": 0,
            "0:living": 0,
            "living": 0,
            "alive": 0,
        }
    )


def _prepare_xena_cohort(dataset_id: str, raw_dir: Path, processed_dir: Path, max_genes: int) -> Path:
    cohort = XENA_COHORTS[dataset_id]
    prefix = cohort["prefix"]
    cohort_raw_dir = raw_dir / dataset_id
    survival_path = cohort_raw_dir / f"{prefix}.survival.tsv.gz"
    clinical_path = cohort_raw_dir / f"{prefix}.clinical.tsv.gz"
    expression_path = cohort_raw_dir / f"{prefix}.star_counts.tsv.gz"

    base_url = "https://gdc-hub.s3.us-east-1.amazonaws.com/download"
    _download(f"{base_url}/{prefix}.survival.tsv.gz", survival_path)
    _download(f"{base_url}/{prefix}.clinical.tsv.gz", clinical_path)
    _download(f"{base_url}/{prefix}.star_counts.tsv.gz", expression_path)

    survival = pd.read_csv(survival_path, sep="\t", compression="gzip")
    survival = survival.rename(columns={"sample": "sample_id", "OS.time": "time", "OS": "event"})
    survival = survival[["sample_id", "time", "event"]].dropna()
    survival["time"] = pd.to_numeric(survival["time"], errors="coerce")
    survival["event"] = pd.to_numeric(survival["event"], errors="coerce")
    survival = survival.dropna(subset=["time", "event"])
    survival = survival.loc[survival["time"] > 0].copy()

    expression = pd.read_csv(expression_path, sep="\t", compression="gzip")
    expression = expression.set_index("Ensembl_ID").T
    expression.index.name = "sample_id"
    expression = expression.apply(pd.to_numeric, errors="coerce")
    variances = expression.var(axis=0, skipna=True).sort_values(ascending=False)
    selected_genes = variances.head(max_genes).index.tolist()
    expression = expression[selected_genes].reset_index()

    frame = survival.merge(expression, on="sample_id", how="inner")
    output_path = processed_dir / f"{dataset_id}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} ({len(frame)} rows, {len(frame.columns) - 3} genes)")
    return output_path


def _prepare_metabric(raw_dir: Path, processed_dir: Path) -> Path:
    dataset_id = "metabric_cbioportal"
    raw_path = raw_dir / dataset_id / "patient_clinical.json"
    clinical = _read_cbioportal_patient_clinical("brca_metabric", raw_path)
    molecular = _fetch_cbioportal_molecular_profile(
        molecular_profile_id="brca_metabric_mrna",
        sample_list_id="brca_metabric_all",
        entrez_gene_ids=list(METABRIC_GENES.values()),
    )

    frame = clinical.merge(molecular, on="patient_id", how="inner")
    frame["time"] = pd.to_numeric(frame["OS_MONTHS"], errors="coerce")
    frame["event"] = _normalize_event_status(frame["OS_STATUS"])
    feature_cols = ["patient_id", "time", "event"] + list(METABRIC_GENES)
    frame = frame[feature_cols].replace({"": np.nan, "NA": np.nan, "NaN": np.nan})
    frame = frame.dropna(subset=["time", "event"])
    frame = frame.loc[frame["time"] > 0].copy()

    for gene in METABRIC_GENES:
        frame[gene] = pd.to_numeric(frame[gene], errors="coerce")

    output_path = processed_dir / f"{dataset_id}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} ({len(frame)} rows, {len(METABRIC_GENES)} genes)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare optional cancer survival datasets for SurvArena.")
    parser.add_argument(
        "--dataset",
        choices=["all", "tcga_luad_xena", "tcga_kirc_xena", "metabric_cbioportal"],
        default="all",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/cancer_survival"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/cancer_survival"))
    parser.add_argument("--max-genes", type=int, default=1000)
    args = parser.parse_args()

    selected = list(XENA_COHORTS) + ["metabric_cbioportal"] if args.dataset == "all" else [args.dataset]
    for dataset_id in selected:
        if dataset_id in XENA_COHORTS:
            _prepare_xena_cohort(dataset_id, args.raw_dir, args.processed_dir, args.max_genes)
        elif dataset_id == "metabric_cbioportal":
            _prepare_metabric(args.raw_dir, args.processed_dir)
        else:
            raise ValueError(f"Unknown dataset: {dataset_id}")


if __name__ == "__main__":
    main()
