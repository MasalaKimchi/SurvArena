from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd


XENA_COHORTS = {
    "tcga_brca_xena": {
        "prefix": "TCGA-BRCA",
        "name": "TCGA Breast Invasive Carcinoma",
    },
    "tcga_luad_xena": {
        "prefix": "TCGA-LUAD",
        "name": "TCGA Lung Adenocarcinoma",
    },
    "tcga_kirc_xena": {
        "prefix": "TCGA-KIRC",
        "name": "TCGA Kidney Renal Clear Cell Carcinoma",
    },
    "tcga_skcm_xena": {
        "prefix": "TCGA-SKCM",
        "name": "TCGA Skin Cutaneous Melanoma",
    },
    "tcga_ov_xena": {
        "prefix": "TCGA-OV",
        "name": "TCGA Ovarian Serous Cystadenocarcinoma",
    },
}

def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, path.open("wb") as handle:
        handle.write(response.read())


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare optional cancer survival datasets for SurvArena.")
    parser.add_argument(
        "--dataset",
        choices=["all", *XENA_COHORTS],
        default="all",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/cancer_survival"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/cancer_survival"))
    parser.add_argument("--max-genes", type=int, default=1000)
    args = parser.parse_args()

    selected = list(XENA_COHORTS) if args.dataset == "all" else [args.dataset]
    for dataset_id in selected:
        if dataset_id in XENA_COHORTS:
            _prepare_xena_cohort(dataset_id, args.raw_dir, args.processed_dir, args.max_genes)
        else:
            raise ValueError(f"Unknown dataset: {dataset_id}")


if __name__ == "__main__":
    main()
