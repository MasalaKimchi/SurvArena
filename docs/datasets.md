# Datasets

Built-in benchmark dataset configs live in `configs/datasets/`.
User datasets can also be passed directly to `SurvivalPredictor.fit(...)` or
`compare_survival_models(...)` from a `DataFrame`, CSV, or Parquet file.

## Built-in Benchmarks

| Dataset | Track | Rows | Features | Event Rate | Source | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `support` | Standard | 8,873 | 14 | 68.03% | pycox | Large mixed clinical cohort. |
| `metabric` | Standard | 1,904 | 9 | 57.93% | pycox | Breast cancer benchmark. |
| `aids` | Standard | 1,151 | 11 | 8.34% | scikit-survival | Heavy censoring, mixed covariates. |
| `gbsg2` | Standard | 686 | 8 | 56.41% | scikit-survival | Compact clinical benchmark. |
| `flchain` | Standard | 7,874 | 9 | 27.55% | scikit-survival | Heavy censoring. |
| `whas500` | Standard | 500 | 14 | 43.00% | scikit-survival | Small cardiovascular cohort. |
| `kkbox` | Large | N/A | N/A | N/A | pycox/Kaggle | Large-scale local-cache dataset prepared through pycox with Kaggle credentials. |

## Notes

- counts reflect the current built-in loaders
- event rate is `events / rows`
- `kkbox` is not shipped as a ready-made local dataset; prepare the pycox KKBox cache with Kaggle credentials before use

## KKBox Download

KKBox requires the optional downloader dependencies and Kaggle credentials:

```bash
python -m pip install -e ".[kkbox]"
chmod 600 ~/.kaggle/kaggle.json
python -c "from pycox.datasets import kkbox; kkbox.download_kkbox()"
python -c "from pathlib import Path; from survarena.data.loaders import load_dataset; print(load_dataset('kkbox', Path.cwd()).X.shape)"
```

Before running the download command, create a Kaggle API token at
`~/.kaggle/kaggle.json` and accept the `kkbox-churn-prediction-challenge` terms
from the Kaggle website.

## Dataset Metadata

Each dataset config records:

- dataset id and source
- `time_col` and `event_col`
- coarse `feature_types`
- recommended split and metric settings
- notes

## Contributing

- Adding new built-in or local-only datasets: `docs/contributing_datasets.md`
