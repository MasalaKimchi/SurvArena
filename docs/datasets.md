# Datasets

Built-in benchmark dataset configs live in `configs/datasets/`.
User datasets can also be passed directly to `SurvivalPredictor.fit(...)` or
`compare_survival_models(...)` from a `DataFrame`, CSV, or Parquet file.

## Built-in Benchmarks

| Dataset | Track | Rows | Features | Event Rate | Source | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `support` | Standard | 8,873 | 14 | 68.03% | pycox | Large mixed clinical cohort. |
| `metabric` | Standard | 1,904 | 9 | 57.93% | pycox | Breast cancer benchmark. |
| `aids` | Standard | 1,151 | 11 | 91.66% | scikit-survival | Light censoring, mixed covariates. |
| `gbsg2` | Standard | 686 | 8 | 56.41% | scikit-survival | Compact clinical benchmark. |
| `flchain` | Standard | 7,874 | 9 | 27.55% | scikit-survival | Heavy censoring. |
| `whas500` | Standard | 500 | 14 | 43.00% | scikit-survival | Small cardiovascular cohort. |
| `kkbox` | Large | N/A | N/A | N/A | custom | Placeholder large-scale track requiring a local loader. |

## Notes

- counts reflect the current built-in loaders
- event rate is `events / rows`
- `kkbox` is config-only today and is not shipped as a ready-made local dataset

## Dataset Metadata

Each dataset config records:

- dataset id and source
- `time_col` and `event_col`
- feature hints
- recommended split strategy
- notes and citation metadata
