# Datasets

SurvArena uses right-censored tabular survival datasets configured in
`configs/datasets/`.

## Dataset Overview

The table below summarizes dataset scale and censoring characteristics used for
benchmark planning and track selection.

| Dataset | Track | Source | Rows (N) | Features (P) | Events | Censored | Event Rate | Censor Rate | Description |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `support` | Standard | pycox | 8,873 | 14 | 6,036 | 2,837 | 68.03% | 31.97% | Mixed ICU/hospital clinical cohort with moderate censoring and heterogeneous feature types. |
| `metabric` | Standard | pycox | 1,904 | 9 | 1,103 | 801 | 57.93% | 42.07% | Breast cancer prognosis benchmark commonly used in survival modeling literature. |
| `gbsg2` | Standard | scikit-survival | 686 | 8 | 387 | 299 | 56.41% | 43.59% | German Breast Cancer Study Group cohort; compact clinical benchmark. |
| `flchain` | Standard | scikit-survival | 7,874 | 9 | 2,169 | 5,705 | 27.55% | 72.45% | Serum free light chain study with heavy censoring and long follow-up tails. |
| `whas500` | Standard | scikit-survival | 500 | 14 | 215 | 285 | 43.00% | 57.00% | Worcester Heart Attack Study subset; small cardiovascular risk benchmark. |
| `pbc` | Standard | lifelines | N/A | N/A | N/A | N/A | N/A | N/A | Primary biliary cirrhosis dataset; loader currently unavailable in this environment. |
| `kkbox` | Large | custom | N/A | N/A | N/A | N/A | N/A | N/A | Large-scale churn survival placeholder requiring a custom local data loader. |

## Notes

- Rates are computed as `events / N` and `censored / N`.
- Counts are measured from the current dataset loaders in `src/data/loaders.py`.
- `kkbox` is intentionally a placeholder for a user-provided large dataset.
- `pbc` metadata exists, but the configured lifelines loader is not currently
  resolvable in this environment.

## Metadata Fields

Each dataset config records:

- dataset identifiers and source
- target columns (`time_col`, `event_col`)
- feature type hints
- recommended split strategy and primary metric
- notes and citation metadata
