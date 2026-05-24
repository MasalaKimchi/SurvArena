# Datasets

Built-in benchmark dataset configs live in `configs/datasets/`.
User datasets can also be passed directly to `SurvivalPredictor.fit(...)` or
`compare_survival_models(...)` from a `DataFrame`, CSV, or Parquet file.

Last reviewed against `configs/datasets/` and benchmark config: 2026-05-24.

## User Dataset Pilot

Use `survarena pilot` when you want to substitute your own dataset and get a
small benchmark-style estimate before committing to a larger run:

```bash
survarena pilot \
  --data train.csv \
  --time-col time \
  --event-col event \
  --dataset-name my_dataset
```

The command writes benchmark-style CSV artifacts and prints the aggregate
leaderboard metrics in the CLI summary. Use `--id-col patient_id` or
`--drop-columns col_a,col_b` for non-feature columns, and add `--repeated` for a
3-fold x 2-repeat pilot.

## Built-in Benchmarks

| Dataset | Track | Rows | Features | Event Rate | Source | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `support` | Standard | 8,873 | 14 | 68.03% | pycox | Large mixed clinical cohort. |
| `metabric` | Standard | 1,904 | 9 | 57.93% | pycox | Breast cancer benchmark. |
| `nwtco` | Standard | 4,028 | 6 | 14.18% | pycox | National Wilms Tumor Study cohort. |
| `aids` | Standard | 1,151 | 11 | 8.34% | scikit-survival | Heavy censoring, mixed covariates. |
| `gbsg2` | Standard | 686 | 8 | 56.41% | scikit-survival | Compact clinical benchmark. |
| `flchain` | Standard | 7,874 | 9 | 27.55% | scikit-survival | Heavy censoring. |
| `whas500` | Standard | 500 | 14 | 43.00% | scikit-survival | Small cardiovascular cohort. |

The seven `Standard` datasets are the default and retained manuscript suite for
`configs/benchmark/manuscript_v1.yaml`.

## Notes

- counts reflect the current built-in loaders
- event rate is `events / rows`
## Dataset Metadata

Each dataset config records:

- dataset id and source
- `time_col` and `event_col`
- coarse `feature_types`
- recommended split and metric settings
- notes

## Contributing

- Adding new built-in or local-only datasets: `docs/contributing_datasets.md`
