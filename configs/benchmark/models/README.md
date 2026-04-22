# Per-Model Benchmark Configs

These configs isolate training so one benchmark invocation fits one method instead
of the full portfolio. Files ending in `_default.yaml` run without HPO. Files
ending in `_autogluon.yaml` use the AutoGluon-backed survival risk adapter.

Example:

```bash
python -m survarena.run_benchmark --benchmark-config configs/benchmark/models/standard_v1_autogluon_survival_default.yaml
```
