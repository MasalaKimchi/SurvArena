# Manuscript Evidence

Last reviewed against local result artifacts and benchmark docs: 2026-06-05.

This page records the retained manuscript evidence bundle and the local machine
context used to build it. It is reference material, not the first-stop setup or
benchmarking guide.

## Current No-HPO Evidence Bundle

The retained clinical manuscript no-HPO evidence bundle uses
`configs/benchmark/manuscript_v1.yaml` and covers:

- 7 built-in benchmark datasets
- 27 methods
- 5 folds x 3 repeats per dataset/method pair
- 2,835 successful fold rows in the current complete matrix
- compact CSV artifacts plus one retained Elo/report bundle under
  `results/manuscript_grade/clinical_no_hpo/elo/`

The preview below shows the Uno C Elo ladder. The retained Elo directory also
contains one ladder per comparable metric, paired win-rate tables, rank
summaries, coverage summaries, method summaries, figures, and
`metric_suite_index.csv`.

![Manuscript no-HPO Elo ratings by Uno C](assets/elo_manuscript_no_hpo_uno_c.png)

Rebuild the metric-specific Elo tables, figures, and index from local result
artifacts with:

```bash
python scripts/build_manuscript_elo.py
```

Use `--metric <metric>` for a single-metric rebuild.

## Local Reference Machine

The current evidence bundle was calibrated for CPU-default execution on the
local MacBook used for the run:

- MacBook Pro, Mac15,6
- Apple M3 Pro, 11 CPU cores (5 performance, 6 efficiency)
- 14-core integrated Apple GPU with Metal support
- 18 GB unified memory
- macOS 26.3.1
- Python 3.12.2 in `.venv`
- PyTorch 2.6.0
- `torch.backends.mps.is_available() == True`
- `torch.cuda.is_available() == False`

For manuscript-grade local Elo construction on this machine, use CPU defaults.
The deep survival adapters resolve `device: auto` to CUDA when available and CPU
otherwise; they do not auto-select Apple MPS. A direct MPS probe of
`torchsurv.loss.cox.neg_partial_log_likelihood` fails on this environment
because PyTorch MPS does not implement `aten::_logcumsumexp`, so Cox-loss neural
training remains CPU-only here.
