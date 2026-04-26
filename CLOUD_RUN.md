# Cloud Runs Through Codex

This guide is for running SurvArena benchmark jobs on a cloud or remote worker
through Codex, not on your local device. Use it when the runtime, memory, or
dependency footprint is too large for a laptop.

## What Codex Should Do Remotely

Ask Codex to create or use a cloud worker with this repository checked out, then
run the benchmark command on that worker. The local machine should only be used
to prepare the branch, review logs, and collect artifacts.

Recommended Codex prompt:

```text
Run this SurvArena benchmark on a cloud worker, not on my local device.

Repository: SurvArena
Branch or commit: <branch-or-commit>
Benchmark config: configs/benchmark/standard_v1.yaml
Scope: <full config, or dataset/method shard>
Python: 3.11
Setup command: PYTHON_BIN=python3.11 ./scripts/setup_env.sh
Validation command: python scripts/check_environment.py
Run command: python -m survarena.run_benchmark --benchmark-config <config> <optional shard flags>
Artifacts to return: results/summary/<experiment_dir> plus terminal logs

Use --dry-run first, then start the real run only if the plan resolves
correctly. Keep stdout unbuffered and preserve the output directory so the run
can be resumed.
```

Replace the placeholders with the exact branch, commit, dataset, method, and
output directory you want Codex to use.

## Cloud Worker Requirements

- Python 3.11 is preferred.
- The worker must have enough disk space for `data/`, dependency caches, and
  `results/summary/`.
- The worker must be allowed to install Python packages from `pyproject.toml`.
- Long-running jobs should keep terminal logs and the `results/summary/`
  directory even if the process exits early.
- If foundation adapters are included, pass `HF_TOKEN` or
  `HUGGINGFACE_HUB_TOKEN` to the worker environment after accepting the gated
  model terms where required.

Do not run full manuscript or cloud-scale jobs on a local laptop unless you are
only doing a tiny scoped smoke run.

## Remote Setup

On the cloud worker, Codex should run:

```bash
PYTHON_BIN=python3.11 ./scripts/setup_env.sh
source .venv/bin/activate
python scripts/check_environment.py
```

For foundation-model runs:

```bash
INSTALL_EXTRAS=dev,foundation PYTHON_BIN=python3.11 ./scripts/setup_env.sh
source .venv/bin/activate
python scripts/check_environment.py --include-foundation
survarena foundation-check
```

## Dry Run First

Before fitting models, validate the resolved benchmark plan. Use the same
benchmark config that the real cloud run will use:

```bash
source .venv/bin/activate
python -m survarena.run_benchmark \
  --benchmark-config configs/benchmark/standard_v1.yaml \
  --dry-run
```

For a smaller manuscript or standard config dry run:

```bash
python -m survarena.run_benchmark \
  --benchmark-config configs/benchmark/standard_v1.yaml \
  --dry-run
```

The dry run should confirm the intended datasets, methods, seeds, repeats, and
comparison modes before any expensive fitting starts.

## Full Cloud Run

Use a shipped benchmark config directly:

```bash
source .venv/bin/activate
python -m survarena.run_benchmark \
  --benchmark-config configs/benchmark/standard_v1.yaml
```

Common cloud targets:

- `configs/benchmark/standard_v1.yaml`: standard no-HPO plus HPO comparison
- `configs/benchmark/manuscript_v1.yaml`: full native no-HPO manuscript run
- `configs/benchmark/manuscript_autogluon_v1.yaml`: AutoGluon-managed run
- `configs/benchmark/smoke_foundation.yaml`: optional foundation smoke run
- `configs/benchmark/smoke_aft.yaml`: AFT-only no-HPO plus minimal-HPO smoke
  run for Weibull, LogNormal, LogLogistic, XGBoost AFT, and CatBoost AFT

Keep the terminal session attached to the remote job manager or run it through
the cloud platform's persistent job mechanism. The important contract is that
the process continues on the remote worker after your local device disconnects.

## Sharded Cloud Runs

For practical cloud execution, prefer one dataset/method shard per worker. This
reduces wall-clock time and makes retries cheaper.

Use `--dataset` and `--method` to scope a worker:

```bash
python -m survarena.run_benchmark \
  --benchmark-config configs/benchmark/standard_v1.yaml \
  --dataset support \
  --method rsf
```

For the full native manuscript portfolio, shard the manuscript config the same
way:

```bash
python -m survarena.run_benchmark \
  --benchmark-config configs/benchmark/manuscript_v1.yaml \
  --dataset support \
  --method coxph
```

Give each cloud worker a clear shard name in the job label, for example
`survarena-support-rsf` or `survarena-whas500-deepsurv`.

## Resuming Failed Or Interrupted Runs

If a worker stops after creating an output directory, resume into that same
directory:

```bash
python -m survarena.run_benchmark \
  --benchmark-config <same-config-used-for-original-run> \
  --output-dir results/summary/<experiment_dir> \
  --resume \
  --max-retries 2
```

Use the same benchmark config, dataset, method, seeds, and comparison modes that
created the original output directory.

## Artifacts To Bring Back

At the end of each cloud job, ask Codex to return or persist:

- `results/summary/<experiment_dir>/README.md`
- `results/summary/<experiment_dir>/experiment_navigator.json`
- fold results, leaderboards, rank summaries, and statistical summaries
- compact run ledgers and indexes
- full terminal logs

Do not only copy the final leaderboard. The compact ledgers and manifest are
needed to audit and resume benchmark collections.

## Minimal Codex Completion Checklist

Ask Codex to report:

- the commit SHA and branch used on the cloud worker
- the exact setup, validation, dry-run, and benchmark commands
- the output directory path
- whether the run completed, failed, or was resumed
- the key artifact paths returned from `results/summary/`
- any failed method/dataset records emitted by the benchmark runner
