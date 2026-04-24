# External Integrations

**Analysis Date:** 2026-04-23

## APIs & External Services

**Dataset and model ecosystems:**
- `scikit-survival` datasets are consumed in `survarena/data/loaders.py` (`load_aids`, `load_gbsg2`, `load_flchain`, `load_whas500`).
  - SDK/Client: `sksurv.datasets`
  - Auth: Not required
- `pycox` datasets are consumed in `survarena/data/loaders.py` (`support`, `metabric`).
  - SDK/Client: `pycox.datasets`
  - Auth: Not required
- Optional gated TabPFN checkpoint access is validated in `survarena/methods/foundation/readiness.py`.
  - SDK/Client: `tabpfn`, `huggingface_hub`
  - Auth: `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` (or `hf auth login`)

## Data Storage

**Databases:**
- Not detected in `survarena/` (no SQL/NoSQL client usage in runtime paths).
  - Connection: Not applicable
  - Client: Not applicable

**File Storage:**
- Local filesystem only for artifacts and manifests, written by `survarena/logging/export.py` and `survarena/logging/tracker.py`.
- Split cache persisted under `data/splits/` by `survarena/data/splitters.py`.

**Caching:**
- Local matplotlib cache rooted under `/tmp/survarena_mpl_cache` from `survarena/api/predictor.py`.
- No Redis/Memcached/distributed cache integration detected.

## Authentication & Identity

**Auth Provider:**
- Custom user auth is not implemented.
  - Implementation: optional token-based Hugging Face auth checks only for foundation model readiness in `survarena/methods/foundation/readiness.py`.

## Monitoring & Observability

**Error Tracking:**
- None detected (no Sentry/Datadog/Honeycomb integration in `survarena/`).

**Logs:**
- Structured benchmark and run payload logs are written to JSON/CSV/JSONL.GZ via `survarena/logging/export.py` and `survarena/logging/tracker.py`.
- Runtime memory sampling uses `psutil` in `survarena/logging/tracker.py`.

## CI/CD & Deployment

**Hosting:**
- Not applicable for the current codebase shape (CLI toolkit; no service deployment manifests).

**CI Pipeline:**
- Not detected; `.github/workflows/` exists and is empty.

## Environment Configuration

**Required env vars:**
- No mandatory env vars for base predictor/benchmark flows in `survarena/cli.py` and `survarena/run_benchmark.py`.
- Optional foundation auth env vars: `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN` in `survarena/methods/foundation/readiness.py`.
- Runtime process env vars set by code: `MPLCONFIGDIR`, `XDG_CACHE_HOME` in `survarena/api/predictor.py`; `PYTHONHASHSEED` in `survarena/utils/seeds.py`.

**Secrets location:**
- Process environment and Hugging Face local auth state, referenced by `survarena/methods/foundation/readiness.py`.

## Webhooks & Callbacks

**Incoming:**
- None detected.

**Outgoing:**
- None detected.

---

*Integration audit: 2026-04-23*
# External Integrations

**Analysis Date:** 2026-04-23

## APIs & External Services

**Public data/model ecosystems:**
- PyCox dataset provider (`support`, `metabric`) used by built-in loaders.
  - SDK/Client: `pycox.datasets` via `survarena/data/loaders.py`
  - Auth: None
- scikit-survival dataset provider (`aids`, `gbsg2`, `flchain`, `whas500`) used by built-in loaders.
  - SDK/Client: `sksurv.datasets` via `survarena/data/loaders.py`
  - Auth: None
- Hugging Face-hosted gated TabPFN checkpoint (default TabPFN runtime path when checkpoint is not local).
  - SDK/Client: `tabpfn` plus `huggingface_hub` resolution in `survarena/methods/foundation/readiness.py`
  - Auth: `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` (or `hf auth login`)

**ML platform backends:**
- AutoGluon Tabular backend for `autogluon_survival` and Mitra foundation adapter.
  - SDK/Client: `autogluon.tabular` in `survarena/automl/autogluon_backend.py` and `survarena/methods/foundation/mitra_survival.py`
  - Auth: None
- Optional experiment-tracking extras are declared but not wired in repository runtime paths.
  - SDK/Client: `mlflow`, `wandb` in optional dependency group `tracking` (`pyproject.toml`)
  - Auth: Not applicable in current code paths

## Data Storage

**Databases:**
- Not detected.
  - Connection: Not applicable
  - Client: Not applicable

**File Storage:**
- Local filesystem only.
  - Experiment artifacts: `results/summary/exp_*` written via `survarena/logging/export.py`
  - Predictor artifacts: `results/predictor/<dataset_name>/` managed via `survarena/api/predictor.py`
  - Split cache: `data/splits/<task_id>/` managed via split logic in `survarena/data/splitters.py`
  - Data IO formats: CSV and Parquet through `survarena/data/io.py`

**Caching:**
- Runtime temp/cache directories are local only (`MPLCONFIGDIR`, `XDG_CACHE_HOME`) in `survarena/api/predictor.py`.
- No Redis/Memcached or distributed cache integration detected.

## Authentication & Identity

**Auth Provider:**
- Custom application auth not implemented (toolkit is local CLI-driven).
  - Implementation: Optional token-based Hugging Face auth for gated TabPFN model access in `survarena/methods/foundation/readiness.py`.

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry/Datadog/Rollbar integrations detected in `survarena/`).

**Logs:**
- Local structured manifests and ledgers.
  - Run manifests: `survarena/logging/manifest.py`
  - JSON/JSONL.GZ writers and hashing: `survarena/logging/tracker.py`
  - Benchmark exports and experiment navigator: `survarena/logging/export.py`

## CI/CD & Deployment

**Hosting:**
- Not detected (repository ships as CLI/library, not a hosted service).

**CI Pipeline:**
- None detected (`.github/workflows/*.yml` not present).

## Environment Configuration

**Required env vars:**
- No mandatory secret env vars for default local benchmark flow.
- Optional auth vars for gated TabPFN: `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN` (`survarena/methods/foundation/readiness.py`).
- Runtime/script control vars:
  - `MPLCONFIGDIR`, `XDG_CACHE_HOME` (`survarena/api/predictor.py`)
  - `PYTHONHASHSEED` (`survarena/utils/seeds.py`)
  - `PYTHON_BIN`, `VENV_DIR`, `INSTALL_EXTRAS` (`scripts/setup_env.sh`)
  - `DATASET`, `METHOD`, `PYTHONUNBUFFERED` (`scripts/run_cloud_comprehensive.sh`)

**Secrets location:**
- Process environment and local Hugging Face auth store (`hf auth login`) referenced in `survarena/methods/foundation/readiness.py`.

## Webhooks & Callbacks

**Incoming:**
- None.

**Outgoing:**
- None.

---

*Integration audit: 2026-04-23*
