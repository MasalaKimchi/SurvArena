# Canonical SurvArena runtime: a pinned linux/amd64 Python 3.11 image.
#
# The maintainer develops on an arm64 Mac, which is DEV-ONLY. Citable /
# reproducible benchmark numbers are meant to come from this amd64 container
# (and the matching CI runners), never from the Mac. See
# docs/ci_and_reproducibility.md for the amd64-is-canonical rule.
#
# The platform is pinned to linux/amd64 so the image is canonical by
# construction. On an arm64 Mac this builds/runs under emulation (slower); that
# is acceptable because the Mac is dev-only. Native amd64 CI runners build it at
# full speed. Drop the `--platform` pin only if you deliberately want a fast
# arm64 dev image (whose numbers are NOT citable).
FROM --platform=linux/amd64 python:3.11-slim

# Keep pip quiet and cache-free; unbuffer stdout for readable container logs.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# Minimal OS deps for the scientific stack:
#   build-essential - C/C++ toolchain for any dependency lacking a manylinux wheel
#   git             - editable-install metadata and any pip VCS installs
#   libgomp1        - OpenMP runtime required at import time by xgboost / catboost
#                     (and lightgbm), which the torch-free classical models use
# This apt layer sits above every COPY, so it is cached across source changes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the packaging metadata first. README.md is required because pyproject
# declares `readme = "README.md"`. This is the layer where a hashed lockfile
# (uv / pip-tools) SHOULD be installed once it exists - at that point the
# dependency install can run before the source copy for real layer caching.
# Until then, an editable install needs the full source, so the heavy install
# below still runs after `COPY . .` (see docs/ci_and_reproducibility.md TODO).
COPY pyproject.toml README.md ./

# Copy the rest of the repo (respecting .dockerignore) and install editable
# with the dev tooling. No foundation extras - heavy adapters stay opt-in.
COPY . .
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -e ".[dev]"

# Default entrypoint: show the CLI help. Override to run real work, e.g.:
#   docker run --rm survarena \
#     benchmark run --config configs/benchmark/manuscript_v1.yaml --dry-run
CMD ["survarena", "--help"]
