from __future__ import annotations

import os
import sys
from pathlib import Path

# Stabilize OpenMP-backed extension modules (XGBoost, etc.) during pytest;
# keeps default if the user already set these.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("MKL_NUM_THREADS", "1")


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
