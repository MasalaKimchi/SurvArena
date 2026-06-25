from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_no_hpo_benchmark_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_no_hpo_benchmark_report", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_classify_cell_distinguishes_coverage_states() -> None:
    module = _load_module()

    assert module.classify_cell(success_splits=15, attempted_splits=15, expected_splits=15) == "complete"
    assert module.classify_cell(success_splits=4, attempted_splits=15, expected_splits=15) == "partial"
    assert module.classify_cell(success_splits=0, attempted_splits=15, expected_splits=15) == "failed"
    assert module.classify_cell(success_splits=0, attempted_splits=3, expected_splits=15) == "attempted_incomplete"
    assert module.classify_cell(success_splits=0, attempted_splits=0, expected_splits=15) == "missing"
