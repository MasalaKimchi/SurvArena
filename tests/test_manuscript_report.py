from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_no_hpo_benchmark_report.py"
AUDIT_SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_manuscript_publishability.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_no_hpo_benchmark_report", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_manuscript_publishability", AUDIT_SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_classify_cell_distinguishes_coverage_states() -> None:
    module = _load_module()

    assert module.classify_cell(success_splits=15, attempted_splits=15, expected_splits=15) == "complete"
    assert module.classify_cell(success_splits=4, attempted_splits=15, expected_splits=15) == "partial"
    assert module.classify_cell(success_splits=0, attempted_splits=15, expected_splits=15) == "failed"
    assert module.classify_cell(success_splits=0, attempted_splits=3, expected_splits=15) == "attempted_incomplete"
    assert module.classify_cell(success_splits=0, attempted_splits=0, expected_splits=15) == "missing"


def test_audit_markdown_table_is_dependency_free_and_deterministic() -> None:
    module = _load_audit_module()
    frame = pd.DataFrame(
        [
            {"name": "alpha|beta", "detail": "line 1\nline 2", "value": 3},
            {"name": r"path\part", "detail": None, "value": np.nan},
        ]
    )

    rendered = module._markdown_table(frame, ["name", "detail", "value"])

    assert rendered == "\n".join(
        [
            "| name | detail | value |",
            "| --- | --- | --- |",
            r"| alpha\|beta | line 1<br>line 2 | 3.0 |",
            r"| path\\part | NA | NA |",
        ]
    )
    assert module._markdown_table(frame.iloc[0:0], ["name", "detail", "value"]) == "_No rows._"


def test_strict_audit_reaches_verdict_without_rendering_traceback(tmp_path: Path) -> None:
    report_path = tmp_path / "audit.md"
    completed = subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT_PATH), "--strict", "--write-doc", str(report_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "publishable=false" in completed.stdout
    assert "Traceback" not in completed.stderr
    report = report_path.read_text(encoding="utf-8")
    assert report.startswith("# Manuscript Publishability Audit")
    assert "invalidated as current release evidence" in report
    assert "Regenerate all citable matrices" in report
