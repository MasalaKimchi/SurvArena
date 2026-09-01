from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
SMOKE_WORKFLOW = ROOT / ".github" / "workflows" / "benchmark-smoke.yml"
DOCS_INDEX = ROOT / "docs" / "index.md"

CHECKOUT_ACTION = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
SETUP_UV_ACTION = "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9"
UPLOAD_ACTION = "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
AUDITED_ACTIONS = {CHECKOUT_ACTION, SETUP_UV_ACTION, UPLOAD_ACTION}


def _load_pyproject() -> dict[str, object]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _load_workflow(path: Path) -> dict[str, object]:
    loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(loaded, dict)
    return loaded


def _steps(job: dict[str, object]) -> list[dict[str, object]]:
    steps = job.get("steps")
    assert isinstance(steps, list)
    assert all(isinstance(step, dict) for step in steps)
    return steps


def _commands(job: dict[str, object]) -> str:
    return "\n".join(str(step["run"]) for step in _steps(job) if "run" in step)


def _setup_uv_step(job: dict[str, object]) -> dict[str, object]:
    matches = [step for step in _steps(job) if step.get("uses") == SETUP_UV_ACTION]
    assert len(matches) == 1
    return matches[0]


def _dependency_profile(job: dict[str, object]) -> str:
    commands = _commands(job)
    markers = {
        "quality": "--only-group quality",
        "docs": "--only-group docs",
        "import-smoke": "--only-group import-smoke",
        "full-tests": "--no-default-groups --group test",
        "benchmark-smoke": "survarena benchmark run",
    }
    matches = [profile for profile, marker in markers.items() if marker in commands]
    assert len(matches) == 1, commands
    return matches[0]


def _tracked_markdown_guides() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", "docs"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    guides = set()
    for item in result.stdout.splitlines():
        path = Path(item)
        if path.suffix == ".md" and path != Path("docs/index.md"):
            guides.add(path.relative_to("docs").with_suffix("").as_posix())
    return guides


def _toctree_documents() -> list[str]:
    entries: list[str] = []
    in_toctree = False
    for line in DOCS_INDEX.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item == "```{toctree}":
            assert not in_toctree
            in_toctree = True
            continue
        if in_toctree and item == "```":
            in_toctree = False
            continue
        if not in_toctree or not item or item.startswith(":"):
            continue
        titled_target = re.fullmatch(r".+\s+<([^>]+)>", item)
        target = titled_target.group(1) if titled_target else item
        entries.append(target.removesuffix(".md"))
    assert not in_toctree
    return entries


def test_uv_dependency_groups_and_pytest_contract() -> None:
    config = _load_pyproject()
    project = config["project"]
    groups = config["dependency-groups"]
    pytest_options = config["tool"]["pytest"]["ini_options"]
    uv_config = config["tool"]["uv"]

    tomli = "tomli>=2.4.1,<3; python_version < '3.11'"
    quality = {"mypy==2.3.1", "ruff>=0.15,<0.16"}
    test = {"pytest>=8.3,<9", tomli}
    import_smoke = {
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scipy==1.13.1",
        "scikit-learn==1.6.1",
        "PyYAML==6.0.2",
        "pyarrow==20.0.0",
    }
    docs = {"Sphinx>=8.2.3,<9", "myst-parser>=5.1,<6"}

    assert project["requires-python"] == ">=3.10,<3.13"
    assert set(project["optional-dependencies"]["dev"]) == quality | test
    assert set(groups["quality"]) == quality
    assert set(groups["test"]) == test
    assert set(groups["import-smoke"]) == import_smoke
    assert set(groups["docs"]) == docs
    assert groups["dev"] == [{"include-group": "quality"}, {"include-group": "test"}]
    assert uv_config["required-version"] == "==0.12.8"
    assert uv_config["dependency-groups"]["docs"]["requires-python"] == ">=3.11"
    assert {"-ra", "--strict-config", "--strict-markers"} <= set(pytest_options["addopts"])
    assert pytest_options["xfail_strict"] is True


def test_all_tracked_markdown_guides_are_in_exactly_one_toctree() -> None:
    entries = _toctree_documents()
    assert len(entries) == len(set(entries))
    assert set(entries) == _tracked_markdown_guides()


def test_workflow_triggers_permissions_and_resource_guards() -> None:
    ci = _load_workflow(CI_WORKFLOW)
    smoke = _load_workflow(SMOKE_WORKFLOW)

    assert set(ci["on"]) == {"push", "pull_request", "workflow_dispatch"}
    assert ci["on"]["push"]["branches"] == ["main"]
    assert ci["on"]["pull_request"]["branches"] == ["main"]
    assert set(smoke["on"]) == {"workflow_dispatch"}

    for workflow in (ci, smoke):
        assert workflow["permissions"] == {"contents": "read"}
        assert workflow["concurrency"]["cancel-in-progress"] == "true"
        assert workflow["concurrency"]["group"]
        for job in workflow["jobs"].values():
            assert int(job["timeout-minutes"]) > 0
            assert "permissions" not in job

    for path in (CI_WORKFLOW, SMOKE_WORKFLOW):
        source = path.read_text(encoding="utf-8")
        lowered = source.lower()
        assert "pull_request_target" not in lowered
        assert re.search(r"(?m)^\s*schedule:", source) is None
        assert "${{ secrets." not in lowered
        assert "permissions: write" not in lowered
        assert re.search(r"\bdeploy(?:ment|ing)?\b", lowered) is None
        assert re.search(r"\bpublish(?:ing)?\b", lowered) is None


def test_workflow_matrices_actions_checkout_and_cache_profiles() -> None:
    ci = _load_workflow(CI_WORKFLOW)
    smoke = _load_workflow(SMOKE_WORKFLOW)
    ci_jobs = ci["jobs"]
    smoke_job = smoke["jobs"]["smoke"]

    assert set(ci_jobs) == {"quality", "type", "import-smoke", "full-tests", "docs"}
    assert ci_jobs["import-smoke"]["strategy"]["matrix"]["python-version"] == ["3.10", "3.11", "3.12"]
    assert ci_jobs["full-tests"]["strategy"]["matrix"]["python-version"] == ["3.10", "3.12"]

    expected_python = {
        "quality": "3.11",
        "type": "3.11",
        "import-smoke": "${{ matrix.python-version }}",
        "full-tests": "${{ matrix.python-version }}",
        "docs": "3.11",
    }

    all_jobs = [(name, job) for name, job in ci_jobs.items()]
    all_jobs.append(("smoke", smoke_job))
    for name, job in all_jobs:
        checkout_steps = [step for step in _steps(job) if step.get("uses") == CHECKOUT_ACTION]
        assert len(checkout_steps) == 1
        assert checkout_steps[0]["with"]["persist-credentials"] == "false"

        setup_step = _setup_uv_step(job)
        setup_options = setup_step["with"]
        assert setup_options["version"] == "0.12.8"
        assert setup_options["enable-cache"] == "true"
        assert setup_options["cache-dependency-glob"] == "uv.lock"
        assert setup_options["cache-suffix"] == _dependency_profile(job)
        assert setup_options["python-version"] == expected_python.get(name, "3.11")

        for step in _steps(job):
            action = step.get("uses")
            if action is None:
                continue
            assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", action)
            assert action in AUDITED_ACTIONS


def test_workflow_commands_are_locked_and_match_local_quality_gates() -> None:
    ci = _load_workflow(CI_WORKFLOW)
    smoke = _load_workflow(SMOKE_WORKFLOW)
    ci_jobs = ci["jobs"]

    assert "uv lock --check" in _commands(ci_jobs["quality"])
    assert (
        "uv run --no-sync ruff check --output-format=github survarena tests scripts"
        in _commands(ci_jobs["quality"])
    )
    mypy_commands = [
        str(step["run"])
        for step in _steps(ci_jobs["type"])
        if "python -m mypy" in str(step.get("run", ""))
    ]
    assert len(mypy_commands) == 1
    mypy_tokens = shlex.split(mypy_commands[0])
    assert mypy_tokens[mypy_tokens.index("mypy") + 1 :] == [
        "survarena/core",
        "survarena/benchmark/resume.py",
        "survarena/data/splitters.py",
        "scripts/audit_manuscript_publishability.py",
    ]
    assert "uv run --no-sync python -c \"import survarena" in _commands(ci_jobs["import-smoke"])
    assert "uv run --no-sync python -m pytest -q" in _commands(ci_jobs["full-tests"])
    assert "uv run --no-sync python -m compileall -q survarena" in _commands(ci_jobs["full-tests"])
    assert "uv run --no-sync sphinx-build -n -W --keep-going" in _commands(ci_jobs["docs"])
    assert "uv run --no-sync survarena benchmark run" in _commands(smoke["jobs"]["smoke"])

    for workflow in (ci, smoke):
        for job in workflow["jobs"].values():
            locked_sync_indices = []
            run_steps = [(index, step) for index, step in enumerate(_steps(job)) if "run" in step]
            for index, step in run_steps:
                command = step.get("run")
                command = str(command).strip()
                if command.startswith("uv sync"):
                    assert "--locked" in command
                    locked_sync_indices.append(index)
                elif command == "uv lock --check":
                    continue
                else:
                    assert command.startswith("uv run --no-sync")
            assert locked_sync_indices
            for index, step in run_steps:
                if re.search(r"(?m)^\s*uv\s+run\s+--no-sync\b", str(step["run"])):
                    assert any(sync_index < index for sync_index in locked_sync_indices)
