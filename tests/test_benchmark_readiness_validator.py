from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import subprocess
import sys
import traceback

import pytest

from scripts import validate_benchmark_readiness as validator


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assert_sensitive_absent(sensitive: str, *outputs: str) -> None:
    if any(sensitive in output for output in outputs):
        pytest.fail("credential-shaped test value reached output", pytrace=False)


def _capture_validation_error(operation: Callable[[], object]) -> validator.ValidationError:
    caught: validator.ValidationError | None = None
    unexpected_error = False
    try:
        operation()
    except validator.ValidationError as error:
        caught = error
    except Exception:
        unexpected_error = True
    if unexpected_error:
        pytest.fail("validator exposed an unexpected exception type", pytrace=False)
    if caught is None:
        pytest.fail("validator did not fail closed", pytrace=False)
    return caught


def _safe_exception_surfaces(
    error: validator.ValidationError,
    sensitive_values: list[str],
) -> tuple[str, str, str]:
    if error.__context__ is not None:
        pytest.fail("sanitized exception retained raw context", pytrace=False)
    if error.__cause__ is not None:
        pytest.fail("sanitized exception retained raw cause", pytrace=False)
    surfaces = (
        str(error),
        repr(error),
        "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    )
    for sensitive in sensitive_values:
        _assert_sensitive_absent(sensitive, *surfaces)
    return surfaces


def _credential_filename(kind: str) -> str:
    values = {
        "hugging_face": "hf_" + "a" * 32,
        "aws_access_id": "AKIA" + "B" * 16,
        "aws_access_assignment": "AWS_ACCESS_KEY_ID=" + "C" * 20,
        "aws_secret_assignment": "AWS_SECRET_ACCESS_KEY=" + "D" * 40,
        "aws_session_assignment": "AWS_SESSION_TOKEN=" + "E" * 40,
        "github": "ghp_" + "c" * 36,
        "openai": "sk-proj-" + "d" * 32,
        "bearer": "Bearer " + "e" * 32,
        "private_key": "-----BEGIN PRIVATE KEY-----",
        "encrypted_private_key": "-----BEGIN ENCRYPTED PRIVATE KEY-----",
        "gitlab": "glpat-" + "g" * 24,
        "slack": "xoxb-" + "1" * 12 + "-" + "h" * 24,
        "generic": "password=" + "f" * 24,
    }
    return values[kind] + ".txt"


def _git_add_sensitive_fixture(repo: Path, filename: str) -> None:
    failed = False
    try:
        validator._git_text(repo, "add", "--", filename)
    except Exception:
        failed = True
    if failed:
        pytest.fail("credential-path Git fixture setup failed", pytrace=False)


def _write_sensitive_fixture(path: Path, value: str) -> None:
    failed = False
    try:
        path.write_text(value, encoding="utf-8")
    except Exception:
        failed = True
    if failed:
        pytest.fail("credential-path file fixture setup failed", pytrace=False)


def _commit_sensitive_fixture(repo: Path, filename: str) -> None:
    _git_add_sensitive_fixture(repo, filename)
    failed = False
    try:
        validator._git_text(repo, "commit", "-q", "-m", "credential-path fixture")
    except Exception:
        failed = True
    if failed:
        pytest.fail("credential-path commit fixture setup failed", pytrace=False)


def _current_inputs() -> tuple[str, str, str, str]:
    return (
        _read(validator.READINESS_PATH),
        _read(validator.INDEX_PATH),
        _read(validator.ROADMAP_PATH),
        _read(validator.REQUIREMENTS_PATH),
    )


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1
    return text.replace(old, new, 1)


def _set_register_status(text: str, row_id: str, old: str, new: str) -> str:
    lines = text.splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines) if line.startswith(f"| {row_id} |")]
    assert len(matches) == 1
    index = matches[0]
    needle = f" | {old} | "
    assert lines[index].count(needle) == 1
    lines[index] = lines[index].replace(needle, f" | {new} | ", 1)
    return "".join(lines)


def test_current_readiness_contract_passes_offline() -> None:
    counts = validator.validate_files(repo=ROOT)

    assert counts == {
        "wrong_rows": 9,
        "domain_rows": 13,
        "register_rows": 19,
        "crosswalk_rows": 22,
        "verified_tags": 93,
        "local_links": 25,
        "git_scope_checked": 0,
        "git_ignored_paths_checked": 0,
    }


@pytest.mark.parametrize(
    ("value", "expected_pattern"),
    [
        ("AKIA" + "A" * 16, "aws_access_key_id"),
        ("AWS_ACCESS_KEY_ID=" + "B" * 20, "aws_access_key_id_assignment"),
        ("AWS_SECRET_ACCESS_KEY=" + "C" * 40, "aws_secret_access_key_assignment"),
        ("AWS_SESSION_TOKEN=" + "D" * 40, "aws_session_token_assignment"),
        ("ghp_" + "e" * 36, "github_token"),
        ("sk-proj-" + "f" * 32, "openai_token"),
        ("hf_" + "g" * 32, "hugging_face_token"),
        ("Bearer " + "h" * 32, "bearer_token"),
        ("-----BEGIN " + "PRIVATE KEY-----", "private_key"),
        ("client_secret=" + "i" * 24, "credential_assignment"),
        ("token=" + "j" * 24, "credential_assignment"),
        ("X-Amz-Signature=" + "k" * 64, "credential_assignment"),
    ],
)
def test_secret_scanner_reports_only_safe_metadata(
    tmp_path: Path,
    value: str,
    expected_pattern: str,
) -> None:
    target = tmp_path / "target.md"
    target.write_text(f"safe prefix {value} safe suffix\n", encoding="utf-8")

    findings = validator.scan_credentials([target], repo=tmp_path)
    rendered = validator.format_secret_findings(findings)

    assert expected_pattern in {finding.pattern for finding in findings}
    _assert_sensitive_absent(value, rendered)
    assert set(rendered) >= {"{", "}"}


def test_secret_gate_precedes_document_and_network_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "AWS_SESSION_TOKEN=" + "j" * 40
    readiness = tmp_path / "readiness.md"
    index = tmp_path / "index.md"
    roadmap = tmp_path / "roadmap.md"
    requirements = tmp_path / "requirements.md"
    readiness.write_text(f"[unsafe](https://example.test/?value={secret})\n", encoding="utf-8")
    for path in (index, roadmap, requirements):
        path.write_text("malformed by design\n", encoding="utf-8")

    parsed = False
    fetched = False

    def fail_if_parsed(*_args: object, **_kwargs: object) -> dict[str, int]:
        nonlocal parsed
        parsed = True
        raise AssertionError("document parsing ran before the secret gate")

    def fail_if_fetched(_url: str) -> tuple[int, str]:
        nonlocal fetched
        fetched = True
        raise AssertionError("network access ran before the secret gate")

    monkeypatch.setattr(validator, "validate_structure", fail_if_parsed)
    with pytest.raises(validator.ValidationError) as caught:
        validator.validate_files(
            readiness_path=readiness,
            index_path=index,
            roadmap_path=roadmap,
            requirements_path=requirements,
            repo=ROOT,
            external=True,
            fetch=fail_if_fetched,
        )

    assert caught.value.exit_code == validator.EXIT_SECRET
    _assert_sensitive_absent(secret, str(caught.value))
    assert not parsed
    assert not fetched


def test_percent_encoded_secret_is_detected_without_value_disclosure(tmp_path: Path) -> None:
    value = "ghp_" + "k" * 36
    encoded_once = "".join(f"%{ord(character):02X}" for character in value)
    encoded = encoded_once.replace("%", "%25")
    target = tmp_path / "encoded.md"
    target.write_text(encoded + "\n", encoding="utf-8")

    findings = validator.scan_credentials([target], repo=tmp_path)
    rendered = validator.format_secret_findings(findings)

    assert any(finding.pattern == "github_token" for finding in findings)
    _assert_sensitive_absent(value, rendered)
    _assert_sensitive_absent(encoded, rendered)


def test_secret_finding_json_redacts_a_credential_shaped_filename(tmp_path: Path) -> None:
    filename = _credential_filename("hugging_face")
    target = tmp_path / filename
    target.write_text("AWS_SESSION_TOKEN=" + "x" * 40 + "\n", encoding="utf-8")

    rendered = validator.format_secret_findings(validator.scan_credentials([target], repo=tmp_path))

    _assert_sensitive_absent(filename, rendered)
    assert "<redacted-path>" in rendered


@pytest.mark.parametrize("credential_kind", ["hugging_face", "gitlab"])
def test_credential_scan_fails_closed_without_leaking_self_referential_symlink_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    credential_kind: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    filename = _credential_filename(credential_kind)
    target = repo / filename
    setup_failed = False
    try:
        target.symlink_to(filename)
    except Exception:
        setup_failed = True
    if setup_failed:
        pytest.fail("credential-path symlink fixture setup failed", pytrace=False)

    error = _capture_validation_error(lambda: validator.scan_credentials([target], repo=repo))
    surfaces = _safe_exception_surfaces(error, [filename, str(target)])
    rendered = surfaces[0]

    print("\n".join(surfaces))
    print("\n".join(surfaces), file=sys.stderr)
    captured = capsys.readouterr()
    outputs = (rendered, captured.out, captured.err)
    _assert_sensitive_absent(filename, *outputs)
    _assert_sensitive_absent(str(target), *outputs)
    assert "unable to read credential-scan input" in rendered
    assert "<redacted-path>" in rendered
    assert "path_id=" in rendered


@pytest.mark.parametrize("failure_kind", ["read", "stat", "decode"])
def test_path_failures_clear_context_cause_and_all_rendered_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_kind: str,
) -> None:
    filename = _credential_filename("hugging_face")
    target = tmp_path / filename
    operation: Callable[[], object]

    if failure_kind == "read":
        _write_sensitive_fixture(target, "read failure\n")
        original_read_text = Path.read_text

        def fail_read(path: Path, *args: object, **kwargs: object) -> str:
            if path == target:
                raise PermissionError(str(path))
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", fail_read)

        def operation() -> object:
            return validator.scan_credentials([target], repo=tmp_path)

    elif failure_kind == "stat":
        repo = tmp_path / "repo"
        repo.mkdir()
        validator._init_scope_test_repo(repo)
        target = repo / filename
        _write_sensitive_fixture(target, "stat failure\n")
        original_stat = Path.stat

        def fail_stat(path: Path, *args: object, **kwargs: object) -> object:
            if path == target:
                raise OSError(str(path))
            return original_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", fail_stat)

        def operation() -> object:
            return validator.create_git_baseline(repo)

    else:
        try:
            target.write_bytes(b"\xff\xfeinvalid-utf8")
        except Exception:
            pytest.fail("credential-path decode fixture setup failed", pytrace=False)
        def operation() -> object:
            return validator.validate_files(
                readiness_path=target,
                index_path=validator.INDEX_PATH,
                roadmap_path=validator.ROADMAP_PATH,
                requirements_path=validator.REQUIREMENTS_PATH,
                repo=tmp_path,
            )

    error = _capture_validation_error(operation)
    surfaces = _safe_exception_surfaces(error, [filename, str(target)])
    print("\n".join(surfaces))
    print("\n".join(surfaces), file=sys.stderr)
    captured = capsys.readouterr()
    _assert_sensitive_absent(filename, captured.out, captured.err)
    _assert_sensitive_absent(str(target), captured.out, captured.err)


def test_malformed_cli_never_echoes_user_controlled_arguments(tmp_path: Path) -> None:
    script = ROOT / "scripts/validate_benchmark_readiness.py"
    baseline_name = _credential_filename("slack")
    baseline_loop = tmp_path / baseline_name
    setup_failed = False
    try:
        baseline_loop.symlink_to(baseline_name)
    except Exception:
        setup_failed = True
    if setup_failed:
        pytest.fail("credential-path CLI fixture setup failed", pytrace=False)

    url_userinfo = "url-user-marker:url-password-marker"
    url_query = "api_key=" + "z" * 24
    url_fragment = "fragment-marker"
    unsafe_url = f"https://{url_userinfo}@example.invalid/path?{url_query}#{url_fragment}"
    cases = [
        (["self-test", _credential_filename("hugging_face")], [_credential_filename("hugging_face")]),
        ([_credential_filename("github")], [_credential_filename("github")]),
        (["check", "--external", _credential_filename("openai")], [_credential_filename("openai")]),
        (["check", "--" + _credential_filename("generic")], [_credential_filename("generic")]),
        (
            ["check", "--baseline", str(baseline_loop), "--baseline-sha256", "0" * 64],
            [baseline_name, str(baseline_loop)],
        ),
        (["self-test", unsafe_url], [unsafe_url, url_userinfo, url_query, url_fragment]),
    ]

    for arguments, sensitive_values in cases:
        result: subprocess.CompletedProcess[str] | None = None
        execution_failed = False
        try:
            result = subprocess.run(
                [sys.executable, str(script), *arguments],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            execution_failed = True
        if execution_failed or result is None:
            pytest.fail("malformed CLI fixture could not execute", pytrace=False)
        if result.returncode == 0:
            pytest.fail("malformed CLI input did not fail closed", pytrace=False)
        for sensitive in sensitive_values:
            _assert_sensitive_absent(sensitive, result.stdout, result.stderr)
        if "Traceback" in result.stdout or "Traceback" in result.stderr:
            pytest.fail("malformed CLI emitted a traceback", pytrace=False)


def test_url_diagnostics_mask_userinfo_query_and_fragment() -> None:
    user = "user-marker"
    password = "password-marker"
    query = "query-marker"
    fragment = "fragment-marker"
    url = f"https://{user}:{password}@www.acm.org/path?value={query}#{fragment}"

    masked = validator.mask_url(url)
    with pytest.raises(validator.ValidationError) as caught:
        validator._validate_static_external_url(url)

    for marker in (user, password, query, fragment):
        assert marker not in masked
        assert marker not in str(caught.value)
    assert masked == "https://www.acm.org/path"
    assert "url_id=" in str(caught.value)


def test_effective_url_is_secret_scanned_before_diagnostics() -> None:
    secret = "ghp_" + "z" * 36
    source = "https://www.acm.org/publications/policies/artifact-review-and-badging-current"
    readiness = f"[source]({source})"

    with pytest.raises(validator.ValidationError) as caught:
        validator.check_external_links(
            readiness,
            fetch=lambda _url: (200, f"https://www.acm.org/path?token={secret}"),
        )

    assert caught.value.exit_code == validator.EXIT_SECRET
    _assert_sensitive_absent(secret, str(caught.value))


def test_unapproved_external_host_is_rejected_before_transport() -> None:
    url = "https://127.0.0.1/private"

    with pytest.raises(validator.ValidationError) as caught:
        validator._validate_static_external_url(url)

    assert "127.0.0.1" not in str(caught.value)
    assert "url_id=" in str(caught.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.acm.org/path?",
        "https://www.acm.org/path#",
        "https://www.acm.org/path%253Fhidden",
        "https://www.acm.org/path%250Ahidden",
        "https://www.acm.org/path%255Chidden",
    ],
)
def test_encoded_or_empty_url_delimiters_are_rejected_without_echo(url: str) -> None:
    with pytest.raises(validator.ValidationError) as caught:
        validator._validate_static_external_url(url)

    assert url not in str(caught.value)
    assert "url_id=" in str(caught.value)


@pytest.mark.parametrize(
    "value",
    [
        "see BR-001",
        "`BR-001`",
        "BR-001,BR-002",
        "BR-002, BR-001",
        "BR-001, BR-001",
        "BR-999",
        "BR-001, None",
        "BR-001\u00a0",
    ],
)
def test_reference_cells_reject_noncanonical_or_unknown_values(value: str) -> None:
    with pytest.raises(validator.ValidationError):
        validator.parse_reference_list(value, prefix="BR", known={"BR-001", "BR-002"})


def test_structure_rejects_unknown_crosswalk_reference() -> None:
    readiness, index, roadmap, requirements = _current_inputs()
    readiness = _replace_once(readiness, "| P0-A | BR-001 |", "| P0-A | BR-999 |")

    with pytest.raises(validator.ValidationError, match="unknown BR"):
        validator.validate_structure(readiness, index, roadmap, requirements)


def test_requirement_checklist_and_traceability_status_must_agree() -> None:
    readiness, index, roadmap, requirements = _current_inputs()
    requirements = _replace_once(
        requirements,
        "| EXEC-01 | Phase 3 | Pending |",
        "| EXEC-01 | Phase 3 | Complete |",
    )

    with pytest.raises(validator.ValidationError, match="statuses differ"):
        validator.validate_structure(readiness, index, roadmap, requirements)


def test_related_phase_list_must_be_sorted_and_unique() -> None:
    readiness, index, roadmap, requirements = _current_inputs()
    readiness = _replace_once(
        readiness,
        "Phases 2 and 6 / STAT-01, STAT-02, STAT-03, STAT-07, REPR-05",
        "Phases 6 and 2 / STAT-01, STAT-02, STAT-03, STAT-07, REPR-05",
    )

    with pytest.raises(validator.ValidationError, match="sorted and unique"):
        validator.validate_structure(readiness, index, roadmap, requirements)


def test_structure_rejects_self_dependency() -> None:
    readiness, index, roadmap, requirements = _current_inputs()
    readiness = _replace_once(
        readiness,
        "| BR-004, BR-007, BR-008, BR-010 | Gate B |",
        "| BR-004, BR-007, BR-008, BR-009, BR-010 | Gate B |",
    )

    with pytest.raises(validator.ValidationError, match="self-dependency"):
        validator.validate_structure(readiness, index, roadmap, requirements)


def test_structure_rejects_dependency_cycle() -> None:
    readiness, index, roadmap, requirements = _current_inputs()
    readiness = _replace_once(
        readiness,
        "| None | Gate B | Phases 4 and 5 / CORE-02, STORE-01, SUIT-01, SUIT-02, SUIT-03 |",
        "| BR-009 | Gate B | Phases 4 and 5 / CORE-02, STORE-01, SUIT-01, SUIT-02, SUIT-03 |",
    )

    with pytest.raises(validator.ValidationError, match="cycle"):
        validator.validate_structure(readiness, index, roadmap, requirements)


def test_partial_release_row_cannot_be_marked_verified() -> None:
    readiness, index, roadmap, requirements = _current_inputs()
    readiness = _set_register_status(readiness, "BR-001", "in_progress", "verified")

    with pytest.raises(validator.ValidationError, match="must remain in_progress"):
        validator.validate_structure(readiness, index, roadmap, requirements)


def test_deferred_scope_row_cannot_be_framed_as_current_work() -> None:
    readiness, index, roadmap, requirements = _current_inputs()
    readiness = _replace_once(
        readiness,
        "| Benchmark maintainer | deferred_scope | [VERIFIED: `.planning/PROJECT.md:44-50`] | DOM-12 |",
        "| Benchmark maintainer | open | [VERIFIED: `.planning/PROJECT.md:44-50`] | DOM-12 |",
    )

    with pytest.raises(validator.ValidationError, match="future-scope semantics"):
        validator.validate_structure(readiness, index, roadmap, requirements)


def test_superseded_status_requires_known_replacement_ids() -> None:
    readiness, index, roadmap, requirements = _current_inputs()
    readiness = _set_register_status(readiness, "BR-003", "open", "superseded")

    with pytest.raises(validator.ValidationError, match="replacement IDs"):
        validator.validate_structure(readiness, index, roadmap, requirements)


def test_malformed_verified_tag_is_rejected(tmp_path: Path) -> None:
    readiness = tmp_path / "readiness.md"
    readiness.write_text(_read(validator.READINESS_PATH) + "\n[VERIFIED: malformed]\n", encoding="utf-8")

    with pytest.raises(validator.ValidationError, match="malformed VERIFIED"):
        validator.validate_provenance(readiness, repo=ROOT)


def test_invalid_last_verified_date_is_rejected() -> None:
    readiness, index, roadmap, requirements = _current_inputs()
    readiness = _replace_once(readiness, "| 2026-09-01 |\n| BR-002 |", "| 2026-02-30 |\n| BR-002 |")

    with pytest.raises(validator.ValidationError, match="valid ISO date"):
        validator.validate_structure(readiness, index, roadmap, requirements)


@pytest.mark.parametrize(
    "credential_kind",
    [
        "hugging_face",
        "aws_access_id",
        "aws_access_assignment",
        "aws_secret_assignment",
        "aws_session_assignment",
        "github",
        "openai",
        "bearer",
        "private_key",
        "encrypted_private_key",
        "gitlab",
        "slack",
        "generic",
    ],
)
@pytest.mark.parametrize("change_kind", ["dirty", "untracked", "committed"])
def test_credential_shaped_git_paths_never_reach_metadata_or_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    credential_kind: str,
    change_kind: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    validator._init_scope_test_repo(repo)
    filename = _credential_filename(credential_kind)
    target = repo / filename

    if change_kind == "dirty":
        _write_sensitive_fixture(target, "committed\n")
        _commit_sensitive_fixture(repo, filename)
        _write_sensitive_fixture(target, "baseline dirty\n")
        baseline = validator.create_git_baseline(repo)
        _write_sensitive_fixture(target, "changed dirty\n")
    else:
        baseline = validator.create_git_baseline(repo)
        _write_sensitive_fixture(target, "new\n")
        if change_kind == "committed":
            _commit_sensitive_fixture(repo, filename)

    issues = validator.check_git_scope(baseline, repo=repo, allow_paths=[])
    if not issues:
        pytest.fail("credential-path scope fixture was not detected", pytrace=False)
    baseline_json = json.dumps(baseline, sort_keys=True)
    diagnostics = validator.format_scope_issues(issues)
    error_text = str(validator.ValidationError("Git scope violations:\n" + diagnostics))
    print(diagnostics)
    print(error_text, file=sys.stderr)
    captured = capsys.readouterr()

    _assert_sensitive_absent(
        filename,
        baseline_json,
        diagnostics,
        error_text,
        captured.out,
        captured.err,
    )
    if "<redacted-path>" not in diagnostics or "path_id=" not in diagnostics:
        pytest.fail("scope diagnostic omitted its safe redacted path identity", pytrace=False)
    if any(len(path_key) != 64 for path_key in baseline["dirty_entries"]):
        pytest.fail("baseline path identity is not a digest", pytrace=False)


def test_unreadable_credential_shaped_path_raises_only_a_safe_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    validator._init_scope_test_repo(repo)
    filename = _credential_filename("gitlab")
    target = repo / filename
    _write_sensitive_fixture(target, "unreadable\n")
    original_read_bytes = Path.read_bytes

    def fail_sensitive_read(path: Path) -> bytes:
        if path == target:
            raise PermissionError(str(path))
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_sensitive_read)
    rendered: str | None = None
    unexpected_error = False
    try:
        validator.create_git_baseline(repo)
    except validator.ValidationError as error:
        rendered = str(error)
    except Exception:
        unexpected_error = True
    if unexpected_error:
        pytest.fail("path identity exposed a non-redacted exception boundary", pytrace=False)
    if rendered is None:
        pytest.fail("unreadable credential-path fixture was not rejected", pytrace=False)
    _assert_sensitive_absent(filename, rendered)
    if not rendered.startswith("unable to inspect Git-visible path identity: <redacted-path> [path_id="):
        pytest.fail("path identity error omitted its safe redacted identity", pytrace=False)


def test_git_ignored_paths_are_explicitly_excluded_from_scope(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    validator._init_scope_test_repo(repo)
    (repo / ".gitignore").write_text("ignored-cache/\n", encoding="utf-8")
    validator._git_text(repo, "add", ".gitignore")
    validator._git_text(repo, "commit", "-q", "-m", "declare ignored cache")
    baseline = validator.create_git_baseline(repo)
    ignored_directory = repo / "ignored-cache"
    ignored_directory.mkdir()
    (ignored_directory / "cache.bin").write_text("ignored\n", encoding="utf-8")

    issues = validator.check_git_scope(baseline, repo=repo, allow_paths=[])

    assert issues == []
    assert baseline["scope_contract"]["git_ignored"] == "excluded"


def test_cli_discloses_that_git_ignored_paths_are_not_checked(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert validator.main(["check", "--external", "offline"]) == 0

    captured = capsys.readouterr()
    assert "git_ignored_paths_checked=0" in captured.out
    assert captured.err == ""


def test_scope_self_tests_cover_dirty_index_paths_renames_and_clean_commits() -> None:
    results = validator.run_scope_self_tests()

    assert results == {
        "changed_preexisting_dirty_content": True,
        "index_only_change": True,
        "new_untracked_path": True,
        "removed_untracked_path": True,
        "rename": True,
        "unauthorized_path": True,
        "allowlisted_change": True,
        "preexisting_dirty_allowlisted_change": True,
        "explicitly_accepted_dirty_allowlisted_change": True,
        "untracked_mode_change": True,
        "unauthorized_clean_commit": True,
        "unauthorized_commit_then_revert": True,
        "credential_shaped_path_redaction": True,
        "git_ignored_path_excluded_by_contract": True,
    }
