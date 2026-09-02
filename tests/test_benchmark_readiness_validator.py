from __future__ import annotations

from pathlib import Path

import pytest

from scripts import validate_benchmark_readiness as validator


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
    assert value not in rendered
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
    assert secret not in str(caught.value)
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
    assert value not in rendered
    assert encoded not in rendered


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
    assert secret not in str(caught.value)


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
    }
