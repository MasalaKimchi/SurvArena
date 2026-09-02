from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
import hashlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import subprocess
import sys
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
READINESS_PATH = ROOT / "docs/benchmark_readiness.md"
INDEX_PATH = ROOT / "docs/index.md"
ROADMAP_PATH = ROOT / ".planning/ROADMAP.md"
REQUIREMENTS_PATH = ROOT / ".planning/REQUIREMENTS.md"

EXIT_SECRET = 2
EXIT_DOCUMENT = 3
EXIT_SCOPE = 4
EXIT_EXTERNAL = 5
EXIT_USAGE = 64
BASELINE_VERSION = 4
MAX_REDIRECTS = 5

WRONG_COLUMNS = [
    "ID",
    "Finding",
    "Evidence class",
    "Scientific / user consequence",
    "Repository evidence",
    "Authoritative criterion",
    "Confidence",
]
DOMAIN_COLUMNS = [
    "ID",
    "Domain",
    "Current assessment",
    "Evidence class",
    "Repository evidence",
    "Authoritative criterion",
    "Linked backlog IDs / rationale",
    "Assessment-change criterion",
]
REGISTER_COLUMNS = [
    "ID",
    "Finding",
    "Evidence class",
    "Severity",
    "Impact",
    "Confidence",
    "Owner",
    "Status",
    "Repository evidence",
    "Authoritative criterion",
    "Remediation",
    "Verification / exit criterion",
    "Dependencies",
    "Target gate",
    "Related phase / requirement",
    "Last verified",
]
CROSSWALK_COLUMNS = ["Research item", "Register IDs", "Coverage note"]
EXPECTED_RESEARCH_ITEMS = [
    *[f"P0-{letter}" for letter in "ABCDEFGHI"],
    *[f"P1-{letter}" for letter in "ABCDEF"],
    *[f"P2-{letter}" for letter in "ABC"],
    "VALIDATION",
    "SECURITY",
    "DECISIONS",
    "ASSUMPTIONS",
]
EXPECTED_REGISTER_IDS = [f"BR-{number:03d}" for number in range(1, 20)]
EVIDENCE_CLASSES = {"OBSERVED DEFECT", "MISSING EVIDENCE", "FUTURE SCOPE"}
OWNERS = {
    "Benchmark maintainer",
    "Scientific-methods maintainer",
    "Data steward",
    "Release steward",
    "Security maintainer",
}
STATUSES = {"open", "in_progress", "blocked", "verified", "accepted_risk", "deferred_scope", "superseded"}
TERMINAL_STATUSES = {"verified", "accepted_risk", "superseded"}
TARGET_GATES = {"Gate A", "Gate B", "Gate C", "Broadened claim"}
PARTIAL_RELEASE_ROWS = {"BR-001", "BR-002", "BR-005"}

BR_ID_PATTERN = re.compile(r"BR-[0-9]{3}")
DOM_ID_PATTERN = re.compile(r"DOM-(?:0[1-9]|1[0-3])")
WRONG_ID_PATTERN = re.compile(r"WRONG-0[1-9]")
BOUNDED_BR_PATTERN = re.compile(r"(?<![A-Za-z0-9_-])BR-[0-9]{3}(?![A-Za-z0-9_-])")
BOUNDED_DOM_PATTERN = re.compile(r"(?<![A-Za-z0-9_-])DOM-(?:0[1-9]|1[0-3])(?![A-Za-z0-9_-])")
BOUNDED_WRONG_PATTERN = re.compile(r"(?<![A-Za-z0-9_-])WRONG-0[1-9](?![A-Za-z0-9_-])")
VERIFIED_TAG_PATTERN = re.compile(
    r"\[VERIFIED: `(?P<path>[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*):"
    r"(?P<start>[1-9][0-9]*)-(?P<end>[1-9][0-9]*)`\]"
)
MARKDOWN_LINK_PATTERN = re.compile(r"\[(?P<label>[^\]\r\n]+)\]\((?P<url>https://[^()\s]+)\)")
MARKDOWN_DESTINATION_PATTERN = re.compile(r"(?<!\!)\[[^\]\r\n]+\]\((?P<url>[^)\s]+)\)")
REQUIREMENT_ID_PATTERN = re.compile(
    r"(?:BASE|STAT|METR|EXEC|DATA|MODL|CORE|STORE|SUIT|PROT|REPR|LIVE|TASK)-[0-9]{2}"
)
EXTERNAL_HOST_ALLOWLIST = {
    "docs.mlcommons.org",
    "docs.openml.org",
    "openml.github.io",
    "owasp.org",
    "pmc.ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "www.acm.org",
    "www.jmlr.org",
    "www.microsoft.com",
    "www.tripod-statement.org",
    "www.w3.org",
}

CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    (
        "aws_access_key_id_assignment",
        re.compile(r"(?i)\bAWS_ACCESS_KEY_ID\s*[:=]\s*[\"']?[^\s\"'`]{8,}"),
    ),
    (
        "aws_secret_access_key_assignment",
        re.compile(r"(?i)\bAWS_SECRET_ACCESS_KEY\s*[:=]\s*[\"']?[^\s\"'`]{8,}"),
    ),
    (
        "aws_session_token_assignment",
        re.compile(r"(?i)\bAWS_SESSION_TOKEN\s*[:=]\s*[\"']?[^\s\"'`]{8,}"),
    ),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("openai_token", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("hugging_face_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    (
        "credential_assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|secret|password|"
            r"token|credential|signature|x-amz-signature)"
            r"\s*[:=]\s*[\"']?[A-Za-z0-9._~+/=-]{8,}"
        ),
    ),
)


class ValidationError(RuntimeError):
    def __init__(self, message: str, *, exit_code: int = EXIT_DOCUMENT) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class SecretFinding:
    file: str
    line: int
    pattern: str


@dataclass(frozen=True)
class ExternalResult:
    source_url: str = field(repr=False)
    label: str
    state: str
    http_code: int
    effective_url: str = field(repr=False)


@dataclass(frozen=True)
class ScopeIssue:
    path: str
    code: str
    before: str
    after: str


@dataclass(frozen=True)
class RequirementInfo:
    status: str
    phase: int | None


def _path_digest(value: str) -> str:
    return hashlib.sha256(os.fsencode(value)).hexdigest()


def sanitize_path(value: str) -> str:
    """Return a digest-only path label; raw path text never crosses an output boundary."""
    if re.fullmatch(
        r"<redacted-path> \[path_id=[0-9a-f]{12}\](?: \[snapshot_key=[0-9a-f]{12}\])?",
        value,
    ):
        return value
    return f"<redacted-path> [path_id={_path_digest(value)[:12]}]"


PATH_ACCESS_ERRORS = (OSError, RuntimeError, ValueError)


def _safe_file_label(path: Path, repo: Path) -> str:
    try:
        resolved = path.resolve()
        relative = resolved.relative_to(repo.resolve()).as_posix()
    except PATH_ACCESS_ERRORS:
        relative = path.name
    return sanitize_path(relative)


def _safe_path_error(
    action: str,
    path: Path,
    *,
    repo: Path,
    exit_code: int = EXIT_DOCUMENT,
) -> ValidationError:
    """Build a path-bearing failure without retaining path or exception text."""
    return ValidationError(f"{action}: {_safe_file_label(path, repo)}", exit_code=exit_code)


def _run_git(
    repo: Path,
    args: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(["git", *args], cwd=repo, check=check, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        raise ValidationError("Git command failed; command details redacted", exit_code=EXIT_SCOPE) from None


def _repo_root(start: Path = ROOT) -> Path:
    result = _run_git(start, ["rev-parse", "--show-toplevel"])
    candidate = Path(os.fsdecode(result.stdout).strip())
    try:
        return candidate.resolve()
    except PATH_ACCESS_ERRORS:
        raise _safe_path_error(
            "unable to resolve repository root",
            candidate,
            repo=start,
            exit_code=EXIT_SCOPE,
        ) from None


def _canonical_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or "\\" in value:
        raise ValidationError(f"noncanonical repository path: {sanitize_path(value)}", exit_code=EXIT_SCOPE)
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValidationError(f"unsafe repository path: {sanitize_path(value)}", exit_code=EXIT_SCOPE)
    return value


def _decoded_candidates(line: str) -> tuple[str, ...]:
    candidates = [line]
    for _ in range(3):
        decoded = unquote(candidates[-1])
        if decoded == candidates[-1]:
            break
        candidates.append(decoded)
    return tuple(candidates)


def scan_credential_texts(documents: Iterable[tuple[str, str]]) -> list[SecretFinding]:
    findings: list[SecretFinding] = []
    for label, text in documents:
        safe_label = sanitize_path(label)
        for line_number, line in enumerate(text.splitlines(), start=1):
            candidates = _decoded_candidates(line)
            for pattern_name, pattern in CREDENTIAL_PATTERNS:
                if any(pattern.search(candidate) for candidate in candidates):
                    findings.append(SecretFinding(file=safe_label, line=line_number, pattern=pattern_name))
    return sorted(findings, key=lambda item: (item.file, item.line, item.pattern))


def scan_credentials(paths: Iterable[Path], *, repo: Path = ROOT) -> list[SecretFinding]:
    documents: list[tuple[str, str]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except PATH_ACCESS_ERRORS:
            raise _safe_path_error("unable to read credential-scan input", path, repo=repo) from None
        documents.append((_safe_file_label(path, repo), text))
    return scan_credential_texts(documents)


def format_secret_findings(findings: Sequence[SecretFinding]) -> str:
    metadata = [
        {"file": sanitize_path(finding.file), "line": finding.line, "pattern": finding.pattern}
        for finding in findings
    ]
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def _secret_gate(documents: Sequence[tuple[str, str]]) -> None:
    findings = scan_credential_texts(documents)
    if findings:
        raise ValidationError(
            "credential patterns detected; values redacted: " + format_secret_findings(findings),
            exit_code=EXIT_SECRET,
        )


def url_identity(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"url_id={digest}"


def mask_url(url: str) -> str:
    parsed = urlsplit(url)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{hostname}{port}{parsed.path}"


def _secret_gate_url(url: str, *, label: str) -> None:
    findings = scan_credential_texts([(label, url)])
    if findings:
        raise ValidationError(
            "credential patterns detected; values redacted: " + format_secret_findings(findings),
            exit_code=EXIT_SECRET,
        )


def _parse_table(text: str, heading: str, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    lines = text.splitlines()
    if lines.count(heading) != 1:
        raise ValidationError(f"expected one table heading: {heading}")
    index = lines.index(heading) + 1
    while index < len(lines) and not lines[index].startswith("|"):
        index += 1
    raw: list[str] = []
    while index < len(lines) and lines[index].startswith("|"):
        raw.append(lines[index])
        index += 1
    if len(raw) < 2:
        raise ValidationError(f"missing Markdown table: {heading}")

    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip()[1:-1].split("|")]

    header = cells(raw[0])
    if header != list(expected_columns):
        raise ValidationError(f"unexpected columns: {heading}")
    if not all(set(cell) <= {"-", ":"} for cell in cells(raw[1])):
        raise ValidationError(f"invalid table delimiter: {heading}")
    rows: list[dict[str, str]] = []
    for line in raw[2:]:
        values = cells(line)
        if len(values) != len(header):
            raise ValidationError(f"ragged Markdown table: {heading}")
        row = dict(zip(header, values, strict=True))
        if any(not row[column] for column in header):
            raise ValidationError(f"empty required table cell: {heading}")
        rows.append(row)
    return rows


def parse_reference_list(value: str, *, prefix: str, known: set[str]) -> list[str]:
    pattern = BR_ID_PATTERN if prefix == "BR" else DOM_ID_PATTERN
    separator_pattern = rf"{pattern.pattern}(?:, {pattern.pattern})*"
    if re.fullmatch(separator_pattern, value) is None:
        raise ValidationError(f"malformed {prefix} whole-cell reference")
    references = value.split(", ")
    if references != sorted(references) or len(references) != len(set(references)):
        raise ValidationError(f"duplicate or unsorted {prefix} reference list")
    unknown = sorted(set(references) - known)
    if unknown:
        raise ValidationError(f"unknown {prefix} reference")
    return references


def _validate_literal_tokens(text: str, *, prefix: str, pattern: re.Pattern[str]) -> None:
    occurrences = [match.start() for match in re.finditer(re.escape(prefix + "-"), text)]
    matches = [match.start() for match in pattern.finditer(text)]
    if occurrences != matches:
        raise ValidationError(f"malformed {prefix} token")


def _validate_link_expression(value: str) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    position = 0
    for match in MARKDOWN_LINK_PATTERN.finditer(value):
        if match.start() != position:
            if value[position : match.start()] != " and ":
                raise ValidationError("criterion link cell is not a bounded whole-cell expression")
        label = match.group("label").strip()
        url = match.group("url")
        if not label or label.startswith(("http://", "https://")):
            raise ValidationError("external criterion link lacks a descriptive label")
        _validate_static_external_url(url)
        links.append((label, url))
        position = match.end()
    if not links or position != len(value):
        raise ValidationError("criterion link cell is not a bounded whole-cell expression")
    return links


def _validate_static_external_url(url: str) -> None:
    _secret_gate_url(url, label="external-url")
    identity = url_identity(url)
    try:
        parsed = urlsplit(url)
        _ = parsed.port
        hostname = (parsed.hostname or "").encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError):
        raise ValidationError(f"invalid external URL: {identity}") from None
    if parsed.scheme != "https" or not hostname:
        raise ValidationError(f"external URL must use HTTPS with a hostname: {identity}")
    if hostname not in EXTERNAL_HOST_ALLOWLIST:
        raise ValidationError(f"external URL host is not approved: {identity}")
    if parsed.username is not None or parsed.password is not None:
        raise ValidationError(f"external URL userinfo is forbidden: {identity}")
    if "?" in url or "#" in url:
        raise ValidationError(f"external URL query or fragment is forbidden: {identity}")
    if "\\" in url or any(ord(character) < 32 for character in url):
        raise ValidationError(f"external URL contains unsafe characters: {identity}")
    decoded = url
    for _ in range(3):
        next_decoded = unquote(decoded)
        if next_decoded == decoded:
            break
        decoded = next_decoded
        decoded_parts = urlsplit(decoded)
        if (
            decoded_parts.username is not None
            or "?" in decoded
            or "#" in decoded
            or "\\" in decoded
            or any(ord(character) < 32 for character in decoded)
        ):
            raise ValidationError(f"encoded URL delimiters are forbidden: {identity}")


def _validate_no_insecure_markdown_links(text: str) -> None:
    for match in MARKDOWN_DESTINATION_PATTERN.finditer(text):
        destination = match.group("url")
        if destination.lower().startswith("http://"):
            raise ValidationError("insecure HTTP Markdown link is forbidden")


def _extract_external_links(text: str) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    for match in MARKDOWN_LINK_PATTERN.finditer(text):
        label = match.group("label").strip()
        url = match.group("url")
        _validate_static_external_url(url)
        links.append((label, url))
    return links


def _parse_requirements(text: str) -> dict[str, RequirementInfo]:
    checklist: dict[str, str] = {}
    for match in re.finditer(
        r"(?m)^- \[(?P<mark>[ xX])\] \*\*(?P<id>[A-Z]+-[0-9]{2})\*\*:",
        text,
    ):
        status = "Complete" if match.group("mark").lower() == "x" else "Pending"
        requirement_id = match.group("id")
        if requirement_id in checklist:
            raise ValidationError("duplicate requirement checklist ID")
        checklist[requirement_id] = status
    future: list[str] = []
    for match in re.finditer(r"(?m)^- \*\*(?P<id>(?:LIVE|TASK)-[0-9]{2})\*\*:", text):
        requirement_id = match.group("id")
        if requirement_id in checklist or requirement_id in future:
            raise ValidationError("duplicate future requirement ID")
        future.append(requirement_id)
    traceability: dict[str, tuple[int, str]] = {}
    for requirement_id, phase_number, status in re.findall(
        r"(?m)^\| ([A-Z]+-[0-9]{2}) \| Phase ([1-6]) \| (Complete|Pending) \|$",
        text,
    ):
        if requirement_id not in checklist:
            raise ValidationError("traceability references an unknown requirement")
        if requirement_id in traceability:
            raise ValidationError("duplicate traceability requirement ID")
        traceability[requirement_id] = (int(phase_number), status)
    if set(traceability) != set(checklist):
        raise ValidationError("requirement checklist and traceability sets differ")
    requirements: dict[str, RequirementInfo] = {}
    for requirement_id, checklist_status in checklist.items():
        phase, traceability_status = traceability[requirement_id]
        if traceability_status != checklist_status:
            raise ValidationError("requirement checklist and traceability statuses differ")
        requirements[requirement_id] = RequirementInfo(status=checklist_status, phase=phase)
    for requirement_id in future:
        requirements[requirement_id] = RequirementInfo(status="Future", phase=None)
    return requirements


def _validate_roadmap_requirements(
    roadmap_text: str,
    requirement_info: Mapping[str, RequirementInfo],
) -> None:
    phase_names = {
        1: "Truthful Green Baseline",
        2: "Scientific Comparison Kernel",
        3: "Execution and Data Safety",
        4: "Authoritative Protocol and Results",
        5: "Representative Protocol Pilot",
        6: "Release Reproduction",
    }
    for phase, name in phase_names.items():
        heading = f"### Phase {phase}: {name}"
        if roadmap_text.count(heading) != 1:
            raise ValidationError("roadmap phase heading set has drifted")
        section = roadmap_text.split(heading, 1)[1]
        section = section.split("\n### Phase ", 1)[0]
        matches = re.findall(r"(?m)^\*\*Requirements:\*\* (.+)$", section)
        if len(matches) != 1:
            raise ValidationError("roadmap phase requirements declaration is missing or duplicated")
        declared = matches[0].split(", ")
        if any(REQUIREMENT_ID_PATTERN.fullmatch(item) is None for item in declared):
            raise ValidationError("roadmap requirement list is not canonical")
        expected = [
            requirement_id
            for requirement_id, info in requirement_info.items()
            if info.phase == phase
        ]
        if declared != expected:
            raise ValidationError("roadmap and requirements traceability differ")


def _parse_related_requirements(
    value: str,
    requirement_info: Mapping[str, RequirementInfo],
) -> tuple[set[int], list[str]]:
    if " / " not in value:
        raise ValidationError("related phase/requirement cell lacks bounded separator")
    phase_text, requirement_text = value.split(" / ", 1)
    if phase_text == "Future":
        phases: set[int] = set()
    elif re.fullmatch(r"Phase [1-6]", phase_text):
        phases = {int(phase_text[-1])}
    elif re.fullmatch(r"Phases [1-6](?:, [1-6])*(?:,? and [1-6])", phase_text):
        phase_values = [int(item) for item in re.findall(r"[1-6]", phase_text)]
        if phase_values != sorted(set(phase_values)):
            raise ValidationError("related phase list is not sorted and unique")
        phases = set(phase_values)
    else:
        raise ValidationError("related phase cell is not canonical")
    if not phases and phase_text != "Future":
        raise ValidationError("related phase cell has no phase")
    requirement_pattern = rf"{REQUIREMENT_ID_PATTERN.pattern}(?:, {REQUIREMENT_ID_PATTERN.pattern})*"
    if re.fullmatch(requirement_pattern, requirement_text) is None:
        raise ValidationError("related requirement list is not a bounded whole-cell list")
    requirement_ids = requirement_text.split(", ")
    if len(requirement_ids) != len(set(requirement_ids)):
        raise ValidationError("related requirement list contains duplicates")
    unknown_requirements = [item for item in requirement_ids if item not in requirement_info]
    if unknown_requirements:
        raise ValidationError("register references an unknown roadmap requirement")
    requirement_order = {requirement_id: index for index, requirement_id in enumerate(requirement_info)}
    if requirement_ids != sorted(requirement_ids, key=requirement_order.__getitem__):
        raise ValidationError("related requirement list is not in canonical project order")
    for requirement_id in requirement_ids:
        info = requirement_info[requirement_id]
        if info.phase is None:
            if phase_text != "Future":
                raise ValidationError("future requirement is mapped to a numbered phase")
        elif info.phase not in phases:
            raise ValidationError("requirement is mapped to the wrong roadmap phase")
    return phases, requirement_ids


def _validate_dependency_graph(rows: Sequence[Mapping[str, str]], ids: set[str]) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = {}
    by_id = {row["ID"]: row for row in rows}
    for row in rows:
        value = row["Dependencies"]
        dependencies = [] if value == "None" else parse_reference_list(value, prefix="BR", known=ids)
        if row["ID"] in dependencies:
            raise ValidationError("register row has a self-dependency")
        graph[row["ID"]] = dependencies

    visiting: list[str] = []
    states: dict[str, int] = {}

    def visit(node: str) -> None:
        if states.get(node) == 1:
            raise ValidationError("register dependency cycle detected")
        if states.get(node) == 2:
            return
        states[node] = 1
        visiting.append(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.pop()
        states[node] = 2

    for row_id in sorted(by_id):
        visit(row_id)
    return graph


def _validate_status_semantics(
    rows: Sequence[Mapping[str, str]],
    graph: Mapping[str, Sequence[str]],
    requirement_info: Mapping[str, RequirementInfo],
) -> None:
    by_id = {row["ID"]: row for row in rows}
    replacement_graph: dict[str, list[str]] = {}
    related_phases: dict[str, set[int]] = {}
    unresolved_markers = re.compile(
        r"(?i)\b(?:release closure|additionally requires|requires|pending|remaining|remains|not yet|"
        r"until|awaiting|incomplete|regenerat(?:e|ed|ion))\b"
    )
    for row in rows:
        row_id = row["ID"]
        status = row["Status"]
        phases, requirement_ids = _parse_related_requirements(
            row["Related phase / requirement"],
            requirement_info,
        )
        related_phases[row_id] = phases
        if (
            row_id in PARTIAL_RELEASE_ROWS
            and any(requirement_info[item].status != "Complete" for item in requirement_ids)
            and status != "in_progress"
        ):
            raise ValidationError("partial kernel/release row must remain in_progress")
        if status == "verified":
            criterion = row["Verification / exit criterion"]
            if "Observed closure:" not in criterion or unresolved_markers.search(criterion):
                raise ValidationError("verified row has an unresolved or non-atomic exit criterion")
            if any(by_id[dependency]["Status"] not in TERMINAL_STATUSES for dependency in graph[row_id]):
                raise ValidationError("verified row depends on an unsatisfied row")
            if any(requirement_info[item].status != "Complete" for item in requirement_ids):
                raise ValidationError("verified row maps to an incomplete requirement")
        if status == "accepted_risk":
            if re.fullmatch(
                r"Risk accepted by (?:Benchmark maintainer|Scientific-methods maintainer|Data steward|"
                r"Release steward|Security maintainer) on [0-9]{4}-[0-9]{2}-[0-9]{2}: .+",
                row["Verification / exit criterion"],
            ) is None:
                raise ValidationError("accepted-risk row lacks named, dated acceptance evidence")
        if status == "blocked" and not any(
            by_id[dependency]["Status"] not in TERMINAL_STATUSES for dependency in graph[row_id]
        ):
            raise ValidationError("blocked row lacks an unsatisfied dependency")
        if status == "superseded":
            criterion = row["Verification / exit criterion"]
            marker = "Replaced by "
            if marker not in criterion:
                raise ValidationError("superseded row lacks replacement IDs")
            replacements = parse_reference_list(
                criterion.split(marker, 1)[1],
                prefix="BR",
                known=set(by_id),
            )
            if row_id in replacements:
                raise ValidationError("superseded row replaces itself")
            replacement_graph[row_id] = replacements
        if row["Severity"] == "P2" or status == "deferred_scope" or row["Target gate"] == "Broadened claim":
            expected_impact = "Does not block Gate A or Gate B for the current scoped package."
            if (
                status != "deferred_scope"
                or row["Evidence class"] != "FUTURE SCOPE"
                or row["Severity"] != "P2"
                or row["Target gate"] not in {"Gate C", "Broadened claim"}
                or expected_impact not in row["Impact"]
                or phases
            ):
                raise ValidationError("deferred-scope row violates future-scope semantics")

    replacement_state: dict[str, int] = {}

    def visit_replacement(node: str) -> None:
        if replacement_state.get(node) == 1:
            raise ValidationError("superseded replacement cycle detected")
        if replacement_state.get(node) == 2:
            return
        replacement_state[node] = 1
        for replacement in replacement_graph.get(node, []):
            visit_replacement(replacement)
        replacement_state[node] = 2

    for row_id in replacement_graph:
        visit_replacement(row_id)

    gate_order = {"Gate A": 1, "Gate B": 2, "Gate C": 3, "Broadened claim": 3}
    for row in rows:
        row_id = row["ID"]
        for dependency_id in graph[row_id]:
            dependency = by_id[dependency_id]
            if gate_order[dependency["Target gate"]] > gate_order[row["Target gate"]]:
                raise ValidationError(f"dependency points to a later target gate: {row_id}->{dependency_id}")
            dependency_phases = related_phases[dependency_id]
            if not related_phases[row_id] and dependency_phases:
                continue
            if related_phases[row_id] and not dependency_phases:
                raise ValidationError(f"current-scope row depends on future-scope work: {row_id}->{dependency_id}")
            if dependency_phases and min(dependency_phases) > max(related_phases[row_id]):
                raise ValidationError(f"dependency points to a later roadmap phase: {row_id}->{dependency_id}")


def _validate_provenance_cell(value: str) -> None:
    tags = list(VERIFIED_TAG_PATTERN.finditer(value))
    if not tags:
        raise ValidationError("repository-evidence cell lacks a canonical VERIFIED tag")
    position = 0
    for tag in tags:
        if tag.start() != position:
            if value[position : tag.start()] != " ":
                raise ValidationError("repository-evidence cell is not a bounded VERIFIED-tag expression")
        position = tag.end()
    if position != len(value):
        raise ValidationError("repository-evidence cell is not a bounded VERIFIED-tag expression")


def _validate_iso_date(value: str) -> None:
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValidationError("Last verified is not a valid ISO date") from None
    if parsed > date.today():
        raise ValidationError("Last verified is in the future")


def validate_structure(
    readiness_text: str,
    index_text: str,
    roadmap_text: str,
    requirements_text: str,
) -> dict[str, int]:
    wrong_rows = _parse_table(readiness_text, "## Where SurvArena Is Wrong Today", WRONG_COLUMNS)
    domain_rows = _parse_table(readiness_text, "## Audit-Domain Coverage", DOMAIN_COLUMNS)
    register_rows = _parse_table(readiness_text, "## Prioritized Improvement Register", REGISTER_COLUMNS)
    crosswalk_rows = _parse_table(readiness_text, "## Research Coverage Crosswalk", CROSSWALK_COLUMNS)
    index_rows = _parse_table(index_text, "## Reference Docs", ["Topic", "Read"])

    wrong_ids = [row["ID"] for row in wrong_rows]
    domain_ids = [row["ID"] for row in domain_rows]
    register_ids = [row["ID"] for row in register_rows]
    if wrong_ids != [f"WRONG-{number:02d}" for number in range(1, 10)]:
        raise ValidationError("WRONG row set/order mismatch")
    if domain_ids != [f"DOM-{number:02d}" for number in range(1, 14)]:
        raise ValidationError("DOM row set/order mismatch")
    if register_ids != EXPECTED_REGISTER_IDS:
        raise ValidationError("BR row set/order mismatch")
    if [row["Research item"] for row in crosswalk_rows] != EXPECTED_RESEARCH_ITEMS:
        raise ValidationError("research crosswalk set/order mismatch")

    _validate_literal_tokens(readiness_text, prefix="WRONG", pattern=BOUNDED_WRONG_PATTERN)
    _validate_literal_tokens(readiness_text, prefix="DOM", pattern=BOUNDED_DOM_PATTERN)
    _validate_literal_tokens(readiness_text, prefix="BR", pattern=BOUNDED_BR_PATTERN)
    if set(BOUNDED_WRONG_PATTERN.findall(readiness_text)) - set(wrong_ids):
        raise ValidationError("unknown WRONG token")
    if set(BOUNDED_DOM_PATTERN.findall(readiness_text)) - set(domain_ids):
        raise ValidationError("unknown DOM token")
    if set(BOUNDED_BR_PATTERN.findall(readiness_text)) - set(register_ids):
        raise ValidationError("unknown BR token")

    expected_wrong_classes = {
        **{f"WRONG-{number:02d}": "OBSERVED DEFECT" for number in range(1, 9)},
        "WRONG-09": "MISSING EVIDENCE",
    }
    for row in wrong_rows:
        if row["Evidence class"] != expected_wrong_classes[row["ID"]]:
            raise ValidationError("WRONG evidence class mismatch")
        criterion = row["Authoritative criterion"]
        if MARKDOWN_LINK_PATTERN.search(criterion):
            _validate_link_expression(criterion)
        else:
            parse_reference_list(criterion, prefix="DOM", known=set(domain_ids))
        _validate_provenance_cell(row["Repository evidence"])

    for row in domain_rows:
        if row["Evidence class"] not in EVIDENCE_CLASSES:
            raise ValidationError("invalid domain evidence class")
        _validate_link_expression(row["Authoritative criterion"])
        linkage = row["Linked backlog IDs / rationale"]
        if linkage.startswith("No current gap — "):
            if len(linkage.removeprefix("No current gap — ").strip()) < 12 or "BR-" in linkage:
                raise ValidationError("domain no-gap rationale is not substantive")
        else:
            parse_reference_list(linkage, prefix="BR", known=set(register_ids))
        _validate_provenance_cell(row["Repository evidence"])

    for row in register_rows:
        if row["Evidence class"] not in EVIDENCE_CLASSES:
            raise ValidationError("invalid register evidence class")
        if row["Severity"] not in {"P0", "P1", "P2"}:
            raise ValidationError("invalid register severity")
        if row["Confidence"] not in {"HIGH", "MEDIUM", "LOW"}:
            raise ValidationError("invalid register confidence")
        if row["Owner"] not in OWNERS or row["Status"] not in STATUSES or row["Target gate"] not in TARGET_GATES:
            raise ValidationError("invalid register owner, status, or target gate")
        parse_reference_list(row["Authoritative criterion"], prefix="DOM", known=set(domain_ids))
        _validate_provenance_cell(row["Repository evidence"])
        _validate_iso_date(row["Last verified"])

    graph = _validate_dependency_graph(register_rows, set(register_ids))
    requirement_info = _parse_requirements(requirements_text)
    _validate_roadmap_requirements(roadmap_text, requirement_info)
    _validate_status_semantics(register_rows, graph, requirement_info)

    reached: set[str] = set()
    for row in crosswalk_rows:
        reached.update(parse_reference_list(row["Register IDs"], prefix="BR", known=set(register_ids)))
    if set(register_ids) - reached:
        raise ValidationError("research crosswalk leaves orphan register rows")

    routes = [
        row
        for row in index_rows
        if re.fullmatch(r"\[[^]]+\]\(benchmark_readiness\.md\)", row["Read"])
    ]
    if len(routes) != 1 or index_text.count("benchmark_readiness.md") != 1:
        raise ValidationError("docs index must contain exactly one readiness Markdown link")
    if "**Immediate implementation priority:**" not in readiness_text or "Phase 3" not in readiness_text:
        raise ValidationError("readiness document lacks the Phase 3 immediate priority")
    if "**Highest subsequent release-provenance priority:**" not in readiness_text:
        raise ValidationError("readiness document lacks the qualified BR-009 priority")
    br009 = next(row for row in register_rows if row["ID"] == "BR-009")
    if br009["Related phase / requirement"].split(" / ", 1)[0] != "Phase 4":
        raise ValidationError("BR-009 must map to Phase 4")
    if graph["BR-009"] != ["BR-004", "BR-007", "BR-008", "BR-010"]:
        raise ValidationError("BR-009 prerequisites are incomplete")
    return {
        "wrong_rows": len(wrong_rows),
        "domain_rows": len(domain_rows),
        "register_rows": len(register_rows),
        "crosswalk_rows": len(crosswalk_rows),
    }


def validate_provenance_text(text: str, *, repo: Path) -> int:
    occurrences = [match.start() for match in re.finditer(r"\[VERIFIED:", text)]
    tags = list(VERIFIED_TAG_PATTERN.finditer(text))
    if occurrences != [tag.start() for tag in tags]:
        raise ValidationError("malformed VERIFIED provenance tag")
    for tag in tags:
        relative = _canonical_relative_path(tag.group("path"))
        candidate = repo / relative
        try:
            target = candidate.resolve()
            repo_root = repo.resolve()
            target_exists = target.is_file()
        except PATH_ACCESS_ERRORS:
            raise _safe_path_error("unable to inspect VERIFIED target", candidate, repo=repo) from None
        if not target.is_relative_to(repo_root) or not target_exists:
            raise ValidationError(
                f"VERIFIED target is missing or escapes the repository: {sanitize_path(relative)}"
            )
        tracked = _run_git(repo, ["ls-files", "--error-unmatch", "--", relative], check=False)
        if tracked.returncode != 0:
            raise ValidationError(f"VERIFIED target is untracked: {sanitize_path(relative)}")
        try:
            line_count = len(target.read_text(encoding="utf-8").splitlines())
        except PATH_ACCESS_ERRORS:
            raise _safe_path_error("unable to read VERIFIED target", candidate, repo=repo) from None
        start = int(tag.group("start"))
        end = int(tag.group("end"))
        if not 1 <= start <= end <= line_count:
            raise ValidationError(f"VERIFIED line range is invalid: {sanitize_path(relative)}")
    return len(tags)


def validate_provenance(readiness_path: Path, *, repo: Path) -> int:
    try:
        text = readiness_path.read_text(encoding="utf-8")
    except PATH_ACCESS_ERRORS:
        raise _safe_path_error("unable to read provenance input", readiness_path, repo=repo) from None
    return validate_provenance_text(text, repo=repo)


def _heading_slugs(path: Path, *, repo: Path = ROOT) -> set[str]:
    counts: dict[str, int] = {}
    slugs: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except PATH_ACCESS_ERRORS:
        raise _safe_path_error("unable to read local-link target", path, repo=repo) from None
    for line in lines:
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line)
        if not match:
            continue
        raw = re.sub(r"<[^>]+>", "", match.group(1))
        raw = re.sub(r"[`*_~]", "", raw).lower()
        base = re.sub(r"[^\w\- ]", "", raw)
        base = re.sub(r"\s+", "-", base.strip())
        count = counts.get(base, 0)
        counts[base] = count + 1
        slugs.add(base if count == 0 else f"{base}-{count}")
    return slugs


def validate_local_link_texts(documents: Sequence[tuple[Path, str]], *, repo: Path) -> int:
    count = 0
    for source, source_text in documents:
        for target in re.findall(r"(?<!\!)\[[^]]+\]\(([^)]+)\)", source_text):
            parsed = urlsplit(target)
            if parsed.scheme in {"http", "https"}:
                continue
            if parsed.scheme or parsed.netloc:
                raise ValidationError("unsupported local-link scheme")
            candidate = source if not parsed.path else source.parent / unquote(parsed.path)
            try:
                destination = candidate.resolve()
                repo_root = repo.resolve()
                destination_exists = destination.is_file()
            except PATH_ACCESS_ERRORS:
                raise _safe_path_error("unable to inspect local-link target", candidate, repo=repo) from None
            if not destination.is_relative_to(repo_root) or not destination_exists:
                raise ValidationError("local link is missing or escapes the repository")
            if parsed.fragment and unquote(parsed.fragment) not in _heading_slugs(destination, repo=repo):
                raise ValidationError("local link fragment is missing")
            count += 1
    return count


def validate_local_links(paths: Sequence[Path], *, repo: Path) -> int:
    documents: list[tuple[Path, str]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except PATH_ACCESS_ERRORS:
            raise _safe_path_error("unable to read local-link source", path, repo=repo) from None
        documents.append((path, text))
    return validate_local_link_texts(documents, repo=repo)


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _require_public_resolution(url: str) -> None:
    parsed = urlsplit(url)
    hostname = parsed.hostname
    if hostname is None:
        raise ValidationError(f"external destination lacks a hostname: {url_identity(url)}", exit_code=EXIT_EXTERNAL)
    try:
        addresses = {
            ipaddress.ip_address(result[4][0])
            for result in socket.getaddrinfo(hostname, parsed.port or 443, type=socket.SOCK_STREAM)
        }
    except (OSError, ValueError):
        raise URLError("resolution failed") from None
    if not addresses or any(not address.is_global for address in addresses):
        raise ValidationError(
            f"external destination resolved outside the public Internet: {url_identity(url)}",
            exit_code=EXIT_EXTERNAL,
        )


def _fetch_external(url: str, *, timeout: float = 12.0) -> tuple[int, str]:
    opener = build_opener(_NoRedirectHandler())
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        _validate_static_external_url(current)
        try:
            _require_public_resolution(current)
            request = Request(current, headers={"User-Agent": "SurvArena-readiness-validator/1"})
            try:
                with opener.open(request, timeout=timeout) as response:  # noqa: S310 - allowlisted HTTPS only.
                    code = int(response.status)
                    location = response.headers.get("Location")
            except HTTPError as error:
                code = int(error.code)
                location = error.headers.get("Location")
        except (TimeoutError, URLError, OSError):
            return 0, current
        if code not in range(300, 400):
            return code, current
        if not location:
            raise ValidationError(
                f"external redirect lacks a destination: {url_identity(current)}",
                exit_code=EXIT_EXTERNAL,
            )
        _secret_gate_url(location, label="external-redirect")
        current = urljoin(current, location)
    raise ValidationError(
        f"external redirect limit exceeded: {url_identity(url)}",
        exit_code=EXIT_EXTERNAL,
    )


def check_external_links(
    readiness_text: str,
    *,
    fetch: Callable[[str], tuple[int, str]] = _fetch_external,
) -> list[ExternalResult]:
    links = _extract_external_links(readiness_text)
    unique: dict[str, str] = {}
    for label, url in links:
        unique.setdefault(url, label)
    results: list[ExternalResult] = []
    for url, label in sorted(unique.items(), key=lambda item: item[1].lower()):
        try:
            code, effective = fetch(url)
        except ValidationError:
            raise
        except Exception:
            raise ValidationError(
                f"external transport failed without exposing transport data: {url_identity(url)}",
                exit_code=EXIT_EXTERNAL,
            ) from None
        _secret_gate_url(effective, label="external-effective-url")
        _validate_static_external_url(effective)
        if code in range(200, 400):
            state = "ok"
        elif code == 0 or code in {403, 406, 418, 429, 451} or code in range(500, 600):
            state = "manual_required"
        else:
            state = "invalid"
        results.append(
            ExternalResult(
                source_url=url,
                label=label,
                state=state,
                http_code=code,
                effective_url=effective,
            )
        )
    invalid = [result for result in results if result.state == "invalid"]
    if invalid:
        identities = ", ".join(url_identity(result.source_url) for result in invalid)
        raise ValidationError(f"invalid external source: {identities}", exit_code=EXIT_EXTERNAL)
    _validate_manual_external_rows(readiness_text, results)
    return results


def _validate_manual_external_rows(readiness_text: str, results: Sequence[ExternalResult]) -> None:
    rows = _parse_table(
        readiness_text,
        "## Manual External-Source Checks",
        ["URL", "Automation result", "Final destination", "Source identity", "Checked on"],
    )
    records: dict[str, tuple[Mapping[str, str], str]] = {}
    for row in rows:
        links = _validate_link_expression(row["URL"])
        if len(links) != 1 or links[0][1] in records:
            raise ValidationError("manual-source URL must be one unique Markdown link")
        destinations = _validate_link_expression(row["Final destination"])
        if len(destinations) != 1:
            raise ValidationError("manual-source final destination must be one Markdown link")
        try:
            checked_on = date.fromisoformat(row["Checked on"])
        except ValueError:
            raise ValidationError("manual-source review date is invalid") from None
        if checked_on > date.today():
            raise ValidationError("manual-source review date is in the future")
        records[links[0][1]] = (row, destinations[0][1])
    manual_results = {result.source_url: result for result in results if result.state == "manual_required"}
    if set(records) != set(manual_results):
        raise ValidationError("manual-source rows do not exactly match current manual-required results")
    for result in results:
        if result.state != "manual_required":
            continue
        row, final_destination = records[result.source_url]
        expected_result = (
            f"manual_required — HTTP {result.http_code}"
            if result.http_code
            else "manual_required — network failure"
        )
        if (
            row["Automation result"] != expected_result
            or final_destination != result.effective_url
            or not row["Source identity"].startswith("confirmed — ")
        ):
            raise ValidationError(
                f"manual-source review row is incomplete: {url_identity(result.source_url)}",
                exit_code=EXIT_EXTERNAL,
            )


def _status_paths(repo: Path) -> dict[str, str]:
    raw = _run_git(repo, ["status", "--porcelain=v1", "-z", "--untracked-files=all"]).stdout
    fields = raw.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    statuses: dict[str, str] = {}
    index = 0
    while index < len(fields):
        record = fields[index]
        index += 1
        if len(record) < 4 or record[2:3] != b" ":
            raise ValidationError("malformed Git status output", exit_code=EXIT_SCOPE)
        status = record[:2].decode("ascii")
        destination = os.fsdecode(record[3:])
        statuses[destination] = status
        if "R" in status or "C" in status:
            if index >= len(fields):
                raise ValidationError("incomplete Git rename status", exit_code=EXIT_SCOPE)
            source = os.fsdecode(fields[index])
            index += 1
            statuses[source] = status
    return statuses


def _worktree_identity(path: Path) -> tuple[str, str | None]:
    try:
        path.lstat()
        if path.is_symlink():
            return "symlink", hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()
        if path.is_file():
            mode = stat.S_IMODE(path.stat().st_mode)
            return f"file:{mode:04o}", hashlib.sha256(path.read_bytes()).hexdigest()
        if path.is_dir():
            mode = stat.S_IMODE(path.stat().st_mode)
            digest = hashlib.sha256()
            for child in sorted(path.rglob("*"), key=lambda item: os.fsencode(str(item.relative_to(path)))):
                relative = os.fsencode(str(child.relative_to(path)))
                kind, child_hash = _worktree_identity(child)
                digest.update(relative)
                digest.update(kind.encode("ascii"))
                if child_hash is not None:
                    digest.update(bytes.fromhex(child_hash))
            return f"directory:{mode:04o}", digest.hexdigest()
        return "special", hashlib.sha256(str(path.lstat().st_mode).encode("ascii")).hexdigest()
    except FileNotFoundError:
        return "missing", None
    except PATH_ACCESS_ERRORS:
        raise _safe_path_error(
            "unable to inspect Git-visible path identity",
            path,
            repo=path.parent,
            exit_code=EXIT_SCOPE,
        ) from None


def _index_entries(repo: Path, relative: str) -> list[dict[str, str]]:
    try:
        raw = _run_git(repo, ["ls-files", "--stage", "-z", "--", relative]).stdout
    except ValidationError:
        raise _safe_path_error(
            "unable to inspect Git index identity",
            repo / relative,
            repo=repo,
            exit_code=EXIT_SCOPE,
        ) from None
    entries: list[dict[str, str]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        if not separator:
            raise ValidationError("malformed Git index entry", exit_code=EXIT_SCOPE)
        try:
            returned_path = os.fsdecode(raw_path)
            mode, object_id, stage = metadata.decode("ascii").split()
        except PATH_ACCESS_ERRORS:
            raise _safe_path_error(
                "unable to decode Git index identity",
                repo / relative,
                repo=repo,
                exit_code=EXIT_SCOPE,
            ) from None
        if _path_digest(returned_path) != _path_digest(relative):
            raise ValidationError("Git index returned an unexpected path", exit_code=EXIT_SCOPE)
        entries.append({"mode": mode, "object_id": object_id, "stage": stage})
    return entries


def _dirty_snapshot(repo: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative, status in sorted(_status_paths(repo).items(), key=lambda item: os.fsencode(item[0])):
        kind, sha256 = _worktree_identity(repo / relative)
        path_key = _path_digest(relative)
        if path_key in result:
            raise ValidationError("Git path-identity collision", exit_code=EXIT_SCOPE)
        result[path_key] = {
            "path_label": sanitize_path(relative),
            "status": status,
            "worktree": {"kind": kind, "sha256": sha256},
            "index": _index_entries(repo, relative),
        }
    return result


def _repository_fingerprint(repo: Path) -> str:
    roots = _run_git(repo, ["rev-list", "--max-parents=0", "HEAD"]).stdout.splitlines()
    return hashlib.sha256(b"\0".join(sorted(roots))).hexdigest()


def create_git_baseline(repo: Path) -> dict[str, Any]:
    head = os.fsdecode(_run_git(repo, ["rev-parse", "HEAD"]).stdout).strip()
    tree = os.fsdecode(_run_git(repo, ["rev-parse", "HEAD^{tree}"]).stdout).strip()
    return {
        "version": BASELINE_VERSION,
        "repository_fingerprint": _repository_fingerprint(repo),
        "baseline_commit": head,
        "baseline_tree": tree,
        "scope_contract": {
            "tracked_commit_tree": "included",
            "staged": "included",
            "unstaged": "included",
            "git_untracked": "included",
            "git_ignored": "excluded",
        },
        "dirty_entries": _dirty_snapshot(repo),
    }


def write_git_baseline(output: Path, *, repo: Path) -> str:
    try:
        resolved_output = output.resolve()
        repo_root = repo.resolve()
    except PATH_ACCESS_ERRORS:
        raise _safe_path_error(
            "unable to resolve Git baseline output",
            output,
            repo=repo,
            exit_code=EXIT_SCOPE,
        ) from None
    if resolved_output == repo_root or resolved_output.is_relative_to(repo_root):
        raise ValidationError("Git baseline output must be outside the repository", exit_code=EXIT_SCOPE)
    payload = create_git_baseline(repo)
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(serialized)
    except PATH_ACCESS_ERRORS:
        raise _safe_path_error(
            "unable to write Git baseline output",
            output,
            repo=repo,
            exit_code=EXIT_SCOPE,
        ) from None
    return hashlib.sha256(serialized).hexdigest()


def _state_label(entry: Mapping[str, Any] | None) -> str:
    if entry is None:
        return "absent"
    worktree = entry.get("worktree", {})
    raw_status = entry.get("status", "??")
    status = raw_status if isinstance(raw_status, str) and re.fullmatch(r"[ MADRCU?!]{2}", raw_status) else "invalid"
    raw_kind = worktree.get("kind", "unknown") if isinstance(worktree, Mapping) else "unknown"
    kind = (
        raw_kind
        if isinstance(raw_kind, str)
        and re.fullmatch(r"(?:missing|symlink|special|file:[0-7]{4}|directory:[0-7]{4})", raw_kind)
        else "invalid"
    )
    index_label = "present" if entry.get("index") else "absent"
    return f"status={status!r},worktree={kind},index={index_label}"


def _snapshot_path_label(
    path_key: str,
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> str:
    entry = after or before or {}
    candidate = str(entry.get("path_label", "<unavailable-path>"))
    label = sanitize_path(candidate)
    return f"{label} [snapshot_key={path_key[:12]}]"


def _parse_name_status(raw: bytes) -> list[tuple[str, list[str]]]:
    fields = raw.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    changes: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(fields):
        status = fields[index].decode("ascii")
        index += 1
        count = 2 if status.startswith(("R", "C")) else 1
        if index + count > len(fields):
            raise ValidationError("malformed Git history diff", exit_code=EXIT_SCOPE)
        paths = [os.fsdecode(item) for item in fields[index : index + count]]
        index += count
        changes.append((status, paths))
    return changes


def check_git_scope(
    baseline: Mapping[str, Any],
    *,
    repo: Path,
    allow_paths: Sequence[str],
    accept_dirty_allow_paths: Sequence[str] = (),
) -> list[ScopeIssue]:
    allowlist = {_canonical_relative_path(path) for path in allow_paths}
    accepted_dirty = {_canonical_relative_path(path) for path in accept_dirty_allow_paths}
    if not accepted_dirty <= allowlist:
        raise ValidationError("accepted dirty paths must also be allowlisted", exit_code=EXIT_SCOPE)
    if baseline.get("version") != BASELINE_VERSION:
        raise ValidationError("incompatible Git baseline version", exit_code=EXIT_SCOPE)
    expected_contract = {
        "tracked_commit_tree": "included",
        "staged": "included",
        "unstaged": "included",
        "git_untracked": "included",
        "git_ignored": "excluded",
    }
    if baseline.get("scope_contract") != expected_contract:
        raise ValidationError("Git baseline scope contract is missing or incompatible", exit_code=EXIT_SCOPE)
    if baseline.get("repository_fingerprint") != _repository_fingerprint(repo):
        raise ValidationError("Git baseline belongs to another repository", exit_code=EXIT_SCOPE)
    commit = str(baseline.get("baseline_commit", ""))
    tree = str(baseline.get("baseline_tree", ""))
    commit_check = _run_git(repo, ["cat-file", "-e", f"{commit}^{{commit}}"], check=False)
    if commit_check.returncode != 0:
        raise ValidationError("Git baseline commit is unavailable", exit_code=EXIT_SCOPE)
    actual_tree = os.fsdecode(_run_git(repo, ["rev-parse", f"{commit}^{{tree}}"] ).stdout).strip()
    if actual_tree != tree:
        raise ValidationError("Git baseline tree does not match its commit", exit_code=EXIT_SCOPE)
    ancestor = _run_git(repo, ["merge-base", "--is-ancestor", commit, "HEAD"], check=False)
    if ancestor.returncode != 0:
        raise ValidationError("Git baseline is not an ancestor of HEAD", exit_code=EXIT_SCOPE)

    issues: list[ScopeIssue] = []
    commits = _run_git(repo, ["rev-list", "--reverse", f"{commit}..HEAD"]).stdout.splitlines()
    for raw_commit in commits:
        current = os.fsdecode(raw_commit)
        parent_fields = _run_git(repo, ["rev-list", "--parents", "-n", "1", current]).stdout.split()
        parents = [os.fsdecode(parent) for parent in parent_fields[1:]]
        if len(parents) != 1:
            issues.append(
                ScopeIssue(path="<history>", code="unapproved_merge", before="single-parent", after=current[:12])
            )
            continue
        raw_changes = _run_git(
            repo,
            ["diff", "--name-status", "-z", "--find-renames", parents[0], current],
        ).stdout
        for status, paths in _parse_name_status(raw_changes):
            for path in paths:
                canonical = _canonical_relative_path(path)
                if canonical not in allowlist:
                    issues.append(
                        ScopeIssue(
                            path=sanitize_path(canonical),
                            code="committed_path",
                            before=f"baseline={commit[:12]}",
                            after=f"commit={current[:12]},status={status}",
                        )
                    )

    before = baseline.get("dirty_entries")
    if not isinstance(before, dict):
        raise ValidationError("Git baseline lacks dirty identities", exit_code=EXIT_SCOPE)
    if any(
        re.fullmatch(r"[0-9a-f]{64}", key) is None or not isinstance(entry, dict)
        for key, entry in before.items()
    ):
        raise ValidationError("Git baseline contains malformed path identities", exit_code=EXIT_SCOPE)
    after = _dirty_snapshot(repo)
    allow_keys = {_path_digest(path) for path in allowlist}
    accepted_dirty_keys = {_path_digest(path) for path in accepted_dirty}
    for path_key in sorted(set(before) | set(after)):
        old = before.get(path_key)
        new = after.get(path_key)
        path_label = _snapshot_path_label(path_key, old, new)
        if path_key in allow_keys:
            if old is not None and old != new and path_key not in accepted_dirty_keys:
                issues.append(
                    ScopeIssue(
                        path=path_label,
                        code="changed_preexisting_allowlisted_path",
                        before=_state_label(old),
                        after=_state_label(new),
                    )
                )
            continue
        if old != new:
            issues.append(
                ScopeIssue(
                    path=path_label,
                    code="dirty_identity",
                    before=_state_label(old),
                    after=_state_label(new),
                )
            )
    return sorted(issues, key=lambda issue: (issue.path, issue.code, issue.after))


def format_scope_issues(issues: Sequence[ScopeIssue]) -> str:
    return "\n".join(
        f"- {sanitize_path(issue.path)}: code={issue.code} before[{issue.before}] after[{issue.after}]"
        for issue in issues
    )


def validate_files(
    *,
    readiness_path: Path = READINESS_PATH,
    index_path: Path = INDEX_PATH,
    roadmap_path: Path = ROADMAP_PATH,
    requirements_path: Path = REQUIREMENTS_PATH,
    repo: Path = ROOT,
    baseline_path: Path | None = None,
    baseline_sha256: str | None = None,
    allow_paths: Sequence[str] = (),
    accept_dirty_allow_paths: Sequence[str] = (),
    external: bool = False,
    fetch: Callable[[str], tuple[int, str]] = _fetch_external,
) -> dict[str, int]:
    target_paths = [readiness_path, index_path, roadmap_path, requirements_path]
    texts: dict[Path, str] = {}
    for path in target_paths:
        try:
            texts[path] = path.read_bytes().decode("utf-8")
        except PATH_ACCESS_ERRORS:
            raise _safe_path_error("unable to read or decode validation input", path, repo=repo) from None
    documents = [(_safe_file_label(path, repo), texts[path]) for path in target_paths]
    _secret_gate(documents)

    readiness_text = texts[readiness_path]
    index_text = texts[index_path]
    roadmap_text = texts[roadmap_path]
    requirements_text = texts[requirements_path]
    _validate_no_insecure_markdown_links(readiness_text)
    _validate_no_insecure_markdown_links(index_text)
    counts = validate_structure(readiness_text, index_text, roadmap_text, requirements_text)
    counts["verified_tags"] = validate_provenance_text(readiness_text, repo=repo)
    counts["local_links"] = validate_local_link_texts(
        [(readiness_path, readiness_text), (index_path, index_text)],
        repo=repo,
    )
    counts["git_scope_checked"] = 0
    counts["git_ignored_paths_checked"] = 0
    if baseline_path is not None:
        try:
            raw_baseline = baseline_path.read_bytes()
        except PATH_ACCESS_ERRORS:
            raise _safe_path_error(
                "unable to read Git baseline",
                baseline_path,
                repo=repo,
                exit_code=EXIT_SCOPE,
            ) from None
        if baseline_sha256 is None or re.fullmatch(r"[0-9a-f]{64}", baseline_sha256) is None:
            raise ValidationError("Git baseline requires an externally retained SHA-256", exit_code=EXIT_SCOPE)
        if hashlib.sha256(raw_baseline).hexdigest() != baseline_sha256:
            raise ValidationError("Git baseline SHA-256 mismatch", exit_code=EXIT_SCOPE)
        try:
            baseline = json.loads(raw_baseline.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            raise _safe_path_error(
                "unable to decode Git baseline",
                baseline_path,
                repo=repo,
                exit_code=EXIT_SCOPE,
            ) from None
        issues = check_git_scope(
            baseline,
            repo=repo,
            allow_paths=allow_paths,
            accept_dirty_allow_paths=accept_dirty_allow_paths,
        )
        if issues:
            raise ValidationError("Git scope violations:\n" + format_scope_issues(issues), exit_code=EXIT_SCOPE)
        counts["git_scope_checked"] = 1
    if external:
        counts["external_links"] = len(check_external_links(readiness_text, fetch=fetch))
    return counts


def _git_text(repo: Path, *args: str) -> None:
    _run_git(repo, list(args))


def _write(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _init_scope_test_repo(repo: Path) -> None:
    _git_text(repo, "init", "-q")
    _git_text(repo, "config", "user.email", "readiness-validator@example.invalid")
    _git_text(repo, "config", "user.name", "Readiness Validator")
    _write(repo / "tracked.txt", "baseline\n")
    _write(repo / "allowed.txt", "baseline\n")
    _git_text(repo, "add", "tracked.txt", "allowed.txt")
    _git_text(repo, "commit", "-q", "-m", "baseline")


def run_scope_self_tests() -> dict[str, bool]:
    results: dict[str, bool] = {}

    def run_case(
        name: str,
        mutate: Callable[[Path], None],
        *,
        setup: Callable[[Path], None] | None = None,
        allow_paths: Sequence[str] = (),
        accept_dirty_allow_paths: Sequence[str] = (),
        sensitive_markers: Sequence[str] = ("TOP_SECRET_MARKER",),
        expect_issues: bool = True,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="survarena-readiness-scope-") as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            _init_scope_test_repo(repo)
            if setup is not None:
                setup(repo)
            baseline = create_git_baseline(repo)
            mutate(repo)
            issues = check_git_scope(
                baseline,
                repo=repo,
                allow_paths=allow_paths,
                accept_dirty_allow_paths=accept_dirty_allow_paths,
            )
            diagnostics = format_scope_issues(issues)
            serialized_baseline = json.dumps(baseline, sort_keys=True)
            safe = all(
                marker not in diagnostics and marker not in serialized_baseline
                for marker in sensitive_markers
            )
            results[name] = bool(issues) == expect_issues and safe

    run_case(
        "changed_preexisting_dirty_content",
        lambda repo: _write(repo / "tracked.txt", "TOP_SECRET_MARKER second\n"),
        setup=lambda repo: _write(repo / "tracked.txt", "TOP_SECRET_MARKER first\n"),
    )
    run_case(
        "index_only_change",
        lambda repo: (_write(repo / "tracked.txt", "staged\n"), _git_text(repo, "add", "tracked.txt")),
    )
    run_case("new_untracked_path", lambda repo: _write(repo / "new.txt", "new\n"))
    run_case(
        "removed_untracked_path",
        lambda repo: (repo / "untracked.txt").unlink(),
        setup=lambda repo: _write(repo / "untracked.txt", "existing\n"),
    )
    run_case("rename", lambda repo: _git_text(repo, "mv", "tracked.txt", "renamed.txt"))
    run_case("unauthorized_path", lambda repo: _write(repo / "tracked.txt", "changed\n"))
    run_case(
        "allowlisted_change",
        lambda repo: _write(repo / "allowed.txt", "allowed change\n"),
        allow_paths=["allowed.txt"],
        expect_issues=False,
    )
    run_case(
        "preexisting_dirty_allowlisted_change",
        lambda repo: _write(repo / "allowed.txt", "second dirty value\n"),
        setup=lambda repo: _write(repo / "allowed.txt", "first dirty value\n"),
        allow_paths=["allowed.txt"],
    )
    run_case(
        "explicitly_accepted_dirty_allowlisted_change",
        lambda repo: _write(repo / "allowed.txt", "second dirty value\n"),
        setup=lambda repo: _write(repo / "allowed.txt", "first dirty value\n"),
        allow_paths=["allowed.txt"],
        accept_dirty_allow_paths=["allowed.txt"],
        expect_issues=False,
    )
    run_case(
        "untracked_mode_change",
        lambda repo: (repo / "untracked.txt").chmod(0o755),
        setup=lambda repo: _write(repo / "untracked.txt", "mode test\n"),
    )

    def clean_commit(repo: Path) -> None:
        _write(repo / "tracked.txt", "committed change\n")
        _git_text(repo, "add", "tracked.txt")
        _git_text(repo, "commit", "-q", "-m", "unauthorized clean commit")

    run_case("unauthorized_clean_commit", clean_commit)

    def commit_and_revert(repo: Path) -> None:
        clean_commit(repo)
        _write(repo / "tracked.txt", "baseline\n")
        _git_text(repo, "add", "tracked.txt")
        _git_text(repo, "commit", "-q", "-m", "revert bytes")

    run_case("unauthorized_commit_then_revert", commit_and_revert)

    credential_path = "hf_" + "q" * 32 + ".txt"
    run_case(
        "credential_shaped_path_redaction",
        lambda repo: _write(repo / credential_path, "content\n"),
        sensitive_markers=[credential_path],
    )

    def ignored_setup(repo: Path) -> None:
        _write(repo / ".gitignore", "ignored-cache/\n")
        _git_text(repo, "add", ".gitignore")
        _git_text(repo, "commit", "-q", "-m", "declare ignored cache")

    run_case(
        "git_ignored_path_excluded_by_contract",
        lambda repo: (_write(repo / "ignored-cache" / "cache.bin", "ignored\n")),
        setup=lambda repo: (
            ignored_setup(repo),
            (repo / "ignored-cache").mkdir(),
        ),
        expect_issues=False,
    )
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the SurvArena benchmark-readiness register.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot-git", help="Record HEAD/tree plus dirty and index identities.")
    snapshot.add_argument("--output", required=True, type=Path)

    check = subparsers.add_parser("check", help="Run secret-first static validation and optional link/scope checks.")
    check.add_argument("--readiness", type=Path, default=READINESS_PATH)
    check.add_argument("--index", type=Path, default=INDEX_PATH)
    check.add_argument("--roadmap", type=Path, default=ROADMAP_PATH)
    check.add_argument("--requirements", type=Path, default=REQUIREMENTS_PATH)
    check.add_argument("--baseline", type=Path)
    check.add_argument("--baseline-sha256")
    check.add_argument("--allow-path", action="append", default=[])
    check.add_argument("--accept-dirty-allow-path", action="append", default=[])
    check.add_argument("--external", choices=["offline", "check"], default="offline")

    subparsers.add_parser("self-test", help="Run disposable-repository scope-guard adversarial cases.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        repo = _repo_root(ROOT)
        if args.command == "snapshot-git":
            baseline_sha256 = write_git_baseline(args.output, repo=repo)
            print(f"benchmark_readiness_validation: git baseline recorded sha256={baseline_sha256}")
            return 0
        if args.command == "self-test":
            results = run_scope_self_tests()
            for name, passed in results.items():
                print(f"benchmark_readiness_self_test: {name}={'PASS' if passed else 'FAIL'}")
            if not all(results.values()):
                return EXIT_SCOPE
            print("benchmark_readiness_self_test: all adversarial cases passed")
            return 0
        counts = validate_files(
            readiness_path=args.readiness,
            index_path=args.index,
            roadmap_path=args.roadmap,
            requirements_path=args.requirements,
            repo=repo,
            baseline_path=args.baseline,
            baseline_sha256=args.baseline_sha256,
            allow_paths=args.allow_path,
            accept_dirty_allow_paths=args.accept_dirty_allow_path,
            external=args.external == "check",
        )
        summary = " ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        print(f"benchmark_readiness_validation: PASS {summary}")
        return 0
    except ValidationError as error:
        print(f"benchmark_readiness_validation: FAIL {error}", file=sys.stderr)
        return error.exit_code
    except Exception:
        print("benchmark_readiness_validation: FAIL code=unexpected_error details=redacted", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
