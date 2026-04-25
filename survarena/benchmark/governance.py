from __future__ import annotations

from typing import Any


DUAL_HPO_MODE_ORDER: tuple[str, str] = ("no_hpo", "hpo")
VALID_HPO_MODES = frozenset(DUAL_HPO_MODE_ORDER)


def resolve_comparison_modes(benchmark_cfg: dict[str, Any]) -> tuple[str, ...]:
    configured = benchmark_cfg.get("comparison_modes", benchmark_cfg.get("hpo_modes"))
    if configured is None:
        return DUAL_HPO_MODE_ORDER
    if isinstance(configured, str):
        modes = [configured]
    else:
        try:
            modes = list(configured)
        except TypeError as exc:
            raise ValueError("comparison_modes must be a non-empty list containing 'no_hpo' and/or 'hpo'.") from exc

    normalized: list[str] = []
    for mode in modes:
        mode_name = str(mode).strip().lower()
        if mode_name not in VALID_HPO_MODES:
            raise ValueError(
                f"Invalid comparison mode '{mode}'. Allowed modes: {', '.join(DUAL_HPO_MODE_ORDER)}."
            )
        if mode_name not in normalized:
            normalized.append(mode_name)
    if not normalized:
        raise ValueError("comparison_modes must include at least one mode.")
    return tuple(normalized)


def normalize_hpo_budget_telemetry(
    *,
    hpo_metadata: dict[str, Any],
    hpo_cfg: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(hpo_metadata)
    realized_trial_count = int(normalized.get("realized_trial_count", normalized.get("trial_count", 0)))
    normalized["realized_trial_count"] = realized_trial_count
    normalized["trial_count"] = realized_trial_count
    normalized["requested_max_trials"] = int(normalized.get("requested_max_trials", hpo_cfg.get("max_trials", 20)))
    timeout_value = normalized.get("requested_timeout_seconds", hpo_cfg.get("timeout_seconds"))
    normalized["requested_timeout_seconds"] = None if timeout_value is None else float(timeout_value)
    normalized["requested_sampler"] = str(normalized.get("requested_sampler", hpo_cfg.get("sampler", "tpe"))).lower()
    normalized["requested_pruner"] = str(normalized.get("requested_pruner", hpo_cfg.get("pruner", "median"))).lower()
    return normalized


def apply_parity_governance(
    *,
    run_records: list[dict[str, Any]],
    fold_records: list[dict[str, Any]],
    comparison_modes: tuple[str, ...],
) -> None:
    parity_modes: dict[str, set[str]] = {}
    for run_payload in run_records:
        metrics = run_payload.get("metrics", {})
        status = str(run_payload.get("status", metrics.get("status", ""))).lower()
        parity_key = str(metrics.get("parity_key", ""))
        hpo_mode = str(metrics.get("hpo_mode", ""))
        if status == "success" and parity_key and hpo_mode:
            parity_modes.setdefault(parity_key, set()).add(hpo_mode)

    for metrics in _parity_targets(run_records=run_records, fold_records=fold_records):
        parity_key = str(metrics.get("parity_key", ""))
        modes = parity_modes.get(parity_key, set())
        missing_modes = [mode for mode in comparison_modes if mode not in modes]
        has_all_modes = not missing_modes
        if len(comparison_modes) == 1 or has_all_modes:
            metrics["parity_eligible"] = len(comparison_modes) > 1
            metrics["comparison_ineligible"] = False
            metrics["parity_reason"] = None
            metrics["missing_modes"] = []
        else:
            metrics["parity_eligible"] = False
            metrics["comparison_ineligible"] = True
            metrics["parity_reason"] = "missing_counterpart_mode"
            metrics["missing_modes"] = missing_modes


def _parity_targets(
    *,
    run_records: list[dict[str, Any]],
    fold_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    targets = []
    for run_payload in run_records:
        metrics = run_payload.get("metrics", {})
        if isinstance(metrics, dict):
            targets.append(metrics)
    targets.extend(fold_records)
    return targets
