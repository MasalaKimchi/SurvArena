from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from survarena.evaluation._eligibility import eligible_frame

_CORE_IDENTITY_COLUMNS = ("benchmark_id", "dataset_id", "method_id")
_STRATUM_CANDIDATES = ("benchmark_id", "hpo_mode")
_CELL_CANDIDATES = (
    "benchmark_id",
    "hpo_mode",
    "dataset_id",
    "split_id",
    "seed",
    "scenario_id",
    "robustness_scenario",
    "robustness_level",
    "perturbation_id",
)


@dataclass(slots=True)
class ComparisonPopulation:
    frame: pd.DataFrame
    coverage: pd.DataFrame
    cell_keys: tuple[str, ...]
    required_methods: tuple[str, ...]
    support_digest: str


def stratum_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in _STRATUM_CANDIDATES if column in frame.columns]


def comparison_cell_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in _CELL_CANDIDATES if column in frame.columns]
    if "dataset_id" not in columns:
        raise ValueError("Comparison input requires dataset_id.")
    return columns


def validate_unique_comparison_cells(frame: pd.DataFrame) -> list[str]:
    missing = [column for column in _CORE_IDENTITY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Comparison input is missing identity columns: {missing}")
    cell_keys = comparison_cell_columns(frame)
    identity = [*cell_keys, "method_id"]
    null_identity = frame[identity].isna().any(axis=1)
    if bool(null_identity.any()):
        bad_columns = [column for column in identity if bool(frame.loc[null_identity, column].isna().any())]
        raise ValueError(f"Comparison identity contains missing values in: {bad_columns}")
    duplicate_mask = frame.duplicated(identity, keep=False)
    if bool(duplicate_mask.any()):
        examples = frame.loc[duplicate_mask, identity].drop_duplicates().head(5).to_dict(orient="records")
        raise ValueError(f"Duplicate comparison cell for method identity {identity}: {examples}")
    return cell_keys


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"})


def _method_roster(sub: pd.DataFrame, required_methods: Iterable[str] | None) -> tuple[str, ...]:
    if required_methods is not None:
        roster = tuple(sorted({str(method) for method in required_methods}))
    else:
        roster = tuple(sorted(sub["method_id"].dropna().astype(str).unique()))
    if not roster:
        raise ValueError("Comparison method roster is empty.")
    return roster


def _cell_set(frame: pd.DataFrame, cell_keys: list[str]) -> set[tuple[object, ...]]:
    if frame.empty:
        return set()
    return set(frame[cell_keys].itertuples(index=False, name=None))


def coverage_summary(
    frame: pd.DataFrame,
    *,
    metric: str,
    required_methods: Iterable[str] | None = None,
) -> pd.DataFrame:
    if metric not in frame.columns:
        raise ValueError(f"Metric '{metric}' not found in frame.")
    cell_keys = validate_unique_comparison_cells(frame)
    strata = stratum_columns(frame)
    rows: list[dict[str, object]] = []

    grouper: str | list[str]
    if len(strata) == 1:
        grouper = strata[0]
    else:
        grouper = strata
    for stratum_key, stratum_sub in frame.groupby(grouper, sort=True, dropna=False):
        key_tuple = stratum_key if isinstance(stratum_key, tuple) else (stratum_key,)
        stratum_values = dict(zip(strata, key_tuple, strict=True))
        roster = _method_roster(stratum_sub, required_methods)
        for dataset_id, dataset_sub in stratum_sub.groupby("dataset_id", sort=True, dropna=False):
            expected_cells = _cell_set(dataset_sub, cell_keys)
            method_payloads: list[dict[str, object]] = []
            dataset_complete = bool(expected_cells)
            for method_id in roster:
                method_sub = dataset_sub[dataset_sub["method_id"].astype(str) == method_id]
                eligible = eligible_frame(method_sub, metric=metric)
                attempted_cells = _cell_set(method_sub, cell_keys)
                eligible_cells = _cell_set(eligible, cell_keys)
                status = (
                    method_sub["status"].fillna("").astype(str).str.lower()
                    if "status" in method_sub.columns
                    else pd.Series("success", index=method_sub.index, dtype=object)
                )
                reason = (
                    method_sub["ineligible_reason"].fillna("").astype(str).str.lower()
                    if "ineligible_reason" in method_sub.columns
                    else pd.Series("", index=method_sub.index, dtype=object)
                )
                failure_type = (
                    method_sub["failure_type"].fillna("").astype(str).str.lower()
                    if "failure_type" in method_sub.columns
                    else pd.Series("", index=method_sub.index, dtype=object)
                )
                fallback = reason.str.contains("fallback", regex=False)
                for column in ("used_fallback", "foundation_used_fallback", "foundation_discrete_hazard_fallback"):
                    fallback = fallback | _bool_series(method_sub, column)
                method_complete = eligible_cells == expected_cells
                dataset_complete = dataset_complete and method_complete
                method_payloads.append(
                    {
                        **stratum_values,
                        "dataset_id": dataset_id,
                        "method_id": method_id,
                        "metric": metric,
                        "n_expected_cells": len(expected_cells),
                        "n_attempted": len(method_sub),
                        "n_successful": int((status == "success").sum()),
                        "n_failed": int((status != "success").sum()),
                        "n_timeout": int((status.str.contains("timeout") | failure_type.str.contains("timeout")).sum()),
                        "n_invalid_prediction": int(
                            (status.str.contains("invalid_prediction") | reason.str.contains("invalid_prediction")).sum()
                        ),
                        "n_fallback": int(fallback.sum()),
                        "n_eligible": len(eligible),
                        "n_missing_cells": len(expected_cells - attempted_cells),
                        "n_ineligible_cells": len(expected_cells - eligible_cells),
                        "method_on_common_support": method_complete,
                    }
                )
            for payload in method_payloads:
                payload["dataset_on_common_support"] = dataset_complete
                rows.append(payload)

    columns = [
        *strata,
        "dataset_id",
        "method_id",
        "metric",
        "n_expected_cells",
        "n_attempted",
        "n_successful",
        "n_failed",
        "n_timeout",
        "n_invalid_prediction",
        "n_fallback",
        "n_eligible",
        "n_missing_cells",
        "n_ineligible_cells",
        "method_on_common_support",
        "dataset_on_common_support",
    ]
    return pd.DataFrame(rows, columns=columns)


def _support_digest(coverage: pd.DataFrame, *, metric: str, cell_keys: list[str]) -> str:
    supported = coverage[coverage["dataset_on_common_support"]]
    payload = {
        "metric": metric,
        "cell_keys": cell_keys,
        "support": supported[
            [column for column in [*stratum_columns(coverage), "dataset_id", "method_id"] if column in supported.columns]
        ]
        .astype(str)
        .sort_values(list(c for c in [*stratum_columns(coverage), "dataset_id", "method_id"] if c in supported.columns))
        .to_dict(orient="records"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_comparison_population(
    frame: pd.DataFrame,
    *,
    metric: str,
    required_methods: Iterable[str] | None = None,
    require_complete: bool = True,
) -> ComparisonPopulation:
    cell_keys = validate_unique_comparison_cells(frame)
    coverage = coverage_summary(frame, metric=metric, required_methods=required_methods)
    eligible = eligible_frame(frame, metric=metric)
    strata = stratum_columns(frame)
    support_keys = [*strata, "dataset_id"]
    supported = coverage[coverage["dataset_on_common_support"]][support_keys].drop_duplicates()
    if require_complete:
        if supported.empty:
            population_frame = eligible.iloc[0:0].copy()
        else:
            population_frame = eligible.merge(supported, on=support_keys, how="inner", validate="many_to_one")
    else:
        population_frame = eligible.copy()
    methods = (
        tuple(sorted({str(method) for method in required_methods}))
        if required_methods is not None
        else tuple(sorted(frame["method_id"].dropna().astype(str).unique()))
    )
    digest = _support_digest(coverage, metric=metric, cell_keys=cell_keys)
    return ComparisonPopulation(
        frame=population_frame.reset_index(drop=True),
        coverage=coverage.reset_index(drop=True),
        cell_keys=tuple(cell_keys),
        required_methods=methods,
        support_digest=digest,
    )


def dataset_method_scores(
    frame: pd.DataFrame,
    *,
    metric: str,
    required_methods: Iterable[str] | None = None,
    require_complete: bool = True,
) -> tuple[pd.DataFrame, ComparisonPopulation]:
    population = build_comparison_population(
        frame,
        metric=metric,
        required_methods=required_methods,
        require_complete=require_complete,
    )
    strata = stratum_columns(population.frame)
    keys = [*strata, "dataset_id", "method_id"]
    if population.frame.empty:
        return pd.DataFrame(columns=[*keys, metric, "n_cells", "support_digest"]), population
    scores = population.frame.groupby(keys, as_index=False).agg(
        **{metric: (metric, "mean")},
        n_cells=(metric, "size"),
    )
    scores["support_digest"] = population.support_digest
    return scores, population
