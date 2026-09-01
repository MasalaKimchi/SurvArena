"""Shared filtering for statistical summaries.

Manuscript-grade comparisons must be computed only over rows that represent a
genuine, comparable model fit. A row is *ineligible* for ranking / significance /
rating / bootstrap aggregation when any of the following hold:

- it has a ``status`` column whose value is not ``"success"`` (the fit failed or
  was skipped), or
- it is explicitly flagged ``comparison_ineligible`` (e.g. a degenerate
  trivial-predictor fallback that produced a plausible-but-meaningless score, or
  a parity counterpart that was never run), or
- (when a metric is supplied) its value for that metric is missing.

Filtering is intentionally *tolerant of absent columns*: a frame that carries no
``status`` / ``comparison_ineligible`` column is treated as fully eligible, so
lightweight callers and tests that build minimal frames keep working.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ineligible_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    valid = normalized.isin({"", "0", "1", "false", "true", "no", "yes"})
    # An ambiguous serialized value is unsafe evidence, so fail closed.
    return normalized.isin({"1", "true", "yes"}) | ~valid


def eligible_frame(frame: pd.DataFrame, *, metric: str | None = None) -> pd.DataFrame:
    """Return the subset of ``frame`` eligible for statistical aggregation.

    Parameters
    ----------
    frame:
        Per-fold benchmark results.
    metric:
        Optional metric column; when provided, rows with a missing value for
        that metric are also dropped.
    """
    out = frame
    if "status" in out.columns:
        out = out[out["status"].astype(str) == "success"]
    if "comparison_ineligible" in out.columns:
        ineligible = _ineligible_mask(out["comparison_ineligible"])
        out = out[~ineligible]
    if metric is not None and metric in out.columns:
        numeric = pd.to_numeric(out[metric], errors="coerce")
        out = out[np.isfinite(numeric.to_numpy(dtype=float))]
    return out.copy()
