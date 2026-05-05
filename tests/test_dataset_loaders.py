from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import survarena.data.loaders as loader_mod
from survarena.data.loaders import load_dataset
from survarena.methods.registry import get_method_class


@pytest.mark.parametrize(
    ("dataset_id", "shape", "event_sum"),
    [
        ("aids", (1151, 11), 96),
        ("gbsg2", (686, 8), 299),
        ("flchain", (7874, 9), 2169),
        ("whas500", (500, 14), 215),
    ],
)
def test_load_sksurv_dataset_matches_documented_event_counts(
    dataset_id: str,
    shape: tuple[int, int],
    event_sum: int,
) -> None:
    dataset = load_dataset(dataset_id, repo_root=Path(__file__).resolve().parents[1])

    assert dataset.metadata.dataset_id == dataset_id
    assert dataset.metadata.source == "scikit-survival"
    assert dataset.X.shape == shape
    assert dataset.time.shape == (shape[0],)
    assert np.all(dataset.time > 0.0)
    assert dataset.event.shape == (shape[0],)
    assert int(dataset.event.sum()) == event_sum


def test_load_kkbox_uses_pycox_cache_and_drops_survival_and_identifier_columns(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "msno": ["user_a", "user_b", "user_c"],
            "duration": [10.0, 20.0, 30.0],
            "event": [1, 0, 1],
            "censor_duration": [30.0, 30.0, 30.0],
            "log_payment_plan_days": [3.1, 2.7, 4.0],
            "payment_method_id": pd.Series([1, 2, 1], dtype="category"),
        }
    )
    monkeypatch.setattr(loader_mod, "_read_kkbox_pycox_frame", lambda: frame)

    dataset = load_dataset("kkbox", repo_root=Path(__file__).resolve().parents[1])

    assert dataset.metadata.dataset_id == "kkbox"
    assert dataset.metadata.source == "pycox"
    assert dataset.X.columns.tolist() == ["log_payment_plan_days", "payment_method_id"]
    np.testing.assert_allclose(dataset.time, np.array([10.0, 20.0, 30.0]))
    np.testing.assert_array_equal(dataset.event, np.array([1, 0, 1]))


def test_load_kkbox_reports_missing_pycox_cache(monkeypatch) -> None:
    monkeypatch.setattr(loader_mod, "_read_kkbox_pycox_frame", lambda: None)

    with pytest.raises(FileNotFoundError, match="KKBox data is not locally available"):
        load_dataset("kkbox", repo_root=Path(__file__).resolve().parents[1])


def test_load_kkbox_smoke_fits_coxph_from_loaded_features(monkeypatch) -> None:
    n_rows = 24
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "msno": [f"user_{idx}" for idx in range(n_rows)],
            "duration": rng.uniform(5.0, 40.0, size=n_rows),
            "event": [idx % 3 != 0 for idx in range(n_rows)],
            "censor_duration": np.full(n_rows, 45.0),
            "log_payment_plan_days": rng.normal(size=n_rows),
            "log_actual_amount_paid": rng.normal(size=n_rows),
        }
    )
    monkeypatch.setattr(loader_mod, "_read_kkbox_pycox_frame", lambda: frame)
    dataset = load_dataset("kkbox", repo_root=Path(__file__).resolve().parents[1])

    coxph = get_method_class("coxph")(alpha=0.01)
    coxph.fit(dataset.X.to_numpy(dtype=float), dataset.time, dataset.event)
    risk = coxph.predict_risk(dataset.X.to_numpy(dtype=float))

    assert risk.shape == (n_rows,)
    assert np.isfinite(risk).all()
