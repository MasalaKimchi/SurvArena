from __future__ import annotations

import numpy as np
import pytest

from survarena.methods.registry import get_method_class


def _toy_survival_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(64, 6))
    linear = 0.8 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2]
    time = np.exp(1.2 - linear + 0.1 * rng.normal(size=64))
    event = (rng.random(64) < 0.75).astype(np.int32)
    return (
        X[:48].astype(np.float64),
        np.maximum(time[:48], 0.05).astype(np.float64),
        event[:48].astype(np.int32),
        X[48:].astype(np.float64),
        np.maximum(time[48:], 0.05).astype(np.float64),
        event[48:].astype(np.int32),
    )


def test_deepsurv_moco_fit_predict_works_without_momentum_encoder() -> None:
    X_train, time_train, event_train, X_test, time_test, event_test = _toy_survival_arrays()
    method = get_method_class("deepsurv_moco")(
        hidden_layers="16-8",
        batch_size=16,
        max_epochs=6,
        patience=2,
        queue_size=64,
        use_momentum_encoder=False,
        seed=0,
    )

    method.fit(X_train, time_train, event_train, X_test, time_test, event_test)
    risk = method.predict_risk(X_test)
    survival = method.predict_survival(X_test, np.asarray([0.5, 1.0, 2.0, 4.0], dtype=np.float64))

    assert risk.shape == (X_test.shape[0],)
    assert np.isfinite(risk).all()
    assert survival.shape == (X_test.shape[0], 4)
    assert np.isfinite(survival).all()
    assert np.all((survival >= 0.0) & (survival <= 1.0))
    assert np.all(np.diff(survival, axis=1) <= 1e-8)


def test_deepsurv_moco_requires_observed_events() -> None:
    X_train, time_train, event_train, *_ = _toy_survival_arrays()
    method = get_method_class("deepsurv_moco")(max_epochs=2, patience=1, batch_size=16, queue_size=32, seed=0)
    with pytest.raises(ValueError, match="requires at least one observed event"):
        method.fit(X_train, time_train, np.zeros_like(event_train))
