from __future__ import annotations

from src.api.predictor import SurvivalPredictor


def test_foundation_model_catalog_exposes_current_and_planned_backbones() -> None:
    predictor = SurvivalPredictor(label_time="time", label_event="event")

    catalog = predictor.foundation_model_catalog()

    assert "tabpfn_survival" in catalog["method_id"].tolist()
    assert "tabicl_survival" in catalog["method_id"].tolist()
    implemented = dict(zip(catalog["method_id"], catalog["implemented"], strict=False))
    assert implemented["tabpfn_survival"] is True
    assert implemented["tabicl_survival"] is False
