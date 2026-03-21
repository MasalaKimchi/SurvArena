from __future__ import annotations

from src.automl.presets import resolve_preset


def test_resolve_preset_skips_high_capacity_models_for_low_event_data() -> None:
    preset = resolve_preset(
        "best",
        n_rows=120,
        n_features=24,
        event_count=8,
        event_fraction=8 / 120,
        enable_foundation_models=True,
    )

    assert "coxph" in preset.method_ids
    assert "coxnet" in preset.method_ids
    assert "rsf" in preset.method_ids
    assert "deepsurv" not in preset.method_ids
    assert "deepsurv_moco" not in preset.method_ids
    assert "tabpfn_survival" not in preset.method_ids
    assert any("only 8 observed events" in note for note in preset.portfolio_notes)


def test_resolve_preset_skips_foundation_models_for_unsupported_feature_shapes() -> None:
    preset = resolve_preset(
        "medium",
        n_rows=1000,
        n_features=40,
        event_count=180,
        event_fraction=0.18,
        high_cardinality_feature_count=3,
        has_datetime_features=True,
        enable_foundation_models=True,
    )

    assert "tabpfn_survival" not in preset.method_ids
    assert any("high-cardinality categorical features" in note for note in preset.portfolio_notes)
    assert any("datetime-aware feature handling" in note for note in preset.portfolio_notes)
