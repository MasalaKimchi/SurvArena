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


def test_foundation_preset_requests_foundation_models_without_extra_flag() -> None:
    preset = resolve_preset(
        "foundation",
        n_rows=500,
        n_features=30,
        event_count=150,
        event_fraction=0.3,
    )

    assert preset.method_ids == ("coxph", "tabpfn_survival", "mitra_survival")


def test_foundation_preset_reports_when_no_current_adapter_is_eligible() -> None:
    preset = resolve_preset(
        "foundation",
        n_rows=50_000,
        n_features=40,
        event_count=2_000,
        event_fraction=0.04,
    )

    assert preset.method_ids == ("coxph",)
    assert any("No currently implemented foundation-model adapters were eligible" in note for note in preset.portfolio_notes)
