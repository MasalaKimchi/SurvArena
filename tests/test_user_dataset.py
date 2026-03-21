from __future__ import annotations

import pandas as pd

from src.data.user_dataset import load_user_dataset


def test_load_user_dataset_populates_feature_metadata_and_diagnostics() -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 0, 0],
            "age": [61, 57, 70, 66],
            "patient_id": ["a1", "a2", "a3", "a4"],
            "visit_date": ["2024-01-01", "2024-01-03", "2024-01-05", "2024-01-07"],
            "notes": [
                "progression observed with several complications",
                "stable follow-up with no major changes recorded",
                "stable follow-up with no major changes recorded",
                "stable follow-up with no major changes recorded",
            ],
        }
    )

    dataset = load_user_dataset(frame, time_col="time", event_col="event", dataset_id="toy")

    assert dataset.metadata.feature_types == ["numerical", "categorical", "datetime", "text"]
    assert dataset.metadata.diagnostics is not None
    assert dataset.metadata.diagnostics.n_events == 1
    assert "patient_id" in dataset.metadata.diagnostics.id_like_features
    assert any("Very few observed events" in warning for warning in dataset.metadata.diagnostics.warnings)
    assert any(feature.inferred_type == "datetime" for feature in dataset.metadata.feature_metadata)
    assert any(feature.inferred_type == "text" for feature in dataset.metadata.feature_metadata)


def test_load_user_dataset_preserves_binary_string_event_labels() -> None:
    frame = pd.DataFrame(
        {
            "time": [1, 2, 3],
            "event": ["dead", "alive", "dead"],
            "x1": [0.2, 0.5, 0.7],
        }
    )

    dataset = load_user_dataset(frame, time_col="time", event_col="event")

    assert dataset.event.tolist() == [1, 0, 1]
