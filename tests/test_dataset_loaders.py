from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from survarena.data.loaders import load_dataset


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
