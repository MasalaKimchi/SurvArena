from __future__ import annotations

from pathlib import Path

from src.data.loaders import load_dataset


def test_load_aids_dataset_from_sksurv() -> None:
    dataset = load_dataset("aids", repo_root=Path(__file__).resolve().parents[1])

    assert dataset.metadata.dataset_id == "aids"
    assert dataset.metadata.source == "scikit-survival"
    assert dataset.X.shape == (1151, 11)
    assert dataset.time.shape == (1151,)
    assert dataset.event.shape == (1151,)
    assert int(dataset.event.sum()) == 1055
