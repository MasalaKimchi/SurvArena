from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
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


def test_load_nwtco_pycox_dataset_matches_documented_shape() -> None:
    dataset = load_dataset("nwtco", repo_root=Path(__file__).resolve().parents[1])

    assert dataset.metadata.dataset_id == "nwtco"
    assert dataset.metadata.source == "pycox"
    assert dataset.X.shape == (4028, 6)
    assert dataset.time.shape == (4028,)
    assert np.all(dataset.time > 0.0)
    assert dataset.event.shape == (4028,)
    assert int(dataset.event.sum()) == 571


def test_load_local_file_dataset_from_config(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs" / "datasets"
    configs_dir.mkdir(parents=True)
    data_path = tmp_path / "data" / "processed" / "toy.csv"
    data_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3"],
            "time": [5.0, 10.0, 12.0],
            "event": [1, 0, 1],
            "gene_a": [0.1, 0.2, 0.3],
            "gene_b": [2.0, 1.5, 1.0],
        }
    ).to_csv(data_path, index=False)
    (configs_dir / "toy_local.yaml").write_text(
        "\n".join(
            [
                "dataset_id: toy_local",
                "name: Toy local file",
                "source: local_file",
                "local_path: data/processed/toy.csv",
                "time_col: time",
                "event_col: event",
                "id_col: patient_id",
                "primary_metric: uno_c",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_dataset("toy_local", repo_root=tmp_path)

    assert dataset.metadata.dataset_id == "toy_local"
    assert dataset.metadata.source == "local_file"
    assert dataset.X.shape == (3, 2)
    assert dataset.time.tolist() == [5.0, 10.0, 12.0]
    assert dataset.event.tolist() == [1, 0, 1]


def test_load_local_file_dataset_missing_file_has_actionable_error(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs" / "datasets"
    configs_dir.mkdir(parents=True)
    (configs_dir / "missing_local.yaml").write_text(
        "\n".join(
            [
                "dataset_id: missing_local",
                "source: local_file",
                "local_path: data/processed/missing.csv",
                "time_col: time",
                "event_col: event",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Run the documented preparation command"):
        load_dataset("missing_local", repo_root=tmp_path)
