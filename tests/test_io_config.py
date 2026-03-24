from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from survarena.config import read_yaml
from survarena.data.io import read_tabular_data


def test_read_tabular_data_returns_dataframe_copy() -> None:
    frame = pd.DataFrame({"age": [61, 57], "stage": ["i", "ii"]})

    loaded = read_tabular_data(frame)

    assert loaded.equals(frame)
    assert loaded is not frame


def test_read_tabular_data_supports_csv_and_parquet(tmp_path: Path) -> None:
    frame = pd.DataFrame({"age": [61, 57], "stage": ["i", "ii"]})
    csv_path = tmp_path / "toy.csv"
    parquet_path = tmp_path / "toy.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)

    csv_loaded = read_tabular_data(csv_path)
    parquet_loaded = read_tabular_data(parquet_path)

    pd.testing.assert_frame_equal(csv_loaded, frame)
    pd.testing.assert_frame_equal(parquet_loaded, frame)


def test_read_tabular_data_rejects_unknown_file_extensions(tmp_path: Path) -> None:
    path = tmp_path / "toy.tsv"
    path.write_text("age\tstage\n61\ti\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        read_tabular_data(path)


def test_read_yaml_loads_mapping_payload(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("dataset_id: toy\nn_trials: 4\n", encoding="utf-8")

    loaded = read_yaml(path)

    assert loaded == {"dataset_id": "toy", "n_trials": 4}
