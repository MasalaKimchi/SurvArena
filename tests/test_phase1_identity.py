from __future__ import annotations

import json

import numpy as np

from survarena.benchmark.runner import _benchmark_run_id, _save_model_artifacts


def test_benchmark_run_id_is_arm_qualified() -> None:
    common = {
        "dataset_id": "whas500__baseline",
        "method_id": "coxph",
        "split_id": "repeat_0_fold_0__baseline",
        "seed": 11,
    }

    no_hpo_id = _benchmark_run_id(**common, hpo_mode="no_hpo")
    hpo_id = _benchmark_run_id(**common, hpo_mode="hpo")

    assert no_hpo_id != hpo_id
    assert no_hpo_id.endswith("_no_hpo")
    assert hpo_id.endswith("_hpo")


def test_artifact_manifest_carries_arm_identity(tmp_path) -> None:
    metadata = _save_model_artifacts(
        artifact_dir=tmp_path,
        benchmark_id="identity_test",
        dataset_id="whas500",
        method_id="coxph",
        split_id="repeat_0_fold_0",
        seed=11,
        hpo_mode="hpo",
        model={"model": "state"},
        preprocessor={"preprocessor": "state"},
        best_params={},
        eval_times=np.asarray([1.0, 2.0]),
        horizons=np.asarray([1.0, 2.0, 3.0]),
        train_idx=np.asarray([0, 1]),
        test_idx=np.asarray([2]),
        train_time=np.asarray([1.0, 2.0]),
        train_event=np.asarray([1, 0]),
        test_time=np.asarray([3.0]),
        test_event=np.asarray([1]),
        risk_scores=np.asarray([0.5]),
        survival_probs=np.asarray([[0.8, 0.6]]),
    )

    manifest = json.loads(
        open(metadata["artifact_manifest_path"], encoding="utf-8").read()
    )
    assert manifest["hpo_mode"] == "hpo"
    assert "hpo" in metadata["artifact_manifest_path"]
