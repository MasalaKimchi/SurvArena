from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import torch


def _import_torch():
    import torch

    return torch


@dataclass
class SurvivalDatasetConfig:
    config: list[Any]
    X_raw: Any
    surrogate_y_raw: Any
    time_raw: np.ndarray
    event_raw: np.ndarray
    cat_ix: list[int]


@dataclass
class SurvivalBatch:
    X_context: list[torch.Tensor]
    X_query: list[torch.Tensor]
    y_context: list[torch.Tensor]
    cat_indices: list[list[int] | None] | list[list[list[int] | None]]
    configs: list[Any]
    time_query_raw: torch.Tensor
    event_query_raw: torch.Tensor


def _take(obj: Any, idx: np.ndarray) -> Any:
    return obj.iloc[idx] if hasattr(obj, "iloc") else obj[idx]


def shuffle_and_chunk_survival_data(
    X_raw: Any,
    surrogate_y_raw: Any,
    time_raw: np.ndarray,
    event_raw: np.ndarray,
    *,
    max_chunk_size: int,
    seed: int,
) -> tuple[list[Any], list[Any], list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(time_raw))
    X_shuffled = _take(X_raw, order)
    surrogate_y_shuffled = _take(surrogate_y_raw, order)
    time_shuffled = np.asarray(time_raw)[order]
    event_shuffled = np.asarray(event_raw)[order]

    num_chunks = max(1, int(np.ceil(len(time_raw) / max_chunk_size)))
    chunk_indices = np.array_split(np.arange(len(time_raw)), num_chunks)

    X_chunks = [_take(X_shuffled, idx) for idx in chunk_indices if len(idx) >= 2]
    y_chunks = [_take(surrogate_y_shuffled, idx) for idx in chunk_indices if len(idx) >= 2]
    time_chunks = [time_shuffled[idx] for idx in chunk_indices if len(idx) >= 2]
    event_chunks = [event_shuffled[idx] for idx in chunk_indices if len(idx) >= 2]
    return X_chunks, y_chunks, time_chunks, event_chunks


class SurvivalDatasetCollection:
    def __init__(
        self,
        split_fn: Any,
        rng: np.random.Generator,
        dataset_configs: list[SurvivalDatasetConfig],
        *,
        backbone_model_type: Literal["classifier", "regressor"],
    ) -> None:
        self.split_fn = split_fn
        self.rng = rng
        self.dataset_configs = dataset_configs
        self.backbone_model_type = backbone_model_type

    def __len__(self) -> int:
        return len(self.dataset_configs)

    def __getitem__(self, index: int) -> SurvivalBatch:
        torch = _import_torch()
        from tabpfn.preprocessing import fit_preprocessing
        from tabpfn.preprocessing.datamodel import FeatureModality, FeatureSchema

        config = self.dataset_configs[index]
        (
            x_train_raw,
            x_test_raw,
            y_train_surrogate,
            _y_test_surrogate,
            _time_train_raw,
            time_test_raw,
            _event_train_raw,
            event_test_raw,
        ) = self.split_fn(
            config.X_raw,
            config.surrogate_y_raw,
            config.time_raw,
            config.event_raw,
        )

        feature_schema = FeatureSchema.from_only_categorical_indices(config.cat_ix, x_train_raw.shape[1])
        preprocessing_iterator = fit_preprocessing(
            configs=config.config,
            X_train=x_train_raw,
            y_train=y_train_surrogate,
            feature_schema=feature_schema,
            random_state=self.rng,
            n_preprocessing_jobs=1,
            parallel_mode="block",
        )
        (configs, preprocessors, X_trains_preprocessed, y_trains_preprocessed, feature_schema_preprocessed) = list(
            zip(*preprocessing_iterator)
        )
        X_trains_preprocessed = list(X_trains_preprocessed)
        y_trains_preprocessed = list(y_trains_preprocessed)

        X_tests_preprocessed = []
        for preprocessor in preprocessors:
            X_tests_preprocessed.append(preprocessor.transform(x_test_raw).X)

        for i in range(len(X_trains_preprocessed)):
            if not isinstance(X_trains_preprocessed[i], torch.Tensor):
                X_trains_preprocessed[i] = torch.as_tensor(X_trains_preprocessed[i], dtype=torch.float32)
            if not isinstance(X_tests_preprocessed[i], torch.Tensor):
                X_tests_preprocessed[i] = torch.as_tensor(X_tests_preprocessed[i], dtype=torch.float32)
            if not isinstance(y_trains_preprocessed[i], torch.Tensor):
                dtype = torch.long if self.backbone_model_type == "classifier" else torch.float32
                y_trains_preprocessed[i] = torch.as_tensor(y_trains_preprocessed[i], dtype=dtype)

        cat_indices = [
            modality.indices_for(FeatureModality.CATEGORICAL) for modality in feature_schema_preprocessed
        ]
        return SurvivalBatch(
            X_context=X_trains_preprocessed,
            X_query=X_tests_preprocessed,
            y_context=y_trains_preprocessed,
            cat_indices=cat_indices,
            configs=list(configs),
            time_query_raw=torch.as_tensor(np.asarray(time_test_raw), dtype=torch.float32),
            event_query_raw=torch.as_tensor(np.asarray(event_test_raw).astype(bool), dtype=torch.bool),
        )


def survival_meta_collator(batch: list[SurvivalBatch], padding_val: float = 0.0) -> SurvivalBatch:
    torch = _import_torch()
    from tabpfn.utils import pad_tensors

    assert len(batch) == 1, "Only batch_size=1 is currently supported for TabPFN survival fine-tuning."
    item = batch[0]
    num_estimators = len(item.X_context)

    def collate_list_field(field_name: str, labels: bool) -> list[torch.Tensor]:
        values = [getattr(item, field_name)[idx] for idx in range(num_estimators)]
        return [torch.stack(pad_tensors([value], padding_val=padding_val, labels=labels)) for value in values]

    return SurvivalBatch(
        X_context=collate_list_field("X_context", labels=False),
        X_query=collate_list_field("X_query", labels=False),
        y_context=collate_list_field("y_context", labels=(item.y_context[0].ndim == 1)),
        cat_indices=[[cat_ix] for cat_ix in item.cat_indices],
        configs=[[cfg] for cfg in item.configs],
        time_query_raw=torch.stack([item.time_query_raw]),
        event_query_raw=torch.stack([item.event_query_raw]),
    )
