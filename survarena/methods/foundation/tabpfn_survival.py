from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.model_selection import train_test_split

from survarena.methods.base import BaseSurvivalMethod
from survarena.methods.foundation.readiness import ensure_foundation_runtime_ready, rewrite_foundation_runtime_error

if TYPE_CHECKING:
    import torch
    from torch import nn


def _import_torch():
    import torch

    return torch


def _import_nn():
    from torch import nn

    return nn


def _import_cox_loss():
    from torchsurv.loss.cox import neg_partial_log_likelihood

    return neg_partial_log_likelihood


def _clip_grad_norm(parameters: Any, max_norm: float) -> None:
    from torch.nn.utils import clip_grad_norm_

    clip_grad_norm_(parameters, max_norm)


def _parse_hidden_layers(value: Any) -> list[int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split("-") if part.strip()]
        return [int(part) for part in parts]
    if isinstance(value, (list, tuple)):
        return [int(part) for part in value]
    raise ValueError(f"Unsupported hidden_layers value: {value!r}")


def _activation_cls(name: str) -> type[Any]:
    nn = _import_nn()
    mapping: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "gelu": nn.GELU,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation '{name}'. Choices: {sorted(mapping)}")
    return mapping[name]


@dataclass
class _SurvivalDatasetConfig:
    config: list[Any]
    X_raw: Any
    surrogate_y_raw: Any
    time_raw: np.ndarray
    event_raw: np.ndarray
    cat_ix: list[int]


@dataclass
class _SurvivalBatch:
    X_context: list[torch.Tensor]
    X_query: list[torch.Tensor]
    y_context: list[torch.Tensor]
    cat_indices: list[list[int] | None] | list[list[list[int] | None]]
    configs: list[Any]
    time_query_raw: torch.Tensor
    event_query_raw: torch.Tensor


def _take(obj: Any, idx: np.ndarray) -> Any:
    return obj.iloc[idx] if hasattr(obj, "iloc") else obj[idx]


def _shuffle_and_chunk_survival_data(
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


class _SurvivalDatasetCollection:
    def __init__(
        self,
        split_fn: Any,
        rng: np.random.Generator,
        dataset_configs: list[_SurvivalDatasetConfig],
        *,
        backbone_model_type: Literal["classifier", "regressor"],
    ) -> None:
        self.split_fn = split_fn
        self.rng = rng
        self.dataset_configs = dataset_configs
        self.backbone_model_type = backbone_model_type

    def __len__(self) -> int:
        return len(self.dataset_configs)

    def __getitem__(self, index: int) -> _SurvivalBatch:
        torch = _import_torch()
        from tabpfn.preprocessing import fit_preprocessing
        from tabpfn.preprocessing.datamodel import FeatureModality, FeatureSchema

        config = self.dataset_configs[index]
        (
            x_train_raw,
            x_test_raw,
            y_train_surrogate,
            _y_test_surrogate,
            time_train_raw,
            time_test_raw,
            event_train_raw,
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
        return _SurvivalBatch(
            X_context=X_trains_preprocessed,
            X_query=X_tests_preprocessed,
            y_context=y_trains_preprocessed,
            cat_indices=cat_indices,
            configs=list(configs),
            time_query_raw=torch.as_tensor(np.asarray(time_test_raw), dtype=torch.float32),
            event_query_raw=torch.as_tensor(np.asarray(event_test_raw).astype(bool), dtype=torch.bool),
        )


def _survival_meta_collator(batch: list[_SurvivalBatch], padding_val: float = 0.0) -> _SurvivalBatch:
    torch = _import_torch()
    from tabpfn.utils import pad_tensors

    assert len(batch) == 1, "Only batch_size=1 is currently supported for TabPFN survival fine-tuning."
    item = batch[0]
    num_estimators = len(item.X_context)

    def collate_list_field(field_name: str, labels: bool) -> list[torch.Tensor]:
        values = [getattr(item, field_name)[idx] for idx in range(num_estimators)]
        return [torch.stack(pad_tensors([value], padding_val=padding_val, labels=labels)) for value in values]

    return _SurvivalBatch(
        X_context=collate_list_field("X_context", labels=False),
        X_query=collate_list_field("X_query", labels=False),
        y_context=collate_list_field("y_context", labels=(item.y_context[0].ndim == 1)),
        cat_indices=[[cat_ix] for cat_ix in item.cat_indices],
        configs=[[cfg] for cfg in item.configs],
        time_query_raw=torch.stack([item.time_query_raw]),
        event_query_raw=torch.stack([item.event_query_raw]),
    )


class TabPFNSurvivalMethod(BaseSurvivalMethod):
    def __init__(
        self,
        n_estimators: int = 8,
        fit_mode: str = "fit_preprocessors",
        model_version: str = "auto",
        checkpoint_path: str | None = None,
        backbone_task: str = "classification_event",
        backbone_training: str = "frozen",
        aggregate_estimators: str = "mean",
        hidden_layers: str | list[int] = "128-64",
        activation: str = "relu",
        dropout: float = 0.1,
        optimizer: str = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 150,
        patience: int = 20,
        grad_clip_value: float | None = 1.0,
        n_estimators_finetune: int = 2,
        n_estimators_final_inference: int = 8,
        finetune_ctx_plus_query_samples: int = 10_000,
        finetune_ctx_query_split_ratio: float = 0.2,
        device: str = "auto",
        seed: int | None = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            fit_mode=fit_mode,
            model_version=model_version,
            checkpoint_path=checkpoint_path,
            backbone_task=backbone_task,
            backbone_training=backbone_training,
            aggregate_estimators=aggregate_estimators,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
            optimizer=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            grad_clip_value=grad_clip_value,
            n_estimators_finetune=n_estimators_finetune,
            n_estimators_final_inference=n_estimators_final_inference,
            finetune_ctx_plus_query_samples=finetune_ctx_plus_query_samples,
            finetune_ctx_query_split_ratio=finetune_ctx_query_split_ratio,
            device=device,
            seed=seed,
        )
        self.finetuned_estimator_: Any = None
        self.backbone = None
        self.head: Any | None = None
        self.device_: Any | None = None
        self.head_input_dim_: int | None = None
        self.baseline_event_times_: np.ndarray | None = None
        self.baseline_survival_: np.ndarray | None = None
        self._backbone_cls: Any = None
        self._train_surrogate_target_: np.ndarray | None = None

    def _resolve_device(self) -> torch.device:
        torch = _import_torch()
        raw_device = str(self.params["device"])
        if raw_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(raw_device)

    @staticmethod
    def _set_torch_seed(seed: int) -> None:
        torch = _import_torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _backbone_model_type(self) -> Literal["classifier", "regressor"]:
        return "classifier" if str(self.params["backbone_task"]) == "classification_event" else "regressor"

    def _build_surrogate_target(self, time_train: np.ndarray, event_train: np.ndarray) -> np.ndarray:
        if self._backbone_model_type() == "classifier":
            return np.asarray(event_train, dtype=np.int32)
        return np.log1p(np.asarray(time_train, dtype=np.float32))

    def _build_backbone(self, *, n_estimators: int, fit_mode: str) -> Any:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion

        backbone_cls = TabPFNClassifier if self._backbone_model_type() == "classifier" else TabPFNRegressor
        self._backbone_cls = backbone_cls
        base_kwargs = {
            "n_estimators": int(n_estimators),
            "fit_mode": fit_mode,
            "device": str(self.params["device"]),
            "random_state": self.params.get("seed"),
            "ignore_pretraining_limits": True,
            "differentiable_input": False,
        }

        checkpoint_path = self.params.get("checkpoint_path")
        if checkpoint_path:
            return backbone_cls(model_path=str(checkpoint_path), **base_kwargs)

        model_version = str(self.params["model_version"]).lower()
        if model_version in {"auto", "default"}:
            return backbone_cls(**base_kwargs)

        version_map = {
            "v2": ModelVersion.V2,
            "v2.5": ModelVersion.V2_5,
            "v2_5": ModelVersion.V2_5,
        }
        if model_version not in version_map:
            raise ValueError("model_version must be one of {'auto', 'v2', 'v2.5'}.")
        return backbone_cls.create_default_for_version(version=version_map[model_version], **base_kwargs)

    def _build_head(self, in_features: int) -> nn.Module:
        nn = _import_nn()
        hidden_layers = _parse_hidden_layers(self.params["hidden_layers"])
        activation = _activation_cls(str(self.params["activation"]))
        dropout = float(self.params["dropout"])

        layers: list[nn.Module] = []
        prev = in_features
        for width in hidden_layers:
            width_i = int(width)
            layers.append(nn.Linear(prev, width_i))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = width_i
        layers.append(nn.Linear(prev, 1, bias=False))
        return nn.Sequential(*layers)

    def _extract_batch_embeddings(self, X_query_batch: list[torch.Tensor]) -> torch.Tensor:
        torch = _import_torch()
        if self.finetuned_estimator_ is None:
            raise RuntimeError("Backbone estimator must be initialized before embedding extraction.")

        executor = self.finetuned_estimator_.executor_
        self.finetuned_estimator_.executor_.use_torch_inference_mode(use_inference=False)
        # TabPFN versions differ in how they store per-estimator schemas/configs.
        # Ensure schema/config lists can index all query estimator slots.
        target_estimators = len(X_query_batch)
        if hasattr(executor, "feature_schema_list"):
            feature_schema_list = getattr(executor, "feature_schema_list")
            if isinstance(feature_schema_list, list):
                for i, schema_item in enumerate(feature_schema_list):
                    if isinstance(schema_item, list) and len(schema_item) == 1 and target_estimators > 1:
                        feature_schema_list[i] = schema_item * target_estimators
        if hasattr(executor, "ensemble_configs"):
            ensemble_configs = getattr(executor, "ensemble_configs")
            if isinstance(ensemble_configs, list):
                for i, cfg_item in enumerate(ensemble_configs):
                    if isinstance(cfg_item, list) and len(cfg_item) == 1 and target_estimators > 1:
                        ensemble_configs[i] = cfg_item * target_estimators
        embeddings: list[torch.Tensor] = []
        iterator = None
        try:
            iterator = executor.iter_outputs(
                X_query_batch,
                autocast=self.finetuned_estimator_.use_autocast_,
                only_return_standard_out=False,
            )
        except TypeError as exc:
            if "only_return_standard_out" not in str(exc):
                raise
            iterator = executor.iter_outputs(
                X_query_batch,
                autocast=self.finetuned_estimator_.use_autocast_,
            )

        for output, _config in iterator:
            if isinstance(output, dict):
                embed = output["test_embeddings"]
            else:
                embed = output
            if not isinstance(embed, torch.Tensor):
                embed = torch.as_tensor(embed, dtype=torch.float32, device=self.device_)
            if embed.ndim >= 3 and embed.shape[1] == 1:
                embed = embed.squeeze(1)
            embeddings.append(embed)

        stacked = torch.stack(embeddings, dim=0)
        if str(self.params["aggregate_estimators"]) == "concat":
            return stacked.permute(1, 0, 2).reshape(stacked.shape[1], -1)
        return stacked.mean(dim=0)

    def _extract_inference_embeddings(self, X: np.ndarray) -> np.ndarray:
        if self.backbone is None:
            raise RuntimeError("Backbone must be fit before extracting embeddings.")
        embeddings = self.backbone.get_embeddings(np.asarray(X, dtype=np.float32), data_source="test")
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 2:
            embeddings = embeddings[:, None, :]
        if str(self.params["aggregate_estimators"]) == "concat":
            features = embeddings.transpose(1, 0, 2).reshape(embeddings.shape[1], -1)
        else:
            features = embeddings.mean(axis=0)
        if self.head_input_dim_ is not None:
            features = self._align_embedding_dim(features, int(self.head_input_dim_))
        return features

    @staticmethod
    def _align_embedding_dim(embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        if embeddings.shape[1] == target_dim:
            return embeddings
        if embeddings.shape[1] > target_dim:
            return embeddings[:, :target_dim]
        pad_width = target_dim - embeddings.shape[1]
        padding = np.zeros((embeddings.shape[0], pad_width), dtype=embeddings.dtype)
        return np.concatenate([embeddings, padding], axis=1)

    @staticmethod
    def _cox_loss(log_risk: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return _import_cox_loss()(log_risk, event, time)

    def _fit_baseline_survival(
        self,
        time_train: np.ndarray,
        event_train: np.ndarray,
        train_log_risk: np.ndarray,
    ) -> None:
        event_mask = event_train.astype(bool)
        event_times = np.unique(time_train[event_mask])
        if event_times.size == 0:
            self.baseline_event_times_ = np.asarray([1.0], dtype=np.float64)
            self.baseline_survival_ = np.asarray([1.0], dtype=np.float64)
            return

        exp_risk = np.exp(train_log_risk.astype(np.float64))
        hazards: list[float] = []
        for event_time in event_times:
            d_j = float(np.sum((time_train == event_time) & event_mask))
            r_j = float(np.sum(exp_risk[time_train >= event_time]))
            hazards.append(d_j / max(r_j, 1e-12))

        cumulative_hazard = np.cumsum(np.asarray(hazards, dtype=np.float64))
        self.baseline_event_times_ = event_times.astype(np.float64)
        self.baseline_survival_ = np.exp(-cumulative_hazard).astype(np.float64)

    def _build_training_dataset(
        self,
        X_train: Any,
        surrogate_y_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        *,
        epoch: int,
    ) -> _SurvivalDatasetCollection:
        from tabpfn.utils import infer_random_state
        from tabpfn.preprocessing.datamodel import FeatureModality

        if not hasattr(self.finetuned_estimator_, "models_") or self.finetuned_estimator_.models_ is None:
            _, rng = self.finetuned_estimator_._initialize_model_variables()
        else:
            _, rng = infer_random_state(self.finetuned_estimator_.random_state)

        X_parts, surrogate_parts, time_parts, event_parts = _shuffle_and_chunk_survival_data(
            X_train,
            surrogate_y_train,
            time_train,
            event_train,
            max_chunk_size=min(int(self.params["finetune_ctx_plus_query_samples"]), len(time_train)),
            seed=int(self.params["seed"] or 0) + epoch,
        )

        dataset_configs: list[_SurvivalDatasetConfig] = []
        for X_part, surrogate_part, time_part, event_part in zip(X_parts, surrogate_parts, time_parts, event_parts):
            if self._backbone_model_type() == "classifier":
                ensemble_configs, X_mod, y_mod = self.finetuned_estimator_._initialize_dataset_preprocessing(
                    X_part, surrogate_part, rng
                )
            else:
                ensemble_configs, X_mod, y_mod, _bardist = self.finetuned_estimator_._initialize_dataset_preprocessing(
                    X_part, surrogate_part, rng
                )
            current_cat_ix = self.finetuned_estimator_.inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL)
            dataset_configs.append(
                _SurvivalDatasetConfig(
                    config=ensemble_configs,
                    X_raw=X_mod,
                    surrogate_y_raw=y_mod,
                    time_raw=np.asarray(time_part, dtype=np.float64),
                    event_raw=np.asarray(event_part, dtype=np.int32),
                    cat_ix=current_cat_ix,
                )
            )

        query_size = max(2, int(len(time_train) * float(self.params["finetune_ctx_query_split_ratio"])))
        query_size = min(query_size, max(2, min(int(self.params["finetune_ctx_plus_query_samples"]) - 1, len(time_train) - 1)))
        training_splitter = partial(
            train_test_split,
            test_size=query_size,
            random_state=int(self.params["seed"] or 0) + epoch,
        )
        return _SurvivalDatasetCollection(
            split_fn=training_splitter,
            rng=rng,
            dataset_configs=dataset_configs,
            backbone_model_type=self._backbone_model_type(),
        )

    def _build_evaluation_backbone(self) -> Any:
        try:
            from tabpfn.finetune_utils import clone_model_for_evaluation
        except ModuleNotFoundError:
            return self._build_backbone(
                n_estimators=int(self.params["n_estimators_final_inference"]),
                fit_mode=str(self.params["fit_mode"]),
            )

        return clone_model_for_evaluation(
            self.finetuned_estimator_,
            {
                "n_estimators": int(self.params["n_estimators_final_inference"]),
                "device": str(self.params["device"]),
                "random_state": self.params.get("seed"),
                "fit_mode": str(self.params["fit_mode"]),
                "ignore_pretraining_limits": True,
            },
            self._backbone_cls,
        )

    def _evaluate_validation_loss(
        self,
        X_train: Any,
        surrogate_y_train: np.ndarray,
        X_val: Any,
        time_val: np.ndarray,
        event_val: np.ndarray,
    ) -> float:
        torch = _import_torch()
        if self.head is None:
            raise RuntimeError("Survival head must be initialized before validation.")

        eval_backbone = self._build_evaluation_backbone()
        eval_backbone.fit(X_train, surrogate_y_train)
        embeddings_np = eval_backbone.get_embeddings(np.asarray(X_val, dtype=np.float32), data_source="test")
        embeddings_np = np.asarray(embeddings_np, dtype=np.float32)
        if embeddings_np.ndim == 2:
            embeddings_np = embeddings_np[:, None, :]
        if str(self.params["aggregate_estimators"]) == "concat":
            embeddings_np = embeddings_np.transpose(1, 0, 2).reshape(embeddings_np.shape[1], -1)
        else:
            embeddings_np = embeddings_np.mean(axis=0)
        if self.head_input_dim_ is not None:
            embeddings_np = self._align_embedding_dim(embeddings_np, int(self.head_input_dim_))

        self.head.eval()
        embeddings = torch.as_tensor(embeddings_np, dtype=torch.float32, device=self.device_)
        val_time = torch.as_tensor(np.asarray(time_val, dtype=np.float32), dtype=torch.float32, device=self.device_)
        val_event = torch.as_tensor(np.asarray(event_val).astype(bool), dtype=torch.bool, device=self.device_)
        with torch.no_grad():
            log_risk = self.head(embeddings).squeeze(-1)
            if int(val_event.sum().item()) <= 0:
                return float("inf")
            return float(self._cox_loss(log_risk, val_event, val_time).item())

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "TabPFNSurvivalMethod":
        ensure_foundation_runtime_ready("tabpfn_survival", checkpoint_path=self.params.get("checkpoint_path"))
        try:
            torch = _import_torch()
            self.device_ = self._resolve_device()
            self._set_torch_seed(int(self.params["seed"] or 0))

            X_train_np = np.asarray(X_train, dtype=np.float32)
            time_train_np = np.asarray(time_train, dtype=np.float64)
            event_train_np = np.asarray(event_train, dtype=np.int32)
            if int(event_train_np.sum()) <= 0:
                raise ValueError("TabPFN survival training requires at least one observed event in the training data.")

            surrogate_y_train = self._build_surrogate_target(time_train_np, event_train_np)
            self._train_surrogate_target_ = np.asarray(surrogate_y_train)
            self.finetuned_estimator_ = self._build_backbone(
                n_estimators=int(self.params["n_estimators_finetune"]),
                fit_mode="batched",
            )
            self.finetuned_estimator_._initialize_model_variables()
            self.finetuned_estimator_.model_.to(self.device_)

            best_backbone = None
            best_head_state = None
            best_monitor = float("inf")
            stale_epochs = 0
            optimizer = None

            for epoch in range(int(self.params["max_epochs"])):
                training_datasets = self._build_training_dataset(
                    X_train_np,
                    surrogate_y_train,
                    time_train_np,
                    event_train_np,
                    epoch=epoch,
                )
                finetuning_dataloader = torch.utils.data.DataLoader(
                    training_datasets,
                    batch_size=1,
                    collate_fn=_survival_meta_collator,
                    shuffle=True,
                    generator=torch.Generator().manual_seed(int(self.params["seed"] or 0) + epoch),
                )

                epoch_loss_sum = 0.0
                epoch_batches = 0
                for batch in finetuning_dataloader:
                    if self._backbone_model_type() == "classifier":
                        context_labels = torch.unique(torch.cat([labels.reshape(-1) for labels in batch.y_context]))
                        query_labels = torch.unique(batch.event_query_raw.reshape(-1).long())
                        if not bool(torch.isin(query_labels, context_labels, assume_unique=True).all()):
                            continue

                    self.finetuned_estimator_.fit_from_preprocessed(
                        batch.X_context,
                        batch.y_context,
                        batch.cat_indices,
                        batch.configs,
                    )
                    embeddings = self._extract_batch_embeddings(batch.X_query)
                    if self.head is None:
                        self.head_input_dim_ = int(embeddings.shape[1])
                        self.head = self._build_head(self.head_input_dim_).to(self.device_)
                        params = list(self.head.parameters())
                        if str(self.params["backbone_training"]).lower() == "finetune":
                            params.extend(list(self.finetuned_estimator_.model_.parameters()))
                        opt_name = str(self.params["optimizer"]).lower()
                        optimizer_cls = torch.optim.AdamW if opt_name == "adamw" else torch.optim.Adam
                        optimizer = optimizer_cls(
                            params,
                            lr=float(self.params["lr"]),
                            weight_decay=float(self.params["weight_decay"]),
                        )
                    if optimizer is None or self.head is None:
                        raise RuntimeError("Optimizer and head must be initialized before training.")

                    optimizer.zero_grad(set_to_none=True)
                    log_risk = self.head(embeddings).squeeze(-1)
                    event_query = batch.event_query_raw.reshape(-1).to(self.device_)
                    time_query = batch.time_query_raw.reshape(-1).to(self.device_)
                    if int(event_query.sum().item()) <= 0:
                        continue
                    loss = self._cox_loss(log_risk, event_query, time_query)
                    loss.backward()
                    if self.params["grad_clip_value"] is not None:
                        _clip_grad_norm(self.head.parameters(), float(self.params["grad_clip_value"]))
                        if str(self.params["backbone_training"]).lower() == "finetune":
                            _clip_grad_norm(
                                self.finetuned_estimator_.model_.parameters(),
                                float(self.params["grad_clip_value"]),
                            )
                    optimizer.step()

                    epoch_loss_sum += float(loss.detach().item())
                    epoch_batches += 1

                if self.head is None:
                    raise RuntimeError("No valid survival batches were available for TabPFN survival training.")

                monitor = epoch_loss_sum / max(epoch_batches, 1)
                if X_val is not None and time_val is not None and event_val is not None:
                    monitor = self._evaluate_validation_loss(
                        X_train_np,
                        surrogate_y_train,
                        np.asarray(X_val, dtype=np.float32),
                        np.asarray(time_val, dtype=np.float64),
                        np.asarray(event_val, dtype=np.int32),
                    )

                if monitor + 1e-8 < best_monitor:
                    best_monitor = monitor
                    best_backbone = deepcopy(self.finetuned_estimator_)
                    best_head_state = deepcopy(self.head.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= int(self.params["patience"]):
                        break

            if best_backbone is not None:
                self.finetuned_estimator_ = best_backbone
            if best_head_state is not None and self.head is not None:
                self.head.load_state_dict(best_head_state)
            if self.head is None:
                raise RuntimeError("TabPFN survival head was not initialized.")
            self.backbone = self._build_evaluation_backbone()
            self.backbone.fit(X_train_np, surrogate_y_train)

            train_embeddings = self._extract_inference_embeddings(X_train_np)
            train_embeddings_t = torch.as_tensor(train_embeddings, dtype=torch.float32, device=self.device_)
            self.head.eval()
            with torch.no_grad():
                train_log_risk = self.head(train_embeddings_t).squeeze(-1).detach().cpu().numpy()
            self._fit_baseline_survival(time_train_np, event_train_np, train_log_risk)
            return self
        except Exception as exc:
            raise rewrite_foundation_runtime_error(
                "tabpfn_survival",
                exc,
                checkpoint_path=self.params.get("checkpoint_path"),
            ) from exc

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        torch = _import_torch()
        if self.backbone is None or self.head is None or self.device_ is None:
            raise RuntimeError("TabPFNSurvivalMethod must be fit before prediction.")
        embeddings = self._extract_inference_embeddings(np.asarray(X, dtype=np.float32))
        embeddings_t = torch.as_tensor(embeddings, dtype=torch.float32, device=self.device_)
        self.head.eval()
        with torch.no_grad():
            risk = self.head(embeddings_t).squeeze(-1).detach().cpu().numpy()
        return risk.astype(np.float64)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.baseline_event_times_ is None or self.baseline_survival_ is None:
            raise RuntimeError("TabPFNSurvivalMethod must be fit before prediction.")
        eval_times = np.asarray(times, dtype=np.float64).reshape(-1)
        risk_scores = self.predict_risk(X)
        rel_risk = np.exp(risk_scores)

        last_surv = float(self.baseline_survival_[-1]) if self.baseline_survival_.size else 1.0
        baseline_at_times = np.interp(
            eval_times,
            self.baseline_event_times_,
            self.baseline_survival_,
            left=1.0,
            right=last_surv,
        )
        survival = np.power(np.clip(baseline_at_times, 1e-8, 1.0)[None, :], rel_risk[:, None])
        return np.clip(survival, 1e-8, 1.0).astype(np.float64)
