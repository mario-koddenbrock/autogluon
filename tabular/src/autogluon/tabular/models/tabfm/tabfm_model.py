from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


# Activation-chunking thresholds (from TabArena PR #434).
# When n_rows × n_features exceeds the trigger, chunk sizes are applied to
# every network module that exposes row/col/ffn_chunk_size knobs.  This bounds
# peak VRAM without changing predictions (chunking is mathematically exact).
_CHUNK_TRIGGER_CELLS = 1_000_000
_CHUNK_ROW = 4_096
_CHUNK_COL = 64
_CHUNK_FFN = 1 << 18  # 262 144


def _memory_chunk_sizes(n_rows: int, n_features: int) -> "dict[str, int] | None":
    if n_rows * n_features <= _CHUNK_TRIGGER_CELLS:
        return None
    return {"row": _CHUNK_ROW, "col": _CHUNK_COL, "ffn": _CHUNK_FFN}


def _apply_chunk_sizes(network, chunk_sizes: "dict[str, int] | None") -> None:
    row = col = ffn = None
    if chunk_sizes is not None:
        row, col, ffn = chunk_sizes["row"], chunk_sizes["col"], chunk_sizes["ffn"]
    for module in network.modules():
        if hasattr(module, "row_chunk_size"):
            module.row_chunk_size = row
        if hasattr(module, "col_chunk_size"):
            module.col_chunk_size = col
        if hasattr(module, "ffn_chunk_size"):
            module.ffn_chunk_size = ffn


# TabFM: Google Research's tabular foundation model (in-context learning, like
# TabPFN / TabDPT). scikit-learn-compatible ``TabFMClassifier`` / ``TabFMRegressor``
# wrapping a pretrained checkpoint downloaded from the HuggingFace Hub.
#
# Repo:   https://github.com/google-research/tabfm
# Install (no PyPI): pip install -e .[pytorch]   (or .[jax]); we use the pytorch backend.
#
# API (verified against tabfm/src/classifier_and_regressor.py):
#   model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="classification"|"regression")
#   clf = TabFMClassifier(model=model, n_estimators=32, softmax_temperature=0.9,
#                         max_num_features=500, batch_size=1, random_state=...)
#   clf.fit(X, y); clf.predict_proba(X)          # classification
#   reg = TabFMRegressor(model=model, ...); reg.fit(X, y); reg.predict(X)  # regression
#
# Device: ``load(model_type=..., device=...)`` places the checkpoint on the target
# device; the sklearn wrappers infer their compute device from that model object.
class TabFMModel(AbstractTorchModel):
    ag_key = "TABFM"
    ag_name = "TabFM"
    seed_name = "random_state"
    ag_priority = 45
    default_random_seed = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._model_type = None  # "classification" | "regression"; set in _fit

    def _get_default_searchspace(self) -> dict:
        """HPO search space over TabFM's inference-time knobs.

        TabFM ships a single pretrained checkpoint (no tunable checkpoints), so
        HPO tunes inference behaviour: ensembling, softmax temperature, and the
        normalization method. ``softmax_temperature`` is classification-only and
        is dropped for regression in ``_get_tabfm_params``. First value of each
        Categorical is the model default.
        """
        from autogluon.common import space

        searchspace = super()._get_default_searchspace()
        searchspace.update(
            {
                "n_estimators": space.Categorical(32, 8, 16, 64),
                "softmax_temperature": space.Categorical(
                    0.9, 0.5, 0.7, 0.8, 1.0, 1.25
                ),
                "norm_methods": space.Categorical(None, "power", "quantile"),
            }
        )
        return searchspace

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        import torch

        if num_gpus != 0 and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        import tabfm

        is_classification = self.problem_type in [BINARY, MULTICLASS]
        model_type = "classification" if is_classification else "regression"
        self._model_type = model_type

        # Load the pretrained checkpoint (HF Hub) directly onto the target
        # device; TabFM's loader accepts a device= kwarg and the sklearn
        # wrappers infer their compute device from this model object.
        base_model = tabfm.tabfm_v1_0_0_pytorch.load(model_type=model_type, device=device)

        model_cls = tabfm.TabFMClassifier if is_classification else tabfm.TabFMRegressor
        params = self._get_tabfm_params()

        X = self.preprocess(X, y=y)
        import numpy as np
        y = y.to_numpy()
        # MPS does not support float64; cast regression targets to float32.
        if y.dtype == np.float64:
            y = y.astype(np.float32)
        self._chunk_sizes = _memory_chunk_sizes(n_rows=X.shape[0], n_features=X.shape[1])
        _apply_chunk_sizes(base_model, self._chunk_sizes)
        self.model = model_cls(model=base_model, **params)
        self.model.fit(X=X, y=y)

    def _get_tabfm_params(self) -> dict:
        """Collect TabFM constructor params from the model's hyperparameters.

        Keeps only args TabFM's constructors accept, maps the AG seed, and drops
        classification-only args for regression.
        """
        model_params = self._get_model_params()

        common = (
            "n_estimators",
            "norm_methods",
            "feat_shuffle_method",
            "permute_categorical",
            "outlier_threshold",
            "max_num_features",
            "max_num_rows",
            "use_amp",
            "batch_size",
            "cat_encoder_mode",
            "n_feature_crosses",
            "n_svd_features",
            "enable_nnls",
            "nnls_beta",
            self.seed_name,
        )
        clf_only = (
            "softmax_temperature",
            "average_logits",
            "class_shift",
            "binary_calibration_method",
            "multiclass_calibration_method",
            "calibration_lambda",
        )
        allowed = common + clf_only if self.problem_type in [BINARY, MULTICLASS] else common

        params = {k: v for k, v in model_params.items() if k in allowed}
        params.setdefault(self.seed_name, self.default_random_seed)
        # Foundation-model fork intent: don't cap features at 500 (Raman data is
        # already subsampled upstream). Callers can override via hyperparameters.
        params.setdefault("max_num_features", None)
        params.setdefault("verbose", False)
        return params

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)

        if self.problem_type in [REGRESSION]:
            return self.model.predict(X)

        import numpy as np

        y_pred_proba = np.asarray(self.model.predict_proba(X))
        return self._convert_proba_to_unified_form(y_pred_proba)

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """TabFM takes a numpy array; label-encode any non-numeric columns."""
        X = super()._preprocess(X, **kwargs)
        if self._feature_generator is None:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.to_numpy()

    def _inner_torch_model(self):
        """The underlying TabFM torch module, passed to the sklearn wrapper as
        ``model=`` and stored on it as ``.model``."""
        return getattr(self.model, "model", None)

    def get_device(self) -> str:
        import torch

        inner = self._inner_torch_model()
        if inner is not None:
            try:
                return str(next(inner.parameters()).device)
            except Exception:
                pass
        return "cpu" if not torch.cuda.is_available() else "cuda"

    def _set_device(self, device: str):
        """Move the underlying TabFM torch model onto ``device``.

        Required by ``AbstractTorchModel`` (its base ``_set_device`` raises
        NotImplementedError) — it is called during ``save()`` (move to CPU) and
        on load/predict. The sklearn wrapper reads its compute device from this
        model object, so moving it here is sufficient.
        """
        inner = self._inner_torch_model()
        if inner is not None and hasattr(inner, "to"):
            inner.to(device)

    def save(self, path: str = None, verbose: bool = True) -> str:
        """Persist without pickling the TabFM checkpoint.

        The pretrained TabFM torch module contains unpicklable objects (a
        ``get_activation`` lambda), so we detach it before AutoGluon pickles the
        model and restore it afterward. The checkpoint is reloaded from the HF
        cache in ``load()`` — it is identical across runs, so nothing is lost.
        """
        inner = self._inner_torch_model()
        if inner is not None:
            self.model.model = None
        try:
            return super().save(path=path, verbose=verbose)
        finally:
            if inner is not None:
                self.model.model = inner

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True):
        """Reload the model and re-attach the TabFM checkpoint dropped on save."""
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        wrapper = getattr(model, "model", None)
        if wrapper is not None and getattr(wrapper, "model", None) is None:
            import tabfm
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            model_type = getattr(model, "_model_type", None) or "classification"
            wrapper.model = tabfm.tabfm_v1_0_0_pytorch.load(
                model_type=model_type, device=device
            )
            _apply_chunk_sizes(wrapper.model, getattr(model, "_chunk_sizes", None))
        return model

    def _get_default_resources(self) -> tuple[int, int]:
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 0.5 if is_gpu_available else 0,
        }

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_rows": None,
                "max_features": None,
                "max_classes": None,
            }
        )
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        default_ag_args_ensemble.update(
            {
                # Foundation-model checkpoints aren't pre-downloaded on worker
                # forks; run folds sequentially to avoid a parallel download race.
                "fold_fitting_strategy": "sequential_local",
                "refit_folds": True,
            }
        )
        return default_ag_args_ensemble
