"""Tests for ManyClassClassifier integration in TabPFN models."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from autogluon.tabular.models.tabpfnv2.tabpfnv2_5_model import RealTabPFNv2Model, RealTabPFNv25Model


def _make_multiclass_data(n_classes: int, n_samples_per_class: int = 5):
    """Create a simple multiclass DataFrame with `n_classes` distinct labels."""
    X = pd.DataFrame({"f0": range(n_classes * n_samples_per_class), "f1": range(n_classes * n_samples_per_class)})
    y = pd.Series(np.repeat(np.arange(n_classes), n_samples_per_class), name="label")
    return X, y


def _make_mock_tabpfn_classifier(predict_proba_value=None):
    """Return a mock that behaves like a fitted TabPFNClassifier."""
    mock = MagicMock()
    mock.fit.return_value = mock
    if predict_proba_value is not None:
        mock.predict_proba.return_value = predict_proba_value
    return mock


@pytest.mark.parametrize("model_cls", [RealTabPFNv2Model, RealTabPFNv25Model])
def test_many_class_classifier_used_when_above_threshold(model_cls):
    """ManyClassClassifier must wrap the base model when num_classes > many_class_threshold."""
    n_classes = 15
    threshold = 10
    X, y = _make_multiclass_data(n_classes)

    mock_base = _make_mock_tabpfn_classifier()
    mock_many = MagicMock()
    mock_many.fit.return_value = mock_many

    with (
        patch("tabpfn.TabPFNClassifier", return_value=mock_base),
        patch("tabpfn_extensions.many_class.ManyClassClassifier", return_value=mock_many) as MockManyClass,
        patch("tabpfn.model.loading.resolve_model_path", return_value=(None, [None], None, None)),
        patch("torch.cuda.is_available", return_value=False),
    ):
        model = model_cls(
            path="./tmp_test/",
            name="test",
            problem_type="multiclass",
            eval_metric="accuracy",
            num_classes=n_classes,
            hyperparameters={"n_estimators": 1},
        )
        model.params_aux["many_class_threshold"] = threshold
        model._fit(X=X, y=y, num_cpus=1, num_gpus=0)

    MockManyClass.assert_called_once()
    mock_many.fit.assert_called_once()


@pytest.mark.parametrize("model_cls", [RealTabPFNv2Model, RealTabPFNv25Model])
def test_many_class_classifier_not_used_when_below_threshold(model_cls):
    """ManyClassClassifier must NOT be used when num_classes <= many_class_threshold."""
    n_classes = 5
    threshold = 10
    X, y = _make_multiclass_data(n_classes)

    mock_base = _make_mock_tabpfn_classifier()

    with (
        patch("tabpfn.TabPFNClassifier", return_value=mock_base),
        patch("tabpfn_extensions.many_class.ManyClassClassifier") as MockManyClass,
        patch("tabpfn.model.loading.resolve_model_path", return_value=(None, [None], None, None)),
        patch("torch.cuda.is_available", return_value=False),
    ):
        model = model_cls(
            path="./tmp_test/",
            name="test",
            problem_type="multiclass",
            eval_metric="accuracy",
            num_classes=n_classes,
            hyperparameters={"n_estimators": 1},
        )
        model.params_aux["many_class_threshold"] = threshold
        model._fit(X=X, y=y, num_cpus=1, num_gpus=0)

    MockManyClass.assert_not_called()
    mock_base.fit.assert_called_once()


@pytest.mark.parametrize("model_cls", [RealTabPFNv2Model, RealTabPFNv25Model])
def test_many_class_classifier_not_used_for_regression(model_cls):
    """ManyClassClassifier must never be used for regression tasks."""
    X = pd.DataFrame({"f0": range(20), "f1": range(20)})
    y = pd.Series(np.random.rand(20), name="label")

    mock_base = _make_mock_tabpfn_classifier()

    with (
        patch("tabpfn.TabPFNRegressor", return_value=mock_base),
        patch("tabpfn_extensions.many_class.ManyClassClassifier") as MockManyClass,
        patch("tabpfn.model.loading.resolve_model_path", return_value=(None, [None], None, None)),
        patch("torch.cuda.is_available", return_value=False),
    ):
        model = model_cls(
            path="./tmp_test/",
            name="test",
            problem_type="regression",
            eval_metric="rmse",
            hyperparameters={"n_estimators": 1},
        )
        model._fit(X=X, y=y, num_cpus=1, num_gpus=0)

    MockManyClass.assert_not_called()
    mock_base.fit.assert_called_once()


@pytest.mark.parametrize("model_cls", [RealTabPFNv2Model, RealTabPFNv25Model])
def test_many_class_fallback_when_extension_missing(model_cls):
    """When tabpfn-extensions is not installed, fall back to base model without raising."""
    n_classes = 15
    threshold = 10
    X, y = _make_multiclass_data(n_classes)

    mock_base = _make_mock_tabpfn_classifier()

    with (
        patch("tabpfn.TabPFNClassifier", return_value=mock_base),
        patch("tabpfn.model.loading.resolve_model_path", return_value=(None, [None], None, None)),
        patch("torch.cuda.is_available", return_value=False),
        patch.dict("sys.modules", {"tabpfn_extensions.many_class": None}),
    ):
        model = model_cls(
            path="./tmp_test/",
            name="test",
            problem_type="multiclass",
            eval_metric="accuracy",
            num_classes=n_classes,
            hyperparameters={"n_estimators": 1},
        )
        model.params_aux["many_class_threshold"] = threshold
        # Should not raise even though the extension is unavailable
        model._fit(X=X, y=y, num_cpus=1, num_gpus=0)

    assert model.model is mock_base
    mock_base.fit.assert_called_once()


def test_many_class_threshold_default_value():
    """Default many_class_threshold must be 10 (TabPFN's native class limit)."""
    model = RealTabPFNv25Model(
        path="./tmp_test/",
        name="test",
        problem_type="multiclass",
        eval_metric="accuracy",
    )
    assert model._get_default_auxiliary_params()["many_class_threshold"] == 10