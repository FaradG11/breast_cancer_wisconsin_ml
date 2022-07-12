import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.compose import ColumnTransformer

from ml_breast_cancer.data import split_train_val_data
from ml_breast_cancer.enities import SplittingParams
from ml_breast_cancer.enities.feature_params import FeatureParams
from ml_breast_cancer.features.build_features import make_features


def test_make_features(
    dataset: pd.DataFrame,
    fitted_transformer,
    feature_params: FeatureParams,
    dataset_path: str,
):
    features, target = make_features(fitted_transformer, dataset, feature_params)
    assert features.shape[1] > 2
    assert not pd.isnull(features).any().any()


def test_split_train_features(
    dataset: pd.DataFrame,
    fitted_transformer: ColumnTransformer,
    feature_params: FeatureParams,
    dataset_path: str,
):
    val_size = 0.2
    train_df, val_df = split_train_val_data(dataset, SplittingParams(val_size=val_size))

    train_features, _ = make_features(fitted_transformer, train_df, feature_params)
    val_features, _ = make_features(fitted_transformer, val_df, feature_params)

    assert train_features.shape[1] == val_features.shape[1]
    assert train_features.shape[0] + val_features.shape[0] == dataset.shape[0]
    assert dataset.shape[0] * val_size == pytest.approx(val_features.shape[0], 1)

