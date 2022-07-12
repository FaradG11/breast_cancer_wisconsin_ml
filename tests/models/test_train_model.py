import os
import pickle
from typing import Tuple

import pandas as pd
import pytest
from py._path.local import LocalPath
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

from ml_breast_cancer.enities import TrainingParams
from ml_breast_cancer.enities.feature_params import FeatureParams
from ml_breast_cancer.features.build_features import make_features
from ml_breast_cancer.models.model_fit_predict import train_model, serialize_model


@pytest.fixture
def features_and_target(
    dataset: pd.DataFrame, fitted_transformer, feature_params: FeatureParams,
) -> Tuple[pd.DataFrame, pd.Series]:
    features, target = make_features(fitted_transformer, dataset, feature_params)
    return features, target


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(features, target, train_params=TrainingParams())
    assert isinstance(model, KNeighborsClassifier)
    assert model.predict(features).shape[0] == target.shape[0]


def test_serialize_model(tmpdir: LocalPath):
    expected_output = tmpdir.join("model.pkl")
    model = KNeighborsClassifier()
    real_output = serialize_model(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists
    with open(real_output, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, KNeighborsClassifier)
