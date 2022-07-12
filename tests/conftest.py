import os
from typing import List

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from ml_breast_cancer.data import read_data
from ml_breast_cancer.enities import FeatureParams
from ml_breast_cancer.features.build_features import column_transformer


@pytest.fixture()
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "iris.csv")


@pytest.fixture()
def target_col():
    return "species"


@pytest.fixture()
def categorical_features() -> List[str]:
    return ["some_feature"]


@pytest.fixture
def numerical_features() -> List[str]:
    return [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]


@pytest.fixture()
def features_to_drop() -> List[str]:
    return ["some_features"]


@pytest.fixture()
def fitted_transformer(
    dataset: pd.DataFrame, feature_params: FeatureParams
) -> ColumnTransformer:
    fitted_transformer = column_transformer(feature_params)
    fitted_transformer.fit(dataset)
    return fitted_transformer


@pytest.fixture
def feature_params(
    categorical_features: List[str],
    features_to_drop: List[str],
    numerical_features: List[str],
    target_col: str,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=features_to_drop,
        target_col=target_col,
        use_log_trick=True,
    )
    return params


@pytest.fixture()
def dataset(dataset_path: str) -> pd.DataFrame:
    data = read_data(dataset_path)
    return data
