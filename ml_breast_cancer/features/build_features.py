from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.impute._base import _BaseImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from ml_breast_cancer.enities.feature_params import FeatureParams


def target_encoding(
        df: pd.DataFrame,
    params: FeatureParams):
    le = LabelEncoder()
    le.fit(df[params.target_col])
    df[params.target_col] = le.transform(df[params.target_col])
    return df


def get_imputer(strategy: str) -> _BaseImputer:
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    return imputer

def get_categorical_imputer() -> _BaseImputer:
    return get_imputer(strategy="most_frequent")


def get_numerical_imputer() -> _BaseImputer:
    return get_imputer(strategy="mean")


def process_categorical_features(
    pipeline: Pipeline, categorical_df: pd.DataFrame
) -> pd.DataFrame:
    one_df = pd.DataFrame(
        pipeline.transform(categorical_df).toarray(),
        columns=pipeline["encoder"].get_feature_names(),
    )
    return one_df


def categorical_pipeline() -> Pipeline:
    # imputer = get_categorical_imputer()
    encoder = LabelEncoder()
    pipeline = Pipeline([("encoder", encoder)])
    return pipeline


def numerical_pipeline() -> Pipeline:
    imputer = get_numerical_imputer()
    return Pipeline([("imputer", imputer), ("scaler", StandardScaler())])


def make_features(
    transformer: ColumnTransformer,
    df: pd.DataFrame,
    params: FeatureParams,
    test_mode: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    ready_features_df = transformer.transform(df)
    if test_mode:
        return ready_features_df, None
    else:
        return extract_target(df, params, ready_features_df)


def column_transformer(params) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            ("numerical", numerical_pipeline(), params.numerical_features),
        ]
    )
    return transformer


def extract_target(
    df: pd.DataFrame, params: FeatureParams, ready_features_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    target = df[params.target_col]
    return ready_features_df, target
