# -*- coding: utf-8 -*-
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from enities import SplittingParams, FeatureParams


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_train_val_data(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    :rtype: object
    """

    col = params.stratify
    if col:
        target = data[col]
    else:
        target = None
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state, stratify=target
    )
    return train_data, val_data


def drop_columns(df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    print("Columns_to_drop:   ", params.features_to_drop)
    updated_data = df.drop(params.features_to_drop, axis=1)
    return updated_data
