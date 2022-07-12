import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ml_breast_cancer.enities.train_params import TrainingParams

SklearnClassifierModel = Union[KNeighborsClassifier,LogisticRegression]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnClassifierModel:
    if train_params.model_type == "KNeighborsClassifier":
        model = KNeighborsClassifier(
            n_neighbors=5
        )
    elif train_params.model_type == "LinearRegression":
        model = LinearRegression()
    model.fit(features, target)
    return model


def predict_model(
    model: SklearnClassifierModel, features: pd.DataFrame, use_log_trick: bool = True
) -> np.ndarray:
    predicts = model.predict(features)
    if use_log_trick:
        predicts = np.exp(predicts)
    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series, use_log_trick: bool = False
) -> Dict[str, float]:
    if use_log_trick:
        target = np.exp(target)
    return {
        "accuracy": accuracy_score(predicts, target),
        "precision": precision_score(predicts, target),
        "recall": recall_score(predicts, target),
        "f1": f1_score(predicts, target)
    }


def serialize_model(model: SklearnClassifierModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
