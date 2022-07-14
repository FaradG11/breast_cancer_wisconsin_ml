import json
import logging
import sys
import shutil
import os
from datetime import datetime

import click
from sklearn.pipeline import make_pipeline
from data import read_data, split_train_val_data, drop_columns
from enities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from features import make_features
from features.build_features import column_transformer, target_encoding
from models import (
    train_model,
    serialize_model,
    serialize_pipe,
    predict_model,
    evaluate_model,
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


def train_pipeline(training_pipeline_params: TrainingPipelineParams, config_path):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)

    # Кодировка целевого признака
    data = target_encoding(data, training_pipeline_params.feature_params)

    # Разделение данных для обучения и для валидации
    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")

    # Трансформирование признаков
    feature_transformer = column_transformer(training_pipeline_params.feature_params)
    feature_transformer.fit(train_df)
    logger.info(f"data.shape is {data.shape}")


    # Выделение колонки с целевым показателем
    train_features, train_target = make_features(
        feature_transformer,
        train_df,
        training_pipeline_params.feature_params,
        test_mode=False,
    )
    logger.info(f"train_features.shape is {train_features.shape}")

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    pipe = make_pipeline(feature_transformer, model)

    val_features, val_target = make_features(
        feature_transformer,
        val_df,
        training_pipeline_params.feature_params,
        test_mode=False,
    )

    logger.info(f"val_features.shape is {val_features.shape}")
    predicts = predict_model(
        model, val_features, training_pipeline_params.feature_params.use_log_trick,
    )

    metrics = evaluate_model(
        predicts,
        val_target,
    )

    vers_path = os.path.join(
        os.getcwd(),
        'models',
        'versions',
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
    )

    os.makedirs(vers_path, exist_ok=True)

    with open(os.path.join(vers_path,training_pipeline_params.metric_path), "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    path_to_model = serialize_pipe(pipe, training_pipeline_params.output_model_path)

    shutil.copy(training_pipeline_params.output_model_path, vers_path)
    shutil.copy(training_pipeline_params.input_data_path, vers_path)
    shutil.copy(config_path, vers_path)

    return path_to_model, metrics


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params, config_path)


if __name__ == "__main__":
    train_pipeline_command()


