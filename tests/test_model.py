import pandas as pd

from src.app import load_best_model
from src.preprocess import (
    load_data,
    split_features_target,
)


def test_model_prediction_shape_and_type():
    model, _ = load_best_model()

    df = load_data(
        "data/online_shoppers_intention.csv"
    )

    X, _ = split_features_target(
        df,
        "Revenue",
    )

    sample = X.iloc[:5]

    predictions = model.predict(sample)

    assert len(predictions) == 5
    assert predictions.ndim == 1

    for prediction in predictions:
        assert int(prediction) in [0, 1]


def test_model_meets_minimum_performance():
    model, best_run = load_best_model()

    f1_score = float(
        best_run["metrics.f1"]
    )

    roc_auc = float(
        best_run["metrics.roc_auc"]
    )

    assert f1_score >= 0.60
    assert roc_auc >= 0.90