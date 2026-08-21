import mlflow
import mlflow.sklearn
import pandas as pd


EXPERIMENT_NAME = "ecommerce_conversion"
PRIMARY_METRIC = "f1"


# ---------------------------------------------------------
# Feature schema
# ---------------------------------------------------------

FEATURE_SCHEMA = {
    "Administrative": int,
    "Administrative_Duration": float,
    "Informational": int,
    "Informational_Duration": float,
    "ProductRelated": int,
    "ProductRelated_Duration": float,
    "BounceRates": float,
    "ExitRates": float,
    "PageValues": float,
    "SpecialDay": float,
    "Month": str,
    "OperatingSystems": int,
    "Browser": int,
    "Region": int,
    "TrafficType": int,
    "VisitorType": str,
    "Weekend": bool,
}


# ---------------------------------------------------------
# MLflow model loading
# ---------------------------------------------------------

def load_best_model():
    """
    Load the highest-F1 completed MLflow model that can
    actually be loaded successfully.

    This makes the application resilient if a completed
    MLflow run exists but its model artifact is unavailable.
    """

    experiment = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        raise ValueError(
            f"MLflow experiment '{EXPERIMENT_NAME}' was not found."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{PRIMARY_METRIC} DESC"],
    )

    if runs.empty:
        raise ValueError(
            "No completed MLflow runs were found."
        )

    load_errors = []

    for _, run in runs.iterrows():
        run_id = run["run_id"]
        model_uri = f"runs:/{run_id}/model"

        try:
            model = mlflow.sklearn.load_model(
                model_uri
            )

            return model, run

        except Exception as error:
            load_errors.append(
                f"{run_id}: {error}"
            )

    raise ValueError(
        "No loadable MLflow model was found. "
        + " | ".join(load_errors)
    )


# ---------------------------------------------------------
# Feature validation
# ---------------------------------------------------------

def validate_features(features):
    """
    Validate that all required model features are present
    and can be converted to the expected data types.

    Returns:
        cleaned_features
        missing_features
        errors
    """

    missing_features = []
    errors = []
    cleaned_features = {}

    for feature_name, expected_type in FEATURE_SCHEMA.items():

        if feature_name not in features:
            missing_features.append(
                feature_name
            )
            continue

        value = features[feature_name]

        if value is None:
            missing_features.append(
                feature_name
            )
            continue

        try:

            if expected_type is bool:

                if isinstance(value, bool):
                    cleaned_value = value

                elif isinstance(value, str):
                    normalized = value.strip().lower()

                    if normalized in [
                        "true",
                        "yes",
                        "1",
                        "y",
                    ]:
                        cleaned_value = True

                    elif normalized in [
                        "false",
                        "no",
                        "0",
                        "n",
                    ]:
                        cleaned_value = False

                    else:
                        raise ValueError(
                            "must be true/false or yes/no"
                        )

                else:
                    cleaned_value = bool(
                        value
                    )

            else:
                cleaned_value = expected_type(
                    value
                )

            cleaned_features[
                feature_name
            ] = cleaned_value

        except (ValueError, TypeError):

            errors.append(
                f"{feature_name} has invalid value: {value}"
            )

    return (
        cleaned_features,
        missing_features,
        errors,
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict_conversion(
    features,
    model=None,
):
    """
    Validate ecommerce session features and run the
    trained conversion model.

    Returns a structured prediction result.
    """

    (
        cleaned_features,
        missing_features,
        errors,
    ) = validate_features(
        features
    )

    if missing_features:
        return {
            "success": False,
            "error_type": "missing_features",
            "missing_features": missing_features,
            "message": (
                "More information is required before "
                "a prediction can be made."
            ),
        }

    if errors:
        return {
            "success": False,
            "error_type": "invalid_features",
            "errors": errors,
            "message": (
                "Some feature values are invalid."
            ),
        }

    if model is None:
        model, _ = load_best_model()

    input_df = pd.DataFrame(
        [cleaned_features],
        columns=FEATURE_SCHEMA.keys(),
    )

    prediction = model.predict(
        input_df
    )[0]

    probability = model.predict_proba(
        input_df
    )[0][1]

    return {
        "success": True,
        "prediction": bool(
            prediction
        ),
        "conversion_probability": float(
            probability
        ),
        "conversion_percentage": float(
            probability * 100
        ),
    }


# ---------------------------------------------------------
# Optional direct test
# ---------------------------------------------------------

def test_model_loading():
    """
    Verify that a loadable MLflow model can be found.
    """

    model, run = load_best_model()

    print("\nBest Loadable Model")
    print("=" * 50)
    print(
        f"Run ID: {run['run_id']}"
    )
    print(
        f"Model: {run.get('params.type', 'Unknown')}"
    )
    print(
        f"F1 Score: {float(run['metrics.f1']):.4f}"
    )
    print(
        f"ROC-AUC: {float(run['metrics.roc_auc']):.4f}"
    )

    return model, run


if __name__ == "__main__":
    test_model_loading()