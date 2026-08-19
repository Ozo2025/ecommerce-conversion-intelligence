import yaml
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from src.preprocess import (
    load_data,
    split_features_target,
    build_preprocessor,
)

from src.evaluate import (
    evaluate_model,
    print_metrics,
)


def load_config(path="configs/config.yaml"):
    """Load project configuration from YAML."""
    with open(path, "r") as file:
        return yaml.safe_load(file)


def build_model(config):
    """Build the model specified in config.yaml."""

    model_type = config["model"]["type"]
    random_state = config["data"]["random_state"]

    if model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=config["model"].get("max_iter", 1000),
            class_weight=config["model"].get("class_weight", "balanced"),
            random_state=random_state,
        )

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=config["model"].get("n_estimators", 200),
            max_depth=config["model"].get("max_depth", 10),
            class_weight=config["model"].get("class_weight", "balanced"),
            random_state=random_state,
            n_jobs=-1,
        )

    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=config["model"].get("n_estimators", 100),
            learning_rate=config["model"].get("learning_rate", 0.1),
            max_depth=config["model"].get("max_depth", 3),
            random_state=random_state,
        )

    raise ValueError(
        f"Unsupported model type: {model_type}"
    )


def train_model():
    """Train, evaluate, and track an ecommerce conversion model."""

    config = load_config()

    # Load dataset
    df = load_data(
        config["data"]["raw_path"]
    )

    X, y = split_features_target(
        df,
        config["data"]["target"]
    )

    print(f"Dataset shape: {df.shape}")
    print(f"Features: {X.shape}")
    print(f"Target: {y.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Preprocessing
    preprocessor = build_preprocessor(
        X_train
    )

    # Build selected model
    model = build_model(config)

    # Full ML pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    # Configure MLflow
    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    # Track experiment
    with mlflow.start_run():

        mlflow.log_params({
            **config["model"],
            "test_size": config["data"]["test_size"],
            "random_state": config["data"]["random_state"],
            "data_version": config["mlflow"]["data_version"],
        })

        # Train
        pipeline.fit(
            X_train,
            y_train
        )

        # Evaluate
        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test
        )

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log trained model
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=["numpy.dtype"],
        )

        # Print results
        print_metrics(
            metrics,
            config["model"]["type"]
        )


if __name__ == "__main__":
    train_model()