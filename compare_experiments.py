import mlflow


EXPERIMENT_NAME = "ecommerce_conversion"
PRIMARY_METRIC = "f1"


def compare_experiments():
    """Compare MLflow runs and identify the best model."""

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(
            f"MLflow experiment '{EXPERIMENT_NAME}' was not found."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{PRIMARY_METRIC} DESC"],
    )

    if runs.empty:
        raise ValueError("No MLflow runs were found.")

    columns = [
        "run_id",
        "params.type",
        "params.n_estimators",
        "params.max_depth",
        "params.learning_rate",
        "metrics.accuracy",
        "metrics.precision",
        "metrics.recall",
        "metrics.f1",
        "metrics.roc_auc",
        "status",
    ]

    available_columns = [
        column for column in columns
        if column in runs.columns
    ]

    results = runs[available_columns].copy()

    print("\nMLflow Experiment Comparison")
    print("=" * 100)
    print(results.to_string(index=False))

    successful_runs = runs[
        runs["status"] == "FINISHED"
    ].copy()

    if successful_runs.empty:
        raise ValueError("No successful MLflow runs were found.")

    successful_runs = successful_runs.sort_values(
        by=f"metrics.{PRIMARY_METRIC}",
        ascending=False,
    )

    best_run = successful_runs.iloc[0]

    print("\nBest Model")
    print("=" * 50)
    print(f"Run ID:     {best_run['run_id']}")
    print(f"Model:      {best_run.get('params.type', 'Unknown')}")
    print(f"F1 Score:   {best_run[f'metrics.{PRIMARY_METRIC}']:.4f}")
    print(f"ROC-AUC:    {best_run['metrics.roc_auc']:.4f}")
    print(f"Accuracy:   {best_run['metrics.accuracy']:.4f}")
    print(f"Precision:  {best_run['metrics.precision']:.4f}")
    print(f"Recall:     {best_run['metrics.recall']:.4f}")


if __name__ == "__main__":
    compare_experiments()