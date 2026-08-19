import pandas as pd
import numpy as np

from src.preprocess import (
    split_features_target,
    build_preprocessor,
)


def make_sample_dataframe():
    return pd.DataFrame(
        {
            "Administrative": [1, 2, np.nan],
            "Administrative_Duration": [10.0, 20.0, 30.0],
            "Month": ["Nov", "Dec", None],
            "VisitorType": [
                "Returning_Visitor",
                "New_Visitor",
                "Returning_Visitor",
            ],
            "Weekend": [True, False, True],
            "Revenue": [True, False, True],
        }
    )


def test_split_features_target():
    df = make_sample_dataframe()

    X, y = split_features_target(
        df,
        "Revenue",
    )

    assert "Revenue" not in X.columns
    assert list(y) == [1, 0, 1]


def test_preprocessor_handles_missing_values():
    df = make_sample_dataframe()

    X, _ = split_features_target(
        df,
        "Revenue",
    )

    preprocessor = build_preprocessor(X)

    transformed = preprocessor.fit_transform(X)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    assert not np.isnan(transformed).any()


def test_preprocessor_encodes_categorical_features():
    df = make_sample_dataframe()

    X, _ = split_features_target(
        df,
        "Revenue",
    )

    preprocessor = build_preprocessor(X)

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[1] > X.shape[1]


def test_preprocessor_does_not_modify_original_dataframe():
    df = make_sample_dataframe()

    X, _ = split_features_target(
        df,
        "Revenue",
    )

    original = X.copy(deep=True)

    preprocessor = build_preprocessor(X)

    preprocessor.fit_transform(X)

    pd.testing.assert_frame_equal(
        X,
        original,
    )