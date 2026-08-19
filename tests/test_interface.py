from src.app import (
    validate_features,
    predict_conversion,
)


def test_interface_parses_valid_features():
    features = {
        "Administrative": 2,
        "Administrative_Duration": 45.0,
        "Informational": 1,
        "Informational_Duration": 20.0,
        "ProductRelated": 25,
        "ProductRelated_Duration": 720.0,
        "BounceRates": 0.02,
        "ExitRates": 0.04,
        "PageValues": 35.5,
        "SpecialDay": 0.0,
        "Month": "Nov",
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 2,
        "VisitorType": "Returning_Visitor",
        "Weekend": True,
    }

    cleaned, missing, errors = validate_features(features)

    assert missing == []
    assert errors == []
    assert cleaned["ProductRelated"] == 25
    assert cleaned["Month"] == "Nov"
    assert cleaned["Weekend"] is True


def test_interface_handles_incomplete_input():
    incomplete_features = {
        "ProductRelated": 25,
        "Month": "Nov",
        "VisitorType": "Returning_Visitor",
    }

    result = predict_conversion(incomplete_features)

    assert result["success"] is False
    assert result["error_type"] == "missing_features"
    assert len(result["missing_features"]) > 0
    assert "Administrative" in result["missing_features"]