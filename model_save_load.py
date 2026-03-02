import os
import json
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from confidence_model import ConfidenceResult, make_confidence_model
from model_constants import DEFAULT_MODEL_NAME
from model_utils import make_predictions_jsonable


def _get_model_file_name(model_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "model_data", model_name)


def _get_predictions_file_name(predictions_filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    predictions_dir = os.path.join(current_dir, "model_data", "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    return os.path.join(predictions_dir, predictions_filename)


def save_confidence_model(confidence_result: ConfidenceResult, model_name: str) -> None:
    filename = _get_model_file_name(model_name)
    joblib.dump(confidence_result, filename)


def load_confidence_model(model_name: str) -> ConfidenceResult:
    filename = _get_model_file_name(model_name)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Confidence model file not found: {filename}")
    return joblib.load(filename)


def save_model(model, model_name=DEFAULT_MODEL_NAME):
    filename = _get_model_file_name(model_name)
    joblib.dump(model, filename)


def load_model(model_name=DEFAULT_MODEL_NAME) -> lgb.LGBMRegressor:
    filename = _get_model_file_name(model_name)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file not found: {filename}")
    return joblib.load(filename)


def load_and_predict(model_filename: str, X: pd.DataFrame = None) -> np.ndarray:
    model = load_model(model_filename)
    assert (
        type(model) == lgb.LGBMRegressor
    ), f"Loaded model is not a LightGBM regressor: {type(model)}"
    return model.predict(X)


def retrieve_predictions(predictions_filename: str) -> list[dict]:
    filepath = _get_predictions_file_name(predictions_filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Predictions file not found: {filepath}")

    with open(filepath, "r") as f:
        return json.load(f)


def _validate_predictions(predictions: list[dict]) -> None:
    for i, record in enumerate(predictions):
        assert isinstance(
            record.get("id"), int
        ), f"record[{i}]['id'] must be int, got {type(record.get('id')).__name__}"
        assert isinstance(
            record.get("predicted_recovery"), float
        ), f"record[{i}]['predicted_recovery'] must be float, got {type(record.get('predicted_recovery')).__name__}"
        pfv = record.get("prediction_feature_values")
        assert isinstance(
            pfv, dict
        ), f"record[{i}]['prediction_feature_values'] must be dict"
        assert isinstance(
            pfv.get("facility"), str
        ), f"record[{i}]['prediction_feature_values']['facility'] must be str, got {type(pfv.get('facility')).__name__}"
        assert isinstance(
            pfv.get("time_start_hour_minute"), float
        ), f"record[{i}]['prediction_feature_values']['time_start_hour_minute'] must be float, got {type(pfv.get('time_start_hour_minute')).__name__}"
        assert isinstance(
            pfv.get("supplier"), str
        ), f"record[{i}]['prediction_feature_values']['supplier'] must be str, got {type(pfv.get('supplier')).__name__}"
        if "supplied_m3" in pfv:
            assert isinstance(
                pfv["supplied_m3"], float
            ), f"record[{i}]['prediction_feature_values']['supplied_m3'] must be float, got {type(pfv['supplied_m3']).__name__}"
        if "confidence" in record:
            conf = record["confidence"]
            assert isinstance(conf, dict), f"record[{i}]['confidence'] must be dict"
            assert isinstance(
                conf.get("cluster"), int
            ), f"record[{i}]['confidence']['cluster'] must be int"
            for conf_key in ("lower", "center", "upper"):
                assert isinstance(
                    conf.get(conf_key), float
                ), f"record[{i}]['confidence']['{conf_key}'] must be float"
            tpct = conf.get("threshold_pct_sd")
            assert tpct is None or isinstance(
                tpct, float
            ), f"record[{i}]['confidence']['threshold_pct_sd'] must be float or null"
            tabs = conf.get("threshold_abs")
            assert tabs is None or isinstance(
                tabs, float
            ), f"record[{i}]['confidence']['threshold_abs'] must be float or null"


def save_predictions(predictions: list[dict], predictions_filename: str):
    _validate_predictions(predictions)
    output_filepath = _get_predictions_file_name(predictions_filename)
    with open(output_filepath, "w") as f:
        json.dump(predictions, f, indent=4)


def make_and_save_predictions(
    X: pd.DataFrame,
    model_type: str,
    model_filename: str = DEFAULT_MODEL_NAME,
    predictions_filename: str = "predictions.json",
    X_not_encoded: pd.DataFrame = None,
    confidence_model_filename: str = None,
):
    predictions = load_and_predict(model_filename, X)

    confidence_assignment = None
    if confidence_model_filename is not None:
        conf_model = load_confidence_model(confidence_model_filename)
        confidence_assignment = conf_model.apply_to(X)

    predictions_dict = make_predictions_jsonable(
        X,
        predictions,
        X_not_encoded=X_not_encoded,
        confidence_dict=confidence_assignment,
    )
    save_predictions(predictions_dict, predictions_filename)
    return predictions
