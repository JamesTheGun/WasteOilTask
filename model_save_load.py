import os
import json
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from model_constants import DEFAULT_MODEL_NAME
from model_utils import make_predictions_jsonable
from confidence_model import do_confidence


def _get_model_file_name(model_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "model_data", model_name)


def save_model(model, model_name=DEFAULT_MODEL_NAME):
    filename = _get_model_file_name(model_name)
    joblib.dump(model, filename)


def load_model(model_name=DEFAULT_MODEL_NAME):
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
    filepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model_data", predictions_filename
    )
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Predictions file not found: {filepath}")

    with open(filepath, "r") as f:
        return json.load(f)


def save_predictions(predictions: list[dict], predictions_filename: str):
    output_filepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model_data", predictions_filename
    )
    with open(output_filepath, "w") as f:
        json.dump(predictions, f, indent=4)


def make_and_save_predictions(
    X: pd.DataFrame,
    model_type: str,
    model_filename: str = DEFAULT_MODEL_NAME,
    predictions_filename: str = "predictions.json",
    X_not_encoded: pd.DataFrame = None,
):
    predictions = load_and_predict(model_filename, X)

    confidence_df = do_confidence(X, model_filename, model_type, visualise=False)

    predictions_dict = make_predictions_jsonable(
        X, predictions, X_not_encoded=X_not_encoded, confidence_df=confidence_df
    )
    save_predictions(predictions_dict, predictions_filename)
    return predictions
