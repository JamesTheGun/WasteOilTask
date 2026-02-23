import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
from typing import Any, List, Tuple, Literal
import pandas as pd
import json
import os
import joblib

from visualisation import visualize_model_predictions
from data_managment import (
    get_schedule_model_features,
    load_train_test_sets_target_recovery_ratio,
    load_train_test_sets_target_recovery_volume,
    train_test_time_series_split,
    load_scheduled_data,
)


def train_lgbm_recovery_volume_model(
    visualse_model: bool,
) -> Tuple[dict, lgb.LGBMRegressor]:
    return _train_lgbm_recovery_model(visualse_model, type="volume")


def train_lgbm_recovery_ratio_model(
    visualse_model: bool,
) -> Tuple[dict, lgb.LGBMRegressor]:
    return _train_lgbm_recovery_model(visualse_model, type="ratio")


def _get_results(
    y_test,
    y_test_pred,
    type: Literal["ratio", "volume"],
    model=None,
    feature_names: List[str] = None,
    y_train=None,
    y_train_pred=None,
    supplied_m3=None,
):
    """Calculate model results. Train scores are optional."""
    results = {
        "test_r2": r2_score(y_test, y_test_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
    }

    if model is not None:
        results["model"] = model
        if feature_names is not None:
            results["feature_importance"] = dict(
                zip(feature_names, model.feature_importances_)
            )

    if y_train is not None and y_train_pred is not None:
        results["train_r2"] = r2_score(y_train, y_train_pred)
        results["train_mae"] = mean_absolute_error(y_train, y_train_pred)
        results["train_rmse"] = np.sqrt(mean_squared_error(y_train, y_train_pred))

    if type == "ratio" and supplied_m3 is not None:
        actual_recovery_volume = supplied_m3 * y_test
        implied_recovery_volume = supplied_m3 * y_test_pred
        results.update(
            {
                "r2_if_used_to_infer_m3": r2_score(
                    actual_recovery_volume, implied_recovery_volume
                ),
                "mae_if_used_to_infer_m3": mean_absolute_error(
                    actual_recovery_volume, implied_recovery_volume
                ),
                "rmse_if_used_to_infer_m3": np.sqrt(
                    mean_squared_error(actual_recovery_volume, implied_recovery_volume)
                ),
            }
        )

    return results


def _train_lgbm_recovery_model(
    visualse_model: bool, type: Literal["ratio", "volume"]
) -> Tuple[dict, lgb.LGBMRegressor]:

    np.random.seed(1)
    if type == "ratio":
        X, y, not_selected = load_train_test_sets_target_recovery_ratio()
    else:
        X, y, not_selected = load_train_test_sets_target_recovery_volume()

    X_train, X_test, y_train, y_test, not_selected_train, not_selected_test = (
        train_test_time_series_split(X, y, not_selected)
    )

    model = lgb.LGBMRegressor(
        n_estimators=200, num_leaves=31, learning_rate=0.05, random_state=1, verbose=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(10)],
    )

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    supplied_m3 = not_selected_test["supplied_m3"] if type == "ratio" else None
    results = _get_results(
        y_test=y_test,
        y_test_pred=y_test_pred,
        type=type,
        model=model,
        feature_names=X_train.columns.tolist(),
        y_train=y_train,
        y_train_pred=y_train_pred,
        supplied_m3=supplied_m3,
    )

    if visualse_model:
        visualize_model_predictions(
            y_true=y_test,
            y_pred=y_test_pred,
            feature_importance=results["feature_importance"],
            X_features=X_test,
            target_name=f"recovery_{type}",
            model=model,
            image_filename=f"recovery_{type}_model.png",
        )

    return results, model


DEFAULT_MODEL_NAME = "lgbm_recovery_model.pkl"


def _get_model_file_name(model_name):
    import os

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


def make_predictions_jsonable(
    X: pd.DataFrame,
    predictions: np.ndarray,
    X_not_encoded: pd.DataFrame = None,
):
    display = X_not_encoded if X_not_encoded is not None else X
    features = display.columns.tolist()
    prediction_feature_values = [dict(zip(features, row)) for row in display.values]
    predictions_with_id = pd.DataFrame(
        {
            "id": X.index,
            "predicted_recovery": predictions,
            "prediction_feature_values": prediction_feature_values,
        }
    )
    return predictions_with_id.to_dict(orient="records")


def retrieve_predictions(predictions_filename: str) -> list[dict]:
    filepath = os.path.join(
        os.path.dirname(__file__), "model_data", predictions_filename
    )
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Predictions file not found: {filepath}")

    with open(filepath, "r") as f:
        combined = json.load(f)

    return combined


def save_predictions(predictions: list[dict], predictions_filename: str):
    output_filepath = os.path.join(
        os.path.dirname(__file__), "model_data", predictions_filename
    )
    with open(output_filepath, "w") as f:
        json.dump(predictions, f, indent=4)


def make_and_save_predictions(
    X: pd.DataFrame,
    model_filename: str = DEFAULT_MODEL_NAME,
    predictions_filename: str = "predictions.json",
    X_not_encoded: pd.DataFrame = None,
):
    predictions = load_and_predict(model_filename, X)
    predictions_dict = make_predictions_jsonable(
        X, predictions, X_not_encoded=X_not_encoded
    )
    save_predictions(predictions_dict, predictions_filename)
    return predictions


def print_model_results(results, model_name=None):
    if not model_name:
        model_name = "MODEL NOT NAMED"
    print(f"Model: {model_name}")
    for statistic in results.keys():
        if statistic not in ("model", "feature_importance"):
            print(f"{statistic}: {results[statistic]:.4f}")
    if "feature_importance" in results:
        sorted_features = sorted(
            results["feature_importance"].items(), key=lambda x: x[1], reverse=True
        )
        for feature, importance in sorted_features[:5]:
            print(f"  {feature}: {importance:.4f}")
    print("=" * 60)


def do_recovery_ratio_model(
    model_filename: str = "lgbm_recovery_ratio_model.pkl", visualse_model: bool = False
):
    ratio_results, model = train_lgbm_recovery_ratio_model(visualse_model)
    print_model_results(ratio_results, model_name="Recovery Ratio Model")
    save_model(model, model_filename)


def do_recovery_volume_model(
    model_filename: str = "lgbm_recovery_volume_model.pkl", visualse_model: bool = False
):
    volume_results, model = train_lgbm_recovery_volume_model(visualse_model)
    print_model_results(volume_results, model_name="Recovery Volume Model")
    save_model(model, model_filename)


RATIO_MODEL_FILENAME = "lgbm_recovery_ratio_model.pkl"
VOLUME_MODEL_FILENAME = "lgbm_recovery_volume_model.pkl"

RESULTS_RATIO_MODEL_FILENAME = "lgbm_recovery_ratio_model_results.json"
RESULTS_VOLUME_MODEL_FILENAME = "lgbm_recovery_volume_model_results.json"


def ImpliedModelResultsInferedRatioFromVolume(
    model,
) -> Tuple[dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame]:
    """Returns (results, actual_ratios, predicted_ratios, X_test) for visualization."""
    X, Y, not_selected = load_train_test_sets_target_recovery_volume()

    X_train, X_test, y_train, y_test, not_selected_train, not_selected_test = (
        train_test_time_series_split(X, Y, not_selected)
    )

    volume_predictions = load_and_predict(VOLUME_MODEL_FILENAME, X_test)
    actual_ratios = y_test / X_test["supplied_m3"]
    predicted_ratios = volume_predictions / X_test["supplied_m3"]

    model_results = _get_results(
        model=model,
        y_test=actual_ratios,
        y_test_pred=predicted_ratios,
        type="ratio",
        supplied_m3=X_test["supplied_m3"],
        feature_names=X_train.columns.tolist(),
    )

    return model_results, actual_ratios, predicted_ratios, X_test


def _do_implied_volume_recovery_model(
    X, X_not_encoded, predictions_volume, visualise_model=False
):
    try:
        model = load_model(VOLUME_MODEL_FILENAME)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"implied recovery ratio model relies on volume model existing. Please run do_recovery_volume_model() first to train and save the model to {VOLUME_MODEL_FILENAME}."
        )

    # Get test set results and data for visualization
    results, actual_ratios, predicted_ratios, X_test = (
        ImpliedModelResultsInferedRatioFromVolume(model)
    )

    # Save predictions for scheduled data
    supplied_volume = X["supplied_m3"].values
    volume_recovery_predictions = retrieve_predictions(RESULTS_VOLUME_MODEL_FILENAME)
    volume_recovery_predictions = np.array(
        [pred["predicted_recovery"] for pred in volume_recovery_predictions]
    )
    implied_ratios_for_schedule = volume_recovery_predictions / supplied_volume
    implied_recovery_predictions_jsonable = make_predictions_jsonable(
        X,
        implied_ratios_for_schedule,
        X_not_encoded=X_not_encoded,
    )

    save_predictions(
        implied_recovery_predictions_jsonable,
        "implied_recovery_ratio_predictions.json",
    )

    print_model_results(results, model_name="Implied Ratio Model")

    if visualise_model:
        visualize_model_predictions(
            y_true=actual_ratios,
            y_pred=predicted_ratios,
            feature_importance=results["feature_importance"],
            X_features=X_test,
            target_name="implied_recovery_ratio",
            model=model,
            image_filename="implied_recovery_ratio_model.png",
        )


if __name__ == "__main__":
    do_recovery_ratio_model(RATIO_MODEL_FILENAME, visualse_model=True)
    do_recovery_volume_model(VOLUME_MODEL_FILENAME, visualse_model=True)

    X_ratio, X_ratio_not_encoded = get_schedule_model_features(m_type="ratio")

    predictions_ratio = make_and_save_predictions(
        X_ratio,
        model_filename=RATIO_MODEL_FILENAME,
        predictions_filename=RESULTS_RATIO_MODEL_FILENAME,
        X_not_encoded=X_ratio_not_encoded,
    )

    X, X_not_encoded = get_schedule_model_features(m_type="volume")

    predictions_volume = make_and_save_predictions(
        X,
        model_filename=VOLUME_MODEL_FILENAME,
        predictions_filename=RESULTS_VOLUME_MODEL_FILENAME,
        X_not_encoded=X_not_encoded,
    )

    _do_implied_volume_recovery_model(
        X, X_not_encoded, predictions_volume, visualise_model=True
    )
