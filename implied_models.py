"""Implied models: derive one target from another model's predictions.

- Implied ratio from volume: volume_pred / supplied_m3
- Implied volume from ratio: ratio_pred * supplied_m3
"""

import numpy as np
import pandas as pd
from typing import Any, Tuple

from visualisation import visualize_model_predictions
from data_managment import (
    get_schedule_model_features,
    load_train_test_sets_target_recovery_ratio_with_supplied_volume,
    load_train_test_sets_target_recovery_volume,
    load_train_test_sets_target_recovery_ratio,
    train_test_time_series_split,
    load_encoder,
    encode_with_encoder,
)
from model_constants import (
    VOLUME_MODEL_FILENAME,
    RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME,
    RATIO_WITHOUT_VOL_MODEL_FILENAME,
    RESULTS_VOLUME_MODEL_FILENAME,
    RESULTS_RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME,
    RESULTS_RATIO_MODEL_FILENAME,
    IMPLIED_RECOVERY_RATIO_PREDICTIONS_FILENAME,
    IMPLIED_RECOVERY_VOLUME_PREDICTIONS_FILENAME,
)

from model_save_load import (
    load_model,
    load_and_predict,
    retrieve_predictions,
    save_predictions,
)
from model_utils import make_predictions_jsonable, print_model_results
from models import _get_results


def _implied_ratio_from_volume_test_results(
    model,
) -> Tuple[dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame]:
    """Evaluate implied ratios (volume_pred / supplied_m3) on the test set.

    Returns (results, actual_ratios, predicted_ratios, X_test).
    """
    X, y, not_selected = load_train_test_sets_target_recovery_volume()
    X_train, X_test, y_train, y_test, _, _ = train_test_time_series_split(
        X, y, not_selected
    )

    # Encode features using the saved encoder from training
    encoder = load_encoder()
    X_test_encoded = encode_with_encoder(X_test, encoder)

    volume_predictions = load_and_predict(VOLUME_MODEL_FILENAME, X_test_encoded)
    actual_ratios = y_test / X_test["supplied_m3"]
    predicted_ratios = volume_predictions / X_test["supplied_m3"]

    results = _get_results(
        model=model,
        y_test=actual_ratios,
        y_test_pred=predicted_ratios,
        type="ratio",
        supplied_m3=X_test["supplied_m3"],
        feature_names=X_train.columns.tolist(),
    )

    return results, actual_ratios, predicted_ratios, X_test_encoded


def _implied_volume_from_ratio_test_results(
    model,
    model_filename: str,
) -> Tuple[dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame]:
    """Evaluate implied volumes (ratio_pred * supplied_m3) on the test set.

    Uses ratio_with_supplied_volume data so X_test includes supplied_m3.
    Returns (results, actual_volumes, predicted_volumes, X_test).
    """
    X, y, not_selected = (
        load_train_test_sets_target_recovery_ratio_with_supplied_volume()
    )
    X_train, X_test, y_train, y_test, _, not_selected_test = (
        train_test_time_series_split(X, y, not_selected)
    )

    # Encode features using the saved encoder from training
    encoder = load_encoder()
    X_test_encoded = encode_with_encoder(X_test, encoder)

    supplied_m3 = X_test["supplied_m3"]
    ratio_predictions = load_and_predict(model_filename, X_test_encoded)
    actual_volumes = y_test * supplied_m3
    predicted_volumes = ratio_predictions * supplied_m3

    results = _get_results(
        model=model,
        y_test=actual_volumes,
        y_test_pred=predicted_volumes,
        type="volume",
        feature_names=X_train.columns.tolist(),
    )

    return results, actual_volumes, predicted_volumes, X_test_encoded


def do_implied_ratio_model(
    visualise_model=False,
):
    """Implied ratio model: use volume model predictions / supplied_m3."""
    try:
        model = load_model(VOLUME_MODEL_FILENAME)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Implied ratio model requires the volume model. "
            f"Run do_recovery_volume_model() first to save {VOLUME_MODEL_FILENAME}."
        )

    X, X_not_encoded = get_schedule_model_features(
        m_type="volume"
    )  # features for volume model we are calling...

    results, actual_ratios, predicted_ratios, X_test = (
        _implied_ratio_from_volume_test_results(model)
    )

    # Derive implied ratios for scheduled data
    supplied_volume = X["supplied_m3"].values
    volume_predictions = retrieve_predictions(RESULTS_VOLUME_MODEL_FILENAME)
    volume_predictions = np.array([p["predicted_recovery"] for p in volume_predictions])
    implied_ratios = volume_predictions / supplied_volume

    predictions_jsonable = make_predictions_jsonable(
        X,
        implied_ratios,
        X_not_encoded=X_not_encoded,
    )
    save_predictions(predictions_jsonable, IMPLIED_RECOVERY_RATIO_PREDICTIONS_FILENAME)

    print_model_results(results, model_name="Implied Ratio Model (from Volume)")

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


def do_implied_volume_model(
    visualise_model=False,
):
    """Implied volume model: use ratio model predictions * supplied_m3."""
    try:
        model = load_model(RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Implied volume model requires the ratio model. "
            f"Run do_recovery_ratio_model() first to save {RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME}."
        )

    X, X_not_encoded = get_schedule_model_features(
        m_type="ratio_with_supplied_volume"
    )  # features for ratio model we are calling... change back to straight ratio features if we want to use the ratio model without volume

    results, actual_volumes, predicted_volumes, X_test = (
        _implied_volume_from_ratio_test_results(
            model, RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME
        )
    )

    # Derive implied volumes for scheduled data
    supplied_volume = X["supplied_m3"].values
    ratio_predictions = retrieve_predictions(
        RESULTS_RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME
    )
    ratio_predictions = np.array([p["predicted_recovery"] for p in ratio_predictions])
    implied_volumes = ratio_predictions * supplied_volume

    predictions_jsonable = make_predictions_jsonable(
        X,
        implied_volumes,
        X_not_encoded=X_not_encoded,
    )
    save_predictions(predictions_jsonable, IMPLIED_RECOVERY_VOLUME_PREDICTIONS_FILENAME)

    print_model_results(results, model_name="Implied Volume Model (from Ratio)")

    if visualise_model:
        visualize_model_predictions(
            y_true=actual_volumes,
            y_pred=predicted_volumes,
            feature_importance=results["feature_importance"],
            X_features=X_test,
            target_name="implied_recovery_volume",
            model=model,
            image_filename="implied_recovery_volume_model.png",
        )
