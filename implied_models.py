import numpy as np
import pandas as pd
from typing import Any, Literal, Tuple

from visualisation import visualize_model_predictions
from data_managment import (
    deterministic_encoded_train_test_split,
    get_schedule_model_features,
    load_train_test_sets_target_recovery_ratio_with_supplied_volume,
    load_train_test_sets_target_recovery_volume,
    load_train_test_sets_target_recovery_ratio_without_supplied_volume,
    train_test_time_series_split,
    load_encoder,
    encode_with_encoder,
)
from model_constants import (
    MODEL_TYPE_DISPLAY_NAMES,
    VOLUME_MODEL_FILENAME,
    RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME,
    RATIO_WITHOUT_SUPPLIED_VOLUME_MODEL_FILENAME,
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
from data_managment import deterministic_encoded_train_test_split
from confidence_model import make_confidence_model
from models import _get_results
from lightgbm import LGBMRegressor


def do_implied_ratio_model(
    visualise_model=False,
):
    try:
        model = load_model(VOLUME_MODEL_FILENAME)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Implied ratio model requires the volume model. "
            f"Run do_recovery_volume_model() first to save {VOLUME_MODEL_FILENAME}."
        )

    do_implied_model(
        visualise_model, model_type="ratio_without_supplied_volume", model=model
    )


def do_implied_volume_model_with_supplied_volume(
    visualise_model=False,
):
    try:
        model = load_model(RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Implied volume model requires the ratio with supplied volume model. "
            f"Run do_recovery_ratio_with_supplied_volume_model() first to save {RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME}."
        )
    do_implied_model(visualise_model, model_type="volume", model=model)


def do_implied_volume_model_without_supplied_volume(
    visualise_model=False,
):
    try:
        model = load_model(RATIO_WITHOUT_SUPPLIED_VOLUME_MODEL_FILENAME)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Implied volume model requires the ratio without supplied volume model. "
            f"Run do_recovery_ratio_without_supplied_volume_model() first to save {RATIO_WITHOUT_SUPPLIED_VOLUME_MODEL_FILENAME}."
        )
    do_implied_model(visualise_model, model_type="volume", model=model)


def get_implied_results_from_base_model(
    base_model: LGBMRegressor,
    base_model_type: str,
    X_test_base_model,
    y_test_base_model,
    not_selected_test,
) -> dict[str, Any]:
    if base_model_type == "volume":
        volume_predictions = base_model.predict(X_test_base_model)
        implied_ratios = volume_predictions / X_test_base_model["supplied_m3"].values
        true_ratios = y_test_base_model / X_test_base_model["supplied_m3"].values
        y_true = true_ratios
        y_pred = implied_ratios

    if base_model_type == "ratio_with_supplied_volume":
        ratio_predictions = base_model.predict(X_test_base_model)
        volume_predictions = ratio_predictions * X_test_base_model["supplied_m3"].values
        y_true = y_test_base_model * X_test_base_model["supplied_m3"].values
        y_pred = volume_predictions

    if base_model_type == "ratio_without_supplied_volume":
        ratio_predictions = base_model.predict(X_test_base_model)
        implied_volumes = ratio_predictions * not_selected_test["supplied_m3"].values
        true_volumes = y_test_base_model * not_selected_test["supplied_m3"].values
        y_true = true_volumes
        y_pred = implied_volumes

    return y_true, y_pred


def do_implied_model(
    visualise_model,
    base_model_type: Literal[
        "volume",
        "ratio_with_supplied_volume",
        "ratio_without_supplied_volume",
    ],
):
    if base_model_type == "volume":
        out_model_type = "implied_ratio_without_supplied_volume"
        base_model = load_model(VOLUME_MODEL_FILENAME)
    elif base_model_type == "ratio_with_supplied_volume":
        out_model_type = "implied_volume_with_supplied_volume"
        base_model = load_model(RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME)
    elif base_model_type == "ratio_without_supplied_volume":
        out_model_type = "implied_volume_without_supplied_volume"
        base_model = load_model(RATIO_WITHOUT_SUPPLIED_VOLUME_MODEL_FILENAME)
    else:
        raise ValueError(f"Invalid base_model_type: {base_model_type}")

    X_train, X_test, y_train, y_test, not_selected_train, not_selected_test, X_test_not_encoded = (
        deterministic_encoded_train_test_split(base_model_type)
    )

    y_true_test_impl, y_pred_test_impl = get_implied_results_from_base_model(
        base_model, base_model_type, X_test, y_test, not_selected_test
    )

    y_true_train_impl, y_pred_train_impl = get_implied_results_from_base_model(
        base_model, base_model_type, X_train, y_train, not_selected_train
    )

    confidence_dict = make_confidence_model(
        X_train,
        X_test,
        y_true_train_impl,
        y_true_test_impl,
        y_pred_test_impl,
        out_model_type,
        "pct",
        correctness_delta_thresholds_pct=[0.1, 0.3, 0.5, 0.75, 1.0],
        visualise_model=visualise_model,
    )

    predictions = make_predictions_jsonable(
        X_test,
        y_pred_test_impl,
        X_not_encoded=X_test_not_encoded,
        confidence_dict=confidence_dict,
    )

    save_predictions(predictions, IMPLIED_RECOVERY_RATIO_PREDICTIONS_FILENAME)

    results = _get_results(
        y_test=y_true_test_impl,
        y_test_pred=y_pred_test_impl,
        type=out_model_type,
        model=base_model,
        feature_names=X_train.columns.tolist(),
        y_train=y_true_train_impl,
        y_train_pred=y_pred_train_impl,
    )

    print_model_results(results, model_name="Implied Ratio Model (from Volume)")

    if visualise_model:
        visualize_model_predictions(
            y_true=y_true_test_impl,
            y_pred=y_pred_test_impl,
            feature_importance=results["feature_importance"],
            X_features=X_test,
            target_name=MODEL_TYPE_DISPLAY_NAMES[out_model_type],
            model=base_model,
            image_filename="implied_recovery_ratio_model.png",
        )
