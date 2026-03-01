import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
from typing import Any, List, Tuple, Literal
import pandas as pd

from confidence_model import get_confidence_results
from visualisation import visualize_model_predictions
from data_managment import (
    get_schedule_model_features,
    deterministic_encoded_train_test_split,
)
from model_constants import (
    RATIO_MODELS,
    MODEL_TYPE_DISPLAY_NAMES,
    RATIO_WITHOUT_VOL_MODEL_FILENAME,
    VOLUME_MODEL_FILENAME,
    RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME,
    RESULTS_RATIO_MODEL_FILENAME,
    RESULTS_VOLUME_MODEL_FILENAME,
    RESULTS_RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME,
)
from model_save_load import (
    save_model,
    make_and_save_predictions,
)
from model_utils import print_model_results


def _train_lgbm_recovery_volume_model(
    visualise_model: bool,
) -> Tuple[dict, lgb.LGBMRegressor]:
    return _train_ratio_with_vol_lgbm_recovery_ratio_model(
        visualise_model, type="volume"
    )


def _train_lgbm_recovery_ratio_without_vol_model(
    visualise_model: bool,
) -> Tuple[dict, lgb.LGBMRegressor]:
    return _train_ratio_with_vol_lgbm_recovery_ratio_model(
        visualise_model, type="ratio"
    )


def _get_results(
    y_test,
    y_test_pred,
    type: Literal["ratio", "volume", "ratio_with_supplied_volume"],
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

    if type in RATIO_MODELS and supplied_m3 is not None:
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


def _train_ratio_with_vol_lgbm_recovery_ratio_model(
    visualise_model: bool,
    type: Literal["ratio", "volume", "ratio_with_supplied_volume"],
) -> Tuple[dict, lgb.LGBMRegressor]:

    X_train, X_test, y_train, y_test, not_selected_train, not_selected_test = (
        deterministic_encoded_train_test_split(type)
    )

    # Split training data further into train/val for early stopping
    # (using the last 20% of training data as validation)
    val_split = int(len(X_train) * 0.8)
    X_val = X_train.iloc[val_split:]
    y_val = y_train.iloc[val_split:]
    X_train_fit = X_train.iloc[:val_split]
    y_train_fit = y_train.iloc[:val_split]

    model = lgb.LGBMRegressor(
        n_estimators=200, num_leaves=31, learning_rate=0.05, random_state=1, verbose=-1
    )

    model.fit(
        X_train_fit,
        y_train_fit,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(10)],
    )

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    supplied_m3 = not_selected_test.get("supplied_m3")

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

    if visualise_model:
        display_type = MODEL_TYPE_DISPLAY_NAMES.get(type, type)
        visualize_model_predictions(
            y_true=y_test,
            y_pred=y_test_pred,
            feature_importance=results["feature_importance"],
            X_features=X_test,
            target_name=f"recovery_{display_type}",
            model=model,
            image_filename=f"recovery_{display_type}_model.png",
        )

    get_confidence_results(X_test, model, type, visualise=visualise_model)

    return results, model


def do_recovery_ratio_model_without_vol(
    model_filename: str = RATIO_WITHOUT_VOL_MODEL_FILENAME,
    visualise_model: bool = False,
):
    ratio_results, model = _train_lgbm_recovery_ratio_without_vol_model(visualise_model)
    print_model_results(ratio_results, model_name="Recovery Ratio Model")
    save_model(model, model_filename)


def do_recovery_volume_model(
    model_filename: str = VOLUME_MODEL_FILENAME, visualise_model: bool = False
):
    volume_results, model = _train_lgbm_recovery_volume_model(visualise_model)
    print_model_results(volume_results, model_name="Recovery Volume Model")
    save_model(model, model_filename)


def do_recovery_ratio_model_with_supplied_volume(
    model_filename: str = RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME,
    visualise_model: bool = False,
):
    ratio_results, model = _train_ratio_with_vol_lgbm_recovery_ratio_model(
        visualise_model, type="ratio_with_supplied_volume"
    )
    print_model_results(
        ratio_results, model_name="Recovery Ratio Model with Supplied Volume"
    )
    save_model(model, model_filename)


if __name__ == "__main__":
    from implied_models import do_implied_ratio_model, do_implied_volume_model

    do_recovery_ratio_model_without_vol(
        RATIO_WITHOUT_VOL_MODEL_FILENAME, visualise_model=True
    )
    do_recovery_volume_model(VOLUME_MODEL_FILENAME, visualise_model=True)
    do_recovery_ratio_model_with_supplied_volume(
        RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME, visualise_model=True
    )

    X_ratio, X_ratio_not_encoded = get_schedule_model_features(m_type="ratio")
    make_and_save_predictions(
        X_ratio,
        "ratio",
        model_filename=RATIO_WITHOUT_VOL_MODEL_FILENAME,
        predictions_filename=RESULTS_RATIO_MODEL_FILENAME,
        X_not_encoded=X_ratio_not_encoded,
    )

    X_volume, X_volume_not_encoded = get_schedule_model_features(m_type="volume")
    make_and_save_predictions(
        X_volume,
        "volume",
        model_filename=VOLUME_MODEL_FILENAME,
        predictions_filename=RESULTS_VOLUME_MODEL_FILENAME,
        X_not_encoded=X_volume_not_encoded,
    )

    X_rsv, X_rsv_not_encoded = get_schedule_model_features(
        m_type="ratio_with_supplied_volume"
    )
    make_and_save_predictions(
        X_rsv,
        "ratio_with_supplied_volume",
        model_filename=RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME,
        predictions_filename=RESULTS_RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME,
        X_not_encoded=X_rsv_not_encoded,
    )

    do_implied_ratio_model(visualise_model=True)
    do_implied_volume_model(visualise_model=True)
