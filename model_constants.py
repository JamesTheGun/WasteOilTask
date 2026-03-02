RATIO_MODELS = ["ratio_without_supplied_volume", "ratio_with_supplied_volume"]

MODEL_TYPE_DISPLAY_NAMES = {
    "ratio_without_supplied_volume": "ratio_without_supplied_volume",
    "volume": "volume",
    "ratio_with_supplied_volume": "ratio_with_supplied_volume",
    "implied_volume_without_supplied_volume": "implied_volume_without_supplied_volume",
    "implied_ratio_without_supplied_volume": "implied_ratio_without_supplied_volume",
    "implied_volume_with_supplied_volume": "implied_volume_with_supplied_volume",
}

DEFAULT_MODEL_NAME = "lgbm_recovery_model.pkl"

RATIO_WITHOUT_SUPPLIED_VOLUME_MODEL_FILENAME = "lgbm_recovery_ratio_model.pkl"
VOLUME_MODEL_FILENAME = "lgbm_recovery_volume_model.pkl"
RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME = (
    "lgbm_recovery_ratio_with_supplied_volume_model.pkl"
)

RESULTS_RATIO_MODEL_FILENAME = "lgbm_recovery_ratio_model_results.json"
RESULTS_VOLUME_MODEL_FILENAME = "lgbm_recovery_volume_model_results.json"
RESULTS_RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME = (
    "lgbm_recovery_ratio_with_supplied_volume_model_results.json"
)

IMPLIED_RECOVERY_RATIO_PREDICTIONS_FILENAME = "implied_recovery_ratio_predictions.json"
IMPLIED_RECOVERY_VOLUME_PREDICTIONS_FILENAME = (
    "implied_recovery_volume_predictions.json"
)

CONFIDENCE_MODEL_FILENAMES = {
    "ratio_without_supplied_volume": "confidence_model_ratio_without_supplied_volume.pkl",
    "volume": "confidence_model_volume.pkl",
    "ratio_with_supplied_volume": "confidence_model_ratio_with_supplied_volume.pkl",
}
