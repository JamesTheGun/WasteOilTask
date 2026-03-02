import numpy as np
import pandas as pd
from confidence_model import ConfidenceResult


def _to_native(val):
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


def make_predictions_jsonable(
    X: pd.DataFrame,
    predictions: np.ndarray,
    X_not_encoded: pd.DataFrame = None,
    confidence_dict: ConfidenceResult = None,
):
    X = X.reset_index(drop=True)
    display = X_not_encoded.reset_index(drop=True) if X_not_encoded is not None else X
    features = display.columns.tolist()
    prediction_feature_values = [
        {k: _to_native(v) for k, v in zip(features, row)} for row in display.values
    ]
    predictions_with_id = pd.DataFrame(
        {
            "id": [int(i) for i in X.index],
            "predicted_recovery": [float(p) for p in predictions],
            "prediction_feature_values": prediction_feature_values,
        }
    )

    if confidence_dict is not None:
        clusters = list(confidence_dict.clusters.values)
        lowers = list(confidence_dict.confidence_lower.values)
        centers = list(confidence_dict.confidence_center.values)
        uppers = list(confidence_dict.confidence_upper.values)
        tpct = confidence_dict.threshold_pct_sd
        tabs = confidence_dict.threshold_abs
        predictions_with_id["confidence"] = [
            {
                "cluster": int(clusters[i]),
                "lower": float(lowers[i]),
                "center": float(centers[i]),
                "upper": float(uppers[i]),
                "threshold_pct_sd": float(tpct) if tpct is not None else None,
                "threshold_abs": float(tabs) if tabs is not None else None,
            }
            for i in range(len(clusters))
        ]

    return predictions_with_id.to_dict(orient="records")


def print_model_results(results: dict[str, float], model_name: str = None):
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
