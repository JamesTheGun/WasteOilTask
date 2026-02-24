import numpy as np
import pandas as pd


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
