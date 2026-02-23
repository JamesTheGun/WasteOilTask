from datetime import timedelta
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split as sk_train_test_split
from data_loading import load_data_with_derived_features, load_scheduled_data
from sklearn.preprocessing import LabelEncoder


def load_train_test_sets_target_recovery_volume() -> (
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
):
    data_with_derived = load_data_with_derived_features()
    X, y, not_selected = get_model_features_target_recovery_volume(data_with_derived)
    return X, y, not_selected


def load_train_test_sets_target_recovery_ratio() -> (
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
):
    data_with_derived = load_data_with_derived_features()
    X, y, not_selected = get_model_features_target_recovery_ratio(data_with_derived)
    return X, y, not_selected


def train_test_time_series_split(
    X: pd.DataFrame, y: pd.Series, not_selected: pd.DataFrame, test_size=0.2
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame
]:

    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X must have a DatetimeIndex for time series splitting.")
    if not X.index.equals(y.index):
        raise ValueError("X and y must have the same index.")

    split_point = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    not_selected_train = not_selected.iloc[:split_point]
    not_selected_test = not_selected.iloc[split_point:]

    return X_train, X_test, y_train, y_test, not_selected_train, not_selected_test


def encode_features(model_features: pd.DataFrame) -> pd.DataFrame:
    df = model_features.copy()
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df


MODEL_FEATURES = [
    "facility",
    "time_start_hour_minute",
    "supplier",
    "supplied_m3",
]

MODEL_TARGET = "recovered_m3"


def get_model_features_target_recovery_volume(
    data_with_derived: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    encoded_features = encode_features(data_with_derived[MODEL_FEATURES])
    target = data_with_derived[MODEL_TARGET]
    not_selected = data_with_derived.drop(columns=MODEL_FEATURES + [MODEL_TARGET])
    return (
        encoded_features,
        target,
        not_selected,
    )


MODEL_FEATURES_TARGET_RECOVERY_RATIO = MODEL_FEATURES.copy()
MODEL_FEATURES_TARGET_RECOVERY_RATIO.remove("supplied_m3")
MODEL_TARGET_RECOVERY_RATIO = "recovery_ratio"


def get_model_features_target_recovery_ratio(
    data_with_derived: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    encoded_features = encode_features(
        data_with_derived[MODEL_FEATURES_TARGET_RECOVERY_RATIO]
    )
    target = data_with_derived[MODEL_TARGET_RECOVERY_RATIO]
    not_selected = data_with_derived.drop(
        columns=MODEL_FEATURES_TARGET_RECOVERY_RATIO + [MODEL_TARGET_RECOVERY_RATIO]
    )
    return (encoded_features, target, not_selected)


def get_schedule_model_features(m_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load scheduled delivery data and prepare features for model prediction.

    Returns:
        Tuple of (encoded features DataFrame, raw features DataFrame)
    """
    scheduled_data = load_scheduled_data()
    if m_type == "ratio":
        features = MODEL_FEATURES_TARGET_RECOVERY_RATIO
    else:
        features = MODEL_FEATURES
    raw_features = scheduled_data[features].copy()
    encoded_features = encode_features(scheduled_data[features])
    return encoded_features, raw_features


if __name__ == "__main__":
    # Example usage
    features = get_schedule_model_features(m_type="ratio")
    print("Features shape:", features.shape)
    print("Feature columns:", features.columns)
    print("Sample features:\n", features.head())
