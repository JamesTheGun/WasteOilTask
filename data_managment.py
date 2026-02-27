from datetime import timedelta
import os
import joblib
import pandas as pd
from typing import Literal, Tuple
from data_loading import load_data_with_derived_features, load_scheduled_data
from sklearn.preprocessing import OrdinalEncoder


ENCODER_FILENAME = "label_encoder.pkl"


def _get_encoder_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model_data", ENCODER_FILENAME
    )


def fit_encoder(df: pd.DataFrame) -> OrdinalEncoder:
    """Fit an OrdinalEncoder on the categorical columns of the training data."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(df[cat_cols])
    encoder.cat_cols_ = cat_cols
    return encoder


def save_encoder(encoder: OrdinalEncoder):
    joblib.dump(encoder, _get_encoder_path())


def load_encoder() -> OrdinalEncoder:
    path = _get_encoder_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Encoder not found at {path}. Train a model first to create the encoder."
        )
    return joblib.load(path)


def encode_with_encoder(df: pd.DataFrame, encoder: OrdinalEncoder) -> pd.DataFrame:
    """Transform a DataFrame using a previously-fit encoder."""
    encoded = df.copy()
    encoded[encoder.cat_cols_] = encoder.transform(df[encoder.cat_cols_]).astype(int)
    return encoded


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


def load_train_test_sets_target_recovery_ratio_with_supplied_volume() -> (
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
):
    data_with_derived = load_data_with_derived_features()
    X, y, not_selected = get_model_features_target_recovery_ratio_with_supplied_volume(
        data_with_derived
    )
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
    """Encode categorical features. Only for standalone/exploratory use.
    For training pipelines, use fit_encoder + encode_with_encoder instead."""
    encoder = fit_encoder(model_features)
    return encode_with_encoder(model_features, encoder)


def deterministic_encoded_train_test_split(
    split_type: Literal["ratio", "volume", "ratio_with_supplied_volume"],
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame
]:
    if split_type == "ratio":
        X, y, not_selected = load_train_test_sets_target_recovery_ratio()
    elif split_type == "volume":
        X, y, not_selected = load_train_test_sets_target_recovery_volume()
    elif split_type == "ratio_with_supplied_volume":
        X, y, not_selected = (
            load_train_test_sets_target_recovery_ratio_with_supplied_volume()
        )
    else:
        raise ValueError(f"Unknown split_type '{split_type}'")

    X_train, X_test, y_train, y_test, not_selected_train, not_selected_test = (
        train_test_time_series_split(X, y, not_selected)
    )

    encoder_x = fit_encoder(X_train)
    save_encoder(encoder_x)

    X_train = encode_with_encoder(X_train, encoder_x)
    X_test = encode_with_encoder(X_test, encoder_x)

    return X_train, X_test, y_train, y_test, not_selected_train, not_selected_test


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
    features = data_with_derived[MODEL_FEATURES].copy()
    target = data_with_derived[MODEL_TARGET]
    not_selected = data_with_derived.drop(columns=MODEL_FEATURES + [MODEL_TARGET])
    return (
        features,
        target,
        not_selected,
    )


MODEL_FEATURES_TARGET_RECOVERY_RATIO = MODEL_FEATURES.copy()
MODEL_FEATURES_TARGET_RECOVERY_RATIO.remove("supplied_m3")
MODEL_TARGET_RECOVERY_RATIO = "recovery_ratio"


def get_model_features_target_recovery_ratio(
    data_with_derived: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    features = data_with_derived[MODEL_FEATURES_TARGET_RECOVERY_RATIO].copy()
    target = data_with_derived[MODEL_TARGET_RECOVERY_RATIO]
    not_selected = data_with_derived.drop(
        columns=MODEL_FEATURES_TARGET_RECOVERY_RATIO + [MODEL_TARGET_RECOVERY_RATIO]
    )
    return (features, target, not_selected)


def get_model_features_target_recovery_ratio_with_supplied_volume(
    data_with_derived: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    features = data_with_derived[MODEL_FEATURES].copy()
    target = data_with_derived[MODEL_TARGET_RECOVERY_RATIO]
    not_selected = data_with_derived.drop(
        columns=MODEL_FEATURES + [MODEL_TARGET_RECOVERY_RATIO]
    )
    return (features, target, not_selected)


def get_schedule_model_features(
    m_type: Literal["ratio", "volume", "ratio_with_supplied_volume"],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load scheduled delivery data and prepare features for model prediction.
    Uses the saved encoder from training to ensure consistent encoding.

    Returns:
        Tuple of (encoded features DataFrame, raw features DataFrame)
    """
    scheduled_data = load_scheduled_data()
    if m_type == "ratio":
        features = MODEL_FEATURES_TARGET_RECOVERY_RATIO
    elif m_type == "volume":
        features = MODEL_FEATURES
    elif m_type == "ratio_with_supplied_volume":
        features = MODEL_FEATURES
    raw_features = scheduled_data[features].copy()
    encoder = load_encoder()
    encoded_features = encode_with_encoder(raw_features, encoder)
    return encoded_features, raw_features


if __name__ == "__main__":
    # Example usage
    features = get_schedule_model_features(m_type="ratio")
    print("Features shape:", features.shape)
    print("Feature columns:", features.columns)
    print("Sample features:\n", features.head())
