import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from scipy.stats import gaussian_kde


def generate_plots(data, targets=None, alpha=0.8):
    """
    Generate histogram or bar plots for each feature in the dataset.

    Args:
        data:    DataFrame containing the data
        targets: Optional list of column names to plot. Plots all columns if None.
        alpha:   Opacity of the bars/histogram (0.0 transparent, 1.0 opaque). Default 0.8.
    """

    cols = targets if targets is not None else data.columns.tolist()

    for col in cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        col_data = data[col].dropna()

        if col_data.dtype == "object" or hasattr(col_data, "cat"):
            counts = col_data.value_counts().sort_index()
            ax.bar(
                counts.index.astype(str),
                counts.values,
                color="steelblue",
                edgecolor="white",
                alpha=alpha,
            )
            ax.set_xticklabels(
                counts.index.astype(str), rotation=45, ha="right", fontsize=8
            )
        else:
            ax.hist(
                col_data, bins=20, color="steelblue", edgecolor="white", alpha=alpha
            )

        ax.set_title(col, fontsize=12)
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)

        plt.tight_layout()
        plt.show()
        plt.close(fig)


def scatter_coloured(data, x_col, y_col, colour_col):
    """
    Create a colour-coded scatter plot across three dimensions.

    Args:
        data:        DataFrame containing the data
        x_col:       Column name for the x-axis
        y_col:       Column name for the y-axis
        colour_col:  Column name used to colour the points
    """

    df = data[[x_col, y_col, colour_col]].dropna()

    fig, ax = plt.subplots(figsize=(9, 6))

    # Categorical colour column
    if df[colour_col].dtype == "object" or str(df[colour_col].dtype) == "category":
        categories = df[colour_col].unique()
        cmap = plt.get_cmap("tab10", len(categories))
        for i, cat in enumerate(sorted(categories)):
            mask = df[colour_col] == cat
            ax.scatter(
                df.loc[mask, x_col],
                df.loc[mask, y_col],
                label=str(cat),
                color=cmap(i),
                alpha=0.7,
                s=20,
            )
        ax.legend(
            title=colour_col, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8
        )
    else:
        # Numeric colour column — use a continuous colourmap
        sc = ax.scatter(
            df[x_col], df[y_col], c=df[colour_col], cmap="viridis", alpha=0.7, s=20
        )
        plt.colorbar(sc, ax=ax, label=colour_col)

    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    ax.set_title(f"{x_col} vs {y_col}  |  colour: {colour_col}", fontsize=11)
    plt.tight_layout()

    plt.show()
    plt.close(fig)


def density_scatter(
    data, x_col, y_col, colour_col, bw=0.3, point_size=15, alpha=0.7, pad=0.5
):
    """
    Scatter plot with a KDE density heatmap rendered behind the points,
    colour-coded by a third column.

    Args:
        data:        DataFrame containing the data
        x_col:       Column name for the x-axis
        y_col:       Column name for the y-axis
        colour_col:  Column name used to colour the points
        bw:          KDE bandwidth (smaller = tighter fit). Default 0.3.
        point_size:  Size of scatter points. Default 15.
        alpha:       Opacity of scatter points. Default 0.7.
        pad:         Padding added around the data extent for the KDE grid and axis
                     limits. Useful when an axis has discrete categories encoded as
                     integers so points aren't flush against the edge. Default 0.5.
    """

    df = data[[x_col, y_col, colour_col]].dropna().copy()
    df_numeric = df.copy()

    for col in [x_col, y_col]:
        if (
            df_numeric[col].dtype == "object"
            or str(df_numeric[col].dtype) == "category"
        ):
            df_numeric[col] = df_numeric[col].astype("category").cat.codes
        elif np.issubdtype(df_numeric[col].dtype, np.datetime64):
            df_numeric[col] = df_numeric[col].astype(np.int64)

    x = df_numeric[x_col].values.astype(float)
    y = df_numeric[y_col].values.astype(float)

    fig, ax = plt.subplots(figsize=(9, 6))

    xmin, xmax = x.min() - pad, x.max() + pad
    ymin, ymax = y.min() - pad, y.max() + pad
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contourf(xx, yy, density, levels=14, cmap="Blues", alpha=0.5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if df[colour_col].dtype == "object" or str(df[colour_col].dtype) == "category":
        categories = sorted(df[colour_col].unique())
        cmap = plt.get_cmap("tab10", len(categories))
        for i, cat in enumerate(categories):
            mask = df[colour_col] == cat
            ax.scatter(
                x[mask],
                y[mask],
                label=str(cat),
                color=cmap(i),
                alpha=alpha,
                s=point_size,
                linewidths=0,
            )
        ax.legend(
            title=colour_col, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8
        )
    else:
        c_vals = df[colour_col].values.astype(float)
        sc = ax.scatter(
            x, y, c=c_vals, cmap="viridis", alpha=alpha, s=point_size, linewidths=0
        )
        plt.colorbar(sc, ax=ax, label=colour_col)

    if df[x_col].dtype == "object" or str(df[x_col].dtype) == "category":
        unique_x = sorted(df[x_col].unique())
        x_codes = [df_numeric[x_col].unique() for _ in range(1)]
        x_ticks = np.sort(df_numeric[x_col].unique())
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([unique_x[int(i)] for i in x_ticks], rotation=45, ha="right")

    if df[y_col].dtype == "object" or str(df[y_col].dtype) == "category":
        unique_y = sorted(df[y_col].unique())
        y_ticks = np.sort(df_numeric[y_col].unique())
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([unique_y[int(i)] for i in y_ticks])

    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    ax.set_title(
        f"{x_col} vs {y_col}  |  colour: {colour_col}  |  KDE bw={bw}", fontsize=11
    )
    plt.tight_layout()

    plt.show()
    plt.close(fig)


def _encode_for_model(series):
    """Encode a column to float for use in LightGBM. Returns (encoded array, label_encoder or None)."""
    if series.dtype == "object" or str(series.dtype) == "category":
        le = LabelEncoder()
        encoded = le.fit_transform(series.astype(str)).astype(float)
        return encoded, le
    elif np.issubdtype(series.dtype, np.datetime64):
        return series.astype(np.int64).astype(float), None
    else:
        return series.values.astype(float), None


def plot_lgbm_model(data, feature_cols: list[str], target_col: str, title=None):
    """
    Train a simple LightGBM model predicting target_col from feature_cols,
    then plot feature importances and predicted vs actual.

    Args:
        data:         DataFrame
        feature_cols: List of feature column names (can be >2)
        target_col:   Target column to predict

    Returns:
        Trained LightGBM model
    """
    assert type(feature_cols) == list and all(
        isinstance(c, str) for c in feature_cols
    ), "feature_cols must be a list of strings"

    df = data[feature_cols + [target_col]].dropna().copy()
    X = []
    for col in feature_cols:
        enc, _ = _encode_for_model(df[col])
        X.append(enc)
    X = np.column_stack(X)

    is_classifier = (
        df[target_col].dtype == "object" or str(df[target_col].dtype) == "category"
    )

    if is_classifier:
        target_le = LabelEncoder()
        y = target_le.fit_transform(df[target_col].astype(str))
        n_classes = len(target_le.classes_)
        extra = {"num_class": n_classes} if n_classes > 2 else {}
        model = lgb.LGBMClassifier(
            objective="multiclass" if n_classes > 2 else "binary",
            num_leaves=15,
            n_estimators=100,
            verbose=-1,
            **extra,
        )
    else:
        target_le = None
        y = df[target_col].values.astype(float)
        model = lgb.LGBMRegressor(
            objective="regression", num_leaves=15, n_estimators=100, verbose=-1
        )

    model.fit(X, y)
    preds = model.predict(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Feature importance
    axes[0].barh(
        feature_cols, model.feature_importances_, color="steelblue", edgecolor="white"
    )
    axes[0].set_xlabel("Importance (split count)")
    axes[0].set_title(f"Feature Importance → {target_col}")

    # Predicted vs actual
    if is_classifier:
        correct = preds == y
        axes[1].scatter(range(len(y)), y, c=correct, cmap="RdYlGn", alpha=0.5, s=10)
        axes[1].set_title(f"Correct predictions  |  accuracy={correct.mean():.2%}")
        axes[1].set_xlabel("Sample index")
        axes[1].set_ylabel(f"{target_col} (encoded)")
    else:
        axes[1].scatter(y, preds, alpha=0.4, s=10, color="steelblue")
        lims = [min(y.min(), preds.min()), max(y.max(), preds.max())]
        axes[1].plot(lims, lims, "r--", linewidth=1, label="Perfect fit")
        axes[1].set_xlabel(f"Actual {target_col}")
        axes[1].set_ylabel(f"Predicted {target_col}")
        axes[1].set_title(
            f"Predicted vs Actual  |  features: {', '.join(feature_cols)}"
        )
        axes[1].legend()

    plt.suptitle(
        title if title is not None else f"LightGBM: predicting {target_col}",
        fontsize=12,
    )
    plt.tight_layout()

    plt.show()
    plt.close(fig)
    return model


import pandas as pd


def display_simple_model_with_lgbm_and_density_scatter(
    data: pd.DataFrame, x_col, y_col, target_col, bw=0.3, alpha=0.5
):
    """
    Train a simple LightGBM model then show both the model diagnostics
    and a KDE density scatter for the same three columns.

    The last argument (target_col) is the prediction target for the model
    and the colour dimension for the scatter.

    Args:
        data:       DataFrame
        x_col:      Feature / x-axis column
        y_col:      Feature / y-axis column
        target_col: Target for the model; colour column for the scatter
        bw:         KDE bandwidth. Default 0.3.
        alpha:      Point opacity. Default 0.5.
    """
    print(f"--- LightGBM: {x_col}, {y_col}  →  {target_col} ---")
    print(target_col)

    plot_lgbm_model(
        data,
        [x_col, y_col],
        target_col,
        title=f"Model: {x_col}, {y_col} → {target_col}",
    )

    print(f"--- Density scatter: {x_col} vs {y_col}, colour={target_col} ---")
    density_scatter(data, x_col, y_col, target_col, bw=bw, alpha=alpha)


# model visualisation functions:
# ...existing code...


def plot_model_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_importance: dict,
    feature_values: dict,
    target_name: str = "target",
    save_path: str = None,
    figsize: tuple = (10, 8),
    kde_bw: float = 0.3,
):
    """
    Visualise model results: true vs predicted, coloured by most important feature.
    Includes KDE density heatmap, statistics, and saves as image.

    Args:
        y_true:             Array of true target values
        y_pred:             Array of predicted values
        feature_importance: Dict of {feature_name: importance_score}
        feature_values:     Dict of {feature_name: array of values} (same length as y_true)
        target_name:        Name of target column for labels
        save_path:          Path to save image. If None, uses f"model_results_{target_name}.png"
        figsize:            Figure size tuple. Default (10, 8).
        kde_bw:             KDE bandwidth. Default 0.3.

    Returns:
        Path to saved image
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)

    # Find most important feature
    top_feature = max(feature_importance, key=feature_importance.get)
    colour_values = feature_values[top_feature]

    # Calculate statistics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    correlation = np.corrcoef(y_true, y_pred)[0, 1]

    fig, ax = plt.subplots(figsize=figsize)

    # KDE density heatmap
    pad = 0.05 * (y_true.max() - y_true.min())
    xmin, xmax = y_true.min() - pad, y_true.max() + pad
    ymin, ymax = y_pred.min() - pad, y_pred.max() + pad
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    try:
        kde = gaussian_kde(np.vstack([y_true, y_pred]), bw_method=kde_bw)
        density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.contourf(xx, yy, density, levels=12, cmap="Blues", alpha=0.4)
    except Exception:
        pass  # Skip KDE if it fails (e.g., singular matrix)

    # Determine if colour column is categorical
    is_categorical = isinstance(colour_values, (list, np.ndarray)) and (
        isinstance(colour_values[0], str) or len(np.unique(colour_values)) < 15
    )

    if is_categorical:
        categories = sorted(np.unique(colour_values))
        cmap = plt.get_cmap("tab10", len(categories))
        for i, cat in enumerate(categories):
            mask = colour_values == cat
            ax.scatter(
                np.array(y_true)[mask],
                np.array(y_pred)[mask],
                label=str(cat),
                color=cmap(i),
                alpha=0.7,
                s=30,
            )
        ax.legend(
            title=top_feature, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8
        )
    else:
        sc = ax.scatter(
            y_true, y_pred, c=colour_values, cmap="viridis", alpha=0.7, s=30
        )
        plt.colorbar(sc, ax=ax, label=top_feature)

    # Perfect fit line
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit", zorder=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Labels and title
    ax.set_xlabel(f"Actual {target_name}", fontsize=11)
    ax.set_ylabel(f"Predicted {target_name}", fontsize=11)
    ax.set_title(
        f"Model Results: {target_name}\nColoured by: {top_feature} (most important)",
        fontsize=12,
    )

    # Statistics text box
    stats_text = (
        f"R² = {r2:.4f}\n"
        f"MAE = {mae:.4f}\n"
        f"RMSE = {rmse:.4f}\n"
        f"Correlation = {correlation:.4f}\n"
        f"n = {len(y_true)}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    # Feature importance text
    importance_text = "Feature Importance:\n" + "\n".join(
        f"  {k}: {v:.1f}"
        for k, v in sorted(feature_importance.items(), key=lambda x: -x[1])[:5]
    )
    ax.text(
        0.98,
        0.02,
        importance_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round", facecolor="lightyellow", alpha=0.9, edgecolor="gray"
        ),
    )

    plt.tight_layout()

    if not save_path:
        save_path = f"model_results_{target_name}.png"

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()
    plt.close(fig)

    return save_path


def visualize_model_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_importance: dict,
    X_features: pd.DataFrame,
    target_name: str,
    model=None,
    image_filename: str = "model_results.png",
):
    """
    Visualise actual vs predicted, coloured by most important feature.
    Optionally includes SHAP summary plot if model is provided.

    Args:
        y_true:             Actual target values
        y_pred:             Predicted values
        feature_importance: Dict of {feature_name: importance_score}
        X_features:         Feature DataFrame for colouring
        target_name:        Name for labels (e.g. "recovery_ratio")
        model:              Trained model for SHAP analysis (optional)
        image_filename:     Output filename in data_exploration_visualisations/models/

    Returns:
        Path to saved image
    """
    import os

    feature_values = {col: X_features[col].values for col in X_features.columns}

    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data_exploration_visualisations",
        "models",
    )
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, image_filename)

    result_path = plot_model_results(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        feature_importance=feature_importance,
        feature_values=feature_values,
        target_name=target_name,
        save_path=save_path,
    )

    # SHAP plot if model provided
    if model is not None:
        try:
            import shap

            shap_path = os.path.join(
                save_dir, image_filename.replace(".png", "_shap.png")
            )

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_features)

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_features, show=False)
            plt.tight_layout()
            plt.savefig(shap_path, dpi=150, bbox_inches="tight")
            plt.show()
            plt.close()
        except ImportError:
            print("SHAP not installed. Run: pip install shap")
        except Exception as e:
            print(f"SHAP plot failed: {e}")

    return result_path
