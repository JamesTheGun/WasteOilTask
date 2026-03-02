from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from scipy.stats import gaussian_kde
from typing import TYPE_CHECKING, List, Optional, Tuple, Any
import pandas as pd
import os

from model_constants import MODEL_TYPE_DISPLAY_NAMES

if TYPE_CHECKING:
    from confidence_model import ConfidenceResult


def generate_plots(
    data: pd.DataFrame, targets: Optional[List[str]] = None, alpha: float = 0.8
) -> None:
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
        plt.close(fig)


def base_scatter_builder(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    colour_col: str,
    figsize: Tuple[float, float] = (9, 6),
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    df = data[[x_col, y_col, colour_col]].dropna().copy()
    fig, ax = plt.subplots(figsize=figsize)
    return df, fig, ax


def _finalise_scatter_axes(
    ax: plt.Axes,
    df: pd.DataFrame,
    df_numeric: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    dark: bool = True,
) -> None:
    text_colour = "white" if dark else "black"

    if df[x_col].dtype == "object" or str(df[x_col].dtype) == "category":
        unique_x = sorted(df[x_col].unique())
        x_ticks = np.sort(df_numeric[x_col].unique())
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([unique_x[int(i)] for i in x_ticks], rotation=45, ha="right")

    if df[y_col].dtype == "object" or str(df[y_col].dtype) == "category":
        unique_y = sorted(df[y_col].unique())
        y_ticks = np.sort(df_numeric[y_col].unique())
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([unique_y[int(i)] for i in y_ticks])

    ax.set_xlabel(x_col, fontsize=10, color=text_colour)
    ax.set_ylabel(y_col, fontsize=10, color=text_colour)
    ax.set_title(title, fontsize=11, color=text_colour)


def _plot_coloured_data(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    colour_col: str,
    point_size: int = 20,
    alpha: float = 0.7,
    dark: bool = True,
) -> None:
    if dark:
        bg = "black"
        ax.set_facecolor(bg)
        ax.figure.patch.set_facecolor(bg)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
        cat_palette = "Set2"
        seq_palette = "plasma"
        legend_kw = dict(
            facecolor="#2a2a2a",
            edgecolor="#555555",
            labelcolor="white",
        )
    else:
        cat_palette = "tab10"
        seq_palette = "viridis"
        legend_kw = {}

    if df[colour_col].dtype == "object" or str(df[colour_col].dtype) == "category":
        categories = sorted(df[colour_col].unique())
        cmap = plt.get_cmap(cat_palette, max(len(categories), 3))
        for i, cat in enumerate(categories):
            mask = df[colour_col] == cat
            ax.scatter(
                df.loc[mask, x_col],
                df.loc[mask, y_col],
                label=str(cat),
                color=cmap(i),
                alpha=alpha,
                s=point_size,
                linewidths=0,
            )
        leg = ax.legend(
            title=colour_col, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8
        )
        if dark and leg is not None:
            leg.get_frame().set_facecolor(legend_kw.get("facecolor", "white"))
            leg.get_frame().set_edgecolor(legend_kw.get("edgecolor", "black"))
            for text in leg.get_texts():
                text.set_color("white")
            leg.get_title().set_color("white")
    else:
        sc = ax.scatter(
            df[x_col],
            df[y_col],
            c=df[colour_col],
            cmap=seq_palette,
            alpha=alpha,
            s=point_size,
        )
        cbar = plt.colorbar(sc, ax=ax, label=colour_col)
        if dark:
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
            cbar.set_label(colour_col, color="white")


def scatter_coloured(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    colour_col: str,
    title: str = None,
    alpha: float = 0.7,
    point_size: int = 20,
) -> None:
    df, fig, ax = base_scatter_builder(data, x_col, y_col, colour_col)

    _plot_coloured_data(
        ax, df, x_col, y_col, colour_col, point_size=point_size, alpha=alpha
    )

    if not title:
        title = f"{x_col} vs {y_col} coloured by {colour_col}"

    _finalise_scatter_axes(
        ax,
        df,
        df,
        x_col,
        y_col,
        title=title,
    )
    plt.tight_layout()

    plt.close(fig)


def scatter_confidence(
    true_delta: pd.Series,
    results: ConfidenceResult,
    colour_col: str,
    model_type: str,
    threshold: Optional[float] = None,
    point_scale: float = 100,
) -> None:
    t = threshold if threshold is not None else results.primary_threshold
    _df = results.for_threshold(t) if threshold is not None else results.primary_df

    conf = _df["confidence_center"].astype(float).reset_index(drop=True)
    delta = pd.Series(true_delta.values, index=range(len(true_delta)))
    colour_data = _df[colour_col].reset_index(drop=True)

    norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-9)
    sizes = norm * point_scale + point_scale * 0.1

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    ax.grid(True, linestyle="--", alpha=0.3, color="#555555")

    if colour_data.dtype == "object" or str(colour_data.dtype) == "category":
        categories = sorted(colour_data.unique())
        cat_cmap = plt.get_cmap("Set2", max(len(categories), 3))
        for i, cat in enumerate(categories):
            mask = colour_data == cat
            ax.scatter(
                conf[mask],
                delta[mask],
                label=str(cat),
                color=cat_cmap(i),
                alpha=0.75,
                s=sizes[mask],
                linewidths=0,
            )
        leg = ax.legend(
            title=colour_col,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=8,
            facecolor="#2a2a2a",
            edgecolor="#555555",
            labelcolor="white",
        )
        if leg is not None:
            leg.get_title().set_color("white")
    else:
        sc = ax.scatter(
            conf,
            delta,
            c=colour_data,
            cmap="plasma",
            alpha=0.75,
            s=sizes,
            linewidths=0,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(colour_col, color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.axhline(
        t,
        color="#ff4444",
        linewidth=1.5,
        linestyle="--",
        label=f"threshold = {t}",
        zorder=3,
    )

    ax.set_xlabel("Confidence (center)", fontsize=10, color="white")
    ax.set_ylabel("Prediction delta", fontsize=10, color="white")
    display_name = MODEL_TYPE_DISPLAY_NAMES.get(model_type, model_type)
    ax.set_title(
        f"Confidence vs Delta  |  threshold={t}  |  {display_name}",
        fontsize=11,
        color="white",
    )
    plt.tight_layout()

    model_dir = _make_model_vis_dir(display_name)
    save_dir = os.path.join(model_dir, "confidence_visualisations")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        os.path.join(save_dir, f"confidence_vs_delta_t{t}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def confidence_histogram(
    data: pd.DataFrame,
    confidence_col: str,
    true_col: str,
    pred_col: str,
    bins: int = 20,
) -> None:
    df = data[[confidence_col, true_col, pred_col]].dropna().copy()
    df["correct"] = df[true_col] == df[pred_col]

    plt.figure(figsize=(8, 5))
    correct_vals = df.loc[df["correct"], confidence_col]
    incorrect_vals = df.loc[~df["correct"], confidence_col]

    plt.hist(
        [correct_vals, incorrect_vals],
        bins=bins,
        color=["green", "red"],
        label=["correct", "incorrect"],
        stacked=False,
        alpha=0.7,
    )
    plt.xlabel(confidence_col)
    plt.ylabel("Count")
    plt.title(f"Confidence histogram coloured by correctness")
    plt.legend()
    plt.tight_layout()
    plt.close()


def _compute_density_scatter_defaults(
    x: np.ndarray,
    y: np.ndarray,
    bw: Optional[float] = None,
    pad: Optional[float] = None,
) -> Tuple[float, float]:
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    combined_range = (x_range + y_range) / 2.0

    if bw is None:
        bw = max(combined_range / 10.0, 0.01)
    if pad is None:
        pad = max(combined_range / 20.0, 0.1)

    return float(bw), float(pad)


def density_scatter(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    colour_col: str,
    bw: Optional[float] = None,
    point_size: int = 15,
    point_alpha: float = 0.7,
    kde_vis_alpha: float = 0.5,
    pad: Optional[float] = None,
    dark: bool = True,
    title: str = None,
    save_path: Optional[str] = None,
) -> None:
    df, fig, ax = base_scatter_builder(data, x_col, y_col, colour_col)
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

    bw, pad = _compute_density_scatter_defaults(x, y, bw=bw, pad=pad)

    xmin, xmax = x.min() - pad, x.max() + pad
    ymin, ymax = y.min() - pad, y.max() + pad
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contourf(xx, yy, density, levels=14, cmap="Blues", alpha=kde_vis_alpha)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if dark:
        ax.grid(True, linestyle="--", alpha=0.4, color="#555555")
        ax.minorticks_on()
        ax.grid(which="minor", linestyle=":", alpha=0.2, color="#555555")
    else:
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.minorticks_on()
        ax.grid(which="minor", linestyle=":", alpha=0.3)

    df_plot = df.copy()
    df_plot[x_col] = x
    _plot_coloured_data(
        ax,
        df_plot,
        x_col,
        y_col,
        colour_col,
        point_size=point_size,
        alpha=point_alpha,
        dark=dark,
    )

    if not title:
        title = f"{x_col} vs {y_col}  |  colour: {colour_col}  |  KDE bw={bw}"

    _finalise_scatter_axes(
        ax,
        df,
        df_numeric,
        x_col,
        y_col,
        title=title,
        dark=dark,
    )

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def _encode_for_model(series):
    if series.dtype == "object" or str(series.dtype) == "category":
        le = LabelEncoder()
        encoded = le.fit_transform(series.astype(str)).astype(float)
        return encoded, le
    elif np.issubdtype(series.dtype, np.datetime64):
        return series.astype(np.int64).astype(float), None
    else:
        return series.values.astype(float), None


def plot_lgbm_model(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    title: Optional[str] = None,
) -> Any:
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

    axes[0].barh(
        feature_cols, model.feature_importances_, color="steelblue", edgecolor="white"
    )
    axes[0].set_xlabel("Importance (split count)")
    axes[0].set_title(f"Feature Importance → {target_col}")

    if is_classifier:
        from sklearn.metrics import accuracy_score, f1_score

        correct = preds == y
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="weighted")
        axes[1].scatter(range(len(y)), y, c=correct, cmap="RdYlGn", alpha=0.5, s=10)
        axes[1].set_title(f"Correct predictions  |  accuracy={acc:.2%}")
        axes[1].set_xlabel("Sample index")
        axes[1].set_ylabel(f"{target_col} (encoded)")
        stats_text = f"Accuracy = {acc:.4f}\nF1 (weighted) = {f1:.4f}\nn = {len(y)}"
        axes[1].text(
            0.02,
            0.98,
            stats_text,
            transform=axes[1].transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
        )
    else:
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        correlation = np.corrcoef(y, preds)[0, 1]
        axes[1].scatter(y, preds, alpha=0.4, s=10, color="steelblue")
        lims = [min(y.min(), preds.min()), max(y.max(), preds.max())]
        axes[1].plot(lims, lims, "r--", linewidth=1, label="Perfect fit")
        axes[1].set_xlabel(f"Actual {target_col}")
        axes[1].set_ylabel(f"Predicted {target_col}")
        axes[1].set_title(
            f"Predicted vs Actual  |  features: {', '.join(feature_cols)}"
        )
        axes[1].legend()
        stats_text = (
            f"R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}\n"
            f"Corr = {correlation:.4f}\nn = {len(y)}"
        )
        axes[1].text(
            0.02,
            0.98,
            stats_text,
            transform=axes[1].transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
        )

    plt.suptitle(
        title if title is not None else f"LightGBM: predicting {target_col}",
        fontsize=12,
    )
    plt.tight_layout()

    plt.close(fig)
    return model


import pandas as pd


def display_simple_model_with_lgbm_and_density_scatter(
    data: pd.DataFrame,
    feat_1: str,
    feat_2: str,
    target: str,
    bw: float = 0.3,
    alpha: float = 0.5,
) -> None:

    print(f"--- LightGBM: {feat_1}, {feat_2}  →  {target} ---")

    plot_lgbm_model(
        data,
        [feat_1, feat_2],
        target,
        title=f"Model: {feat_1}, {feat_2} → {target}",
    )

    print(f"--- Density scatter: {feat_1} vs {target}, colour={feat_2} ---")
    density_scatter(data, feat_1, target, feat_2, bw=bw, alpha=alpha)


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
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)

    top_feature = max(feature_importance, key=feature_importance.get)
    colour_values = feature_values[top_feature]

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    correlation = np.corrcoef(y_true, y_pred)[0, 1]

    fig, ax = plt.subplots(figsize=figsize)

    pad = 0.05 * (y_true.max() - y_true.min())
    xmin, xmax = y_true.min() - pad, y_true.max() + pad
    ymin, ymax = y_pred.min() - pad, y_pred.max() + pad
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    try:
        kde = gaussian_kde(np.vstack([y_true, y_pred]), bw_method=kde_bw)
        density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.contourf(xx, yy, density, levels=12, cmap="Blues", alpha=0.4)
    except Exception:
        pass

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

    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit", zorder=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel(f"Actual {target_name}", fontsize=11)
    ax.set_ylabel(f"Predicted {target_name}", fontsize=11)
    ax.set_title(
        f"Model Results: {target_name}\nColoured by: {top_feature} (most important)",
        fontsize=12,
    )

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

    plt.close(fig)

    return save_path


def _make_model_vis_dir(target_name: str) -> str:

    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data_exploration_visualisations",
        "models",
        f"recovery_{target_name}",
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _visualise_confidence(
    plot_df: pd.DataFrame,
    save_path: str,
    model_type: str,
) -> None:

    density_scatter(
        plot_df,
        x_col="confidence_lower",
        y_col="correctness_delta_pct_sd",
        colour_col="clusters",
        title=f"Confidence Lower vs Error (threshold={threshold} SD) | {model_type}",
        point_alpha=0.7,
        point_size=25,
        save_path=save_path,
    )


def visualize_model_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_importance: dict,
    X_features: pd.DataFrame,
    target_name: str,
    model: Any = None,
    image_filename: str = "model_results.png",
) -> str:
    feature_values = {col: X_features[col].values for col in X_features.columns}

    save_dir = _make_model_vis_dir(target_name)

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
            plt.close()
        except ImportError:
            print("SHAP not installed. Run: pip install shap")
        except Exception as e:
            import traceback

            print(f"SHAP plot failed: {e}")
            traceback.print_exc()

    return result_path
