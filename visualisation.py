from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from model_constants import MODEL_TYPE_DISPLAY_NAMES

if TYPE_CHECKING:
    from confidence_model import ConfidenceResult


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
