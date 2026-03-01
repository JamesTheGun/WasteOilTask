import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from scipy.stats import gaussian_kde
from typing import List, Optional, Tuple, Any
import pandas as pd


def generate_plots(
    data: pd.DataFrame, targets: Optional[List[str]] = None, alpha: float = 0.8
) -> None:
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
        plt.close(fig)


def base_scatter_builder(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    colour_col: str,
    figsize: Tuple[float, float] = (9, 6),
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """Initialise a scatter plot and return core objects.

    This helper performs the common pre-processing used by the various
    scatter-based visualisations defined in this module.  It:

    * selects and drops NA rows for the three columns
    * creates a ``matplotlib`` ``fig``/``ax`` pair with a sensible size

    Returns ``(df, fig, ax)`` where ``df`` is the filtered DataFrame.
    """

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
    """Apply labels/ticks/title to a scatter axis.

    The logic handles categorical x/y axes by substituting numeric ticks with
    the original category strings, using ``df_numeric`` for location values.
    ``title`` should already include any extra information (e.g. KDE bandwidth).
    When ``dark=True`` labels and title are rendered in white.
    """

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
    """Draw scatter points on ``ax`` coloured by ``colour_col`` in ``df``.

    Handles both categorical and numeric colour columns.  If the column is
    categorical, each category gets a distinct colour and a legend is created;
    otherwise a continuous colourmap is used and a colourbar is attached to
    ``ax``.

    When ``dark=True`` (the default) the axes and parent figure backgrounds
    are set to a dark colour and brighter palettes / white text are used so
    that points remain clearly visible.

    Parameters mirror those used in ``scatter_coloured`` and ``density_scatter``.
    """

    if dark:
        bg = "black"
        ax.set_facecolor(bg)
        ax.figure.patch.set_facecolor(bg)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
        cat_palette = "Set2"  # bright pastels that pop on dark bg
        seq_palette = "plasma"  # vivid sequential map
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
    """
    Create a colour-coded scatter plot across three dimensions.

    Args:
        data:        DataFrame containing the data
        x_col:       Column name for the x-axis
        y_col:       Column name for the y-axis
        colour_col:  Column name used to colour the points
        alpha:       Opacity of the points. Default 0.7.
        point_size:  Size of scatter points. Default 20.
    """

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
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    colour_col: str,
    confidence_col: str,
    true_col: Optional[str] = None,
    pred_col: Optional[str] = None,
    point_scale: float = 100,
) -> None:
    """Confidence scatter plot with size and outline semantics.

    * **point size** – proportional to ``confidence_col``; higher confidence =
      larger marker.
    * **outline colour/width** – when ``true_col`` and ``pred_col`` are given,
      mismatched points will receive a **red outline** (thicker than the
      default) to highlight incorrect predictions.

    ``colour_col`` still controls point colour exactly as in
    :func:`scatter_coloured`.  ``confidence_col`` must be numeric and is
    normalised before scaling by ``point_scale``.
    """

    cols = [x_col, y_col, colour_col, confidence_col]
    if true_col:
        cols.append(true_col)
    if pred_col:
        cols.append(pred_col)

    df = data[cols].dropna().copy()

    conf = df[confidence_col].astype(float)
    norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-9)
    sizes = norm * point_scale + point_scale * 0.1

    wrong_mask = None
    if true_col and pred_col:
        wrong_mask = df[true_col] != df[pred_col]
        delta = (df[true_col].astype(float) - df[pred_col].astype(float)).abs()
        dnorm = (delta - delta.min()) / (delta.max() - delta.min() + 1e-9)
        cmap = plt.get_cmap("Reds")

    fig, ax = plt.subplots(figsize=(9, 6))

    if df[colour_col].dtype == "object" or str(df[colour_col].dtype) == "category":
        categories = sorted(df[colour_col].unique())
        cmap = plt.get_cmap("tab10", len(categories))
        for i, cat in enumerate(categories):
            mask = df[colour_col] == cat
            if wrong_mask is not None:
                edgecolors = []
                for idx, m in df[mask].iterrows():
                    if wrong_mask.loc[idx]:
                        edgecolors.append(cmap(dnorm.loc[idx]))
                    else:
                        edgecolors.append("white")
            else:
                edgecolors = "none"
            ax.scatter(
                df.loc[mask, x_col],
                df.loc[mask, y_col],
                label=str(cat),
                color=cmap(i),
                alpha=0.7,
                s=sizes[mask],
                edgecolors=edgecolors,
                linewidths=1.5,
            )
        ax.legend(
            title=colour_col, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8
        )
    else:
        if wrong_mask is not None:
            edgecolors = []
            for idx, m in df.iterrows():
                if wrong_mask.loc[idx]:
                    edgecolors.append(cmap(dnorm.loc[idx]))
                else:
                    edgecolors.append("white")
        else:
            edgecolors = "none"
        sc = ax.scatter(
            df[x_col],
            df[y_col],
            c=df[colour_col],
            cmap="viridis",
            alpha=0.7,
            s=sizes,
            edgecolors=edgecolors,
            linewidths=1.5,
        )
        plt.colorbar(sc, ax=ax, label=colour_col)

    _finalise_scatter_axes(
        ax,
        df,
        df,
        x_col,
        y_col,
        title=f"{x_col} vs {y_col}  |  colour: {colour_col}  |  confidence: {confidence_col}",
    )
    plt.tight_layout()
    plt.close(fig)


def confidence_histogram(
    data: pd.DataFrame,
    confidence_col: str,
    true_col: str,
    pred_col: str,
    bins: int = 20,
) -> None:
    """Histogram of confidence scores coloured by correctness.

    Arguments mirror :func:`scatter_confidence` but only ``confidence_col`` is
    required.  Bars show the count of points within each confidence bin; green
    for correct predictions, red for incorrect.
    """
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
    """
    Scatter plot with a KDE density heatmap rendered behind the points,
    colour-coded by a third column.

    Args:
        data:        DataFrame containing the data
        x_col:       Column name for the x-axis
        y_col:       Column name for the y-axis
        colour_col:  Column name used to colour the points
        bw:            KDE bandwidth (smaller = tighter fit). If None, computed from data range.
        point_size:    Size of scatter points. Default 15.
        point_alpha:   Opacity of scatter points. Default 0.7.
        kde_vis_alpha: Opacity of the KDE density contour fill. Default 0.5.
        pad:           Padding around the data extent. If None, computed from data range.
        dark:          Use dark theme for background and text. Default True.
        title:         Optional title for the plot. Default generates one from column names.
        save_path:     If provided, save the figure to this path instead of displaying it.
    """

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
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
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


def plot_lgbm_model(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    title: Optional[str] = None,
) -> Any:
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
    """Train a simple LightGBM model and display a density scatter plot for two features against a target.

    Args:
        data (pd.DataFrame): DataFrame containing the features and target columns.
        feat_1 (str): First feature column name, used as model input and scatter x-axis.
        target (str): Target column name to predict.
        feat_2 (str): Second feature column name, used as model input and scatter colour.
        bw (float, optional): KDE bandwidth for the density scatter. Defaults to 0.3.
        alpha (float, optional): Opacity of scatter points. Defaults to 0.5.
    """
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


def visualize_model_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_importance: dict,
    X_features: pd.DataFrame,
    target_name: str,
    model: Any = None,
    image_filename: str = "model_results.png",
) -> str:
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
        image_filename:     Output filename in data_exploration_visualisations/models/{target_name}/

    Returns:
        Path to saved image
    """
    import os

    feature_values = {col: X_features[col].values for col in X_features.columns}

    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data_exploration_visualisations",
        "models",
        target_name,
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
