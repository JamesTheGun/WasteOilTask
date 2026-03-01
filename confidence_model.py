from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import statsmodels.stats.proportion as smp
import lightgbm

CIKind = Literal["wilson", "beta"]
BetaPrior = Tuple[float, float]


@dataclass(frozen=True)
class ClusterConfidence:
    """
    Represents cluster-level probability that |y - yhat| <= threshold.
    center: point estimate of probability (either p_hat or posterior mean)
    lower/upper: interval bounds (Wilson CI or Beta credible interval)
    n: number of points used to estimate confidence in this cluster
    k: number of "successes" (|residual| <= threshold)
    """

    center: float
    lower: float
    upper: float
    n: int
    k: int


def _validate_params(critical_percentage_delta, critical_abs_delta):
    if (critical_percentage_delta is None) == (critical_abs_delta is None):
        raise ValueError(
            "Provide exactly one of critical_percentage_delta or critical_abs_delta."
        )


def _get_deltas_for_percentage(
    deltas_abs, sd_reference_y, critical_percentage_delta, actual_y
):
    if sd_reference_y is None:
        sd_reference_y = actual_y
    sd = float(pd.Series(sd_reference_y).std())
    if sd <= 0 or not np.isfinite(sd):
        raise ValueError(f"Invalid sd_reference_y std: {sd}")
    deltas = deltas_abs / sd
    critical_delta = float(critical_percentage_delta)
    return deltas, critical_delta


def _get_deltas_for_abs(deltas_abs, critical_abs_delta):
    deltas = deltas_abs
    critical_delta = float(critical_abs_delta)
    return deltas, critical_delta


def confidence_model(
    to_train_cluster: pd.DataFrame,
    to_get_confidences: pd.DataFrame,
    predicted_y: pd.Series,
    actual_y: pd.Series,
    *,
    features: Sequence[str],
    n_clusters: int,
    critical_percentage_delta: float | None = None,
    critical_abs_delta: float | None = None,
    sd_reference_y: Optional[pd.Series] = None,
    ci_kind: CIKind = "wilson",
    alpha: float = 0.05,
    beta_prior: BetaPrior = (0.5, 0.5),
    beta_credible_level: float = 0.95,
    random_state: int = 1,
    n_init: int | str = "auto",
) -> tuple[Pipeline, Dict[int, ClusterConfidence], list[str]]:

    _validate_params(
        critical_percentage_delta=critical_percentage_delta,
        critical_abs_delta=critical_abs_delta,
    )

    X_conf = to_get_confidences.copy()
    X_train = to_train_cluster.copy()

    predicted_y = pd.Series(predicted_y, index=actual_y.index)
    actual_y = pd.Series(actual_y, index=actual_y.index)

    deltas_abs = (actual_y - predicted_y).abs()
    if critical_percentage_delta is not None:
        deltas, critical_delta = _get_deltas_for_percentage(
            deltas_abs=deltas_abs,
            sd_reference_y=sd_reference_y,
            critical_percentage_delta=critical_percentage_delta,
            actual_y=actual_y,
        )
    else:
        deltas, critical_delta = _get_deltas_for_abs(
            deltas_abs=deltas_abs,
            critical_abs_delta=critical_abs_delta,
        )

    cluster_model: Pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "kmeans",
                KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init),
            ),
        ]
    )
    cluster_model.fit(X_train[list(features)])

    predicted_clusters = cluster_model.predict(X_conf[list(features)])

    clusters_and_deltas = pd.DataFrame(
        {
            "cluster": predicted_clusters,
            "delta": pd.Series(deltas.values, index=X_conf.index),
        },
        index=X_conf.index,
    )

    cluster_confs: Dict[int, ClusterConfidence] = {}
    for cluster_id in np.unique(predicted_clusters):
        cluster_points = clusters_and_deltas.loc[
            clusters_and_deltas["cluster"] == cluster_id, "delta"
        ]

        conf = get_cluster_confidence(
            delta_threshold=critical_delta,
            cluster_points=cluster_points,
            ci_kind=ci_kind,
            alpha=alpha,
            beta_prior=beta_prior,
            beta_credible_level=beta_credible_level,
        )
        cluster_confs[int(cluster_id)] = conf

    return cluster_model, cluster_confs, list(features)


def _do_wilson_ci(k, n, alpha=0.05) -> ClusterConfidence:
    p_hat = k / n
    lower, upper = smp.proportion_confint(k, n, alpha=alpha, method="wilson")
    return ClusterConfidence(
        center=float(p_hat), lower=float(lower), upper=float(upper), n=n, k=k
    )


def _do_beta_ci(k, n, a, b, beta_credible_level=0.95) -> ClusterConfidence:
    if a <= 0 or b <= 0:
        raise ValueError("beta_prior must have positive (a,b).")

    post_a = a + k
    post_b = b + (n - k)

    center = post_a / (post_a + post_b)

    tail = (1.0 - beta_credible_level) / 2.0
    lower = _beta_ppf(tail, post_a, post_b)
    upper = _beta_ppf(1.0 - tail, post_a, post_b)

    return ClusterConfidence(
        center=float(center), lower=float(lower), upper=float(upper), n=n, k=k
    )


def get_cluster_confidence(
    delta_threshold: float,
    cluster_points: pd.Series,
    ci_kind: CIKind = "wilson",
    alpha: float = 0.05,
    beta_prior: BetaPrior = (0.5, 0.5),
    beta_credible_level: float = 0.95,
) -> ClusterConfidence:
    points = pd.Series(cluster_points).dropna()
    n = int(points.shape[0])
    if n <= 0:
        return ClusterConfidence(
            center=float("nan"), lower=float("nan"), upper=float("nan"), n=0, k=0
        )

    k = int((points <= delta_threshold).sum())

    if ci_kind == "wilson":
        return _do_wilson_ci(k, n, alpha=alpha)

    if ci_kind == "beta":
        return _do_beta_ci(
            k,
            n,
            a=beta_prior[0],
            b=beta_prior[1],
            beta_credible_level=beta_credible_level,
        )

    raise ValueError(f"Unknown ci_kind: {ci_kind}")


def add_cluster_confidences(
    data: pd.DataFrame,
    cluster_col_name: str,
    clust_confidences: Dict[int, ClusterConfidence],
    threshold_confs: Optional[Dict[float, Dict[int, ClusterConfidence]]] = None,
) -> pd.DataFrame:
    """
    Adds confidence_center/lower/upper/n/k columns by mapping cluster id -> confidence stats.
    If threshold_confs is provided (mapping SD threshold -> cluster_confs), also adds
    confidence_lower_within_{threshold}_sd columns for each threshold.
    """
    out = data.copy()

    def _get(cluster: int, field: str, confs: Dict[int, ClusterConfidence]) -> float:
        cc = confs.get(int(cluster))
        return getattr(cc, field) if cc is not None else float("nan")

    out["confidence_lower"] = out[cluster_col_name].apply(
        lambda c: _get(c, "lower", clust_confidences)
    )
    out["confidence_upper"] = out[cluster_col_name].apply(
        lambda c: _get(c, "upper", clust_confidences)
    )
    out["confidence_center"] = out[cluster_col_name].apply(
        lambda c: _get(c, "center", clust_confidences)
    )
    out["confidence_n"] = out[cluster_col_name].apply(
        lambda c: _get(c, "n", clust_confidences)
    )
    out["confidence_k"] = out[cluster_col_name].apply(
        lambda c: _get(c, "k", clust_confidences)
    )

    if threshold_confs is not None:
        for threshold, confs in threshold_confs.items():
            label = str(threshold).replace(".", "p")
            col = f"confidence_lower_within_{label}_sd"
            out[col] = out[cluster_col_name].apply(lambda c: _get(c, "lower", confs))

    return out


def _beta_ppf(q: float, a: float, b: float) -> float:
    try:
        from scipy.stats import beta as sp_beta  # type: ignore

        return float(sp_beta.ppf(q, a, b))
    except Exception:
        rng = np.random.default_rng(0)
        samples = rng.beta(a, b, size=200_000)
        return float(np.quantile(samples, q))


def _visualise_confidence(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred_test: pd.Series,
    model_type: str,
) -> None:
    from visualisation import density_scatter

    sd_thresholds = [0.1, 0.3, 0.5, 0.75, 1.0]
    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data_exploration_visualisations",
        "models",
        f"recovery_{model_type}",
        "confidence",
    )
    os.makedirs(save_dir, exist_ok=True)

    sd = float(y_train.std())
    correctness_delta_abs = (y_test - y_pred_test).abs()
    correctness_delta_pct_sd = correctness_delta_abs / sd

    for threshold in sd_thresholds:
        cm, confs, features = confidence_model(
            to_train_cluster=X_train,
            to_get_confidences=X_test,
            predicted_y=y_pred_test,
            actual_y=y_test,
            features=list(X_train.columns),
            n_clusters=8,
            critical_percentage_delta=threshold,
            sd_reference_y=y_train,
            ci_kind="beta",
            beta_prior=(0.5, 0.5),
            beta_credible_level=0.95,
        )

        clusters = cm.predict(X_test[features])
        plot_df = X_test.copy()
        plot_df["clusters"] = clusters
        plot_df = add_cluster_confidences(plot_df, "clusters", confs)
        plot_df["correctness_delta_pct_sd"] = correctness_delta_pct_sd.values

        threshold_label = str(threshold).replace(".", "p")
        save_path = os.path.join(
            save_dir, f"confidence_lower_at_{threshold_label}sd.png"
        )
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


def do_confidence(
    X: pd.DataFrame,
    model_or_model_file_name: str | lightgbm.LGBMRegressor,
    model_type: str,
    visualise: bool = False,
) -> pd.DataFrame:
    """
    Fit a confidence model on the train/test split for ``model_type`` and
    apply it to ``X``, returning a DataFrame with cluster and confidence columns.
    """
    from data_managment import deterministic_encoded_train_test_split
    from model_save_load import load_and_predict

    X_train, X_test, y_train, y_test, _, _ = deterministic_encoded_train_test_split(
        model_type
    )

    if isinstance(model_or_model_file_name, str):
        model_name = model_or_model_file_name
        y_pred_test = load_and_predict(model_name, X_test)
    elif isinstance(model_or_model_file_name, lightgbm.LGBMRegressor):
        model = model_or_model_file_name
        y_pred_test = model.predict(X_test)
    else:
        raise ValueError(
            "model_or_model_name must be either a model filename (str) or a fitted LGBMRegressor instance."
        )

    SD_THRESHOLDS = [0.1, 0.3, 0.5, 0.75, 1.0]

    threshold_confs: Dict[float, Dict[int, ClusterConfidence]] = {}
    for t in SD_THRESHOLDS:
        t_cm, t_confs, _ = confidence_model(
            to_train_cluster=X_train,
            to_get_confidences=X_test,
            predicted_y=pd.Series(y_pred_test, index=X_test.index),
            actual_y=y_test,
            features=list(X_train.columns),
            n_clusters=8,
            critical_percentage_delta=t,
            sd_reference_y=y_train,
            ci_kind="beta",
            beta_prior=(0.5, 0.5),
            beta_credible_level=0.95,
        )
        threshold_confs[t] = t_confs

    cluster_model, cluster_confs, cluster_features = confidence_model(
        to_train_cluster=X_train,
        to_get_confidences=X_test,
        predicted_y=y_pred_test,
        actual_y=y_test,
        features=list(X_train.columns),
        n_clusters=8,
        critical_percentage_delta=0.75,
        sd_reference_y=y_train,
        ci_kind="beta",
        beta_prior=(0.5, 0.5),
        beta_credible_level=0.95,
    )

    clusters = cluster_model.predict(X[cluster_features])
    conf_df = X.copy()
    conf_df["clusters"] = clusters

    if visualise:
        _visualise_confidence(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            y_pred_test=pd.Series(y_pred_test, index=X_test.index),
            model_type=model_type,
        )
    return add_cluster_confidences(
        conf_df, "clusters", cluster_confs, threshold_confs=threshold_confs
    )
