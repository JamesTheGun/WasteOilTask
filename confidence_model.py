from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import statsmodels.stats.proportion as smp
import lightgbm
from visualisation import _make_model_vis_dir, scatter_confidence
from model_constants import MODEL_TYPE_DISPLAY_NAMES

CIKind = Literal["wilson", "beta"]
BetaPrior = Tuple[float, float]


@dataclass(frozen=True)
class ClusterConfidence:
    center: float
    lower: float
    upper: float
    n: int
    k: int


@dataclass
class ConfidenceResult:
    results: Dict[float, pd.DataFrame]
    primary_threshold: float
    threshold_type: Literal["pct", "abs"]
    cluster_model: object
    cluster_confidences_by_threshold: Dict[float, Dict[int, "ClusterConfidence"]]
    features: List[str]

    @property
    def thresholds(self) -> List[float]:
        return list(self.results.keys())

    @property
    def primary_df(self) -> pd.DataFrame:
        return self.results[self.primary_threshold]

    @property
    def clusters(self) -> pd.Series:
        return self.primary_df["cluster"]

    @property
    def confidence_lower(self) -> pd.Series:
        return self.primary_df["confidence_lower"]

    @property
    def confidence_center(self) -> pd.Series:
        return self.primary_df["confidence_center"]

    @property
    def confidence_upper(self) -> pd.Series:
        return self.primary_df["confidence_upper"]

    @property
    def threshold_pct_sd(self) -> Optional[float]:
        return self.primary_threshold if self.threshold_type == "pct" else None

    @property
    def threshold_abs(self) -> Optional[float]:
        return self.primary_threshold if self.threshold_type == "abs" else None

    def for_threshold(self, t: float) -> pd.DataFrame:
        return self.results[t]

    def apply_to(
        self, X: pd.DataFrame, threshold: float = None
    ) -> "ConfidenceAssignment":
        t = threshold if threshold is not None else self.primary_threshold
        confs = self.cluster_confidences_by_threshold[t]
        clusters = self.cluster_model.predict(X[self.features])
        df = pd.DataFrame(
            {
                "cluster": [int(c) for c in clusters],
                "confidence_center": [float(confs[int(c)].center) for c in clusters],
                "confidence_lower": [float(confs[int(c)].lower) for c in clusters],
                "confidence_upper": [float(confs[int(c)].upper) for c in clusters],
            },
            index=X.index,
        )
        tpct = t if self.threshold_type == "pct" else None
        tabs = t if self.threshold_type == "abs" else None
        return ConfidenceAssignment(df, threshold_pct_sd=tpct, threshold_abs=tabs)


@dataclass
class ConfidenceAssignment:
    _df: pd.DataFrame
    threshold_pct_sd: Optional[float]
    threshold_abs: Optional[float]

    @property
    def clusters(self) -> pd.Series:
        return self._df["cluster"]

    @property
    def confidence_center(self) -> pd.Series:
        return self._df["confidence_center"]

    @property
    def confidence_lower(self) -> pd.Series:
        return self._df["confidence_lower"]

    @property
    def confidence_upper(self) -> pd.Series:
        return self._df["confidence_upper"]


def _validate_params(critical_percentage_delta, critical_abs_delta):
    if (critical_percentage_delta is None) == (critical_abs_delta is None):
        raise ValueError(
            "Provide exactly one of critical_percentage_delta or critical_abs_delta."
        )


def _get_deltas_for_percentage(deltas_abs, sd_reference_y):
    sd = float(pd.Series(sd_reference_y).std())
    if sd <= 0 or not np.isfinite(sd):
        raise ValueError(f"Invalid sd_reference_y std: {sd}")
    deltas = deltas_abs / sd
    return deltas


def _get_deltas_for_abs(deltas_abs, critical_abs_delta):
    deltas = deltas_abs
    critical_delta = float(critical_abs_delta)
    return deltas, critical_delta


def make_cluster_model(
    to_train_cluster: pd.DataFrame,
    to_get_confidences: pd.DataFrame,
    *,
    features: Sequence[str],
    n_clusters: int,
    critical_percentage_delta: float | None = None,
    critical_abs_delta: float | None = None,
    random_state: int = 1,
    n_init: int | str = "auto",
) -> tuple[Pipeline, list[str]]:

    _validate_params(
        critical_percentage_delta=critical_percentage_delta,
        critical_abs_delta=critical_abs_delta,
    )

    X_train = to_train_cluster.copy()

    cluster_model: Pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "kmeans",
                KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init),
            ),
        ]
    )
    cluster_model.fit(X_train[features])

    return cluster_model, features


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


def make_nan_confidence() -> ClusterConfidence:
    return ClusterConfidence(
        center=float("nan"), lower=float("nan"), upper=float("nan"), n=0, k=0
    )


OVERIDE_THINGO = True


def get_cluster_confidence(
    delta_threshold: float,
    cluster_deltas: pd.Series,
    ci_kind: CIKind = "wilson",
    alpha: float = 0.05,
    beta_prior: BetaPrior = (0.5, 0.5),
    beta_credible_level: float = 0.95,
) -> ClusterConfidence:
    points = pd.Series(cluster_deltas).dropna()
    k = int((points <= delta_threshold).sum())

    if k <= 0 and not OVERIDE_THINGO:
        print(
            f"WARNING: No points within threshold {delta_threshold} in cluster, returning NaN conf."
        )
        return make_nan_confidence()

    if k >= points.shape[0] and not OVERIDE_THINGO:
        print(
            f"WARNING: All points within threshold {delta_threshold} in cluster, returning NaN conf."
        )
        return make_nan_confidence()

    n = int(points.shape[0])
    if n <= 0:
        print(
            f"WARNING: No points in cluster to calculate confidence, returning NaN conf."
        )
        return make_nan_confidence()

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


def _beta_ppf(q: float, a: float, b: float) -> float:
    try:
        from scipy.stats import beta as sp_beta  # type: ignore

        return float(sp_beta.ppf(q, a, b))
    except Exception:
        rng = np.random.default_rng(0)
        samples = rng.beta(a, b, size=200_000)
        return float(np.quantile(samples, q))


def confirm_good_for_threshold_type(
    threshold_type: Literal["abs", "pct"],
    correctness_delta_thresholds_pct: Optional[list] = None,
    correctness_delta_thresholds_abs: Optional[list] = None,
):
    if threshold_type == "pct":
        if correctness_delta_thresholds_pct is None:
            raise ValueError(
                "For pct threshold_type, provide correctness_delta_thresholds_pct."
            )
    elif threshold_type == "abs":
        if correctness_delta_thresholds_abs is None:
            raise ValueError(
                "For abs threshold_type, provide correctness_delta_thresholds_abs."
            )
    else:
        raise ValueError(f"Invalid threshold_type: {threshold_type}")


def make_deltas_for_type(
    type: Literal["abs", "pct"],
    y_true: pd.Series,
    y_pred,
    y_for_sd: Optional[pd.Series] = None,
):
    if y_for_sd is None:
        y_for_sd = y_true
    deltas_abs = (y_true - y_pred).abs()
    if type == "pct":
        return _get_deltas_for_percentage(deltas_abs, y_for_sd)
    elif type == "abs":
        return _get_deltas_for_abs(deltas_abs)


def make_confidence_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_test_pred: pd.Series,
    model_type: str,
    threshold_type: Literal["abs", "pct"],
    correctness_delta_thresholds_pct: Optional[list] = [0.1, 0.3, 0.5, 0.75, 1.0],
    correctness_delta_thresholds_abs: Optional[list] = None,
    visualise_model: bool = True,
) -> ConfidenceResult:
    confirm_good_for_threshold_type(
        threshold_type,
        correctness_delta_thresholds_pct=correctness_delta_thresholds_pct,
        correctness_delta_thresholds_abs=correctness_delta_thresholds_abs,
    )

    results = _generate_results(
        X_train,
        X_test,
        y_train,
        y_test,
        y_test_pred,
        threshold_type,
        correctness_delta_thresholds_pct=correctness_delta_thresholds_pct,
        correctness_delta_thresholds_abs=correctness_delta_thresholds_abs,
    )

    if results and visualise_model:
        delta_test = make_deltas_for_type(
            type=threshold_type,
            y_true=y_test,
            y_pred=y_test_pred,
            y_for_sd=y_train,
        )
        for threshold in results.thresholds:
            scatter_confidence(
                delta_test,
                results,
                colour_col="cluster",
                model_type=model_type,
                threshold=threshold,
            )

    return results


def _generate_results(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred_test: pd.Series,
    threshold_type: Literal["abs", "pct"],
    correctness_delta_thresholds_pct: Optional[list] = [0.1, 0.3, 0.5, 0.75, 1.0],
    correctness_delta_thresholds_abs: Optional[list] = None,
) -> ConfidenceResult:

    delta_test = make_deltas_for_type(
        type=threshold_type, y_true=y_test, y_pred=y_pred_test, y_for_sd=y_train
    )

    df_with_delta_test = X_test.copy()
    df_with_delta_test["deltas"] = delta_test.values

    cluster_model = None
    features = None
    predicted_clusters = None
    thresholds = (
        correctness_delta_thresholds_pct
        if threshold_type == "pct"
        else correctness_delta_thresholds_abs
    )

    results_at_thresholds = {}
    cluster_confidences_by_threshold = {}

    for theshold in thresholds:
        if predicted_clusters is None:
            cluster_model, features = make_cluster_model(
                X_train,
                X_test,
                features=X_train.columns.tolist(),
                n_clusters=5,
                critical_abs_delta=theshold,
            )
            predicted_clusters = cluster_model.predict(X_test[features])

        cluster_confidences = {}
        for cluster in np.unique(predicted_clusters):
            cluster_deltas = df_with_delta_test[predicted_clusters == cluster]["deltas"]
            confidence = get_cluster_confidence(
                theshold,
                cluster_deltas,
                ci_kind="wilson",
                alpha=0.05,
                beta_prior=(0.5, 0.5),
                beta_credible_level=0.95,
            )
            cluster_confidences[int(cluster)] = confidence

        cluster_confidences_by_threshold[theshold] = cluster_confidences

        results_dict = pd.DataFrame(
            {
                "cluster": predicted_clusters,
                "confidence_upper": [
                    cluster_confidences[int(c)].upper for c in predicted_clusters
                ],
                "confidence_center": [
                    cluster_confidences[int(c)].center for c in predicted_clusters
                ],
                "confidence_lower": [
                    cluster_confidences[int(c)].lower for c in predicted_clusters
                ],
            }
        )
        results_at_thresholds[theshold] = results_dict

    return ConfidenceResult(
        results=results_at_thresholds,
        primary_threshold=thresholds[0],
        threshold_type=threshold_type,
        cluster_model=cluster_model,
        cluster_confidences_by_threshold=cluster_confidences_by_threshold,
        features=features,
    )
