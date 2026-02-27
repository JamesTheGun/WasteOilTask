# using k-means because instead of something a like GMM because the data does no follow a guassian on any dimension
from typing import List

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

from data_managment import deterministic_encoded_train_test_split


def confidence_model(
    to_train_cluster: pd.DataFrame,
    to_get_confidences: pd.DataFrame,
    predicted_y: pd.Series,
    actual_y: pd.Series,
    critical_percentage_delta: float = None,
    critical_abs_delta: float = None,
) -> tuple[KMeans, pd.Series]:

    if critical_percentage_delta is None and critical_abs_delta is None:
        raise ValueError(
            "At least one of critical_percentage_delta or critical_abs_delta must be provided."
        )

    if critical_percentage_delta is not None and critical_abs_delta is not None:
        raise ValueError(
            "Only one of critical_percentage_delta or critical_abs_delta should be provided, not both."
        )
    to_get_confidences = to_get_confidences.copy()
    cluster_model, features = get_best_model_and_features(to_train_cluster)
    predicted_clusters = cluster_model.predict(to_get_confidences[features])

    is_percentage_delta = critical_percentage_delta is not None

    deltas = (actual_y - predicted_y).reset_index(drop=True)

    clusters_and_deltas = to_get_confidences.copy()
    clusters_and_deltas["clusters"] = predicted_clusters
    clusters_and_deltas["deltas"] = deltas
    get_cluster_points = lambda cluster: clusters_and_deltas["clusters"] == cluster

    cluster_Ps: dict[int, float] = {
        cluster: get_p_of_exceeding_delta_for_cluster(
            critical_percentage_delta,
            is_percentage_delta,
            get_cluster_points(cluster),
        )
        for cluster in np.unique(predicted_clusters)
    }
    return cluster_model, cluster_Ps, features


def get_p_of_exceeding_delta_for_cluster(
    delta_threshold: float, is_percentage_delta: bool, cluster_points: pd.Series
) -> pd.Series:
    # delta_threshold =
    if is_percentage_delta:
        cluster_points = cluster_points.apply(
            lambda x: abs(x) / abs(x - 1) if x - 1 != 0 else 0
        )
    print(f"cluster_points: {cluster_points}")
    F_hat = np.mean(cluster_points <= delta_threshold)
    n = len(cluster_points)
    prob_exceeding = 1 - F_hat**n
    print(prob_exceeding)
    return prob_exceeding


def get_best_model_and_features(
    target_features: pd.DataFrame,
) -> tuple[KMeans, pd.Series]:
    dimensions = target_features.columns.tolist()
    best_k, best_model, best_inertia, features = search_dimension_combinations(
        target_features, dimensions
    )
    return best_model, features


def search_dimension_combinations(
    df_with_target_feature: pd.DataFrame,
    dimensions: list[str],
    combo_size: int = 3,
    max_k: int = 10,
) -> tuple[int, KMeans, float, pd.Series]:
    if combo_size > len(dimensions):
        print(
            "WARNING: combo_size is greater than the number of available dimensions. Reducing combo_size to match the number of dimensions."
        )
        combo_size = len(dimensions)
    search_space = combo_size * (combo_size - 1) // 2
    print(f"Evaluating {search_space} combinations of {combo_size} dimensions...")
    if search_space > max_k:
        raise ValueError(
            f"Search space of {search_space} exceeds max_k of {max_k}. Consider reducing combo_size or increasing max_k, or pass a higher max_k"
        )
    dim_combos = [
        [dim_1, dim_2]
        for i, dim_1 in enumerate(dimensions)
        for dim_2 in dimensions[i + 1 :]
    ]
    results: list[tuple[int, KMeans, float, pd.Series]] = [
        (*evaluate_dimensions(df_with_target_feature, combo), pd.Series(combo))
        for combo in dim_combos
    ]
    best_model_and_params = max(results, key=lambda x: x[0])
    print(best_model_and_params)
    return best_model_and_params


def evaluate_dimensions(
    target_features: pd.DataFrame, dimension: List[str]
) -> tuple[int, KMeans, float]:
    features = target_features.copy()
    best_inertia, best_model, best_k = find_best_best_k_clusters(
        features, max_k=10, dimensions=dimension
    )
    return best_inertia, best_model, best_k


def find_best_best_k_clusters(
    target_features: pd.DataFrame, max_k: int, dimensions: list[str] = None
) -> tuple[int, KMeans, float]:
    features = target_features.copy()
    best_inertia = np.inf
    best_model = None
    for k in range(1, max_k + 1):
        kmeans = get_model(features, n_clusters=k, dimensions=dimensions)
        kmeans.fit(features if dimensions is None else features[dimensions])
        inertia = kmeans.inertia_
        if inertia < best_inertia:
            best_inertia = inertia
            best_k = k
            best_model = kmeans
    return best_inertia, best_model, best_k


def get_model(
    target_features: pd.DataFrame, n_clusters: int = 3, dimensions: list[str] = None
) -> KMeans:
    features = target_features.copy()
    if dimensions:
        features = features[dimensions]
    kmeans = KMeans(n_clusters=n_clusters, random_state=1)
    return kmeans


def add_pcas_of_features(target_features: pd.DataFrame) -> pd.DataFrame:
    from sklearn.decomposition import PCA

    features = target_features.copy()
    pca = PCA()
    pca_result = pca.fit_transform(features)
    pca_df = pd.DataFrame(
        pca_result, columns=[f"pca_{i+1}" for i in range(pca_result.shape[1])]
    )
    combined = pd.concat([features.reset_index(drop=True), pca_df], axis=1)
    return combined


def add_cluster_confidences(
    data: pd.DataFrame, cluster_col_name: str, clust_confidences: pd.Series
) -> pd.DataFrame:
    data = data.copy()
    data["confidence"] = data[cluster_col_name].map(clust_confidences)
    return data


if __name__ == "__main__":
    # load real data and perform a basic clustering confidence check
    import pandas as pd
    import numpy as np

    X_train, X_test, y_train, y_test, not_selected_train, not_selected_test = (
        deterministic_encoded_train_test_split("ratio_with_supplied_volume")
    )

    # load pre‑trained recovery ratio model and predict on encoded features
    from model_save_load import load_and_predict

    # from model_constants import RATIO_MODEL_FILENAME
    from model_constants import RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME

    y_pred_test = pd.Series(
        load_and_predict(RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME, X_test),
        index=X_test.index,
    )

    y_pred_train = pd.Series(
        load_and_predict(RATIO_WITH_SUPPLIED_VOLUME_MODEL_FILENAME, X_train),
        index=X_train.index,
    )

    cluster_model_test, cluster_Ps_test, cluster_features_test = confidence_model(
        X_train,
        X_test,
        y_pred_test,
        y_test,
        critical_percentage_delta=0.2,
    )

    cluster_model_train, cluster_Ps_train, cluster_features_train = confidence_model(
        X_train,
        X_train,
        y_pred_train,
        y_train,
        critical_percentage_delta=0.2,
    )

    # optionally visualise clusters using the scatter plot helper
    from visualisation import scatter_coloured, scatter_confidence

    clusters = cluster_model_test.predict(X_test[cluster_features_test])
    viz_df = X_test.copy()
    pcaed_viz_df = add_pcas_of_features(viz_df)
    pcaed_viz_df["clusters"] = clusters
    data_with_confidences = add_cluster_confidences(
        pcaed_viz_df, "clusters", cluster_Ps_test
    )
    data_with_confidences = add_cluster_confidences(
        pcaed_viz_df, "clusters", cluster_Ps_train
    )
    print(cluster_Ps_train)
    data_with_trues_and_preds = data_with_confidences.copy()
    data_with_trues_and_preds["true"] = y_test.reset_index(drop=True)
    data_with_trues_and_preds["pred"] = y_pred_test.reset_index(drop=True)
    correctness_delta = abs(y_test - y_pred_test)
    data_with_trues_and_preds["correctness_delta"] = correctness_delta.reset_index(
        drop=True
    )
    scatter_coloured(
        data_with_trues_and_preds,
        x_col="confidence",
        y_col="correctness_delta",
        colour_col="clusters",
    )
