import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List, Dict, Any, Optional


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan

    dominance = 0
    for xi in x:
        dominance += np.sum(xi > y) - np.sum(xi < y)

    return dominance / (n1 * n2)


def cramers_v(contingency_table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1

    if min_dim == 0 or n == 0:
        return np.nan

    return np.sqrt(chi2 / (n * min_dim))


def compare_numeric(group_a: pd.Series, group_b: pd.Series) -> Dict[str, Any]:
    a_clean = group_a.dropna()
    b_clean = group_b.dropna()

    if len(a_clean) < 2 or len(b_clean) < 2:
        return {"test": "insufficient_data", "p_value": np.nan, "effect_size": np.nan}

    stat, p_value = stats.mannwhitneyu(a_clean, b_clean, alternative="two-sided")
    effect = cliffs_delta(a_clean.values, b_clean.values)

    return {
        "test": "mann_whitney_u",
        "p_value": p_value,
        "effect_size": effect,
        "effect_type": "cliffs_delta",
        "median_a": a_clean.median(),
        "median_b": b_clean.median(),
        "viz": "boxplot",
    }


def compare_categorical(group_a: pd.Series, group_b: pd.Series) -> Dict[str, Any]:
    a_clean = group_a.dropna()
    b_clean = group_b.dropna()

    if len(a_clean) < 5 or len(b_clean) < 5:
        return {"test": "insufficient_data", "p_value": np.nan, "effect_size": np.nan}

    combined = pd.DataFrame(
        {
            "value": pd.concat([a_clean, b_clean]),
            "group": ["A"] * len(a_clean) + ["B"] * len(b_clean),
        }
    )
    contingency = pd.crosstab(combined["value"], combined["group"])

    contingency = contingency.loc[(contingency > 0).any(axis=1)]
    contingency = contingency.loc[:, (contingency > 0).any(axis=0)]

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {
            "test": "insufficient_categories",
            "p_value": np.nan,
            "effect_size": np.nan,
        }

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    effect = cramers_v(contingency)

    return {
        "test": "chi_square",
        "p_value": p_value,
        "effect_size": effect,
        "effect_type": "cramers_v",
        "viz": "stacked_bar",
    }


def analyze_condition_impact(
    df: pd.DataFrame,
    condition_col: str,
    condition_value: Any,
    exclude_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    exclude_cols = exclude_cols or []
    exclude_cols.append(condition_col)

    group_a = df[df[condition_col] == condition_value]
    group_b = df[df[condition_col] != condition_value]

    results = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        dtype = df[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            result = compare_numeric(group_a[col], group_b[col])
        elif pd.api.types.is_categorical_dtype(dtype) or dtype == object:
            result = compare_categorical(group_a[col], group_b[col])
        else:
            continue

        result["column"] = col
        result["n_condition"] = len(group_a)
        result["n_other"] = len(group_b)
        results.append(result)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values("p_value").reset_index(drop=True)

    return results_df


def compare_two_conditions(
    df: pd.DataFrame,
    condition_col: str,
    value_a: Any,
    value_b: Any,
    exclude_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    exclude_cols = exclude_cols or []
    exclude_cols.append(condition_col)

    group_a = df[df[condition_col] == value_a]
    group_b = df[df[condition_col] == value_b]

    results = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        dtype = df[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            result = compare_numeric(group_a[col], group_b[col])
        elif pd.api.types.is_categorical_dtype(dtype) or dtype == object:
            result = compare_categorical(group_a[col], group_b[col])
        else:
            continue

        result["column"] = col
        result["n_a"] = len(group_a)
        result["n_b"] = len(group_b)
        results.append(result)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values("p_value").reset_index(drop=True)

    return results_df


def compare_all_cols_to_target(data: pd.DataFrame, target_col: str):
    all_results = []
    for col in data.columns:
        if col != target_col:
            results = compare_two_columns(data, col, target_col)
            results["column"] = col
            all_results.append(results)

    results_df = pd.DataFrame(all_results)
    print_impact_summary(results_df, alpha=0.1)
    return results_df


def compare_all_cols_to_target_visualise(data: pd.DataFrame, target_col: str):
    results_df = compare_all_cols_to_target(data, target_col)
    visualise_impact_summary(results_df, alpha=0.1)


def compare_two_columns(df: pd.DataFrame, col_a: str, col_b: str) -> Dict[str, Any]:
    a_clean = df[col_a].dropna()
    b_clean = df[col_b].dropna()
    common_index = a_clean.index.intersection(b_clean.index)
    a_clean = a_clean.loc[common_index]
    b_clean = b_clean.loc[common_index]

    a_numeric = pd.api.types.is_numeric_dtype(a_clean)
    b_numeric = pd.api.types.is_numeric_dtype(b_clean)

    if a_numeric and b_numeric:
        corr, p_value = stats.spearmanr(a_clean, b_clean)
        return {
            "test": "spearman_correlation",
            "statistic": corr,
            "p_value": p_value,
            "effect_size": corr,
            "effect_type": "spearman_r",
            "n": len(a_clean),
            "viz": "scatter",
        }

    elif not a_numeric and not b_numeric:
        combined = pd.DataFrame({"a": a_clean.values, "b": b_clean.values})
        contingency = pd.crosstab(combined["a"], combined["b"])
        contingency = contingency.loc[(contingency > 0).any(axis=1)]
        contingency = contingency.loc[:, (contingency > 0).any(axis=0)]
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return {
                "test": "insufficient_categories",
                "p_value": np.nan,
                "effect_size": np.nan,
            }
        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
        effect = cramers_v(contingency)
        return {
            "test": "chi_square",
            "statistic": chi2,
            "p_value": p_value,
            "effect_size": effect,
            "effect_type": "cramers_v",
            "n": len(a_clean),
        }

    else:
        numeric_col, cat_col = (a_clean, b_clean) if a_numeric else (b_clean, a_clean)
        groups = [numeric_col[cat_col == cat].values for cat in cat_col.unique()]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            return {
                "test": "insufficient_data",
                "p_value": np.nan,
                "effect_size": np.nan,
            }
        stat, p_value = stats.kruskal(*groups)
        n = len(numeric_col)
        k = len(groups)
        eta_sq = (stat - k + 1) / (n - k) if n > k else np.nan
        return {
            "test": "kruskal_wallis",
            "statistic": stat,
            "p_value": p_value,
            "effect_size": eta_sq,
            "effect_type": "eta_squared",
            "n": n,
        }


def print_impact_summary(results: pd.DataFrame | dict, alpha: float = None) -> None:
    if isinstance(results, dict):
        results = pd.DataFrame([results])

    significant = results[results["p_value"] < alpha] if alpha else results

    print(f"{'='*60}")
    if alpha:
        print(f"SIGNIFICANT FINDINGS (p < {alpha}): {len(significant)}/{len(results)}")
    else:
        print(f"ALL FINDINGS: {len(results)} columns analyzed")

    print(f"{'='*60}\n")

    for _, row in significant.iterrows():
        effect_label = (
            "small"
            if abs(row["effect_size"]) < 0.3
            else "medium" if abs(row["effect_size"]) < 0.5 else "large"
        )

        print(f"Column: {row.get('column', 'N/A')}")
        print(f"  Test: {row['test']}, p-value: {row['p_value']:.10f}")
        print(f"  Effect size: {row['effect_size']:.3f} ({effect_label})")
        print(f"  Suggested viz: {row.get('viz', 'N/A')}")
        print()


def visualise_impact_summary(results, alpha: float = None) -> None:
    if isinstance(results, dict):
        results = pd.DataFrame([results])

    df = results.dropna(subset=["effect_size"]).copy()
    if df.empty:
        print("No results to visualise.")
        return

    df = df.sort_values("effect_size", key=abs, ascending=True).reset_index(drop=True)

    labels = df.get("column", pd.Series(["comparison"] * len(df))).fillna("comparison")
    effects = df["effect_size"].abs()
    p_values = df["p_value"]

    if alpha is not None:
        colours = ["steelblue" if p < alpha else "lightgrey" for p in p_values]
    else:
        colours = "steelblue"

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))
    bars = ax.barh(labels, effects, color=colours)

    ax.axvline(0.3, color="orange", linestyle="--", linewidth=1, label="small (0.3)")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="medium (0.5)")

    ax.set_xlabel("Effect Size (absolute)")
    ax.set_title("Impact Summary" + (f" (α={alpha})" if alpha else ""))
    ax.legend(fontsize=8)

    for i, (effect, p) in enumerate(zip(effects, p_values)):
        ax.text(effect + 0.005, i, f"p={p:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()
    plt.close(fig)
