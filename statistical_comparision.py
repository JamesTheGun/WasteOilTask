import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, List, Dict, Any, Optional


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cliff's delta effect size for two groups."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan

    dominance = 0
    for xi in x:
        dominance += np.sum(xi > y) - np.sum(xi < y)

    return dominance / (n1 * n2)


def cramers_v(contingency_table: pd.DataFrame) -> float:
    """Calculate Cramér's V effect size for categorical association."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1

    if min_dim == 0 or n == 0:
        return np.nan

    return np.sqrt(chi2 / (n * min_dim))


def compare_numeric(group_a: pd.Series, group_b: pd.Series) -> Dict[str, Any]:
    """Compare two groups on a numeric column."""
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
    """Compare two groups on a categorical column."""
    a_clean = group_a.dropna()
    b_clean = group_b.dropna()

    if len(a_clean) < 5 or len(b_clean) < 5:
        return {"test": "insufficient_data", "p_value": np.nan, "effect_size": np.nan}

    # Build contingency table
    combined = pd.DataFrame(
        {
            "value": pd.concat([a_clean, b_clean]),
            "group": ["A"] * len(a_clean) + ["B"] * len(b_clean),
        }
    )
    contingency = pd.crosstab(combined["value"], combined["group"])

    # Remove zero rows/cols
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
    """
    Analyze how a condition affects all other columns.

    Parameters:
        df: DataFrame to analyze
        condition_col: Column containing the condition
        condition_value: Value to compare (this group vs all others)
        exclude_cols: Columns to skip

    Returns:
        DataFrame with statistical comparison results for each column
    """
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
    """
    Compare two specific condition values against each other.
    """
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


def print_impact_summary(results: pd.DataFrame, alpha: float = None) -> None:
    """Print a readable summary of impact analysis results."""
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

        print(f"Column: {row['column']}")
        print(f"  Test: {row['test']}, p-value: {row['p_value']:.10f}")
        print(f"  Effect size: {row['effect_size']:.3f} ({effect_label})")
        print(f"  Suggested viz: {row.get('viz', 'N/A')}")
        print()
