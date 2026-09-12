from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import ExtraTreesRegressor
from .stats_utils import has_outliers_iqr as _has_outliers_iqr

PVALUE_THRESHOLD = 0.05
SKEW_SYMMETRY_THRESHOLD = 0.5


def _missingness_min_pvalue(df: pd.DataFrame, target_col: str) -> float:
    is_missing = df[target_col].isna()

    if is_missing.nunique() < 2:
        return 1.0

    p_values = []

    for other_col in df.columns:
        if other_col == target_col:
            continue
        other = df[other_col]

        try:
            if pd.api.types.is_numeric_dtype(other):
                group_missing = other[is_missing].dropna()
                group_present = other[~is_missing].dropna()
                if len(group_missing) < 2 or len(group_present) < 2:
                    continue
                _, p = stats.ttest_ind(group_missing, group_present, equal_var=False, nan_policy="omit")
            else:
                contingency = pd.crosstab(is_missing, other.astype(str))
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                _, p, _, _ = stats.chi2_contingency(contingency)

            if pd.notna(p):
                p_values.append(p)
        except (ValueError, ZeroDivisionError):
            continue

    return float(min(p_values)) if p_values else 1.0


def _most_associated_categorical_column(df: pd.DataFrame, target_col: str) -> str | None:
    best_col, best_p = None, 1.0

    for other_col in df.columns:
        if other_col == target_col:
            continue
        try:
            other = df[other_col]
            if pd.api.types.is_numeric_dtype(other):
                if other.nunique(dropna=True) < 4:
                    continue
                grouping_values = pd.qcut(other, q=4, duplicates="drop")
            else:
                grouping_values = other.astype(str)

            contingency = pd.crosstab(df[target_col].astype(str), grouping_values)
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue
            _, p, _, _ = stats.chi2_contingency(contingency)
            if p < best_p:
                best_col, best_p = other_col, p
        except (ValueError, ZeroDivisionError):
            continue

    return best_col


def recommend_strategy(
    df: pd.DataFrame,
    datetime_cols: list[str],
    is_time_series: bool,
) -> list[dict]:
    recommendations = []

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        if missing_count == 0:
            continue

        missing_rate = round(missing_count / len(df), 4)
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_datetime = col in datetime_cols

        if is_datetime:
            recommendations.append({
                "column": col, "missing_rate": missing_rate, "p_value": None,
                "missingness_type": "N/A",
                "recommended_method": "aucune (colonne temporelle)",
                "rationale": "Les colonnes datetime ne sont pas imputées automatiquement.",
            })
            continue

        p_value = round(_missingness_min_pvalue(df, col), 4)
        is_mcar = p_value >= PVALUE_THRESHOLD
        missingness_type = "MCAR" if is_mcar else "MAR/MNAR"

        if is_numeric:
            has_outliers = _has_outliers_iqr(series)
            symmetric = abs(series.dropna().skew()) < SKEW_SYMMETRY_THRESHOLD if series.dropna().shape[0] > 2 else True

            if is_mcar:
                if is_time_series:
                    method = "interpolation_lineaire"
                    rationale = f"p={p_value} >= 0.05 (MCAR/MAR) + série temporelle -> interpolation linéaire (bfill/ffill en alternative)."
                elif has_outliers:
                    method = "suppression"
                    rationale = f"p={p_value} >= 0.05 (MCAR/MAR), présence d'outliers -> suppression des lignes (moyenne/médiane seraient faussées)."
                elif symmetric:
                    method = "moyenne"
                    rationale = f"p={p_value} >= 0.05 (MCAR/MAR), pas d'outlier, distribution symétrique -> moyenne."
                else:
                    method = "mediane"
                    rationale = f"p={p_value} >= 0.05 (MCAR/MAR), pas d'outlier, distribution asymétrique -> médiane."
            else:
                if is_time_series:
                    method = "groupe_temporel"
                    rationale = f"p={p_value} < 0.05 (MAR/MNAR) + série temporelle -> imputation par statistiques du groupe (période)."
                elif has_outliers:
                    method = "knn_imputer"
                    rationale = f"p={p_value} < 0.05 (MAR/MNAR), présence d'outliers, variable corrélée -> KNNImputer (tolère les outliers)."
                else:
                    method = "iterative_imputer"
                    rationale = f"p={p_value} < 0.05 (MAR/MNAR), pas d'outlier, variable corrélée -> IterativeImputer (ExtraTrees)."

            recommendations.append({
                "column": col, "missing_rate": missing_rate, "p_value": p_value,
                "missingness_type": missingness_type,
                "recommended_method": method, "rationale": rationale,
            })

        else:
            if is_mcar:
                method = "mode"
                rationale = f"p={p_value} >= 0.05 (MCAR) -> imputation par le mode."
            else:
                grouping_col = _most_associated_categorical_column(df, col)
                if grouping_col:
                    method = "mode_par_groupe"
                    rationale = f"p={p_value} < 0.05 (MAR/MNAR), forte association avec '{grouping_col}' -> mode calculé par groupe de '{grouping_col}'."
                else:
                    method = "mode"
                    rationale = f"p={p_value} < 0.05 (MAR/MNAR) mais aucune colonne associée exploitable -> mode global par défaut."

            recommendations.append({
                "column": col, "missing_rate": missing_rate, "p_value": p_value,
                "missingness_type": missingness_type,
                "recommended_method": method, "rationale": rationale,
            })

    return recommendations


def apply_missing_value_treatment(
    df: pd.DataFrame,
    datetime_cols: list[str],
    strategies: dict[str, str],
) -> tuple[pd.DataFrame, list[dict]]:
    df = df.copy()
    log = []

    simple_methods = {"moyenne", "mediane", "mode", "interpolation_lineaire", "bfill", "ffill"}
    knn_cols = [c for c, m in strategies.items() if m == "knn_imputer"]
    iterative_cols = [c for c, m in strategies.items() if m == "iterative_imputer"]
    group_temporal_cols = [c for c, m in strategies.items() if m == "groupe_temporel"]
    mode_group_cols = [c for c, m in strategies.items() if m == "mode_par_groupe"]
    deletion_cols = [c for c, m in strategies.items() if m == "suppression"]

    for col in deletion_cols:
        if col not in df.columns:
            continue
        before = int(df[col].isna().sum())
        df = df.dropna(subset=[col])
        log.append({"column": col, "method": "suppression", "values_imputed": before})

    for col, method in strategies.items():
        if method not in simple_methods or col not in df.columns:
            continue
        before = int(df[col].isna().sum())
        if before == 0:
            continue

        if method == "moyenne":
            df[col] = df[col].fillna(df[col].mean())
        elif method == "mediane":
            df[col] = df[col].fillna(df[col].median())
        elif method == "mode":
            mode_val = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Inconnu")
        elif method == "interpolation_lineaire":
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
        elif method == "bfill":
            df[col] = df[col].bfill()
        elif method == "ffill":
            df[col] = df[col].ffill()

        log.append({"column": col, "method": method, "values_imputed": before})

    if knn_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        imputer = KNNImputer(n_neighbors=5)
        imputed_array = imputer.fit_transform(df[numeric_cols])
        imputed_df = pd.DataFrame(imputed_array, columns=numeric_cols, index=df.index)
        for col in knn_cols:
            before = int(df[col].isna().sum())
            df[col] = imputed_df[col]
            log.append({"column": col, "method": "knn_imputer", "values_imputed": before})

    if iterative_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=50, random_state=42),
            random_state=42, max_iter=10,
        )
        imputed_array = imputer.fit_transform(df[numeric_cols])
        imputed_df = pd.DataFrame(imputed_array, columns=numeric_cols, index=df.index)
        for col in iterative_cols:
            before = int(df[col].isna().sum())
            df[col] = imputed_df[col]
            log.append({"column": col, "method": "iterative_imputer", "values_imputed": before})

    if group_temporal_cols and datetime_cols:
        period_col = datetime_cols[0]
        period_series = pd.to_datetime(df[period_col], errors="coerce").dt.to_period("M")
        for col in group_temporal_cols:
            before = int(df[col].isna().sum())
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df.groupby(period_series)[col].transform(lambda s: s.fillna(s.mean()))
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df.groupby(period_series)[col].transform(
                    lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Inconnu")
                )
            log.append({"column": col, "method": "groupe_temporel", "values_imputed": before})

    for col in mode_group_cols:
        if col not in df.columns:
            continue
        before = int(df[col].isna().sum())
        grouping_col = _most_associated_categorical_column(df, col)
        if grouping_col:
            if pd.api.types.is_numeric_dtype(df[grouping_col]):
                grouping_values = pd.qcut(df[grouping_col], q=4, duplicates="drop")
            else:
                grouping_values = df[grouping_col]
            df[col] = df.groupby(grouping_values, observed=True)[col].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else np.nan)
            )
        global_mode = df[col].mode(dropna=True)
        df[col] = df[col].fillna(global_mode.iloc[0] if not global_mode.empty else "Inconnu")
        log.append({"column": col, "method": "mode_par_groupe", "values_imputed": before})

    return df, log
