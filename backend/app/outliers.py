from __future__ import annotations

import numpy as np
import pandas as pd

SKEW_ZSCORE_THRESHOLD = 0.5
SKEW_WINSOR_THRESHOLD = 1.5
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3
WINSOR_LOWER_PCT = 0.05
WINSOR_UPPER_PCT = 0.95


def recommend_outlier_strategy(df: pd.DataFrame) -> list[dict]:
    recommendations = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if series.shape[0] < 3:
            continue
        skew = float(series.skew())
        abs_skew = abs(skew)

        if abs_skew < SKEW_ZSCORE_THRESHOLD:
            method, rationale = "zscore", f"Distribution symétrique/quasi-normale (skew={skew:.2f}) -> Z-score."
        elif abs_skew >= SKEW_WINSOR_THRESHOLD:
            method, rationale = "winsorisation", f"Distribution très asymétrique (skew={skew:.2f}) -> Winsorisation (plafonnement 5%/95%)."
        else:
            method, rationale = "iqr", f"Distribution asymétrique modérée (skew={skew:.2f}) -> règle IQR."

        n_outliers = _count_outliers(series, method)

        recommendations.append({
            "column": col,
            "skewness": round(skew, 4),
            "n_outliers_detected": n_outliers,
            "recommended_method": method,
            "rationale": rationale,
        })

    return recommendations


def _count_outliers(series: pd.Series, method: str) -> int:
    if method == "zscore":
        z = (series - series.mean()) / series.std(ddof=0)
        return int((z.abs() > ZSCORE_THRESHOLD).sum())
    elif method == "iqr":
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
        return int(((series < lower) | (series > upper)).sum())
    elif method == "winsorisation":
        lower, upper = series.quantile(WINSOR_LOWER_PCT), series.quantile(WINSOR_UPPER_PCT)
        return int(((series < lower) | (series > upper)).sum())
    return 0


def apply_outlier_treatment(df: pd.DataFrame, strategies: dict[str, str]) -> tuple[pd.DataFrame, list[dict]]:
    """Applique le traitement des outliers colonne par colonne, par plafonnement (capping)."""
    df = df.copy()
    log = []

    for col, method in strategies.items():
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col]

        if method == "zscore":
            mean, std = series.mean(), series.std(ddof=0)
            if std == 0:
                continue
            z = (series - mean) / std
            outlier_mask = z.abs() > ZSCORE_THRESHOLD
            df.loc[outlier_mask, col] = series.median()

        elif method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
            outlier_mask = (series < lower) | (series > upper)
            df[col] = series.clip(lower=lower, upper=upper)

        elif method == "winsorisation":
            lower, upper = series.quantile(WINSOR_LOWER_PCT), series.quantile(WINSOR_UPPER_PCT)
            outlier_mask = (series < lower) | (series > upper)
            df[col] = series.clip(lower=lower, upper=upper)

        else:
            continue

        log.append({
            "column": col,
            "method": method,
            "values_treated": int(outlier_mask.sum()),
        })

    return df, log
