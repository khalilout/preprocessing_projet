"""
Étape 4 — FEATURE SCALING (MISE À L'ÉCHELLE)

  Pour chaque colonne numérique :

      1) Présence d'outliers (règle IQR) ?
         OUI, peu importe la symétrie -> RobustScaler
             (basé sur médiane/IQR, insensible aux valeurs extrêmes)

      2) Sinon, vérifier la symétrie :
         Symétrique  (|skew| < 0.5) -> StandardScaler (centre-réduit)
         Asymétrique (|skew| >= 0.5) -> MinMaxScaler (ramène entre [0,1])

  Une colonne peut aussi être exclue explicitement ("aucun" / ne pas scaler).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

SKEW_SYMMETRY_THRESHOLD = 0.5

SCALERS = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
    "robust": RobustScaler,
}


def _has_outliers_iqr(series: pd.Series, min_proportion: float = 0.01) -> bool:
    """
    Un point isolé au-delà de 1.5*IQR est statistiquement normal même sur une
    vraie loi normale (~0.7% des points). On ne considère qu'il y a "des outliers"
    que si leur proportion dépasse un seuil (1% par défaut), pour éviter de
    déclencher RobustScaler sur des données parfaitement propres.
    """
    s = series.dropna()
    if s.shape[0] < 4:
        return False
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return False
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_proportion = ((s < lower) | (s > upper)).mean()
    return bool(outlier_proportion > min_proportion)


def recommend_scaling_strategy(df: pd.DataFrame) -> list[dict]:
    recommendations = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if series.shape[0] < 3:
            continue
        skew = float(series.skew())
        abs_skew = abs(skew)
        has_outliers = _has_outliers_iqr(series)

        if has_outliers:
            method = "robust"
            rationale = f"Outliers détectés (skew={skew:.2f}) -> RobustScaler (robuste aux valeurs extrêmes), quelle que soit la symétrie."
        elif abs_skew < SKEW_SYMMETRY_THRESHOLD:
            method = "standard"
            rationale = f"Pas d'outlier, distribution symétrique (skew={skew:.2f}) -> StandardScaler."
        else:
            method = "minmax"
            rationale = f"Pas d'outlier, distribution asymétrique (skew={skew:.2f}) -> MinMaxScaler."

        recommendations.append({
            "column": col,
            "skewness": round(skew, 4),
            "recommended_method": method,
            "rationale": rationale,
        })

    return recommendations


def apply_scaling(df: pd.DataFrame, strategies: dict[str, str]) -> tuple[pd.DataFrame, list[dict]]:
    df = df.copy()
    log = []

    by_method: dict[str, list[str]] = {}
    for col, method in strategies.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and method in SCALERS:
            by_method.setdefault(method, []).append(col)

    for method, cols in by_method.items():
        scaler = SCALERS[method]()
        df[cols] = scaler.fit_transform(df[cols])
        for col in cols:
            log.append({"column": col, "method": method})

    return df, log