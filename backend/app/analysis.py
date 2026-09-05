"""
Étape 1 — ANALYSE INITIALE DES DONNÉES

Reproduit le bloc "1. ANALYSE INITIALE" du schéma :
- types de colonnes
- taux de valeurs manquantes
- symétrie (skewness)
- corrélation
- détection time series
"""
import pandas as pd
import numpy as np

from .schemas import ColumnInfo, InitialAnalysisResponse


def infer_column_type(series: pd.Series) -> str:
    """Détermine un type 'métier' plus lisible que le dtype pandas brut."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "booléenne"
    if pd.api.types.is_numeric_dtype(series):
        return "numérique"
    return "catégorielle"


def try_parse_datetime(series: pd.Series) -> bool:
    """
    Tente de détecter si une colonne object est en fait une date
    (ex: '2023-01-15' stockée en texte). On ne modifie pas la colonne ici,
    on se contente de tester si le parsing réussit sur un échantillon.
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    sample = series.dropna().head(50)
    if sample.empty:
        return False
    try:
        pd.to_datetime(sample, errors="raise", format="mixed")
        return True
    except (ValueError, TypeError):
        return False


def detect_time_series(df: pd.DataFrame, datetime_cols: list[str]) -> bool:
    """
    Une table est considérée comme 'time series' si :
    - au moins une colonne datetime existe, ET
    - les valeurs de cette colonne sont (quasi) uniques et ordonnables
      (ce qui exclut par ex. une colonne 'date de naissance' répétée sans logique temporelle globale)
    """
    if not datetime_cols:
        return False
    for col in datetime_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        uniqueness_ratio = parsed.nunique(dropna=True) / max(len(parsed), 1)
        if uniqueness_ratio > 0.8:
            return True
    return False


def analyze_dataframe(df: pd.DataFrame) -> InitialAnalysisResponse:
    columns_info: list[ColumnInfo] = []
    datetime_cols: list[str] = []

    for col in df.columns:
        series = df[col]

        is_dt = pd.api.types.is_datetime64_any_dtype(series) or try_parse_datetime(series)
        if is_dt:
            datetime_cols.append(col)

        inferred_type = "datetime" if is_dt else infer_column_type(series)

        missing_count = int(series.isna().sum())
        missing_rate = round(missing_count / len(df), 4) if len(df) else 0.0

        skewness = None
        if pd.api.types.is_numeric_dtype(series) and series.dropna().shape[0] > 2:
            skewness = round(float(series.skew()), 4)

        columns_info.append(
            ColumnInfo(
                name=col,
                dtype=str(series.dtype),
                inferred_type=inferred_type,
                missing_count=missing_count,
                missing_rate=missing_rate,
                n_unique=int(series.nunique(dropna=True)),
                skewness=skewness,
            )
        )

    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = None
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(numeric_only=True).round(3)
        corr = corr.where(pd.notnull(corr), None)
        correlation_matrix = corr.to_dict()

    is_ts = detect_time_series(df, datetime_cols)

    return InitialAnalysisResponse(
        n_rows=df.shape[0],
        n_columns=df.shape[1],
        columns=columns_info,
        correlation_matrix=correlation_matrix,
        is_time_series=is_ts,
        time_series_columns=datetime_cols,
        duplicated_rows=int(df.duplicated().sum()),
        memory_usage_mb=round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
    )
