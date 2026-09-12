from __future__ import annotations

import pandas as pd


def has_outliers_iqr(series: pd.Series, min_proportion: float = 0.01) -> bool:
    clean = series.dropna()
    if len(clean) == 0:
        return False

    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return False

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    n_outliers = ((clean < lower_bound) | (clean > upper_bound)).sum()

    return bool((n_outliers / len(clean)) > min_proportion)