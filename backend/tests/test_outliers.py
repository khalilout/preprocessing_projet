import numpy as np
import pandas as pd

from backend.app.outliers import recommend_outlier_strategy, apply_outlier_treatment


def test_symmetric_clean_column_recommends_zscore(clean_dataframe):
    recos = recommend_outlier_strategy(clean_dataframe)
    age_reco = next(r for r in recos if r["column"] == "age")
    assert age_reco["recommended_method"] == "zscore"


def test_outliers_detected_on_column_with_extremes(dataframe_with_outliers):
    recos = recommend_outlier_strategy(dataframe_with_outliers)
    salaire_reco = next(r for r in recos if r["column"] == "salaire")
    assert salaire_reco["n_outliers_detected"] >= 5


def test_apply_iqr_caps_values_within_bounds(dataframe_with_outliers):
    df_result, log = apply_outlier_treatment(dataframe_with_outliers, {"salaire": "iqr"})
    q1, q3 = dataframe_with_outliers["salaire"].quantile(0.25), dataframe_with_outliers["salaire"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    assert df_result["salaire"].max() <= upper_bound + 1e-6
    assert log[0]["column"] == "salaire"


def test_apply_winsorisation_caps_at_percentiles(dataframe_with_outliers):
    df_result, _ = apply_outlier_treatment(dataframe_with_outliers, {"salaire": "winsorisation"})
    expected_upper = dataframe_with_outliers["salaire"].quantile(0.95)
    assert df_result["salaire"].max() <= expected_upper + 1e-6


def test_skip_option_leaves_outliers_untouched(dataframe_with_outliers):
    df_result, log = apply_outlier_treatment(dataframe_with_outliers, {"salaire": "ne_pas_traiter"})
    pd.testing.assert_series_equal(df_result["salaire"], dataframe_with_outliers["salaire"])
    assert log == []


def test_apply_reduces_outlier_count(dataframe_with_outliers):
    before_recos = recommend_outlier_strategy(dataframe_with_outliers)
    before_count = next(r for r in before_recos if r["column"] == "salaire")["n_outliers_detected"]

    df_result, _ = apply_outlier_treatment(dataframe_with_outliers, {"salaire": "iqr"})
    after_recos = recommend_outlier_strategy(df_result)
    after_count = next((r for r in after_recos if r["column"] == "salaire"), {"n_outliers_detected": 0})["n_outliers_detected"]

    assert after_count < before_count