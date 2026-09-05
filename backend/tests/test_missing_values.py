import numpy as np
import pandas as pd

from backend.app.missing_values import (
    recommend_strategy,
    apply_missing_value_treatment,
    _has_outliers_iqr,
)


def test_mcar_column_gets_basic_method(dataframe_with_missing_values):
    recos = recommend_strategy(dataframe_with_missing_values, datetime_cols=[], is_time_series=False)
    age_reco = next(r for r in recos if r["column"] == "age")
    assert age_reco["missingness_type"] == "MCAR"
    assert age_reco["p_value"] >= 0.05
    assert age_reco["recommended_method"] == "moyenne"


def test_mar_column_gets_advanced_method(dataframe_with_missing_values):
    recos = recommend_strategy(dataframe_with_missing_values, datetime_cols=[], is_time_series=False)
    ville_reco = next(r for r in recos if r["column"] == "ville")
    assert ville_reco["missingness_type"] == "MAR/MNAR"
    assert ville_reco["p_value"] < 0.05
    assert ville_reco["recommended_method"] in ("mode_par_groupe", "mode")


def test_columns_without_missing_values_are_skipped(clean_dataframe):
    recos = recommend_strategy(clean_dataframe, datetime_cols=[], is_time_series=False)
    assert recos == []


def test_apply_treatment_removes_all_missing_values(dataframe_with_missing_values):
    recos = recommend_strategy(dataframe_with_missing_values, datetime_cols=[], is_time_series=False)
    strategies = {r["column"]: r["recommended_method"] for r in recos}
    df_clean, log = apply_missing_value_treatment(dataframe_with_missing_values, datetime_cols=[], strategies=strategies)
    assert df_clean.isna().sum().sum() == 0
    assert len(log) == len(strategies)


def test_skip_option_leaves_column_untouched(dataframe_with_missing_values):
    strategies = {"age": "ne_pas_imputer"}
    df_result, log = apply_missing_value_treatment(dataframe_with_missing_values, datetime_cols=[], strategies=strategies)
    assert df_result["age"].isna().sum() == dataframe_with_missing_values["age"].isna().sum()
    assert log == []


def test_has_outliers_iqr_detects_real_outliers(dataframe_with_outliers):
    assert _has_outliers_iqr(dataframe_with_outliers["salaire"]) is True


def test_has_outliers_iqr_false_on_clean_normal_data(clean_dataframe):
    assert _has_outliers_iqr(clean_dataframe["age"]) is False


def test_suppression_reduces_row_count(dataframe_with_outliers):
    df = dataframe_with_outliers.copy()
    df.loc[[10, 20, 30], "salaire"] = np.nan
    df_result, log = apply_missing_value_treatment(df, datetime_cols=[], strategies={"salaire": "suppression"})
    assert df_result.shape[0] == df.shape[0] - 3
