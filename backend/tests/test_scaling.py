import numpy as np
import pandas as pd

from backend.app.scaling import recommend_scaling_strategy, apply_scaling


def test_clean_symmetric_column_recommends_standard(clean_dataframe):
    recos = recommend_scaling_strategy(clean_dataframe)
    age_reco = next(r for r in recos if r["column"] == "age")
    assert age_reco["recommended_method"] == "standard"


def test_column_with_outliers_recommends_robust_regardless_of_skew(dataframe_with_outliers):
    recos = recommend_scaling_strategy(dataframe_with_outliers)
    salaire_reco = next(r for r in recos if r["column"] == "salaire")
    assert salaire_reco["recommended_method"] == "robust"


def test_apply_standard_scaler_centers_data(clean_dataframe):
    df_result, log = apply_scaling(clean_dataframe, {"age": "standard"})
    assert abs(df_result["age"].mean()) < 1e-6
    assert abs(df_result["age"].std(ddof=0) - 1) < 1e-6
    assert log[0]["method"] == "standard"


def test_apply_minmax_scaler_bounds_between_0_and_1():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 100]})
    df_result, _ = apply_scaling(df, {"x": "minmax"})
    assert df_result["x"].min() == 0
    assert df_result["x"].max() == 1


def test_skip_option_leaves_column_unscaled(clean_dataframe):
    df_result, log = apply_scaling(clean_dataframe, {"age": "ne_pas_scaler"})
    pd.testing.assert_series_equal(df_result["age"], clean_dataframe["age"])
    assert log == []