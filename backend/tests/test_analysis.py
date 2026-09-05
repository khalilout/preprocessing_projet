import numpy as np
import pandas as pd

from backend.app.analysis import (
    infer_column_type,
    try_parse_datetime,
    detect_time_series,
    analyze_dataframe,
)


def test_infer_column_type_numeric():
    series = pd.Series([1, 2, 3, 4])
    assert infer_column_type(series) == "numérique"


def test_infer_column_type_categorical():
    series = pd.Series(["a", "b", "c"])
    assert infer_column_type(series) == "catégorielle"


def test_try_parse_datetime_detects_text_dates():
    series = pd.Series(["2023-01-15", "2023-02-20", "2023-03-10"])
    assert try_parse_datetime(series) is True


def test_try_parse_datetime_rejects_plain_text():
    series = pd.Series(["Dakar", "Thies", "Saint-Louis"])
    assert try_parse_datetime(series) is False


def test_detect_time_series_true_for_unique_dates():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({"date": dates, "valeur": range(100)})
    assert detect_time_series(df, datetime_cols=["date"]) is True


def test_detect_time_series_false_without_datetime_column():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert detect_time_series(df, datetime_cols=[]) is False


def test_analyze_dataframe_reports_correct_shape(clean_dataframe):
    result = analyze_dataframe(clean_dataframe)
    assert result.n_rows == clean_dataframe.shape[0]
    assert result.n_columns == clean_dataframe.shape[1]


def test_analyze_dataframe_detects_missing_values(dataframe_with_missing_values):
    result = analyze_dataframe(dataframe_with_missing_values)
    age_col = next(c for c in result.columns if c.name == "age")
    assert age_col.missing_count == 20
    assert age_col.missing_rate == round(20 / 400, 4)


def test_analyze_dataframe_correlation_matrix_has_no_nan(clean_dataframe):
    df = clean_dataframe.copy()
    df["constante"] = 5
    result = analyze_dataframe(df)
    assert result.correlation_matrix is not None
    for row in result.correlation_matrix.values():
        for value in row.values():
            assert value is None or isinstance(value, (int, float))
