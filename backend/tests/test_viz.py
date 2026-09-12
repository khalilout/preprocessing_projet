import numpy as np
import pandas as pd
import pytest

from backend.app.viz import (
    missing_matrix_png,
    boxplot_png,
    distribution_single_png,
    distribution_comparison_png,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def numeric_series():
    np.random.seed(42)
    return pd.Series(np.random.normal(loc=50, scale=10, size=200))


@pytest.fixture
def df_with_missing():
    return pd.DataFrame({
        "a": [1, 2, None, 4, 5],
        "b": [None, 2, 3, None, 5],
        "c": [1, 2, 3, 4, 5],
    })


def test_missing_matrix_png_returns_valid_png(df_with_missing):
    result = missing_matrix_png(df_with_missing)
    assert isinstance(result, bytes)
    assert result.startswith(PNG_SIGNATURE)
    assert len(result) > 0


def test_missing_matrix_png_samples_large_dataframe_without_crash():
    big_df = pd.DataFrame({
        "a": np.random.choice([1, 2, None], size=50_000),
        "b": np.random.choice([1, 2, None], size=50_000),
    })
    result = missing_matrix_png(big_df, max_rows=500)
    assert result.startswith(PNG_SIGNATURE)


def test_missing_matrix_png_handles_dataframe_without_missing_values():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = missing_matrix_png(df)
    assert result.startswith(PNG_SIGNATURE)


def test_boxplot_png_returns_valid_png(numeric_series):
    result = boxplot_png(numeric_series, title="Test boxplot")
    assert isinstance(result, bytes)
    assert result.startswith(PNG_SIGNATURE)


def test_boxplot_png_handles_series_with_nan():
    series_with_nan = pd.Series([1, 2, np.nan, 4, 5, np.nan])
    result = boxplot_png(series_with_nan, title="Avec NaN")
    assert result.startswith(PNG_SIGNATURE)


def test_distribution_single_png_returns_valid_png(numeric_series):
    result = distribution_single_png(numeric_series, title="AVANT", color="#f7a3a3")
    assert isinstance(result, bytes)
    assert result.startswith(PNG_SIGNATURE)


def test_distribution_single_png_handles_series_with_nan():
    series_with_nan = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    result = distribution_single_png(series_with_nan, title="Test", color="#a3f7b0")
    assert result.startswith(PNG_SIGNATURE)


def test_distribution_comparison_png_returns_valid_png():
    np.random.seed(0)
    before = pd.Series(np.random.normal(0, 5, 100))
    after = pd.Series(np.random.normal(0, 1, 100))
    result = distribution_comparison_png(before, after, column="salaire")
    assert isinstance(result, bytes)
    assert result.startswith(PNG_SIGNATURE)


def test_distribution_comparison_png_handles_empty_series_gracefully():
    empty = pd.Series([], dtype=float)
    non_empty = pd.Series([1.0, 2.0, 3.0])
    result = distribution_comparison_png(empty, non_empty, column="test")
    assert result.startswith(PNG_SIGNATURE)