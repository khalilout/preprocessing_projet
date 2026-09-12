import pandas as pd
import pytest

from backend.app.encoding import recommend_encoding_strategy, apply_encoding


@pytest.fixture
def dataframe_with_ordinal_column():
    return pd.DataFrame({
        "niveau_etude": ["bac", "master", "licence", "bac", "doctorat", "licence", None],
        "genre": ["M", "F", "F", "M", "F", "M", "F"],
        "salaire": [2500, 3200, 2100, 2600, 5000, 2000, 2400],
    })


def test_recommend_encoding_includes_unique_values(dataframe_with_ordinal_column):
    recos = recommend_encoding_strategy(dataframe_with_ordinal_column)
    reco_niveau = next(r for r in recos if r["column"] == "niveau_etude")

    assert "unique_values" in reco_niveau
    assert set(reco_niveau["unique_values"]) == {"bac", "master", "licence", "doctorat"}
    assert reco_niveau["unique_values"] == sorted(reco_niveau["unique_values"])


def test_apply_encoding_ordinal_respects_custom_order(dataframe_with_ordinal_column):
    strategies = {"niveau_etude": "ordinal"}
    ordinal_orders = {"niveau_etude": ["bac", "licence", "master", "doctorat"]}

    df_out, log = apply_encoding(dataframe_with_ordinal_column, strategies, ordinal_orders)

    expected = {"bac": 0.0, "licence": 1.0, "master": 2.0, "doctorat": 3.0}
    original = dataframe_with_ordinal_column["niveau_etude"]
    for original_value, encoded_value in zip(original, df_out["niveau_etude"]):
        if pd.isna(original_value):
            continue
        assert encoded_value == expected[original_value]

    assert log[0]["method"] == "ordinal"
    assert log[0]["order"] == ["bac", "licence", "master", "doctorat"]


def test_apply_encoding_ordinal_preserves_nan(dataframe_with_ordinal_column):
    strategies = {"niveau_etude": "ordinal"}
    ordinal_orders = {"niveau_etude": ["bac", "licence", "master", "doctorat"]}

    df_out, _ = apply_encoding(dataframe_with_ordinal_column, strategies, ordinal_orders)

    assert pd.isna(df_out["niveau_etude"].iloc[6])


def test_apply_encoding_ordinal_without_order_is_noop(dataframe_with_ordinal_column):
    strategies = {"niveau_etude": "ordinal"}

    df_out, log = apply_encoding(dataframe_with_ordinal_column, strategies, ordinal_orders=None)

    pd.testing.assert_series_equal(df_out["niveau_etude"], dataframe_with_ordinal_column["niveau_etude"])
    assert log == []


def test_apply_encoding_ordinal_unknown_category_gets_minus_one(dataframe_with_ordinal_column):
    strategies = {"niveau_etude": "ordinal"}
    ordinal_orders = {"niveau_etude": ["bac", "licence", "master"]}  # 'doctorat' absent

    df_out, _ = apply_encoding(dataframe_with_ordinal_column, strategies, ordinal_orders)

    doctorat_rows = dataframe_with_ordinal_column["niveau_etude"] == "doctorat"
    assert (df_out.loc[doctorat_rows, "niveau_etude"] == -1.0).all()


def test_apply_encoding_ordinal_combined_with_other_methods(dataframe_with_ordinal_column):
    strategies = {"niveau_etude": "ordinal", "genre": "label"}
    ordinal_orders = {"niveau_etude": ["bac", "licence", "master", "doctorat"]}

    df_out, log = apply_encoding(dataframe_with_ordinal_column, strategies, ordinal_orders)

    methods_applied = {entry["method"] for entry in log}
    assert methods_applied == {"ordinal", "label"}
    assert df_out["salaire"].tolist() == dataframe_with_ordinal_column["salaire"].tolist()