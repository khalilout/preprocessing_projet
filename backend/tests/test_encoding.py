import pandas as pd

from backend.app.encoding import recommend_encoding_strategy, apply_encoding


def test_binary_column_recommends_label(clean_dataframe):
    recos = recommend_encoding_strategy(clean_dataframe)
    genre_reco = next(r for r in recos if r["column"] == "genre")
    assert genre_reco["recommended_method"] == "label"


def test_low_cardinality_column_recommends_one_hot(clean_dataframe):
    recos = recommend_encoding_strategy(clean_dataframe)
    ville_reco = next(r for r in recos if r["column"] == "ville")
    assert ville_reco["recommended_method"] == "one_hot"


def test_high_cardinality_column_recommends_label():
    df = pd.DataFrame({"code_client": [f"C{i}" for i in range(50)]})
    recos = recommend_encoding_strategy(df)
    assert recos[0]["recommended_method"] == "label"


def test_apply_one_hot_creates_expected_columns(clean_dataframe):
    df_result, log = apply_encoding(clean_dataframe, {"ville": "one_hot"})
    expected_cols = {"ville_Dakar", "ville_Thies", "ville_Saint-Louis"}
    assert expected_cols.issubset(set(df_result.columns))
    assert "ville" not in df_result.columns


def test_apply_label_encoding_produces_integers(clean_dataframe):
    df_result, log = apply_encoding(clean_dataframe, {"genre": "label"})
    assert set(df_result["genre"].unique()).issubset({0, 1, "0", "1"})


def test_apply_ordinal_respects_given_order():
    df = pd.DataFrame({"niveau": ["faible", "fort", "moyen", "faible", "fort"]})
    df_result, log = apply_encoding(
        df, {"niveau": "ordinal"}, ordinal_orders={"niveau": ["faible", "moyen", "fort"]}
    )
    mapping = dict(zip(df["niveau"], df_result["niveau"]))
    assert mapping["faible"] < mapping["moyen"] < mapping["fort"]