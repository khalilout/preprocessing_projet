"""
Étape 5 — ENCODAGE DES VARIABLES CATÉGORIELLES

  Nominal Encoding  (pas d'ordre)         -> One-Hot Encoding (pd.get_dummies)
  Ordinal Encoding  (ordre naturel)       -> OrdinalEncoder (nécessite l'ordre des catégories)
  Label Encoding    (nominal, arbre/etc.) -> LabelEncoder

  Comme la distinction nominal/ordinal dépend du sens métier (ex: "faible/moyen/fort"
  est ordinal, "Dakar/Thies/Saint-Louis" ne l'est pas), l'automatisation ne peut
  que proposer un défaut raisonnable basé sur la cardinalité :
      - 2 catégories                -> label (binaire, l'ordre n'a pas d'impact)
      - <= 10 catégories            -> one-hot (nominal probable, cardinalité gérable)
      - > 10 catégories             -> label (évite l'explosion dimensionnelle du one-hot)
  L'utilisateur reste libre de forcer "ordinal" en fournissant l'ordre des catégories.
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

ONE_HOT_MAX_CARDINALITY = 10


def recommend_encoding_strategy(df: pd.DataFrame) -> list[dict]:
    recommendations = []
    categorical_cols = df.select_dtypes(exclude=["number", "datetime64[ns]", "bool"]).columns

    for col in categorical_cols:
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= 1:
            continue

        if n_unique == 2:
            method, rationale = "label", "2 catégories -> Label Encoding (binaire, l'ordre n'a pas d'impact)."
        elif n_unique <= ONE_HOT_MAX_CARDINALITY:
            method, rationale = "one_hot", f"{n_unique} catégories (<= {ONE_HOT_MAX_CARDINALITY}) -> One-Hot Encoding (pas d'ordre supposé)."
        else:
            method, rationale = "label", f"{n_unique} catégories (> {ONE_HOT_MAX_CARDINALITY}) -> Label Encoding (évite l'explosion dimensionnelle)."

        recommendations.append({
            "column": col,
            "n_unique": int(n_unique),
            "recommended_method": method,
            "rationale": rationale,
        })

    return recommendations


def apply_encoding(
    df: pd.DataFrame,
    strategies: dict[str, str],
    ordinal_orders: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    strategies: {colonne: "one_hot" | "label" | "ordinal"}
    ordinal_orders: {colonne: [catégories dans l'ordre croissant]} — requis si method="ordinal"
    """
    df = df.copy()
    log = []
    ordinal_orders = ordinal_orders or {}

    one_hot_cols = [c for c, m in strategies.items() if m == "one_hot" and c in df.columns]

    for col, method in strategies.items():
        if col not in df.columns or method == "one_hot":
            continue

        if method == "label":
            le = LabelEncoder()
            df[col] = df[col].astype(object)
            non_null_mask = df[col].notna()
            df.loc[non_null_mask, col] = le.fit_transform(df.loc[non_null_mask, col].astype(str))
            log.append({"column": col, "method": "label", "n_categories": len(le.classes_)})

        elif method == "ordinal":
            categories = ordinal_orders.get(col)
            if not categories:
                continue
            df[col] = df[col].astype(object)
            encoder = OrdinalEncoder(categories=[categories], handle_unknown="use_encoded_value", unknown_value=-1)
            non_null_mask = df[col].notna()
            df.loc[non_null_mask, col] = encoder.fit_transform(df.loc[non_null_mask, [col]]).ravel()
            log.append({"column": col, "method": "ordinal", "order": categories})

    if one_hot_cols:
        before_cols = set(df.columns)
        df = pd.get_dummies(df, columns=one_hot_cols, prefix=one_hot_cols, dtype=int)
        new_cols = list(set(df.columns) - before_cols)
        log.append({"column": ", ".join(one_hot_cols), "method": "one_hot", "new_columns_created": len(new_cols)})

    return df, log