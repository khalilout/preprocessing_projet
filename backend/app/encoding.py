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

        unique_values = sorted(str(v) for v in df[col].dropna().unique().tolist())

        recommendations.append({
            "column": col,
            "n_unique": int(n_unique),
            "recommended_method": method,
            "rationale": rationale,
            "unique_values": unique_values,
        })

    return recommendations


def apply_encoding(
    df: pd.DataFrame,
    strategies: dict[str, str],
    ordinal_orders: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    
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