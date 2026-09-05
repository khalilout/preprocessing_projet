"""
Génère un script Python autonome (pandas/scikit-learn) qui reproduit,
étape par étape, exactement les traitements appliqués via l'API/le dashboard.

Utile pour le portfolio : montre que l'app ne fait pas de la "boîte noire",
et donne à l'utilisateur un script réutilisable/versionnable dans un repo Git,
indépendant de l'application elle-même.
"""

HEADER = '''"""
Script de prétraitement généré automatiquement.
Reproduit les étapes appliquées via l'application (analyse, valeurs manquantes,
outliers, scaling, encodage).
"""
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OrdinalEncoder

df = pd.read_csv("votre_fichier.csv")

'''


def _missing_values_code(strategies: dict[str, str]) -> str:
    if not strategies:
        return ""
    lines = []

    knn_cols = [c for c, m in strategies.items() if m == "knn_imputer"]
    iterative_cols = [c for c, m in strategies.items() if m == "iterative_imputer"]

    for col, method in strategies.items():
        if method == "moyenne":
            lines.append(f'df["{col}"] = df["{col}"].fillna(df["{col}"].mean())')
        elif method == "mediane":
            lines.append(f'df["{col}"] = df["{col}"].fillna(df["{col}"].median())')
        elif method == "mode":
            lines.append(f'df["{col}"] = df["{col}"].fillna(df["{col}"].mode().iloc[0])')
        elif method == "interpolation_lineaire":
            lines.append(f'df["{col}"] = df["{col}"].interpolate(method="linear", limit_direction="both")')
        elif method == "bfill":
            lines.append(f'df["{col}"] = df["{col}"].bfill()')
        elif method == "ffill":
            lines.append(f'df["{col}"] = df["{col}"].ffill()')
        elif method == "suppression":
            lines.append(f'df = df.dropna(subset=["{col}"])')
        elif method == "mode_par_groupe":
            lines.append(f'df["{col}"] = df["{col}"].fillna(df["{col}"].mode().iloc[0])')
        elif method in ("knn_imputer", "iterative_imputer"):
            continue
        elif method == "groupe_temporel":
            pass

    if knn_cols:
        lines.append(f"knn_cols = {knn_cols!r}")
        lines.append("_imputer = KNNImputer(n_neighbors=5)")
        lines.append("df[knn_cols] = _imputer.fit_transform(df[knn_cols])")

    if iterative_cols:
        lines.append(f"iterative_cols = {iterative_cols!r}")
        lines.append("_imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=50, random_state=42), random_state=42, max_iter=10)")
        lines.append("df[iterative_cols] = _imputer.fit_transform(df[iterative_cols])")

    lines.append("")
    return "\n".join(lines) + "\n"


def _outliers_code(strategies: dict[str, str]) -> str:
    if not strategies:
        return ""
    lines = []

    for col, method in strategies.items():
        if method == "zscore":
            lines.append(f'_mean, _std = df["{col}"].mean(), df["{col}"].std()')
            lines.append(f'_z = (df["{col}"] - _mean) / _std')
            lines.append(f'df.loc[_z.abs() > 3, "{col}"] = df["{col}"].median()')
        elif method == "iqr":
            lines.append(f'_q1, _q3 = df["{col}"].quantile(0.25), df["{col}"].quantile(0.75)')
            lines.append("_iqr = _q3 - _q1")
            lines.append(f'df["{col}"] = df["{col}"].clip(lower=_q1 - 1.5*_iqr, upper=_q3 + 1.5*_iqr)')
        elif method == "winsorisation":
            lines.append(f'_lower, _upper = df["{col}"].quantile(0.05), df["{col}"].quantile(0.95)')
            lines.append(f'df["{col}"] = df["{col}"].clip(lower=_lower, upper=_upper)')

    lines.append("")
    return "\n".join(lines) + "\n"


def _scaling_code(strategies: dict[str, str]) -> str:
    if not strategies:
        return ""
    lines = []

    scaler_map = {"minmax": "MinMaxScaler", "standard": "StandardScaler", "robust": "RobustScaler"}
    by_method: dict[str, list[str]] = {}
    for col, method in strategies.items():
        by_method.setdefault(method, []).append(col)

    for method, cols in by_method.items():
        if method not in scaler_map:
            continue
        lines.append(f"_cols = {cols!r}")
        lines.append(f"_scaler = {scaler_map[method]}()")
        lines.append("df[_cols] = _scaler.fit_transform(df[_cols])")

    lines.append("")
    return "\n".join(lines) + "\n"


def _encoding_code(strategies: dict[str, str]) -> str:
    if not strategies:
        return ""
    lines = []

    one_hot_cols = [c for c, m in strategies.items() if m == "one_hot"]

    for col, method in strategies.items():
        if method == "label":
            lines.append(f'df["{col}"] = LabelEncoder().fit_transform(df["{col}"].astype(str))')
        elif method == "ordinal":
            lines.append(f'_order = []')
            lines.append(f'df["{col}"] = OrdinalEncoder(categories=[_order]).fit_transform(df[["{col}"]])')

    if one_hot_cols:
        lines.append(f"df = pd.get_dummies(df, columns={one_hot_cols!r}, prefix={one_hot_cols!r}, dtype=int)")

    lines.append("")
    return "\n".join(lines) + "\n"


def generate_pipeline_code(applied_strategies: dict) -> str:
    """
    applied_strategies: {
        "missing_values": {col: method, ...},
        "outliers": {col: method, ...},
        "scaling": {col: method, ...},
        "encoding": {col: method, ...},
    }
    """
    code = HEADER
    code += _missing_values_code(applied_strategies.get("missing_values", {}))
    code += _outliers_code(applied_strategies.get("outliers", {}))
    code += _scaling_code(applied_strategies.get("scaling", {}))
    code += _encoding_code(applied_strategies.get("encoding", {}))
    code += 'df.to_csv("dataset_nettoye.csv", index=False)\n'
    code += 'print("Terminé :", df.shape)\n'
    return code
