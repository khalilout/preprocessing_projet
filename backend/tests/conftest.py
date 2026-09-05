"""
Fixtures partagées entre les fichiers de test.

Une fixture pytest est une fonction qui prépare des données ou un état
réutilisable par plusieurs tests, sans dupliquer le code de préparation
dans chaque test.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clean_dataframe() -> pd.DataFrame:
    """Un DataFrame simple, sans valeurs manquantes ni outliers, pour les cas de base."""
    np.random.seed(42)
    n = 2000
    return pd.DataFrame({
        "age": np.random.normal(35, 8, n),
        "salaire": np.random.normal(3000, 500, n),
        "ville": np.random.choice(["Dakar", "Thies", "Saint-Louis"], n),
        "genre": np.random.choice(["M", "F"], n),
    })


@pytest.fixture
def dataframe_with_missing_values() -> pd.DataFrame:
    """
    Un DataFrame avec deux types de valeurs manquantes :
    - 'age' : manquants complètement aléatoires (MCAR)
    - 'ville' : manquants dépendants de 'salaire' (MAR) - simule un vrai biais
    """
    np.random.seed(0)
    n = 400
    df = pd.DataFrame({
        "age": np.random.normal(35, 10, n),
        "salaire": np.random.exponential(3000, n),
        "ville": np.random.choice(["Dakar", "Thies", "Saint-Louis"], n),
        "genre": np.random.choice(["M", "F"], n),
    })
    rng = np.random.default_rng(1)
    df.loc[rng.choice(n, 20, replace=False), "age"] = np.nan
    mask_mar = df["salaire"] > df["salaire"].quantile(0.85)
    df.loc[mask_mar, "ville"] = np.nan
    return df


@pytest.fixture
def dataframe_with_outliers() -> pd.DataFrame:
    """Un DataFrame avec des outliers injectés délibérément sur 'salaire'."""
    np.random.seed(0)
    n = 300
    values = np.random.normal(3000, 300, n)
    values[:5] = [50000, -20000, 45000, -18000, 60000]
    return pd.DataFrame({"salaire": values, "age": np.random.normal(35, 8, n)})
