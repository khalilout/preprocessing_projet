# 🧹 Application de Prétraitement des Données

Application web de nettoyage et préparation automatisée de données tabulaires (CSV/Excel), guidée par des recommandations statistiques à chaque étape. Projet portfolio démontrant une chaîne de valeur data complète : traitement des données, API, interface interactive, tests, et conteneurisation.

---

## Sommaire

- [Aperçu](#aperçu)
- [Architecture](#architecture)
- [Fonctionnalités](#fonctionnalités)
- [Stack technique](#stack-technique)
- [Installation locale](#installation-locale)
- [Lancer avec Docker](#lancer-avec-docker)
- [Tests](#tests)
- [Structure du projet](#structure-du-projet)
- [Limites connues](#limites-connues)
- [Pistes d'amélioration](#pistes-damélioration)

---

## Aperçu

L'utilisateur charge un fichier CSV ou Excel et progresse à travers un pipeline en 6 étapes. À chaque étape, l'application **analyse statistiquement** les données et **recommande une méthode de traitement adaptée** (imputation, gestion des outliers, scaling, encodage) — l'utilisateur garde la main pour valider ou changer chaque recommandation avant de l'appliquer.

En sortie : un dataset nettoyé exportable en CSV, et un **script Python autonome** reproduisant fidèlement tout le pipeline appliqué (aucune dépendance à l'application — utile pour intégrer le traitement ailleurs, ou simplement vérifier qu'il n'y a pas de boîte noire).

---

## Architecture

```
┌────────────────────┐      HTTP       ┌──────────────────────┐
│   Streamlit UI      │ ──────────────> │     FastAPI backend    │
│  (frontend/app.py)  │ <────────────── │     (backend/app/)      │
└────────────────────┘                 └──────────────────────┘
                                                  │
                                      DATASETS: dict[str, pd.DataFrame]
                                      (stockage en mémoire, clé = dataset_id UUID)
```

Le frontend Streamlit ne contient **aucune logique de traitement de données** : il présente l'UI, appelle l'API, affiche les résultats. Toute la logique métier vit côté FastAPI. Ce découplage montre une séparation propre entre présentation et traitement, et permettrait de brancher un autre frontend (mobile, autre framework) sans toucher au backend.

**Principe clé** : le fichier est uploadé une seule fois (`POST /upload`), l'API renvoie un `dataset_id`. Toutes les étapes suivantes référencent cet identifiant plutôt que de renvoyer le fichier entier à chaque appel — le DataFrame vit en mémoire côté serveur et se transforme progressivement à chaque étape validée.

---

## Fonctionnalités

### 1. Analyse initiale
Détection automatique des types de colonnes, taux de valeurs manquantes, asymétrie (skewness), corrélations, et détection de séries temporelles.

### 2. Traitement des valeurs manquantes
Pour chaque colonne, un test statistique (Student pour le numérique, Chi² pour le catégoriel) détermine si l'absence de valeur est aléatoire (**MCAR**) ou liée à d'autres colonnes (**MAR/MNAR**), et recommande en conséquence :
- Méthodes simples (moyenne, médiane, mode, interpolation) pour le hasard pur
- Méthodes avancées (KNNImputer, IterativeImputer, imputation par groupe temporel) quand une dépendance statistique est détectée

Visualisation du motif des valeurs manquantes via une matrice `missingno` (échantillonnée pour rester performante sur de gros volumes).

### 3. Traitement des valeurs aberrantes (outliers)
Méthode choisie selon l'asymétrie de la distribution : Z-score (symétrique), IQR (asymétrie modérée), ou Winsorisation (forte asymétrie). Comparaison visuelle avant/après (boxplot + histogramme).

### 4. Feature scaling
RobustScaler en priorité si des outliers sont détectés, sinon StandardScaler (distribution symétrique) ou MinMaxScaler (asymétrique).

### 5. Encodage des variables catégorielles
Label Encoding (binaire ou forte cardinalité), One-Hot Encoding (cardinalité modérée), ou **Encodage Ordinal** avec ordre personnalisable directement depuis l'interface.

### 6. Visualisation & export
Résumé avant/après, journal complet des traitements appliqués, statistiques descriptives, export du dataset nettoyé en CSV, et génération d'un **script Python reproductible** téléchargeable.

À chaque étape, l'option **"ne pas traiter"** reste disponible pour laisser une colonne inchangée.

---

## Stack technique

| Composant | Technologies |
|---|---|
| Backend | FastAPI, Pandas, NumPy, Scikit-learn, SciPy |
| Frontend | Streamlit, Plotly |
| Visualisation | Matplotlib, missingno |
| Tests | Pytest, httpx (TestClient) |
| Conteneurisation | Docker, docker-compose |
| CI/CD | GitHub Actions |

---

## Installation locale

Prérequis : Python 3.11+, `pip`.

```powershell
# Créer et activer l'environnement virtuel
python -m venv venv
venv\Scripts\activate

# Installer les dépendances
python -m pip install -r requirements-backend.txt -r requirements-frontend.txt -r requirements-dev.txt

# Lancer le backend (terminal 1)
uvicorn backend.app.main:app --reload

# Lancer le frontend (terminal 2)
streamlit run frontend/app.py
```

L'application est accessible sur http://localhost:8501 (le frontend appelle le backend sur http://localhost:8000 par défaut).

---

## Lancer avec Docker

```powershell
docker compose up --build
```

Puis ouvrir http://localhost:8501. Les deux services (`backend`, `frontend`) communiquent via un réseau Docker interne ; le frontend résout l'API par nom de service (`API_URL=http://backend:8000`), sans configuration manuelle.

---

## Tests

```powershell
pytest backend/tests/ -v
```

58 tests couvrant : l'analyse initiale, les valeurs manquantes (logique MCAR/MAR), les outliers, le scaling, l'encodage (y compris l'encodage ordinal avec ordre personnalisé), la génération du script Python reproductible, les visualisations, et un test d'intégration end-to-end du pipeline complet via `TestClient`.

---

## Structure du projet

```
data-preprocessing-app/
├── backend/
│   ├── app/
│   │   ├── main.py              # routes FastAPI
│   │   ├── schemas.py           # modèles Pydantic
│   │   ├── analysis.py          # étape 1
│   │   ├── missing_values.py    # étape 2
│   │   ├── outliers.py          # étape 3
│   │   ├── scaling.py           # étape 4
│   │   ├── encoding.py          # étape 5
│   │   ├── stats_utils.py       # fonctions statistiques partagées
│   │   ├── viz.py               # génération des visualisations (PNG)
│   │   └── code_generator.py    # étape 6 — script Python reproductible
│   └── tests/
├── frontend/
│   └── app.py                   # dashboard Streamlit
├── docker-compose.yml
├── requirements-backend.txt
├── requirements-frontend.txt
├── requirements-dev.txt
└── .github/workflows/ci.yml     # CI : lance les tests à chaque push/PR
```

---

## Limites connues

- **Stockage en mémoire** : aucune base de données — un redémarrage du backend efface les données uploadées. Choix assumé pour un projet de démonstration ; une vraie mise en production nécessiterait Redis ou une base de données pour la persistance multi-utilisateurs.
- **CORS ouvert** (`allow_origins=["*"]`) — à restreindre en environnement de production réel.
- **Encodage ordinal** : l'ordre doit être défini manuellement via l'interface à chaque session (pas de sauvegarde de configuration entre deux uploads).

---

## Pistes d'amélioration

- Persistance des datasets (Redis / base de données) pour supporter plusieurs utilisateurs simultanés
- Authentification et gestion de sessions utilisateur
- Sauvegarde/chargement de configurations de prétraitement réutilisables
- Déploiement (backend + frontend) — voir la section déploiement du dépôt une fois en ligne