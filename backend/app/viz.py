import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd


def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def missing_matrix_png(df: pd.DataFrame, max_rows: int = 5000) -> bytes:
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42).sort_index()

    fig = plt.figure(figsize=(10, 5))
    ax = msno.matrix(df, fontsize=10, sparkline=True)
    fig = ax.get_figure()
    return _fig_to_png_bytes(fig)


def boxplot_png(series: pd.Series, title: str) -> bytes:
    """Boxplot simple d'une colonne numérique, pour repérer visuellement les outliers."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(series.dropna(), vert=False, patch_artist=True,
               boxprops=dict(facecolor="#a3c9f7"))
    ax.set_title(title)
    ax.set_xlabel(series.name)
    return _fig_to_png_bytes(fig)


def distribution_comparison_png(before: pd.Series, after: pd.Series, column: str) -> bytes:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    axes[0, 0].boxplot(before.dropna(), vert=False, patch_artist=True,
                        boxprops=dict(facecolor="#f7a3a3"))
    axes[0, 0].set_title(f"{column} — Boxplot AVANT")

    axes[0, 1].boxplot(after.dropna(), vert=False, patch_artist=True,
                        boxprops=dict(facecolor="#a3f7b0"))
    axes[0, 1].set_title(f"{column} — Boxplot APRÈS")

    axes[1, 0].hist(before.dropna(), bins=30, color="#f7a3a3", edgecolor="white")
    axes[1, 0].set_title(f"Distribution AVANT (skew={before.dropna().skew():.2f})")

    axes[1, 1].hist(after.dropna(), bins=30, color="#a3f7b0", edgecolor="white")
    axes[1, 1].set_title(f"Distribution APRÈS (skew={after.dropna().skew():.2f})")

    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def distribution_single_png(series: pd.Series, title: str, color: str = "#a3c9f7") -> bytes:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].boxplot(series.dropna(), vert=False, patch_artist=True,
                     boxprops=dict(facecolor=color))
    axes[0].set_title("Boxplot")

    axes[1].hist(series.dropna(), bins=30, color=color, edgecolor="white")
    axes[1].set_title(f"Histogramme (skew={series.dropna().skew():.2f})")

    fig.suptitle(title)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)