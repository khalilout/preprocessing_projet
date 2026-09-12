import io
import uuid

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .analysis import analyze_dataframe
from .missing_values import recommend_strategy, apply_missing_value_treatment
from .outliers import recommend_outlier_strategy, apply_outlier_treatment
from .scaling import recommend_scaling_strategy, apply_scaling
from .encoding import recommend_encoding_strategy, apply_encoding
from .viz import missing_matrix_png, boxplot_png, distribution_comparison_png, distribution_single_png
from .code_generator import generate_pipeline_code
from .schemas import (
    InitialAnalysisResponse,
    UploadResponse,
    MissingValueStrategyResponse,
    ApplyImputationRequest,
    ImputationResponse,
    OutlierStrategyResponse,
    ApplyOutlierRequest,
    ScalingStrategyResponse,
    ApplyScalingRequest,
    EncodingStrategyResponse,
    ApplyEncodingRequest,
    GenericTreatmentResponse,
)

app = FastAPI(
    title="Data Preprocessing API",
    description="API de nettoyage, imputation, gestion des outliers, scaling et encodage",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASETS: dict[str, pd.DataFrame] = {}
INITIAL_ANALYSIS: dict[str, dict] = {}
TREATMENT_LOG: dict[str, list[dict]] = {}
APPLIED_STRATEGIES: dict[str, dict] = {}


def _read_upload_to_dataframe(file: UploadFile) -> pd.DataFrame:
    filename = file.filename or ""
    content = file.file.read()

    if filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(content))
    else:
        raise HTTPException(
            status_code=400,
            detail="Format non supporté. Utilisez un fichier .csv, .xlsx ou .xls",
        )

    df.columns = [str(c) for c in df.columns]

    return df


@app.get("/")
def root():
    return {"status": "ok", "message": "Data Preprocessing API is running"}


@app.post("/upload", response_model=UploadResponse)
def upload_dataset(file: UploadFile = File(...)):
    df = _read_upload_to_dataframe(file)

    if df.empty:
        raise HTTPException(status_code=400, detail="Le fichier est vide.")

    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = df

    analysis = analyze_dataframe(df)
    INITIAL_ANALYSIS[dataset_id] = analysis.model_dump()
    TREATMENT_LOG[dataset_id] = []
    APPLIED_STRATEGIES[dataset_id] = {"missing_values": {}, "outliers": {}, "scaling": {}, "encoding": {}}

    result = analysis.model_dump()
    result["dataset_id"] = dataset_id
    return result


@app.get("/datasets/{dataset_id}/preview")
def preview_dataset(dataset_id: str, n_rows: int = 10):
    df = _get_dataset_or_404(dataset_id)
    preview = df.head(n_rows).replace([np.inf, -np.inf], np.nan)
    preview = preview.astype(object).where(pd.notnull(preview), None)
    return preview.to_dict(orient="records")


@app.get("/datasets/{dataset_id}/analysis", response_model=InitialAnalysisResponse)
def get_analysis(dataset_id: str):
    df = _get_dataset_or_404(dataset_id)
    return analyze_dataframe(df)


@app.get("/datasets/{dataset_id}/missing-strategy", response_model=MissingValueStrategyResponse)
def get_missing_value_strategy(dataset_id: str):
    df = _get_dataset_or_404(dataset_id)
    analysis = analyze_dataframe(df)
    recommendations = recommend_strategy(
        df,
        datetime_cols=analysis.time_series_columns,
        is_time_series=analysis.is_time_series,
    )
    return {"dataset_id": dataset_id, "recommendations": recommendations}


@app.post("/datasets/{dataset_id}/impute-missing", response_model=ImputationResponse)
def impute_missing_values(dataset_id: str, request: ApplyImputationRequest):
    df = _get_dataset_or_404(dataset_id)
    analysis = analyze_dataframe(df)

    df_clean, log = apply_missing_value_treatment(
        df,
        datetime_cols=analysis.time_series_columns,
        strategies=request.strategies,
    )

    DATASETS[dataset_id] = df_clean
    new_analysis = analyze_dataframe(df_clean)

    TREATMENT_LOG.setdefault(dataset_id, []).append({"step": "valeurs_manquantes", "entries": log})
    APPLIED_STRATEGIES.setdefault(dataset_id, {}).setdefault("missing_values", {}).update(request.strategies)

    return {"dataset_id": dataset_id, "log": log, "analysis": new_analysis}


def _get_dataset_or_404(dataset_id: str) -> pd.DataFrame:
    df = DATASETS.get(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="dataset_id introuvable. Ré-uploadez le fichier.")
    return df


@app.get("/datasets/{dataset_id}/outlier-strategy", response_model=OutlierStrategyResponse)
def get_outlier_strategy(dataset_id: str):
    df = _get_dataset_or_404(dataset_id)
    recommendations = recommend_outlier_strategy(df)
    return {"dataset_id": dataset_id, "recommendations": recommendations}


@app.post("/datasets/{dataset_id}/treat-outliers", response_model=GenericTreatmentResponse)
def treat_outliers(dataset_id: str, request: ApplyOutlierRequest):
    df = _get_dataset_or_404(dataset_id)
    df_clean, log = apply_outlier_treatment(df, request.strategies)
    DATASETS[dataset_id] = df_clean
    TREATMENT_LOG.setdefault(dataset_id, []).append({"step": "outliers", "entries": log})
    APPLIED_STRATEGIES.setdefault(dataset_id, {}).setdefault("outliers", {}).update(request.strategies)
    return {"dataset_id": dataset_id, "log": log, "analysis": analyze_dataframe(df_clean)}


@app.get("/datasets/{dataset_id}/scaling-strategy", response_model=ScalingStrategyResponse)
def get_scaling_strategy(dataset_id: str):
    df = _get_dataset_or_404(dataset_id)
    recommendations = recommend_scaling_strategy(df)
    return {"dataset_id": dataset_id, "recommendations": recommendations}


@app.post("/datasets/{dataset_id}/apply-scaling", response_model=GenericTreatmentResponse)
def apply_scaling_route(dataset_id: str, request: ApplyScalingRequest):
    df = _get_dataset_or_404(dataset_id)
    df_scaled, log = apply_scaling(df, request.strategies)
    DATASETS[dataset_id] = df_scaled
    TREATMENT_LOG.setdefault(dataset_id, []).append({"step": "scaling", "entries": log})
    APPLIED_STRATEGIES.setdefault(dataset_id, {}).setdefault("scaling", {}).update(request.strategies)
    return {"dataset_id": dataset_id, "log": log, "analysis": analyze_dataframe(df_scaled)}


@app.get("/datasets/{dataset_id}/encoding-strategy", response_model=EncodingStrategyResponse)
def get_encoding_strategy(dataset_id: str):
    df = _get_dataset_or_404(dataset_id)
    recommendations = recommend_encoding_strategy(df)
    return {"dataset_id": dataset_id, "recommendations": recommendations}


@app.post("/datasets/{dataset_id}/apply-encoding", response_model=GenericTreatmentResponse)
def apply_encoding_route(dataset_id: str, request: ApplyEncodingRequest):
    df = _get_dataset_or_404(dataset_id)
    df_encoded, log = apply_encoding(df, request.strategies, request.ordinal_orders)
    DATASETS[dataset_id] = df_encoded
    TREATMENT_LOG.setdefault(dataset_id, []).append({"step": "encodage", "entries": log})
    APPLIED_STRATEGIES.setdefault(dataset_id, {}).setdefault("encoding", {}).update(request.strategies)
    return {"dataset_id": dataset_id, "log": log, "analysis": analyze_dataframe(df_encoded)}


@app.get("/datasets/{dataset_id}/export-csv")
def export_csv(dataset_id: str):
    df = _get_dataset_or_404(dataset_id)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=dataset_nettoye_{dataset_id[:8]}.csv"},
    )


@app.get("/datasets/{dataset_id}/summary")
def get_summary(dataset_id: str):
    df = _get_dataset_or_404(dataset_id)
    current_analysis = analyze_dataframe(df).model_dump()
    initial_analysis = INITIAL_ANALYSIS.get(dataset_id, current_analysis)

    def _global_missing_rate(analysis: dict) -> float:
        total_cells = analysis["n_rows"] * analysis["n_columns"] if analysis["n_columns"] else 0
        total_missing = sum(c["missing_count"] for c in analysis["columns"])
        return round(total_missing / total_cells, 4) if total_cells else 0.0

    numeric_df = df.select_dtypes(include=[np.number])
    describe_current = numeric_df.describe().round(3).to_dict() if not numeric_df.empty else {}

    return {
        "dataset_id": dataset_id,
        "initial": {
            "n_rows": initial_analysis["n_rows"],
            "n_columns": initial_analysis["n_columns"],
            "missing_rate_global": _global_missing_rate(initial_analysis),
        },
        "current": {
            "n_rows": current_analysis["n_rows"],
            "n_columns": current_analysis["n_columns"],
            "missing_rate_global": _global_missing_rate(current_analysis),
        },
        "treatment_log": TREATMENT_LOG.get(dataset_id, []),
        "descriptive_stats": describe_current,
    }


@app.get("/datasets/{dataset_id}/generated-code")
def get_generated_code(dataset_id: str):
    _get_dataset_or_404(dataset_id)
    strategies = APPLIED_STRATEGIES.get(dataset_id, {})
    code = generate_pipeline_code(strategies)
    return StreamingResponse(
        io.BytesIO(code.encode("utf-8")),
        media_type="text/x-python",
        headers={"Content-Disposition": "attachment; filename=pipeline_pretraitement.py"},
    )


@app.get("/datasets/{dataset_id}/missing-matrix")
def get_missing_matrix(dataset_id: str):
    df = _get_dataset_or_404(dataset_id)
    png_bytes = missing_matrix_png(df)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/datasets/{dataset_id}/outlier-boxplot")
def get_outlier_boxplot(dataset_id: str, column: str, label: str = "Distribution actuelle"):
    df = _get_dataset_or_404(dataset_id)
    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Colonne '{column}' introuvable.")
    png_bytes = boxplot_png(df[column], f"{column} — {label}")
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/datasets/{dataset_id}/outlier-distribution")
def get_outlier_distribution(dataset_id: str, column: str, label: str = "", color: str = "#a3c9f7"):
    df = _get_dataset_or_404(dataset_id)
    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Colonne '{column}' introuvable.")
    png_bytes = distribution_single_png(df[column], f"{column} {label}".strip(), color=color)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
