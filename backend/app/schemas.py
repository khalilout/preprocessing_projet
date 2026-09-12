from pydantic import BaseModel
from typing import Optional


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    inferred_type: str
    missing_count: int
    missing_rate: float
    n_unique: int
    skewness: Optional[float] = None


class InitialAnalysisResponse(BaseModel):
    n_rows: int
    n_columns: int
    columns: list[ColumnInfo]
    correlation_matrix: Optional[dict] = None
    is_time_series: bool
    time_series_columns: list[str]
    duplicated_rows: int
    memory_usage_mb: float


class UploadResponse(InitialAnalysisResponse):
    dataset_id: str


class MissingValueRecommendation(BaseModel):
    column: str
    missing_rate: float
    p_value: Optional[float] = None
    missingness_type: str
    recommended_method: str
    rationale: str


class MissingValueStrategyResponse(BaseModel):
    dataset_id: str
    recommendations: list[MissingValueRecommendation]


class ApplyImputationRequest(BaseModel):
    strategies: dict[str, str]


class ImputationLogEntry(BaseModel):
    column: str
    method: str
    values_imputed: int


class ImputationResponse(BaseModel):
    dataset_id: str
    log: list[ImputationLogEntry]
    analysis: InitialAnalysisResponse


class OutlierRecommendation(BaseModel):
    column: str
    skewness: float
    n_outliers_detected: int
    recommended_method: str
    rationale: str


class OutlierStrategyResponse(BaseModel):
    dataset_id: str
    recommendations: list[OutlierRecommendation]


class ApplyOutlierRequest(BaseModel):
    strategies: dict[str, str]


class ScalingRecommendation(BaseModel):
    column: str
    skewness: float
    recommended_method: str
    rationale: str


class ScalingStrategyResponse(BaseModel):
    dataset_id: str
    recommendations: list[ScalingRecommendation]


class ApplyScalingRequest(BaseModel):
    strategies: dict[str, str]


class EncodingRecommendation(BaseModel):
    column: str
    n_unique: int
    recommended_method: str
    rationale: str
    unique_values: Optional[list[str]] = None


class EncodingStrategyResponse(BaseModel):
    dataset_id: str
    recommendations: list[EncodingRecommendation]


class ApplyEncodingRequest(BaseModel):
    strategies: dict[str, str]
    ordinal_orders: Optional[dict[str, list[str]]] = None


class GenericTreatmentResponse(BaseModel):
    dataset_id: str
    log: list[dict]
    analysis: InitialAnalysisResponse