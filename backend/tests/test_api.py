"""
Tests d'intégration : appellent l'API FastAPI comme le ferait le frontend
Streamlit, via TestClient (pas besoin de lancer un vrai serveur).
"""
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def _make_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def test_upload_returns_dataset_id_and_analysis(clean_dataframe):
    response = client.post("/upload", files={"file": ("test.csv", _make_csv_bytes(clean_dataframe), "text/csv")})
    assert response.status_code == 200
    body = response.json()
    assert "dataset_id" in body
    assert body["n_rows"] == clean_dataframe.shape[0]


def test_upload_rejects_unsupported_format():
    response = client.post("/upload", files={"file": ("test.txt", b"contenu quelconque", "text/plain")})
    assert response.status_code == 400


def test_missing_dataset_id_returns_404():
    response = client.get("/datasets/id-inexistant/analysis")
    assert response.status_code == 404


def test_full_pipeline_end_to_end(dataframe_with_missing_values):
    upload_response = client.post(
        "/upload", files={"file": ("test.csv", _make_csv_bytes(dataframe_with_missing_values), "text/csv")}
    )
    assert upload_response.status_code == 200
    dataset_id = upload_response.json()["dataset_id"]

    strategy_response = client.get(f"/datasets/{dataset_id}/missing-strategy")
    assert strategy_response.status_code == 200
    recos = strategy_response.json()["recommendations"]
    strategies = {r["column"]: r["recommended_method"] for r in recos}
    impute_response = client.post(f"/datasets/{dataset_id}/impute-missing", json={"strategies": strategies})
    assert impute_response.status_code == 200
    assert all(c["missing_count"] == 0 for c in impute_response.json()["analysis"]["columns"])

    outlier_recos = client.get(f"/datasets/{dataset_id}/outlier-strategy").json()["recommendations"]
    outlier_strategies = {r["column"]: r["recommended_method"] for r in outlier_recos}
    outlier_response = client.post(f"/datasets/{dataset_id}/treat-outliers", json={"strategies": outlier_strategies})
    assert outlier_response.status_code == 200

    scaling_recos = client.get(f"/datasets/{dataset_id}/scaling-strategy").json()["recommendations"]
    scaling_strategies = {r["column"]: r["recommended_method"] for r in scaling_recos}
    scaling_response = client.post(f"/datasets/{dataset_id}/apply-scaling", json={"strategies": scaling_strategies})
    assert scaling_response.status_code == 200

    encoding_recos = client.get(f"/datasets/{dataset_id}/encoding-strategy").json()["recommendations"]
    encoding_strategies = {r["column"]: r["recommended_method"] for r in encoding_recos}
    encoding_response = client.post(f"/datasets/{dataset_id}/apply-encoding", json={"strategies": encoding_strategies})
    assert encoding_response.status_code == 200

    summary_response = client.get(f"/datasets/{dataset_id}/summary")
    assert summary_response.status_code == 200
    summary = summary_response.json()
    assert summary["current"]["missing_rate_global"] == 0.0
    assert len(summary["treatment_log"]) == 4

    code_response = client.get(f"/datasets/{dataset_id}/generated-code")
    assert code_response.status_code == 200
    assert b"import pandas as pd" in code_response.content

    export_response = client.get(f"/datasets/{dataset_id}/export-csv")
    assert export_response.status_code == 200
    assert export_response.headers["content-type"].startswith("text/csv")


def test_preview_handles_nan_as_null(dataframe_with_missing_values):
    upload_response = client.post(
        "/upload", files={"file": ("test.csv", _make_csv_bytes(dataframe_with_missing_values), "text/csv")}
    )
    dataset_id = upload_response.json()["dataset_id"]
    preview_response = client.get(f"/datasets/{dataset_id}/preview", params={"n_rows": 400})
    assert preview_response.status_code == 200


def test_excel_column_names_are_coerced_to_string():
    import io
    import datetime

    df = pd.DataFrame({"nom": ["a", "b"], datetime.datetime(2024, 1, 1): [1, 2]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)

    response = client.post(
        "/upload",
        files={"file": ("test.xlsx", buf.read(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert response.status_code == 200
    column_names = [c["name"] for c in response.json()["columns"]]
    assert all(isinstance(name, str) for name in column_names)
