import py_compile
import tempfile
import os

from backend.app.code_generator import generate_pipeline_code


def test_generated_code_is_valid_python_syntax():
    strategies = {
        "missing_values": {"age": "moyenne", "salaire": "knn_imputer"},
        "outliers": {"salaire": "winsorisation"},
        "scaling": {"age": "standard"},
        "encoding": {"ville": "one_hot"},
    }
    code = generate_pipeline_code(strategies)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        temp_path = f.name

    try:
        py_compile.compile(temp_path, doraise=True)
    finally:
        os.unlink(temp_path)


def test_generated_code_handles_empty_strategies():
    code = generate_pipeline_code({})
    assert "import pandas as pd" in code
    assert "to_csv" in code


def test_generated_code_includes_chosen_methods():
    strategies = {"missing_values": {"age": "mediane"}}
    code = generate_pipeline_code(strategies)
    assert "mediane" not in code
    assert '"age"' in code and ".median()" in code
