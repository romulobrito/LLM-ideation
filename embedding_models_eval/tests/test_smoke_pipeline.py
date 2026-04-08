"""
Smoke tests leves: validam config e gravacao de artefatos sem rodar embeddings/API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from embedding_models_eval.pipeline import load_config
from embedding_models_eval.pipeline.runner import save_artifacts


def _embedding_eval_root() -> Path:
    """Diretorio embedding_models_eval (pai de tests/)."""
    return Path(__file__).resolve().parents[1]


def _pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def test_smoke_load_default_config() -> None:
    """default.yaml existe, carrega e tem secoes minimas."""
    cfg_path = _embedding_eval_root() / "configs" / "default.yaml"
    assert cfg_path.is_file(), f"missing {cfg_path}"
    cfg = load_config(str(cfg_path))
    assert isinstance(cfg["models"], list) and len(cfg["models"]) >= 1
    assert "dataset" in cfg and "path" in cfg["dataset"]
    assert "output" in cfg and "results_dir" in cfg["output"]


def test_smoke_save_artifacts_macro_csv_and_json(tmp_path: Path) -> None:
    """Grava CSV + JSON macro sem detalhado (sem parquet)."""
    macro = pd.DataFrame(
        [
            {"Modelo": "dummy", "MAP@1": 0.25, "MAP@5": 0.1},
        ]
    )
    output_cfg = {
        "results_dir": str(tmp_path),
        "save_summary": True,
        "save_summary_json": True,
        "save_summary_excel": False,
        "save_detailed": False,
    }
    artifacts = save_artifacts({}, macro, output_cfg)
    assert artifacts["summary_csv"] is not None
    assert artifacts["summary_json"] is not None
    assert artifacts["summary_excel"] is None
    assert artifacts["detailed"] == {}
    assert artifacts["detailed_json"] == {}

    csv_p = Path(artifacts["summary_csv"])
    json_p = Path(artifacts["summary_json"])
    assert csv_p.is_file() and json_p.is_file()

    payload = json.loads(json_p.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) == 1
    assert payload[0]["Modelo"] == "dummy"


@pytest.mark.skipif(not _pyarrow_available(), reason="pyarrow needed for to_parquet smoke")
def test_smoke_save_artifacts_detailed_parquet_and_json(tmp_path: Path) -> None:
    """Com pyarrow: parquet + JSON detalhado por modelo."""
    macro = pd.DataFrame([{"Modelo": "m1"}])
    df_scored = pd.DataFrame(
        [
            {"prompt_id": "p1", "doc_id": "d1", "score_to_anchor": 0.9, "x": 1},
        ]
    )
    resultados = {"m1": {"df_scored": df_scored}}
    output_cfg = {
        "results_dir": str(tmp_path),
        "save_summary": True,
        "save_summary_json": True,
        "save_summary_excel": False,
        "save_detailed": True,
        "save_detailed_json": True,
    }
    artifacts = save_artifacts(resultados, macro, output_cfg)
    assert "m1" in artifacts["detailed"]
    assert "m1" in artifacts["detailed_json"]
    assert Path(artifacts["detailed"]["m1"]).is_file()
    detail_json = Path(artifacts["detailed_json"]["m1"])
    assert detail_json.is_file()
    rows = json.loads(detail_json.read_text(encoding="utf-8"))
    assert isinstance(rows, list) and rows[0]["doc_id"] == "d1"
