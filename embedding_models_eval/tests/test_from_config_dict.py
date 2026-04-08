"""
Tests for run_experiment_from_config_dict: validation path and delegation to runner.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest

from embedding_models_eval.pipeline import run_experiment_from_config_dict


def _minimal_valid_config(results_dir: str) -> Dict[str, Any]:
    """Smallest config that passes validate_config (paths need not exist for validation)."""
    return {
        "dataset": {
            "path": "dummy_dataset.json",
            "text_col": "extracted_idea_250",
            "truncate_content": 0,
        },
        "models": [
            {
                "name": "stub_model",
                "provider": "sentence_transformers",
                "config": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "batch_size": 8,
                    "normalize": True,
                    "device": "cpu",
                },
            }
        ],
        "metrics": {
            "k_values": [1, 5, 10],
            "metric_names": ["ir", "votes"],
        },
        "ranking": {
            "group_cols": ["contest_number", "context_prompt_url"],
            "rank_col": "rank_in_prompt",
            "anchor_rank": 1,
        },
        "output": {
            "results_dir": results_dir,
            "save_summary": True,
            "save_summary_json": True,
            "save_summary_excel": False,
            "save_detailed": False,
        },
    }


def test_from_config_dict_rejects_missing_dataset_path() -> None:
    cfg = _minimal_valid_config("/tmp/out")
    del cfg["dataset"]["path"]
    with pytest.raises(ValueError, match="dataset.path"):
        run_experiment_from_config_dict(cfg, load_env=False, verbose=False)


@patch("embedding_models_eval.pipeline.from_config_dict.run_experiment")
def test_from_config_dict_validates_then_calls_run_experiment(
    mock_run: Any,
    tmp_path: Any,
) -> None:
    mock_run.return_value = {"ok": True}
    cfg = _minimal_valid_config(str(tmp_path / "results"))

    out = run_experiment_from_config_dict(
        cfg,
        load_env=False,
        verbose=False,
        continue_on_error=True,
    )

    assert out == {"ok": True}
    mock_run.assert_called_once()
    args, kw = mock_run.call_args
    assert kw == {"verbose": False, "continue_on_error": True}
    passed = args[0]
    assert passed["dataset"]["path"] == "dummy_dataset.json"
    assert passed["models"][0]["name"] == "stub_model"


@patch("embedding_models_eval.pipeline.from_config_dict.load_env_robust")
@patch("embedding_models_eval.pipeline.from_config_dict.run_experiment")
def test_from_config_dict_load_env_false_skips_dotenv(
    mock_run: Any,
    mock_load_env: Any,
    tmp_path: Any,
) -> None:
    mock_run.return_value = {}
    cfg = _minimal_valid_config(str(tmp_path / "r"))
    run_experiment_from_config_dict(cfg, load_env=False, verbose=False)
    mock_load_env.assert_not_called()


@patch("embedding_models_eval.pipeline.from_config_dict.load_env_robust")
@patch("embedding_models_eval.pipeline.from_config_dict.run_experiment")
def test_from_config_dict_load_env_true_calls_dotenv(
    mock_run: Any,
    mock_load_env: Any,
    tmp_path: Any,
) -> None:
    mock_run.return_value = {}
    cfg = _minimal_valid_config(str(tmp_path / "r"))
    run_experiment_from_config_dict(cfg, load_env=True, verbose=False)
    mock_load_env.assert_called_once()
