"""
Testes das etapas opcionais (pipeline_extras): validacao e export CSV per-prompt.
"""

from __future__ import annotations

import pandas as pd

from embedding_models_eval.pipeline.optional_steps import (
    extras_has_enabled_work,
    save_per_prompt_metrics_csvs,
    validate_extras_for_visualizations,
)


def test_validate_visualizations_requires_per_prompt_export() -> None:
    msg = validate_extras_for_visualizations(
        {"run_visualizations": True, "save_per_prompt_metrics": False}
    )
    assert msg is not None
    assert "save_per_prompt_metrics" in msg

    assert (
        validate_extras_for_visualizations(
            {"run_visualizations": True, "save_per_prompt_metrics": True}
        )
        is None
    )


def test_extras_has_enabled_work_false_when_all_off() -> None:
    assert not extras_has_enabled_work(
        {
            "save_per_prompt_metrics": False,
            "run_tfidf_baseline": False,
            "run_visualizations": False,
        }
    )
    assert extras_has_enabled_work({"save_per_prompt_metrics": True})


def test_save_per_prompt_metrics_csvs_minimal(tmp_path) -> None:
    per_ir = pd.DataFrame(
        [
            {"prompt_id": "p1", "F1@5": 0.4, "AP@5": 0.3},
        ]
    )
    resultados = {
        "model_a": {
            "ir_metrics": {"per_prompt": per_ir},
            "votes_metrics": {"per_prompt": pd.DataFrame()},
        }
    }
    paths = save_per_prompt_metrics_csvs(resultados, tmp_path, verbose=False)
    assert len(paths) == 1
    out = tmp_path / "model_a_metrics_per_prompt.csv"
    assert out.is_file()
    reread = pd.read_csv(out)
    assert reread.loc[0, "prompt_id"] == "p1"
