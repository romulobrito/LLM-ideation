"""
Integration-level checks for feasible_range_ratio in pipeline comparison table.
"""

from __future__ import annotations

from embedding_models_eval.pipeline.runner import build_comparison_table


def test_build_comparison_table_includes_feasible_ratio_means() -> None:
    resultados_por_modelo = {
        "model_x": {
            "ir_metrics": {"macro": {"MAP@1": 0.1}},
            "votes_metrics": {"macro": {"norm_mean_votes@1": 0.2}},
            "feasible_range_ratio_metrics": {
                "macro": {
                    "ratio_max@1_mean": 0.9,
                    "ratio_max@1_std": 0.1,  # std should not appear in comparison table
                    "ratio_mean@3_mean": 0.4,
                    "ratio_min@7_mean": 0.2,
                }
            },
        }
    }

    table = build_comparison_table(resultados_por_modelo)
    assert not table.empty
    assert "Modelo" in table.columns
    assert "ratio_max@1_mean" in table.columns
    assert "ratio_mean@3_mean" in table.columns
    assert "ratio_min@7_mean" in table.columns
    assert "ratio_max@1_std" not in table.columns
    assert table.loc[0, "ratio_max@1_mean"] == 0.9

