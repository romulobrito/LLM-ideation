"""
Unit tests for feasible_range_ratio metric.
"""

from __future__ import annotations

import pandas as pd

from embedding_models_eval.metrics.base import get_metric, list_metrics
from embedding_models_eval.metrics.feasible_range_ratio import FeasibleRangeRatioMetrics


def _toy_df() -> pd.DataFrame:
    # Winner row (rank_in_prompt == 1) must be excluded by the metric.
    return pd.DataFrame(
        [
            {
                "prompt_id": "p1",
                "rank_in_prompt": 1,
                "rank_pred": None,
                "rank_gold": None,
                "likes": 100,
            },
            {
                "prompt_id": "p1",
                "rank_in_prompt": 2,
                "rank_pred": 3,
                "rank_gold": 1,
                "likes": 80,
            },
            {
                "prompt_id": "p1",
                "rank_in_prompt": 3,
                "rank_pred": 1,
                "rank_gold": 2,
                "likes": 60,
            },
            {
                "prompt_id": "p1",
                "rank_in_prompt": 4,
                "rank_pred": 2,
                "rank_gold": 3,
                "likes": 40,
            },
            {
                "prompt_id": "p2",
                "rank_in_prompt": 1,
                "rank_pred": None,
                "rank_gold": None,
                "likes": 50,
            },
            {
                "prompt_id": "p2",
                "rank_in_prompt": 2,
                "rank_pred": 1,
                "rank_gold": 1,
                "likes": 10,
            },
            {
                "prompt_id": "p2",
                "rank_in_prompt": 3,
                "rank_pred": 2,
                "rank_gold": 2,
                "likes": 10,
            },
        ]
    )


def test_metric_is_registered() -> None:
    assert "feasible_range_ratio" in list_metrics()
    metric = get_metric("feasible_range_ratio", k_values=[1, 2])
    assert isinstance(metric, FeasibleRangeRatioMetrics)


def test_ratio_values_follow_feasible_range_definition() -> None:
    df = _toy_df()
    metric = FeasibleRangeRatioMetrics(k_values=[1, 2, 3])
    result = metric.compute(df)
    per_prompt = result["per_prompt"].set_index("prompt_id")

    # Prompt p1, remaining votes in predicted order: [60, 40, 80]
    # Desc votes: [80, 60, 40]
    # k=1 -> best=80 worst=40 observed=60 => 0.5 for max/mean/min
    assert per_prompt.loc["p1", "ratio_max@1"] == 0.5
    assert per_prompt.loc["p1", "ratio_mean@1"] == 0.5
    assert per_prompt.loc["p1", "ratio_min@1"] == 0.5

    # k=2:
    # observed top2 votes [60,40] => obs_max=60 obs_mean=50 obs_min=40
    # best slice [80,60], worst slice [60,40]
    # ratio_max=(60-60)/(80-60)=0
    # ratio_mean=(50-50)/(70-50)=0
    # ratio_min=(40-40)/(60-40)=0
    assert per_prompt.loc["p1", "ratio_max@2"] == 0.0
    assert per_prompt.loc["p1", "ratio_mean@2"] == 0.0
    assert per_prompt.loc["p1", "ratio_min@2"] == 0.0

    # k=3 covers all items => denominator zero for all stats, rule returns 1.0
    assert per_prompt.loc["p1", "ratio_max@3"] == 1.0
    assert per_prompt.loc["p1", "ratio_mean@3"] == 1.0
    assert per_prompt.loc["p1", "ratio_min@3"] == 1.0


def test_outputs_have_expected_schema_and_bounds() -> None:
    df = _toy_df()
    metric = FeasibleRangeRatioMetrics(k_values=[1, 2])
    result = metric.compute(df)

    per_prompt = result["per_prompt"]
    macro = result["macro"]

    for col in ["prompt_id", "ratio_max@1", "ratio_mean@1", "ratio_min@1"]:
        assert col in per_prompt.columns

    ratio_cols = [c for c in per_prompt.columns if c.startswith("ratio_")]
    for col in ratio_cols:
        assert ((per_prompt[col] >= 0.0) & (per_prompt[col] <= 1.0)).all()

    for key in [
        "ratio_max@1_mean",
        "ratio_max@1_std",
        "ratio_mean@2_mean",
        "ratio_min@2_std",
    ]:
        assert key in macro

