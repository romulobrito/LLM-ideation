"""
Feasible-range vote ratios for top-k evaluation.

This metric normalizes observed top-k vote statistics against the
best/worst feasible ranges for the same prompt and k.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .base import Metric, register_metric


class FeasibleRangeRatioMetrics(Metric):
    """
    Computes top-k vote ratios normalized by feasible ranges.

    For each prompt and each k:
    - ratio_max@k
    - ratio_mean@k
    - ratio_min@k

    All ratios are clipped to [0, 1].
    """

    def __init__(
        self,
        k_values: List[int] | None = None,
        votes_col: str = "likes",
        rank_pred_col: str = "rank_pred",
        prompt_id_col: str = "prompt_id",
        winner_rank_col: str = "rank_in_prompt",
        winner_rank_value: int = 1,
    ) -> None:
        if k_values is None:
            k_values = [1, 2, 3, 4, 5, 6, 7]
        super().__init__("feasible_range_ratio", k_values)
        self.votes_col = votes_col
        self.rank_pred_col = rank_pred_col
        self.prompt_id_col = prompt_id_col
        self.winner_rank_col = winner_rank_col
        self.winner_rank_value = winner_rank_value

    @staticmethod
    def _safe_ratio(observed: float, lower: float, upper: float) -> float:
        """
        Returns normalized ratio in [0, 1] for an observed value.

        If upper == lower, returns 1.0 (no discriminative range).
        """
        den = upper - lower
        if den == 0:
            return 1.0
        ratio = (observed - lower) / den
        return float(np.clip(ratio, 0.0, 1.0))

    def compute(
        self,
        df: pd.DataFrame,
        rank_pred_col: str = "rank_pred",
        rank_gold_col: str = "rank_gold",
        prompt_id_col: str = "prompt_id",
    ) -> Dict:
        """
        Computes per-prompt and macro feasible-range ratios.

        Args:
            df: Ranked rows by prompt.
            rank_pred_col: Unused override kept for Metric interface compatibility.
            rank_gold_col: Unused override kept for Metric interface compatibility.
            prompt_id_col: Unused override kept for Metric interface compatibility.

        Returns:
            {
                "per_prompt": DataFrame with ratio columns per prompt,
                "macro": Dict with mean/std for each ratio@k
            }
        """
        required_cols = {self.prompt_id_col, self.rank_pred_col, self.votes_col}
        if self.winner_rank_col in df.columns:
            required_cols.add(self.winner_rank_col)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for feasible_range_ratio: {missing}")

        per_prompt_rows: List[Dict[str, float | str]] = []

        for pid, group in df.groupby(self.prompt_id_col):
            work = group.copy()

            # Exclude winner row when source rank is available.
            if self.winner_rank_col in work.columns:
                work = work[work[self.winner_rank_col] != self.winner_rank_value]

            # Keep rows that are rankable and have valid votes.
            work = work[work[self.rank_pred_col].notna() & work[self.votes_col].notna()]
            if work.empty:
                continue

            work = work.sort_values(self.rank_pred_col, ascending=True, kind="mergesort")
            votes_pred_order = work[self.votes_col].astype(float).tolist()
            votes_desc = sorted(votes_pred_order, reverse=True)
            total = len(votes_desc)
            if total == 0:
                continue

            row: Dict[str, float | str] = {"prompt_id": pid}

            for k in self.k_values:
                n_eff = min(int(k), total)
                top_votes = votes_pred_order[:n_eff]

                obs_max = float(max(top_votes))
                obs_mean = float(sum(top_votes) / n_eff)
                obs_min = float(min(top_votes))

                best_slice = votes_desc[:n_eff]
                worst_slice = votes_desc[total - n_eff :]

                max_best = float(best_slice[0])
                max_worst = float(worst_slice[0])

                mean_best = float(sum(best_slice) / n_eff)
                mean_worst = float(sum(worst_slice) / n_eff)

                min_best = float(best_slice[-1])
                min_worst = float(worst_slice[-1])

                row[f"ratio_max@{k}"] = self._safe_ratio(obs_max, max_worst, max_best)
                row[f"ratio_mean@{k}"] = self._safe_ratio(obs_mean, mean_worst, mean_best)
                row[f"ratio_min@{k}"] = self._safe_ratio(obs_min, min_worst, min_best)

            per_prompt_rows.append(row)

        if not per_prompt_rows:
            return {"per_prompt": pd.DataFrame(), "macro": {}}

        per_prompt_df = pd.DataFrame(per_prompt_rows)
        macro_results: Dict[str, float] = {}

        for k in self.k_values:
            for metric_name in (f"ratio_max@{k}", f"ratio_mean@{k}", f"ratio_min@{k}"):
                vals = per_prompt_df[metric_name].dropna().astype(float).values
                macro_results[f"{metric_name}_mean"] = float(np.mean(vals)) if len(vals) else 0.0
                macro_results[f"{metric_name}_std"] = (
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                )

        return {"per_prompt": per_prompt_df, "macro": macro_results}


register_metric("feasible_range_ratio", FeasibleRangeRatioMetrics)
register_metric("feasible_range_ratio_metrics", FeasibleRangeRatioMetrics)

