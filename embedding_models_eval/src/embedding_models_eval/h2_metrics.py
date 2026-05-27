"""
H2 metrics utilities under the same protocol used in H1.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np

from embedding_models_eval.metrics.feasible_range_random_baseline import (
    compute_feasible_ratio_for_votes,
)

PromptId = str
StoryId = str


def group_rows_by_prompt(rows: Iterable[dict]) -> Dict[PromptId, List[dict]]:
    """
    Group legacy rows by prompt id.

    Prompt id convention:
    - "{contest_number}::{context_prompt_url}"
    """
    grouped: Dict[PromptId, List[dict]] = defaultdict(list)
    for row in rows:
        contest_number = str(row.get("contest_number") or "")
        prompt_url = str(row.get("context_prompt_url") or "")
        prompt_id = f"{contest_number}::{prompt_url}"
        grouped[prompt_id].append(row)
    return dict(grouped)


def build_story_votes_map(prompt_rows: Sequence[dict]) -> Dict[StoryId, float]:
    """Build a story_id -> likes map from prompt rows."""
    out: Dict[StoryId, float] = {}
    for row in prompt_rows:
        story_id = str(row.get("story_url") or "").strip()
        likes = row.get("likes")
        if not story_id or likes is None:
            continue
        out[story_id] = float(likes)
    return out


def _exclude_winner(story_votes: Dict[StoryId, float]) -> Dict[StoryId, float]:
    """Exclude the single highest-like story (winner proxy)."""
    if not story_votes:
        return {}
    winner_story = max(story_votes.items(), key=lambda item: (item[1], item[0]))[0]
    return {sid: v for sid, v in story_votes.items() if sid != winner_story}


def compute_prompt_h2_metrics(
    predicted_story_ids: Sequence[StoryId],
    story_votes: Dict[StoryId, float],
    k_values: Sequence[int],
    *,
    exclude_winner: bool = True,
) -> Dict[str, float]:
    """
    Compute prompt-level H2 metrics with feasible-range ratios.

    Returns keys:
    - ratio_max@k, ratio_mean@k, ratio_min@k
    - votes_top1_ratio
    """
    if not predicted_story_ids:
        raise ValueError("predicted_story_ids cannot be empty")

    base_votes = dict(story_votes)
    if exclude_winner:
        base_votes = _exclude_winner(base_votes)

    if not base_votes:
        raise ValueError("No candidate stories available after winner exclusion")

    votes_pred_order: List[float] = []
    for sid in predicted_story_ids:
        if sid in base_votes:
            votes_pred_order.append(float(base_votes[sid]))

    if not votes_pred_order:
        raise ValueError("Predicted ranking has no overlap with candidate stories")

    out: Dict[str, float] = {}
    for k in k_values:
        rmax, rmean, rmin = compute_feasible_ratio_for_votes(votes_pred_order, int(k))
        out[f"ratio_max@{k}"] = float(rmax)
        out[f"ratio_mean@{k}"] = float(rmean)
        out[f"ratio_min@{k}"] = float(rmin)

    top1_votes = float(votes_pred_order[0])
    best_votes = float(max(base_votes.values()))
    out["votes_top1_ratio"] = float(top1_votes / best_votes) if best_votes > 0 else 0.0
    return out


def aggregate_prompt_metrics(prompt_metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate prompt-level metrics into macro mean/std keys."""
    if not prompt_metrics:
        raise ValueError("prompt_metrics cannot be empty")

    metric_names = sorted(prompt_metrics[0].keys())
    out: Dict[str, float] = {}
    for name in metric_names:
        vals = [float(row[name]) for row in prompt_metrics if name in row]
        out[f"{name}_mean"] = float(np.mean(vals)) if vals else 0.0
        out[f"{name}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def build_macro_row(model_name: str, macro: Dict[str, float]) -> Dict[str, float | str]:
    """Build one comparacao_modelos_macro row with expected schema."""
    row: Dict[str, float | str] = {"Modelo": model_name}
    row.update(macro)
    return row


def build_h2_detailed_rows(
    prompt_id: str,
    prompt_rows: Sequence[dict],
    predicted_story_ids: Sequence[str],
) -> List[Dict[str, float | str | None]]:
    """
    Build detailed rows compatible with feasible-range random baseline script.

    Output columns include:
    - prompt_id
    - story_url
    - likes
    - rank_in_prompt (proxy from likes ordering)
    - rank_pred (H2 ranking position, may be None)
    """
    votes_map = build_story_votes_map(prompt_rows)
    if not votes_map:
        return []

    sorted_stories = sorted(votes_map.items(), key=lambda item: (-item[1], item[0]))
    gold_rank_map: Dict[str, int] = {}
    for idx, (story_id, _) in enumerate(sorted_stories, start=1):
        gold_rank_map[story_id] = idx

    pred_rank_map: Dict[str, int] = {}
    for idx, story_id in enumerate(predicted_story_ids, start=1):
        if story_id in votes_map and story_id not in pred_rank_map:
            pred_rank_map[story_id] = idx

    out: List[Dict[str, float | str | None]] = []
    for story_id, likes in sorted(votes_map.items(), key=lambda item: item[0]):
        out.append(
            {
                "prompt_id": str(prompt_id),
                "story_url": str(story_id),
                "likes": float(likes),
                "rank_in_prompt": int(gold_rank_map[story_id]),
                "rank_pred": int(pred_rank_map[story_id]) if story_id in pred_rank_map else None,
            }
        )
    return out
