"""
Minimal tests aligned with sprint item 5:

- Anchor selection per prompt (rank_in_prompt == anchor_rank)
- Deterministic ranking (stable mergesort + tie-break on rank_col)
- Output schema for df_scored
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from embedding_models_eval.embeddings.base import EmbeddingProvider
from embedding_models_eval.ranking.anchor import build_anchor_ranking


class FakeNormalizedProvider(EmbeddingProvider):
    """
    Returns a fixed stack of L2-normalized vectors, one row per call to embed().

    The i-th row of ``vectors`` corresponds to the i-th text in the same order
    ``build_anchor_ranking`` passes to ``embed``.
    """

    def __init__(self, vectors: np.ndarray, config: Optional[Dict] = None) -> None:
        cfg = config if config is not None else {"name": "fake_normalized"}
        super().__init__(cfg)
        v = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        self._vectors = (v / norms).astype(np.float32)

    def embed(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        if n != self._vectors.shape[0]:
            raise ValueError(
                f"Fake provider expected {self._vectors.shape[0]} texts, got {n}"
            )
        return self._vectors.copy()


REQUIRED_SCORED_COLS = (
    "prompt_id",
    "doc_id",
    "score_to_anchor",
    "rank_pred",
    "rank_gold",
)


@pytest.fixture
def group_cols() -> List[str]:
    return ["contest_number", "context_prompt_url"]


def test_anchor_is_unique_rank_one_per_prompt(group_cols: List[str]) -> None:
    """Each prompt group uses exactly one anchor row (gold rank 1); anchor has no rank_pred."""
    # Two prompts, 3 rows each: ranks 1,2,3
    rows = []
    for p in ("url_a", "url_b"):
        for r in (1, 2, 3):
            rows.append(
                {
                    "contest_number": 1,
                    "context_prompt_url": p,
                    "rank_in_prompt": r,
                    "extracted_idea_250": f"idea_{p}_{r}",
                    "story_url": f"https://example.com/{p}/{r}",
                }
            )
    df = pd.DataFrame(rows)
    dim = 4
    emb = np.zeros((len(df), dim), dtype=np.float32)
    for i in range(len(df)):
        emb[i, i % dim] = 1.0
    provider = FakeNormalizedProvider(emb)

    out = build_anchor_ranking(
        df,
        provider,
        text_col="extracted_idea_250",
        group_cols=group_cols,
        rank_col="rank_in_prompt",
        anchor_rank=1,
        show_progress=False,
    )

    for pid in out["prompt_id"].unique():
        g = out[out["prompt_id"] == pid]
        anchors = g[g["rank_in_prompt"] == 1]
        assert len(anchors) == 1
        assert anchors["rank_pred"].isna().all()
        assert anchors["score_to_anchor"].notna().all()


def test_predicted_ranking_is_deterministic(group_cols: List[str]) -> None:
    """Same input twice yields identical rank_pred for all candidate rows."""
    df = pd.DataFrame(
        {
            "contest_number": [1, 1, 1],
            "context_prompt_url": ["u"] * 3,
            "rank_in_prompt": [1, 2, 3],
            "extracted_idea_250": ["a", "b", "c"],
            "story_url": ["s1", "s2", "s3"],
        }
    )
    emb = np.eye(3, 5, dtype=np.float32)
    provider = FakeNormalizedProvider(emb)

    out1 = build_anchor_ranking(
        df, provider, group_cols=group_cols, show_progress=False
    )
    out2 = build_anchor_ranking(
        df, provider, group_cols=group_cols, show_progress=False
    )

    c1 = out1[out1["rank_pred"].notna()]["rank_pred"].values
    c2 = out2[out2["rank_pred"].notna()]["rank_pred"].values
    np.testing.assert_array_equal(c1, c2)


def test_tie_break_uses_gold_rank_col_when_scores_equal(group_cols: List[str]) -> None:
    """
    With mergesort, tied score_to_anchor breaks by rank_in_prompt ascending
    (see ranking/anchor.py sort keys).
    """
    df = pd.DataFrame(
        {
            "contest_number": [1, 1, 1, 1],
            "context_prompt_url": ["u"] * 4,
            "rank_in_prompt": [1, 2, 3, 4],
            "extracted_idea_250": ["t0", "t1", "t2", "t3"],
            "story_url": ["s0", "s1", "s2", "s3"],
        }
    )
    # Same non-anchor embedding for rank 2 and 3 -> tie on cosine to anchor
    v0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_tie = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    v_low = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    emb = np.stack([v0, v_tie, v_tie, v_low], axis=0)
    provider = FakeNormalizedProvider(emb)

    out = build_anchor_ranking(
        df, provider, group_cols=group_cols, show_progress=False
    )
    cand = out[out["rank_pred"].notna()].set_index("story_url")

    # rank 2 row should be predicted before rank 3 when scores tie
    assert cand.loc["s1", "rank_pred"] < cand.loc["s2", "rank_pred"]
    assert cand.loc["s3", "rank_pred"] == 3


def test_output_schema_has_required_columns(group_cols: List[str]) -> None:
    """df_scored exposes the columns promised by the pipeline contract."""
    df = pd.DataFrame(
        {
            "contest_number": [1, 1],
            "context_prompt_url": ["u", "u"],
            "rank_in_prompt": [1, 2],
            "extracted_idea_250": ["x", "y"],
            "story_url": ["a", "b"],
        }
    )
    emb = np.eye(2, 3, dtype=np.float32)
    provider = FakeNormalizedProvider(emb)
    out = build_anchor_ranking(
        df, provider, group_cols=group_cols, show_progress=False
    )
    for col in REQUIRED_SCORED_COLS:
        assert col in out.columns

    cand = out[out["rank_pred"].notna()]
    assert cand["rank_pred"].dtype == np.float64 or np.issubdtype(
        cand["rank_pred"].dtype, np.floating
    )
    assert cand["rank_gold"].notna().all()
