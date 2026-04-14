"""
Similaridade cosseno, dissimilaridade e normalizacao min-max [0, 1].
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from embedding_models_eval.embeddings.base import EmbeddingProvider


def _cosine_sim_01(emb_task: np.ndarray, emb_story: np.ndarray) -> float:
    """Cosseno L2-normalizado mapeado para [0, 1]: (cos + 1) / 2."""
    cos = float(np.dot(emb_task.ravel(), emb_story.ravel()))
    cos = max(-1.0, min(1.0, cos))
    return (cos + 1.0) / 2.0


def rank_stories_by_task_dissimilarity(
    task_description: str,
    stories: List[Dict[str, str]],
    provider: EmbeddingProvider,
    *,
    backend_label: str,
    model_label: str,
    strategy: str = "embedding_dissimilarity",
) -> Dict[str, Any]:
    """
    Embed T e cada h_i; similaridade em [0,1]; dissimilaridade = 1 - similaridade;
    ordena do maior dissimilaridade (melhor) para menor; normalized_score min-max.
    """
    if len(stories) < 1:
        raise ValueError("stories nao pode ser vazio")

    texts = [task_description] + [s["story_text"] for s in stories]
    embeddings = provider.embed(texts)
    if embeddings.shape[0] != len(texts):
        raise RuntimeError("provider.embed retornou contagem inesperada")

    e_task = embeddings[0]
    e_stories = embeddings[1:]

    rows: List[Dict[str, Any]] = []
    for i, s in enumerate(stories):
        sim_01 = _cosine_sim_01(e_task, e_stories[i])
        dis = 1.0 - sim_01
        rows.append(
            {
                "story_id": s["story_id"],
                "raw_similarity": round(sim_01, 6),
                "raw_dissimilarity": round(dis, 6),
            }
        )

    # Ordenar: maior dissimilaridade primeiro; desempate por story_id
    rows.sort(key=lambda r: (-r["raw_dissimilarity"], r["story_id"]))

    dis_vals = [r["raw_dissimilarity"] for r in rows]
    d_min = min(dis_vals)
    d_max = max(dis_vals)
    span = d_max - d_min

    for pos, r in enumerate(rows, start=1):
        if span <= 0.0:
            norm = 1.0
        else:
            norm = (r["raw_dissimilarity"] - d_min) / span
        r["normalized_score"] = round(float(norm), 6)
        r["rank_position"] = pos

    return {
        "strategy": strategy,
        "embedding_backend": backend_label,
        "embedding_model": model_label,
        "ranking": rows,
    }
