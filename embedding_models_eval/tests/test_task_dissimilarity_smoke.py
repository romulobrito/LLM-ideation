"""
Smoke: ranking por dissimilaridade com provider fake (sem rede, sem torch pesado).
"""

from __future__ import annotations

from typing import List

import numpy as np

from embedding_models_eval.embeddings.base import EmbeddingProvider
from embedding_models_eval.task_dissimilarity.io import parse_task_rank_input
from embedding_models_eval.task_dissimilarity.legacy_input import (
    extract_task_inputs_from_legacy_payload,
)
from embedding_models_eval.task_dissimilarity.ranking import (
    rank_stories_by_task_dissimilarity,
)


class _FakeProvider(EmbeddingProvider):
    """Embeddings fixos: historia 0 alinhada com tarefa; historia 1 ortogonal."""

    def __init__(self) -> None:
        super().__init__({})
        self._dim = 4

    def embed(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        out = np.zeros((n, self._dim), dtype=np.float32)
        base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        base = base / np.linalg.norm(base)
        for i in range(n):
            if i == 0:
                out[i] = base
            else:
                v = np.roll(base, i % self._dim)
                out[i] = v / np.linalg.norm(v)
        return out


def test_parse_task_rank_input_fixture() -> None:
    from pathlib import Path

    fix = Path(__file__).resolve().parent / "fixtures" / "task_rank_minimal_input.json"
    import json

    data = json.loads(fix.read_text(encoding="utf-8"))
    task, stories = parse_task_rank_input(data)
    assert "login" in task.lower()
    assert len(stories) == 2


def test_rank_dissimilarity_order_and_normalization() -> None:
    task = "task T"
    stories = [
        {"story_id": "a", "story_text": "same direction as T"},
        {"story_id": "b", "story_text": "different direction"},
    ]
    prov = _FakeProvider()
    out = rank_stories_by_task_dissimilarity(
        task,
        stories,
        prov,
        backend_label="fake",
        model_label="fake-model",
    )
    assert out["strategy"] == "embedding_dissimilarity"
    assert out["embedding_backend"] == "fake"
    assert len(out["ranking"]) == 2
    r0 = out["ranking"][0]
    r1 = out["ranking"][1]
    assert r0["normalized_score"] >= r1["normalized_score"]
    assert r0["rank_position"] == 1
    assert r1["rank_position"] == 2
    assert 0.0 <= r0["normalized_score"] <= 1.0


def test_load_task_yaml_validates_without_loading_model(tmp_path) -> None:
    from embedding_models_eval.task_dissimilarity.config import (
        load_task_embedding_yaml,
        validate_embedding_section,
    )

    y = tmp_path / "t.yaml"
    y.write_text(
        """
embedding:
  backend: sentence_transformers
  model: sentence-transformers/all-MiniLM-L6-v2
  batch_size: 8
  device: cpu
  normalize: true
""",
        encoding="utf-8",
    )
    cfg = load_task_embedding_yaml(str(y))
    validate_embedding_section(cfg["embedding"])
    assert cfg["embedding"]["backend"] == "sentence_transformers"


def test_legacy_json_nested_to_task_format() -> None:
    """JSON legado minimo (concurso/prompt/texts) -> schema task_description + stories."""
    payload = [
        {
            "Number": "1",
            "Title": "Contest",
            "Prompts": [
                {
                    "Title": "Write about cats",
                    "URL": "https://example.com/p1",
                    "Texts": [
                        {
                            "URL": "https://example.com/s1",
                            "Extracted_idea": {"250": "Cat story A"},
                        },
                        {
                            "URL": "https://example.com/s2",
                            "Extracted_idea": {"250": "Cat story B"},
                        },
                    ],
                }
            ],
        }
    ]
    items = extract_task_inputs_from_legacy_payload(payload, max_prompts=10)
    assert len(items) == 1
    assert items[0]["task_description"] == "Write about cats"
    assert len(items[0]["stories"]) == 2
    task, stories = parse_task_rank_input(items[0])
    assert task == "Write about cats"
    assert len(stories) == 2
