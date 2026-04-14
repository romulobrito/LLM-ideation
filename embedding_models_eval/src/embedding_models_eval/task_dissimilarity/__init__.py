"""
Ranking por dissimilaridade de embedding entre uma tarefa T e candidatas h_i.

Modulo aditivo: nao altera o pipeline embedding-eval. Reutiliza EmbeddingProvider.
"""

from .config import load_task_embedding_yaml, provider_from_task_config
from .io import parse_task_rank_input, serialize_task_rank_output
from .legacy_input import extract_task_inputs_from_legacy_payload
from .ranking import rank_stories_by_task_dissimilarity

__all__ = [
    "load_task_embedding_yaml",
    "provider_from_task_config",
    "parse_task_rank_input",
    "serialize_task_rank_output",
    "extract_task_inputs_from_legacy_payload",
    "rank_stories_by_task_dissimilarity",
]
