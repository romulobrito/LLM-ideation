"""
Modulo de Embedding Providers.

Permite adicionar novos providers de embedding sem modificar codigo existente.
Novos providers devem herdar de EmbeddingProvider e ser registrados.
"""

from .base import (
    EmbeddingProvider,
    register_provider,
    get_provider,
    list_providers,
)
from .sentence_transformers import SentenceTransformersProvider
from .openai import OpenAIProvider
from .semdist import SemDistProvider

__all__ = [
    "EmbeddingProvider",
    "register_provider",
    "get_provider",
    "list_providers",
    "SentenceTransformersProvider",
    "OpenAIProvider",
    "SemDistProvider",
]
