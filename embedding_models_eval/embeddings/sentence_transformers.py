"""
Provider de embeddings usando Sentence-Transformers.

Exemplo de como adicionar um novo provider:
1. Herdar de EmbeddingProvider
2. Implementar embed()
3. Registrar automaticamente no __init__.py
"""

from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import EmbeddingProvider, register_provider


class SentenceTransformersProvider(EmbeddingProvider):
    """
    Provider de embeddings usando Sentence-Transformers.
    
    Configuracao exemplo:
    {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 32,
        "normalize": True,
        "device": "cuda"  # ou "cpu"
    }
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.batch_size = config.get("batch_size", 32)
        self.normalize = config.get("normalize", True)
        self.device = config.get("device", "cpu")
        
        # Carrega modelo (lazy loading poderia ser implementado)
        self.model = SentenceTransformer(self.model_name, device=self.device)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Gera embeddings usando Sentence-Transformers.
        
        Args:
            texts: Lista de strings
            
        Returns:
            Array numpy (n_texts, embedding_dim) normalizado
        """
        if not texts:
            return np.array([]).reshape(0, 0)
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        
        return embeddings.astype(np.float32)


# Registra automaticamente
register_provider("sentence_transformers", SentenceTransformersProvider)
register_provider("st", SentenceTransformersProvider)  # Alias curto
