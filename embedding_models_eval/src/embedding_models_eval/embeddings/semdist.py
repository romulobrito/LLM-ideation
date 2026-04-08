"""
Provider de embeddings usando SemDist framework.

Placeholder para integracao futura. Demonstra como adicionar um novo provider.
"""

from typing import Dict, List
import numpy as np

from .base import EmbeddingProvider, register_provider


class SemDistProvider(EmbeddingProvider):
    """
    Provider de embeddings usando SemDist framework.
    
    TODO: Integrar com framework do Ricardo quando disponivel.
    
    Configuracao exemplo (a ser definida):
    {
        "model_name": "...",
        "config_path": "...",
        ...
    }
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        # TODO: Inicializar SemDist quando framework estiver disponivel
        raise NotImplementedError(
            "SemDist provider ainda nao implementado. "
            "Aguardando framework do Ricardo."
        )
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Gera embeddings usando SemDist.
        
        Args:
            texts: Lista de strings
            
        Returns:
            Array numpy (n_texts, embedding_dim)
        """
        # TODO: Implementar quando framework estiver disponivel
        raise NotImplementedError("SemDist embed() ainda nao implementado")


# Registra automaticamente (mesmo sendo placeholder)
register_provider("semdist", SemDistProvider)
