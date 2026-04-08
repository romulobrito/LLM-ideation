"""
Interface base para Embedding Providers.

Permite adicionar novos providers implementando apenas esta interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np


class EmbeddingProvider(ABC):
    """
    Interface base para providers de embedding.
    
    Para adicionar um novo provider:
    1. Herdar desta classe
    2. Implementar embed()
    3. Registrar com @register_provider ou register_provider()
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o provider com configuracoes.
        
        Args:
            config: Dicionario com configuracoes especificas do provider
        """
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Gera embeddings para uma lista de textos.
        
        Args:
            texts: Lista de strings para embedar
            
        Returns:
            Array numpy de shape (n_texts, embedding_dim) com embeddings normalizados
        """
        pass
    
    def __call__(self, texts: List[str]) -> np.ndarray:
        """Permite usar o provider como funcao."""
        return self.embed(texts)


# Sistema de registro para extensibilidade
_PROVIDERS: Dict[str, type] = {}


def register_provider(name: str, provider_class: type):
    """
    Registra um novo provider de embedding.
    
    Permite adicionar novos providers sem modificar codigo existente.
    
    Args:
        name: Nome unico do provider (ex: "sentence_transformers", "openai")
        provider_class: Classe que herda de EmbeddingProvider
        
    Example:
        @register_provider("meu_provider")
        class MeuProvider(EmbeddingProvider):
            ...
    """
    if not issubclass(provider_class, EmbeddingProvider):
        raise TypeError(f"{provider_class} deve herdar de EmbeddingProvider")
    _PROVIDERS[name] = provider_class


def get_provider(name: str, config: Dict) -> EmbeddingProvider:
    """
    Cria uma instancia de um provider registrado.
    
    Args:
        name: Nome do provider registrado
        config: Configuracoes para o provider
        
    Returns:
        Instancia do provider
        
    Raises:
        ValueError: Se o provider nao estiver registrado
    """
    if name not in _PROVIDERS:
        available = ", ".join(_PROVIDERS.keys())
        raise ValueError(
            f"Provider '{name}' nao encontrado. "
            f"Providers disponiveis: {available}"
        )
    return _PROVIDERS[name](config)


def list_providers() -> List[str]:
    """Retorna lista de providers registrados."""
    return list(_PROVIDERS.keys())
