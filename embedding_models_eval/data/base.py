"""
Interface base para Dataset Loaders.

Permite adicionar novos loaders implementando apenas esta interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd


class DatasetLoader(ABC):
    """
    Interface base para loaders de dataset.
    
    Para adicionar um novo loader:
    1. Herdar desta classe
    2. Implementar load()
    3. Registrar com register_loader()
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o loader com configuracoes.
        
        Args:
            config: Dicionario com configuracoes especificas do loader
        """
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
    
    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        """
        Carrega um dataset e retorna DataFrame padronizado.
        
        Args:
            path: Caminho para o arquivo do dataset
            
        Returns:
            DataFrame com schema padronizado:
            - Chaves de agrupamento (ex: contest_number, context_prompt_url)
            - rank_in_prompt (ranking gold baseado em likes)
            - Coluna de texto configurável (ex: extracted_idea_250)
            - Outras colunas do dataset original
        """
        pass
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Valida se o DataFrame tem o schema esperado.
        
        Args:
            df: DataFrame para validar
            
        Returns:
            True se schema valido, False caso contrario
        """
        required_cols = [
            "rank_in_prompt",
        ]
        
        for col in required_cols:
            if col not in df.columns:
                return False
        
        return True


# Sistema de registro para extensibilidade
_LOADERS: Dict[str, type] = {}


def register_loader(name: str, loader_class: type):
    """
    Registra um novo loader de dataset.
    
    Permite adicionar novos loaders sem modificar codigo existente.
    
    Args:
        name: Nome unico do loader (ex: "json", "csv", "parquet")
        loader_class: Classe que herda de DatasetLoader
        
    Example:
        @register_loader("meu_loader")
        class MeuLoader(DatasetLoader):
            ...
    """
    if not issubclass(loader_class, DatasetLoader):
        raise TypeError(f"{loader_class} deve herdar de DatasetLoader")
    _LOADERS[name] = loader_class


def get_loader(name: str, config: Dict) -> DatasetLoader:
    """
    Cria uma instancia de um loader registrado.
    
    Args:
        name: Nome do loader registrado
        config: Configuracoes para o loader
        
    Returns:
        Instancia do loader
        
    Raises:
        ValueError: Se o loader nao estiver registrado
    """
    if name not in _LOADERS:
        available = ", ".join(_LOADERS.keys())
        raise ValueError(
            f"Loader '{name}' nao encontrado. "
            f"Loaders disponiveis: {available}"
        )
    return _LOADERS[name](config)


def list_loaders() -> List[str]:
    """Retorna lista de loaders registrados."""
    return list(_LOADERS.keys())
