"""
Interface base para Metricas de Avaliacao.

Permite adicionar novas metricas implementando apenas esta interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd


class Metric(ABC):
    """
    Interface base para metricas de avaliacao.
    
    Para adicionar uma nova metrica:
    1. Herdar desta classe
    2. Implementar compute()
    3. Registrar com register_metric()
    """
    
    def __init__(self, name: str, k_values: List[int] = [1, 3, 5, 10]):
        """
        Inicializa a metrica.
        
        Args:
            name: Nome da metrica
            k_values: Lista de valores de k para calcular (ex: [1, 3, 5, 10])
        """
        self.name = name
        self.k_values = k_values
    
    @abstractmethod
    def compute(
        self,
        df: pd.DataFrame,
        rank_pred_col: str = "rank_pred",
        rank_gold_col: str = "rank_gold",
        prompt_id_col: str = "prompt_id",
    ) -> Dict:
        """
        Calcula a metrica.
        
        Args:
            df: DataFrame com rankings previstos e gold
            rank_pred_col: Nome da coluna com ranking previsto
            rank_gold_col: Nome da coluna com ranking gold
            prompt_id_col: Nome da coluna com ID do prompt
            
        Returns:
            Dicionario com resultados da metrica
            Formato: {
                "per_prompt": DataFrame com metricas por prompt,
                "macro": Dict com metricas agregadas
            }
        """
        pass


# Sistema de registro para extensibilidade
_METRICS: Dict[str, type] = {}


def register_metric(name: str, metric_class: type):
    """
    Registra uma nova metrica.
    
    Args:
        name: Nome unico da metrica
        metric_class: Classe que herda de Metric
    """
    if not issubclass(metric_class, Metric):
        raise TypeError(f"{metric_class} deve herdar de Metric")
    _METRICS[name] = metric_class


def get_metric(name: str, **kwargs) -> Metric:
    """
    Cria uma instancia de uma metrica registrada.
    
    Args:
        name: Nome da metrica registrada
        **kwargs: Argumentos para o construtor da metrica
        
    Returns:
        Instancia da metrica
    """
    if name not in _METRICS:
        available = ", ".join(_METRICS.keys())
        raise ValueError(
            f"Metrica '{name}' nao encontrada. "
            f"Metricas disponiveis: {available}"
        )
    return _METRICS[name](**kwargs)


def compute_all_metrics(
    df: pd.DataFrame,
    metric_names: Optional[List[str]] = None,
    **kwargs
) -> Dict:
    """
    Calcula todas as metricas registradas (ou um subconjunto).
    
    Args:
        df: DataFrame com dados
        metric_names: Lista de nomes de metricas (None = todas)
        **kwargs: Argumentos adicionais para as metricas
        
    Returns:
        Dicionario com resultados de todas as metricas
    """
    if metric_names is None:
        metric_names = list(_METRICS.keys())
    
    results = {}
    for name in metric_names:
        metric = get_metric(name, **kwargs)
        results[name] = metric.compute(df)
    
    return results


def list_metrics() -> List[str]:
    """Retorna lista de metricas registradas."""
    return list(_METRICS.keys())
