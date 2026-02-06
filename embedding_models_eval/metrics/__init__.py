"""
Modulo de Metricas de Avaliacao.

Permite adicionar novas metricas sem modificar codigo existente.
Novas metricas devem implementar a interface base e ser registradas.
"""

from .base import (
    Metric,
    register_metric,
    get_metric,
    list_metrics,
    compute_all_metrics,
)
from .ir_metrics import IRMetrics
from .votes_metrics import VotesMetrics

__all__ = [
    "Metric",
    "register_metric",
    "get_metric",
    "list_metrics",
    "compute_all_metrics",
    "IRMetrics",
    "VotesMetrics",
]
