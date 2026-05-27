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
from .votes_metrics import VotesMetrics
from .feasible_range_ratio import FeasibleRangeRatioMetrics

try:
    from .ir_metrics import IRMetrics
except Exception:  # pragma: no cover - defensive for optional runtime deps
    IRMetrics = None

__all__ = [
    "Metric",
    "register_metric",
    "get_metric",
    "list_metrics",
    "compute_all_metrics",
    "VotesMetrics",
    "FeasibleRangeRatioMetrics",
]

if IRMetrics is not None:
    __all__.append("IRMetrics")
