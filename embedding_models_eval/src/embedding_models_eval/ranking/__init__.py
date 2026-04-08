"""
Modulo de Metodos de Ranking.

Permite adicionar novos metodos de ranking sem modificar codigo existente.
"""

from .anchor import build_anchor_ranking

__all__ = [
    "build_anchor_ranking",
]
