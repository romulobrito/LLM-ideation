"""
Modulo Pipeline - Orquestracao completa de avaliacao de embeddings.

Fornece funcoes principais para executar o pipeline end-to-end.
"""

from .config_loader import load_config
from .from_config_dict import run_experiment_from_config_dict
from .runner import run_experiment

__all__ = [
    "load_config",
    "run_experiment",
    "run_experiment_from_config_dict",
]
