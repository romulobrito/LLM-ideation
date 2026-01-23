"""
Modulo de Data Loaders.

Permite adicionar novos loaders de dataset sem modificar codigo existente.
Novos loaders devem implementar a interface base e ser registrados.
"""

from .base import DatasetLoader, register_loader, get_loader, list_loaders
from .json_loader import JSONLoader, load_dataset

__all__ = [
    "DatasetLoader",
    "register_loader",
    "get_loader",
    "list_loaders",
    "JSONLoader",
    "load_dataset",  
]
