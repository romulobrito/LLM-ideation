"""
YAML de embedding para ranking por dissimilaridade (schema dedicado).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from embedding_models_eval.embeddings import get_provider
from embedding_models_eval.embeddings.base import EmbeddingProvider
from embedding_models_eval.pipeline.config_loader import load_env_robust, substitute_env_vars


def load_task_embedding_yaml(path: str) -> Dict[str, Any]:
    """
    Carrega arquivo YAML com secao obrigatoria ``embedding``.

    Returns:
        Dict completo (tipicamente so ``embedding`` e comentarios ignorados).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"YAML nao encontrado: {path}")
    load_env_robust()
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("YAML raiz deve ser um mapa")
    cfg = substitute_env_vars(raw)
    if "embedding" not in cfg or not isinstance(cfg["embedding"], dict):
        raise ValueError("YAML deve conter secao 'embedding' (mapa)")
    return cfg


def validate_embedding_section(emb: Dict[str, Any]) -> None:
    """Valida chaves minimas da secao embedding."""
    backend = emb.get("backend")
    if backend not in ("sentence_transformers", "openai"):
        raise ValueError(
            "embedding.backend deve ser 'sentence_transformers' ou 'openai'"
        )
    model = emb.get("model")
    if not model or not str(model).strip():
        raise ValueError("embedding.model e obrigatorio")
    if backend == "openai":
        env_name = str(emb.get("api_key_env", "OPENAI_API_KEY"))
        if not emb.get("api_key") and not os.getenv(env_name):
            raise ValueError(
                f"embedding OpenAI: defina env {env_name} ou embedding.api_key"
            )


def provider_from_task_config(cfg: Dict[str, Any]) -> Tuple[EmbeddingProvider, str, str]:
    """
    Constroi EmbeddingProvider a partir do YAML carregado.

    Returns:
        (provider, backend_label, model_label) para metadados de saida JSON.
    """
    validate_embedding_section(cfg["embedding"])
    emb = cfg["embedding"]
    backend = str(emb["backend"])
    model = str(emb["model"]).strip()

    if backend == "sentence_transformers":
        st_cfg: Dict[str, Any] = {
            "model_name": model,
            "batch_size": int(emb.get("batch_size", 32)),
            "normalize": bool(emb.get("normalize", True)),
            "device": str(emb.get("device", "cpu")),
        }
        return get_provider("sentence_transformers", st_cfg), "sentence_transformers", model

    # openai (OpenAI oficial ou API compativel via base_url, ex. OpenRouter)
    env_name = str(emb.get("api_key_env", "OPENAI_API_KEY"))
    api_key = emb.get("api_key")
    if not api_key:
        api_key = os.getenv(env_name)
    oa_cfg: Dict[str, Any] = {
        "model_name": model,
        "api_key": api_key,
        "batch_size": int(emb.get("batch_size", 100)),
        "normalize": bool(emb.get("normalize", True)),
    }
    if emb.get("base_url"):
        oa_cfg["base_url"] = str(emb["base_url"]).strip()
    return get_provider("openai", oa_cfg), "openai", model
