"""
JSON de entrada e saida para ranking por dissimilaridade.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def parse_task_rank_input(data: Dict[str, Any]) -> tuple[str, List[Dict[str, str]]]:
    """
    Valida payload de entrada e retorna (task_description, stories).

    Raises:
        ValueError: schema invalido.
    """
    if not isinstance(data, dict):
        raise ValueError("entrada deve ser um objeto JSON")
    task = data.get("task_description")
    if task is None or not str(task).strip():
        raise ValueError("task_description e obrigatorio (string nao vazia)")
    stories = data.get("stories")
    if not isinstance(stories, list) or len(stories) == 0:
        raise ValueError("stories deve ser lista nao vazia")
    out: List[Dict[str, str]] = []
    for i, s in enumerate(stories):
        if not isinstance(s, dict):
            raise ValueError(f"stories[{i}] deve ser objeto")
        sid = s.get("story_id")
        stext = s.get("story_text")
        if sid is None or not str(sid).strip():
            raise ValueError(f"stories[{i}].story_id obrigatorio")
        if stext is None or not str(stext).strip():
            raise ValueError(f"stories[{i}].story_text obrigatorio")
        out.append({"story_id": str(sid).strip(), "story_text": str(stext).strip()})
    return str(task).strip(), out


def serialize_task_rank_output(payload: Dict[str, Any]) -> str:
    """Serializa saida com JSON ASCII (uma linha opcional indentada)."""
    return json.dumps(payload, ensure_ascii=True, indent=2)


def parse_task_rank_input_from_path(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
