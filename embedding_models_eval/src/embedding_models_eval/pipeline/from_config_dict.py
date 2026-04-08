"""
Entry point for orchestrators when configuration is already a Python dict.

Additive API: does not replace ``load_config(path)`` or the CLI. Use this when
a platform (e.g. Open WebUI tool) builds or merges config in memory and must
call the same pipeline as ``embedding-eval`` without writing a YAML file.
"""

from __future__ import annotations

from typing import Any, Dict

from .config_loader import load_env_robust, substitute_env_vars, validate_config
from .runner import run_experiment


def run_experiment_from_config_dict(
    config: Dict[str, Any],
    *,
    load_env: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Validate and run the full embedding evaluation pipeline from a config dict.

    Execution steps (same contract as loading YAML via ``load_config``):

    1. Optionally load ``.env`` from the same search paths as ``load_config``.
    2. Recursively substitute ``${VAR}`` in string values (e.g. API keys).
    3. Run ``validate_config`` on the prepared mapping.
    4. Call ``run_experiment`` with that mapping.

    The original ``config`` object is not modified; substitution builds new
    dict/list structure (same behavior as ``load_config`` after YAML parse).

    Args:
        config: Mapping with required sections ``dataset``, ``models``, ``metrics``,
            ``ranking``, ``output``; optional ``pipeline_extras``.
        load_env: If True, call ``load_env_robust`` before substitution.
        **kwargs: Forwarded to ``run_experiment`` (e.g. ``verbose``,
            ``continue_on_error``).

    Returns:
        The dict returned by ``run_experiment`` (macro table, artifacts paths,
        per-prompt metrics, pipeline_extras report).

    Raises:
        ValueError: If validation fails or the runner raises.
    """
    if load_env:
        load_env_robust()
    prepared: Dict[str, Any] = substitute_env_vars(config)
    validate_config(prepared)
    return run_experiment(prepared, **kwargs)
