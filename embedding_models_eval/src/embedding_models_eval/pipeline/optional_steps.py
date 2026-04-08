"""
Etapas opcionais do pipeline (aditivas): export per-prompt CSV, TF-IDF, visualizacoes.

Todas desligadas por padrao via YAML (pipeline_extras). Imports pesados sao lazy
onde faz sentido (ex.: suite de plots).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _resolve_path(
    results_dir: str,
    explicit: Optional[str],
    subdir_fallback: str,
) -> Path:
    """
    Resolve diretorio de saida: path explicito (abs ou relativo ao cwd) ou
    results_dir / subdir_fallback.
    """
    if explicit is not None and str(explicit).strip() != "":
        p = Path(explicit)
        return p if p.is_absolute() else Path.cwd() / p
    return Path(results_dir) / subdir_fallback


def get_pipeline_extras(config: Dict[str, Any]) -> Dict[str, Any]:
    """Secao pipeline_extras com defaults seguros (tudo desligado)."""
    raw = config.get("pipeline_extras")
    if not isinstance(raw, dict):
        return {}
    return raw


def extras_has_enabled_work(extras: Dict[str, Any]) -> bool:
    """True se alguma etapa opcional esta ligada."""
    return bool(
        extras.get("save_per_prompt_metrics")
        or extras.get("run_tfidf_baseline")
        or extras.get("run_visualizations")
    )


def save_per_prompt_metrics_csvs(
    resultados_por_modelo: Dict[str, Dict],
    dest_dir: Path,
    verbose: bool = True,
) -> List[str]:
    """
    Grava {modelo}_metrics_per_prompt.csv no mesmo formato de run_ranking_only.

    Returns:
        Lista de paths gravados.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    for modelo_name, resultados in resultados_por_modelo.items():
        ir_block = resultados.get("ir_metrics") or {}
        votes_block = resultados.get("votes_metrics") or {}
        per_prompt_ir = ir_block.get("per_prompt")
        if per_prompt_ir is None or (
            isinstance(per_prompt_ir, pd.DataFrame) and per_prompt_ir.empty
        ):
            continue
        per_prompt_votes = votes_block.get("per_prompt")
        if (
            per_prompt_votes is not None
            and isinstance(per_prompt_votes, pd.DataFrame)
            and not per_prompt_votes.empty
        ):
            merged = per_prompt_ir.merge(
                per_prompt_votes,
                on="prompt_id",
                how="left",
            )
        else:
            merged = per_prompt_ir

        out_path = dest_dir / f"{modelo_name}_metrics_per_prompt.csv"
        merged.to_csv(out_path, index=False)
        saved.append(str(out_path))
        if verbose:
            print(f"   Per-prompt metrics: {out_path}")

    return saved


def run_tfidf_baseline_integrated(
    df: pd.DataFrame,
    text_col: str,
    config: Dict[str, Any],
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """
    Executa baseline TF-IDF e grava parquet, tfidf_metrics_per_prompt.csv, ranking_summary.
    Reutiliza funcoes de embedding_models_eval.tfidf_baseline_runner.
    """
    from embedding_models_eval.tfidf_baseline_runner import (
        build_tfidf_anchor_ranking,
        compute_ir_metrics,
        compute_votes_metrics,
        save_summary,
    )

    ranking_config = config.get("ranking", {})
    metrics_config = config.get("metrics", {})
    k_values = metrics_config.get("k_values", [1, 3, 5, 10])
    votes_col = ranking_config.get("votes_col", "likes")

    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("   TF-IDF: ranking por ancora...")
    df_scored = build_tfidf_anchor_ranking(
        df,
        text_col=text_col,
        group_cols=ranking_config.get(
            "group_cols", ["contest_number", "context_prompt_url"]
        ),
        rank_col=ranking_config.get("rank_col", "rank_in_prompt"),
        anchor_rank=ranking_config.get("anchor_rank", 1),
        show_progress=verbose,
    )

    parquet_path = output_dir / "tfidf_scored.parquet"
    df_scored.to_parquet(parquet_path, index=False)
    if verbose:
        print(f"   TF-IDF parquet: {parquet_path}")

    ir_result = compute_ir_metrics(df_scored, k_values=k_values, verbose=verbose)
    votes_result = compute_votes_metrics(
        df_scored,
        k_values=k_values,
        votes_col=votes_col,
        verbose=verbose,
    )

    per_prompt_ir = ir_result.get("per_prompt")
    per_prompt_votes = votes_result.get("per_prompt")
    if per_prompt_ir is not None and not per_prompt_ir.empty:
        if (
            per_prompt_votes is not None
            and isinstance(per_prompt_votes, pd.DataFrame)
            and not per_prompt_votes.empty
        ):
            merged = per_prompt_ir.merge(
                per_prompt_votes,
                on="prompt_id",
                how="left",
            )
        else:
            merged = per_prompt_ir
        mpath = output_dir / "tfidf_metrics_per_prompt.csv"
        merged.to_csv(mpath, index=False)
        if verbose:
            print(f"   TF-IDF per-prompt: {mpath}")

    save_summary(df_scored, ir_result, votes_result, str(output_dir), k_values=k_values)


def run_visualization_suite_integrated(
    ranking_dir: str,
    tfidf_dir: str,
    output_dir: str,
    n_bootstrap: int = 10000,
    verbose: bool = True,
) -> None:
    """Wrapper lazy sobre embedding_models_eval.visualization_suite."""
    from embedding_models_eval import visualization_suite as viz_module

    viz_module.run_visualization_suite(
        ranking_dir=ranking_dir,
        tfidf_dir=tfidf_dir,
        output_dir=output_dir,
        n_bootstrap=n_bootstrap,
        verbose=verbose,
    )


def validate_extras_for_visualizations(extras: Dict[str, Any]) -> Optional[str]:
    """
    Retorna mensagem de erro se combinacao invalida; None se OK.
    """
    if not extras.get("run_visualizations"):
        return None
    if not extras.get("save_per_prompt_metrics"):
        return (
            "pipeline_extras.run_visualizations requires "
            "pipeline_extras.save_per_prompt_metrics: true "
            "(CSV per-prompt dos modelos de embedding)."
        )
    return None


def run_optional_pipeline_steps(
    *,
    df: pd.DataFrame,
    text_col: str,
    config: Dict[str, Any],
    resultados_por_modelo: Dict[str, Dict],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executa pipeline_extras apos save_artifacts principal.

    Returns:
        Dict com paths resolvidos e listas de artefatos opcionais (para retorno).
    """
    out_cfg = config.get("output", {})
    results_dir = str(out_cfg.get("results_dir", "results"))
    extras = get_pipeline_extras(config)

    report: Dict[str, Any] = {
        "per_prompt_metrics_paths": [],
        "tfidf_output_dir": None,
        "visualizations_output_dir": None,
        "skipped_visualizations": None,
    }

    msg = validate_extras_for_visualizations(extras)
    if msg:
        raise ValueError(msg)

    if extras.get("save_per_prompt_metrics"):
        dest = _resolve_path(
            results_dir,
            extras.get("per_prompt_metrics_dir"),
            str(extras.get("per_prompt_metrics_subdir", "ranking_only")),
        )
        if verbose:
            print("=" * 70)
            print("OPCIONAL: EXPORT PER-PROMPT METRICS (CSV)")
            print("-" * 70)
        report["per_prompt_metrics_paths"] = save_per_prompt_metrics_csvs(
            resultados_por_modelo,
            dest,
            verbose=verbose,
        )
        report["per_prompt_metrics_dir"] = str(dest)
        if verbose:
            print()

    if extras.get("run_tfidf_baseline"):
        tfidf_out = _resolve_path(
            results_dir,
            extras.get("tfidf_baseline_dir"),
            str(extras.get("tfidf_baseline_subdir", "tfidf_baseline")),
        )
        if verbose:
            print("=" * 70)
            print("OPCIONAL: BASELINE TF-IDF")
            print("-" * 70)
        run_tfidf_baseline_integrated(
            df, text_col, config, tfidf_out, verbose=verbose
        )
        report["tfidf_output_dir"] = str(tfidf_out)
        if verbose:
            print()

    if extras.get("run_visualizations"):
        ranking_dir = extras.get("visualizations_ranking_dir")
        if not ranking_dir:
            ranking_dir = str(
                _resolve_path(
                    results_dir,
                    extras.get("per_prompt_metrics_dir"),
                    str(extras.get("per_prompt_metrics_subdir", "ranking_only")),
                )
            )
        tfidf_dir = extras.get("visualizations_tfidf_dir")
        if not tfidf_dir:
            tfidf_dir = str(
                _resolve_path(
                    results_dir,
                    extras.get("tfidf_baseline_dir"),
                    str(extras.get("tfidf_baseline_subdir", "tfidf_baseline")),
                )
            )
        viz_out = _resolve_path(
            results_dir,
            extras.get("visualizations_output_dir"),
            str(extras.get("visualizations_subdir", "visualizations")),
        )
        n_boot = int(extras.get("n_bootstrap", 10000))
        if verbose:
            print("=" * 70)
            print("OPCIONAL: VISUALIZACOES (BOXPLOT / BOOTSTRAP)")
            print("-" * 70)
        try:
            run_visualization_suite_integrated(
                ranking_dir=ranking_dir,
                tfidf_dir=tfidf_dir,
                output_dir=str(viz_out),
                n_bootstrap=n_boot,
                verbose=verbose,
            )
        except FileNotFoundError as e:
            report["skipped_visualizations"] = str(e)
            if verbose:
                print(f"   AVISO: visualizacoes ignoradas: {e}")
        report["visualizations_output_dir"] = str(viz_out)
        if verbose:
            print()

    return report
