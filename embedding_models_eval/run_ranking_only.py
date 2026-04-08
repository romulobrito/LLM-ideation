"""
Executa o pipeline completo ate metricas @k (ranking por ancora + avaliacao IR + votos).

Este script:
1. Carrega dados reais (saida_final.json)
2. Para cada modelo: gera embeddings e calcula ranking por similaridade com ancora
3. Salva df_scored (prompt_id, doc_id, score_to_anchor, rank_pred, rank_gold) em Parquet
4. Calcula metricas IR @k (MAP@k, P@k, R@k, F1@k) por prompt, com media e desvio padrao
5. Calcula metricas de votos (mean_votes@k, norm_mean_votes@k) por prompt
6. Salva resumo com metricas macro por modelo
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np


def load_real_data(config_path="configs/default.yaml"):
    """Carrega dados reais usando o loader do pipeline (mesmo padrao de run_embeddings_only)."""
    print("=" * 70)
    print("1. CARREGANDO DADOS REAIS")
    print("=" * 70)
    print()

    try:
        from embedding_models_eval.pipeline.config_loader import load_config
        from embedding_models_eval.data import get_loader

        config = load_config(config_path)
        dataset_config = config.get("dataset", {})
        dataset_path = dataset_config.get("path", "saida_final.json")
        text_col = dataset_config.get("text_col", "extracted_idea_250")

        print(f"   Dataset: {dataset_path}")
        print(f"   Coluna de texto: {text_col}")
        print()

        loader = get_loader("json", dataset_config)
        df = loader.load(dataset_path)

        print(f"   Dados carregados!")
        print(f"   Total de linhas: {len(df):,}")
        print()

        if text_col not in df.columns:
            raise ValueError(f"Coluna '{text_col}' nao encontrada no dataset")

        df = df[df[text_col].notna() & (df[text_col].str.strip() != "")]
        print(f"   Textos validos: {len(df):,}")
        print()

        return df, text_col, config

    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def process_single_model_ranking(df, text_col, model_config, ranking_config, save_dir=None, verbose=True):
    """
    Para um modelo: cria provider, chama build_anchor_ranking, salva df_scored.
    Retorna (df_scored, None em caso de erro).
    """
    from embedding_models_eval.embeddings import get_provider
    from embedding_models_eval.ranking import build_anchor_ranking

    model_name = model_config.get("name", "unknown")
    if verbose:
        print(f"   Modelo: {model_name}")
        print(f"   Criando provider e calculando ranking com ancora...")

    try:
        provider = get_provider(model_config["provider"], model_config.get("config", {}))
        df_scored = build_anchor_ranking(
            df,
            provider,
            text_col=text_col,
            group_cols=ranking_config.get("group_cols", ["contest_number", "context_prompt_url"]),
            rank_col=ranking_config.get("rank_col", "rank_in_prompt"),
            anchor_rank=ranking_config.get("anchor_rank", 1),
            show_progress=verbose,
        )

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / f"{model_name}_scored.parquet"
            df_scored.to_parquet(path, index=False)
            if verbose:
                print(f"   Salvo: {path}")

        return df_scored
    except Exception as e:
        if verbose:
            print(f"   Erro: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_ir_metrics(df_scored, k_values=None, verbose=True):
    """
    Calcula metricas IR @k para um DataFrame com ranking.

    Args:
        df_scored: DataFrame com colunas rank_pred, rank_gold, prompt_id, doc_id
        k_values: Lista de valores k para metricas @k (default: [1, 3, 5, 10])
        verbose: Mostrar progresso

    Returns:
        Dict com "per_prompt" (DataFrame) e "macro" (dict com media/std por metrica)
    """
    from embedding_models_eval.metrics import IRMetrics

    if k_values is None:
        k_values = [1, 3, 5, 10]

    # Filtra apenas candidatos (exclui ancora, que tem rank_pred NaN)
    df_cand = df_scored[df_scored["rank_pred"].notna()].copy()

    if df_cand.empty:
        if verbose:
            print("   Nenhum candidato para calcular metricas IR")
        return {"per_prompt": pd.DataFrame(), "macro": {}}

    ir_metrics = IRMetrics(k_values=k_values)
    results = ir_metrics.compute(
        df_cand,
        rank_pred_col="rank_pred",
        rank_gold_col="rank_gold",
        prompt_id_col="prompt_id",
        doc_id_col="doc_id",
    )

    return results


def compute_votes_metrics(df_scored, k_values=None, votes_col="likes", verbose=True):
    """
    Calcula metricas de votos (likes) no top-k previsto vs gold.

    Args:
        df_scored: DataFrame com colunas rank_pred, rank_gold, prompt_id, likes
        k_values: Lista de valores k (default: [1, 3, 5, 10])
        votes_col: Coluna com votos/likes
        verbose: Mostrar progresso

    Returns:
        Dict com "per_prompt" (DataFrame) e "macro" (dict com media/std)
    """
    from embedding_models_eval.metrics import VotesMetrics

    if k_values is None:
        k_values = [1, 3, 5, 10]

    # Filtra apenas candidatos (exclui ancora)
    df_cand = df_scored[df_scored["rank_pred"].notna()].copy()

    if df_cand.empty or votes_col not in df_cand.columns:
        if verbose:
            col_msg = f"coluna '{votes_col}' ausente" if votes_col not in df_scored.columns else "nenhum candidato"
            print(f"   Nenhuma metrica de votos calculada ({col_msg})")
        return {"per_prompt": pd.DataFrame(), "macro": {}}

    votes_metrics = VotesMetrics(k_values=k_values, votes_col=votes_col)
    results = votes_metrics.compute(
        df_cand,
        rank_pred_col="rank_pred",
        rank_gold_col="rank_gold",
        prompt_id_col="prompt_id",
    )

    return results


def save_summary(resultados, ir_por_modelo, votes_por_modelo, output_dir, k_values=None):
    """
    Salva resumo por modelo com:
    - Estatisticas de score_to_anchor
    - Metricas IR @k (media e desvio padrao)
    - Metricas de votos @k (media e desvio padrao)
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    rows = []
    for model_name, df_scored in resultados.items():
        if df_scored is None or df_scored.empty:
            continue
        s = df_scored["score_to_anchor"].dropna()
        row = {
            "modelo": model_name,
            "n_linhas": len(df_scored),
            "n_prompts": df_scored["prompt_id"].nunique(),
            "score_to_anchor_min": s.min() if len(s) else None,
            "score_to_anchor_mean": s.mean() if len(s) else None,
            "score_to_anchor_max": s.max() if len(s) else None,
        }

        # Metricas IR @k (media e desvio padrao)
        if model_name in ir_por_modelo:
            macro = ir_por_modelo[model_name].get("macro", {})
            for k in k_values:
                row[f"MAP@{k}"] = macro.get(f"MAP@{k}", None)
                for m in ["P", "R", "F1"]:
                    row[f"{m}@{k}_mean"] = macro.get(f"{m}@{k}_mean", None)
                    row[f"{m}@{k}_std"] = macro.get(f"{m}@{k}_std", None)

        # Metricas de votos @k
        if model_name in votes_por_modelo:
            macro_v = votes_por_modelo[model_name].get("macro", {})
            for k in k_values:
                row[f"mean_votes@{k}_pred"] = macro_v.get(f"mean_votes@{k}_pred", None)
                row[f"mean_votes@{k}_gold"] = macro_v.get(f"mean_votes@{k}_gold", None)
                row[f"norm_mean_votes@{k}"] = macro_v.get(f"norm_mean_votes@{k}", None)

        rows.append(row)

    if not rows:
        return
    summary = pd.DataFrame(rows)
    path = Path(output_dir) / "ranking_summary.csv"
    summary.to_csv(path, index=False)
    print(f"   Resumo salvo: {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Executa pipeline completo: ranking por ancora + metricas IR @k."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Caminho para arquivo de configuracao YAML",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nome do modelo a processar (se nao especificado, processa todos)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ranking_only",
        help="Diretorio para salvar Parquets e resumo (ex.: results/ranking_only)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Modo verboso (padrao: True)",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_false",
        dest="verbose",
        help="Desativar modo verboso",
    )
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("PIPELINE COMPLETO: RANKING POR ANCORA + METRICAS @k")
    print("=" * 70)
    print()
    print("Fluxo: carrega dados -> por modelo: embeddings + ranking + metricas IR @k")
    print("Saida: Parquet por modelo + resumo com MAP@k")
    print()

    df, text_col, config = load_real_data(args.config)
    if df is None:
        return 1

    ranking_config = config.get("ranking", {})
    models_config = config.get("models", [])

    if args.model:
        models_config = [m for m in models_config if m.get("name") == args.model]
        if not models_config:
            print(f"Modelo '{args.model}' nao encontrado na configuracao")
            return 1

    print("=" * 70)
    print(f"2. PROCESSANDO {len(models_config)} MODELO(S)")
    print("=" * 70)
    print()

    resultados = {}
    for i, model_config in enumerate(models_config, 1):
        model_name = model_config.get("name", f"model_{i}")
        if args.verbose:
            print(f"[{i}/{len(models_config)}] {model_name}")
        df_scored = process_single_model_ranking(
            df, text_col, model_config, ranking_config,
            save_dir=args.output_dir,
            verbose=args.verbose,
        )
        resultados[model_name] = df_scored
        if df_scored is not None and args.verbose:
            print(f"   OK: {len(df_scored):,} linhas")
        print()

    # Calcula metricas IR @k e Votes @k para cada modelo
    k_values = ranking_config.get("k_values", [1, 3, 5, 10])
    votes_col = ranking_config.get("votes_col", "likes")

    print("=" * 70)
    print("3. CALCULANDO METRICAS IR @k + VOTOS @k")
    print("=" * 70)
    print()
    print(f"   Valores de k: {k_values}")
    print(f"   Coluna de votos: {votes_col}")
    print()

    ir_por_modelo = {}
    votes_por_modelo = {}

    for model_name, df_scored in resultados.items():
        if df_scored is None or df_scored.empty:
            continue
        if args.verbose:
            print(f"   [{model_name}] Calculando metricas IR...")

        # -- Metricas IR --
        ir_result = compute_ir_metrics(df_scored, k_values=k_values, verbose=args.verbose)
        ir_por_modelo[model_name] = ir_result

        macro_ir = ir_result.get("macro", {})
        if macro_ir and args.verbose:
            # Exibe media +/- std para as metricas principais @5
            parts = []
            for m in ["P@5", "R@5", "F1@5"]:
                mean_val = macro_ir.get(f"{m}_mean", 0.0)
                std_val = macro_ir.get(f"{m}_std", 0.0)
                parts.append(f"{m}={mean_val:.4f}+/-{std_val:.4f}")
            map5 = macro_ir.get("MAP@5", 0.0)
            parts.append(f"MAP@5={map5:.4f}")
            print(f"   [{model_name}] {', '.join(parts)}")

        # -- Metricas de Votos --
        if args.verbose:
            print(f"   [{model_name}] Calculando metricas de votos...")

        votes_result = compute_votes_metrics(
            df_scored, k_values=k_values, votes_col=votes_col, verbose=args.verbose,
        )
        votes_por_modelo[model_name] = votes_result

        macro_v = votes_result.get("macro", {})
        if macro_v and args.verbose:
            parts_v = []
            for k in [5]:
                pred_val = macro_v.get(f"mean_votes@{k}_pred", 0.0)
                gold_val = macro_v.get(f"mean_votes@{k}_gold", 0.0)
                norm_val = macro_v.get(f"norm_mean_votes@{k}", 0.0)
                parts_v.append(f"votes_pred@{k}={pred_val:.2f}")
                parts_v.append(f"votes_gold@{k}={gold_val:.2f}")
                parts_v.append(f"norm@{k}={norm_val:.4f}")
            print(f"   [{model_name}] {', '.join(parts_v)}")

        # Salva metricas por prompt (IR)
        per_prompt_ir = ir_result.get("per_prompt")
        if per_prompt_ir is not None and not per_prompt_ir.empty:
            # Junta com metricas de votos por prompt se disponivel
            per_prompt_votes = votes_result.get("per_prompt")
            if per_prompt_votes is not None and not per_prompt_votes.empty:
                per_prompt_merged = per_prompt_ir.merge(
                    per_prompt_votes, on="prompt_id", how="left",
                )
            else:
                per_prompt_merged = per_prompt_ir

            metrics_path = Path(args.output_dir) / f"{model_name}_metrics_per_prompt.csv"
            per_prompt_merged.to_csv(metrics_path, index=False)
            if args.verbose:
                print(f"   [{model_name}] Metricas por prompt salvas: {metrics_path}")
        print()

    n_ok = sum(1 for v in resultados.values() if v is not None)
    print("=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)
    print()
    print(f"Modelos processados: {n_ok}/{len(models_config)}")
    print(f"Saida: {args.output_dir}")
    print()
    save_summary(resultados, ir_por_modelo, votes_por_modelo, args.output_dir, k_values=k_values)
    print()
    print("=" * 70)
    print("CONCLUIDO")
    print("=" * 70)
    print()

    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
