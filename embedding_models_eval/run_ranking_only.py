"""
Executa o pipeline ate ranking por ancora (run_ranking_only), sem metricas @k.

Este script:
1. Carrega dados reais (saida_final.json)
2. Para cada modelo: gera embeddings e calcula ranking por similaridade com ancora
3. Salva df_scored (prompt_id, doc_id, score_to_anchor, rank_pred, rank_gold) em Parquet
4. Para por ai (sem metricas IR/votos, sem tabela macro)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def load_real_data(config_path="configs/default.yaml"):
    """Carrega dados reais usando o loader do pipeline (mesmo padrao de run_embeddings_only)."""
    print("=" * 70)
    print("1. CARREGANDO DADOS REAIS")
    print("=" * 70)
    print()

    try:
        from pipeline.config_loader import load_config
        from data import get_loader

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
    from embeddings import get_provider
    from ranking import build_anchor_ranking

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


def save_summary(resultados, output_dir):
    """Salva resumo por modelo (estatisticas de score_to_anchor e contagens)."""
    rows = []
    for model_name, df_scored in resultados.items():
        if df_scored is None or df_scored.empty:
            continue
        s = df_scored["score_to_anchor"].dropna()
        rows.append({
            "modelo": model_name,
            "n_linhas": len(df_scored),
            "n_prompts": df_scored["prompt_id"].nunique(),
            "score_to_anchor_min": s.min() if len(s) else None,
            "score_to_anchor_mean": s.mean() if len(s) else None,
            "score_to_anchor_max": s.max() if len(s) else None,
        })
    if not rows:
        return
    summary = pd.DataFrame(rows)
    path = Path(output_dir) / "ranking_summary.csv"
    summary.to_csv(path, index=False)
    print(f"   Resumo salvo: {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Executa pipeline ate ranking por ancora (run_ranking_only), sem metricas."
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
    print("PIPELINE ATE RANKING POR ANCORA")
    print("=" * 70)
    print()
    print("Fluxo: carrega dados -> por modelo: embeddings + build_anchor_ranking")
    print("Saida: Parquet por modelo (prompt_id, doc_id, score_to_anchor, rank_pred, rank_gold)")
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

    n_ok = sum(1 for v in resultados.values() if v is not None)
    print("=" * 70)
    print("RESUMO")
    print("=" * 70)
    print()
    print(f"Modelos processados: {n_ok}/{len(models_config)}")
    print(f"Saida: {args.output_dir}")
    print()
    save_summary(resultados, args.output_dir)
    print()
    print("=" * 70)
    print("CONCLUIDO")
    print("=" * 70)
    print()

    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
