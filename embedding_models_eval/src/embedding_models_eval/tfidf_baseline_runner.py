"""
Baseline lexical: ranking por ancora usando TF-IDF cosine similarity.

Protocolo identico ao run_ranking_only.py, substituindo embeddings neurais
por representacao esparsa TF-IDF. Serve como piso (floor) para isolar
quanto da recuperacao de ranking vem de sobreposicao lexical vs semantica.

Fluxo:
1. Carrega dados reais (saida_final.json)
2. Calcula TF-IDF sobre os textos
3. Para cada prompt: ancora = rank 1, ranqueia candidatos por cosine(tfidf)
4. Calcula metricas IR @k e metricas de votos @k
5. Salva resultados no mesmo formato que run_ranking_only
"""

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(config_path: str = "configs/default.yaml"):
    """
    Carrega dados reais usando o loader do pipeline.

    Args:
        config_path: Caminho para configuracao YAML.

    Returns:
        Tupla (df, text_col, config) ou (None, None, None) em caso de erro.
    """
    print("=" * 70)
    print("1. CARREGANDO DADOS")
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

        print(f"   Total de linhas: {len(df):,}")

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


def build_tfidf_anchor_ranking(
    df: pd.DataFrame,
    text_col: str = "extracted_idea_250",
    group_cols: Optional[List[str]] = None,
    rank_col: str = "rank_in_prompt",
    anchor_rank: int = 1,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Ranking por ancora usando TF-IDF cosine similarity.

    Protocolo identico a build_anchor_ranking (ranking/anchor.py),
    substituindo embeddings neurais por vetores TF-IDF esparsos.

    Args:
        df: DataFrame com dados.
        text_col: Coluna de texto.
        group_cols: Colunas para criar prompt_id.
        rank_col: Coluna com ranking gold.
        anchor_rank: Valor de rank da ancora (default: 1).
        show_progress: Exibir progresso.

    Returns:
        DataFrame com colunas: prompt_id, doc_id, score_to_anchor,
        rank_pred, rank_gold.
    """
    out = df.copy()

    if group_cols is None:
        group_cols = ["contest_number", "context_prompt_url"]

    required_cols = group_cols + [rank_col, text_col]
    missing_cols = [col for col in required_cols if col not in out.columns]
    if missing_cols:
        raise ValueError(f"Colunas faltando: {missing_cols}")

    # prompt_id e doc_id seguem o mesmo padrao de ranking/anchor.py
    out["prompt_id"] = (
        out[group_cols[0]].astype(str).str.strip()
        + "::"
        + out[group_cols[1]].astype(str).str.strip()
    )

    if "story_url" in out.columns:
        out["doc_id"] = out["story_url"].fillna("").astype(str).str.strip()
    else:
        out["doc_id"] = out["prompt_id"] + "::idx=" + out.index.astype(str)

    empty_mask = out["doc_id"] == ""
    out.loc[empty_mask, "doc_id"] = (
        out.loc[empty_mask, "prompt_id"] + "::idx=" + out.loc[empty_mask].index.astype(str)
    )

    out[text_col] = out[text_col].fillna("").astype(str)

    # TF-IDF sobre todos os textos
    if show_progress:
        print(f"   Calculando TF-IDF para {len(out)} textos...")

    vectorizer = TfidfVectorizer(
        max_features=10000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    tfidf_matrix = vectorizer.fit_transform(out[text_col].tolist())

    if show_progress:
        vocab_size = len(vectorizer.vocabulary_)
        print(f"   TF-IDF: {tfidf_matrix.shape[0]} docs x {vocab_size} termos")

    # Similaridade com ancora por prompt
    out["score_to_anchor"] = np.nan

    prompts = out["prompt_id"].unique()
    for prompt_id in prompts:
        group = out[out["prompt_id"] == prompt_id]

        anchor_mask = group[rank_col] == anchor_rank
        anchor_indices = group[anchor_mask].index.tolist()

        if not anchor_indices:
            continue

        anchor_idx = anchor_indices[0]
        anchor_pos = out.index.get_loc(anchor_idx)
        anchor_vec = tfidf_matrix[anchor_pos]

        group_indices = group.index.tolist()
        group_positions = [out.index.get_loc(idx) for idx in group_indices]
        group_vecs = tfidf_matrix[group_positions]

        similarities = cosine_similarity(group_vecs, anchor_vec).flatten()
        out.loc[group_indices, "score_to_anchor"] = similarities

    # Ranking (mesmo protocolo de ranking/anchor.py)
    out["rank_pred"] = np.nan
    out["rank_gold"] = np.nan

    cand_mask = out[rank_col] > anchor_rank

    if not cand_mask.any():
        return out

    # Gold rank entre candidatos
    cand_gold = out.loc[cand_mask].sort_values(
        ["prompt_id", rank_col],
        ascending=[True, True],
        kind="mergesort",
    )
    out.loc[cand_gold.index, "rank_gold"] = (
        cand_gold.groupby("prompt_id").cumcount() + 1
    )

    # Predicted rank: similaridade DESC, tie-break por rank_col ASC
    cand_pred = out.loc[cand_mask].sort_values(
        ["prompt_id", "score_to_anchor", rank_col],
        ascending=[True, False, True],
        kind="mergesort",
    )
    out.loc[cand_pred.index, "rank_pred"] = (
        cand_pred.groupby("prompt_id").cumcount() + 1
    )

    if show_progress:
        n_cand = cand_mask.sum()
        n_prompts = out["prompt_id"].nunique()
        print(f"   Ranking: {n_cand} candidatos em {n_prompts} prompts")

    return out


def compute_ir_metrics(df_scored, k_values=None, verbose=True):
    """Calcula metricas IR @k (reutiliza metrics.IRMetrics existente)."""
    from embedding_models_eval.metrics import IRMetrics

    if k_values is None:
        k_values = [1, 3, 5, 10]

    df_cand = df_scored[df_scored["rank_pred"].notna()].copy()

    if df_cand.empty:
        if verbose:
            print("   Nenhum candidato para metricas IR")
        return {"per_prompt": pd.DataFrame(), "macro": {}}

    ir_metrics = IRMetrics(k_values=k_values)
    return ir_metrics.compute(
        df_cand,
        rank_pred_col="rank_pred",
        rank_gold_col="rank_gold",
        prompt_id_col="prompt_id",
        doc_id_col="doc_id",
    )


def compute_votes_metrics(df_scored, k_values=None, votes_col="likes", verbose=True):
    """Calcula metricas de votos @k (reutiliza metrics.VotesMetrics existente)."""
    from embedding_models_eval.metrics import VotesMetrics

    if k_values is None:
        k_values = [1, 3, 5, 10]

    df_cand = df_scored[df_scored["rank_pred"].notna()].copy()

    if df_cand.empty or votes_col not in df_cand.columns:
        if verbose:
            reason = (
                f"coluna '{votes_col}' ausente"
                if votes_col not in df_scored.columns
                else "nenhum candidato"
            )
            print(f"   Sem metricas de votos ({reason})")
        return {"per_prompt": pd.DataFrame(), "macro": {}}

    votes_metrics = VotesMetrics(k_values=k_values, votes_col=votes_col)
    return votes_metrics.compute(
        df_cand,
        rank_pred_col="rank_pred",
        rank_gold_col="rank_gold",
        prompt_id_col="prompt_id",
    )


def save_summary(df_scored, ir_result, votes_result, output_dir, k_values=None):
    """
    Salva resumo no mesmo formato de run_ranking_only (ranking_summary.csv).

    Uma unica linha (modelo = tfidf).
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    s = df_scored["score_to_anchor"].dropna()
    row = {
        "modelo": "tfidf",
        "n_linhas": len(df_scored),
        "n_prompts": df_scored["prompt_id"].nunique(),
        "score_to_anchor_min": s.min() if len(s) else None,
        "score_to_anchor_mean": s.mean() if len(s) else None,
        "score_to_anchor_max": s.max() if len(s) else None,
    }

    macro_ir = ir_result.get("macro", {})
    for k in k_values:
        row[f"MAP@{k}"] = macro_ir.get(f"MAP@{k}", None)
        for m in ["P", "R", "F1"]:
            row[f"{m}@{k}_mean"] = macro_ir.get(f"{m}@{k}_mean", None)
            row[f"{m}@{k}_std"] = macro_ir.get(f"{m}@{k}_std", None)

    macro_v = votes_result.get("macro", {})
    for k in k_values:
        row[f"mean_votes@{k}_pred"] = macro_v.get(f"mean_votes@{k}_pred", None)
        row[f"mean_votes@{k}_gold"] = macro_v.get(f"mean_votes@{k}_gold", None)
        row[f"norm_mean_votes@{k}"] = macro_v.get(f"norm_mean_votes@{k}", None)

    summary = pd.DataFrame([row])
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "ranking_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"   Resumo salvo: {csv_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Baseline lexical: ranking por ancora com TF-IDF cosine similarity."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Caminho para configuracao YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tfidf_baseline",
        help="Diretorio de saida",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-verbose",
        action="store_false",
        dest="verbose",
    )
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("BASELINE LEXICAL: TF-IDF ANCHOR RANKING")
    print("=" * 70)
    print()

    df, text_col, config = load_data(args.config)
    if df is None:
        return 1

    ranking_config = config.get("ranking", {})
    k_values = ranking_config.get("k_values", [1, 3, 5, 10])
    votes_col = ranking_config.get("votes_col", "likes")

    # -- Ranking TF-IDF --
    print("=" * 70)
    print("2. RANKING POR ANCORA (TF-IDF)")
    print("=" * 70)
    print()

    df_scored = build_tfidf_anchor_ranking(
        df,
        text_col=text_col,
        group_cols=ranking_config.get("group_cols", ["contest_number", "context_prompt_url"]),
        rank_col=ranking_config.get("rank_col", "rank_in_prompt"),
        anchor_rank=ranking_config.get("anchor_rank", 1),
        show_progress=args.verbose,
    )

    # Salva df_scored em parquet
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "tfidf_scored.parquet"
    df_scored.to_parquet(parquet_path, index=False)
    if args.verbose:
        print(f"   Parquet salvo: {parquet_path}")
    print()

    # -- Metricas IR --
    print("=" * 70)
    print("3. METRICAS IR @k + VOTOS @k")
    print("=" * 70)
    print()
    print(f"   k = {k_values}")
    print()

    ir_result = compute_ir_metrics(df_scored, k_values=k_values, verbose=args.verbose)

    macro_ir = ir_result.get("macro", {})
    if macro_ir and args.verbose:
        parts = []
        for m in ["P@5", "R@5", "F1@5"]:
            mean_val = macro_ir.get(f"{m}_mean", 0.0)
            std_val = macro_ir.get(f"{m}_std", 0.0)
            parts.append(f"{m}={mean_val:.4f}+/-{std_val:.4f}")
        map5 = macro_ir.get("MAP@5", 0.0)
        parts.append(f"MAP@5={map5:.4f}")
        print(f"   [tfidf] {', '.join(parts)}")

    # -- Metricas de votos --
    votes_result = compute_votes_metrics(
        df_scored, k_values=k_values, votes_col=votes_col, verbose=args.verbose,
    )

    macro_v = votes_result.get("macro", {})
    if macro_v and args.verbose:
        pred_val = macro_v.get("mean_votes@5_pred", 0.0)
        gold_val = macro_v.get("mean_votes@5_gold", 0.0)
        norm_val = macro_v.get("norm_mean_votes@5", 0.0)
        print(f"   [tfidf] votes_pred@5={pred_val:.2f}, votes_gold@5={gold_val:.2f}, norm@5={norm_val:.4f}")

    # Salva metricas per_prompt
    per_prompt_ir = ir_result.get("per_prompt")
    per_prompt_votes = votes_result.get("per_prompt")

    if per_prompt_ir is not None and not per_prompt_ir.empty:
        if per_prompt_votes is not None and not per_prompt_votes.empty:
            per_prompt_merged = per_prompt_ir.merge(
                per_prompt_votes, on="prompt_id", how="left",
            )
        else:
            per_prompt_merged = per_prompt_ir

        metrics_path = out_dir / "tfidf_metrics_per_prompt.csv"
        per_prompt_merged.to_csv(metrics_path, index=False)
        if args.verbose:
            print(f"   Metricas por prompt: {metrics_path}")

    print()

    # -- Resumo --
    print("=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)
    print()

    save_summary(df_scored, ir_result, votes_result, args.output_dir, k_values=k_values)

    print()
    print("=" * 70)
    print("CONCLUIDO")
    print("=" * 70)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
