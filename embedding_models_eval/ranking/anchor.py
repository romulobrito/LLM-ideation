"""
Metodo de ranking baseado em ancora (anchor-based ranking).

Para cada prompt:
  - Usa top-1 (rank_in_prompt == 1) como ancora
  - Calcula similaridade semantica de cada candidato com a ancora
  - Gera ranking previsto baseado em similaridade
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from embeddings.base import EmbeddingProvider


def build_anchor_ranking(
    df: pd.DataFrame,
    provider: EmbeddingProvider,
    text_col: str = "extracted_idea_250",
    group_cols: List[str] = None,
    rank_col: str = "rank_in_prompt",
    anchor_rank: int = 1,
    doc_id_col: Optional[str] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Constrói ranking baseado em similaridade com ancora.
    
    Para cada prompt:
      - Identifica ancora (rank_in_prompt == 1)
      - Calcula embeddings de todas as historias
      - Calcula similaridade (cosine) de cada candidato com ancora
      - Ordena candidatos por similaridade → ranking previsto
    
    Args:
        df: DataFrame com dados (deve ter rank_in_prompt)
        provider: Provider de embeddings
        text_col: Nome da coluna de texto para embedar
        group_cols: Colunas para agrupamento (default: ["contest_number", "context_prompt_url"])
        rank_col: Nome da coluna com ranking gold (default: "rank_in_prompt")
        anchor_rank: Rank da ancora (default: 1)
        doc_id_col: Nome da coluna para doc_id (default: usa story_url ou cria sintetico)
        show_progress: Mostrar barra de progresso
        
    Returns:
        DataFrame com colunas adicionais:
        - prompt_id: ID do prompt (combinacao de group_cols)
        - doc_id: ID do documento
        - score_to_anchor: Similaridade com ancora (cosine similarity)
        - rank_pred: Ranking previsto (exclui ancora, comeca em 1)
        - rank_gold: Ranking gold (exclui ancora, comeca em 1)
    """
    out = df.copy()
    
    if group_cols is None:
        group_cols = ["contest_number", "context_prompt_url"]
    
    # Valida colunas necessarias
    required_cols = group_cols + [rank_col, text_col]
    missing_cols = [col for col in required_cols if col not in out.columns]
    if missing_cols:
        raise ValueError(f"Colunas faltando: {missing_cols}")
    
    # Cria prompt_id (combinacao de group_cols)
    out["prompt_id"] = (
        out[group_cols[0]].astype(str).str.strip()
        + "::"
        + out[group_cols[1]].astype(str).str.strip()
    )
    
    # Cria doc_id (prefer story_url, fallback para ID sintetico)
    if doc_id_col and doc_id_col in out.columns:
        out["doc_id"] = out[doc_id_col].fillna("").astype(str).str.strip()
    elif "story_url" in out.columns:
        out["doc_id"] = out["story_url"].fillna("").astype(str).str.strip()
    else:
        out["doc_id"] = out["prompt_id"] + "::idx=" + out.index.astype(str)
    
    # Garante que doc_id nao seja vazio
    empty_mask = out["doc_id"] == ""
    out.loc[empty_mask, "doc_id"] = out.loc[empty_mask, "prompt_id"] + "::idx=" + out.loc[empty_mask].index.astype(str)
    
    # Prepara texto para embedding
    out[text_col] = out[text_col].fillna("").astype(str)
    
    # Gera embeddings de todas as historias
    if show_progress:
        print(f"Gerando embeddings para {len(out)} textos...")
    
    texts = out[text_col].tolist()
    embeddings = provider.embed(texts)
    
    if show_progress:
        print(f"Embeddings gerados: shape {embeddings.shape}")
    
    # Calcula similaridade com ancora por prompt
    out["score_to_anchor"] = np.nan
    
    prompts = out["prompt_id"].unique()
    if show_progress:
        prompts_iter = tqdm(prompts, desc="Calculando similaridades")
    else:
        prompts_iter = prompts
    
    for prompt_id in prompts_iter:
        group = out[out["prompt_id"] == prompt_id]
        
        # Encontra ancora (rank_in_prompt == anchor_rank)
        anchor_mask = group[rank_col] == anchor_rank
        anchor_indices = group[anchor_mask].index.tolist()
        
        if not anchor_indices:
            continue
        
        anchor_idx = anchor_indices[0]
        anchor_emb = embeddings[group.index.get_loc(anchor_idx)]
        
        # Calcula similaridade de todos com ancora
        group_indices = group.index.tolist()
        group_embs = embeddings[[group.index.get_loc(idx) for idx in group_indices]]
        
        # Cosine similarity (dot product de embeddings normalizados)
        similarities = group_embs @ anchor_emb
        
        out.loc[group_indices, "score_to_anchor"] = similarities
    
    # Prepara ranks (excluindo ancora)
    out["rank_pred"] = np.nan
    out["rank_gold"] = np.nan
    
    # Filtra candidatos (exclui ancora)
    cand_mask = out[rank_col] > anchor_rank
    
    if not cand_mask.any():
        # Nenhum candidato, retorna apenas com score_to_anchor
        return out
    
    # Gold rank entre candidatos (ranking original excluindo ancora)
    cand_gold = out.loc[cand_mask].sort_values(
        ["prompt_id", rank_col],
        ascending=[True, True],
        kind="mergesort",
    )
    out.loc[cand_gold.index, "rank_gold"] = (
        cand_gold.groupby("prompt_id").cumcount() + 1
    )
    
    # Predicted rank entre candidatos (ordenado por similaridade descendente)
    # Tie-break determinístico: se similaridade igual, usa rank_in_prompt original
    cand_pred = out.loc[cand_mask].sort_values(
        ["prompt_id", "score_to_anchor", rank_col],
        ascending=[True, False, True],  # Similaridade DESC, rank ASC
        kind="mergesort",
    )
    out.loc[cand_pred.index, "rank_pred"] = (
        cand_pred.groupby("prompt_id").cumcount() + 1
    )
    
    return out
