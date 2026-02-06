"""
Metricas baseadas em votos (likes) no top-k.

Nova metrica customizada que demonstra como adicionar metricas especificas.
"""

from typing import Dict, List
import pandas as pd
import numpy as np

from .base import Metric, register_metric


class VotesMetrics(Metric):
    """
    Metricas baseadas em votos (likes) no top-k do ranking.
    
    Calcula:
    - mean_votes@k (pred): media de likes no top-k previsto
    - max_votes@k (pred): maximo de likes no top-k previsto
    - mean_votes@k (gold): media de likes no top-k gold (topline)
    - max_votes@k (gold): maximo de likes no top-k gold
    - Versoes normalizadas (pred/gold)
    """
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10], votes_col: str = "likes"):
        super().__init__("votes_metrics", k_values)
        self.votes_col = votes_col
    
    def compute(
        self,
        df: pd.DataFrame,
        rank_pred_col: str = "rank_pred",
        rank_gold_col: str = "rank_gold",
        prompt_id_col: str = "prompt_id",
    ) -> Dict:
        """
        Calcula metricas de votos no top-k.
        
        Args:
            df: DataFrame com rankings e votos
            rank_pred_col: Coluna com ranking previsto
            rank_gold_col: Coluna com ranking gold
            prompt_id_col: Coluna com ID do prompt
            
        Returns:
            {
                "per_prompt": DataFrame com metricas por prompt,
                "macro": Dict com metricas agregadas
            }
        """
        if self.votes_col not in df.columns:
            return {
                "per_prompt": pd.DataFrame(),
                "macro": {},
            }
        
        results_per_prompt = []
        macro_results = {}
        
        for k in self.k_values:
            k_results = []
            
            for prompt_id, group in df.groupby(prompt_id_col):
                # Top-k previsto
                topk_pred = group.nsmallest(k, rank_pred_col)
                mean_votes_pred = topk_pred[self.votes_col].mean()
                max_votes_pred = topk_pred[self.votes_col].max()
                
                # Top-k gold
                topk_gold = group.nsmallest(k, rank_gold_col)
                mean_votes_gold = topk_gold[self.votes_col].mean()
                max_votes_gold = topk_gold[self.votes_col].max()
                
                # Normalizacao (trata divisao por zero)
                norm_mean = (
                    mean_votes_pred / mean_votes_gold
                    if mean_votes_gold > 0 else np.nan
                )
                norm_max = (
                    max_votes_pred / max_votes_gold
                    if max_votes_gold > 0 else np.nan
                )
                
                k_results.append({
                    "prompt_id": prompt_id,
                    f"mean_votes@{k}_pred": mean_votes_pred,
                    f"max_votes@{k}_pred": max_votes_pred,
                    f"mean_votes@{k}_gold": mean_votes_gold,
                    f"max_votes@{k}_gold": max_votes_gold,
                    f"norm_mean_votes@{k}": norm_mean,
                    f"norm_max_votes@{k}": norm_max,
                })
            
            k_df = pd.DataFrame(k_results)
            results_per_prompt.append(k_df)
            
            # Macro (media)
            if not k_df.empty:
                macro_results[f"mean_votes@{k}_pred"] = k_df[f"mean_votes@{k}_pred"].mean()
                macro_results[f"max_votes@{k}_pred"] = k_df[f"max_votes@{k}_pred"].mean()
                macro_results[f"mean_votes@{k}_gold"] = k_df[f"mean_votes@{k}_gold"].mean()
                macro_results[f"max_votes@{k}_gold"] = k_df[f"max_votes@{k}_gold"].mean()
                macro_results[f"norm_mean_votes@{k}"] = k_df[f"norm_mean_votes@{k}"].mean()
                macro_results[f"norm_max_votes@{k}"] = k_df[f"norm_max_votes@{k}"].mean()
        
        # Combina todos os k
        all_per_prompt = results_per_prompt[0].copy()
        for k_df in results_per_prompt[1:]:
            all_per_prompt = all_per_prompt.merge(
                k_df,
                on="prompt_id",
                how="outer"
            )
        
        return {
            "per_prompt": all_per_prompt,
            "macro": macro_results,
        }


# Registra automaticamente
register_metric("votes", VotesMetrics)
register_metric("votes_metrics", VotesMetrics)
