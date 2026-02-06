"""
Metricas classicas de Information Retrieval usando ranx.

Demonstra como implementar uma metrica seguindo a interface base.
"""

from typing import Dict, List
import pandas as pd
import ranx
from ranx import Qrels, Run

from .base import Metric, register_metric


class IRMetrics(Metric):
    """
    Metricas classicas de IR: P@k, R@k, F1@k, AP@k, MAP@k.
    
    Usa a biblioteca ranx para calculo eficiente.
    """
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        super().__init__("ir_metrics", k_values)
    
    def compute(
        self,
        df: pd.DataFrame,
        rank_pred_col: str = "rank_pred",
        rank_gold_col: str = "rank_gold",
        prompt_id_col: str = "prompt_id",
        doc_id_col: str = "doc_id",
    ) -> Dict:
        """
        Calcula metricas IR usando ranx.
        
        Args:
            df: DataFrame com rankings
            rank_pred_col: Coluna com ranking previsto
            rank_gold_col: Coluna com ranking gold
            prompt_id_col: Coluna com ID do prompt
            doc_id_col: Coluna com ID do documento
            
        Returns:
            {
                "per_prompt": DataFrame com metricas por prompt,
                "macro": Dict com MAP@k para cada k
            }
        """
        # Prepara Qrels (ground truth) e Run (predicoes) para ranx
        qrels_dict = {}
        run_dict = {}
        
        for prompt_id, group in df.groupby(prompt_id_col):
            # Qrels: documentos relevantes (rank gold <= k_max)
            k_max = max(self.k_values)
            relevant_docs = group[
                group[rank_gold_col] <= k_max
            ][doc_id_col].tolist()
            
            if not relevant_docs:
                continue
            
            qrels_dict[prompt_id] = {
                doc_id: 1 for doc_id in relevant_docs
            }
            
            # Run: ranking previsto
            run_dict[prompt_id] = {
                doc_id: 1.0 / (rank + 1)  # Score inverso do rank
                for doc_id, rank in zip(
                    group[doc_id_col],
                    group[rank_pred_col]
                )
            }
        
        if not qrels_dict:
            return {"per_prompt": pd.DataFrame(), "macro": {}}
        
        qrels = Qrels(qrels_dict)
        run = Run(run_dict)
        
        # Calcula metricas para cada k
        results = {}
        macro_results = {}
        
        for k in self.k_values:
            # Metricas por prompt
            precision = ranx.evaluate(qrels, run, f"precision@{k}")
            recall = ranx.evaluate(qrels, run, f"recall@{k}")
            f1 = ranx.evaluate(qrels, run, f"f1@{k}")
            ap = ranx.evaluate(qrels, run, f"map@{k}")
            
            # ranx.evaluate() retorna dict quando ha multiplos prompts,
            # ou valor escalar quando ha apenas um prompt
            # Normaliza para sempre ser dict
            if not isinstance(precision, dict):
                # Caso unico prompt: converte para dict
                prompt_ids = list(qrels_dict.keys())
                precision = {prompt_ids[0]: float(precision)}
                recall = {prompt_ids[0]: float(recall)}
                f1 = {prompt_ids[0]: float(f1)}
                ap = {prompt_ids[0]: float(ap)}
            
            # Agrega em DataFrame
            per_prompt = pd.DataFrame({
                "prompt_id": list(precision.keys()),
                f"P@{k}": list(precision.values()),
                f"R@{k}": list(recall.values()),
                f"F1@{k}": list(f1.values()),
                f"AP@{k}": list(ap.values()),
            })
            
            results[k] = per_prompt
            
            # Macro (media)
            macro_results[f"MAP@{k}"] = sum(ap.values()) / len(ap) if ap else 0.0
        
        # Combina todos os k em um DataFrame
        all_per_prompt = results[self.k_values[0]].copy()
        for k in self.k_values[1:]:
            all_per_prompt = all_per_prompt.merge(
                results[k],
                on="prompt_id",
                suffixes=("", f"_k{k}")
            )
        
        return {
            "per_prompt": all_per_prompt,
            "macro": macro_results,
        }


# Registra automaticamente
register_metric("ir", IRMetrics)
register_metric("ir_metrics", IRMetrics)
