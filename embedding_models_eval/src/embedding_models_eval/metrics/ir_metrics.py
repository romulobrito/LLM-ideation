"""
Metricas classicas de Information Retrieval usando ranx.

Calcula P@k, R@k, F1@k, AP@k por prompt individual,
e agrega em media (macro) e desvio padrao.
"""

from typing import Dict, List
import numpy as np
import pandas as pd
import ranx
from ranx import Qrels, Run

from .base import Metric, register_metric


class IRMetrics(Metric):
    """
    Metricas classicas de IR: P@k, R@k, F1@k, AP@k, MAP@k.

    Calcula por prompt individual usando ranx, depois agrega
    media e desvio padrao sobre todos os prompts.
    """

    def __init__(self, k_values: List[int] = None):
        if k_values is None:
            k_values = [1, 3, 5, 10]
        super().__init__("ir_metrics", k_values)

    def _compute_single_prompt(
        self,
        qrels_single: Dict[str, int],
        run_single: Dict[str, float],
        prompt_id: str,
        k: int,
    ) -> Dict:
        """
        Calcula metricas IR para um unico prompt e um valor de k.

        Args:
            qrels_single: {doc_id: relevancia} para este prompt
            run_single: {doc_id: score} para este prompt
            prompt_id: ID do prompt
            k: Valor de corte

        Returns:
            Dict com prompt_id, P@k, R@k, F1@k, AP@k
        """
        qrels_obj = Qrels({prompt_id: qrels_single})
        run_obj = Run({prompt_id: run_single})

        precision_val = float(ranx.evaluate(qrels_obj, run_obj, f"precision@{k}"))
        recall_val = float(ranx.evaluate(qrels_obj, run_obj, f"recall@{k}"))
        f1_val = float(ranx.evaluate(qrels_obj, run_obj, f"f1@{k}"))
        ap_val = float(ranx.evaluate(qrels_obj, run_obj, f"map@{k}"))

        return {
            "prompt_id": prompt_id,
            f"P@{k}": precision_val,
            f"R@{k}": recall_val,
            f"F1@{k}": f1_val,
            f"AP@{k}": ap_val,
        }

    def compute(
        self,
        df: pd.DataFrame,
        rank_pred_col: str = "rank_pred",
        rank_gold_col: str = "rank_gold",
        prompt_id_col: str = "prompt_id",
        doc_id_col: str = "doc_id",
    ) -> Dict:
        """
        Calcula metricas IR por prompt individual e agrega.

        Args:
            df: DataFrame com rankings
            rank_pred_col: Coluna com ranking previsto
            rank_gold_col: Coluna com ranking gold
            prompt_id_col: Coluna com ID do prompt
            doc_id_col: Coluna com ID do documento

        Returns:
            {
                "per_prompt": DataFrame com metricas por prompt (N linhas),
                "macro": Dict com media e std de cada metrica
            }
        """
        # Prepara Qrels e Run por prompt
        qrels_dict = {}
        run_dict = {}

        for prompt_id, group in df.groupby(prompt_id_col):
            k_max = max(self.k_values)
            relevant_docs = group[
                group[rank_gold_col] <= k_max
            ][doc_id_col].tolist()

            if not relevant_docs:
                continue

            qrels_dict[prompt_id] = {
                doc_id: 1 for doc_id in relevant_docs
            }

            run_dict[prompt_id] = {
                doc_id: 1.0 / (rank + 1)
                for doc_id, rank in zip(
                    group[doc_id_col],
                    group[rank_pred_col],
                )
            }

        if not qrels_dict:
            return {"per_prompt": pd.DataFrame(), "macro": {}}

        # Calcula metricas por prompt individual para cada k
        all_rows = {pid: {"prompt_id": pid} for pid in qrels_dict}

        for k in self.k_values:
            for prompt_id in qrels_dict:
                result = self._compute_single_prompt(
                    qrels_dict[prompt_id],
                    run_dict[prompt_id],
                    prompt_id,
                    k,
                )
                all_rows[prompt_id][f"P@{k}"] = result[f"P@{k}"]
                all_rows[prompt_id][f"R@{k}"] = result[f"R@{k}"]
                all_rows[prompt_id][f"F1@{k}"] = result[f"F1@{k}"]
                all_rows[prompt_id][f"AP@{k}"] = result[f"AP@{k}"]

        per_prompt_df = pd.DataFrame(list(all_rows.values()))

        # Agrega: media e desvio padrao sobre prompts
        macro_results = {}
        for k in self.k_values:
            for metric_name in [f"P@{k}", f"R@{k}", f"F1@{k}", f"AP@{k}"]:
                vals = per_prompt_df[metric_name].values
                macro_results[f"{metric_name}_mean"] = float(np.mean(vals))
                macro_results[f"{metric_name}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            # MAP@k = media de AP@k (conveniente)
            macro_results[f"MAP@{k}"] = macro_results[f"AP@{k}_mean"]

        return {
            "per_prompt": per_prompt_df,
            "macro": macro_results,
        }


# Registra automaticamente
register_metric("ir", IRMetrics)
register_metric("ir_metrics", IRMetrics)
