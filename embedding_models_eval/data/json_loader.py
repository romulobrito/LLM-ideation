"""
Loader para datasets em formato JSON (saida_final.json).

Implementa o loader baseado no notebook de exemplo.
"""

import json
import re
from typing import Dict, List
import pandas as pd

from .base import DatasetLoader, register_loader


def to_int(x, default=0):
    """Convert a value to int safely (handles None, NaN-like strings, and mixed formats)."""
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return int(x)
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return default
    # Keep only digits and minus sign
    s = re.sub(r"[^\d\-]", "", s)
    if not s or s == "-":
        return default
    try:
        return int(s)
    except Exception:
        return default


def safe_join_tags(tags):
    """Join tag list into a single string; return empty string for None."""
    if isinstance(tags, list):
        return "; ".join(str(t) for t in tags)
    if tags is None:
        return ""
    return str(tags)


def iter_rows(payload):
    """Yield flat rows from the nested JSON structure."""
    contests = payload if isinstance(payload, list) else [payload]

    for contest in contests:
        #  contest context
        contest_number = str(contest.get("Number", "")).strip()
        contest_title = contest.get("Title", "")
        contest_url = contest.get("URL", "")
        prize_value = contest.get("Prize_Value", "")
        ended = contest.get("Ended", "")
        contest_scraped_at = contest.get("Scraped_At", "")

        #  prompt context
        prompts = contest.get("Prompts") or []
        for prompt in prompts:
            prompt_title = prompt.get("Title", "")
            prompt_url = prompt.get("URL", "")
            prompt_posted = prompt.get("Posted", "")
            prompt_texts_count = prompt.get("Texts_Count", "")
            prompt_scraped_at = prompt.get("Scraped_At", "")

            #  historias dentro do prompt
            texts = prompt.get("Texts") or []
            for txt in texts:
                extracted = txt.get("Extracted_idea") or {}

                yield {
                    # Contest contexto
                    "contest_number": contest_number,
                    "contest_title": contest_title,
                    "contest_url": contest_url,
                    "prize_value": prize_value,
                    "ended": ended,
                    "contest_scraped_at": contest_scraped_at,
                    # Prompt contexto
                    "context_prompt_title": prompt_title,
                    "context_prompt_url": prompt_url,
                    "context_prompt_posted": prompt_posted,
                    "context_texts_count": prompt_texts_count,
                    "context_prompt_scraped_at": prompt_scraped_at,
                    # Story/proposal campos
                    "story_title": txt.get("Title", ""),
                    "story_url": txt.get("URL", ""),
                    "story_author": txt.get("Author", ""),
                    "story_posted": txt.get("Posted", ""),
                    "story_award": txt.get("Award", ""),
                    "story_tags": safe_join_tags(txt.get("Tags")),
                    "likes": to_int(txt.get("Likes")),
                    "comments": to_int(txt.get("Comments")),
                    "extracted_idea_50": extracted.get("50", ""),
                    "extracted_idea_150": extracted.get("150", ""),
                    "extracted_idea_250": extracted.get("250", ""),
                    "story_content": txt.get("Content", ""),
                }


class JSONLoader(DatasetLoader):
    """
    Loader para datasets JSON (formato saida_final.json).
    
    Configuracao exemplo:
    {
        "text_col": "extracted_idea_250",  # Coluna de texto padrao
        "truncate_content": 0,  # 0 = sem truncamento
        "group_cols": ["contest_number", "context_prompt_url"],  # Chaves de agrupamento
        "rank_by": "likes",  # "likes" ou "comments" para ranking gold
    }
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.text_col = config.get("text_col", "extracted_idea_250")
        self.truncate_content = config.get("truncate_content", 0)
        self.group_cols = config.get(
            "group_cols",
            ["contest_number", "context_prompt_url"]
        )
        self.rank_by = config.get("rank_by", "likes")  # "likes" ou "comments"
    
    def load(self, path: str) -> pd.DataFrame:
        """
        Carrega dataset JSON e retorna DataFrame padronizado.
        
        Args:
            path: Caminho para arquivo JSON
            
        Returns:
            DataFrame com:
            - Chaves de agrupamento (contest_number, context_prompt_url)
            - rank_in_prompt (ranking gold baseado em likes)
            - Coluna de texto configurável
            - Todas as outras colunas do dataset
        """
        #  Carregar JSON
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        
        #  Aplanar estrutura aninhada
        df = pd.DataFrame(list(iter_rows(payload)))
        if df.empty:
            raise ValueError("No rows produced. Check JSON structure/keys.")
        
        #  Opcionalmente truncar conteudo longo da historia
        if self.truncate_content and self.truncate_content > 0:
            df["story_content"] = df["story_content"].astype(str).str.slice(
                0, int(self.truncate_content)
            )
        
        #  Ordenar historias dentro de cada prompt por engajamento
        #  Usar mergesort para ordenacao estavel
        sort_cols = self.group_cols + [self.rank_by]
        if self.rank_by == "likes":
                # Likes desc, comentarios desc como tie-break
            sort_cols = self.group_cols + ["likes", "comments"]
            ascending = [True, True, False, False]
        else:
            #  Comentarios desc, likes desc como empate
            sort_cols = self.group_cols + ["comments", "likes"]
            ascending = [True, True, False, False]
        
        df = df.sort_values(
            sort_cols,
            ascending=ascending,
            kind="mergesort",
        )
        
        #  Ranking dentro de cada prompt (1 = top story in that prompt)
        df["rank_in_prompt"] = df.groupby(self.group_cols).cumcount() + 1
        
        #  Ordenacao final: contest -> prompt -> rank within prompt
        df = df.sort_values(
            self.group_cols + ["rank_in_prompt"],
            ascending=[True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        
        # Valida schema
        if not self.validate_schema(df):
            raise ValueError(
                "Schema invalido: DataFrame deve ter coluna 'rank_in_prompt'"
            )
        
        return df


# Registra automaticamente
register_loader("json", JSONLoader)


# Funcao de conveniencia para uso rapido
def load_dataset(
    path: str,
    text_col: str = "extracted_idea_250",
    truncate_content: int = 0,
    group_cols: List[str] = None,
    rank_by: str = "likes",
) -> pd.DataFrame:
    """
    Funcao de conveniencia para carregar dataset JSON.
    
    Args:
        path: Caminho para arquivo JSON
        text_col: Nome da coluna de texto (default: "extracted_idea_250")
        truncate_content: Truncar story_content (0 = sem truncamento)
        group_cols: Chaves de agrupamento (default: ["contest_number", "context_prompt_url"])
        rank_by: "likes" ou "comments" para ranking gold (default: "likes")
        
    Returns:
        DataFrame padronizado com rank_in_prompt
        
    Example:
        df = load_dataset("saida_final.json", text_col="extracted_idea_250")
    """
    if group_cols is None:
        group_cols = ["contest_number", "context_prompt_url"]
    
    config = {
        "text_col": text_col,
        "truncate_content": truncate_content,
        "group_cols": group_cols,
        "rank_by": rank_by,
    }
    
    loader = JSONLoader(config)
    return loader.load(path)
