#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa PACKING: Consolida JSON de vibes em bullets.

Este modulo pega o JSON gerado no CRITIQUE e consolida em um formato
de bullets (texto simples) para ser incorporado na diretiva.

Requer: OPENAI_API_KEY ou OPENROUTER_API_KEY no ambiente.
"""

from __future__ import annotations

import re
from typing import List, Dict, Optional

from bleu_minimal_deepseek import call_deepseek


# Template do prompt PACKING
PACKING_PROMPT_TEMPLATE = """You are creating writing directives based on the analysis below.

Original directive:
"- Center your story around two characters who like each other but don't get a happily ever after."

Analysis of patterns (DO = reinforce, DON'T = avoid):
------
{json_critique}
------

Create a bulleted list of writing directives:
1. First bullet: ALWAYS include the original directive exactly as shown above
2. Next bullets (up to 7 total): Convert each DO/DON'T pattern into a clear, actionable directive
   - For "do" patterns: "- DO: [specific action]"
   - For "don't" patterns: "- DON'T: [specific action to avoid]"
3. Be SPECIFIC and ACTIONABLE (avoid vague language)
4. Do NOT mention specific story examples or character names
5. Focus on craft elements: tone, structure, character development, pacing, etc.

Output ONLY the bulleted list, nothing else:"""


def packing_step(
    critique_json: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.5,
    max_tokens: int = 1000,
    api_key_override: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    """
    Etapa PACKING: Consolida JSON em bullets.
    
    Args:
        critique_json: Lista de dicionarios com vibe_pattern e vibe_description
        model: Modelo LLM a usar (default: gpt-4o-mini)
        temperature: Temperatura para geracao (default: 0.5)
        max_tokens: Maximo de tokens (default: 1000)
        api_key_override: API key alternativa (opcional)
        reasoning_effort: Reasoning effort para modelos que suportam (opcional)
    
    Returns:
        String com bullets formatados (DO/DON'T)
    
    Example:
        >>> critique = [{"vibe_pattern": "do", "vibe_description": "Use metaphors"}]
        >>> result = packing_step(critique)
        >>> print(result)
        "- DO: Use metaphors to convey emotions."
    """
    import json
    
    # Converter JSON para string
    json_str = json.dumps(critique_json, indent=2, ensure_ascii=False)
    
    # Montar prompt
    prompt = PACKING_PROMPT_TEMPLATE.format(json_critique=json_str)
    
    # Chamar LLM
    print(f"[PACKING] Chamando modelo {model}...")
    response = call_deepseek(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key_override=api_key_override,
        reasoning_effort=reasoning_effort,
        exclude_reasoning=True,  # Para GPT-5: pegar apenas output final
    )
    
    # Limpar e formatar resposta
    bullets = _clean_bullet_response(response)
    
    print(f"[PACKING] Bullets consolidados com sucesso ({len(bullets.splitlines())} linhas)")
    return bullets


def _clean_bullet_response(response: str) -> str:
    """
    Limpa e formata a resposta do packing.
    
    Remove markdown, espacos extras, e garante formato consistente.
    
    Args:
        response: Resposta bruta da LLM
    
    Returns:
        String limpa com bullets
    """
    response = (response or "").strip()
    
    # Remover code blocks markdown
    response = re.sub(r"```(?:markdown|text)?\s*", "", response)
    response = re.sub(r"```", "", response)
    
    # Dividir em linhas
    lines = response.split("\n")
    
    # Filtrar linhas vazias e processar bullets
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Garantir que comeca com "-"
        if not line.startswith("-"):
            line = f"- {line}"
        
        cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)


def _validate_bullets(bullets: str) -> bool:
    """
    Valida se os bullets estao no formato esperado.
    
    Args:
        bullets: String com bullets
    
    Returns:
        True se valido, False caso contrario
    """
    if not bullets or not bullets.strip():
        return False
    
    lines = bullets.strip().split("\n")
    
    # Pelo menos uma linha
    if len(lines) < 1:
        return False
    
    # Cada linha deve comecar com "-" e conter DO ou DON'T
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if not line.startswith("-"):
            return False
        
        # Verificar se contem DO: ou DON'T:
        if "DO:" not in line.upper() and "DON'T:" not in line.upper():
            return False
    
    return True


# Exemplo de uso (para testes)
if __name__ == "__main__":
    import json
    
    # Dados de teste (JSON do CRITIQUE)
    CRITIQUE_JSON = [
        {
            "vibe_pattern": "don't",
            "vibe_description": "The LLM tends to use whimsical, metaphor-heavy titles that feel slightly fantastical or conceptual."
        },
        {
            "vibe_pattern": "do",
            "vibe_description": "The LLM creates scenarios with subtle, everyday encounters that hint at deeper connections."
        },
        {
            "vibe_pattern": "don't",
            "vibe_description": "The LLM often employs playful or poetic language in the premise descriptions."
        }
    ]
    
    print("=== TESTE: refinement_packing.py ===\n")
    print("Input JSON:")
    print(json.dumps(CRITIQUE_JSON, indent=2, ensure_ascii=False))
    print()
    
    try:
        result = packing_step(
            critique_json=CRITIQUE_JSON,
            model="gpt-4o-mini",
            temperature=0.5
        )
        
        print("\n=== RESULTADO ===")
        print(result)
        
        print("\n=== VALIDACAO ===")
        is_valid = _validate_bullets(result)
        print(f"Bullets validos: {is_valid}")
        
    except Exception as e:
        print(f"\n=== ERRO ===")
        print(f"Tipo: {type(e).__name__}")
        print(f"Mensagem: {e}")
        import traceback
        traceback.print_exc()

