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
from experiment_iterativo import embed_texts, cosine_distance


# Template do prompt PACKING
PACKING_PROMPT_TEMPLATE = """You are creating writing directives based on CONTRASTIVE FEEDBACK below.

Original directive:
"{directive}"

Contrastive Feedback (replace/add/keep actions):
------
{json_critique}
------

Create a bulleted list of writing directives:
1. First bullet: ALWAYS include the original directive exactly as shown above

2. For each feedback item, create a directive:
   - "replace" actions: "- REPLACE [from] WITH [to]"
   - "add" actions: "- ADD: [description]"
   - "keep" actions: "- KEEP: [description]"

3. CRITICAL: Be ULTRA-SPECIFIC and CONTRASTIVE
   GOOD: "- REPLACE archetypal unnamed characters WITH named characters who have specific occupations and backstories"
   GOOD: "- ADD sensory details about settings (sounds, smells, textures)"
   BAD: "- Improve character development"
   BAD: "- Don't use gimmicks"

4. Do NOT mention specific story examples or character names from the sets
5. Focus on CRAFT ELEMENTS: character types, setting details, plot structure, emotional tone, pacing

Output ONLY the bulleted list (up to 8 bullets total), nothing else:"""


def packing_step(
    critique_json: List[Dict[str, str]],
    directive: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.5,
    max_tokens: int = 1000,
    api_key_override: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    previous_bullets: Optional[str] = None,
    max_bullets: int = 15,
    embedder = None,  # Embedder para dedup semantico (opcional)
    dedup_threshold: float = 0.90,  # Threshold para dedup (0.90 = 90% similar)
) -> str:
    """
    Etapa PACKING: Consolida feedback contrastivo em bullets acionáveis.
    
    Args:
        critique_json: Lista de dicionarios com feedback contrastivo
                      (action: replace/add/keep, from/to/description)
        directive: Diretiva original do experimento
        model: Modelo LLM a usar (default: gpt-4o-mini)
        temperature: Temperatura para geracao (default: 0.5)
        max_tokens: Maximo de tokens (default: 1000)
        api_key_override: API key alternativa (opcional)
        reasoning_effort: Reasoning effort para modelos que suportam (opcional)
        previous_bullets: Bullets das iteracoes anteriores (para sumarizacao)
        max_bullets: Numero maximo de bullets (default: 15)
    
    Returns:
        String com bullets formatados (REPLACE/ADD/KEEP)
    
    Example:
        >>> critique = [{"action": "replace", "from": "gimmicks", "to": "character depth"}]
        >>> directive = "Center your story around two characters..."
        >>> result = packing_step(critique, directive)
        >>> print(result)
        "- REPLACE gimmicky concepts WITH character-driven narratives"
    """
    import json
    
    # Converter JSON para string
    json_str = json.dumps(critique_json, indent=2, ensure_ascii=False)
    
    # Montar prompt
    prompt = PACKING_PROMPT_TEMPLATE.format(
        directive=directive,
        json_critique=json_str
    )
    
    # Chamar LLM
    print(f"[PACKING] Chamando modelo {model}...")
    response = call_deepseek(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key_override=api_key_override,
        reasoning_effort=reasoning_effort,
        exclude_reasoning=None,  # Auto-detecção: True para DeepSeek, False para GPT-5
    )
    
    # Limpar e formatar resposta
    bullets = _clean_bullet_response(response)
    
    # Combinar com bullets anteriores
    if previous_bullets:
        all_bullets = previous_bullets + "\n" + bullets
    else:
        all_bullets = bullets
    
    # Sumarizar se exceder o limite
    num_bullets = len([l for l in all_bullets.split("\n") if l.strip().startswith("-")])
    if num_bullets > max_bullets:
        print(f"[PACKING] {num_bullets} bullets excede limite de {max_bullets}, sumarizando...")
        bullets = _summarize_bullets(
            bullets=all_bullets,
            directive=directive,
            max_bullets=max_bullets,
            model=model,
            temperature=temperature,
            api_key_override=api_key_override,
            reasoning_effort=reasoning_effort,
        )
    else:
        bullets = all_bullets
    
    # Aplicar dedup semantico se embedder foi fornecido
    if embedder is not None:
        bullets = _deduplicate_bullets(bullets, embedder, threshold=dedup_threshold)
    
    print(f"[PACKING] Bullets consolidados com sucesso ({len(bullets.splitlines())} linhas)")
    return bullets


def _summarize_bullets(
    bullets: str,
    directive: str,
    max_bullets: int,
    model: str,
    temperature: float,
    api_key_override: Optional[str],
    reasoning_effort: Optional[str],
) -> str:
    """
    Sumariza bullets excessivos mantendo os mais importantes.
    
    Args:
        bullets: String com todos os bullets acumulados
        directive: Diretiva original
        max_bullets: Numero maximo de bullets desejado
        model: Modelo LLM
        temperature: Temperatura
        api_key_override: API key alternativa
        reasoning_effort: Reasoning effort
    
    Returns:
        String com bullets sumarizados
    """
    SUMMARIZE_PROMPT = """You have accumulated writing directives that need consolidation.

Original directive:
"{directive}"

Current bullets (TOO MANY):
------
{bullets}
------

Your task: Consolidate these into the {max_bullets} MOST IMPORTANT directives.

Rules:
1. ALWAYS keep the original directive as the first bullet
2. Merge similar/overlapping directives
3. Keep the most specific and actionable ones
4. Prioritize "REPLACE" and "ADD" over "KEEP"
5. Remove redundant items

Output ONLY the consolidated bulleted list ({max_bullets} bullets max):"""

    prompt = SUMMARIZE_PROMPT.format(
        directive=directive,
        bullets=bullets,
        max_bullets=max_bullets
    )
    
    response = call_deepseek(
        prompt=prompt,
        model=model,
        max_tokens=1500,
        temperature=temperature,
        api_key_override=api_key_override,
        reasoning_effort=reasoning_effort,
        exclude_reasoning=None,
    )
    
    summarized = _clean_bullet_response(response)
    print(f"[PACKING] Sumarizacao concluida: {len(bullets.splitlines())} → {len(summarized.splitlines())} linhas")
    return summarized


def _deduplicate_bullets(bullets: str, embedder, threshold: float = 0.90) -> str:
    """
    Remove bullets semanticamente redundantes usando embeddings.
    
    Args:
        bullets: String com bullets formatados (um por linha com "- ")
        embedder: Modelo de embeddings (do get_embedder)
        threshold: Limiar de similaridade (0.90 = 90% similar = redundante)
    
    Returns:
        String com bullets unicos (sem redundancia)
    
    Example:
        >>> bullets = "- Use named characters\\n- Use characters with names\\n- Add sensory details"
        >>> unique = _deduplicate_bullets(bullets, embedder, threshold=0.90)
        >>> print(unique)
        "- Use named characters\\n- Add sensory details"
    """
    if not bullets or not bullets.strip():
        return bullets
    
    # Separar bullets em lista
    lines = bullets.split("\n")
    bullets_list = [line.strip() for line in lines if line.strip().startswith("-")]
    
    if len(bullets_list) <= 1:
        return bullets
    
    print(f"[DEDUP] Verificando {len(bullets_list)} bullets...")
    
    # Remover o "- " do inicio para embeddings mais limpos
    bullets_text = [b[1:].strip() if b.startswith("-") else b for b in bullets_list]
    
    # Gerar embeddings
    try:
        embeddings = embed_texts(embedder, bullets_text)
    except Exception as e:
        print(f"[DEDUP] Erro ao gerar embeddings: {e}")
        return bullets  # Retornar original se falhar
    
    # Encontrar bullets unicos (sem redundancia)
    keep_indices = []
    
    for i in range(len(embeddings)):
        is_redundant = False
        
        # Comparar com bullets ja mantidos
        for j in keep_indices:
            similarity = 1 - cosine_distance(embeddings[i], embeddings[j])
            
            if similarity >= threshold:
                # Redundante! Remover
                is_redundant = True
                print(f"[DEDUP] Bullet {i+1} redundante com bullet {j+1} (similaridade={similarity:.3f})")
                print(f"  - Removido: {bullets_list[i][:70]}...")
                print(f"  - Mantido:  {bullets_list[j][:70]}...")
                break
        
        if not is_redundant:
            keep_indices.append(i)
    
    # Reconstruir string de bullets
    unique_bullets = [bullets_list[i] for i in keep_indices]
    result = "\n".join(unique_bullets)
    
    removed = len(bullets_list) - len(unique_bullets)
    if removed > 0:
        print(f"[DEDUP] Removidos {removed} bullets redundantes ({len(bullets_list)} → {len(unique_bullets)})")
    else:
        print(f"[DEDUP] Nenhum bullet redundante detectado")
    
    return result


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
    
    Aceita formatos:
    - REPLACE/ADD/KEEP (formato atual do pipeline)
    - DO:/DON'T: (formato legado, ainda suportado)
    
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
    
    # Cada linha deve comecar com "-"
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if not line.startswith("-"):
            return False
        
        # Verificar se contem REPLACE/ADD/KEEP (formato atual) OU DO:/DON'T: (legado)
        line_upper = line.upper()
        has_replace = "REPLACE" in line_upper
        has_add = "ADD:" in line_upper or "ADD " in line_upper
        has_keep = "KEEP:" in line_upper or "KEEP " in line_upper
        has_do = "DO:" in line_upper
        has_dont = "DON'T:" in line_upper or "DONT:" in line_upper
        
        # Aceitar se tiver pelo menos um dos formatos
        if not (has_replace or has_add or has_keep or has_do or has_dont):
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
    
    # Diretiva de exemplo
    DIRECTIVE = "Center your story around two characters who like each other but don't get a happily ever after."
    
    print("=== TESTE: refinement_packing.py ===\n")
    print("Input JSON:")
    print(json.dumps(CRITIQUE_JSON, indent=2, ensure_ascii=False))
    print(f"\nDirective: {DIRECTIVE}\n")
    
    try:
        result = packing_step(
            critique_json=CRITIQUE_JSON,
            directive=DIRECTIVE,
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

