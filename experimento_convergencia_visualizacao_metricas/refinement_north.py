#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NORTH STAR: Gera diretrizes fixas baseadas em ideias humanas.

Este modulo analisa as ideias humanas de referencia e extrai padroes
FUNDAMENTAIS que devem ser mantidos CONSTANTES ao longo de todas as iteracoes.

O objetivo e fornecer uma ancora para evitar oscilacao do feedback.
"""

from __future__ import annotations

from typing import List, Optional

from bleu_minimal_deepseek import call_deepseek


NORTH_STAR_PROMPT_TEMPLATE = """You are analyzing human-written short story ideas to extract CORE PATTERNS that make them effective.

Writing contest invitation:
------
{invitation}
------

Original directive:
------
{directive}
------

Human story ideas (reference examples):
------
{human_ideas}
------

Your task: Identify the TIMELESS, FUNDAMENTAL patterns that characterize these human ideas.

Focus on:
1. CHARACTER PATTERNS: Types, naming, backstories, occupations, development
2. SETTING PATTERNS: Locations, atmospheres, integral vs generic, specificity
3. PLOT PATTERNS: Conflict types, structure, pacing, turning points
4. EMOTIONAL PATTERNS: Tone, endings, emotional arcs, themes
5. CRAFT PATTERNS: Literary devices, metaphors, sensory details, dialogue style

CRITICAL INSTRUCTIONS:
- Extract patterns that appear in MULTIPLE human examples (not just one)
- Be ULTRA-SPECIFIC: mention concrete elements, not vague advice
- Focus on CRAFT ELEMENTS: character types, settings, plot structure, tone
- Output 5-7 core directives that capture the ESSENCE of these stories
- These directives should be TIMELESS (valid for all iterations)
- DO NOT mention specific story titles, character names, or plot details
- Output ONLY the bulleted list, nothing else

Output format:
- [Pattern 1]
- [Pattern 2]
- [Pattern 3]
...

Output the bulleted list now:"""


def generate_north_star(
    invitation: str,
    directive: str,
    human_ideas: List[str],
    model: str = "gpt-4o",
    temperature: float = 0.3,
    max_tokens: int = 1000,
    api_key_override: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    """
    Gera o NORTE FIXO analisando as ideias humanas.
    
    Esta funcao deve ser chamada UMA UNICA VEZ no inicio do refinement loop.
    O norte gerado sera usado em TODAS as iteracoes como ancora.
    
    Args:
        invitation: Convite do concurso de escrita
        directive: Diretiva original
        human_ideas: Lista de ideias humanas (referencias)
        model: Modelo LLM a usar (default: gpt-4o, mais preciso)
        temperature: Temperatura para geracao (default: 0.3, conservador)
        max_tokens: Maximo de tokens (default: 1000)
        api_key_override: API key alternativa (opcional)
        reasoning_effort: Reasoning effort para modelos que suportam (opcional)
    
    Returns:
        String com bullets do norte fixo
    
    Example:
        >>> north = generate_north_star(
        ...     invitation="Strangers Again...",
        ...     directive="Center your story around...",
        ...     human_ideas=["Barney in the Rubble...", "Maybe One Day..."]
        ... )
        >>> print(north)
        - Use named characters with specific occupations (detective, teacher, etc.)
        - Set stories in locations with inherent tension or emotional significance
        - Include central symbols or objects that represent the relationship
        - Create bittersweet endings where connection ends but transforms characters
    """
    # Formatar ideias humanas
    human_ideas_text = "\n\n".join(
        f"IDEA {i+1}:\n{idea}" 
        for i, idea in enumerate(human_ideas)
    )
    
    # Montar prompt
    prompt = NORTH_STAR_PROMPT_TEMPLATE.format(
        invitation=invitation,
        directive=directive,
        human_ideas=human_ideas_text
    )
    
    # Chamar LLM
    print(f"[NORTH] Analisando {len(human_ideas)} ideias humanas...")
    print(f"[NORTH] Usando modelo: {model} (temperature={temperature})")
    
    response = call_deepseek(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key_override=api_key_override,
        reasoning_effort=reasoning_effort,
        exclude_reasoning=True,  # Queremos so o output final
    )
    
    # Limpar resposta (remover linhas vazias extras)
    bullets = _clean_north_response(response)
    
    print(f"[NORTH] Norte fixo gerado com sucesso!")
    num_diretrizes = len([l for l in bullets.split('\n') if l.strip().startswith('-')])
    print(f"[NORTH] Numero de diretrizes: {num_diretrizes}")
    
    return bullets


def _clean_north_response(response: str) -> str:
    """
    Limpa a resposta do LLM para norte fixo.
    
    Args:
        response: Resposta bruta do LLM
    
    Returns:
        Bullets limpos
    """
    # Remover linhas vazias extras
    lines = [line for line in response.split('\n') if line.strip()]
    
    # Garantir que todas as linhas comecem com '-'
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line.startswith('-'):
            # Adicionar '-' se nao tiver
            if line:
                line = f"- {line}"
        if line:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def format_north_with_tactical(north_star: str, tactical_bullets: str) -> str:
    """
    Combina norte fixo com bullets taticos em um unico texto.
    
    Args:
        north_star: Bullets do norte fixo (constantes)
        tactical_bullets: Bullets taticos da iteracao atual (dinamicos)
    
    Returns:
        String formatada com ambos
    
    Example:
        >>> north = "- Use named characters\n- Dramatic settings"
        >>> tactical = "- Reduce excessive dialogue\n- Add more sensory details"
        >>> combined = format_north_with_tactical(north, tactical)
        >>> print(combined)
        CORE DIRECTIVES (ALWAYS FOLLOW):
        - Use named characters
        - Dramatic settings
        
        CURRENT CORRECTIONS (address recent issues):
        - Reduce excessive dialogue
        - Add more sensory details
    """
    # Se tactical estiver vazio, retornar apenas norte
    if not tactical_bullets or not tactical_bullets.strip():
        return north_star
    
    # Formatar com secoes claras
    combined = f"""CORE DIRECTIVES (ALWAYS FOLLOW):
{north_star}

CURRENT CORRECTIONS (address recent issues):
{tactical_bullets}"""
    
    return combined

