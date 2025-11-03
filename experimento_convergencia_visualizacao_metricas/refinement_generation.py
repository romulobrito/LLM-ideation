#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa GENERATION: Gera novas ideias com diretiva revisada.

Este modulo pega a diretiva original, adiciona os bullets de DO/DON'T
e gera novas ideias de historias curtas.

Requer: OPENAI_API_KEY ou OPENROUTER_API_KEY no ambiente.
"""

from __future__ import annotations

import re
from typing import List, Optional

from bleu_minimal_deepseek import call_deepseek


# Template do prompt GENERATION
GENERATION_PROMPT_TEMPLATE = """Consider the following writing contest invitation:
------
{invitation}
------

And also these writing directives:
------
{revised_directive}
------

Your task is to creatively generate {num_ideas} short-story ideas based on the invitation and directives.

CRITICAL CRAFT REQUIREMENT:
- The directives above contain SPECIFIC craft patterns derived from high-quality story ideas
- Follow the patterns and craft elements specified in the directives carefully
- Apply the REPLACE, ADD, and KEEP guidance to incorporate these proven craft elements
- Focus on the SPECIFIC character types, settings, plot structures, and emotional tones indicated

CRITICAL FORMATTING REQUIREMENTS:
- Generate EXACTLY {num_ideas} separate ideas
- Number each idea: 1., 2., 3., etc.
- Each idea should span ~150 words
- Each idea must be distinct in theme, tone, and concept
- Separate ideas with a blank line

Example format:
1. [Title or first line]
[~150 words of story idea]

2. [Title or first line]
[~150 words of story idea]

Now generate {num_ideas} ideas following this exact format:"""


def generation_step(
    invitation: str,
    directive: str,
    bullets: str,
    num_ideas: int = 5,
    model: str = "gpt-4o-mini",
    temperature: float = 0.8,
    max_tokens: int = 4000,
    api_key_override: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    human_examples: Optional[List[str]] = None,
) -> List[str]:
    """
    Etapa GENERATION: Gera novas ideias com diretiva revisada.
    
    Args:
        invitation: Convite do concurso de escrita
        directive: Diretiva original
        bullets: Bullets de DO/DON'T do packing
        num_ideas: Numero de ideias a gerar (default: 5)
        model: Modelo LLM a usar (default: gpt-4o-mini)
        temperature: Temperatura para geracao (default: 0.8)
        max_tokens: Maximo de tokens (default: 4000)
        api_key_override: API key alternativa (opcional)
        reasoning_effort: Reasoning effort para modelos que suportam (opcional)
        human_examples: Exemplos de ideias humanas para few-shot learning (opcional)
    
    Returns:
        Lista de ideias geradas (strings)
    
    Example:
        >>> bullets = "- DO: Use subtle encounters\\n- DON'T: Use whimsical titles"
        >>> examples = ["Example 1...", "Example 2..."]
        >>> ideas = generation_step("Invitation...", "Directive...", bullets, 
        ...                         num_ideas=3, human_examples=examples)
        >>> print(len(ideas))
        3
    """
    # Criar diretiva revisada
    revised_directive = _create_revised_directive(directive, bullets)
    
    # Montar prompt
    prompt = GENERATION_PROMPT_TEMPLATE.format(
        invitation=invitation,
        revised_directive=revised_directive,
        num_ideas=num_ideas
    )
    
    # Adicionar exemplos humanos (few-shot learning) se fornecidos
    if human_examples and len(human_examples) > 0:
        # Usar 2-3 exemplos (não todos para não explodir o prompt)
        num_examples = min(3, len(human_examples))
        examples_to_use = human_examples[:num_examples]
        
        examples_text = "\n\nEXAMPLES OF DESIRED STYLE AND QUALITY:\n"
        examples_text += "Study these examples carefully to understand the tone, structure, and craft:\n\n"
        
        for i, example in enumerate(examples_to_use, 1):
            # Truncar para ~200 palavras se muito longo
            words = example.split()
            if len(words) > 200:
                truncated = " ".join(words[:200]) + "..."
            else:
                truncated = example
            
            examples_text += f"EXAMPLE {i}:\n{truncated}\n\n"
        
        examples_text += "---\n\nNow, generate your own ideas following the STYLE and CRAFT shown above, "
        examples_text += "but with COMPLETELY NEW stories, characters, and concepts:\n"
        
        prompt += examples_text
        
        print(f"[GENERATION] Usando {num_examples} exemplos humanos para few-shot learning")
    
    # Chamar LLM
    print(f"[GENERATION] Chamando modelo {model} para gerar {num_ideas} ideias...")
    response = call_deepseek(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key_override=api_key_override,
        reasoning_effort=reasoning_effort,
        exclude_reasoning=None,  # Auto-detecção: True para DeepSeek, False para GPT-5
    )
    
    # Parsear ideias
    ideas = _parse_ideas_response(response)
    
    print(f"[GENERATION] Geradas {len(ideas)} ideias com sucesso")
    return ideas


def _create_revised_directive(directive: str, bullets: str) -> str:
    """
    Cria diretiva revisada adicionando bullets.
    
    Args:
        directive: Diretiva original
        bullets: Bullets de DO/DON'T
    
    Returns:
        Diretiva revisada com bullets
    """
    revised = f"{directive}\n\n"
    revised += "Additional guidance:\n"
    revised += bullets
    
    return revised


def _parse_ideas_response(response: str) -> List[str]:
    """
    Parseia a resposta da LLM em lista de ideias.
    
    Cada ideia deve estar no formato:
    N. Title
    Description
    
    Args:
        response: Resposta bruta da LLM
    
    Returns:
        Lista de ideias (strings, cada uma com titulo e descricao)
    """
    response = (response or "").strip()
    
    # Tentar parsear com numeros (1., 2., etc)
    # Pattern: N. Titulo\nDescricao
    pattern = r"(\d+)\.\s*(.+?)(?=\n\d+\.|$)"
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        ideas = []
        for num, content in matches:
            content = content.strip()
            if content:
                ideas.append(content)
        
        if ideas:
            return ideas
    
    # Fallback: dividir por linhas vazias duplas
    chunks = re.split(r"\n\s*\n", response)
    ideas = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    if not ideas:
        # Ultimo fallback: a resposta inteira como uma ideia
        ideas = [response]
    
    return ideas


def _validate_ideas(ideas: List[str], expected_count: int) -> bool:
    """
    Valida se as ideias geradas estao no formato esperado.
    
    Args:
        ideas: Lista de ideias
        expected_count: Numero esperado de ideias
    
    Returns:
        True se valido (pelo menos 50% do esperado), False caso contrario
    """
    if not ideas:
        return False
    
    # Aceitar se tivermos pelo menos 50% das ideias esperadas
    min_acceptable = max(1, expected_count // 2)
    
    if len(ideas) < min_acceptable:
        return False
    
    # Verificar se cada ideia tem conteudo minimo
    for idea in ideas:
        if not idea or len(idea.strip()) < 20:
            return False
    
    return True


# Exemplo de uso (para testes)
if __name__ == "__main__":
    # Dados de teste
    INVITATION = """Strangers Again

I've been thinking a lot lately about the need to feel connected - to be seen, remembered, or maybe even just understood. Over the years, I've noticed how connection can manifest in the smallest of things: a shared meal, a passing glance, a familiar name we can't quite place.

This week, let's write stories about that pull between people. From fleeting relationships to chance encounters with strangers who seem like old friends, let's explore yearning and connection, even when things are complicated or just out of reach."""

    DIRECTIVE = "Center your story around two characters who like each other but don't get a happily ever after."

    BULLETS = """- DON'T: Use whimsical, metaphor-heavy titles that feel fantastical
- DO: Create scenarios with subtle, everyday encounters
- DON'T: Employ overly playful or poetic language"""

    print("=== TESTE: refinement_generation.py ===\n")
    
    try:
        result = generation_step(
            invitation=INVITATION,
            directive=DIRECTIVE,
            bullets=BULLETS,
            num_ideas=3,
            model="gpt-4o-mini",
            temperature=0.8
        )
        
        print("\n=== RESULTADO ===")
        for i, idea in enumerate(result, 1):
            print(f"\n--- Ideia {i} ---")
            print(idea[:200] + ("..." if len(idea) > 200 else ""))
        
        print("\n=== VALIDACAO ===")
        is_valid = _validate_ideas(result, expected_count=3)
        print(f"Ideias validas: {is_valid}")
        print(f"Total de ideias: {len(result)}")
        
    except Exception as e:
        print(f"\n=== ERRO ===")
        print(f"Tipo: {type(e).__name__}")
        print(f"Mensagem: {e}")
        import traceback
        traceback.print_exc()

