#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa CRITIQUE: Analisa vibes da LLM vs humanas e gera JSON.

Este modulo compara as caracteristicas gerais (vibes) das ideias geradas
pela LLM com as ideias humanas de referencia, classificando-as como
"do" (fazer) ou "don't" (nao fazer).

Requer: OPENAI_API_KEY ou OPENROUTER_API_KEY no ambiente.
"""

from __future__ import annotations

import json
import re
from typing import List, Dict, Optional
from pathlib import Path

from bleu_minimal_deepseek import call_deepseek


# Template do prompt CRITIQUE
CRITIQUE_PROMPT_TEMPLATE = """Consider the following writing contest invitation:
------
{invitation}
------

And also this writing directive:
------
{directive}
------

Below are two sets of short-story ideas (SET A and SET B) generated based on the invitation and directive.

SET A:
------
{human_ideas}
------

SET B:
------
{llm_ideas}
------

Your task: Analyze the characteristic vibes/patterns in SET B and determine which ones should be reinforced (DO) or avoided (DON'T) to make SET B more similar to SET A.

Follow this process:
1) Identify the overall vibes and patterns in SET A
2) Identify the overall vibes and patterns in SET B
3) For each significant vibe/pattern in SET B:
   - If it MATCHES a vibe in SET A → mark as "do" (reinforce this)
   - If it CONFLICTS with SET A's vibes → mark as "don't" (avoid this)
4) Additionally, identify vibes present in SET A but MISSING in SET B → mark as "do" (add this)

Generate a JSON array where each entry has:
{{
    "vibe_pattern": "[do|don't]",
    "vibe_description": "<describe the vibe/pattern clearly and specifically>"
}}

CRITICAL INSTRUCTIONS:
- Focus on ACTIONABLE patterns (tone, structure, character development, emotional arc, etc.)
- Be SPECIFIC: "Characters lack names and backstories" is better than "underdeveloped characters"
- Output ONLY the JSON array at the end
- NO markdown fences (```), NO comments, NO trailing commas
- Use only standard ASCII quotes (")
- Format: [{{"vibe_pattern": "do/don't", "vibe_description": "..."}}]

Output the JSON array now:"""


def critique_step(
    invitation: str,
    directive: str,
    human_ideas: List[str],
    llm_ideas: List[str],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 2000,
    api_key_override: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Etapa CRITIQUE: Compara vibes e retorna JSON com padroes DO/DON'T.
    
    Args:
        invitation: Convite do concurso de escrita
        directive: Diretiva original
        human_ideas: Lista de ideias humanas (referencias)
        llm_ideas: Lista de ideias geradas pela LLM
        model: Modelo LLM a usar (default: gpt-4o-mini)
        temperature: Temperatura para geracao (default: 0.7)
        max_tokens: Maximo de tokens (default: 2000)
        api_key_override: API key alternativa (opcional)
        reasoning_effort: Reasoning effort para modelos que suportam (opcional)
    
    Returns:
        Lista de dicionarios com vibe_pattern e vibe_description
    
    Example:
        >>> human = ["Idea 1...", "Idea 2..."]
        >>> llm = ["Idea A...", "Idea B..."]
        >>> result = critique_step("Invitation...", "Directive...", human, llm)
        >>> print(result[0]["vibe_pattern"])
        "don't"
    """
    # Formatar ideias humanas
    human_ideas_str = "\n---\n".join(human_ideas)
    
    # Formatar ideias da LLM
    llm_ideas_str = "\n---\n".join(llm_ideas)
    
    # Montar prompt
    prompt = CRITIQUE_PROMPT_TEMPLATE.format(
        invitation=invitation,
        directive=directive,
        human_ideas=human_ideas_str,
        llm_ideas=llm_ideas_str
    )
    
    # Chamar LLM
    print(f"[CRITIQUE] Chamando modelo {model}...")
    
    # IMPORTANTE: GPT-5 com exclude_reasoning=True retorna vazio!
    # Entao usamos exclude_reasoning=False e extraimos JSON do reasoning manualmente
    response = call_deepseek(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key_override=api_key_override,
        reasoning_effort=reasoning_effort,
        exclude_reasoning=False,  # GPT-5 precisa disso para funcionar
    )
    
    # GPT-5 pode retornar muito texto de raciocinio, precisamos extrair apenas o JSON
    # Se a resposta parece conter raciocinio extenso, tentar pegar apenas a parte final
    if len(response) > 3000 and "**" in response:  # Indicadores de reasoning do GPT-5
        print(f"[CRITIQUE] Detectado reasoning extenso ({len(response)} chars), extraindo JSON...")
        # Tentar pegar apenas a ultima parte que parece ser JSON
        lines = response.split('\n')
        # Procurar de tras pra frente por linhas que parecem JSON
        json_lines = []
        in_json = False
        for line in reversed(lines):
            if ']' in line and not in_json:
                in_json = True
            if in_json:
                json_lines.insert(0, line)
            if '[' in line and in_json:
                break
        if json_lines:
            response = '\n'.join(json_lines)
            print(f"[CRITIQUE] Extraido bloco JSON ({len(response)} chars)")
    
    # Parsear JSON da resposta
    try:
        json_critique = _parse_json_response(response)
        print(f"[CRITIQUE] JSON parseado com sucesso: {len(json_critique)} vibes detectadas")
        return json_critique
    except Exception as e:
        print(f"[CRITIQUE] ERRO ao parsear JSON: {e}")
        print(f"[CRITIQUE] Resposta raw (primeiros 500 chars): {response[:500]}...")
        if len(response) > 500:
            print(f"[CRITIQUE] Resposta raw (ultimos 500 chars): ...{response[-500:]}")
        raise


def _json_sanitize(s: str) -> str:
    """
    Sanitiza string JSON removendo comentarios, trailing commas, aspas tipograficas, etc.
    
    Args:
        s: String potencialmente "suja" com JSON
    
    Returns:
        String sanitizada pronta para json.loads()
    """
    import re
    
    # Pega so o que esta entre o primeiro '[' e o ultimo ']'
    first, last = s.find('['), s.rfind(']')
    if first != -1 and last != -1 and first < last:
        s = s[first:last+1]
    
    # Remove code fences ```...```
    s = re.sub(r"^```[a-zA-Z0-9]*\s*|\s*```$", "", s.strip(), flags=re.DOTALL)
    
    # Remove comentarios // ... (single-line)
    s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
    
    # Remove comentarios /* ... */ (multi-line)
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)
    
    # Normaliza aspas tipograficas " " ' ' -> "
    s = s.replace(""", '"').replace(""", '"')
    s = s.replace("'", "'").replace("'", "'")
    
    # Remove virgulas finais (trailing commas) antes de ] ou }
    s = re.sub(r",\s*([\]}])", r"\1", s)
    
    # Normaliza "don't" com apostrofo tipografico -> "don't"
    s = s.replace("don't", "don't").replace("don't", "don't")
    
    return s.strip()


def _parse_json_response(response: str) -> List[Dict[str, str]]:
    """
    Parseia a resposta JSON da LLM.
    
    Tenta diferentes estrategias de parsing:
    0. Sanitizacao (remove comentarios, trailing commas, aspas tipograficas)
    1. JSON direto
    2. Extrair JSON de markdown code block (```json/jsonc ... ```)
    3. Buscar array JSON no texto (do fim para o inicio - GPT-5 reasoning)
    4. Remover texto antes/depois do JSON
    5. Corrigir aspas simples para duplas
    6. Buscar objetos individuais
    
    Args:
        response: Resposta da LLM (pode conter markdown, comentarios, etc)
    
    Returns:
        Lista de dicionarios parseada
    
    Raises:
        ValueError: Se nao conseguir parsear o JSON
    """
    response = (response or "").strip()
    
    if not response:
        raise ValueError("Resposta vazia da LLM")
    
    # Tentativa 0: Sanitizar primeiro (remove comentarios, trailing commas, aspas tipograficas)
    sanitized = _json_sanitize(response)
    
    # Tentativa 1: JSON direto (com sanitizacao)
    try:
        data = json.loads(sanitized)
        if isinstance(data, list):
            print(f"[CRITIQUE] JSON parseado com sucesso (sanitizado)")
            return data
        elif isinstance(data, dict):
            print(f"[CRITIQUE] JSON dict parseado, convertendo para lista")
            return [data]
    except json.JSONDecodeError as e:
        print(f"[CRITIQUE] Tentativa 1 falhou: {e}")
        pass
    
    # Tentativa 2: Extrair de markdown code block (aceita json, jsonc, JSONC)
    code_block_match = re.search(r"```(?:jsonc?|JSONC?)?\s*([\s\S]*?)\s*```", response, re.IGNORECASE)
    if code_block_match:
        try:
            block_content = code_block_match.group(1).strip()
            # Sanitizar o conteudo do bloco
            block_sanitized = _json_sanitize(block_content)
            data = json.loads(block_sanitized)
            if isinstance(data, list):
                print(f"[CRITIQUE] JSON extraido de markdown code block")
                return data
        except json.JSONDecodeError as e:
            print(f"[CRITIQUE] Tentativa 2 falhou: {e}")
            pass
    
    # Tentativa 3: Buscar array JSON no texto (GPT-5 reasoning: tentar do fim para o inicio)
    # GPT-5 coloca reasoning text ANTES do JSON final, entao o ultimo match eh mais confiavel
    try:
        all_json_matches = list(re.finditer(r'\[\s*\{[\s\S]*?\}\s*\]', response, re.DOTALL))
        # Tentar do ultimo para o primeiro (o ultimo eh mais provavel de ser o output final)
        for match in reversed(all_json_matches):
            try:
                json_str = match.group(0)
                data = json.loads(json_str)
                if isinstance(data, list) and len(data) > 0:
                    # Validar se tem a estrutura esperada
                    if all(isinstance(item, dict) and "vibe_pattern" in item for item in data):
                        print(f"[CRITIQUE] JSON extraido de reasoning (posicao {match.start()}/{len(response)})")
                        return data
            except json.JSONDecodeError:
                continue
    except:
        pass
    
    # Tentativa 3b: Fallback para primeira ocorrencia (modo greedy)
    json_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", response, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    
    # Tentativa 4: Remover texto antes/depois do primeiro [ e ultimo ]
    try:
        first_bracket = response.find('[')
        last_bracket = response.rfind(']')
        if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
            json_str = response[first_bracket:last_bracket+1]
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
    except json.JSONDecodeError:
        pass
    
    # Tentativa 5: Corrigir aspas simples para duplas e tentar parsear
    try:
        fixed_response = response.replace("'", '"')
        data = json.loads(fixed_response)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Tentativa 6: Buscar multiplos objetos JSON individuais
    try:
        objects = re.findall(r'\{[^{}]*"vibe_pattern"[^{}]*"vibe_description"[^{}]*\}', response, re.IGNORECASE)
        if objects:
            print(f"[CRITIQUE] Tentativa 6: encontrados {len(objects)} objetos candidatos")
            parsed_objects = []
            for i, obj_str in enumerate(objects):
                try:
                    obj = json.loads(obj_str)
                    if isinstance(obj, dict) and "vibe_pattern" in obj and "vibe_description" in obj:
                        parsed_objects.append(obj)
                except Exception as e:
                    print(f"[CRITIQUE] Tentativa 6, objeto {i+1} falhou: {e}")
                    continue
            if parsed_objects:
                print(f"[CRITIQUE] Tentativa 6: {len(parsed_objects)} objetos parseados com sucesso")
                return parsed_objects
    except Exception as e:
        print(f"[CRITIQUE] Tentativa 6 falhou completamente: {e}")
        pass
    
    # Salvar resposta para debug
    print(f"\n{'='*80}")
    print(f"[CRITIQUE] ERRO - Nao foi possivel parsear JSON da resposta")
    print(f"{'='*80}")
    print(f"[CRITIQUE] Resposta RAW (primeiros 1000 chars):")
    print(response[:1000])
    if len(response) > 2000:
        print(f"\n[CRITIQUE] Resposta RAW (ultimos 1000 chars):")
        print(f"...{response[-1000:]}")
    elif len(response) > 1000:
        print(f"\n[CRITIQUE] Resposta RAW (resto):")
        print(response[1000:])
    
    print(f"\n[CRITIQUE] Resposta SANITIZADA:")
    print(sanitized[:1000])
    if len(sanitized) > 1000:
        print(f"...{sanitized[-500:]}")
    
    print(f"\n[CRITIQUE] Total: {len(response)} chars (raw), {len(sanitized)} chars (sanitized)")
    print(f"{'='*80}\n")
    
    raise ValueError(f"Nao foi possivel parsear JSON da resposta")


def _validate_critique_json(data: List[Dict[str, str]]) -> bool:
    """
    Valida se o JSON do critique esta no formato esperado.
    
    Args:
        data: Lista de dicionarios a validar
    
    Returns:
        True se valido, False caso contrario
    """
    if not isinstance(data, list):
        return False
    
    for item in data:
        if not isinstance(item, dict):
            return False
        if "vibe_pattern" not in item or "vibe_description" not in item:
            return False
        if item["vibe_pattern"] not in ["do", "don't"]:
            return False
        if not isinstance(item["vibe_description"], str) or not item["vibe_description"].strip():
            return False
    
    return True


# Exemplo de uso (para testes)
if __name__ == "__main__":
    # Dados de teste
    INVITATION = """Strangers Again

I've been thinking a lot lately about the need to feel connected - to be seen, remembered, or maybe even just understood. Over the years, I've noticed how connection can manifest in the smallest of things: a shared meal, a passing glance, a familiar name we can't quite place.

This week, let's write stories about that pull between people. From fleeting relationships to chance encounters with strangers who seem like old friends, let's explore yearning and connection, even when things are complicated or just out of reach."""

    DIRECTIVE = "Center your story around two characters who like each other but don't get a happily ever after."

    HUMAN_EXAMPLES = [
        "A story about two childhood friends who reconnect...",
        "A tale of missed connections at a train station..."
    ]

    LLM_EXAMPLES = [
        "Elevator Theory: Two neighbors keep missing each other's names...",
        "The Streetlight Swap: A city electrician and a poet swap lines..."
    ]

    print("=== TESTE: refinement_critique.py ===\n")
    
    try:
        result = critique_step(
            invitation=INVITATION,
            directive=DIRECTIVE,
            human_ideas=HUMAN_EXAMPLES,
            llm_ideas=LLM_EXAMPLES,
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        print("\n=== RESULTADO ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        print("\n=== VALIDACAO ===")
        is_valid = _validate_critique_json(result)
        print(f"JSON valido: {is_valid}")
        
    except Exception as e:
        print(f"\n=== ERRO ===")
        print(f"Tipo: {type(e).__name__}")
        print(f"Mensagem: {e}")
        import traceback
        traceback.print_exc()

