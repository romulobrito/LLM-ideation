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

Your task: Provide CONTRASTIVE FEEDBACK to make SET B more similar to SET A.

CRITICAL: For each pattern, use this format:
1) CONFLICTS (what SET B does wrong):
   - Identify SPECIFIC element in SET B (e.g., "archetypal unnamed characters")
   - Identify CORRESPONDING element in SET A (e.g., "named characters with detailed backstories")
   - Output: {{"action": "replace", "from": "what SET B has", "to": "what SET A has"}}

2) MATCHES (what SET B does right):
   - Identify element present in BOTH sets
   - Output: {{"action": "keep", "description": "what to keep doing"}}

3) MISSING (what SET A has that SET B lacks):
   - Identify element in SET A but absent in SET B
   - Output: {{"action": "add", "description": "what to add"}}

EXAMPLES OF GOOD FEEDBACK:
{{"action": "replace", "from": "Gimmicky concepts centered on objects", "to": "Character-driven stories with emotional depth"}}
{{"action": "add", "description": "Named characters with specific backstories and occupations"}}
{{"action": "keep", "description": "Bittersweet endings without forced resolution"}}

BAD (too vague): {{"action": "replace", "from": "Bad writing", "to": "Good writing"}}
BAD (not contrastive): {{"action": "don't", "description": "Don't use gimmicks"}}

CRITICAL INSTRUCTIONS:
- Be ULTRA-SPECIFIC: mention concrete elements (character types, settings, plot devices, tone markers)
- ALWAYS provide "from" AND "to" for "replace" actions
- Focus on CRAFT ELEMENTS: character development, setting details, emotional arc, pacing, dialogue style
- Output 5-8 feedback items total

CRITICAL OUTPUT FORMAT
YOU MUST END YOUR RESPONSE WITH THE JSON ARRAY.
DO NOT STOP BEFORE GENERATING THE JSON.
After any analysis or thinking, YOU MUST write:

[
  {{"action": "replace", "from": "...", "to": "..."}},
  {{"action": "add", "description": "..."}},
  {{"action": "keep", "description": "..."}}
]

RULES:
- NO markdown fences (```)
- NO comments (//)
- NO trailing commas
- Use ONLY standard ASCII quotes (")
- END with the closing bracket ]

NOW GENERATE THE JSON ARRAY:"""


def critique_step(
    invitation: str,
    directive: str,
    human_ideas: List[str],
    llm_ideas: List[str],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 4000,
    api_key_override: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    previous_feedbacks: Optional[List[List[Dict[str, str]]]] = None,
) -> List[Dict[str, str]]:
    """
    Etapa CRITIQUE: Compara vibes e retorna JSON com feedback contrastivo.
    
    Args:
        invitation: Convite do concurso de escrita
        directive: Diretiva original
        human_ideas: Lista de ideias humanas (referencias)
        llm_ideas: Lista de ideias geradas pela LLM
        model: Modelo LLM a usar (default: gpt-4o-mini)
        temperature: Temperatura para geracao (default: 0.7)
        max_tokens: Maximo de tokens (default: 4000)
        api_key_override: API key alternativa (opcional)
        reasoning_effort: Reasoning effort para modelos que suportam (opcional)
        previous_feedbacks: Lista de feedbacks anteriores para evitar redundancia (opcional)
    
    Returns:
        Lista de dicionarios com feedback contrastivo:
        - action: "replace", "add", ou "keep"
        - from/to: para "replace" (o que substituir e pelo que)
        - description: para "add" e "keep"
    
    Example:
        >>> human = ["Idea 1...", "Idea 2..."]
        >>> llm = ["Idea A...", "Idea B..."]
        >>> result = critique_step("Invitation...", "Directive...", human, llm)
        >>> print(result[0])
        {"action": "replace", "from": "...", "to": "..."}
    """
    # Formatar ideias humanas
    human_ideas_str = "\n---\n".join(human_ideas)
    
    # Formatar ideias da LLM
    llm_ideas_str = "\n---\n".join(llm_ideas)
    
    # Preparar contexto de feedback anterior (se houver)
    previous_context = ""
    if previous_feedbacks and len(previous_feedbacks) > 0:
        import json
        previous_context = "\n\nPREVIOUS FEEDBACK FROM PAST ITERATIONS (DO NOT REPEAT):\n"
        previous_context += "You have already given the following feedback in previous iterations.\n"
        previous_context += "DO NOT repeat these points. Focus on NEW patterns or issues not yet addressed:\n------\n"
        for i, feedback in enumerate(previous_feedbacks[-3:], 1):  # Ultimas 3 iteracoes
            previous_context += f"Iteration {i}:\n"
            previous_context += json.dumps(feedback, indent=2, ensure_ascii=False)
            previous_context += "\n------\n"
        print(f"[CRITIQUE] Incluindo historico de {len(previous_feedbacks)} iteracoes anteriores")
    
    # Montar prompt
    prompt = CRITIQUE_PROMPT_TEMPLATE.format(
        invitation=invitation,
        directive=directive,
        human_ideas=human_ideas_str,
        llm_ideas=llm_ideas_str
    )
    
    # Adicionar contexto de feedback anterior (se houver)
    if previous_context:
        prompt = prompt + previous_context
    
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
        print(f"[CRITIQUE] ERRO ao parsear JSON (tentativa 1): {e}")
        print(f"[CRITIQUE] Tentando retry com exclude_reasoning=True...")
        
        # FALLBACK: Tentar novamente com exclude_reasoning=True (costuma retornar JSON limpo)
        try:
            # Aumentar max_tokens no retry para garantir resposta completa
            retry_max_tokens = max(max_tokens, 2000)  # Minimo 2000 tokens para JSON completo
            print(f"[CRITIQUE] Retry com max_tokens={retry_max_tokens} (original: {max_tokens})")
            
            response2 = call_deepseek(
                prompt=prompt,
                model=model,
                max_tokens=retry_max_tokens,
                temperature=temperature,
                api_key_override=api_key_override,
                reasoning_effort=reasoning_effort,
                exclude_reasoning=True,  # Forcar apenas content
            )
            
            # Validar se a resposta nao esta vazia ou muito curta
            if not response2 or len(response2.strip()) < 50:
                print(f"[CRITIQUE] AVISO: Resposta retry muito curta ou vazia ({len(response2) if response2 else 0} chars)")
                raise ValueError("Resposta retry muito curta ou vazia")
            
            print(f"[CRITIQUE] Resposta retry recebida: {len(response2)} chars")
            json_critique = _parse_json_response(response2)
            print(f"[CRITIQUE] Retry bem-sucedido: {len(json_critique)} vibes detectadas")
            return json_critique
        except Exception as e2:
            print(f"[CRITIQUE] Retry tambem falhou: {e2}")
            
            # Salvar respostas para debug
            import tempfile
            import os
            debug_dir = tempfile.gettempdir()
            
            response1_path = os.path.join(debug_dir, "critique_response1_failed.txt")
            with open(response1_path, "w", encoding="utf-8") as f:
                f.write(response)
            print(f"[CRITIQUE] Resposta original salva em: {response1_path}")
            print(f"[CRITIQUE] Resposta original (primeiros 500 chars): {response[:500]}...")
            if len(response) > 500:
                print(f"[CRITIQUE] Resposta original (ultimos 500 chars): ...{response[-500:]}")
            
            if 'response2' in locals():
                response2_path = os.path.join(debug_dir, "critique_response2_failed.txt")
                with open(response2_path, "w", encoding="utf-8") as f:
                    f.write(response2)
                print(f"[CRITIQUE] Resposta retry salva em: {response2_path}")
                print(f"[CRITIQUE] Resposta retry (primeiros 500 chars): {response2[:500]}...")
                if len(response2) > 500:
                    print(f"[CRITIQUE] Resposta retry (ultimos 500 chars): ...{response2[-500:]}")
            
            raise ValueError(f"Nao foi possivel parsear JSON da resposta apos 2 tentativas. Verifique os arquivos de debug em {debug_dir}")


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
    
    # Tentativa 3: Extrair JSONs numerados (DeepSeek V3.2 coloca numeros: 1. {...}, 2. {...})
    # Exemplo: "1. {"action": "replace", ...}\n\n2. {"action": "add", ...}"
    try:
        # Regex para encontrar objetos JSON precedidos por numeros opcionais
        numbered_json_pattern = r'(?:^\d+\.\s*)?(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
        matches = re.findall(numbered_json_pattern, response, re.MULTILINE)
        
        if matches and len(matches) >= 3:  # Pelo menos 3 items para ser valido
            parsed_items = []
            for match in matches:
                try:
                    # Sanitizar e parsear cada objeto JSON individualmente
                    sanitized_obj = _json_sanitize(match)
                    obj = json.loads(sanitized_obj)
                    if isinstance(obj, dict) and "action" in obj:
                        parsed_items.append(obj)
                except json.JSONDecodeError:
                    continue
            
            if len(parsed_items) >= 3:
                print(f"[CRITIQUE] JSON extraido de items numerados ({len(parsed_items)} items)")
                return parsed_items
    except Exception as e:
        print(f"[CRITIQUE] Tentativa 3 (numerados) falhou: {e}")
        pass
    
    # Tentativa 4: Buscar array JSON no texto (GPT-5 reasoning: tentar do fim para o inicio)
    # GPT-5 coloca reasoning text ANTES do JSON final, entao o ultimo match eh mais confiavel
    try:
        all_json_matches = list(re.finditer(r'\[\s*\{[\s\S]*?\}\s*\]', response, re.DOTALL))
        # Tentar do ultimo para o primeiro (o ultimo eh mais provavel de ser o output final)
        for match in reversed(all_json_matches):
            try:
                json_str = match.group(0)
                data = json.loads(json_str)
                if isinstance(data, list) and len(data) > 0:
                    # Validar se tem estrutura de lista de objetos (funciona com qualquer esquema)
                    if all(isinstance(item, dict) for item in data):
                        print(f"[CRITIQUE] JSON extraido de reasoning (posicao {match.start()}/{len(response)}, {len(data)} items)")
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
    
    # Tentativa 4a: Buscar JSON array DEPOIS de texto explicativo/reasoning
    # DeepSeek V3.2-Exp tende a gerar: "[reasoning text...] JSON array:"
    # Procurar por "JSON array:" ou "Output:" seguido de [...]
    try:
        # Buscar por padrões que indicam início do JSON
        patterns = [
            r'JSON array:\s*(\[[\s\S]*\])',
            r'NOW GENERATE THE JSON ARRAY:\s*(\[[\s\S]*\])',
            r'Output the JSON array now:\s*(\[[\s\S]*\])',
            r'Here is the JSON:\s*(\[[\s\S]*\])',
            r'The JSON array:\s*(\[[\s\S]*\])',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                json_candidate = match.group(1).strip()
                # Sanitizar e parsear
                json_sanitized = _json_sanitize(json_candidate)
                try:
                    data = json.loads(json_sanitized)
                    if isinstance(data, list) and len(data) >= 3:
                        print(f"[CRITIQUE] JSON extraido apos marcador (pattern: {pattern})")
                        return data
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[CRITIQUE] Tentativa 4a falhou: {e}")
        pass
    
    # Tentativa 4b: Remover texto antes/depois do primeiro [ e ultimo ]
    try:
        first_bracket = response.find('[')
        last_bracket = response.rfind(']')
        if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
            json_str = response[first_bracket:last_bracket+1]
            # Sanitizar antes de parsear
            json_str_sanitized = _json_sanitize(json_str)
            data = json.loads(json_str_sanitized)
            if isinstance(data, list):
                print(f"[CRITIQUE] JSON extraido por brackets (tentativa 4b)")
                return data
    except json.JSONDecodeError as e:
        print(f"[CRITIQUE] Tentativa 4b falhou: {e}")
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
    
    # Tentativa 6: Buscar multiplos objetos JSON individuais (esquema flexivel)
    try:
        # Buscar objetos com action (esquema novo) OU vibe_pattern (esquema antigo)
        objects = re.findall(r'\{[^{}]*("action"|"vibe_pattern")[^{}]*\}', response, re.IGNORECASE)
        if objects:
            print(f"[CRITIQUE] Tentativa 6: encontrados {len(objects)} objetos candidatos")
            parsed_objects = []
            for i, obj_str in enumerate(objects):
                try:
                    # Sanitizar objeto individual
                    obj_sanitized = _json_sanitize(obj_str)
                    obj = json.loads(obj_sanitized)
                    if isinstance(obj, dict):
                        # Aceitar se tem action OU vibe_pattern
                        if "action" in obj or "vibe_pattern" in obj:
                            parsed_objects.append(obj)
                except Exception as e:
                    print(f"[CRITIQUE] Tentativa 6, objeto {i+1} falhou: {e}")
                    continue
            if parsed_objects and len(parsed_objects) >= 3:
                print(f"[CRITIQUE] Tentativa 6: {len(parsed_objects)} objetos parseados com sucesso")
                return parsed_objects
    except Exception as e:
        print(f"[CRITIQUE] Tentativa 6 falhou completamente: {e}")
        pass
    
    # Tentativa 7: Extrair pattern mais agressivo - procurar por linhas com "action"
    try:
        # Procurar linhas que contenham "action": "replace|add|keep"
        lines = response.split('\n')
        objects_rebuilt = []
        current_obj = {}
        
        for line in lines:
            line = line.strip()
            
            # Detectar início de novo objeto
            if '"action"' in line or "'action'" in line:
                # Salvar objeto anterior se existir
                if current_obj and ("action" in current_obj):
                    objects_rebuilt.append(current_obj)
                current_obj = {}
                
                # Extrair action
                action_match = re.search(r'"action"\s*:\s*"(replace|add|keep)"', line, re.IGNORECASE)
                if not action_match:
                    action_match = re.search(r"'action'\s*:\s*'(replace|add|keep)'", line, re.IGNORECASE)
                if action_match:
                    current_obj["action"] = action_match.group(1).lower()
            
            # Extrair from
            if '"from"' in line or "'from'" in line:
                from_match = re.search(r'"from"\s*:\s*"([^"]*)"', line)
                if not from_match:
                    from_match = re.search(r"'from'\s*:\s*'([^']*)'", line)
                if from_match:
                    current_obj["from"] = from_match.group(1)
            
            # Extrair to
            if '"to"' in line or "'to'" in line:
                to_match = re.search(r'"to"\s*:\s*"([^"]*)"', line)
                if not to_match:
                    to_match = re.search(r"'to'\s*:\s*'([^']*)'", line)
                if to_match:
                    current_obj["to"] = to_match.group(1)
            
            # Extrair description
            if '"description"' in line or "'description'" in line:
                desc_match = re.search(r'"description"\s*:\s*"([^"]*)"', line)
                if not desc_match:
                    desc_match = re.search(r"'description'\s*:\s*'([^']*)'", line)
                if desc_match:
                    current_obj["description"] = desc_match.group(1)
        
        # Salvar último objeto
        if current_obj and ("action" in current_obj):
            objects_rebuilt.append(current_obj)
        
        if objects_rebuilt and len(objects_rebuilt) >= 3:
            print(f"[CRITIQUE] Tentativa 7: {len(objects_rebuilt)} objetos reconstruidos linha por linha")
            return objects_rebuilt
    except Exception as e:
        print(f"[CRITIQUE] Tentativa 7 falhou: {e}")
        pass
    
    # Tentativa 8: Tentar recuperar JSON truncado - completar objetos incompletos
    try:
        # Se a resposta e muito curta (< 200 chars), pode estar truncada
        if len(response) < 200:
            print(f"[CRITIQUE] Resposta muito curta ({len(response)} chars), pode estar truncada")
            # Tentar extrair objetos JSON mesmo que incompletos
            objects = re.findall(r'\{"action"\s*:\s*"([^"]+)"[^}]*\}', response)
            if objects:
                print(f"[CRITIQUE] Tentativa 8: encontrados {len(objects)} objetos mesmo com truncamento")
                # Tentar reconstruir objetos minimos
                rebuilt = []
                for obj_match in re.finditer(r'\{"action"\s*:\s*"([^"]+)"', response):
                    obj = {"action": obj_match.group(1)}
                    # Tentar extrair from/to/description do texto ao redor
                    start = obj_match.start()
                    end = min(start + 500, len(response))
                    context = response[start:end]
                    from_match = re.search(r'"from"\s*:\s*"([^"]*)"', context)
                    to_match = re.search(r'"to"\s*:\s*"([^"]*)"', context)
                    desc_match = re.search(r'"description"\s*:\s*"([^"]*)"', context)
                    if from_match:
                        obj["from"] = from_match.group(1)
                    if to_match:
                        obj["to"] = to_match.group(1)
                    if desc_match:
                        obj["description"] = desc_match.group(1)
                    rebuilt.append(obj)
                if rebuilt and len(rebuilt) >= 3:
                    print(f"[CRITIQUE] Tentativa 8: {len(rebuilt)} objetos reconstruidos de resposta truncada")
                    return rebuilt
        
        # Se a resposta tem um JSON array incompleto (sem fechamento), tentar completar
        first_bracket = response.find('[')
        if first_bracket != -1:
            # Verificar se tem fechamento
            last_bracket = response.rfind(']')
            if last_bracket == -1 or last_bracket < first_bracket:
                # JSON incompleto - tentar extrair objetos individuais
                json_content = response[first_bracket:]
                # Tentar extrair objetos JSON completos ou parcialmente completos
                objects = re.findall(r'\{"action"\s*:\s*"([^"]+)"[^}]*\}', json_content)
                if objects:
                    print(f"[CRITIQUE] Tentativa 8: JSON incompleto, extraindo {len(objects)} objetos")
                    rebuilt = []
                    for obj_match in re.finditer(r'\{"action"\s*:\s*"([^"]+)"', json_content):
                        obj = {"action": obj_match.group(1)}
                        start = obj_match.start()
                        end = min(start + 300, len(json_content))
                        obj_str = json_content[start:end]
                        # Tentar extrair campos
                        from_match = re.search(r'"from"\s*:\s*"([^"]*)"', obj_str)
                        to_match = re.search(r'"to"\s*:\s*"([^"]*)"', obj_str)
                        desc_match = re.search(r'"description"\s*:\s*"([^"]*)"', obj_str)
                        if from_match:
                            obj["from"] = from_match.group(1)
                        if to_match:
                            obj["to"] = to_match.group(1)
                        if desc_match:
                            obj["description"] = desc_match.group(1)
                        rebuilt.append(obj)
                    if rebuilt and len(rebuilt) >= 3:
                        print(f"[CRITIQUE] Tentativa 8: {len(rebuilt)} objetos extraidos de JSON incompleto")
                        return rebuilt
    except Exception as e:
        print(f"[CRITIQUE] Tentativa 8 falhou: {e}")
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
    
    # Validacao flexivel: aceita tanto esquema antigo (vibe_pattern) quanto novo (action/from/to)
    for item in data:
        if not isinstance(item, dict):
            return False
        
        # Esquema novo: action/from/to
        if "action" in item:
            if item["action"] not in ["replace", "add", "keep"]:
                return False
            continue
        
        # Esquema antigo: vibe_pattern/vibe_description (compatibilidade legada)
        if "vibe_pattern" in item:
            if item["vibe_pattern"] not in ["do", "don't"]:
                return False
            if "vibe_description" not in item:
                return False
            if not isinstance(item["vibe_description"], str) or not item["vibe_description"].strip():
                return False
            continue
        
        # Se nao tem nenhum dos campos esperados, invalido
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

