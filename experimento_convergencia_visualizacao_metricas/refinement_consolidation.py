#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de consolidação semântica de ideias.

Agrupa ideias semanticamente similares e consolida cada grupo
usando LLM para preservar informação e reduzir redundância.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from experiment_iterativo import embed_texts, cosine_distance
from bleu_minimal_deepseek import call_deepseek


def cluster_by_similarity(
    ideas: List[str],
    embeddings: np.ndarray,
    threshold: float = 0.85
) -> List[List[int]]:
    """
    Agrupa ideias por similaridade semântica.
    
    Args:
        ideas: Lista de ideias (strings)
        embeddings: Embeddings das ideias (numpy array)
        threshold: Limiar de similaridade (0.85 = 85% similar = agrupar)
    
    Returns:
        Lista de grupos, onde cada grupo é uma lista de índices
    
    Example:
        >>> ideas = ["Ideia A", "Ideia B similar a A", "Ideia C única"]
        >>> embeddings = embed_texts(embedder, ideas)
        >>> groups = cluster_by_similarity(ideas, embeddings, threshold=0.85)
        >>> print(groups)
        [[0, 1], [2]]  # Grupo 1: ideias 0 e 1 (similares), Grupo 2: ideia 2 (única)
    """
    n = len(ideas)
    if n <= 1:
        return [[i] for i in range(n)]
    
    groups = []
    used = set()
    
    for i in range(n):
        if i in used:
            continue
        
        # Criar novo grupo com ideia i
        group = [i]
        used.add(i)
        
        # Encontrar ideias similares a i
        for j in range(i + 1, n):
            if j in used:
                continue
            
            # Calcular similaridade
            similarity = 1.0 - cosine_distance(embeddings[i], embeddings[j])
            
            if similarity >= threshold:
                group.append(j)
                used.add(j)
                print(f"[CONSOLIDATION] Ideias {i} e {j} agrupadas (similaridade={similarity:.3f})")
        
        groups.append(group)
    
    print(f"[CONSOLIDATION] Formados {len(groups)} grupos de {n} ideias")
    return groups


def llm_consolidate_group(
    ideas: List[str],
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 1000,
    api_key_override: str = None,
    reasoning_effort: str = None,
) -> str:
    """
    Consolida um grupo de ideias similares em uma única ideia usando LLM.
    
    Args:
        ideas: Lista de ideias similares a consolidar
        model: Modelo LLM a usar
        temperature: Temperatura (0.3 = mais conservador)
        max_tokens: Máximo de tokens
        api_key_override: API key alternativa
        reasoning_effort: Reasoning effort
    
    Returns:
        Ideia consolidada (string)
    
    Example:
        >>> ideas = [
        ...     "Sarah, PR partner, meets James, CFO, at gala",
        ...     "Maria, marketing director, meets John, tech exec, at event"
        ... ]
        >>> consolidated = llm_consolidate_group(ideas)
        >>> print(consolidated)
        "Sarah, a PR partner, meets James, a CFO, at a professional gala..."
    """
    if len(ideas) == 1:
        # Ideia única, não precisa consolidar
        return ideas[0]
    
    # Montar prompt de consolidação
    ideas_text = "\n\n".join([f"IDEA {i+1}:\n{idea}" for i, idea in enumerate(ideas)])
    
    prompt = f"""You are given {len(ideas)} similar story ideas that share common themes or patterns.

Your task: Consolidate these ideas into ONE story idea that:
1. Captures the BEST elements from each idea
2. Preserves specific details (character names, occupations, settings, locations)
3. Maintains emotional depth and narrative complexity
4. Combines complementary aspects (if one focuses on career and another on relationships, combine both)

CRITICAL RULES:
- The consolidated idea should be RICHER and MORE SPECIFIC than any individual idea
- NOT a generic summary - include concrete details (names, places, occupations)
- Maintain the narrative structure and tone
- If ideas have different focuses, integrate them (e.g., career + relationships)
- Length: similar to the original ideas (~100-200 words)

IDEAS TO CONSOLIDATE:
{ideas_text}

Now write the CONSOLIDATED IDEA (do not add labels, just write the idea directly):
"""
    
    # Chamar LLM
    print(f"[CONSOLIDATION] Consolidando {len(ideas)} ideias com {model}...")
    
    try:
        response = call_deepseek(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key_override=api_key_override,
            reasoning_effort=reasoning_effort,
            exclude_reasoning=None,  # Auto-detecção
        )
        
        consolidated = response.strip()
        
        # Validação básica
        if not consolidated or len(consolidated) < 50:
            print(f"[CONSOLIDATION] AVISO: Consolidação muito curta, usando primeira ideia como fallback")
            return ideas[0]
        
        print(f"[CONSOLIDATION] Consolidação bem-sucedida ({len(consolidated)} caracteres)")
        return consolidated
        
    except Exception as e:
        print(f"[CONSOLIDATION] ERRO ao consolidar: {e}")
        print(f"[CONSOLIDATION] Fallback: usando primeira ideia do grupo")
        return ideas[0]


def consolidate_similar_ideas(
    ideas: List[str],
    embedder,
    model: str = "gpt-4o-mini",
    threshold: float = 0.85,
    max_group_size: int = 4,
    temperature: float = 0.3,
    max_tokens: int = 1000,
    api_key_override: str = None,
    reasoning_effort: str = None,
) -> Tuple[List[str], dict]:
    """
    Agrupa e consolida ideias semanticamente similares.
    
    Args:
        ideas: Lista de ideias
        embedder: Modelo de embeddings
        model: Modelo LLM para consolidação
        threshold: Limiar de similaridade (0.85 = 85% similar = agrupar)
        max_group_size: Tamanho máximo de grupo (default: 4)
        temperature: Temperatura para LLM
        max_tokens: Máximo de tokens
        api_key_override: API key alternativa
        reasoning_effort: Reasoning effort
    
    Returns:
        Tupla (ideias_consolidadas, metadata)
        - ideias_consolidadas: Lista de ideias após consolidação
        - metadata: Dict com estatísticas (num_groups, num_consolidated, etc.)
    
    Example:
        >>> ideas = ["Ideia A", "Ideia B similar a A", "Ideia C única"]
        >>> consolidated, meta = consolidate_similar_ideas(ideas, embedder)
        >>> print(len(consolidated))
        2  # 1 consolidada (A+B) + 1 única (C)
        >>> print(meta)
        {'num_groups': 2, 'num_consolidated': 1, 'num_unique': 1}
    """
    if len(ideas) <= 1:
        return ideas, {'num_groups': len(ideas), 'num_consolidated': 0, 'num_unique': len(ideas)}
    
    print(f"\n[CONSOLIDATION] Iniciando consolidação de {len(ideas)} ideias")
    print(f"[CONSOLIDATION] Threshold: {threshold}, Max group size: {max_group_size}")
    
    # Fase 1: Gerar embeddings
    print(f"[CONSOLIDATION] Gerando embeddings...")
    embeddings = embed_texts(embedder, ideas)
    
    # Fase 2: Agrupar por similaridade
    print(f"[CONSOLIDATION] Agrupando por similaridade...")
    groups = cluster_by_similarity(ideas, embeddings, threshold)
    
    # Fase 3: Consolidar cada grupo
    consolidated_ideas = []
    num_consolidated = 0
    num_unique = 0
    
    for group_idx, group in enumerate(groups):
        if len(group) == 1:
            # Ideia única, não precisa consolidar
            consolidated_ideas.append(ideas[group[0]])
            num_unique += 1
            print(f"[CONSOLIDATION] Grupo {group_idx + 1}: 1 ideia (única, preservada)")
        elif len(group) > max_group_size:
            # Grupo muito grande: consolidar em subgrupos ou aumentar tokens
            print(f"[CONSOLIDATION] AVISO: Grupo {group_idx + 1} muito grande ({len(group)} ideias > {max_group_size})")
            
            # Estrategia: dividir em subgrupos de max_group_size e consolidar cada um
            # Depois consolidar os resultados finais em uma única ideia
            group_ideas = [ideas[i] for i in group]
            
            # Dividir em subgrupos
            subconsolidated = []
            for sub_idx in range(0, len(group_ideas), max_group_size):
                subgroup = group_ideas[sub_idx:sub_idx + max_group_size]
                print(f"[CONSOLIDATION] Subgrupo {sub_idx//max_group_size + 1}: consolidando {len(subgroup)} ideias...")
                
                sub_consolidated = llm_consolidate_group(
                    ideas=subgroup,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key_override=api_key_override,
                    reasoning_effort=reasoning_effort,
                )
                subconsolidated.append(sub_consolidated)
            
            # Se gerou mais de 1 subconsolidado, consolidar novamente
            if len(subconsolidated) > 1:
                print(f"[CONSOLIDATION] Consolidando {len(subconsolidated)} subconsolidados em uma única ideia...")
                # Aumentar tokens para grupo maior
                final_consolidated = llm_consolidate_group(
                    ideas=subconsolidated,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens * 2,  # Mais tokens para grupo maior
                    api_key_override=api_key_override,
                    reasoning_effort=reasoning_effort,
                )
                consolidated_ideas.append(final_consolidated)
            else:
                consolidated_ideas.append(subconsolidated[0])
            
            num_consolidated += 1
        else:
            # Consolidar grupo (tamanho normal: 2-4 ideias)
            group_ideas = [ideas[i] for i in group]
            print(f"[CONSOLIDATION] Grupo {group_idx + 1}: {len(group)} ideias (consolidando...)")
            
            consolidated = llm_consolidate_group(
                ideas=group_ideas,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key_override=api_key_override,
                reasoning_effort=reasoning_effort,
            )
            
            consolidated_ideas.append(consolidated)
            num_consolidated += 1
    
    # Metadata
    metadata = {
        'num_original': len(ideas),
        'num_groups': len(groups),
        'num_consolidated': num_consolidated,
        'num_unique': num_unique,
        'num_final': len(consolidated_ideas),
        'reduction': len(ideas) - len(consolidated_ideas),
    }
    
    print(f"\n[CONSOLIDATION] Consolidação concluída:")
    print(f"  - Ideias originais: {metadata['num_original']}")
    print(f"  - Grupos formados: {metadata['num_groups']}")
    print(f"  - Grupos consolidados: {metadata['num_consolidated']}")
    print(f"  - Ideias únicas: {metadata['num_unique']}")
    print(f"  - Ideias finais: {metadata['num_final']}")
    print(f"  - Redução: {metadata['reduction']} ideias\n")
    
    return consolidated_ideas, metadata

