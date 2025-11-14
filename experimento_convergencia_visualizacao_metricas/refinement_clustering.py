#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo de clustering para historias humanas.

Agrupa historias similares semanticamente para focar o refinamento
em um estilo especifico, evitando feedback conflitante.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import numpy as np

from experiment_iterativo import embed_texts, cosine_distance


def cluster_human_ideas(
    human_ideas: List[str],
    embedder,
    method: str = "kmeans",
    n_clusters: int = 4,
    distance_threshold: float = 0.3,
) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    Clusteriza historias humanas com base em embeddings semanticos.
    
    Args:
        human_ideas: Lista de historias humanas (strings)
        embedder: Modelo de embeddings (do get_embedder)
        method: Metodo de clustering ("kmeans" ou "agglomerative")
        n_clusters: Numero de clusters (para kmeans)
        distance_threshold: Threshold de distancia (para agglomerative)
    
    Returns:
        labels: Lista de cluster IDs para cada historia (0, 1, 2, ...)
        clusters_dict: Dicionario {cluster_id: [indices das historias]}
    
    Example:
        >>> embedder = get_embedder("all-MiniLM-L6-v2")
        >>> ideas = ["historia 1...", "historia 2...", "historia 3..."]
        >>> labels, clusters = cluster_human_ideas(ideas, embedder, method="kmeans", n_clusters=2)
        >>> print(clusters)
        {0: [0, 2], 1: [1]}  # Cluster 0 tem historias 0 e 2, Cluster 1 tem historia 1
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    
    print(f"[CLUSTERING] Gerando embeddings de {len(human_ideas)} historias...")
    embeddings = embed_texts(embedder, human_ideas)
    
    print(f"[CLUSTERING] Aplicando {method.upper()}...")
    
    if method == "kmeans":
        # KMeans: usuario define numero de clusters
        n_clusters = min(n_clusters, len(human_ideas))  # Nao pode ter mais clusters que historias
        
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(embeddings)
        
        print(f"[CLUSTERING] KMeans com k={n_clusters} clusters")
    
    elif method == "agglomerative":
        # Agglomerative: threshold de distancia define numero de clusters
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average"
        )
        labels = clusterer.fit_predict(embeddings)
        
        n_clusters_found = len(np.unique(labels))
        print(f"[CLUSTERING] Agglomerative com threshold={distance_threshold:.2f} encontrou {n_clusters_found} clusters")
    
    else:
        raise ValueError(f"Metodo {method} nao suportado. Use 'kmeans' ou 'agglomerative'")
    
    # Criar dicionario de clusters
    clusters_dict = {}
    for cluster_id in np.unique(labels):
        indices = [i for i, label in enumerate(labels) if label == cluster_id]
        clusters_dict[int(cluster_id)] = indices
    
    # Mostrar resumo
    print(f"\n[CLUSTERING] Resumo dos clusters:")
    for cluster_id, indices in sorted(clusters_dict.items()):
        print(f"  Cluster {cluster_id}: {len(indices)} historias (indices: {indices})")
    print()
    
    return labels.tolist(), clusters_dict


def select_cluster_representatives(
    human_ideas: List[str],
    embedder,
    cluster_id: int,
    labels: List[int],
    min_size: Optional[int] = None,
) -> List[str]:
    """
    Seleciona historias de um cluster especifico.
    
    Se o cluster tiver menos de min_size historias, completa com as historias
    mais proximas ao centroide do cluster (de outros clusters).
    
    Args:
        human_ideas: Lista completa de historias humanas
        embedder: Modelo de embeddings
        cluster_id: ID do cluster a selecionar
        labels: Labels de cluster para cada historia (do cluster_human_ideas)
        min_size: Tamanho minimo do cluster (None = sem expansao)
    
    Returns:
        Lista de historias do cluster selecionado (expandido se necessario)
    
    Example:
        >>> labels, clusters = cluster_human_ideas(ideas, embedder, n_clusters=3)
        >>> cluster_0_ideas = select_cluster_representatives(ideas, embedder, cluster_id=0, labels=labels, min_size=5)
        >>> print(len(cluster_0_ideas))
        5  # Expandido para 5 historias (se originalmente tinha < 5)
    """
    # Filtrar historias do cluster
    cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
    selected_ideas = [human_ideas[i] for i in cluster_indices]
    
    original_size = len(selected_ideas)
    print(f"[CLUSTERING] Cluster {cluster_id}: {original_size} historias")
    
    # Se min_size especificado e cluster pequeno, expandir
    if min_size and original_size < min_size:
        print(f"[CLUSTERING] Expandindo cluster {cluster_id} de {original_size} para {min_size} historias...")
        
        # Gerar embeddings de todas as historias
        all_embeddings = embed_texts(embedder, human_ideas)
        
        # Calcular centroide do cluster
        cluster_embeddings = all_embeddings[cluster_indices]
        centroid = cluster_embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # Normalizar
        
        # Calcular distancias de TODAS as historias ao centroide
        distances_to_centroid = [
            (i, cosine_distance(emb, centroid))
            for i, emb in enumerate(all_embeddings)
        ]
        
        # Ordenar por distancia (mais proximas primeiro)
        distances_to_centroid.sort(key=lambda x: x[1])
        
        # Pegar as min_size historias mais proximas (incluindo as ja no cluster)
        expanded_indices = [idx for idx, _ in distances_to_centroid[:min_size]]
        
        # Atualizar lista de historias
        selected_ideas = [human_ideas[i] for i in expanded_indices]
        
        # Mostrar quais foram adicionadas
        added_indices = [i for i in expanded_indices if i not in cluster_indices]
        if added_indices:
            print(f"[CLUSTERING] Adicionadas {len(added_indices)} historias vizinhas: indices {added_indices}")
            print(f"[CLUSTERING] Cluster expandido: {original_size} -> {len(selected_ideas)} historias")
    
    return selected_ideas


def analyze_cluster_diversity(
    human_ideas: List[str],
    embedder,
    labels: List[int],
    clusters_dict: Dict[int, List[int]],
) -> Dict[int, Dict[str, float]]:
    """
    Analisa a diversidade intra-cluster e inter-cluster.
    
    Metricas:
    - Coesao (intra-cluster): Quao proximas as historias dentro do cluster estao
    - Separacao (inter-cluster): Quao distantes os clusters estao uns dos outros
    
    Args:
        human_ideas: Lista de historias humanas
        embedder: Modelo de embeddings
        labels: Labels de cluster
        clusters_dict: Dicionario de clusters
    
    Returns:
        Dicionario com metricas por cluster:
        {
            cluster_id: {
                "size": int,
                "cohesion": float,  # Distancia media intra-cluster (menor = mais coeso)
                "centroid": np.ndarray,
            }
        }
    """
    print(f"[CLUSTERING] Analisando diversidade dos clusters...")
    
    # Gerar embeddings
    embeddings = embed_texts(embedder, human_ideas)
    
    cluster_stats = {}
    
    for cluster_id, indices in clusters_dict.items():
        cluster_embeddings = embeddings[indices]
        
        # Centroide do cluster
        centroid = cluster_embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # Normalizar
        
        # Coesao: distancia media das historias ao centroide
        cohesion = np.mean([
            cosine_distance(emb, centroid)
            for emb in cluster_embeddings
        ])
        
        cluster_stats[cluster_id] = {
            "size": len(indices),
            "cohesion": float(cohesion),
            "centroid": centroid,
        }
    
    # Mostrar resumo
    print(f"\n[CLUSTERING] Metricas de diversidade:")
    for cluster_id, stats in sorted(cluster_stats.items()):
        print(f"  Cluster {cluster_id}:")
        print(f"    - Tamanho: {stats['size']} historias")
        print(f"    - Coesao: {stats['cohesion']:.4f} (menor = mais coeso)")
    
    # Calcular separacao inter-cluster
    if len(cluster_stats) > 1:
        print(f"\n[CLUSTERING] Separacao entre clusters (distancia entre centroides):")
        cluster_ids = sorted(cluster_stats.keys())
        for i, id1 in enumerate(cluster_ids):
            for id2 in cluster_ids[i+1:]:
                dist = cosine_distance(
                    cluster_stats[id1]["centroid"],
                    cluster_stats[id2]["centroid"]
                )
                print(f"    Cluster {id1} <-> Cluster {id2}: {dist:.4f}")
    
    print()
    
    return cluster_stats


def get_best_cluster(
    clusters_dict: Dict[int, List[int]],
    cluster_stats: Dict[int, Dict[str, float]],
    criterion: str = "largest",
) -> int:
    """
    Seleciona o melhor cluster baseado em um criterio.
    
    Args:
        clusters_dict: Dicionario de clusters
        cluster_stats: Estatisticas dos clusters (do analyze_cluster_diversity)
        criterion: Criterio de selecao:
            - "largest": Maior cluster (mais historias)
            - "most_cohesive": Cluster mais coeso (menor distancia intra-cluster)
            - "balanced": Balanco entre tamanho e coesao
    
    Returns:
        ID do melhor cluster
    
    Example:
        >>> best_id = get_best_cluster(clusters_dict, cluster_stats, criterion="largest")
        >>> print(f"Melhor cluster: {best_id}")
    """
    if criterion == "largest":
        # Maior cluster
        best_id = max(clusters_dict.keys(), key=lambda k: len(clusters_dict[k]))
        print(f"[CLUSTERING] Melhor cluster (criterion=largest): {best_id} ({len(clusters_dict[best_id])} historias)")
    
    elif criterion == "most_cohesive":
        # Cluster mais coeso (menor cohesion)
        best_id = min(cluster_stats.keys(), key=lambda k: cluster_stats[k]["cohesion"])
        print(f"[CLUSTERING] Melhor cluster (criterion=most_cohesive): {best_id} (coesao={cluster_stats[best_id]['cohesion']:.4f})")
    
    elif criterion == "balanced":
        # Balanco: normalizar tamanho e coesao, maximizar (tamanho / coesao)
        # Penaliza fortemente clusters muito pequenos (1-2 historias)
        max_size = max(stats["size"] for stats in cluster_stats.values())
        max_cohesion = max(stats["cohesion"] for stats in cluster_stats.values())
        
        scores = {}
        for cluster_id, stats in cluster_stats.items():
            size = stats["size"]
            
            # Penalidade exponencial para clusters pequenos
            if size == 1:
                size_penalty = 0.01  # Cluster com 1 historia: penalidade massiva
            elif size == 2:
                size_penalty = 0.25  # Cluster com 2 historias: penalidade media
            else:
                size_penalty = 1.0   # Cluster com 3+ historias: sem penalidade
            
            # Score: (tamanho normalizado * penalidade) / (coesao normalizada + epsilon)
            size_norm = stats["size"] / max_size
            cohesion_norm = stats["cohesion"] / max_cohesion if max_cohesion > 0 else 0
            scores[cluster_id] = (size_norm * size_penalty) / (cohesion_norm + 0.01)
        
        best_id = max(scores.keys(), key=lambda k: scores[k])
        print(f"[CLUSTERING] Melhor cluster (criterion=balanced): {best_id} (tamanho={cluster_stats[best_id]['size']}, score={scores[best_id]:.2f})")
    
    else:
        raise ValueError(f"Criterio {criterion} nao suportado")
    
    return best_id

