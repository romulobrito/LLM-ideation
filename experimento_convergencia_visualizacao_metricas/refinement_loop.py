#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loop principal de refinamento iterativo.

Orquestra as 3 etapas (CRITIQUE, PACKING, GENERATION) em um loop
iterativo, monitorando convergencia por distancia coseno.

Requer: OPENAI_API_KEY ou OPENROUTER_API_KEY no ambiente.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np

from experiment_iterativo import get_embedder, embed_texts, cosine_distance
from refinement_critique import critique_step
from refinement_packing import packing_step
from refinement_generation import generation_step
from refinement_north import generate_north_star, format_north_with_tactical
from refinement_clustering import (
    cluster_human_ideas,
    select_cluster_representatives,
    analyze_cluster_diversity,
    get_best_cluster,
)


@dataclass
class IterationResult:
    """Resultado de uma iteracao do refinamento."""
    iteration: int
    critique_json: List[Dict[str, str]]
    bullets: str
    generated_ideas: List[str]
    avg_distance: float
    min_distance: float
    timestamp: str
    individual_distances: Optional[List[float]] = None  # Distancias individuais de cada ideia
    top3_mean_distance: Optional[float] = None          # Media das 3 melhores
    centroid_distance: Optional[float] = None           # Distancia media ao centroide humano
    centroid_to_centroid: Optional[float] = None        # Distancia entre centroides (LLM vs humano)
    distance_from_iter1: Optional[float] = None         # Distancia do centroide desta iter ao centroide da iter 1
    
    def to_dict(self) -> Dict:
        """Converte para dicionario."""
        return asdict(self)


@dataclass
class RefinementConfig:
    """Configuracao do refinement loop."""
    invitation: str
    directive: str
    human_ideas: List[str]
    model: str = "gpt-4o-mini"
    embedder_name: str = "all-MiniLM-L6-v2"
    device: str = "auto"
    max_iterations: int = 5
    patience: int = 3
    delta_threshold: float = 0.01
    num_ideas_per_iter: int = 5
    temperature: float = 0.8
    max_tokens: int = 4000
    api_key_override: Optional[str] = None
    reasoning_effort: Optional[str] = None
    output_dir: Optional[Path] = None
    use_north_star: bool = True  # NOVO: Usar norte fixo automatico
    north_star_model: str = "gpt-4o"  # NOVO: Modelo para gerar norte (mais preciso)
    # Parametros de clustering
    use_clustering: bool = False  # Se True, usa clustering; se False, usa human_ideas diretamente
    all_human_ideas: Optional[List[str]] = None  # TODAS as historias disponiveis (para clustering)
    clustering_method: str = "kmeans"  # "kmeans" ou "agglomerative"
    n_clusters: int = 4  # Numero de clusters (para kmeans)
    distance_threshold: float = 0.3  # Threshold de distancia (para agglomerative)
    selected_cluster_id: Optional[int] = None  # ID do cluster selecionado (None = auto-select)
    min_cluster_size: int = 5  # Tamanho minimo do cluster (expandir com vizinhos se menor)
    # Parametros de otimizacao e convergencia
    optimize_metric: str = "top3_mean"  # Metrica para convergencia: "avg", "min", "top3_mean", "centroid", "centroid_to_centroid"


class RefinementLoop:
    """
    Loop principal de refinamento iterativo.
    
    Executa as 3 etapas em loop:
    1. CRITIQUE: Analisa vibes e gera JSON
    2. PACKING: Consolida JSON em bullets
    3. GENERATION: Gera novas ideias com diretiva revisada
    
    Monitora convergencia por distancia coseno entre ideias geradas e humanas.
    """
    
    def __init__(self, config: RefinementConfig):
        """
        Inicializa o loop de refinamento.
        
        Args:
            config: Configuracao do refinamento
        """
        self.config = config
        self.embedder = None
        self.human_embeddings = None
        self.results: List[IterationResult] = []
        self.converged = False
        self.convergence_reason = ""
        
        # Informacoes de clustering (NOVO)
        self.cluster_labels: Optional[List[int]] = None
        self.clusters_dict: Optional[Dict[int, List[int]]] = None
        self.selected_cluster_id: Optional[int] = None
        self.all_human_ideas: Optional[List[str]] = None  # Todas historias antes de clustering
        self.iter1_centroid: Optional[np.ndarray] = None  # Centroide da iteracao 1 (baseline)
        
        print("[LOOP] Inicializando RefinementLoop...")
        self._initialize_embedder()
        self._embed_human_ideas()
        
        # Criar diretorio com timestamp para historico
        if self.config.output_dir:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_dir = self.config.output_dir / timestamp
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            print(f"[LOOP] Diretorio do experimento: {self.experiment_dir}")
        else:
            self.experiment_dir = None
    
    def _initialize_embedder(self) -> None:
        """Inicializa o modelo de embeddings."""
        print(f"[LOOP] Carregando embedder: {self.config.embedder_name}")
        
        # Determinar device (apenas para Sentence Transformers locais)
        device = self.config.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        
        # Usar get_embedder() que suporta tanto local quanto OpenAI
        self.embedder = get_embedder(self.config.embedder_name, device=device)
        
        # Log do tipo de embedder
        if isinstance(self.embedder, str) and "text-embedding" in self.embedder:
            print(f"[LOOP] Embedder OpenAI carregado: {self.embedder}")
        else:
            print(f"[LOOP] Embedder local carregado em: {device}")
    
    def _embed_human_ideas(self) -> None:
        """Computa embeddings das ideias humanas."""
        print(f"[LOOP] Computando embeddings de {len(self.config.human_ideas)} ideias humanas...")
        
        # Usar embed_texts() que suporta tanto local quanto OpenAI
        self.human_embeddings = embed_texts(self.embedder, self.config.human_ideas)
        
        print(f"[LOOP] Embeddings humanos: shape {self.human_embeddings.shape}")
    
    def run(self) -> List[IterationResult]:
        """
        Executa o loop de refinamento completo.
        
        Returns:
            Lista de resultados de cada iteracao
        """
        print("\n" + "="*60)
        print("INICIANDO REFINEMENT LOOP")
        print("="*60 + "\n")
        
        # NOVO: CLUSTERING (se ativado)
        cluster_labels = None
        clusters_dict = None
        cluster_stats = None
        
        # Salvar historias originais antes de clustering
        if self.config.use_clustering:
            self.all_human_ideas = self.config.all_human_ideas.copy()
        
        if self.config.use_clustering:
            print("[LOOP] Modo: CLUSTERING ATIVADO")
            print("="*60)
            
            # Validar configuracao
            if self.config.all_human_ideas is None or len(self.config.all_human_ideas) == 0:
                raise ValueError(
                    "use_clustering=True requer all_human_ideas preenchido com todas as historias disponiveis"
                )
            
            # 1. Clusterizar
            print(f"[LOOP] Clusterizando {len(self.config.all_human_ideas)} historias humanas...")
            cluster_labels, clusters_dict = cluster_human_ideas(
                human_ideas=self.config.all_human_ideas,
                embedder=self.embedder,
                method=self.config.clustering_method,
                n_clusters=self.config.n_clusters,
                distance_threshold=self.config.distance_threshold,
            )
            
            # 2. Analisar diversidade
            cluster_stats = analyze_cluster_diversity(
                human_ideas=self.config.all_human_ideas,
                embedder=self.embedder,
                labels=cluster_labels,
                clusters_dict=clusters_dict,
            )
            
            # 3. Selecionar cluster
            if self.config.selected_cluster_id is None:
                # Auto-selecionar: maior cluster (mais historias)
                selected_cluster_id = get_best_cluster(
                    clusters_dict=clusters_dict,
                    cluster_stats=cluster_stats,
                    criterion="largest",
                )
                print(f"[LOOP] Auto-selecionado cluster {selected_cluster_id} (criterion=largest)")
            else:
                selected_cluster_id = self.config.selected_cluster_id
                print(f"[LOOP] Usando cluster pre-selecionado: {selected_cluster_id}")
            
            # 4. Extrair historias do cluster (com expansao se necessario)
            cluster_ideas = select_cluster_representatives(
                human_ideas=self.config.all_human_ideas,
                embedder=self.embedder,
                cluster_id=selected_cluster_id,
                labels=cluster_labels,
                min_size=self.config.min_cluster_size,
            )
            
            # 5. Atualizar config com historias do cluster
            print(f"[LOOP] Atualizando human_ideas: {len(self.config.human_ideas)} -> {len(cluster_ideas)} historias (cluster {selected_cluster_id})")
            self.config.human_ideas = cluster_ideas
            
            # Recomputar embeddings com as novas historias
            self._embed_human_ideas()
            
            # 6. Salvar informacoes de clustering nas variaveis de instancia (NOVO)
            self.cluster_labels = cluster_labels
            self.clusters_dict = clusters_dict
            self.selected_cluster_id = selected_cluster_id
            
            print("="*60)
            print(f"[LOOP] CLUSTERING COMPLETO: Usando {len(cluster_ideas)} historias do cluster {selected_cluster_id}")
            print("="*60 + "\n")
        else:
            print(f"[LOOP] Modo: SEM CLUSTERING (usando {len(self.config.human_ideas)} historias fornecidas)\n")
        
        # NOVO: Gerar norte fixo UMA VEZ antes do loop
        north_star = None
        if self.config.use_north_star:
            print("[LOOP] Modo: NORTE FIXO + Correcoes Taticas")
            print("[LOOP] Gerando norte fixo das ideias humanas...")
            north_star = generate_north_star(
                invitation=self.config.invitation,
                directive=self.config.directive,
                human_ideas=self.config.human_ideas,
                model=self.config.north_star_model,
                temperature=0.3,  # Conservador para analise
                max_tokens=1000,
                api_key_override=self.config.api_key_override,
                reasoning_effort=None,  # Nao precisa reasoning para analise
            )
            print(f"\n[LOOP] NORTE FIXO GERADO:")
            print("-" * 60)
            print(north_star)
            print("-" * 60 + "\n")
        else:
            print("[LOOP] Modo: Feedback 100% dinamico (sem norte fixo)")
        
        # Ideias iniciais da LLM
        current_llm_ideas = self._generate_initial_ideas()
        all_generated_ideas = list(current_llm_ideas)  # NOVO: Acumula histórico
        
        no_improvement_count = 0
        best_avg_distance = float("inf")
        
        for iteration in range(1, self.config.max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"ITERACAO {iteration}/{self.config.max_iterations}")
            print(f"{'='*60}\n")
            
            # ETAPA 1: CRITIQUE
            print(f"[LOOP] Etapa 1/3: CRITIQUE")
            # CORREÇÃO: Usar TODAS as ideias geradas até agora, não apenas as últimas
            # Limitar a 10 ideias mais recentes para evitar prompt muito longo
            # (DeepSeek V3.2-Exp tem problemas com prompts >8K tokens)
            critique_llm_ideas = all_generated_ideas[-10:] if len(all_generated_ideas) > 10 else all_generated_ideas
            print(f"[LOOP] Critique usando {len(critique_llm_ideas)} ideias acumuladas (max 10 para evitar overflow)")
            
            critique_json = critique_step(
                invitation=self.config.invitation,
                directive=self.config.directive,
                human_ideas=self.config.human_ideas,
                llm_ideas=critique_llm_ideas,
                model=self.config.model,
                temperature=0.3,  # REDUZIDO: 0.7 → 0.3 para análise mais consistente
                max_tokens=self.config.max_tokens,
                api_key_override=self.config.api_key_override,
                reasoning_effort=self.config.reasoning_effort,
            )
            
            # ETAPA 2: PACKING
            print(f"[LOOP] Etapa 2/3: PACKING")
            tactical_bullets = packing_step(
                critique_json=critique_json,
                model=self.config.model,
                temperature=0.5,
                max_tokens=1000,
                api_key_override=self.config.api_key_override,
                reasoning_effort=self.config.reasoning_effort,
            )
            
            # NOVO: Combinar norte fixo + bullets taticos
            if self.config.use_north_star and north_star:
                bullets = format_north_with_tactical(north_star, tactical_bullets)
                print(f"[LOOP] Usando NORTE FIXO + correções táticas")
            else:
                bullets = tactical_bullets
                print(f"[LOOP] Usando feedback 100% dinâmico")
            
            # ETAPA 3: GENERATION
            print(f"[LOOP] Etapa 3/3: GENERATION")
            # MUDANÇA: Removido few-shot learning (human_examples=None)
            # Com clustering, o Norte Fixo já fornece diretrizes específicas do cluster
            # Few-shot poderia causar overfitting ou vazamento de informação
            new_ideas = generation_step(
                invitation=self.config.invitation,
                directive=self.config.directive,
                bullets=bullets,
                num_ideas=self.config.num_ideas_per_iter,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key_override=self.config.api_key_override,
                reasoning_effort=self.config.reasoning_effort,
                human_examples=None,  # ZERO-SHOT: sem exemplos humanos no prompt
            )
            
            # Calcular distancias (todas as metricas)
            avg_dist, min_dist, individual_dists, top3_mean, centroid_dist, c2c_dist, dist_from_iter1 = self._compute_distances(new_ideas)
            
            # Criar resultado
            result = IterationResult(
                iteration=iteration,
                critique_json=critique_json,
                bullets=bullets,
                generated_ideas=new_ideas,
                avg_distance=avg_dist,
                min_distance=min_dist,
                individual_distances=individual_dists,
                top3_mean_distance=top3_mean,
                centroid_distance=centroid_dist,
                centroid_to_centroid=c2c_dist,
                distance_from_iter1=dist_from_iter1,
                timestamp=datetime.now().isoformat(),
            )
            self.results.append(result)
            
            # Salvar resultado
            if self.config.output_dir:
                self._save_iteration(result)
            
            # Mostrar metricas (TODAS)
            print(f"\n[LOOP] Metricas da iteracao {iteration}:")
            print(f"  - Distancia media:        {avg_dist:.4f}")
            print(f"  - Distancia minima:       {min_dist:.4f}")
            print(f"  - Top-3 media:            {top3_mean:.4f}")
            print(f"  - Distancia centroide:    {centroid_dist:.4f}")
            print(f"  - Centroid-to-Centroid:   {c2c_dist:.4f}")
            print(f"  - Distancia da Iter 1:    {dist_from_iter1:.4f}")
            print(f"  - Ideias geradas:         {len(new_ideas)}")
            
            # Selecionar metrica de otimizacao
            if self.config.optimize_metric == "avg":
                current_metric = avg_dist
            elif self.config.optimize_metric == "min":
                current_metric = min_dist
            elif self.config.optimize_metric == "top3_mean":
                current_metric = top3_mean
            elif self.config.optimize_metric == "centroid":
                current_metric = centroid_dist
            elif self.config.optimize_metric == "centroid_to_centroid":
                current_metric = c2c_dist
            else:
                current_metric = avg_dist  # Fallback
            
            print(f"  >>> Metrica de otimizacao ({self.config.optimize_metric}): {current_metric:.4f}")
            
            # Verificar convergencia baseado na metrica selecionada
            improvement = best_avg_distance - current_metric
            
            if current_metric < best_avg_distance - self.config.delta_threshold:
                # Houve melhoria significativa
                best_avg_distance = current_metric
                no_improvement_count = 0
                print(f"[LOOP] ✅ Melhoria detectada: {improvement:.4f}")
            else:
                # Sem melhoria significativa
                no_improvement_count += 1
                
                # Verificar se está piorando ou estagnando
                if avg_dist > best_avg_distance:
                    print(f"[LOOP] ⚠️ Piorou: +{avg_dist - best_avg_distance:.4f} ({no_improvement_count}/{self.config.patience})")
                else:
                    print(f"[LOOP] ⏸️ Estagnado: sem melhoria significativa ({no_improvement_count}/{self.config.patience})")
                
                if no_improvement_count >= self.config.patience:
                    # Determinar se convergiu (estabilizou) ou divergiu (piorou)
                    if avg_dist > best_avg_distance * 1.05:  # Piorou mais de 5%
                        self.converged = False
                        self.convergence_reason = f"DIVERGENCIA: Distancias aumentaram por {self.config.patience} iteracoes (piorou {((avg_dist/best_avg_distance - 1)*100):.1f}%)"
                    else:
                        self.converged = True
                        self.convergence_reason = f"ESTABILIZACAO: Sem melhoria significativa por {self.config.patience} iteracoes"
                    
                    print(f"\n[LOOP] {self.convergence_reason}")
                    break
            
            # Atualizar ideias para proxima iteracao
            current_llm_ideas = new_ideas
            all_generated_ideas.extend(new_ideas)  # NOVO: Acumular no histórico
            print(f"[LOOP] Total de ideias acumuladas: {len(all_generated_ideas)}")
        
        # Finalizar
        if not self.converged:
            self.convergence_reason = f"Atingido max_iterations ({self.config.max_iterations})"
            print(f"\n[LOOP] FINALIZACAO: {self.convergence_reason}")
        
        # Salvar resumo
        if self.config.output_dir:
            self._save_summary()
        
        print("\n" + "="*60)
        print("REFINEMENT LOOP CONCLUIDO")
        print("="*60 + "\n")
        
        return self.results
    
    def _generate_initial_ideas(self) -> List[str]:
        """
        Gera ideias iniciais da LLM (sem refinamento).
        
        Returns:
            Lista de ideias iniciais
        """
        print("[LOOP] Gerando ideias iniciais da LLM...")
        
        # Usar generation_step com bullets vazios
        # ZERO-SHOT: sem exemplos humanos para evitar vazamento de informacao
        # (human_examples=None por padrao, mas pode ser habilitado no futuro se necessario)
        initial_ideas = generation_step(
            invitation=self.config.invitation,
            directive=self.config.directive,
            bullets="",
            num_ideas=self.config.num_ideas_per_iter,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key_override=self.config.api_key_override,
            reasoning_effort=self.config.reasoning_effort,
            human_examples=None,  # ZERO-SHOT: sem few-shot learning para evitar vazamento
        )
        
        print(f"[LOOP] Geradas {len(initial_ideas)} ideias iniciais")
        return initial_ideas
    
    def _compute_distances(self, llm_ideas: List[str]) -> Tuple[float, float, List[float], float, float, float, float]:
        """
        Computa multiplas metricas de distancia entre ideias da LLM e humanas.
        
        Args:
            llm_ideas: Lista de ideias da LLM
        
        Returns:
            Tupla (avg_distance, min_distance, individual_distances, top3_mean, centroid_distance, centroid_to_centroid, distance_from_iter1)
        """
        # Embed ideias da LLM usando embed_texts() (suporta OpenAI e local)
        llm_embeddings = embed_texts(self.embedder, llm_ideas)
        
        # Embeddings ja vem normalizados de embed_texts()
        # Calcular distancias usando a funcao cosine_distance()
        min_distances_per_llm = []
        
        for llm_emb in llm_embeddings:
            # Para cada ideia LLM, calcular distancia para cada ideia humana
            distances_to_humans = [
                cosine_distance(llm_emb, human_emb)
                for human_emb in self.human_embeddings
            ]
            # Pegar a menor distancia (ideia humana mais proxima)
            min_dist = min(distances_to_humans)
            min_distances_per_llm.append(min_dist)
        
        # Metricas basicas
        avg_distance = float(np.mean(min_distances_per_llm))
        min_distance = float(np.min(min_distances_per_llm))
        individual_distances = min_distances_per_llm
        
        # Top-K mean: media das K melhores (menores distancias)
        k = min(3, len(min_distances_per_llm))  # top-3 ou menos se tiver poucas ideias
        topk_distances = sorted(min_distances_per_llm)[:k]
        top3_mean = float(np.mean(topk_distances))
        
        # Centroid distance: distancia media ao centroide do cluster humano
        human_centroid = self.human_embeddings.mean(axis=0)
        human_centroid = human_centroid / np.linalg.norm(human_centroid)  # Normalizar
        
        centroid_distances = [
            cosine_distance(llm_emb, human_centroid)
            for llm_emb in llm_embeddings
        ]
        centroid_distance = float(np.mean(centroid_distances))
        
        # NOVO: Centroid-to-Centroid - distancia entre o centro das ideias LLM e o centro das humanas
        llm_centroid = llm_embeddings.mean(axis=0)
        llm_centroid = llm_centroid / np.linalg.norm(llm_centroid)  # Normalizar
        
        centroid_to_centroid = float(cosine_distance(llm_centroid, human_centroid))
        
        # NOVO: Distancia em relacao a iteracao 1 (baseline)
        if self.iter1_centroid is None:
            # Esta eh a iteracao 1, salvar o centroide
            self.iter1_centroid = llm_centroid.copy()
            distance_from_iter1 = 0.0  # Distancia de si mesma = 0
        else:
            # Calcular distancia ao centroide da iteracao 1
            distance_from_iter1 = float(cosine_distance(llm_centroid, self.iter1_centroid))
        
        return avg_distance, min_distance, individual_distances, top3_mean, centroid_distance, centroid_to_centroid, distance_from_iter1
    
    def _save_iteration(self, result: IterationResult) -> None:
        """Salva resultado de uma iteracao em arquivo."""
        if not self.experiment_dir:
            return
        
        output_file = self.experiment_dir / f"iteration_{result.iteration:02d}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"[LOOP] Iteracao {result.iteration} salva em: {output_file}")
    
    def _save_summary(self) -> None:
        """Salva resumo completo do experimento."""
        summary = {
            "config": {
                "model": self.config.model,
                "embedder": self.config.embedder_name,
                "max_iterations": self.config.max_iterations,
                "patience": self.config.patience,
                "delta_threshold": self.config.delta_threshold,
                "num_ideas_per_iter": self.config.num_ideas_per_iter,
                "num_human_ideas": len(self.config.human_ideas),  # NOVO: salvar quantas foram usadas
                "use_clustering": self.config.use_clustering,  # NOVO: se clustering foi usado
            },
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
            "total_iterations": len(self.results),
            "best_avg_distance": min(r.avg_distance for r in self.results) if self.results else None,
            "best_min_distance": min(r.min_distance for r in self.results) if self.results else None,
            "iterations": [
                {
                    "iteration": r.iteration,
                    "avg_distance": r.avg_distance,
                    "min_distance": r.min_distance,
                    "num_ideas": len(r.generated_ideas),
                    "timestamp": r.timestamp,
                }
                for r in self.results
            ]
        }
        
        # Adicionar informacoes de clustering se foi usado (NOVO)
        if self.config.use_clustering and self.cluster_labels is not None:
            summary["clustering"] = {
                "method": self.config.clustering_method,
                "n_clusters_total": len(set(self.cluster_labels)),
                "selected_cluster_id": self.selected_cluster_id,
                "cluster_labels": self.cluster_labels,  # Label de cada historia
                "clusters_sizes": {
                    str(cluster_id): len(indices)
                    for cluster_id, indices in self.clusters_dict.items()
                },
                "total_human_ideas_available": len(self.all_human_ideas) if self.all_human_ideas else 0,
            }
        
        if not self.experiment_dir:
            return
        
        summary_file = self.experiment_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"[LOOP] Resumo salvo em: {summary_file}")


# Exemplo de uso (para testes)
if __name__ == "__main__":
    # Dados de teste
    INVITATION = """Strangers Again

I've been thinking a lot lately about the need to feel connected - to be seen, remembered, or maybe even just understood. Over the years, I've noticed how connection can manifest in the smallest of things: a shared meal, a passing glance, a familiar name we can't quite place.

This week, let's write stories about that pull between people. From fleeting relationships to chance encounters with strangers who seem like old friends, let's explore yearning and connection, even when things are complicated or just out of reach."""

    DIRECTIVE = "Center your story around two characters who like each other but don't get a happily ever after."

    HUMAN_IDEAS = [
        "A story about two childhood friends who reconnect at a funeral...",
        "A tale of missed connections at a train station over 10 years..."
    ]

    print("=== TESTE: refinement_loop.py ===\n")
    
    try:
        # Configurar
        config = RefinementConfig(
            invitation=INVITATION,
            directive=DIRECTIVE,
            human_ideas=HUMAN_IDEAS,
            model="gpt-4o-mini",
            embedder_name="all-MiniLM-L6-v2",
            max_iterations=2,  # Apenas 2 para teste
            patience=1,
            num_ideas_per_iter=3,
            output_dir=Path("test_refinement_output")
        )
        
        # Executar
        loop = RefinementLoop(config)
        results = loop.run()
        
        print("\n=== RESULTADO FINAL ===")
        print(f"Total de iteracoes: {len(results)}")
        print(f"Convergiu: {loop.converged}")
        print(f"Razao: {loop.convergence_reason}")
        
        if results:
            best = min(results, key=lambda r: r.avg_distance)
            print(f"\nMelhor iteracao: {best.iteration}")
            print(f"Melhor distancia media: {best.avg_distance:.4f}")
        
    except Exception as e:
        print(f"\n=== ERRO ===")
        print(f"Tipo: {type(e).__name__}")
        print(f"Mensagem: {e}")
        import traceback
        traceback.print_exc()

