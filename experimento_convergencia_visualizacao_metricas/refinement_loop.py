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
from sentence_transformers import SentenceTransformer

from refinement_critique import critique_step
from refinement_packing import packing_step
from refinement_generation import generation_step
from refinement_north import generate_north_star, format_north_with_tactical


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
        
        print("[LOOP] Inicializando RefinementLoop...")
        self._initialize_embedder()
        self._embed_human_ideas()
        
        if self.config.output_dir:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[LOOP] Diretorio de saida: {self.config.output_dir}")
    
    def _initialize_embedder(self) -> None:
        """Inicializa o modelo de embeddings."""
        print(f"[LOOP] Carregando embedder: {self.config.embedder_name}")
        
        # Determinar device
        device = self.config.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.embedder = SentenceTransformer(self.config.embedder_name, device=device)
        print(f"[LOOP] Embedder carregado em: {device}")
    
    def _embed_human_ideas(self) -> None:
        """Computa embeddings das ideias humanas."""
        print(f"[LOOP] Computando embeddings de {len(self.config.human_ideas)} ideias humanas...")
        self.human_embeddings = self.embedder.encode(
            self.config.human_ideas,
            convert_to_numpy=True,
            show_progress_bar=False
        )
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
                human_examples=self.config.human_ideas,  # NOVO: Few-shot learning
            )
            
            # Calcular distancias
            avg_dist, min_dist = self._compute_distances(new_ideas)
            
            # Criar resultado
            result = IterationResult(
                iteration=iteration,
                critique_json=critique_json,
                bullets=bullets,
                generated_ideas=new_ideas,
                avg_distance=avg_dist,
                min_distance=min_dist,
                timestamp=datetime.now().isoformat(),
            )
            self.results.append(result)
            
            # Salvar resultado
            if self.config.output_dir:
                self._save_iteration(result)
            
            # Mostrar metricas
            print(f"\n[LOOP] Metricas da iteracao {iteration}:")
            print(f"  - Distancia media: {avg_dist:.4f}")
            print(f"  - Distancia minima: {min_dist:.4f}")
            print(f"  - Ideias geradas: {len(new_ideas)}")
            
            # Verificar convergencia
            improvement = best_avg_distance - avg_dist
            
            if avg_dist < best_avg_distance - self.config.delta_threshold:
                # Houve melhoria significativa
                best_avg_distance = avg_dist
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
            human_examples=self.config.human_ideas,  # NOVO: Few-shot desde o início
        )
        
        print(f"[LOOP] Geradas {len(initial_ideas)} ideias iniciais")
        return initial_ideas
    
    def _compute_distances(self, llm_ideas: List[str]) -> Tuple[float, float]:
        """
        Computa distancias entre ideias da LLM e humanas.
        
        Args:
            llm_ideas: Lista de ideias da LLM
        
        Returns:
            Tupla (distancia_media, distancia_minima)
        """
        # Embed ideias da LLM
        llm_embeddings = self.embedder.encode(
            llm_ideas,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Calcular distancias coseno
        # Normalizar vetores
        human_norm = self.human_embeddings / np.linalg.norm(
            self.human_embeddings, axis=1, keepdims=True
        )
        llm_norm = llm_embeddings / np.linalg.norm(
            llm_embeddings, axis=1, keepdims=True
        )
        
        # Similaridade coseno: dot product de vetores normalizados
        similarities = np.dot(llm_norm, human_norm.T)
        
        # Distancia coseno: 1 - similaridade
        distances = 1.0 - similarities
        
        # Para cada ideia LLM, pegar a menor distancia para qualquer ideia humana
        min_distances_per_llm = distances.min(axis=1)
        
        avg_distance = float(min_distances_per_llm.mean())
        min_distance = float(min_distances_per_llm.min())
        
        return avg_distance, min_distance
    
    def _save_iteration(self, result: IterationResult) -> None:
        """Salva resultado de uma iteracao em arquivo."""
        output_file = self.config.output_dir / f"iteration_{result.iteration:02d}.json"
        
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
        
        summary_file = self.config.output_dir / "summary.json"
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

