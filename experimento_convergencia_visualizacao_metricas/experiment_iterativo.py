#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento iterativo de aproximacao por embeddings.

Passos por referencia:
1) Embed da referencia.
2) Gerar 2 ideias (1.txt, 2.txt).
3) Calcular distancias ao embed da referencia; escolher A = mais proxima.
4) Realimentar o modelo com feedback (A preferida a B) e gerar novas 2 ideias.
5) Repetir ate max_iter ou estagnacao (delta/patience).

Requer: OPENROUTER_API_KEY ou OPENROUTER_API_KEY_<VENDOR> no ambiente.
"""

from __future__ import annotations

import os
import csv
import re
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from dotenv import load_dotenv

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

import numpy as np

from bleu_minimal_deepseek import call_deepseek


# ----------------------------- util fs -------------------------------------

def load_references_from_fs(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    if p.is_file():
        data = p.read_text(encoding="utf-8")
        return _parse_references_text(data)
    files = sorted(list(p.rglob("*.txt")) + list(p.rglob("*.md")))
    refs: List[str] = []
    for fp in files:
        try:
            s = fp.read_text(encoding="utf-8").strip()
            if s:
                refs.append(s)
        except Exception:
            continue
    return refs


def _parse_references_text(raw: str) -> List[str]:
    s = (raw or "").strip()
    if not s:
        return []
    # tentar JSON list de strings
    try:
        import json

        v = json.loads(s)
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return [x.strip() for x in v if str(x).strip()]
    except Exception:
        pass
    if "---" in s:
        parts = [p.strip() for p in s.split("---")]
        return [p for p in parts if p]
    # blocos por linhas em branco
    out: List[str] = []
    buf: List[str] = []
    for line in s.splitlines():
        if line.strip() == "":
            if buf:
                out.append(" ".join(buf).strip())
                buf = []
        else:
            buf.append(line.strip())
    if buf:
        out.append(" ".join(buf).strip())
    if not out:
        out = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return [x for x in out if x]


# --------------------------- embeddings -------------------------------------

def get_embedder(model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
    """
    Inicializa um modelo de embeddings.
    
    Args:
        model_name: Nome do modelo (sentence-transformers ou openai)
        device: Device para sentence-transformers (cpu/cuda)
    
    Returns:
        Modelo ou string "openai" para embeddings OpenAI
    
    Supported models:
        - "all-MiniLM-L6-v2" (384D, local, gratis)
        - "openai" ou "text-embedding-3-large" (3072D, API, ~$0.13/1M tokens)
        - "text-embedding-3-small" (1536D, API, ~$0.02/1M tokens)
    """
    if model_name in ["openai", "text-embedding-3-large", "text-embedding-3-small"]:
        # Verificar se OPENAI_API_KEY existe
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY nao encontrada no ambiente. "
                "Necessaria para usar embeddings OpenAI."
            )
        # Retornar identificador especial
        return model_name if model_name != "openai" else "text-embedding-3-large"
    
    # Sentence Transformers (local)
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers nao instalado.")
    if device is None:
        try:
            import torch  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    return SentenceTransformer(model_name, device=device)


def embed_texts(model, texts: List[str]) -> np.ndarray:
    """
    Gera embeddings para uma lista de textos.
    
    Args:
        model: Modelo retornado por get_embedder()
        texts: Lista de strings
    
    Returns:
        Array numpy (N, D) com embeddings normalizados
    """
    # Detectar se e OpenAI ou Sentence Transformers
    if isinstance(model, str) and "text-embedding" in model:
        # OpenAI embeddings
        return _embed_texts_openai(model, texts)
    else:
        # Sentence Transformers (local)
        X = model.encode(texts, normalize_embeddings=True, batch_size=64)
        return np.asarray(X, dtype=float)


def _embed_texts_openai(model_name: str, texts: List[str]) -> np.ndarray:
    """
    Gera embeddings usando API OpenAI.
    
    Args:
        model_name: "text-embedding-3-large" ou "text-embedding-3-small"
        texts: Lista de strings
    
    Returns:
        Array numpy (N, D) com embeddings normalizados
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "openai nao instalado. Execute: pip install openai"
        )
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # OpenAI permite batch de ate 2048 textos
    # Vamos processar em chunks de 100 para seguranca
    all_embeddings = []
    chunk_size = 100
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        response = client.embeddings.create(
            input=chunk,
            model=model_name,
            encoding_format="float"
        )
        chunk_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(chunk_embeddings)
    
    # Converter para numpy e normalizar
    X = np.array(all_embeddings, dtype=float)
    # Normalizar (OpenAI ja retorna normalizado, mas garantir)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-12)
    
    return X


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # entradas normalizadas; distancia = 1 - cos
    sim = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return 1.0 - sim


# ------------------------------- LLM ----------------------------------------

# Base prompt for initial idea generation
# Used in first iteration (without feedback)
BASE_PROMPT = (
    "Consider the following writing contest invitation:\n"
    "------\n"
    "Strangers Again\n\n"
    "I have been thinking a lot lately about the need to feel connected.\n"
    "This week, let us write stories about yearning and connection.\n"
    "------\n\n"
    "And also this writing directive:\n"
    "------\n"
    "Center your story around two characters who like each other but do not get a happily ever after.\n"
    "------\n\n"
    "Your task is to creatively generate 2 short-story ideas based on the invitation and directive.\n"
    "Each idea must have around 150 words and be distinct in theme, tone, and concept.\n"
    "Output strictly in two sections headed by:\n"
    "### 1.txt\n"
    "<idea one>\n"
    "### 2.txt\n"
    "<idea two>\n"
)


def prompt_with_feedback(a_text: str, b_text: str) -> str:
    """
    Generate prompt with user preference feedback.
    
    This prompt simulates a writing contest scenario where:
    1. User has preferred idea A over idea B
    2. The LLM should generate new ideas considering this preference
    3. The goal is to explore variations that align with the preferred direction
    
    Args:
        a_text: Text of preferred idea (chosen by user)
        b_text: Text of non-preferred idea (rejected)
    
    Returns:
        Complete prompt with preference feedback
    """
    header = (
        "Previously, you have generated 2 short-story ideas (A and B below) based on the invitation and directive. "
        "The user preferred idea A over idea B because it was closer to their target vision.\n\n"
        "Here's idea A (preferred):\n------\n" + a_text.strip() + "\n------\n\n"
        "Here's idea B (not preferred):\n------\n" + b_text.strip() + "\n------\n\n"
        "Your task is to generate 2 NEW short-story ideas that REFINE and explore VARIATIONS of idea A's core elements. "
        "Focus on staying CLOSE to the style, theme, tone, and narrative approach of idea A, while introducing subtle creative variations. "
        "Each idea should span 150 words.\n\n"
        "The ideas should be provided in two separate files: 1.txt and 2.txt."
    )
    return BASE_PROMPT + "\n\n" + header
    
    # header = (
    #     "Previously, you have generated 2 short-story ideas (A and B below) based on the invitation and directive. "
    #     "The user preferred idea A over idea B.\n\n"
    #     "Here's idea A:\n------\n" + a_text.strip() + "\n------\n\n"
    #     "Here's idea B:\n------\n" + b_text.strip() + "\n------\n\n"
    #     "Your task is to creatively generate another 2 short-story ideas based on the invitation, directive, and now the feedback. "
    #     "Each idea should span 150 words and be distinct in theme, tone, and concept.\n\n"
    #     "The ideas should be provided in two separate files: 1.txt and 2.txt."
    # )
    # return BASE_PROMPT + "\n\n" + header
    
    # OLD PROMPT 
    # header = (
    #     "Previously, you generated two ideas (A and B). "
    #     "Idea A was semantically closer to a target reference than idea B.\n\n"
    #     "Here is idea A (closer to target):\n### A\n" + a_text.strip() + "\n\n"
    #     "Here is idea B (further from target):\n### B\n" + b_text.strip() + "\n\n"
    #     "Generate two NEW and DISTINCT ideas, 150 words each, that explore variations "
    #     "closer to the semantic space of idea A.\n"
    #     "Output strictly as:\n### 1.txt\n<idea one>\n### 2.txt\n<idea two>\n"
    # )
    # return BASE_PROMPT + "\n\n" + header


def parse_two_ideas(text: str) -> Tuple[str, str]:
    s = (text or "").strip()
    if not s:
        return "", ""
    # procurar seções ### 1.txt / ### 2.txt
    m1 = re.search(r"###\s*1\.txt\s*(.+?)###\s*2\.txt\s*(.+)", s, re.S | re.I)
    if m1:
        i1 = m1.group(1).strip()
        i2 = m1.group(2).strip()
        return i1, i2
    # fallback: separador ---
    if "---" in s:
        parts = [p.strip() for p in s.split("---") if p.strip()]
        if len(parts) >= 2:
            return parts[0], parts[1]
    # ultimo fallback: tentar dividir por marcadores 1. e 2.
    m2 = re.split(r"\n\s*2[\).-]", s, maxsplit=1)
    if len(m2) == 2:
        first = re.sub(r"^\s*1[\).-]", "", m2[0]).strip()
        second = m2[1].strip()
        return first, second
    return s, ""


# ------------------------------ experimento ---------------------------------

@dataclass
class IterConfig:
    model_id: str
    reasoning: Optional[str]
    temperature: float
    max_tokens: int
    max_iters: int
    patience: int
    delta: float
    cands_per_iter: int
    out_dir: Path


def run_for_reference(ref_text: str, ref_id: int, embedder, cfg: IterConfig) -> None:
    out_base = cfg.out_dir / f"ref_{ref_id:03d}"
    out_base.mkdir(parents=True, exist_ok=True)
    log_path = out_base / "log.csv"
    
    # IMPORTANTE: Sempre sobrescrever o log para evitar duplicatas
    # Se o arquivo existe, remover para começar do zero
    if log_path.exists():
        log_path.unlink()
        print(f"    Log anterior removido: {log_path}")
    
    # embed da referencia
    print(f"   Gerando embedding da referencia...")
    r_vec = embed_texts(embedder, [ref_text])[0]
    print(f"   Embedding gerado")

    best = float("inf")
    no_improve = 0
    
    print(f"   Iniciando ciclo iterativo (max {cfg.max_iters} iteracoes)...")

    # Usar modo "w" (write) para criar arquivo novo
    with log_path.open("w", newline="", encoding="utf-8") as lf:
        w = csv.writer(lf)
        # Sempre escrever o header (arquivo novo)
        w.writerow([
            "iter", "cand_id", "dist", "chosen_A", "dmin_so_far",
            "model", "temperature", "reasoning", "max_tokens", "files"
        ])

        # iteracao 1: gerar cfg.cands_per_iter candidatos
        iter_dir = out_base / "iter_001"
        iter_dir.mkdir(exist_ok=True)
        
        cands, paths = [], []
        print(f"   Gerando {cfg.cands_per_iter} candidatos para iteracao 1...")
        while len(cands) < cfg.cands_per_iter:
            txt = call_deepseek(
                prompt=BASE_PROMPT,
                model=cfg.model_id,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                api_key_override=None,
                reasoning_effort=cfg.reasoning,
            )
            i1, i2 = parse_two_ideas(txt)
            for idea in (i1, i2):
                if idea and idea.strip() and len(cands) < cfg.cands_per_iter:
                    cands.append(idea.strip())
                    p = iter_dir / f"{len(cands)}.txt"
                    p.write_text(idea.strip(), encoding="utf-8")
                    paths.append(p)
        
        # Calcular embeddings e distancias
        e = embed_texts(embedder, cands)
        dists = [cosine_distance(vec, r_vec) for vec in e]
        
        # Ordenar por distancia (menor = melhor)
        order = np.argsort(dists)
        top1_idx = int(order[0])
        top2_idx = int(order[1])
        
        # A = melhor (top1), B = segundo melhor (top2)
        A = cands[top1_idx]
        B = cands[top2_idx]
        best = dists[top1_idx]
        
        # Logar todos os candidatos
        for j, (d, pth) in enumerate(zip(dists, paths)):
            chosen = 1 if j == top1_idx else 0
            w.writerow([1, j+1, f"{d:.6f}", chosen, f"{best:.6f}", 
                       cfg.model_id, cfg.temperature, cfg.reasoning or "", cfg.max_tokens, str(pth)])
        
        print(f"   Iteracao 1: melhor=cand_{top1_idx+1} (dist={best:.4f}), segundo=cand_{top2_idx+1} (dist={dists[top2_idx]:.4f})")

        # loop principal (iteracoes k >= 2)
        for k in range(2, cfg.max_iters + 1):
            iter_dir = out_base / f"iter_{k:03d}"
            iter_dir.mkdir(exist_ok=True)
            
            cands, paths = [], []
            while len(cands) < cfg.cands_per_iter:
                txt = call_deepseek(
                    prompt=prompt_with_feedback(A, B),
                    model=cfg.model_id,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    api_key_override=None,
                    reasoning_effort=cfg.reasoning,
                )
                i1, i2 = parse_two_ideas(txt)
                for idea in (i1, i2):
                    if idea and idea.strip() and len(cands) < cfg.cands_per_iter:
                        cands.append(idea.strip())
                        p = iter_dir / f"{len(cands)}.txt"
                        p.write_text(idea.strip(), encoding="utf-8")
                        paths.append(p)
            
            # Calcular embeddings e distancias
            e = embed_texts(embedder, cands)
            dists = [cosine_distance(vec, r_vec) for vec in e]
            
            # Ordenar por distancia (menor = melhor)
            order = np.argsort(dists)
            top1_idx = int(order[0])
            top2_idx = int(order[1])
            
            # Melhor da iteracao atual
            best_iter = dists[top1_idx]
            
            # Atualizar melhor global
            best_prev = best
            best = min(best, best_iter)
            
            # Logar todos os candidatos
            for j, (d, pth) in enumerate(zip(dists, paths)):
                chosen = 1 if j == top1_idx else 0
                w.writerow([k, j+1, f"{d:.6f}", chosen, f"{best:.6f}",
                           cfg.model_id, cfg.temperature, cfg.reasoning or "", cfg.max_tokens, str(pth)])
            
            # MUDANCA CRITICA: Atualizar A SEMPRE para top1 (nao so quando melhora global)
            # Isso permite "pioras temporarias" e evita congelamento em minimo local
            A = cands[top1_idx]
            B = cands[top2_idx]  # B = segundo melhor (nao a pior!)
            
            # Calcular gap entre top1 e top2
            gap = dists[top2_idx] - dists[top1_idx]
            
            # Early stopping com delta RELATIVO
            if best_prev > 0:
                rel_improvement = (best_prev - best) / best_prev
            else:
                rel_improvement = 0.0
            
            if rel_improvement > cfg.delta:
                no_improve = 0
                melhoria_str = f"↓ (-{rel_improvement*100:.1f}%)"
            else:
                no_improve += 1
                melhoria_str = "="
            
            print(f"  {melhoria_str} Iteracao {k}: top1=cand_{top1_idx+1} ({best_iter:.4f}), "
                  f"top2=cand_{top2_idx+1} ({dists[top2_idx]:.4f}), gap={gap:.4f} | "
                  f"best_global={best:.4f}, no_improve={no_improve}/{cfg.patience}")
            
            if no_improve >= cfg.patience:
                print(f"    Early stop: sem melhoria relativa > {cfg.delta*100:.1f}% por {cfg.patience} iteracoes")
                break
        
        print(f"   Concluido: {k} iteracoes, melhor distancia final: {best:.4f}")


def _load_env_robusto() -> None:
    """Carrega .env de locais comuns (cwd, diretorio do script, pai) com override."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for p in candidates:
        try:
            if p.exists():
                load_dotenv(dotenv_path=str(p), override=True)
        except Exception:
            pass


def main() -> None:
    _load_env_robusto()
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs-path", type=str, required=True, help="Arquivo ou pasta com referencias")
    ap.add_argument("--out-dir", type=str, required=True, help="Pasta de saida do experimento")
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="Modelo LLM (sem '/' para OpenAI direta, com '/' para OpenRouter)")
    ap.add_argument("--reasoning", type=str, default="None", help="Reasoning effort (apenas para modelos OpenRouter que suportam)")
    ap.add_argument("--temperature", type=float, default=0.5, help="Temperature para geracao (default 0.5 para convergencia)")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--embedder", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--max-iters", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10, help="Numero de iteracoes sem melhoria antes de parar (default 10)")
    ap.add_argument("--delta", type=float, default=0.01, help="Melhoria relativa minima (default 0.01 = 1%%)")
    ap.add_argument("--cands-per-iter", type=int, default=4, help="Numero de candidatos por iteracao (default 4)")
    ap.add_argument("--clean", action="store_true", help="Limpar diretorio de saida antes de iniciar")
    ap.add_argument("--start-from-ref", type=int, default=None, help="Comecar a partir da referencia N (pula referencias anteriores)")
    ap.add_argument("--skip-complete", action="store_true", help="Pular referencias que ja tem >= 10 iteracoes")
    ap.add_argument("--max-refs", type=int, default=None, help="Numero maximo de referencias a processar (limita o total)")
    args = ap.parse_args()

    print("=" * 80)
    print(" EXPERIMENTO ITERATIVO - INICIANDO")
    print("=" * 80)
    print(f" Arquivo de referencias: {args.refs_path}")
    print(f" Diretorio de saida: {args.out_dir}")
    print(f" Modelo LLM: {args.model}")
    print(f"  Temperatura: {args.temperature}")
    print(f" Max tokens: {args.max_tokens}")
    print(f" Candidatos por iteracao: {args.cands_per_iter}")
    print(f" Max iteracoes: {args.max_iters}")
    print(f"  Paciencia (early stop): {args.patience}")
    print(f" Delta minimo (relativo): {args.delta*100:.1f}%")
    print(f" Embedder: {args.embedder}")
    print(f" Device: {args.device}")
    if args.reasoning and args.reasoning != "None":
        print(f" Reasoning: {args.reasoning}")
    print("=" * 80)
    print()
    
    # Limpar diretorio de saida se solicitado
    out_path = Path(args.out_dir)
    if args.clean and out_path.exists():
        import shutil
        print(f"  Limpando diretorio de saida: {out_path}")
        try:
            shutil.rmtree(out_path)
            print(f" Diretorio {out_path} removido com sucesso")
        except Exception as e:
            print(f"  Aviso: Nao foi possivel remover completamente {out_path}: {e}")
        print()
    
    refs = load_references_from_fs(args.refs_path)
    if not refs:
        raise SystemExit("Sem referencias no caminho informado.")
    
    # Limitar número de referências se especificado
    total_refs_loaded = len(refs)
    if args.max_refs and args.max_refs < total_refs_loaded:
        refs = refs[:args.max_refs]
        print(f" Carregadas {total_refs_loaded} referencias, limitando a {args.max_refs}")
    else:
        print(f" Carregadas {len(refs)} referencias")
    print()

    device = None if args.device == "auto" else args.device
    embedder = get_embedder(args.embedder, device=device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = IterConfig(
        model_id=args.model,
        reasoning=(None if str(args.reasoning).lower() in {"none", "", "null"} else args.reasoning),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_iters=args.max_iters,
        patience=args.patience,
        delta=args.delta,
        cands_per_iter=args.cands_per_iter,
        out_dir=out_dir,
    )

    for i, ref in enumerate(refs, start=1):
        # Pular referências se --start-from-ref especificado
        if args.start_from_ref and i < args.start_from_ref:
            print(f" PULANDO ref_{i:03d} (start-from-ref={args.start_from_ref})")
            continue
        
        # Pular referências completas se --skip-complete especificado
        if args.skip_complete:
            ref_dir = out_dir / f"ref_{i:03d}"
            log_file = ref_dir / "log.csv"
            if log_file.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(log_file)
                    if len(df) > 0 and df['iter'].max() >= 10:
                        print(f" PULANDO ref_{i:03d} (já completa com {df['iter'].max()} iterações)")
                        continue
                except Exception:
                    pass  # Se der erro, processar normalmente
        
        print(f"\n{'='*80}")
        print(f" REFERENCIA {i}/{len(refs)}")
        print(f"{'='*80}")
        run_for_reference(ref, i, embedder, cfg)
    
    print(f"\n{'='*80}")
    print(" EXPERIMENTO CONCLUIDO COM SUCESSO!")
    print(f" Resultados salvos em: {out_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


