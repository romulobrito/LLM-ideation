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
    X = model.encode(texts, normalize_embeddings=True, batch_size=64)
    return np.asarray(X, dtype=float)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # entradas normalizadas; distancia = 1 - cos
    sim = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return 1.0 - sim


# ------------------------------- LLM ----------------------------------------

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
    header = (
        "Previously, you generated two ideas (A preferred over B).\n\n"
        "Here is idea A (preferred):\n### A\n" + a_text.strip() + "\n\n"
        "Here is idea B (not preferred):\n### B\n" + b_text.strip() + "\n\n"
        "Generate two NEW and DISTINCT ideas, 150 words each, improving towards the qualities in A.\n"
        "Output strictly as:\n### 1.txt\n<idea one>\n### 2.txt\n<idea two>\n"
    )
    return BASE_PROMPT + "\n\n" + header


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
    out_dir: Path


def run_for_reference(ref_text: str, ref_id: int, embedder, cfg: IterConfig) -> None:
    out_base = cfg.out_dir / f"ref_{ref_id:03d}"
    out_base.mkdir(parents=True, exist_ok=True)
    log_path = out_base / "log.csv"
    is_new = not log_path.exists()
    # embed da referencia
    print(f"  🧠 Gerando embedding da referencia...")
    r_vec = embed_texts(embedder, [ref_text])[0]
    print(f"  ✓ Embedding gerado")

    best = float("inf")
    no_improve = 0
    
    print(f"  🚀 Iniciando ciclo iterativo (max {cfg.max_iters} iteracoes)...")

    with log_path.open("a", newline="", encoding="utf-8") as lf:
        w = csv.writer(lf)
        if is_new:
            w.writerow([
                "iter", "cand_id", "dist", "chosen_A", "dmin_so_far",
                "model", "temperature", "reasoning", "max_tokens", "files"
            ])

        # iteracao 1
        txt = call_deepseek(
            prompt=BASE_PROMPT,
            model=cfg.model_id,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            api_key_override=None,
            reasoning_effort=cfg.reasoning,
        )
        i1, i2 = parse_two_ideas(txt)
        iter_dir = out_base / "iter_001"
        iter_dir.mkdir(exist_ok=True)
        (iter_dir / "1.txt").write_text(i1, encoding="utf-8")
        (iter_dir / "2.txt").write_text(i2, encoding="utf-8")
        e = embed_texts(embedder, [i1, i2])
        d1 = cosine_distance(e[0], r_vec)
        d2 = cosine_distance(e[1], r_vec)
        if d1 <= d2:
            A, B = i1, i2
            A_idx = 1
        else:
            A, B = i2, i1
            A_idx = 2
        best = min(d1, d2)
        w.writerow([1, 1, f"{d1:.6f}", 1 if A_idx == 1 else 0, f"{best:.6f}", cfg.model_id, cfg.temperature, cfg.reasoning or "", cfg.max_tokens, str(iter_dir / "1.txt")])
        w.writerow([1, 2, f"{d2:.6f}", 1 if A_idx == 2 else 0, f"{best:.6f}", cfg.model_id, cfg.temperature, cfg.reasoning or "", cfg.max_tokens, str(iter_dir / "2.txt")])
        
        print(f"  ✓ Iteracao 1: dist_1={d1:.4f}, dist_2={d2:.4f} → melhor: ideia {A_idx} (dist={best:.4f})")

        # loop
        for k in range(2, cfg.max_iters + 1):
            txt = call_deepseek(
                prompt=prompt_with_feedback(A, B),
                model=cfg.model_id,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                api_key_override=None,
                reasoning_effort=cfg.reasoning,
            )
            i1, i2 = parse_two_ideas(txt)
            iter_dir = out_base / f"iter_{k:03d}"
            iter_dir.mkdir(exist_ok=True)
            (iter_dir / "1.txt").write_text(i1, encoding="utf-8")
            (iter_dir / "2.txt").write_text(i2, encoding="utf-8")
            e = embed_texts(embedder, [i1, i2])
            d1 = cosine_distance(e[0], r_vec)
            d2 = cosine_distance(e[1], r_vec)
            
            # Identificar qual e a melhor da iteracao atual
            if d1 <= d2:
                best_current_idx = 1
                best_current_text = i1
                best_current_dist = d1
                worst_current_text = i2
            else:
                best_current_idx = 2
                best_current_text = i2
                best_current_dist = d2
                worst_current_text = i1
            
            # log
            best_prev = best
            best = min(best, d1, d2)
            w.writerow([k, 1, f"{d1:.6f}", 1 if best_current_idx == 1 else 0, f"{best:.6f}", cfg.model_id, cfg.temperature, cfg.reasoning or "", cfg.max_tokens, str(iter_dir / "1.txt")])
            w.writerow([k, 2, f"{d2:.6f}", 1 if best_current_idx == 2 else 0, f"{best:.6f}", cfg.model_id, cfg.temperature, cfg.reasoning or "", cfg.max_tokens, str(iter_dir / "2.txt")])
            
            # Imprimir progresso
            melhoria = "✓" if best < best_prev else "="
            print(f"  {melhoria} Iteracao {k}: dist_1={d1:.4f}, dist_2={d2:.4f} → melhor: ideia {best_current_idx} (dist={best:.4f}, no_improve={no_improve})")

            # Atualizar A apenas se a melhor atual for melhor que a melhor historica
            # Isso garante que o feedback sempre use a MELHOR ideia ja gerada
            if best_current_dist < best_prev:
                A = best_current_text
            # Caso contrario, manter A anterior (melhor historica)
            
            # B sempre e a pior da iteracao atual (para contraste)
            B = worst_current_text

            # criterios de parada
            if best + cfg.delta < best_prev:
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= cfg.patience:
                print(f"  ⏸️  Early stop: sem melhora significativa por {cfg.patience} iteracoes")
                break
        
        print(f"  ✅ Concluido: {k} iteracoes, melhor distancia final: {best:.4f}")


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
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--embedder", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--max-iters", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--delta", type=float, default=0.005)
    ap.add_argument("--clean", action="store_true", help="Limpar diretorio de saida antes de iniciar")
    args = ap.parse_args()

    print("=" * 80)
    print("🧪 EXPERIMENTO ITERATIVO - INICIANDO")
    print("=" * 80)
    print(f"📁 Arquivo de referencias: {args.refs_path}")
    print(f"📂 Diretorio de saida: {args.out_dir}")
    print(f"🤖 Modelo LLM: {args.model}")
    print(f"🌡️  Temperatura: {args.temperature}")
    print(f"📝 Max tokens: {args.max_tokens}")
    print(f"🔢 Max iteracoes: {args.max_iters}")
    print(f"⏸️  Paciencia (early stop): {args.patience}")
    print(f"📊 Delta minimo: {args.delta}")
    print(f"🧠 Embedder: {args.embedder}")
    print(f"💻 Device: {args.device}")
    if args.reasoning and args.reasoning != "None":
        print(f"🤔 Reasoning: {args.reasoning}")
    print("=" * 80)
    print()
    
    # Limpar diretorio de saida se solicitado
    out_path = Path(args.out_dir)
    if args.clean and out_path.exists():
        import shutil
        print(f"🗑️  Limpando diretorio de saida: {out_path}")
        try:
            shutil.rmtree(out_path)
            print(f"✅ Diretorio {out_path} removido com sucesso")
        except Exception as e:
            print(f"⚠️  Aviso: Nao foi possivel remover completamente {out_path}: {e}")
        print()
    
    refs = load_references_from_fs(args.refs_path)
    if not refs:
        raise SystemExit("Sem referencias no caminho informado.")
    
    print(f"✅ Carregadas {len(refs)} referencias")
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
        out_dir=out_dir,
    )

    for i, ref in enumerate(refs, start=1):
        print(f"\n{'='*80}")
        print(f"📄 REFERENCIA {i}/{len(refs)}")
        print(f"{'='*80}")
        run_for_reference(ref, i, embedder, cfg)
    
    print(f"\n{'='*80}")
    print("✅ EXPERIMENTO CONCLUIDO COM SUCESSO!")
    print(f"📂 Resultados salvos em: {out_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


