#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BLEU minimo para avaliar saidas geradas por LLM (DeepSeek via OpenRouter) em relacao a referencias.
Uso basico:
  python bleu_minimal_deepseek.py --prompt "Explique o que e MAUVE" --reference "MAUVE e uma metrica..."

Opcoes:
  --prompt <str>                Prompt/contexto para o LLM gerar uma resposta.
  --reference <str>             Referencia unica (string).
  --ref-file <path>             Arquivo com uma ou mais referencias (uma por linha ou JSON list).
  --hypothesis <str>            Se voce ja tem a hipotese (saida do LLM), passe aqui para so medir o BLEU.
  --model <name>                Modelo DeepSeek (padrao: deepseek/deepseek-chat).
  --max-tokens <int>            Maximo de tokens para geracao (padrao: 400).
  --temperature <float>         Temperatura (padrao: 0.3).
  --use-sacrebleu               Se presente e biblioteca instalada, usa sacrebleu (senao usa BLEU interno).
  --quiet                       Modo silencioso.
Requerimento para geracao: variavel de ambiente OPENROUTER_API_KEY.
"""

import os, argparse, json, math, sys
from typing import List, Sequence, Tuple
import re

# utilidades basicas

def simple_tokenize(text: str) -> List[str]:
    """Tokenizacao simples por espacos + limpeza leve"""
    text = text.strip()
    # normalizacao leve
    text = re.sub(r'\s+', ' ', text.lower())
    return text.split() if text else []

def ngrams(tokens: List[str], n: int):
    return [tuple(tokens[i:i+n]) for i in range(0, max(0, len(tokens)-n+1))]

# BLEU interno

def clipped_precision(hyp: List[str], refs_tok: List[List[str]], n: int) -> float:
    from collections import Counter
    hyp_ngr = Counter(ngrams(hyp, n))
    if not hyp_ngr:
        return 0.0
    max_ref_counts = {}
    for ref in refs_tok:
        ref_counts = Counter(ngrams(ref, n))
        for g, c in ref_counts.items():
            max_ref_counts[g] = max(max_ref_counts.get(g, 0), c)
    clipped = {g: min(c, max_ref_counts.get(g, 0)) for g, c in hyp_ngr.items()}
    return (sum(clipped.values()) / sum(hyp_ngr.values())) if hyp_ngr else 0.0

def brevity_penalty(c_len: int, r_len: int) -> float:
    if c_len > r_len:
        return 1.0
    if c_len == 0:
        return 0.0
    return math.exp(1.0 - float(r_len)/float(c_len))

def closest_ref_length(hyp_len: int, ref_lens: List[int]) -> int:
    # escolhe o comprimento de referencia mais proximo 
    return min(ref_lens, key=lambda rl: (abs(rl - hyp_len), rl))

def sentence_bleu_internal(hyp: str, refs: Sequence[str], max_n: int = 4, smoothing: bool = True) -> float:
    """BLEU (sentence-level) com smoothing simples, multi-referencia"""
    hyp_tok = simple_tokenize(hyp)
    refs_tok = [simple_tokenize(r) for r in refs]
    if not hyp_tok or not any(refs_tok):
        return 0.0
    precisions = []
    for n in range(1, max_n+1):
        p = clipped_precision(hyp_tok, refs_tok, n)
        if p == 0.0 and smoothing:
            # smoothing M1: substitui zero por um valor pequeno dependente de n-gram
            p = 1e-9
        precisions.append(p)
    # media geometrica
    geo_mean = math.exp(sum((1.0/max_n) * math.log(p) for p in precisions))
    # BP
    c = len(hyp_tok)
    r = closest_ref_length(c, [len(rt) for rt in refs_tok])
    bp = brevity_penalty(c, r)
    return bp * geo_mean

def corpus_bleu_internal(hyps: Sequence[str], list_of_refs: Sequence[Sequence[str]], max_n: int = 4, smoothing: bool=True) -> float:
    """Corpus-level BLEU aproximado somando contagens (como no BLEU classico)."""
    from collections import Counter
    hyp_len_total = 0
    ref_len_total = 0
    numerators = [0]*max_n
    denominators = [0]*max_n

    for hyp, refs in zip(hyps, list_of_refs):
        hyp_tok = simple_tokenize(hyp)
        refs_tok = [simple_tokenize(r) for r in refs]
        hyp_len_total += len(hyp_tok)
        ref_len_total += closest_ref_length(len(hyp_tok), [len(rt) for rt in refs_tok])
        for n in range(1, max_n+1):
            from collections import Counter
            hyp_counts = Counter(ngrams(hyp_tok, n))
            max_ref_counts = {}
            for ref in refs_tok:
                ref_counts = Counter(ngrams(ref, n))
                for g, c in ref_counts.items():
                    max_ref_counts[g] = max(max_ref_counts.get(g, 0), c)
            clipped = {g: min(c, max_ref_counts.get(g, 0)) for g, c in hyp_counts.items()}
            numerators[n-1] += sum(clipped.values())
            denominators[n-1] += max(1, sum(hyp_counts.values()))

    precisions = []
    for n in range(1, max_n+1):
        if denominators[n-1] == 0:
            p = 0.0
        else:
            p = numerators[n-1]/denominators[n-1]
        if p == 0.0 and smoothing:
            p = 1e-9
        precisions.append(p)
    geo_mean = math.exp(sum((1.0/max_n) * math.log(p) for p in precisions))
    bp = brevity_penalty(hyp_len_total, ref_len_total)
    return bp * geo_mean

# metricas de avaliacao

def compute_bleu(hypothesis: str, references: Sequence[str], use_sacrebleu: bool=False, corpus: bool=False) -> float:
    if use_sacrebleu:
        try:
            import sacrebleu
            if corpus:
                # corpus BLEU: sacrebleu expects list of hyps and list of refs lists (but transposed)
                bleu = sacrebleu.corpus_bleu([hypothesis], [references])
                return bleu.score/100.0
            else:
                # sentence BLEU:
                bleu = sacrebleu.sentence_bleu(hypothesis, references)
                return bleu.score/100.0
        except Exception:
            pass
    # interno (fallback)
    if corpus:
        return corpus_bleu_internal([hypothesis], [references])
    else:
        return sentence_bleu_internal(hypothesis, references)

# ROUGE computation (max over multiple references)
def compute_rouge(hypothesis: str, references: Sequence[str]):
    """Compute ROUGE-1/2/Lsum F1, taking the maximum over multiple references.
    Returns a dict with keys 'rouge1', 'rouge2', 'rougeLsum' in [0,1].
    """
    try:
        from rouge_score import rouge_scorer
    except Exception:
        return None
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    best = {"rouge1": 0.0, "rouge2": 0.0, "rougeLsum": 0.0}
    for ref in references:
        scores = scorer.score(ref, hypothesis)
        best["rouge1"] = max(best["rouge1"], float(scores["rouge1"].fmeasure))
        best["rouge2"] = max(best["rouge2"], float(scores["rouge2"].fmeasure))
        best["rougeLsum"] = max(best["rougeLsum"], float(scores["rougeLsum"].fmeasure))
    return best

#  BERTScore computation (max over multiple references)
def compute_bertscore(hypothesis: str, references: Sequence[str], lang: str = "pt"):
    """Compute BERTScore F1, taking maximum over refs. Returns float in [0,1].
    Note: first run downloads a large transformer model.
    """
    try:
        from bert_score import BERTScorer
    except Exception:
        return None
    try:
        scorer = BERTScorer(lang=lang, rescale_with_baseline=True)
        f1_best: float = 0.0
        for ref in references:
            P, R, F1 = scorer.score([hypothesis], [ref])
            f1_val = float(F1.mean().item())
            if f1_val > f1_best:
                f1_best = f1_val
        return f1_best
    except Exception:
        return None

#  BLEU diagnostics helper 
def bleu_diagnostics(hypothesis: str, references: Sequence[str], max_n: int = 4):
    """Return detailed BLEU components for constructive analysis.
    The function returns: precisions per n (with smoothing applied when zero),
    brevity penalty, candidate length, chosen reference length, and example
    overlapping n-grams (top unigrams and bigrams).
    """
    from collections import Counter

    hyp_tok = simple_tokenize(hypothesis)
    refs_tok = [simple_tokenize(r) for r in references]
    if not hyp_tok or not any(refs_tok):
        return {
            "precisions": [0.0]*max_n,
            "brevity_penalty": 0.0,
            "candidate_len": len(hyp_tok),
            "ref_len": 0,
            "overlaps": {"top_1gram": [], "top_2gram": []}
        }

    precisions: List[float] = []
    for n in range(1, max_n+1):
        hyp_counts = Counter(ngrams(hyp_tok, n))
        max_ref_counts: dict = {}
        for ref in refs_tok:
            ref_counts = Counter(ngrams(ref, n))
            for gram, count in ref_counts.items():
                max_ref_counts[gram] = max(max_ref_counts.get(gram, 0), count)
        clipped = {gram: min(count, max_ref_counts.get(gram, 0)) for gram, count in hyp_counts.items()}
        den = max(1, sum(hyp_counts.values()))
        num = sum(clipped.values())
        p = (num/den) if den > 0 else 0.0
        if p == 0.0:
            p = 1e-9
        precisions.append(p)

    cand_len = len(hyp_tok)
    ref_len = closest_ref_length(cand_len, [len(rt) for rt in refs_tok])
    bp = brevity_penalty(cand_len, ref_len)

    overlaps = {}
    for n in (1, 2):
        hyp_counts = Counter(ngrams(hyp_tok, n))
        max_ref_counts = {}
        for ref in refs_tok:
            ref_counts = Counter(ngrams(ref, n))
            for gram, count in ref_counts.items():
                max_ref_counts[gram] = max(max_ref_counts.get(gram, 0), count)
        common = []
        for gram, count in hyp_counts.items():
            match = min(count, max_ref_counts.get(gram, 0))
            if match > 0:
                common.append((gram, match))
        common.sort(key=lambda x: x[1], reverse=True)
        overlaps[f"top_{n}gram"] = [" ".join(g) for g, _ in common[:10]]

    return {
        "precisions": precisions,
        "brevity_penalty": bp,
        "candidate_len": cand_len,
        "ref_len": ref_len,
        "overlaps": overlaps
    }

# diversidade: distinct e self-BLEU

def compute_distinct_scores(hypotheses: Sequence[str]) -> Tuple[float, float]:
    """distinct-1 e distinct-2 sobre o conjunto de hipoteses.
    distinct-n = (# n-grams unicos) / (# n-grams totais), agregando todas as hipoteses.
    """
    tokens_list = [simple_tokenize(h) for h in hypotheses]
    # unigramas
    total_unigrams = sum(len(toks) for toks in tokens_list)
    uniq_unigrams = set()
    for toks in tokens_list:
        for t in toks:
            uniq_unigrams.add(t)
    distinct1 = (len(uniq_unigrams) / total_unigrams) if total_unigrams > 0 else 0.0
    # bigramas
    total_bigrams = 0
    uniq_bigrams = set()
    for toks in tokens_list:
        for i in range(0, max(0, len(toks)-1)):
            bg = (toks[i], toks[i+1])
            uniq_bigrams.add(bg)
            total_bigrams += 1
    distinct2 = (len(uniq_bigrams) / total_bigrams) if total_bigrams > 0 else 0.0
    return distinct1, distinct2

def compute_self_bleu(hypotheses: Sequence[str], use_sacrebleu: bool=True) -> Tuple[float, float, List[float]]:
    """Self-BLEU medio e desvio: para cada h_i, BLEU(h_i, refs = hipoteses \ h_i)."""
    import math
    n = len(hypotheses)
    if n <= 1:
        return 0.0, 0.0, []
    scores: List[float] = []
    for i, hyp in enumerate(hypotheses):
        refs = [hypotheses[j] for j in range(n) if j != i]
        score = compute_bleu(hyp, refs, use_sacrebleu=use_sacrebleu, corpus=False)
        scores.append(float(score))
    mean = sum(scores)/len(scores) if scores else 0.0
    var = sum((s-mean)**2 for s in scores)/len(scores) if scores else 0.0
    std = math.sqrt(var)
    return mean, std, scores

# IO helpers

def load_references_from_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().strip()
    # supports: JSON list ["ref1","ref2"], '---' separators, blank-line blocks, or one-per-line fallback
    try:
        arr = json.loads(data)
        if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
            return arr
    except Exception:
        pass
    # split by '---'
    if "---" in data:
        parts = [p.strip() for p in data.split("---")]
        return [p for p in parts if p]
    # split by blank-line blocks
    blocks: List[str] = []
    buf: List[str] = []
    for line in data.splitlines():
        if line.strip() == "":
            if buf:
                blocks.append(" ".join(buf).strip())
                buf = []
        else:
            buf.append(line.strip())
    if buf:
        blocks.append(" ".join(buf).strip())
    # fallback: one reference per non-empty line
    if len(blocks) <= 1:
        blocks = [ln.strip() for ln in data.splitlines() if ln.strip()]
    return [b for b in blocks if b]

def parse_hypotheses_data(data: str) -> List[str]:
    """Suporta JSON list, separador '---', blocos separados por linhas em branco, ou uma por linha."""
    data = data.strip()
    if not data:
        return []
    try:
        arr = json.loads(data)
        if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
            return [x.strip() for x in arr if x.strip()]
    except Exception:
        pass
    if "---" in data:
        parts = [p.strip() for p in data.split("---")]
        return [p for p in parts if p]
    # blocos por linhas em branco
    blocks: List[str] = []
    buf: List[str] = []
    for line in data.splitlines():
        if line.strip() == "":
            if buf:
                blocks.append(" ".join(buf).strip())
                buf = []
        else:
            buf.append(line.strip())
    if buf:
        blocks.append(" ".join(buf).strip())
    if len(blocks) <= 1:
        blocks = [ln.strip() for ln in data.splitlines() if ln.strip()]
    return [b for b in blocks if b]

def load_hypotheses_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    return parse_hypotheses_data(data)

# OpenRouter client helpers

def _select_openrouter_api_key(model: str, override: str | None = None) -> str:
    """Pick API key for OpenRouter.
    Priority: explicit override -> OPENROUTER_API_KEY_<VENDOR> -> OPENROUTER_API_KEY.
    Vendor is the prefix before '/': e.g., 'openai/gpt-5' -> 'openai'.
    """
    if override and override.strip():
        return override.strip()
    vendor = ""
    if "/" in model:
        vendor = model.split("/", 1)[0].lower().replace("-", "_")
    env_try = []
    if vendor:
        env_try.append(f"OPENROUTER_API_KEY_{vendor.upper()}")
    env_try.append("OPENROUTER_API_KEY")
    for name in env_try:
        val = os.getenv(name, "")
        if val:
            return val
    raise RuntimeError(
        "Nenhuma chave OpenRouter encontrada. Defina OPENROUTER_API_KEY ou OPENROUTER_API_KEY_<VENDOR>."
    )

def call_deepseek(prompt: str, model: str="deepseek/deepseek-chat", max_tokens: int=400, temperature: float=0.3, image_url: str="", api_key_override: str | None = None, reasoning_effort: str | None = None) -> str:
    """
    Generate text using OpenRouter-compatible API (OpenAI client).
    Works with DeepSeek and GPT-5 ids. Optional image_url for vision models.
    Requires env var OPENROUTER_API_KEY.
    """
    api_key = _select_openrouter_api_key(model, api_key_override)
    try:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        # Optional OpenRouter headers
        site_url = os.getenv("OPENROUTER_SITE_URL", "")
        site_title = os.getenv("OPENROUTER_SITE_TITLE", "")
        extra_headers = {}
        if site_url:
            extra_headers["HTTP-Referer"] = site_url
        if site_title:
            extra_headers["X-Title"] = site_title

        # Build messages: text only or multimodal if image_url is provided
        if image_url:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
            msgs = [{"role": "user", "content": content}]
        else:
            msgs = [{"role": "user", "content": prompt}]

        # Optional reasoning body for models that support it (e.g., gpt-5)
        extra_body = None
        if reasoning_effort:
            extra_body = {"reasoning": {"effort": reasoning_effort}}

        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers if extra_headers else None,
            extra_body=extra_body,
        )
        # Extracao robusta de texto (cobre respostas multimodais e modelos de raciocinio)
        if not getattr(resp, "choices", None) or len(resp.choices) == 0:
            raise RuntimeError("Resposta sem choices do modelo.")
        msg = getattr(resp.choices[0], "message", None)
        # 1) content como string direto
        if msg is not None:
            c = getattr(msg, "content", None)
            if isinstance(c, str) and c.strip():
                return c.strip()
            # 2) content como lista de partes multimodais
            if isinstance(c, list):
                texts = []
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        t = part.get("text", "")
                        if isinstance(t, str) and t:
                            texts.append(t)
                joined = " ".join(texts).strip()
                if joined:
                    return joined
            # 3) alguns provedores colocam texto em campo reasoning
            r = getattr(msg, "reasoning", None)
            if isinstance(r, dict):
                rt = r.get("content") or r.get("text")
                if isinstance(rt, str) and rt.strip():
                    return rt.strip()
        # 4) fallback: campo output_text (nem todos possuem)
        ot = getattr(resp, "output_text", None)
        if isinstance(ot, str) and ot.strip():
            return ot.strip()
        # 5) tentativa de retry: remover reasoning e enviar como texto simples
        try:
            resp2 = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=extra_headers if extra_headers else None,
            )
            if getattr(resp2, "choices", None):
                msg2 = getattr(resp2.choices[0], "message", None)
                c2 = getattr(msg2, "content", None)
                if isinstance(c2, str) and c2.strip():
                    return c2.strip()
        except Exception:
            pass
        raise RuntimeError("Resposta vazia do modelo. Verifique chave, acesso ao modelo e quotas.")
    except Exception as e:
        raise RuntimeError(f"Falha ao chamar DeepSeek via OpenRouter: {e}")

# CLI

def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--reference", type=str, default=None)
    parser.add_argument("--ref-file", type=str, default=None)
    parser.add_argument("--hypothesis", type=str, default=None)
    parser.add_argument("--hyp-file", type=str, default=None, help="Arquivo com hipoteses (JSON list, '---' ou linhas)")
    parser.add_argument("--model", type=str, default="deepseek/deepseek-chat")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--use-sacrebleu", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # flags for extra metrics
    parser.add_argument("--rouge", action="store_true", help="Compute ROUGE metrics")
    parser.add_argument("--bertscore", action="store_true", help="Compute BERTScore (downloads model on first run)")
    parser.add_argument("--distinct", action="store_true", help="Compute distinct-1/2 on a set of hypotheses")
    parser.add_argument("--self-bleu", dest="self_bleu", action="store_true", help="Compute Self-BLEU on a set of hypotheses")
    # existing diagnostics flag
    parser.add_argument("--diagnostics", action="store_true", help="Print BLEU components and overlaps")
    args = parser.parse_args()

    # coletas de referencia
    references = []
    if args.reference:
        references.append(args.reference)
    if args.ref_file:
        references.extend(load_references_from_file(args.ref_file))
    if not references:
        print(" Voce precisa fornecer pelo menos uma referencia (--reference ou --ref-file).", file=sys.stderr)
        sys.exit(2)

    # conjunto de hipoteses (para distinct/self-BLEU)
    hyp_set: List[str] = []
    if args.hyp_file:
        try:
            hyp_set = load_hypotheses_from_file(args.hyp_file)
        except Exception as e:
            print(f" Falha ao carregar hipoteses de {args.hyp_file}: {e}", file=sys.stderr)
            sys.exit(2)

    # hipotese unica (gera ou usa) para metricas unitarias
    generated = False
    if args.hypothesis:
        hypothesis = args.hypothesis
    else:
        if not args.prompt and not hyp_set:
            print(" Sem --hypothesis, sem --prompt e sem --hyp-file. Forneca uma das opcoes.", file=sys.stderr)
            sys.exit(2)
        hypothesis = None
        if args.prompt:
            try:
                hypothesis = call_deepseek(args.prompt, model=args.model, max_tokens=args.max_tokens, temperature=args.temperature)
                generated = True
            except Exception as e:
                print(f"Nao foi possivel gerar com DeepSeek ({e}).", file=sys.stderr)
                print("   Voce ainda pode avaliar BLEU passando --hypothesis com sua saida ou usar --hyp-file.", file=sys.stderr)
                sys.exit(3)
        elif hyp_set:
            # se apenas conjunto foi fornecido, usar a primeira hipotese para metricas unitarias
            hypothesis = hyp_set[0]

    # metricas unitarias
    bleu_sent = compute_bleu(hypothesis, references, use_sacrebleu=args.use_sacrebleu, corpus=False)
    bleu_corp = compute_bleu(hypothesis, references, use_sacrebleu=args.use_sacrebleu, corpus=True)

    rouge_scores = None
    if args.rouge:
        rouge_scores = compute_rouge(hypothesis, references)

    bertscore_f1 = None
    if args.bertscore:
        bertscore_f1 = compute_bertscore(hypothesis, references, lang="pt")

    # metricas de conjunto
    distinct_res: Tuple[float, float] = (0.0, 0.0)
    self_bleu_mean = 0.0
    self_bleu_std = 0.0
    self_bleu_list: List[float] = []
    if (args.distinct or args.self_bleu) and len(hyp_set) < 2:
        print(" Aviso: para distinct/self-BLEU forneca ao menos 2 hipoteses via --hyp-file.", file=sys.stderr)
    else:
        if args.distinct and hyp_set:
            distinct_res = compute_distinct_scores(hyp_set)
        if args.self_bleu and hyp_set:
            m, s, lst = compute_self_bleu(hyp_set, use_sacrebleu=args.use_sacrebleu)
            self_bleu_mean, self_bleu_std, self_bleu_list = m, s, lst

    if not args.quiet:
        print("=== BLEU minimo (sentence & corpus) ===")
        print(f"Hipotese ({'gerada' if generated else 'fornecida'}):\n{hypothesis}\n")
        print(f"Referencias ({len(references)}):")
        for i, r in enumerate(references, 1):
            print(f"  [{i}] {r}")
        print("\nResultados:")
        print(f"   Sentence-BLEU: {bleu_sent:.4f}")
        print(f"   Corpus-BLEU:   {bleu_corp:.4f}")
        if rouge_scores is not None:
            print("   ROUGE:")
            print(f"     ROUGE-1 (F1):   {rouge_scores['rouge1']:.4f}")
            print(f"     ROUGE-2 (F1):   {rouge_scores['rouge2']:.4f}")
            print(f"     ROUGE-Lsum (F1): {rouge_scores['rougeLsum']:.4f}")
        if bertscore_f1 is not None:
            print(f"   BERTScore (F1): {bertscore_f1:.4f}")
        if args.diagnostics:
            diag = bleu_diagnostics(hypothesis, references)
            p_str = ", ".join(f"{p:.4f}" for p in diag["precisions"]) 
            print("\nBLEU diagnostics:")
            print(f"   BP: {diag['brevity_penalty']:.4f}  c_len: {diag['candidate_len']}  r_len: {diag['ref_len']}")
            print(f"   p1..p4: {p_str}")
            tops1 = diag.get("overlaps", {}).get("top_1gram", [])
            tops2 = diag.get("overlaps", {}).get("top_2gram", [])
            if tops1:
                print("   top 1-grams: " + ", ".join(tops1))
            if tops2:
                print("   top 2-grams: " + ", ".join(tops2))
        if args.distinct and hyp_set:
            print("\nDiversidade (conjunto):")
            print(f"   distinct-1: {distinct_res[0]:.4f}")
            print(f"   distinct-2: {distinct_res[1]:.4f}")
        if args.self_bleu and hyp_set:
            print("   Self-BLEU mean: {:.4f} (std {:.4f})".format(self_bleu_mean, self_bleu_std))
            print("   Self-BLEU por hipotese:")
            for idx, val in enumerate(self_bleu_list, 1):
                print(f"     H{idx}: {val:.4f}")

    # saida em JSON 
    out = {
        "hypothesis": hypothesis,
        "references": references,
        "sentence_bleu": bleu_sent,
        "corpus_bleu": bleu_corp,
        "rouge": rouge_scores,
        "bertscore_f1": bertscore_f1,
        "distinct": {
            "distinct1": distinct_res[0],
            "distinct2": distinct_res[1],
            "num_hypotheses": len(hyp_set)
        } if hyp_set else None,
        "self_bleu": {
            "mean": self_bleu_mean,
            "std": self_bleu_std,
            "per_hypothesis": self_bleu_list,
            "num_hypotheses": len(hyp_set)
        } if hyp_set else None
    }
    print("\nJSON:", json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
