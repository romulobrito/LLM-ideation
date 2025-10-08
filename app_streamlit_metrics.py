"""
App Streamlit minimo para avaliar geracoes com BLEU/ROUGE/BERTScore.
Permite:
- Inserir referencias (uma por linha ou JSON list)
- Gerar hipotese via DeepSeek (OpenRouter) ou informar hipotese manualmente
- Calcular BLEU (com opcao sacrebleu), ROUGE e BERTScore
- Exibir diagnosticos (p1..p4, BP, c_len, r_len, n-grams)
- Calcular diversidade em conjunto: distinct-1/2 e Self-BLEU (opcional)

Requer variavel OPENROUTER_API_KEY no ambiente para gerar via DeepSeek.
"""

import os
from typing import List, Sequence, Optional, Dict, Any, Tuple

import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import io
import matplotlib.pyplot as plt
from nngs import nngs_curve, NNGSResult, load_array

# Optional semantic embeddings
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore


def _get_st_model(model_name: str = "all-mpnet-base-v2", device: Optional[str] = None):
    """Load a sentence-transformers model once (cached). Uses CPU fallback by default."""
    if SentenceTransformer is None:
        return None
    if device is None:
        use_cuda = bool(torch and torch.cuda.is_available())
        device = "cuda" if use_cuda else "cpu"
    key = f"_st_model_{model_name}_{device}"
    if key not in st.session_state:
        st.session_state[key] = SentenceTransformer(model_name, device=device)
    return st.session_state[key]


# Importa funcoes do modulo existente
from bleu_minimal_deepseek import (
    compute_bleu,
    compute_rouge,
    compute_bertscore,
    bleu_diagnostics,
    call_deepseek,
)
# Tokenizador simples reutilizado para distinct
from bleu_minimal_deepseek import simple_tokenize


def parse_references(raw: str) -> List[str]:
    """Converte entrada de referencias em lista de strings.
    Suporta JSON list, separador '---', blocos separados por linha em branco,
    ou uma referencia por linha (fallback). Limpa prefixos [n], aspas e virgulas finais.
    """
    import json, re

    def _clean_block(text: str) -> str:
        s = text.strip()
        # remover prefixo numerado tipo [1]
        s = re.sub(r'^\[\d+\]\s*', "", s)
        # remover aspas externas e virgula final
        s = re.sub(r'^"', "", s)
        s = re.sub(r'",\s*$', "", s)
        s = re.sub(r'"\s*$', "", s)
        # colapsar espacos
        s = re.sub(r'\s+', " ", s).strip()
        return s

    raw_clean = (raw or "").strip()
    if not raw_clean:
        return []

    # Tenta JSON list diretamente
    try:
        value = json.loads(raw_clean)
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            return [_clean_block(x) for x in value if _clean_block(x)]
    except Exception:
        pass

    # Separador '---' em blocos
    if "---" in raw_clean:
        parts = [p.strip() for p in raw_clean.split("---")]
        cleaned = []
        for p in parts:
            if not p:
                continue
            # juntar linhas internas e limpar
            lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
            s = _clean_block(" ".join(lines))
            if s:
                cleaned.append(s)
        return cleaned

    # Blocos por linha em branco
    blocks: List[str] = []
    buf: List[str] = []
    for line in raw_clean.splitlines():
        if line.strip() == "":
            if buf:
                blocks.append(" ".join(buf).strip())
                buf = []
        else:
            buf.append(line.strip())
    if buf:
        blocks.append(" ".join(buf).strip())

    if blocks:
        cleaned = []
        for b in blocks:
            s = _clean_block(b)
            if s:
                cleaned.append(s)
        if cleaned:
            return cleaned

    # Fallback: uma por linha
    refs = [_clean_block(ln) for ln in raw_clean.splitlines() if ln.strip()]
    return [r for r in refs if r]


def load_references_from_fs(path: str) -> List[str]:
    """Carrega referencias a partir de um arquivo ou pasta.
    - Se for arquivo: interpreta o conteudo com parse_references (suporta JSON, '---', blocos, linhas).
    - Se for pasta: coleta recursivamente *.txt e *.md; cada arquivo vira uma referencia (multi-paragrafo preservado).
    """
    import os, glob
    p = (path or "").strip()
    if not p:
        return []
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = f.read()
            return parse_references(data)
        except Exception:
            return []
    if os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, "**", "*.txt"), recursive=True) +
                       glob.glob(os.path.join(p, "**", "*.md"), recursive=True))
        refs: List[str] = []
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    s = f.read().strip()
                if s:
                    refs.append(s)
            except Exception:
                continue
        return refs
    return []

def parse_hypotheses(raw: str) -> List[str]:
    """Converte entrada de hipoteses em lista de textos.
    Suporta JSON list de strings, separador por linhas com linhas em branco,
    ou separadores '---'.
    """
    import json

    raw_clean = (raw or "").strip()
    if not raw_clean:
        return []
    # JSON list
    try:
        value = json.loads(raw_clean)
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            return [x.strip() for x in value if x.strip()]
    except Exception:
        pass
    # Separar por '---'
    if "---" in raw_clean:
        parts = [p.strip() for p in raw_clean.split("---")]
        return [p for p in parts if p]
    # Separar por blocos (linhas em branco duplas)
    blocks: List[str] = []
    buf: List[str] = []
    for line in raw_clean.splitlines():
        if line.strip() == "":
            if buf:
                blocks.append(" ".join(buf).strip())
                buf = []
        else:
            buf.append(line.strip())
    if buf:
        blocks.append(" ".join(buf).strip())
    # Se ainda sobrou uma unica linha, tratar como uma hipotese por linha
    if len(blocks) <= 1:
        blocks = [ln.strip() for ln in raw_clean.splitlines() if ln.strip()]
    return [b for b in blocks if b]


def compute_distinct_scores(hypotheses: List[str]) -> Tuple[float, float]:
    """Calcula distinct-1 e distinct-2 sobre o conjunto de hipoteses.
    distinct-n = (# n-grams unicos) / (# n-grams totais) agregando sobre todas as hipoteses.
    """
    from collections import Counter

    tokens_list = [simple_tokenize(h) for h in hypotheses]
    # Unigramas
    total_unigrams = sum(len(toks) for toks in tokens_list)
    uniq_unigrams = set()
    for toks in tokens_list:
        uniq_unigrams.update(toks)
    distinct1 = (len(uniq_unigrams) / total_unigrams) if total_unigrams > 0 else 0.0
    # Bigramas
    def bigrams(toks: List[str]) -> List[Tuple[str, str]]:
        return [(toks[i], toks[i+1]) for i in range(0, max(0, len(toks)-1))]
    total_bigrams = sum(max(0, len(toks)-1) for toks in tokens_list)
    uniq_bigrams = set()
    for toks in tokens_list:
        for bg in bigrams(toks):
            uniq_bigrams.add(bg)
    distinct2 = (len(uniq_bigrams) / total_bigrams) if total_bigrams > 0 else 0.0
    return distinct1, distinct2


def compute_self_bleu(hypotheses: List[str], use_sacrebleu: bool = True) -> Tuple[float, float, List[float]]:
    """Calcula Self-BLEU medio e desvio padrao.
    Para cada hipotese h_i, BLEU(h_i, refs = hipoteses \ h_i). Retorna (media, desvio, lista)."""
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


# Simple BoW embedding para NNGS automático (sem deps extras)
def _embed_texts_bow(texts: List[str], existing_vocab: Optional[Dict[str, int]] = None) -> Tuple[Any, Dict[str, int]]:
    from collections import Counter
    import numpy as _np
    # vocab
    if existing_vocab is None:
        vocab: Dict[str, int] = {}
        idx = 0
        for t in texts:
            for tok in simple_tokenize(t):
                if tok not in vocab:
                    vocab[tok] = idx
                    idx += 1
    else:
        vocab = existing_vocab
    n = len(texts)
    d = len(vocab)
    M = _np.zeros((n, d), dtype=float)
    for i, t in enumerate(texts):
        cnt = Counter(simple_tokenize(t))
        for tok, c in cnt.items():
            j = vocab.get(tok)
            if j is not None:
                M[i, j] = float(c)
    norms = _np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    M = M / norms
    return M, vocab

def _load_env_robusto() -> None:
    # tenta carregar .env de varios locais comuns
    candidatos = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for p in candidatos:
        try:
            if p.exists():
                load_dotenv(dotenv_path=str(p), override=True)
        except Exception:
            pass

def _load_default_prompt() -> str:
    """Tenta carregar prompt padrao do arquivo 'prompt'.
    Se nao existir, retorna um fallback interno do experimento.
    """
    try:
        candidates = [
            Path.cwd() / "prompt",
            Path(__file__).resolve().parent / "prompt",
            Path(__file__).resolve().parent.parent / "prompt",
        ]
        for p in candidates:
            if p.exists() and p.is_file():
                return p.read_text(encoding="utf-8")
    except Exception:
        pass
    # Fallback interno (ASCII)
    return (
        "Consider the following writing contest invitation:\n"
        "------\n"
        "Strangers Again\n\n"
        "I have been thinking about the need to feel connected.\n"
        "This week, write stories about yearning and connection,\n"
        "even when things are complicated or out of reach.\n"
        "------\n\n"
        "And also this directive:\n"
        "------\n"
        "Center your story around two characters who like each other\n"
        "but do not get a happily ever after.\n"
        "------\n\n"
        "Task: creatively generate 2 short-story ideas, each with 150 words,\n"
        "distinct in theme, tone, and concept. Name them 1.txt and 2.txt.\n"
    )

def main() -> None:
    """UI principal do app Streamlit minimo."""
    _load_env_robusto()

    st.set_page_config(
        page_title="Métricas de texto (BLEU/ROUGE/BERTScore)",
        page_icon="📏",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.title("Avaliação de Respostas com BLEU/ROUGE/BERTScore")
    st.caption(
        "Cole referências e gere uma hipótese via LLM ou informe uma hipótese manualmente."
    )

    with st.sidebar:
        st.header("Configuracoes")
        mode = st.radio(
            "Modo de entrada",
            options=["Gerar via LLM", "Usar hipótese fornecida"],
            index=1,
        )
        use_sacrebleu = st.checkbox("Usar sacrebleu", value=True)
        use_rouge = st.checkbox("Calcular ROUGE", value=True)
        use_bertscore = st.checkbox("Calcular BERTScore", value=False,
                                    help="Pode baixar modelo grande na primeira vez")
        show_diag = st.checkbox("Mostrar diagnósticos (BLEU)", value=True)
        
        eval_set = st.checkbox("Avaliar conjunto (distinct-1/2 e Self-BLEU)", value=False)

        st.divider()
        st.subheader("Parâmetros de geração (se usar LLM)")
        provider = st.selectbox(
            "Provedor/Modelo",
            options=["DeepSeek", "GPT-5 (OpenRouter)", "Custom (OpenRouter)"],
            index=0,
            help="Todos os modelos usam a API do OpenRouter via cliente OpenAI.",
        )
        default_model = "deepseek/deepseek-chat" if provider == "DeepSeek" else ("openai/gpt-5" if provider == "GPT-5 (OpenRouter)" else "")
        model_name = st.text_input("Model ID (OpenRouter)", value=default_model, help="Ex.: deepseek/deepseek-chat, openai/gpt-5")
        image_url = st.text_input("Image URL (opcional, para modelos vision)", value="", help="Se preenchido, envia prompt multimodal (texto+imagem)")
        api_key_override = st.text_input("API Key (override opcional)", value="", type="password", help="Se preencher, esta chave sera usada em vez das do .env")
        # Raciocinio (apenas modelos que suportam)
        reasoning_effort = st.selectbox(
            "Raciocinio (se suportado pelo modelo)",
            options=["Nenhum", "low", "medium", "high"],
            index=0,
            help="Para modelos como openai/gpt-5 que expõem reasoning, define o effort.",
        )
        # Diagnostico de chave: mostra qual variavel sera usada (sem expor o valor)
        with st.expander("Diagnostico de chave OpenRouter", expanded=False):
            mdl = (model_name or "").strip()
            vendor = mdl.split("/", 1)[0].lower().replace("-", "_") if "/" in mdl else ""
            cand = ([f"OPENROUTER_API_KEY_{vendor.upper()}"] if vendor else []) + ["OPENROUTER_API_KEY"]
            found = None
            for name in cand:
                if os.getenv(name, ""):
                    found = name
                    break
            if api_key_override.strip():
                st.info("Override ativo: uma chave fornecida na UI sera usada nesta chamada.")
            st.write("Modelo:", mdl or "(vazio)")
            st.write("Vendor detectado:", vendor or "(indefinido)")
            st.write("Variaveis candidatas:", ", ".join(cand) if cand else "OPENROUTER_API_KEY")
            st.write("Variavel selecionada:", found or "nenhuma encontrada")

        st.divider()
        st.subheader("Fonte das Referências")
        refs_path = st.text_input(
            "Caminho (arquivo ou pasta)", value="",
            help="Se preencher, o app vai carregar referencias deste caminho. Em pasta, cada arquivo .txt/.md vira uma referencia."
        )
        temperature = st.slider("Temperatura", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        max_tokens = st.number_input("Max tokens", min_value=16, max_value=4000, value=400, step=16)

    st.subheader("Referências")
    refs_raw = st.text_area(
        "Digite referências (uma por linha) ou JSON list",
        height=140,
        placeholder='Exemplos:\n- Texto referencia 1\n- Texto referencia 2\nOu JSON: ["Ref 1","Ref 2"]',
        disabled=bool((refs_path or "").strip()),
    )
    # Preview quando usar caminho
    if (refs_path or "").strip():
        loaded_preview = load_references_from_fs(refs_path)
        st.caption(f"Carregadas {len(loaded_preview)} referências de '{refs_path}'. O campo acima fica desabilitado quando caminho for fornecido.")

    hypothesis: Optional[str] = None
    prompt_text: Optional[str] = None

    if mode == "Usar hipotese fornecida":
        st.subheader("Hipotese")
        hypothesis = st.text_area(
            "Cole a hipotese (texto a avaliar)",
            height=160,
            placeholder="Cole aqui o texto gerado que deseja avaliar",
        )
    else:
        st.subheader("Prompt para geração via LLM (OpenRouter)")
        prompt_text = st.text_area(
            "Prompt (contexto/pergunta)",
            height=160,
            placeholder="Descreva o que o modelo deve responder",
        )
        if not (prompt_text or "").strip():
            # Carrega prompt padrao se usuario nao forneceu
            default_prompt = _load_default_prompt()
            st.info("Usando prompt padrao do experimento (arquivo 'prompt' ou fallback interno).")
            prompt_text = default_prompt
        st.info("Requer OPENROUTER_API_KEY no ambiente (.env) para gerar via OpenRouter.")

    # entrada de conjunto de hipoteses
    hyps_raw: Optional[str] = None
    if eval_set:
        st.subheader("Conjunto de hipoteses (opcional)")
        hyps_raw = st.text_area(
            "Cole multiplas hipoteses para medir diversidade (JSON list, '---' como separador, ou blocos separados por linha em branco)",
            height=160,
            placeholder='Ex.: ["texto 1", "texto 2", "texto 3"]\nOu:\ntexto 1\n\ntexto 2\n\ntexto 3\nOu use --- como separador',
        )

    do_run = st.button("Calcular metricas", type="primary")

    if not do_run:
        st.stop()

    # Valida referencias (preferir caminho se fornecido)
    references = []
    if (refs_path or "").strip():
        references = load_references_from_fs(refs_path)
    else:
        references = parse_references(refs_raw)
    if not references:
        st.error("Forneça ao menos uma referência.")
        st.stop()

    # Obtem hipotese (gera se necessario)
    if hypothesis is None:
        # Nao bloqueie por falta de OPENROUTER_API_KEY generica; a selecao de chave por vendor
        # e feita dentro de call_deepseek (override -> OPENROUTER_API_KEY_<VENDOR> -> OPENROUTER_API_KEY)
        if not prompt_text or not prompt_text.strip():
            st.error("Forneça um prompt para geração.")
            st.stop()
        try:
            with st.spinner("Gerando resposta via OpenRouter..."):
                hypothesis = call_deepseek(
                    prompt=prompt_text.strip(),
                    model=model_name.strip() or "deepseek/deepseek-chat",
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    image_url=image_url.strip(),
                    api_key_override=(api_key_override.strip() or None),
                    reasoning_effort=(None if reasoning_effort == "Nenhum" else reasoning_effort),
                )
            if not hypothesis or not hypothesis.strip():
                st.error("LLM retornou resposta vazia. Verifique o Model ID, a API key e eventuais limites de cota.")
                st.stop()
        except Exception as e:
            st.error(f"Falha na geração: {e}")
            st.stop()

    # Auto-parse multiple generated hypotheses (split by '---') for diversity/NNGS
    auto_hyps: List[str] = []
    if hypothesis and "---" in hypothesis:
        parts = [p.strip() for p in hypothesis.split("---")]
        auto_hyps = [p for p in parts if p]

    # If we have >=3 references and >=3 auto_hyps, compute a quick semantic or BoW NNGS
    if len(references) >= 3 and len(auto_hyps) >= 3:
        st.markdown("---")
        st.subheader("NNGS automático — Gerados vs Referências")
        emb_mode = "BoW"
        model_name_choice = None
        device_choice = "auto"
        batch_size = 32
        if SentenceTransformer is not None:
            cols = st.columns(3)
            with cols[0]:
                model_name_choice = st.selectbox(
                    "Modelo (ST)",
                    ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"],
                    index=0,
                    key="nngs_auto_st_model",
                )
            with cols[1]:
                device_choice = st.selectbox(
                    "Device", ["cuda", "cpu", "auto"], index=(0 if (torch and torch.cuda.is_available()) else 1), key="nngs_auto_device"
                )
            with cols[2]:
                default_bs = 128 if (torch and torch.cuda.is_available()) else 64
                batch_size = int(st.number_input("batch_size", min_value=8, max_value=512, value=default_bs, step=8))
        try:
            # Prefer sentence-transformers if available
            if SentenceTransformer is not None:
                dev = None if device_choice == "auto" else device_choice
                try:
                    model = _get_st_model(model_name_choice or "all-MiniLM-L6-v2", device=dev)
                except Exception:
                    # If CUDA/device init fails, fallback to CPU model
                    model = _get_st_model(model_name_choice or "all-MiniLM-L6-v2", device="cpu")
                    device_choice = "cpu"
                emb_mode = f"ST: {model_name_choice} ({getattr(model, 'device', 'cpu')})"
                try:
                    X_refs = model.encode(references, normalize_embeddings=True, batch_size=batch_size)
                    Y_hyps = model.encode(auto_hyps, normalize_embeddings=True, batch_size=batch_size)
                except Exception:
                    # Retry on CPU if encode fails on CUDA
                    model = _get_st_model(model_name_choice or "all-MiniLM-L6-v2", device="cpu")
                    emb_mode = f"ST: {model_name_choice} (cpu)"
                    X_refs = model.encode(references, normalize_embeddings=True, batch_size=max(16, batch_size // 2))
                    Y_hyps = model.encode(auto_hyps, normalize_embeddings=True, batch_size=max(16, batch_size // 2))
            else:
                X_refs, vocab = _embed_texts_bow(references)
                Y_hyps, _ = _embed_texts_bow(auto_hyps, existing_vocab=vocab)
            # Align by min length
            import numpy as _np
            n = min(len(references), len(auto_hyps))
            X_use = X_refs[:n]
            Y_use = Y_hyps[:n]
            from nngs import nngs_curve, NNGSResult
            k_max_quick = max(2, min(10, n - 1))
            k_values_quick = list(range(1, k_max_quick + 1))
            res_quick: NNGSResult = nngs_curve(X_use, Y_use, k_values=k_values_quick, metric="cosine")
            figq = plt.figure(figsize=(5, 3))
            plt.plot(res_quick.k_values, res_quick.nngs_values, label="NNGS(k)")
            plt.plot(res_quick.k_values, res_quick.baseline_values, linestyle="--", label="H(k)")
            plt.xlabel("k")
            plt.ylabel("Jaccard médio")
            plt.legend()
            st.pyplot(figq, clear_figure=True)
            st.caption(f"Auto-NNGS usando {emb_mode} (n={n}). Use a seção NNGS para uploads de X/Y se preferir.")
        except Exception as e:
            st.info(f"NNGS automático não pôde ser calculado: {e}")
    elif len(auto_hyps) > 0 and len(auto_hyps) < 3:
        st.info("Várias saídas detectadas, mas são necessárias pelo menos 3 para calcular NNGS automático.")

    # Calcula metricas unitarias
    try:
        bleu_sent = compute_bleu(hypothesis, references, use_sacrebleu=use_sacrebleu, corpus=False)
        bleu_corp = compute_bleu(hypothesis, references, use_sacrebleu=use_sacrebleu, corpus=True)
    except Exception as e:
        st.error(f"Erro ao calcular BLEU: {e}")
        st.stop()

    rouge_scores: Optional[Dict[str, float]] = None
    if use_rouge:
        try:
            rouge_scores = compute_rouge(hypothesis, references)
        except Exception as e:
            st.warning(f"ROUGE indisponível: {e}")

    bert_f1: Optional[float] = None
    if use_bertscore:
        try:
            bert_f1 = compute_bertscore(hypothesis, references, lang="pt")
        except Exception as e:
            st.warning(f"BERTScore indisponível: {e}")

    # Exibe resultados unitarios
    st.subheader("Resultados")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Sentence-BLEU", f"{bleu_sent:.4f}")
    with c2:
        st.metric("Corpus-BLEU", f"{bleu_corp:.4f}")
    with c3:
        if rouge_scores is not None:
            st.metric("ROUGE-1 (F1)", f"{rouge_scores.get('rouge1', 0.0):.4f}")
        else:
            st.metric("ROUGE-1 (F1)", "-")

    c4, c5, c6 = st.columns(3)
    with c4:
        if rouge_scores is not None:
            st.metric("ROUGE-2 (F1)", f"{rouge_scores.get('rouge2', 0.0):.4f}")
        else:
            st.metric("ROUGE-2 (F1)", "-")
    with c5:
        if rouge_scores is not None:
            st.metric("ROUGE-Lsum (F1)", f"{rouge_scores.get('rougeLsum', 0.0):.4f}")
        else:
            st.metric("ROUGE-Lsum (F1)", "-")
    with c6:
        if bert_f1 is not None:
            st.metric("BERTScore (F1)", f"{bert_f1:.4f}")
        else:
            st.metric("BERTScore (F1)", "-")

    st.markdown("---")
    st.subheader("Hipotese avaliada")
    # aumentar altura para evitar aparente "corte" visual ao exibir respostas longas
    st.text_area("Texto", value=hypothesis or "", height=420, disabled=True)
    # opcional: download do texto completo
    st.download_button(
        label="Baixar hipotese (.txt)",
        data=(hypothesis or "").encode("utf-8"),
        file_name="hipotese.txt",
        mime="text/plain",
    )

    st.subheader("Referencias")
    for i, r in enumerate(references, start=1):
        st.write(f"[{i}] {r}")

    if show_diag:
        try:
            diag = bleu_diagnostics(hypothesis, references)
            st.markdown("---")
            st.subheader("BLEU Diagnostics")
            st.write(
                f"BP: {diag.get('brevity_penalty', 0.0):.4f}  c_len: {diag.get('candidate_len', 0)}  r_len: {diag.get('ref_len', 0)}"
            )
            pvals = diag.get("precisions", [])
            st.write("p1..p4: " + ", ".join(f"{p:.4f}" for p in pvals))
            tops = diag.get("overlaps", {})
            if tops.get("top_1gram"):
                st.write("top 1-grams: " + ", ".join(tops["top_1gram"]))
            if tops.get("top_2gram"):
                st.write("top 2-grams: " + ", ".join(tops["top_2gram"]))
        except Exception as e:
            st.warning(f"Falha ao gerar diagnósticos: {e}")

    # Diversidade do conjunto - calcular antes do JSON
    distinct_vals: Optional[Tuple[float, float]] = None
    self_bleu_vals: Optional[Tuple[float, float, List[float]]] = None
    hyps: List[str] = []
    if eval_set and hyps_raw and hyps_raw.strip():
        hyps = parse_hypotheses(hyps_raw)
        hyps = [h for h in hyps if h.strip()]
        if len(hyps) >= 2:
            st.markdown("---")
            st.subheader("Diversidade do conjunto")
            d1, d2 = compute_distinct_scores(hyps)
            mean_sb, std_sb, sb_list = compute_self_bleu(hyps, use_sacrebleu)
            distinct_vals = (d1, d2)
            self_bleu_vals = (mean_sb, std_sb, sb_list)
            cA, cB, cC = st.columns(3)
            with cA:
                st.metric("distinct-1", f"{d1:.4f}")
            with cB:
                st.metric("distinct-2", f"{d2:.4f}")
            with cC:
                st.metric("Self-BLEU (mean)", f"{mean_sb:.4f}")
            # Tabela simples com Self-BLEU por hipotese
            if sb_list:
                st.write("Self-BLEU por hipotese:")
                for idx, val in enumerate(sb_list, start=1):
                    st.write(f"- H{idx}: {val:.4f}")
        else:
            st.info("Forneça pelo menos 2 hipóteses para calcular diversidade.")

    # JSON de saida (inclui conjunto se disponivel)
    import json as _json
    output: Dict[str, Any] = {
        "hypothesis": hypothesis,
        "references": references,
        "sentence_bleu": float(bleu_sent),
        "corpus_bleu": float(bleu_corp),
        "rouge": rouge_scores,
        "bertscore_f1": bert_f1,
        "distinct": (
            {
                "distinct1": float(distinct_vals[0]),
                "distinct2": float(distinct_vals[1]),
                "num_hypotheses": len(hyps),
            }
            if distinct_vals is not None
            else None
        ),
        "self_bleu": (
            {
                "mean": float(self_bleu_vals[0]),
                "std": float(self_bleu_vals[1]),
                "per_hypothesis": [float(x) for x in (self_bleu_vals[2] or [])],
                "num_hypotheses": len(hyps),
            }
            if self_bleu_vals is not None
            else None
        ),
    }
    st.markdown("---")
    st.subheader("JSON")
    st.code(_json.dumps(output, ensure_ascii=True, indent=2))

    # NNGS section (optional)
    st.markdown("---")
    with st.expander("NNGS — Similaridade de Grafos kNN (opcional)", expanded=False):
        st.caption("Compare dois espaços de embeddings pareados (X e Y) usando a curva NNGS(k) e o baseline analítico H(k).")
        metric = st.selectbox("Métrica de distância", ["cosine", "euclidean", "manhattan"], index=0, key="nngs_metric")
        k_max = st.number_input("k máximo (<= n-1)", min_value=1, value=15, step=1, key="nngs_kmax")
        do_hist = st.checkbox("Mostrar histograma de Jaccard (k final)", value=True, key="nngs_hist")

        st.subheader("Carregue os embeddings")
        st.write("• X (source) e Y (target) devem ter o mesmo número de linhas. Formatos: .npy ou .csv (sem header).")
        upX = st.file_uploader("Arquivo X (.npy ou .csv)", type=["npy", "csv"], key="nngs_upx")
        upY = st.file_uploader("Arquivo Y (.npy ou .csv)", type=["npy", "csv"], key="nngs_upy")
        example = st.checkbox("Usar exemplo sintético (duas nuvens rotacionadas + ruído)", value=False, key="nngs_example")

        X = Y = None
        if example:
            st.info("Gerando exemplo sintético (n=300, d=32).")
            import numpy as np
            rng = np.random.default_rng(42)
            n, d = 300, 32
            X = rng.normal(size=(n, d))
            Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
            Y = (X @ Q) + 0.05 * rng.normal(size=(n, d))
        elif upX is not None and upY is not None:
            try:
                X = load_array(io.BytesIO(upX.read()))
                Y = load_array(io.BytesIO(upY.read()))
            except Exception as e:
                st.error(f"Falha ao carregar arrays: {e}")

        if X is not None and Y is not None:
            n = X.shape[0]
            if n != Y.shape[0]:
                st.error("X e Y precisam ter o mesmo número de linhas.")
            elif n < 3:
                st.error("Precisa de pelo menos 3 amostras.")
            else:
                k_max = int(min(int(k_max), n - 1))
                k_values = list(range(1, k_max + 1))
                with st.spinner("Calculando NNGS(k)..."):
                    try:
                        res: NNGSResult = nngs_curve(X, Y, k_values=k_values, metric=metric)
                    except Exception as e:
                        st.error(f"Falha no cálculo do NNGS: {e}")
                        res = None  # type: ignore
                if res is not None:
                    st.subheader("Curva NNGS vs k")
                    fig1 = plt.figure(figsize=(6, 4))
                    plt.plot(res.k_values, res.nngs_values, label="NNGS(k)")
                    plt.plot(res.k_values, res.baseline_values, linestyle="--", label="Baseline H(k)")
                    plt.xlabel("k (vizinhos)")
                    plt.ylabel("similaridade média de Jaccard")
                    plt.legend()
                    st.pyplot(fig1, clear_figure=True)

                    last_k = res.k_values[-1]
                    st.write(f"**Resumo:** n = {n},  último k = {last_k},  NNGS(k={last_k}) = {res.nngs_values[-1]:.4f},  H(k={last_k}) = {res.baseline_values[-1]:.4f}")

                    if res.per_point_jaccard is not None and do_hist:
                        st.subheader(f"Distribuição de Jaccard por amostra (k={last_k})")
                        fig2 = plt.figure(figsize=(6, 4))
                        plt.hist(res.per_point_jaccard, bins=20)
                        plt.xlabel("Jaccard por amostra")
                        plt.ylabel("contagem")
                        st.pyplot(fig2, clear_figure=True)

        st.markdown("---")
        st.markdown("**Dicas:**")
        st.markdown("- Varie **k** para focar em estrutura local (k pequeno) vs global (k grande).")
        st.markdown("- A curva **H(k)** é o baseline analítico para dados i.i.d. aleatórios: se NNGS ~ H(k), há pouca correspondência estrutural entre X e Y.")


if __name__ == "__main__":
    main() 