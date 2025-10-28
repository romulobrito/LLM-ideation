#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface Streamlit para Refinement Loop.

Permite executar o loop de refinamento iterativo com interface visual,
mostrando progresso em tempo real e resultados de cada iteracao.

Uso:
    streamlit run app_refinement.py
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv

from refinement_loop import RefinementLoop, RefinementConfig, IterationResult

# Import UMAP with fallback
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# Configuracao da pagina
st.set_page_config(
    page_title="Refinement Loop - Diversidade LLM",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Carregar variaveis de ambiente
def load_env_vars() -> None:
    """Carrega variaveis de ambiente do .env."""
    env_paths = [
        Path.home() / "Documentos" / "MAI-DAI-USP" / "experimento_convergencia_visualizacao_metricas" / ".env",
        Path(".env"),
        Path("../.env"),
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"[ENV] Carregado: {env_path}")
            break


load_env_vars()


# Titulo principal
st.title("🔄 Refinement Loop - Diversidade LLM")

# Opção de carregar experimento anterior
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Carregar Experimento Salvo")

exp_dir = Path("exp_refinement")
if exp_dir.exists() and (exp_dir / "summary.json").exists():
    if st.sidebar.button("📊 Visualizar Último Experimento", use_container_width=True):
        st.session_state['show_saved_exp'] = True
else:
    st.sidebar.info("Nenhum experimento salvo ainda")

# Verificar se deve mostrar experimento salvo
if st.session_state.get('show_saved_exp', False):
    st.subheader("📊 Experimento Salvo")
    
    # Carregar summary
    summary_file = exp_dir / "summary.json"
    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Iterações", summary["total_iterations"])
    with col2:
        st.metric("Convergiu?", "✅" if summary["converged"] else "❌")
    with col3:
        st.metric("Melhor Dist. Média", f"{summary['best_avg_distance']:.4f}")
    with col4:
        st.metric("Melhor Dist. Mínima", f"{summary['best_min_distance']:.4f}")
    
    # Gráfico de convergência
    st.subheader("📈 Gráfico de Convergência")
    
    iterations = [item["iteration"] for item in summary["iterations"]]
    avg_distances = [item["avg_distance"] for item in summary["iterations"]]
    min_distances = [item["min_distance"] for item in summary["iterations"]]
    
    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        x=iterations, y=avg_distances,
        mode='lines+markers', name='Dist. Média',
        line=dict(color='blue', width=2), marker=dict(size=8)
    ))
    fig_conv.add_trace(go.Scatter(
        x=iterations, y=min_distances,
        mode='lines+markers', name='Dist. Mínima',
        line=dict(color='green', width=2), marker=dict(size=8)
    ))
    fig_conv.update_layout(
        xaxis_title="Iteração",
        yaxis_title="Distância Coseno",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig_conv, use_container_width=True)
    
    # UMAP 3D
    if UMAP_AVAILABLE:
        st.subheader("🌐 Visualização UMAP 3D")
        
        try:
            from experiment_iterativo import get_embedder, embed_texts, load_references_from_fs
            
            embedder_name = summary['config']['embedder']
            with st.spinner(f"Carregando embedder `{embedder_name}`..."):
                embedder = get_embedder(embedder_name)
            
            with st.spinner("Processando embeddings..."):
                # Carregar TODAS as ideias humanas disponíveis
                human_ideas_all = load_references_from_fs("ideas-exp/human")
                human_embeddings_all = embed_texts(embedder, human_ideas_all)
                
                # Tentar descobrir quantas foram usadas no experimento
                # Ler do summary (novo campo adicionado)
                num_human_used = summary['config'].get('num_human_ideas', len(human_ideas_all))
                
                # Se o summary não tem essa info (experimentos antigos), assumir todas
                if num_human_used > len(human_ideas_all):
                    num_human_used = len(human_ideas_all)
                
                # Informar ao usuário
                st.info(f"""
                **📊 Ideias Humanas:**
                - Total disponíveis: {len(human_ideas_all)}
                - Usadas no experimento: {num_human_used}
                - Não usadas (contexto): {len(human_ideas_all) - num_human_used}
                """)
                
                all_embeddings = []
                all_labels = []
                all_types = []
                all_iterations_viz = []
                all_distances = []
                all_used_status = []  # NOVO: rastrear se foi usada ou não
                
                # Adicionar ideias humanas (USADAS e NÃO USADAS)
                for i, emb in enumerate(human_embeddings_all):
                    all_embeddings.append(emb)
                    all_labels.append(f"Humana {i+1}")
                    
                    # Determinar se foi usada no experimento
                    if i < num_human_used:
                        all_types.append("humana_usada")
                        all_used_status.append("Usada")
                    else:
                        all_types.append("humana_nao_usada")
                        all_used_status.append("Não Usada")
                    
                    all_iterations_viz.append(0)
                    all_distances.append(0.0)
                
                # Iterações
                for iter_num in range(1, summary["total_iterations"] + 1):
                    iter_file = exp_dir / f"iteration_{iter_num:02d}.json"
                    if iter_file.exists():
                        with open(iter_file, "r", encoding="utf-8") as f:
                            iter_data = json.load(f)
                        
                        iter_embeddings = embed_texts(embedder, iter_data["generated_ideas"])
                        
                        for i, emb in enumerate(iter_embeddings):
                            all_embeddings.append(emb)
                            all_labels.append(f"Iter {iter_num} - Ideia {i+1}")
                            all_types.append("gerada")
                            all_used_status.append("N/A")
                            all_iterations_viz.append(iter_num)
                            
                            # Calcular distância apenas para as humanas USADAS
                            human_emb_used = human_embeddings_all[:num_human_used]
                            dists = [np.dot(emb, h_emb) for h_emb in human_emb_used]
                            min_dist = 1.0 - max(dists)
                            all_distances.append(min_dist)
                
                embeddings_array = np.array(all_embeddings)
                
                n_neighbors = min(15, len(embeddings_array) - 1)
                umap_reducer = UMAP(
                    n_components=3, random_state=42,
                    n_neighbors=n_neighbors, min_dist=0.1, metric='cosine'
                )
                embeddings_umap = umap_reducer.fit_transform(embeddings_array)
                
                df_umap = pd.DataFrame({
                    'x': embeddings_umap[:, 0],
                    'y': embeddings_umap[:, 1],
                    'z': embeddings_umap[:, 2],
                    'label': all_labels,
                    'tipo': all_types,
                    'used_status': all_used_status,
                    'iteracao': all_iterations_viz,
                    'distancia': all_distances
                })
                
                fig_umap = go.Figure()
                
                # Humanas USADAS (vermelho escuro)
                df_human_used = df_umap[df_umap['tipo'] == 'humana_usada']
                if len(df_human_used) > 0:
                    fig_umap.add_trace(go.Scatter3d(
                        x=df_human_used['x'], y=df_human_used['y'], z=df_human_used['z'],
                        mode='markers',
                        marker=dict(size=12, color='darkred', symbol='diamond', line=dict(color='black', width=2)),
                        name='🔴 Humanas (USADAS)',
                        text=df_human_used['label'],
                        hovertemplate="<b>%{text} - USADA</b><br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<br>UMAP3: %{z:.3f}<extra></extra>"
                    ))
                
                # Humanas NÃO USADAS (laranja claro)
                df_human_not_used = df_umap[df_umap['tipo'] == 'humana_nao_usada']
                if len(df_human_not_used) > 0:
                    fig_umap.add_trace(go.Scatter3d(
                        x=df_human_not_used['x'], y=df_human_not_used['y'], z=df_human_not_used['z'],
                        mode='markers',
                        marker=dict(size=10, color='orange', symbol='diamond', line=dict(color='darkorange', width=1), opacity=0.5),
                        name='🟠 Humanas (NÃO USADAS)',
                        text=df_human_not_used['label'],
                        hovertemplate="<b>%{text} - NÃO USADA</b><br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<br>UMAP3: %{z:.3f}<extra></extra>"
                    ))
                
                # Geradas
                df_generated = df_umap[df_umap['tipo'] == 'gerada']
                if len(df_generated) > 0:
                    best_idx = df_generated['distancia'].idxmin()
                    best_row = df_generated.loc[best_idx]
                    df_gen_normal = df_generated.drop(best_idx)
                    
                    if len(df_gen_normal) > 0:
                        fig_umap.add_trace(go.Scatter3d(
                            x=df_gen_normal['x'], y=df_gen_normal['y'], z=df_gen_normal['z'],
                            mode='markers',
                            marker=dict(
                                size=6, color=df_gen_normal['iteracao'],
                                colorscale='Viridis',
                                colorbar=dict(title="Iteração", x=1.15),
                                symbol='circle'
                            ),
                            name='Ideias Geradas',
                            text=df_gen_normal['label'],
                            customdata=df_gen_normal['distancia'],
                            hovertemplate="<b>%{text}</b><br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<br>UMAP3: %{z:.3f}<br>Dist: %{customdata:.4f}<extra></extra>"
                        ))
                    
                    fig_umap.add_trace(go.Scatter3d(
                        x=[best_row['x']], y=[best_row['y']], z=[best_row['z']],
                        mode='markers',
                        marker=dict(size=20, color='gold', symbol='diamond', line=dict(color='orange', width=4)),
                        name='⭐ Melhor Ideia',
                        text=[best_row['label']],
                        customdata=[best_row['distancia']],
                        hovertemplate="<b>⭐ %{text}</b><br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<br>UMAP3: %{z:.3f}<br>Dist: %{customdata:.4f}<extra></extra>"
                    ))
                
                fig_umap.update_layout(
                    scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
                    title="Evolução das Ideias (UMAP 3D)",
                    height=700
                )
                
                st.plotly_chart(fig_umap, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Erro ao gerar visualizações: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Tabela de resultados
    st.subheader("📋 Tabela de Resultados")
    df_results = pd.DataFrame([
        {
            "Iteração": item["iteration"],
            "Dist. Média": f"{item['avg_distance']:.4f}",
            "Dist. Mínima": f"{item['min_distance']:.4f}",
            "Num Ideias": item["num_ideas"]
        }
        for item in summary["iterations"]
    ])
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    # Botão para voltar
    if st.button("🔙 Voltar para Configuração", type="primary"):
        st.session_state['show_saved_exp'] = False
        st.rerun()
    
    st.stop()

# # Aviso sobre correções implementadas
# st.info("""
# **✅ CORREÇÕES IMPLEMENTADAS (v2.0):**

# 1. **🔄 Histórico Acumulado**: CRITIQUE agora analisa até 20 ideias geradas anteriormente (não apenas as 5 últimas)
#    - **Antes**: Feedback oscilava porque comparava apenas 5 ideias por vez
#    - **Depois**: Feedback estável baseado no histórico acumulado

# 2. **🎯 Temperature Ajustada**: Padrão mudou de 0.8 → 0.5
#    - **Recomendado**: 0.3-0.5 para convergência estável
#    - **Evitar**: >0.7 causa alta variação

# 3. **📊 Detecção de Divergência**: Sistema agora distingue convergência vs. divergência
#    - **Antes**: Qualquer estagnação = "convergiu"
#    - **Depois**: Analisa se melhorou/piorou/estabilizou

# **💡 RESULTADO ESPERADO**: Gráficos mais suaves e convergência real em vez de zigue-zague
# """)

# st.markdown("---")


# Sidebar: Configuracoes
st.sidebar.header("⚙️ Configuracao")

# Modelo LLM com provedor
st.sidebar.subheader("🤖 Modelo LLM")

provider = st.sidebar.selectbox(
    "Provedor:",
    ["OpenAI Direto", "GPT-4o via OpenRouter", "DeepSeek (OpenRouter)", "GPT-5 (OpenRouter) ⚠️ Experimental", "Personalizado"],
    index=0,
    help="Provedor de API para chamadas LLM"
)

if provider == "OpenAI Direto":
    model = st.sidebar.selectbox(
        "Modelo:",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0,
        help="Modelo OpenAI direto"
    )
elif provider == "GPT-4o via OpenRouter":
    model = "openai/gpt-4o"
    st.sidebar.info(f"📌 Modelo: {model}")
    st.sidebar.success("✅ **ESTÁVEL**: Recomendado para uso em produção")
elif provider == "DeepSeek (OpenRouter)":
    model = st.sidebar.selectbox(
        "Modelo:",
        ["deepseek/deepseek-chat", "deepseek/deepseek-v3.2-exp", "deepseek/deepseek-r1"],
        index=0,
        help="Modelos DeepSeek via OpenRouter"
    )
    
    if model == "deepseek/deepseek-r1":
        st.sidebar.warning("""
        ⚠️ **ATENÇÃO**: `deepseek-r1` pode falhar no CRITIQUE!
        
        O modelo retorna apenas reasoning sem JSON final.
        
        **RECOMENDAÇÃO**: Use `deepseek-chat` ou `deepseek-v3.2-exp`.
        """)
    elif model == "deepseek/deepseek-v3.2-exp":
        st.sidebar.warning("""
        🆕 **EXPERIMENTAL**: DeepSeek V3.2-Exp
        
        ⚠️ **ATENÇÃO**: Pode falhar com prompts grandes!
        
        Problemas conhecidos:
        - Rate limit mais restritivo
        - Timeout em prompts longos
        
        **RECOMENDAÇÃO**: Use `deepseek-chat` para estabilidade.
        """)
    else:
        st.sidebar.success("✅ **ESTÁVEL**: Funciona bem, barato e eficiente!")
    
    st.sidebar.info("""
    **📋 Modelos:**
    - `deepseek-chat`: ✅ Recomendado (estável)
    - `deepseek-v3.2-exp`: 🆕 Novo (experimental)
    - `deepseek-r1`: ⚠️ Pode falhar no CRITIQUE
    """)
elif provider == "GPT-5 (OpenRouter) ⚠️ Experimental":
    model = "openai/gpt-5"
    st.sidebar.info(f"📌 Modelo: {model}")
    st.sidebar.error("⚠️ **EXPERIMENTAL**: GPT-5 é instável e pode retornar apenas reasoning sem JSON.")
    st.sidebar.warning("💡 **RECOMENDAÇÃO**: Use gpt-4o-mini ou gpt-4o para melhor estabilidade.")
else:
    model = st.sidebar.text_input(
        "Nome do modelo:",
        value="gpt-4o-mini",
        help="Digite o nome completo do modelo (ex: openai/gpt-4o-mini)"
    )

# Embedder
st.sidebar.subheader("🧮 Modelo de Embeddings")

embedder_name = st.sidebar.selectbox(
    "Modelo:",
    [
        "all-MiniLM-L6-v2",
        "text-embedding-3-large",
        "text-embedding-3-small",
        "all-mpnet-base-v2",
        "paraphrase-multilingual-MiniLM-L12-v2"
    ],
    index=0,
    help="Modelo para gerar embeddings e calcular distancias semanticas"
)

# Info sobre embeddings OpenAI
if "text-embedding" in embedder_name:
    st.sidebar.info(f"""
    **🌐 OpenAI Embeddings**
    
    **Modelo**: {embedder_name}
    
    **Dimensões**:
    - 3-large: 3072D (melhor qualidade)
    - 3-small: 1536D (mais rapido)
    
    **Custo**: ~$0.13/1M tokens (3-large)
    
    **Vantagem**: Captura nuances e detalhes que 
    embeddings locais (384D) nao detectam.
    
    ⚠️ **Requer**: OPENAI_API_KEY no .env
    """)
    
    # Verificar se OPENAI_API_KEY existe
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        st.sidebar.error("""
        ❌ **ERRO**: OPENAI_API_KEY nao encontrada!
        
        Adicione ao arquivo `.env`:
        ```
        OPENAI_API_KEY=sk-...
        ```
        """)
else:
    st.sidebar.success(f"""
    **💻 Local (Sentence Transformers)**
    
    **Modelo**: {embedder_name}
    **Dimensões**: 384D (MiniLM) ou 768D (MPNet)
    **Custo**: Grátis (roda localmente)
    **Device**: CPU/CUDA
    """)

# Device
device = st.sidebar.selectbox(
    "Device",
    ["auto", "cpu", "cuda"],
    index=0,
    help="Dispositivo para computacao de embeddings"
)

# Parametros de convergencia
st.sidebar.subheader("Convergencia")

max_iterations = st.sidebar.slider(
    "Max Iteracoes",
    min_value=1,
    max_value=20,
    value=5,
    help="Numero maximo de iteracoes"
)

patience = st.sidebar.slider(
    "Patience",
    min_value=1,
    max_value=10,
    value=3,
    help="Numero de iteracoes sem melhoria antes de parar"
)

delta_threshold = st.sidebar.number_input(
    "Delta Threshold",
    min_value=0.001,
    max_value=0.1,
    value=0.01,
    step=0.001,
    format="%.3f",
    help="Melhoria minima para considerar progresso"
)

# Parametros de geracao
st.sidebar.subheader("Geracao")

num_ideas_per_iter = st.sidebar.slider(
    "Ideias por Iteracao",
    min_value=1,
    max_value=20,
    value=5,
    help="Numero de ideias a gerar em cada iteracao"
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=2.0,
    value=0.5,  # REDUZIDO: 0.8 → 0.5 para melhor convergência
    step=0.1,
    help="Temperatura para geracao de ideias. RECOMENDADO: 0.3-0.5 para convergência estável"
)

if temperature > 0.7:
    st.sidebar.warning("⚠️ Temperature alta (>0.7) pode causar oscilação. Recomenda-se 0.3-0.5 para convergência.")

max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=1000,
    max_value=8000,
    value=4000,
    step=500,
    help="Número máximo de tokens por resposta. Maior = respostas mais completas, mas mais caro. DeepSeek: 4000+, GPT-4o: 2000-4000"
)

reasoning_effort = st.sidebar.selectbox(
    "Reasoning Effort",
    ["None", "minimal", "low", "medium", "high"],
    index=0,
    help="Nivel de reasoning para modelos que suportam (ex: GPT-5). Use 'minimal' para GPT-5 com JSON estruturado."
)

# NOVO: Norte Fixo
st.sidebar.subheader("⭐ Norte Fixo (Experimental)")

use_north_star = st.sidebar.checkbox(
    "Usar Norte Fixo Automático",
    value=True,
    help="Gera diretrizes FIXAS analisando ideias humanas (reduz oscilacao do feedback)"
)

if use_north_star:
    north_star_model = st.sidebar.selectbox(
        "Modelo para Norte",
        ["gpt-4o", "gpt-4o-mini", "deepseek/deepseek-chat"],
        index=0,
        help="Modelo para analisar ideias humanas e gerar norte fixo (use gpt-4o para melhor qualidade)"
    )
    
    st.sidebar.info("""
    **💡 Norte Fixo:**
    - Analisa ideias humanas UMA VEZ
    - Extrai padrões fundamentais
    - Mantém direção constante
    - Reduz oscilação do feedback
    """)
else:
    north_star_model = "gpt-4o"
    st.sidebar.warning("⚠️ Feedback 100% dinâmico (pode oscilar)")

# Diretorio de saida
st.sidebar.subheader("Saida")

default_output_dir = str(Path.home() / "Documentos" / "MAI-DAI-USP" / "experimento_convergencia_visualizacao_metricas" / "exp_refinement")

output_dir = st.sidebar.text_input(
    "Diretorio de Saida",
    value=default_output_dir,
    help="Diretorio para salvar resultados"
)

st.sidebar.markdown("---")


# Area principal: Inputs
st.header("📝 Inputs do Experimento")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Convite do Concurso")
    invitation = st.text_area(
        "Invitation",
        value="""Strangers Again

I've been thinking a lot lately about the need to feel connected - to be seen, remembered, or maybe even just understood. Over the years, I've noticed how connection can manifest in the smallest of things: a shared meal, a passing glance, a familiar name we can't quite place.

This week, let's write stories about that pull between people. From fleeting relationships to chance encounters with strangers who seem like old friends, let's explore yearning and connection, even when things are complicated or just out of reach.""",
        height=250,
        help="Texto do convite do concurso de escrita"
    )

with col2:
    st.subheader("Diretiva Original")
    directive = st.text_area(
        "Directive",
        value="Center your story around two characters who like each other but don't get a happily ever after.",
        height=100,
        help="Diretiva original para geracao de ideias"
    )
    
    st.subheader("Ideias Humanas")
    human_ideas_mode = st.radio(
        "Fonte das Ideias Humanas",
        ["Digitar manualmente", "Carregar de arquivo"],
        index=1
    )
    
    if human_ideas_mode == "Digitar manualmente":
        human_ideas_text = st.text_area(
            "Ideias (uma por linha)",
            value="",
            height=150,
            help="Digite uma ideia humana por linha"
        )
        human_ideas = [line.strip() for line in human_ideas_text.split("\n") if line.strip()]
        
        # Slider para limitar numero de ideias (se digitadas manualmente)
        if human_ideas:
            num_human_ideas = st.slider(
                "Numero de Ideias Humanas a Usar",
                min_value=1,
                max_value=len(human_ideas),
                value=min(5, len(human_ideas)),
                help="Quantas ideias humanas usar no experimento"
            )
            human_ideas = human_ideas[:num_human_ideas]
    else:
        human_ideas_path = st.text_input(
            "Caminho para arquivo",
            value=str(Path.home() / "Documentos" / "MAI-DAI-USP" / "experimento_convergencia_visualizacao_metricas" / "ideas-exp" / "human"),
            help="Diretorio com arquivos .txt de ideias humanas"
        )
        
        # Carregar temporariamente para contar e permitir selecao
        temp_path = Path(human_ideas_path)
        if temp_path.exists():
            temp_ideas = []
            if temp_path.is_dir():
                for txt_file in sorted(temp_path.glob("*.txt")):
                    try:
                        content = txt_file.read_text(encoding="utf-8").strip()
                        if content:
                            temp_ideas.append(content)
                    except:
                        pass
            else:
                try:
                    content = temp_path.read_text(encoding="utf-8").strip()
                    temp_ideas = [line.strip() for line in content.split("\n") if line.strip()]
                except:
                    pass
            
            if temp_ideas:
                st.info(f"📁 Encontradas {len(temp_ideas)} ideias humanas")
                num_human_ideas = st.slider(
                    "Numero de Ideias Humanas a Usar",
                    min_value=1,
                    max_value=len(temp_ideas),
                    value=min(5, len(temp_ideas)),
                    help="Quantas ideias humanas usar no experimento"
                )
            else:
                num_human_ideas = None
        else:
            num_human_ideas = None
        
        human_ideas = []

st.markdown("---")

# Validacao de chaves de API
st.header("🔑 Status de Configuracao")

col_status1, col_status2 = st.columns(2)

with col_status1:
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    has_openrouter_key = bool(os.getenv("OPENROUTER_API_KEY")) or bool(os.getenv("OPENROUTER_API_KEY_OPENAI")) or bool(os.getenv("OPENROUTER_API_KEY_DEEPSEEK"))
    
    if "/" in model:
        # Modelo via OpenRouter
        can_run = has_openrouter_key
        if can_run:
            st.success("✅ Chave OpenRouter configurada")
        else:
            st.error("❌ Nenhuma chave OpenRouter encontrada no .env")
            st.info("Configure OPENROUTER_API_KEY no arquivo .env")
    else:
        # Modelo OpenAI direto
        can_run = has_openai_key
        if can_run:
            st.success("✅ Chave OpenAI configurada")
        else:
            st.error("❌ OPENAI_API_KEY não encontrada no .env")
            st.info("Configure OPENAI_API_KEY no arquivo .env")

with col_status2:
    st.info(f"🤖 **Modelo selecionado:** `{model}`")
    st.info(f"🔢 **Iterações máximas:** {max_iterations}")
    st.info(f"💡 **Ideias por iteração:** {num_ideas_per_iter}")

st.markdown("---")

# Inicializar session_state para controle de execucao
if "refinement_running" not in st.session_state:
    st.session_state["refinement_running"] = False
if "refinement_stop_requested" not in st.session_state:
    st.session_state["refinement_stop_requested"] = False

# Botoes de controle
col_btn1, col_btn2 = st.columns([3, 1])

with col_btn1:
    run_button = st.button(
        "🚀 Executar Refinement Loop", 
        type="primary", 
        use_container_width=True, 
        disabled=not can_run or st.session_state["refinement_running"]
    )

with col_btn2:
    if st.session_state["refinement_running"]:
        if st.button("⛔ PARAR", type="secondary", use_container_width=True):
            st.session_state["refinement_stop_requested"] = True
            st.warning("⏸️ Solicitação de parada enviada...")
            st.rerun()

# Mostrar status de execucao
if st.session_state["refinement_running"]:
    st.info("⚙️ **Status:** Experimento em execução... Clique em **PARAR** para interromper.")

if run_button:
    
    # Validar inputs
    if not invitation.strip():
        st.error("❌ Invitation nao pode estar vazio")
        st.stop()
    
    if not directive.strip():
        st.error("❌ Directive nao pode estar vazio")
        st.stop()
    
    # Carregar ideias humanas de arquivo se necessario
    if human_ideas_mode == "Carregar de arquivo":
        ideas_path = Path(human_ideas_path)
        if not ideas_path.exists():
            st.error(f"❌ Caminho nao existe: {ideas_path}")
            st.stop()
        
        # Carregar todos os .txt do diretorio
        human_ideas = []
        if ideas_path.is_dir():
            for txt_file in sorted(ideas_path.glob("*.txt")):
                content = txt_file.read_text(encoding="utf-8").strip()
                if content:
                    human_ideas.append(content)
        else:
            content = ideas_path.read_text(encoding="utf-8").strip()
            human_ideas = [line.strip() for line in content.split("\n") if line.strip()]
        
        # Aplicar limite do slider
        if num_human_ideas and num_human_ideas < len(human_ideas):
            human_ideas = human_ideas[:num_human_ideas]
            st.info(f"✅ Usando {num_human_ideas} de {len(human_ideas)} ideias humanas de: {ideas_path}")
        else:
            st.info(f"✅ Carregadas {len(human_ideas)} ideias humanas de: {ideas_path}")
    
    if not human_ideas:
        st.error("❌ Nenhuma ideia humana fornecida")
        st.stop()
    
    # Criar configuracao
    reasoning_arg = None if reasoning_effort == "None" else reasoning_effort
    
    config = RefinementConfig(
        invitation=invitation,
        directive=directive,
        human_ideas=human_ideas,
        model=model,
        embedder_name=embedder_name,
        device=device,
        max_iterations=max_iterations,
        patience=patience,
        delta_threshold=delta_threshold,
        num_ideas_per_iter=num_ideas_per_iter,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_arg,
        output_dir=Path(output_dir),
        use_north_star=use_north_star,  # NOVO
        north_star_model=north_star_model,  # NOVO
    )
    
    # Marcar como rodando
    st.session_state["refinement_running"] = True
    st.session_state["refinement_stop_requested"] = False
    
    # Criar placeholders para progresso
    progress_bar = st.progress(0.0, text="Inicializando...")
    status_text = st.empty()
    
    # Container para resultados em tempo real
    results_container = st.container()
    
    try:
        # Executar loop
        with st.spinner("Executando Refinement Loop..."):
            
            # Criar loop
            loop = RefinementLoop(config)
            
            # Armazenar resultados em session_state
            if "refinement_results" not in st.session_state:
                st.session_state["refinement_results"] = []
            
            # Executar (nota: nao conseguimos mostrar progresso em tempo real
            # dentro do loop sem modificar a classe, entao mostramos depois)
            results = loop.run()
            
            st.session_state["refinement_results"] = results
            st.session_state["refinement_config"] = config
            st.session_state["refinement_loop"] = loop
        
        # Atualizar progresso
        progress_bar.progress(1.0, text="Concluido!")
        status_text.success(f"✅ Refinement Loop concluido: {len(results)} iteracoes")
        
    except KeyboardInterrupt:
        st.warning("⏸️ Experimento interrompido pelo usuário!")
        status_text.warning("Execução interrompida")
    except Exception as e:
        st.error(f"❌ Erro ao executar Refinement Loop: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        # Marcar como nao rodando
        st.session_state["refinement_running"] = False
        st.session_state["refinement_stop_requested"] = False


# Exibir resultados se existirem
if "refinement_results" in st.session_state and st.session_state["refinement_results"]:
    
    st.markdown("---")
    st.header("📊 Resultados")
    
    results: List[IterationResult] = st.session_state["refinement_results"]
    config: RefinementConfig = st.session_state["refinement_config"]
    loop: RefinementLoop = st.session_state["refinement_loop"]
    
    # Metricas gerais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Iteracoes", len(results))
    
    with col2:
        best_avg = min(r.avg_distance for r in results)
        st.metric("Melhor Distancia Media", f"{best_avg:.4f}")
    
    with col3:
        best_min = min(r.min_distance for r in results)
        st.metric("Melhor Distancia Minima", f"{best_min:.4f}")
    
    with col4:
        convergence_status = "✅ Sim" if loop.converged else "❌ Nao"
        st.metric("Convergiu?", convergence_status)
    
    st.info(f"**Razao:** {loop.convergence_reason}")
    
    # Grafico de convergencia
    st.subheader("📈 Grafico de Convergencia")
    
    iterations = [r.iteration for r in results]
    avg_distances = [r.avg_distance for r in results]
    min_distances = [r.min_distance for r in results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=avg_distances,
        mode="lines+markers",
        name="Distancia Media",
        line=dict(color="blue", width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=min_distances,
        mode="lines+markers",
        name="Distancia Minima",
        line=dict(color="green", width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        xaxis_title="Iteracao",
        yaxis_title="Distancia Coseno",
        hovermode="x unified",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ============================================================================
    # VISUALIZACAO UMAP 3D
    # ============================================================================
    if UMAP_AVAILABLE:
        st.markdown("---")
        st.subheader("🌐 Visualização UMAP 3D - Evolução das Ideias")
        
        with st.spinner("Calculando UMAP 3D..."):
            try:
                # Coletar todos os embeddings
                all_embeddings = []
                all_labels = []
                all_types = []
                all_iterations = []
                all_distances = []
                
                # Adicionar ideias humanas (referências)
                for emb in loop.human_embeddings:
                    all_embeddings.append(emb)
                    all_labels.append("Referencia Humana")
                    all_types.append("humana")
                    all_iterations.append(0)
                    all_distances.append(0.0)
                
                # Adicionar ideias geradas de cada iteração
                for result in results:
                    # Embeddings das ideias geradas nesta iteração
                    # Usar embed_texts() para suportar tanto OpenAI quanto local
                    from experiment_iterativo import embed_texts
                    iter_embeddings = embed_texts(loop.embedder, result.generated_ideas)
                    
                    for i, emb in enumerate(iter_embeddings):
                        all_embeddings.append(emb)
                        all_labels.append(f"Iter {result.iteration} - Ideia {i+1}")
                        all_types.append("gerada")
                        all_iterations.append(result.iteration)
                        
                        # Calcular distância mínima dessa ideia para as humanas
                        dists = [np.dot(emb, h_emb) for h_emb in loop.human_embeddings]
                        min_dist = 1.0 - max(dists)  # cosine distance = 1 - cosine similarity
                        all_distances.append(min_dist)
                
                # Converter para numpy array
                embeddings_array = np.array(all_embeddings)
                
                # Aplicar UMAP 3D
                n_neighbors = min(15, len(embeddings_array) - 1)
                umap_reducer = UMAP(
                    n_components=3,
                    random_state=42,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric='cosine'
                )
                embeddings_umap = umap_reducer.fit_transform(embeddings_array)
                
                # Criar DataFrame
                df_umap = pd.DataFrame({
                    'x': embeddings_umap[:, 0],
                    'y': embeddings_umap[:, 1],
                    'z': embeddings_umap[:, 2],
                    'label': all_labels,
                    'tipo': all_types,
                    'iteracao': all_iterations,
                    'distancia': all_distances
                })
                
                # Criar figura 3D
                fig_umap = go.Figure()
                
                # Adicionar referências humanas
                df_human = df_umap[df_umap['tipo'] == 'humana']
                fig_umap.add_trace(go.Scatter3d(
                    x=df_human['x'],
                    y=df_human['y'],
                    z=df_human['z'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='diamond',
                        line=dict(color='darkred', width=2)
                    ),
                    name='Ideias Humanas',
                    text=df_human['label'],
                    hovertemplate="<b>%{text}</b><br>" +
                                "UMAP 1: %{x:.3f}<br>" +
                                "UMAP 2: %{y:.3f}<br>" +
                                "UMAP 3: %{z:.3f}<extra></extra>"
                ))
                
                # Adicionar ideias geradas (coloridas por iteração)
                df_generated = df_umap[df_umap['tipo'] == 'gerada']
                
                if len(df_generated) > 0:
                    # Encontrar a melhor ideia global (menor distância)
                    best_idx = df_generated['distancia'].idxmin()
                    best_row = df_generated.loc[best_idx]
                    
                    # Plotar ideias normais (exceto a melhor)
                    df_gen_normal = df_generated.drop(best_idx)
                    
                    if len(df_gen_normal) > 0:
                        fig_umap.add_trace(go.Scatter3d(
                            x=df_gen_normal['x'],
                            y=df_gen_normal['y'],
                            z=df_gen_normal['z'],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color=df_gen_normal['iteracao'],
                                colorscale='Viridis',
                                colorbar=dict(
                                    title="Iteração",
                                    x=1.15,
                                    xanchor='left',
                                    thickness=15,
                                    len=0.7
                                ),
                                symbol='circle',
                                line=dict(color='white', width=0.5)
                            ),
                            name='Ideias Geradas',
                            text=df_gen_normal['label'],
                            customdata=df_gen_normal['distancia'],
                            hovertemplate="<b>%{text}</b><br>" +
                                        "UMAP 1: %{x:.3f}<br>" +
                                        "UMAP 2: %{y:.3f}<br>" +
                                        "UMAP 3: %{z:.3f}<br>" +
                                        "Dist. Coseno: %{customdata:.4f}<extra></extra>"
                        ))
                    
                    # Destacar a MELHOR ideia global
                    fig_umap.add_trace(go.Scatter3d(
                        x=[best_row['x']],
                        y=[best_row['y']],
                        z=[best_row['z']],
                        mode='markers',
                        marker=dict(
                            size=20,
                            color='gold',
                            symbol='diamond',  # Scatter3D não suporta 'star', usar 'diamond'
                            line=dict(color='orange', width=4)
                        ),
                        name='⭐ Melhor Ideia',
                        text=[best_row['label']],
                        customdata=[best_row['distancia']],
                        hovertemplate="<b>⭐ %{text}</b><br>" +
                                    "UMAP 1: %{x:.3f}<br>" +
                                    "UMAP 2: %{y:.3f}<br>" +
                                    "UMAP 3: %{z:.3f}<br>" +
                                    "Dist. Coseno: %{customdata:.4f}<extra></extra>"
                    ))
                
                # Layout
                fig_umap.update_layout(
                    scene=dict(
                        xaxis_title="UMAP 1",
                        yaxis_title="UMAP 2",
                        zaxis_title="UMAP 3",
                        bgcolor='rgba(240,240,240,0.9)',
                        xaxis=dict(backgroundcolor="white", gridcolor="lightgray"),
                        yaxis=dict(backgroundcolor="white", gridcolor="lightgray"),
                        zaxis=dict(backgroundcolor="white", gridcolor="lightgray"),
                    ),
                    title=dict(
                        text="Evolução das Ideias no Espaço Semântico (UMAP 3D)",
                        x=0.5,
                        xanchor='center'
                    ),
                    showlegend=True,
                    legend=dict(
                        x=0.02,
                        y=0.98,
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='black',
                        borderwidth=1
                    ),
                    height=700,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_umap, use_container_width=True)
                
                st.info("""
                **📊 Como interpretar o UMAP 3D:**
                
                - **Diamantes vermelhos**: Ideias humanas (referências)
                - **Círculos coloridos**: Ideias geradas (cor = iteração)
                - **Diamante dourado ⭐**: Melhor ideia gerada (menor distância)
                
                **💡 O que observar:**
                - Ideias mais próximas dos diamantes vermelhos são mais parecidas com as humanas
                - A trajetória das cores mostra a evolução ao longo das iterações
                - Se as ideias estão se aproximando (convergência) ou afastando (divergência) das humanas
                
                ⚠️ **Importante**: UMAP reduz de 384D para 3D (~99% de perda de informação).
                As distâncias visuais são aproximadas. Use os valores de hover para distâncias reais.
                """)
                
            except Exception as e:
                st.error(f"❌ Erro ao calcular UMAP 3D: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.markdown("---")
        st.warning("📦 UMAP não disponível. Instale com: `pip install umap-learn`")
    
    # Tabela de resultados
    st.markdown("---")
    st.subheader("📋 Tabela de Resultados")
    
    df = pd.DataFrame([
        {
            "Iteracao": r.iteration,
            "Distancia Media": f"{r.avg_distance:.4f}",
            "Distancia Minima": f"{r.min_distance:.4f}",
            "Num Ideias": len(r.generated_ideas),
            "Timestamp": r.timestamp
        }
        for r in results
    ])
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Detalhes de cada iteracao
    st.subheader("🔍 Detalhes por Iteracao")
    
    selected_iteration = st.selectbox(
        "Selecione uma iteracao",
        options=range(1, len(results) + 1),
        format_func=lambda x: f"Iteracao {x}"
    )
    
    selected_result = results[selected_iteration - 1]
    
    tab1, tab2, tab3 = st.tabs(["Critique JSON", "Bullets", "Ideias Geradas"])
    
    with tab1:
        st.json(selected_result.critique_json)
    
    with tab2:
        st.text(selected_result.bullets)
    
    with tab3:
        for i, idea in enumerate(selected_result.generated_ideas, 1):
            with st.expander(f"Ideia {i}"):
                st.write(idea)
    
    # Download de resultados
    st.markdown("---")
    st.subheader("💾 Download")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download JSON completo
        summary = {
            "config": {
                "model": config.model,
                "embedder": config.embedder_name,
                "max_iterations": config.max_iterations,
                "patience": config.patience,
                "delta_threshold": config.delta_threshold,
            },
            "converged": loop.converged,
            "convergence_reason": loop.convergence_reason,
            "results": [r.to_dict() for r in results]
        }
        
        json_str = json.dumps(summary, indent=2, ensure_ascii=False)
        
        st.download_button(
            label="📥 Download JSON Completo",
            data=json_str,
            file_name=f"refinement_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Download CSV
        csv_str = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv_str,
            file_name=f"refinement_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Refinement Loop - Sistema de Refinamento Iterativo para Diversidade LLM</p>
        <p>MAI/DAI - USP | 2025</p>
    </div>
    """,
    unsafe_allow_html=True
)

