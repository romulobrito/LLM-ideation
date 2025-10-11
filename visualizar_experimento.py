#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualização gráfica dos resultados do experimento iterativo.
Mostra distâncias, embeddings e evolução das ideias.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Import UMAP with fallback
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

st.set_page_config(
    page_title="Visualização Experimento",
    page_icon="",
    layout="wide"
)

st.title(" Visualização do Experimento Iterativo")
st.markdown("Análise gráfica das distâncias e embeddings gerados")
st.markdown("---")

# Sidebar - Seleção de experimento
with st.sidebar:
    st.header(" Configurações")
    
    exp_dir = st.text_input(
        "Diretório do experimento:",
        value="exp_out"
    )
    
    exp_path = Path(exp_dir)
    
    if not exp_path.exists():
        st.error(f"Diretório não encontrado: {exp_dir}")
        st.stop()
    
    # Listar referências disponíveis
    ref_dirs = sorted([d for d in exp_path.iterdir() if d.is_dir() and d.name.startswith("ref_")])
    
    if not ref_dirs:
        st.error("Nenhuma referência encontrada")
        st.stop()
    
    st.success(f" {len(ref_dirs)} referências encontradas")
    
    ref_names = [d.name for d in ref_dirs]
    selected_ref = st.selectbox(
        "Selecione a referência:",
        ref_names,
        index=0
    )
    
    st.markdown("---")
    
    # Opções de visualização
    st.subheader("Gráficos")
    st.markdown("**Convergência:**")
    show_convergence_detailed = st.checkbox("Gráfico de convergência (Detalhado)", value=True, 
                                            help="Mostra 3 camadas: pontos, melhor histórico e trajetória escolhida")
    show_convergence_simple = st.checkbox("Gráfico de convergência (Simples)", value=False,
                                          help="Versão original mais compacta")
    st.markdown("**Embeddings:**")
    show_embeddings_3d = st.checkbox("Embeddings 3D (PCA Interativo)", value=True)
    show_embeddings_tsne = st.checkbox("Embeddings 3D (t-SNE Interativo)", value=False)
    if UMAP_AVAILABLE:
        show_embeddings_umap = st.checkbox("Embeddings 3D (UMAP Interativo)", value=True)
    else:
        show_embeddings_umap = False
        st.info(" UMAP não disponível. Instale com: pip install umap-learn")
    show_heatmap = st.checkbox("Heatmap de distâncias", value=True)
    show_best_worst = st.checkbox("Comparação melhor vs pior", value=True)
    
    st.markdown("---")
    st.subheader(" Grafos e Hierarquia")
    show_dendrogram = st.checkbox("Dendrograma Hierárquico", value=True)
    show_evolution_graph = st.checkbox("Grafo de Evolução", value=True)
    
    st.markdown("---")
    st.subheader(" Métricas de Exploração")
    show_diversity = st.checkbox("Diversidade Intra-Iteração", value=True,
                                  help="Mostra quão diferentes são as 2 ideias geradas em cada iteração")
    show_novelty = st.checkbox("Novidade (Distância às Anteriores)", value=True,
                                help="Mostra quão diferente cada ideia é de todas as anteriores")
    show_exploration_rate = st.checkbox("Taxa de Exploração vs. Refinamento", value=True,
                                        help="Mostra o percentual de iterações que exploraram vs. refinaram")
    
    st.markdown("---")
    st.subheader(" Contexto Global")
    show_all_refs_umap = st.checkbox("UMAP 3D - Todas as Referências", value=True, 
                                      help="Visualiza TODAS as referências do arquivo junto com as iterações geradas")

# Carregar dados da referência selecionada
ref_path = exp_path / selected_ref
log_file = ref_path / "log.csv"

if not log_file.exists():
    st.error(f"Arquivo log.csv não encontrado em {ref_path}")
    st.stop()

# Ler log
df = pd.read_csv(log_file)

st.header(f" Referência: {selected_ref}")

# Métricas resumidas
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de iterações", df['iter'].max())

with col2:
    best_dist = df['dist'].min()
    st.metric("Melhor distância", f"{best_dist:.4f}")

with col3:
    worst_dist = df['dist'].max()
    st.metric("Pior distância", f"{worst_dist:.4f}")

with col4:
    improvement = ((worst_dist - best_dist) / worst_dist) * 100
    st.metric("Melhoria (%)", f"{improvement:.1f}%")

st.markdown("---")

# ============================================================================
# 1. GRAFICO DE CONVERGENCIA DETALHADO (2D - CONFIÁVEL)
# ============================================================================
if show_convergence_detailed:
    st.subheader(" Gráfico de Convergência (Distância vs. Iteração)")
    
    st.info("""
     **Este gráfico mostra as distâncias REAIS no espaço de 384D.**
    Diferente do UMAP 3D (que perde 99.22% da informação), aqui as distâncias são exatas.
    """)
    
    fig = go.Figure()
    
    # 1. Pontos individuais (escolhidos e rejeitados)
    for iter_num in sorted(df['iter'].unique()):
        iter_data = df[df['iter'] == iter_num]
        
        for _, row in iter_data.iterrows():
            cand_id = row['cand_id']
            dist = row['dist']
            chosen = row['chosen_A']
            
            # Cor: verde se escolhido, vermelho se rejeitado
            color = 'green' if chosen == 1 else 'red'
            symbol = 'star' if chosen == 1 else 'circle'
            
            fig.add_trace(go.Scatter(
                x=[iter_num],
                y=[dist],
                mode='markers',
                marker=dict(size=12, color=color, symbol=symbol),
                name=f"Iter {iter_num} - Cand {cand_id}",
                showlegend=False,
                hovertemplate=f"<b>Iteração {iter_num}</b><br>" +
                              f"Candidato: {cand_id}<br>" +
                              f"Distância: {dist:.4f}<br>" +
                              f"Status: {'Escolhido (A)' if chosen else 'Rejeitado (B)'}<extra></extra>"
            ))
    
    # 2. Linha da melhor distância acumulada (MONOTÔNICA)
    best_so_far = df.groupby('iter')['dmin_so_far'].first()
    fig.add_trace(go.Scatter(
        x=best_so_far.index,
        y=best_so_far.values,
        mode='lines+markers',
        line=dict(color='blue', width=3, dash='dash'),
        marker=dict(size=10, color='blue', symbol='diamond'),
        name='Melhor Histórico (Monotônica)',
        hovertemplate="<b>Melhor até iter %{x}</b><br>Distância: %{y:.4f}<extra></extra>"
    ))
    
    # 3. Linha das ideias escolhidas (NÃO-MONOTÔNICA)
    chosen_data = df[df['chosen_A'] == 1].sort_values('iter')
    if len(chosen_data) > 0:
        fig.add_trace(go.Scatter(
            x=chosen_data['iter'],
            y=chosen_data['dist'],
            mode='lines',
            line=dict(color='green', width=2, dash='dot'),
            name='Trajetória Escolhida (Não-Monotônica)',
            hovertemplate="<b>Iter %{x}</b><br>Distância escolhida: %{y:.4f}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Evolução das Distâncias por Iteração (Espaço Real 384D)",
        xaxis_title="Iteração",
        yaxis_title="Distância Cosseno (menor = melhor)",
        hovermode='closest',
        height=600,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Análise de convergência
    col1, col2, col3 = st.columns(3)
    
    chosen_dists = chosen_data['dist'].values
    improvements = sum(1 for i in range(1, len(chosen_dists)) if chosen_dists[i] < chosen_dists[i-1])
    deteriorations = sum(1 for i in range(1, len(chosen_dists)) if chosen_dists[i] > chosen_dists[i-1])
    
    with col1:
        st.metric("Melhorias", f"{improvements} ({improvements/(len(chosen_dists)-1)*100:.1f}%)")
    with col2:
        st.metric("Pioras", f"{deteriorations} ({deteriorations/(len(chosen_dists)-1)*100:.1f}%)", 
                  delta=f"-{deteriorations}", delta_color="inverse")
    with col3:
        total_improvement = ((chosen_dists[0] - chosen_dists[-1]) / chosen_dists[0]) * 100
        st.metric("Melhoria Total", f"{total_improvement:.1f}%")
    
    # Legenda explicativa
    st.markdown("""
    **Como interpretar:**
    
    - **Estrela verde**: Ideia escolhida (menor distância naquela iteração)
    - **Círculo vermelho**: Ideia rejeitada (maior distância naquela iteração)
    - **Linha azul tracejada**: Melhor distância histórica (SEMPRE melhora ou mantém)
    -  **Linha verde pontilhada**: Trajetória das ideias escolhidas (pode piorar!)
    
    **Observações importantes:**
    1. A linha azul (melhor histórico) é **monotônica decrescente** - nunca piora
    2. A linha verde (escolhidas) é **não-monotônica** - pode subir e descer
    3. Isso mostra que o algoritmo é uma **busca estocástica**, não convergência determinística
    4. **Diferença entre linhas**: Quando verde está acima da azul, a iteração não melhorou o histórico
    """)
    
    # Análise detalhada
    with st.expander(" Ver análise detalhada de convergência"):
        st.write("**Distâncias por iteração:**")
        analysis_df = df[['iter', 'cand_id', 'dist', 'chosen_A']].copy()
        analysis_df['status'] = analysis_df['chosen_A'].map({1: 'Escolhido ', 0: 'Rejeitado ❌'})
        analysis_df = analysis_df.sort_values(['iter', 'cand_id'])
        st.dataframe(analysis_df, use_container_width=True, hide_index=True)
        
        # Estatísticas adicionais
        st.write("**Estatísticas:**")
        st.write(f"- Distância inicial (Iter 1, melhor): {chosen_dists[0]:.4f}")
        st.write(f"- Distância final (Iter {len(chosen_dists)}, escolhida): {chosen_dists[-1]:.4f}")
        st.write(f"- Melhor distância global: {df['dist'].min():.4f} (Iter {df.loc[df['dist'].idxmin(), 'iter']:.0f})")
        st.write(f"- Pior distância global: {df['dist'].max():.4f} (Iter {df.loc[df['dist'].idxmax(), 'iter']:.0f})")
        st.write(f"- Variação (range): {df['dist'].max() - df['dist'].min():.4f}")
        st.write(f"- Desvio padrão: {df['dist'].std():.4f}")

st.markdown("---")

# ============================================================================
# 1B. GRAFICO DE CONVERGENCIA SIMPLES (VERSÃO ORIGINAL)
# ============================================================================
if show_convergence_simple:
    st.subheader(" Gráfico de Convergência (Versão Simples)")
    
    fig_simple = go.Figure()
    
    # Agrupar por iteração
    for iter_num in sorted(df['iter'].unique()):
        iter_data = df[df['iter'] == iter_num]
        
        for _, row in iter_data.iterrows():
            cand_id = row['cand_id']
            dist = row['dist']
            chosen = row['chosen_A']
            
            # Cor: verde se escolhido, vermelho se rejeitado
            color = 'green' if chosen == 1 else 'red'
            symbol = 'star' if chosen == 1 else 'circle'
            
            fig_simple.add_trace(go.Scatter(
                x=[iter_num],
                y=[dist],
                mode='markers',
                marker=dict(size=12, color=color, symbol=symbol),
                name=f"Iter {iter_num} - Cand {cand_id}",
                showlegend=False,
                hovertemplate=f"<b>Iteração {iter_num}</b><br>" +
                              f"Candidato: {cand_id}<br>" +
                              f"Distância: {dist:.4f}<br>" +
                              f"Status: {'Escolhido (A)' if chosen else 'Rejeitado (B)'}<extra></extra>"
            ))
    
    # Linha da melhor distância acumulada
    best_so_far = df.groupby('iter')['dmin_so_far'].first()
    fig_simple.add_trace(go.Scatter(
        x=best_so_far.index,
        y=best_so_far.values,
        mode='lines+markers',
        line=dict(color='blue', width=2, dash='dash'),
        marker=dict(size=8, color='blue'),
        name='Melhor acumulada',
        hovertemplate="<b>Melhor até iter %{x}</b><br>Distância: %{y:.4f}<extra></extra>"
    ))
    
    fig_simple.update_layout(
        title="Evolução das Distâncias por Iteração",
        xaxis_title="Iteração",
        yaxis_title="Distância Cosseno",
        hovermode='closest',
        height=500,
        showlegend=True,
        legend=dict(x=0.7, y=0.95)
    )
    
    st.plotly_chart(fig_simple, use_container_width=True)
    
    # Legenda
    st.markdown("""
    **Legenda:**
    -  **Estrela verde**: Ideia escolhida como A (melhor)
    -  **Círculo vermelho**: Ideia rejeitada como B (pior)
    -  **Linha azul tracejada**: Melhor distância acumulada
    """)

st.markdown("---")

# ============================================================================
# 1C. DIVERSIDADE INTRA-ITERAÇÃO
# ============================================================================
if show_diversity:
    st.subheader(" Diversidade Intra-Iteração")
    
    st.info("""
     **O que esta métrica mede:** Quão diferentes são as 2 ideias geradas em cada iteração.
    - **Alta diversidade** (≥0.5): LLM está explorando amplamente
    - **Baixa diversidade** (≤0.2): LLM está refinando localmente
    - **Decrescente**: Comportamento ideal de convergência
    """)
    
    # Carregar embeddings para calcular diversidade
    with st.spinner(" Calculando diversidade..."):
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Tentar usar GPU, fallback para CPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            
            diversity_data = []
            
            for iter_num in sorted(df['iter'].unique()):
                iter_data = df[df['iter'] == iter_num]
                
                if len(iter_data) == 2:  # Precisa de exatamente 2 candidatos
                    # Carregar textos dos 2 candidatos
                    texts = []
                    for _, row in iter_data.iterrows():
                        cand_id = int(row['cand_id'])
                        file_path = ref_path / f"iter_{iter_num:03d}" / f"{cand_id}.txt"
                        
                        if file_path.exists():
                            with open(file_path, 'r', encoding='utf-8') as f:
                                texts.append(f.read().strip())
                    
                    if len(texts) == 2:
                        # Gerar embeddings
                        embeddings = model.encode(texts, normalize_embeddings=True)
                        
                        # Calcular distância cosseno entre os 2
                        from scipy.spatial.distance import cosine
                        diversity = cosine(embeddings[0], embeddings[1])
                        
                        diversity_data.append({
                            'iter': iter_num,
                            'diversity': diversity
                        })
            
            if diversity_data:
                diversity_df = pd.DataFrame(diversity_data)
                
                # Criar gráfico
                fig_div = go.Figure()
                
                # Linha de diversidade
                fig_div.add_trace(go.Scatter(
                    x=diversity_df['iter'],
                    y=diversity_df['diversity'],
                    mode='lines+markers',
                    line=dict(color='purple', width=3),
                    marker=dict(size=10, color='purple', symbol='circle'),
                    name='Diversidade',
                    hovertemplate="<b>Iteração %{x}</b><br>Diversidade: %{y:.4f}<extra></extra>"
                ))
                
                # Linha de referência: diversidade alta (0.5)
                fig_div.add_hline(y=0.5, line_dash="dash", line_color="red", 
                                  annotation_text="Alta diversidade (exploração)")
                
                # Linha de referência: diversidade baixa (0.2)
                fig_div.add_hline(y=0.2, line_dash="dash", line_color="green",
                                  annotation_text="Baixa diversidade (refinamento)")
                
                fig_div.update_layout(
                    title="Diversidade Intra-Iteração ao Longo do Tempo",
                    xaxis_title="Iteração",
                    yaxis_title="Diversidade Cosseno (0 = idênticas, 1 = ortogonais)",
                    hovermode='closest',
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig_div, use_container_width=True)
                
                # Análise estatística
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_diversity = diversity_df['diversity'].mean()
                    st.metric("Diversidade Média", f"{avg_diversity:.3f}")
                
                with col2:
                    first_half = diversity_df[diversity_df['iter'] <= diversity_df['iter'].max() // 2]['diversity'].mean()
                    second_half = diversity_df[diversity_df['iter'] > diversity_df['iter'].max() // 2]['diversity'].mean()
                    change = ((second_half - first_half) / first_half) * 100
                    st.metric("Mudança (1ª→2ª metade)", f"{change:+.1f}%",
                             delta=f"{second_half - first_half:.3f}")
                
                with col3:
                    # Contar iterações em cada regime
                    high_div = sum(diversity_df['diversity'] >= 0.5)
                    low_div = sum(diversity_df['diversity'] <= 0.2)
                    st.metric("Exploração/Refinamento", f"{high_div}/{low_div}")
                
                # Interpretação
                st.markdown("""
                **Como interpretar:**
                
                - **Linha roxa**: Diversidade entre as 2 ideias geradas em cada iteração
                - **Linha vermelha tracejada**: Limiar de alta diversidade (≥0.5)
                - **Linha verde tracejada**: Limiar de baixa diversidade (≤0.2)
                
                **Padrões esperados:**
                1. **Decrescente**: LLM está convergindo (explorando → refinando) 
                2. **Constante alta**: LLM não está aprendendo com feedback ❌
                3. **Constante baixa**: Convergência prematura (preso em mínimo local) 
                4. **Oscilante**: Busca estocástica (normal para seu algoritmo)
                """)
                
                # Análise detalhada
                with st.expander(" Ver análise detalhada"):
                    st.dataframe(diversity_df, use_container_width=True, hide_index=True)
                    
                    # Diagnóstico
                    st.write("**Diagnóstico:**")
                    if avg_diversity >= 0.5:
                        st.warning(" Diversidade média ALTA - LLM pode não estar convergindo")
                    elif avg_diversity <= 0.2:
                        st.warning(" Diversidade média BAIXA - Possível convergência prematura")
                    else:
                        st.success(" Diversidade média MODERADA - Comportamento balanceado")
                    
                    if change < -20:
                        st.success(" Diversidade DECRESCENTE - Convergência saudável")
                    elif change > 20:
                        st.warning(" Diversidade CRESCENTE - Pode indicar problema no prompt")
                    else:
                        st.info(" Diversidade ESTÁVEL - Exploração constante")
            
            else:
                st.warning(" Não foi possível calcular diversidade (precisa de 2 candidatos por iteração)")
        
        except Exception as e:
            st.error(f"Erro ao calcular diversidade: {e}")
            import traceback
            st.code(traceback.format_exc())

st.markdown("---")

# ============================================================================
# 1D. NOVIDADE (DISTÂNCIA ÀS ANTERIORES)
# ============================================================================
if show_novelty:
    st.subheader(" Novidade (Distância às Anteriores)")
    
    st.info("""
     **O que esta métrica mede:** Quão diferente cada nova ideia é de TODAS as ideias anteriores.
    - **Alta novidade** (≥0.4): Explorando novas regiões do espaço semântico
    - **Baixa novidade** (≤0.1): Revisitando regiões já exploradas (refinamento)
    - **Decrescente**: Convergindo para uma região específica
    """)
    
    # Carregar embeddings para calcular novidade
    with st.spinner(" Calculando novidade..."):
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Tentar usar GPU, fallback para CPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            
            novelty_data = []
            all_embeddings = []  # Histórico de embeddings
            
            for iter_num in sorted(df['iter'].unique()):
                iter_data = df[df['iter'] == iter_num]
                
                for _, row in iter_data.iterrows():
                    cand_id = int(row['cand_id'])
                    file_path = ref_path / f"iter_{iter_num:03d}" / f"{cand_id}.txt"
                    
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        
                        # Gerar embedding (normalizado para consistência)
                        embedding = model.encode([text], normalize_embeddings=True)[0]
                        
                        # Calcular novidade (distância mínima às anteriores)
                        if all_embeddings:
                            from scipy.spatial.distance import cosine
                            distances_to_previous = [cosine(embedding, prev_emb) 
                                                    for prev_emb in all_embeddings]
                            novelty = min(distances_to_previous)
                        else:
                            novelty = None  # Primeira ideia não tem anteriores
                        
                        novelty_data.append({
                            'iter': iter_num,
                            'cand_id': cand_id,
                            'novelty': novelty,
                            'chosen': row['chosen_A']
                        })
                        
                        # Adicionar ao histórico
                        all_embeddings.append(embedding)
            
            if novelty_data:
                novelty_df = pd.DataFrame(novelty_data)
                # Remover primeira linha (novelty=None)
                novelty_df = novelty_df[novelty_df['novelty'].notna()]
                
                # Criar gráfico
                fig_nov = go.Figure()
                
                # Separar escolhidas e rejeitadas
                chosen = novelty_df[novelty_df['chosen'] == 1]
                rejected = novelty_df[novelty_df['chosen'] == 0]
                
                # Pontos escolhidos (verde)
                fig_nov.add_trace(go.Scatter(
                    x=chosen['iter'],
                    y=chosen['novelty'],
                    mode='markers',
                    marker=dict(size=12, color='green', symbol='star'),
                    name='Escolhida (A)',
                    hovertemplate="<b>Iter %{x} - Escolhida</b><br>Novidade: %{y:.4f}<extra></extra>"
                ))
                
                # Pontos rejeitados (vermelho)
                fig_nov.add_trace(go.Scatter(
                    x=rejected['iter'],
                    y=rejected['novelty'],
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='circle'),
                    name='Rejeitada (B)',
                    hovertemplate="<b>Iter %{x} - Rejeitada</b><br>Novidade: %{y:.4f}<extra></extra>"
                ))
                
                # Linha de tendência (apenas escolhidas)
                fig_nov.add_trace(go.Scatter(
                    x=chosen['iter'],
                    y=chosen['novelty'],
                    mode='lines',
                    line=dict(color='orange', width=2, dash='dot'),
                    name='Tendência (Escolhidas)',
                    hovertemplate="<b>Iter %{x}</b><br>Novidade: %{y:.4f}<extra></extra>"
                ))
                
                # Linha de referência: alta novidade (0.4)
                fig_nov.add_hline(y=0.4, line_dash="dash", line_color="red", 
                                  annotation_text="Alta novidade (exploração)")
                
                # Linha de referência: baixa novidade (0.1)
                fig_nov.add_hline(y=0.1, line_dash="dash", line_color="green",
                                  annotation_text="Baixa novidade (refinamento)")
                
                fig_nov.update_layout(
                    title="Novidade das Ideias ao Longo do Tempo",
                    xaxis_title="Iteração",
                    yaxis_title="Novidade (distância mínima às anteriores)",
                    hovermode='closest',
                    height=500,
                    showlegend=True,
                    legend=dict(x=0.7, y=0.95)
                )
                
                st.plotly_chart(fig_nov, use_container_width=True)
                
                # Análise estatística
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_novelty = chosen['novelty'].mean()
                    st.metric("Novidade Média (Escolhidas)", f"{avg_novelty:.3f}")
                
                with col2:
                    first_half = chosen[chosen['iter'] <= chosen['iter'].max() // 2]['novelty'].mean()
                    second_half = chosen[chosen['iter'] > chosen['iter'].max() // 2]['novelty'].mean()
                    change = ((second_half - first_half) / first_half) * 100
                    st.metric("Mudança (1ª→2ª metade)", f"{change:+.1f}%",
                             delta=f"{second_half - first_half:.3f}")
                
                with col3:
                    # Contar ideias em cada regime
                    high_nov = sum(chosen['novelty'] >= 0.4)
                    low_nov = sum(chosen['novelty'] <= 0.1)
                    st.metric("Exploração/Refinamento", f"{high_nov}/{low_nov}")
                
                # Interpretação
                st.markdown("""
                **Como interpretar:**
                
                - **Estrela verde**: Ideia escolhida (menor distância à referência)
                - **Círculo vermelho**: Ideia rejeitada
                -  **Linha laranja pontilhada**: Tendência das ideias escolhidas
                - **Linha vermelha tracejada**: Limiar de alta novidade (≥0.4)
                - **Linha verde tracejada**: Limiar de baixa novidade (≤0.1)
                
                **Padrões esperados:**
                1. **Decrescente**: Convergindo para uma região (refinamento local) 
                2. **Constante alta**: Explorando amplamente (não convergindo) 
                3. **Constante baixa**: Preso em região local (convergência prematura) 
                4. **Oscilante**: Alternando entre exploração e refinamento
                
                **Diferença vs. Diversidade:**
                - **Diversidade**: Compara as 2 ideias da MESMA iteração
                - **Novidade**: Compara cada ideia com TODAS as anteriores
                """)
                
                # Análise detalhada
                with st.expander(" Ver análise detalhada"):
                    st.dataframe(novelty_df[['iter', 'cand_id', 'novelty', 'chosen']], 
                                use_container_width=True, hide_index=True)
                    
                    # Diagnóstico
                    st.write("**Diagnóstico:**")
                    if avg_novelty >= 0.4:
                        st.warning(" Novidade média ALTA - Explorando amplamente (pode não convergir)")
                    elif avg_novelty <= 0.1:
                        st.warning(" Novidade média BAIXA - Possível convergência prematura")
                    else:
                        st.success(" Novidade média MODERADA - Balanceando exploração e refinamento")
                    
                    if change < -30:
                        st.success(" Novidade DECRESCENTE FORTE - Convergência clara")
                    elif change < -10:
                        st.success(" Novidade DECRESCENTE - Convergindo gradualmente")
                    elif change > 20:
                        st.warning(" Novidade CRESCENTE - Divergindo (problema no prompt?)")
                    else:
                        st.info(" Novidade ESTÁVEL - Exploração constante")
                    
                    # Análise de repetição
                    very_low_novelty = sum(novelty_df['novelty'] < 0.05)
                    if very_low_novelty > 0:
                        st.warning(f" {very_low_novelty} ideias com novidade < 0.05 (possível repetição)")
            
            else:
                st.warning(" Não foi possível calcular novidade")
        
        except Exception as e:
            st.error(f"Erro ao calcular novidade: {e}")
            import traceback
            st.code(traceback.format_exc())

st.markdown("---")

# ============================================================================
# 1E. TAXA DE EXPLORAÇÃO VS. REFINAMENTO
# ============================================================================
if show_exploration_rate:
    st.subheader(" Taxa de Exploração vs. Refinamento")
    
    st.info("""
     **O que esta métrica mede:** Percentual de iterações onde a ideia escolhida PIOROU em relação ao melhor histórico.
    - **Exploração**: Escolher uma ideia pior que o melhor histórico (buscar novas regiões)
    - **Refinamento**: Escolher uma ideia melhor que o melhor histórico (melhorar localmente)
    - **Balanceado**: ~30-40% de exploração é saudável para busca estocástica
    """)
    
    # Calcular taxa de exploração (não precisa de embeddings, usa log.csv)
    try:
        chosen_data = df[df['chosen_A'] == 1].sort_values('iter')
        
        if len(chosen_data) > 0:
            # Classificar cada iteração
            exploration_data = []
            
            for _, row in chosen_data.iterrows():
                iter_num = int(row['iter'])
                chosen_dist = float(row['dist'])
                best_so_far = float(row['dmin_so_far'])
                
                # Se escolhido > melhor histórico = exploração
                # Se escolhido <= melhor histórico = refinamento
                is_exploration = chosen_dist > best_so_far
                
                exploration_data.append({
                    'iter': iter_num,
                    'chosen_dist': chosen_dist,
                    'best_so_far': best_so_far,
                    'type': 'Exploração' if is_exploration else 'Refinamento',
                    'is_exploration': 1 if is_exploration else 0
                })
            
            exp_df = pd.DataFrame(exploration_data)
            
            # Calcular taxa acumulada
            exp_df['exploration_rate_cumulative'] = exp_df['is_exploration'].expanding().mean() * 100
            
            # Criar figura com 2 subplots
            from plotly.subplots import make_subplots
            
            fig_exp = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Classificação por Iteração', 'Taxa de Exploração Acumulada'),
                vertical_spacing=0.15,
                row_heights=[0.5, 0.5]
            )
            
            # Subplot 1: Barras de exploração/refinamento
            exploration = exp_df[exp_df['type'] == 'Exploração']
            refinement = exp_df[exp_df['type'] == 'Refinamento']
            
            fig_exp.add_trace(
                go.Bar(
                    x=exploration['iter'],
                    y=[1] * len(exploration),
                    name='Exploração',
                    marker_color='orange',
                    hovertemplate="<b>Iter %{x}: Exploração</b><br>Escolhida piorou o histórico<extra></extra>"
                ),
                row=1, col=1
            )
            
            fig_exp.add_trace(
                go.Bar(
                    x=refinement['iter'],
                    y=[1] * len(refinement),
                    name='Refinamento',
                    marker_color='blue',
                    hovertemplate="<b>Iter %{x}: Refinamento</b><br>Escolhida melhorou o histórico<extra></extra>"
                ),
                row=1, col=1
            )
            
            # Subplot 2: Taxa acumulada
            fig_exp.add_trace(
                go.Scatter(
                    x=exp_df['iter'],
                    y=exp_df['exploration_rate_cumulative'],
                    mode='lines+markers',
                    line=dict(color='purple', width=3),
                    marker=dict(size=8),
                    name='Taxa Acumulada',
                    hovertemplate="<b>Iter %{x}</b><br>Taxa de exploração: %{y:.1f}%<extra></extra>"
                ),
                row=2, col=1
            )
            
            # Linha de referência: 30-40% (ideal)
            fig_exp.add_hline(y=30, line_dash="dash", line_color="green", 
                             annotation_text="Mínimo ideal (30%)", row=2, col=1)
            fig_exp.add_hline(y=40, line_dash="dash", line_color="green",
                             annotation_text="Máximo ideal (40%)", row=2, col=1)
            
            fig_exp.update_xaxes(title_text="Iteração", row=1, col=1)
            fig_exp.update_xaxes(title_text="Iteração", row=2, col=1)
            fig_exp.update_yaxes(title_text="Tipo", row=1, col=1, showticklabels=False)
            fig_exp.update_yaxes(title_text="Taxa de Exploração (%)", row=2, col=1)
            
            fig_exp.update_layout(
                height=700,
                showlegend=True,
                legend=dict(x=0.7, y=1.0),
                barmode='stack'
            )
            
            st.plotly_chart(fig_exp, use_container_width=True)
            
            # Análise estatística
            col1, col2, col3, col4 = st.columns(4)
            
            total_iters = len(exp_df)
            num_exploration = sum(exp_df['is_exploration'])
            num_refinement = total_iters - num_exploration
            exploration_rate = (num_exploration / total_iters) * 100
            
            with col1:
                st.metric("Taxa de Exploração", f"{exploration_rate:.1f}%")
            
            with col2:
                st.metric("Iterações de Exploração", f"{num_exploration}/{total_iters}")
            
            with col3:
                st.metric("Iterações de Refinamento", f"{num_refinement}/{total_iters}")
            
            with col4:
                # Classificar comportamento
                if 30 <= exploration_rate <= 40:
                    behavior = "Balanceado "
                    color = "normal"
                elif exploration_rate > 50:
                    behavior = "Alta Exploração "
                    color = "inverse"
                else:
                    behavior = "Alto Refinamento "
                    color = "inverse"
                st.metric("Comportamento", behavior)
            
            # Interpretação
            st.markdown("""
            **Como interpretar:**
            
            **Gráfico Superior (Barras):**
            -  **Laranja**: Iteração de exploração (escolhida piorou o histórico)
            -  **Azul**: Iteração de refinamento (escolhida melhorou o histórico)
            
            **Gráfico Inferior (Linha):**
            -  **Linha roxa**: Taxa de exploração acumulada ao longo do tempo
            -  **Linhas verdes**: Faixa ideal (30-40%)
            
            **Comportamentos:**
            1. **30-40% exploração**: Balanceado (busca estocástica saudável) 
            2. **>50% exploração**: Explorando demais (pode não convergir) 
            3. **<20% exploração**: Refinando demais (pode estar preso) 
            
            **Por que exploração é importante:**
            - Evita mínimos locais
            - Descobre regiões melhores do espaço
            - Permite "escape" de regiões ruins
            
            **Relação com outras métricas:**
            - Alta exploração + Alta diversidade = Buscando amplamente
            - Baixa exploração + Baixa diversidade = Convergiu (ou preso)
            """)
            
            # Análise detalhada
            with st.expander(" Ver análise detalhada"):
                st.dataframe(exp_df[['iter', 'chosen_dist', 'best_so_far', 'type']], 
                            use_container_width=True, hide_index=True)
                
                # Diagnóstico
                st.write("**Diagnóstico:**")
                if 30 <= exploration_rate <= 40:
                    st.success(f" Taxa de exploração IDEAL ({exploration_rate:.1f}%) - Busca estocástica balanceada")
                elif exploration_rate > 50:
                    st.warning(f" Taxa de exploração ALTA ({exploration_rate:.1f}%) - Pode indicar:")
                    st.write("  • Prompt incentivando muita diversidade")
                    st.write("  • LLM não está aprendendo com feedback")
                    st.write("  • Espaço semântico muito complexo")
                elif exploration_rate < 20:
                    st.warning(f" Taxa de exploração BAIXA ({exploration_rate:.1f}%) - Pode indicar:")
                    st.write("  • Convergência prematura (preso em mínimo local)")
                    st.write("  • Prompt incentivando pouca diversidade")
                    st.write("  • Espaço semântico muito simples")
                else:
                    st.info(f" Taxa de exploração MODERADA ({exploration_rate:.1f}%)")
                
                # Análise temporal
                first_half_rate = exp_df[exp_df['iter'] <= exp_df['iter'].max() // 2]['is_exploration'].mean() * 100
                second_half_rate = exp_df[exp_df['iter'] > exp_df['iter'].max() // 2]['is_exploration'].mean() * 100
                
                st.write(f"\n**Análise temporal:**")
                st.write(f"• 1ª metade: {first_half_rate:.1f}% exploração")
                st.write(f"• 2ª metade: {second_half_rate:.1f}% exploração")
                
                if first_half_rate > second_half_rate + 10:
                    st.success(" Exploração DECRESCENTE - Começou explorando, terminou refinando (ideal)")
                elif second_half_rate > first_half_rate + 10:
                    st.warning(" Exploração CRESCENTE - Pode indicar divergência")
                else:
                    st.info(" Exploração CONSTANTE - Taxa estável ao longo do tempo")
        
        else:
            st.warning(" Não há dados suficientes para calcular taxa de exploração")
    
    except Exception as e:
        st.error(f"Erro ao calcular taxa de exploração: {e}")
        import traceback
        st.code(traceback.format_exc())

st.markdown("---")

# ============================================================================
# 2. EMBEDDINGS 3D (PCA INTERATIVO)
# ============================================================================
if show_embeddings_3d:
    st.subheader("Visualização de Embeddings (PCA 3D Interativo)")
    
    with st.spinner("Carregando embeddings..."):
        # Carregar todos os textos e gerar embeddings
        try:
            from sentence_transformers import SentenceTransformer
            
            @st.cache_resource
            def load_embedder():
                return SentenceTransformer('all-MiniLM-L6-v2')
            
            embedder = load_embedder()
            
            texts = []
            labels = []
            colors_list = []
            types = []  # 'referencia' ou 'gerada'
            
            # Tentar carregar a referencia original
            ref_text = None
            ref_file_candidates = [
                ref_path.parent.parent / "refs_combined.txt",  # arquivo principal
                ref_path / "referencia.txt",  # se houver arquivo específico
            ]
            
            # Extrair número da referencia (ex: ref_001 -> 1)
            ref_num = int(selected_ref.split('_')[1])
            
            # Ler arquivo de referencias e pegar a correta
            for ref_file in ref_file_candidates:
                if ref_file.exists():
                    with open(ref_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Separar por --- e pegar a referencia correta
                        refs = [r.strip() for r in content.split('---') if r.strip()]
                        if ref_num <= len(refs):
                            ref_text = refs[ref_num - 1]
                            break
            
            # Adicionar referencia se encontrada
            if ref_text:
                texts.append(ref_text)
                labels.append("REFERENCIA ORIGINAL")
                colors_list.append(-1)  # valor especial para referencia
                types.append('referencia')
            
            # Carregar ideias geradas
            for _, row in df.iterrows():
                iter_num = row['iter']
                cand_id = row['cand_id']
                file_path = Path(row['files'])
                
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        texts.append(text)
                        labels.append(f"Iter{iter_num}-Cand{cand_id}")  # SEM ESPAÇOS para match com código de conexões
                        colors_list.append(iter_num)
                        types.append('gerada')
            
            if len(texts) > 0:
                # Gerar embeddings
                embeddings = embedder.encode(texts, show_progress_bar=False)
                
                # NORMALIZAR embeddings para que distância euclidiana seja proporcional ao cosseno
                # Apos normalização: dist_euclidiana^2 = 2 * (1 - cos_similarity)
                embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                # Calcular distâncias da referencia (se existir)
                if ref_text:
                    ref_embedding = embeddings_norm[0]  # primeiro embedding é a referência
                    gen_embeddings = embeddings_norm[1:]  # resto são ideias geradas
                    
                    # Distância cosseno (como no experimento)
                    cos_distances = 1 - np.dot(gen_embeddings, ref_embedding)
                    
                    # Distância euclidiana (após normalização)
                    eucl_distances = np.linalg.norm(gen_embeddings - ref_embedding, axis=1)
                    
                    # Adicionar ao DataFrame para exibir
                    st.info(f" Distâncias calculadas: {len(cos_distances)} ideias vs referencia")
                    st.write(f"**Distância cosseno:** min={cos_distances.min():.4f}, max={cos_distances.max():.4f}, média={cos_distances.mean():.4f}")
                    st.write(f"**Distância euclidiana:** min={eucl_distances.min():.4f}, max={eucl_distances.max():.4f}, média={eucl_distances.mean():.4f}")
                    st.write(f"**Relação teórica:** dist_eucl = √(2 × dist_cos) → verificação: {np.allclose(eucl_distances, np.sqrt(2 * cos_distances), atol=1e-5)}")
                
                # Reduzir para 3D com PCA (usando embeddings normalizados)
                pca = PCA(n_components=3)
                embeddings_3d = pca.fit_transform(embeddings_norm)
                
                # Criar DataFrame
                df_viz = pd.DataFrame({
                    'x': embeddings_3d[:, 0],
                    'y': embeddings_3d[:, 1],
                    'z': embeddings_3d[:, 2],
                    'label': labels,
                    'iteração': colors_list,
                    'tipo': types,
                    'dist_cos': [0.0] + list(cos_distances) if ref_text else [0.0] * len(labels),
                    'dist_eucl': [0.0] + list(eucl_distances) if ref_text else [0.0] * len(labels)
                })
                
                # Criar figura 3D com plotly graph_objects
                fig = go.Figure()
                
                # Adicionar referencia (se existir)
                df_ref = df_viz[df_viz['tipo'] == 'referencia']
                if len(df_ref) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=df_ref['x'],
                        y=df_ref['y'],
                        z=df_ref['z'],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color='red',
                            symbol='diamond',
                            line=dict(color='darkred', width=2)
                        ),
                        name='Referencia Original',
                        text=df_ref['label'],
                        hovertemplate="<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>"
                    ))
                
                # Adicionar ideias geradas (com gradiente de cor)
                df_gen = df_viz[df_viz['tipo'] == 'gerada']
                if len(df_gen) > 0:
                    # Encontrar a melhor ideia (menor distância cosseno)
                    best_idx = df_gen['dist_cos'].idxmin()
                    best_row = df_gen.loc[best_idx]
                    
                    # Plotar ideias normais
                    df_gen_normal = df_gen.drop(best_idx)
                    if len(df_gen_normal) > 0:
                        fig.add_trace(go.Scatter3d(
                            x=df_gen_normal['x'],
                            y=df_gen_normal['y'],
                            z=df_gen_normal['z'],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color=df_gen_normal['iteração'],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(
                                    title="Iteração",
                                    x=1.15,
                                    xanchor='left',
                                    thickness=15,
                                    len=0.7
                                ),
                                symbol='circle'
                            ),
                            name='Ideias Geradas',
                            customdata=np.column_stack((df_gen_normal['dist_cos'], df_gen_normal['dist_eucl'])),
                            hovertemplate="<b>%{text}</b><br>" +
                                        "PC1: %{x:.3f}<br>" +
                                        "PC2: %{y:.3f}<br>" +
                                        "PC3: %{z:.3f}<br>" +
                                        "Dist Cosseno: %{customdata[0]:.4f}<br>" +
                                        "Dist Euclidiana: %{customdata[1]:.4f}<extra></extra>",
                            text=df_gen_normal['label']
                        ))
                    
                    # Destacar a MELHOR ideia
                    fig.add_trace(go.Scatter3d(
                        x=[best_row['x']],
                        y=[best_row['y']],
                        z=[best_row['z']],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color='gold',
                            symbol='diamond',
                            line=dict(color='orange', width=3)
                        ),
                        name=f'MELHOR IDEIA (dist={best_row["dist_cos"]:.4f})',
                        text=[best_row['label']],
                        hovertemplate="<b> MELHOR IDEIA</b><br>" +
                                    "<b>%{text}</b><br>" +
                                    "PC1: %{x:.3f}<br>" +
                                    "PC2: %{y:.3f}<br>" +
                                    "PC3: %{z:.3f}<br>" +
                                    f"Dist Cosseno: {best_row['dist_cos']:.4f}<br>" +
                                    f"Dist Euclidiana: {best_row['dist_eucl']:.4f}<extra></extra>"
                    ))
                
                fig.update_layout(
                    title='Embeddings das Ideias Geradas (PCA 3D)',
                    scene=dict(
                        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)",
                        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)",
                        zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% var)",
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.5)
                        )
                    ),
                    height=700,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f" Variancia explicada: {sum(pca.explained_variance_ratio_)*100:.1f}%")
                
                # Legenda
                st.markdown(f"""
                **Legenda:**
                -  **Diamante vermelho grande**: Referencia original (alvo)
                -  **Diamante dourado**: MELHOR ideia gerada (menor distância)
                -  **Esferas coloridas**: Outras ideias geradas (cor = iteração)
                - **Objetivo**: Ideias devem se aproximar do diamante vermelho!
                
                **Controles Interativos:**
                - **Rotacionar**: Clique e arraste
                - **Zoom**: Scroll do mouse ou pinch
                - **Pan**: Shift + arraste
                - **Reset**: Duplo clique
                
                **Vantagens do 3D:**
                - Captura **{sum(pca.explained_variance_ratio_)*100:.1f}%** da variância (vs ~34% em 2D)
                - Melhor percepção de distâncias e agrupamentos
                - Explore diferentes ângulos para entender a convergência
                
                **Nota técnica:** Embeddings foram **normalizados** antes do PCA, garantindo que a 
                distância euclidiana no gráfico seja **proporcional** à distância cosseno usada no experimento.
                Use o hover para ver as distâncias reais de cada ideia.
                """)
            else:
                st.warning(" Nenhum texto encontrado para visualizar")
                
        except Exception as e:
            st.error(f"Erro ao gerar embeddings: {e}")

st.markdown("---")

# ============================================================================
# 3. t-SNE 3D INTERATIVO
# ============================================================================
if show_embeddings_tsne:
    st.subheader("Visualização de Embeddings (t-SNE 3D Interativo)")
    
    with st.spinner("Calculando t-SNE 3D (pode demorar)..."):
        try:
            if len(texts) > 0 and 'embeddings_norm' in locals():
                # t-SNE 3D (usando embeddings normalizados)
                tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(texts)-1))
                embeddings_tsne = tsne.fit_transform(embeddings_norm)
                
                df_tsne = pd.DataFrame({
                    'x': embeddings_tsne[:, 0],
                    'y': embeddings_tsne[:, 1],
                    'z': embeddings_tsne[:, 2],
                    'label': labels,
                    'iteração': colors_list,
                    'tipo': types,
                    'dist_cos': [0.0] + list(cos_distances) if ref_text else [0.0] * len(labels),
                    'dist_eucl': [0.0] + list(eucl_distances) if ref_text else [0.0] * len(labels)
                })
                
                # Criar figura 3D com plotly graph_objects
                fig = go.Figure()
                
                # Adicionar referencia (se existir)
                df_ref_tsne = df_tsne[df_tsne['tipo'] == 'referencia']
                if len(df_ref_tsne) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=df_ref_tsne['x'],
                        y=df_ref_tsne['y'],
                        z=df_ref_tsne['z'],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color='red',
                            symbol='diamond',
                            line=dict(color='darkred', width=2)
                        ),
                        name='Referencia Original',
                        text=df_ref_tsne['label'],
                        hovertemplate="<b>%{text}</b><br>t-SNE 1: %{x:.3f}<br>t-SNE 2: %{y:.3f}<br>t-SNE 3: %{z:.3f}<extra></extra>"
                    ))
                
                # Adicionar ideias geradas
                df_gen_tsne = df_tsne[df_tsne['tipo'] == 'gerada']
                if len(df_gen_tsne) > 0:
                    # Encontrar a melhor ideia (menor distância cosseno)
                    best_idx = df_gen_tsne['dist_cos'].idxmin()
                    best_row = df_gen_tsne.loc[best_idx]
                    
                    # Plotar ideias normais
                    df_gen_normal = df_gen_tsne.drop(best_idx)
                    if len(df_gen_normal) > 0:
                        fig.add_trace(go.Scatter3d(
                            x=df_gen_normal['x'],
                            y=df_gen_normal['y'],
                            z=df_gen_normal['z'],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color=df_gen_normal['iteração'],
                                colorscale='Plasma',
                                showscale=True,
                                colorbar=dict(
                                    title="Iteração",
                                    x=1.15,
                                    xanchor='left',
                                    thickness=15,
                                    len=0.7
                                ),
                                symbol='circle'
                            ),
                            name='Ideias Geradas',
                            customdata=np.column_stack((df_gen_normal['dist_cos'], df_gen_normal['dist_eucl'])),
                            hovertemplate="<b>%{text}</b><br>" +
                                        "t-SNE 1: %{x:.3f}<br>" +
                                        "t-SNE 2: %{y:.3f}<br>" +
                                        "t-SNE 3: %{z:.3f}<br>" +
                                        "Dist Cosseno: %{customdata[0]:.4f}<br>" +
                                        "Dist Euclidiana: %{customdata[1]:.4f}<extra></extra>",
                            text=df_gen_normal['label']
                        ))
                    
                    # Destacar a MELHOR ideia
                    fig.add_trace(go.Scatter3d(
                        x=[best_row['x']],
                        y=[best_row['y']],
                        z=[best_row['z']],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color='gold',
                            symbol='diamond',
                            line=dict(color='orange', width=3)
                        ),
                        name=f'MELHOR IDEIA (dist={best_row["dist_cos"]:.4f})',
                        text=[best_row['label']],
                        hovertemplate="<b> MELHOR IDEIA</b><br>" +
                                    "<b>%{text}</b><br>" +
                                    "t-SNE 1: %{x:.3f}<br>" +
                                    "t-SNE 2: %{y:.3f}<br>" +
                                    "t-SNE 3: %{z:.3f}<br>" +
                                    f"Dist Cosseno: {best_row['dist_cos']:.4f}<br>" +
                                    f"Dist Euclidiana: {best_row['dist_eucl']:.4f}<extra></extra>"
                    ))
                
                fig.update_layout(
                    title='Embeddings das Ideias Geradas (t-SNE 3D)',
                    scene=dict(
                        xaxis_title="t-SNE 1",
                        yaxis_title="t-SNE 2",
                        zaxis_title="t-SNE 3",
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.5)
                        )
                    ),
                    height=700,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Legenda
                st.markdown("""
                **Legenda:**
                -  **Diamante vermelho grande**: Referencia original (alvo)
                -  **Diamante dourado**: MELHOR ideia gerada (menor distância)
                -  **Esferas coloridas**: Outras ideias geradas (cor = iteração)
                - **Objetivo**: Ideias devem se aproximar do diamante vermelho!
                
                **Controles Interativos:**
                - **Rotacionar**: Clique e arraste
                - **Zoom**: Scroll do mouse ou pinch
                - **Pan**: Shift + arraste
                - **Reset**: Duplo clique
                
                **Vantagens do t-SNE 3D:**
                - Melhor visualização de **clusters** e agrupamentos
                - Estrutura local preservada em 3 dimensões
                - Identifica "famílias" de ideias similares
                - Explore diferentes ângulos para ver padrões
                
                **Nota técnica:** Embeddings foram **normalizados** antes do t-SNE, garantindo que a 
                distância euclidiana no gráfico seja **proporcional** à distância cosseno usada no experimento.
                
                **Importante:** t-SNE preserva estrutura **local** (clusters), não distâncias globais. 
                Ideias próximas no gráfico são semanticamente similares, mas distâncias absolutas podem 
                não refletir as distâncias reais. Use o hover para ver as distâncias verdadeiras!
                """)
            else:
                st.warning(" Carregue os embeddings primeiro (ative PCA)")
        except Exception as e:
            st.error(f"Erro ao calcular t-SNE: {e}")

st.markdown("---")

# ============================================================================
# 3.5. VISUALIZACAO UMAP 3D INTERATIVO
# ============================================================================
if show_embeddings_umap and UMAP_AVAILABLE:
    st.subheader(" Visualização de Embeddings (UMAP 3D Interativo)")
    
    with st.spinner("Calculando UMAP 3D..."):
        try:
            if len(texts) > 0 and 'embeddings_norm' in locals():
                # UMAP 3D (usando embeddings normalizados)
                n_neighbors = min(15, len(texts) - 1)
                umap_reducer = UMAP(
                    n_components=3,
                    random_state=42,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric='cosine'
                )
                embeddings_umap = umap_reducer.fit_transform(embeddings_norm)
                
                df_umap = pd.DataFrame({
                    'x': embeddings_umap[:, 0],
                    'y': embeddings_umap[:, 1],
                    'z': embeddings_umap[:, 2],
                    'label': labels,
                    'iteração': colors_list,
                    'tipo': types,
                    'dist_cos': [0.0] + list(cos_distances) if ref_text else [0.0] * len(labels),
                    'dist_eucl': [0.0] + list(eucl_distances) if ref_text else [0.0] * len(labels)
                })
                
                # Criar figura 3D com plotly graph_objects
                fig = go.Figure()
                
                # Adicionar referencia (se existir)
                df_ref_umap = df_umap[df_umap['tipo'] == 'referencia']
                if len(df_ref_umap) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=df_ref_umap['x'],
                        y=df_ref_umap['y'],
                        z=df_ref_umap['z'],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color='red',
                            symbol='diamond',
                            line=dict(color='darkred', width=2)
                        ),
                        name='Referencia Original',
                        text=df_ref_umap['label'],
                        hovertemplate="<b>%{text}</b><br>UMAP 1: %{x:.3f}<br>UMAP 2: %{y:.3f}<br>UMAP 3: %{z:.3f}<extra></extra>"
                    ))
                
                # Adicionar ideias geradas
                df_gen_umap = df_umap[df_umap['tipo'] == 'gerada']
                if len(df_gen_umap) > 0:
                    # Encontrar a melhor ideia (menor distância cosseno)
                    best_idx = df_gen_umap['dist_cos'].idxmin()
                    best_row = df_gen_umap.loc[best_idx]
                    
                    # Plotar ideias normais
                    df_gen_normal = df_gen_umap.drop(best_idx)
                    if len(df_gen_normal) > 0:
                        fig.add_trace(go.Scatter3d(
                            x=df_gen_normal['x'],
                            y=df_gen_normal['y'],
                            z=df_gen_normal['z'],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color=df_gen_normal['iteração'],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(
                                    title="Iteração",
                                    x=1.15,
                                    xanchor='left',
                                    thickness=15,
                                    len=0.7
                                ),
                                symbol='circle'
                            ),
                            name='Ideias Geradas',
                            customdata=np.column_stack((df_gen_normal['dist_cos'], df_gen_normal['dist_eucl'])),
                            hovertemplate="<b>%{text}</b><br>" +
                                        "UMAP 1: %{x:.3f}<br>" +
                                        "UMAP 2: %{y:.3f}<br>" +
                                        "UMAP 3: %{z:.3f}<br>" +
                                        "Dist Cosseno: %{customdata[0]:.4f}<br>" +
                                        "Dist Euclidiana: %{customdata[1]:.4f}<extra></extra>",
                            text=df_gen_normal['label']
                        ))
                    
                    # Destacar a MELHOR ideia
                    fig.add_trace(go.Scatter3d(
                        x=[best_row['x']],
                        y=[best_row['y']],
                        z=[best_row['z']],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color='gold',
                            symbol='diamond',
                            line=dict(color='orange', width=3)
                        ),
                        name=f'MELHOR IDEIA (dist={best_row["dist_cos"]:.4f})',
                        text=[best_row['label']],
                        hovertemplate="<b> MELHOR IDEIA</b><br>" +
                                    "<b>%{text}</b><br>" +
                                    "UMAP 1: %{x:.3f}<br>" +
                                    "UMAP 2: %{y:.3f}<br>" +
                                    "UMAP 3: %{z:.3f}<br>" +
                                    f"Dist Cosseno: {best_row['dist_cos']:.4f}<br>" +
                                    f"Dist Euclidiana: {best_row['dist_eucl']:.4f}<extra></extra>"
                    ))
                
                fig.update_layout(
                    title='Embeddings das Ideias Geradas (UMAP 3D)',
                    scene=dict(
                        xaxis_title="UMAP 1",
                        yaxis_title="UMAP 2",
                        zaxis_title="UMAP 3",
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.5)
                        )
                    ),
                    height=700,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Legenda
                st.markdown("""
                **Legenda:**
                -  **Diamante vermelho grande**: Referencia original (alvo)
                -  **Diamante dourado**: MELHOR ideia gerada (menor distância)
                -  **Esferas coloridas**: Outras ideias geradas (cor = iteração)
                - **Objetivo**: Ideias devem se aproximar do diamante vermelho!
                
                **Controles Interativos:**
                - **Rotacionar**: Clique e arraste
                - **Zoom**: Scroll do mouse ou pinch
                - **Pan**: Shift + arraste
                - **Reset**: Duplo clique
                
                **Vantagens do UMAP 3D:**
                -  **MELHOR DOS DOIS MUNDOS**: Preserva estrutura **local E global**
                -  **Mais rápido** que t-SNE 3D
                - **Mais consistente**: Mesmo resultado sempre (determinístico)
                -  **Distâncias confiáveis**: Usa métrica cosseno diretamente
                -  **Convergência clara**: Veja o "caminho" completo das ideias
                -  **Mais informação**: 3D preserva muito mais estrutura que 2D
                
                **Interpretação:** 
                UMAP 3D é a **melhor visualização** para entender convergência! Ele mostra:
                - **Estrutura local**: Clusters de ideias similares (como t-SNE)
                - **Estrutura global**: Distâncias reais entre grupos (como PCA)
                - **Caminho de convergência**: Ideias iniciais (azul) → finais (amarelo) → referência (vermelho)
                
                Explore diferentes ângulos para ver como as ideias se aproximam da referência!
                
                **Nota técnica:** Embeddings foram **normalizados** antes do UMAP, garantindo que a 
                distância euclidiana no gráfico seja **proporcional** à distância cosseno usada no experimento.
                Use o hover para ver as distâncias reais de cada ideia.
                """)
            else:
                st.warning(" Carregue os embeddings primeiro (ative PCA)")
        except Exception as e:
            st.error(f"Erro ao calcular UMAP: {e}")

st.markdown("---")

# ============================================================================
# 4. HEATMAP DE DISTANCIAS
# ============================================================================
if show_heatmap:
    st.subheader(" Heatmap de Distâncias")
    
    try:
        # Criar matriz de distâncias por iteração e candidato
        # Usar pivot_table com agregacao (mean) para lidar com duplicatas
        pivot = df.pivot_table(
            index='iter', 
            columns='cand_id', 
            values='dist',
            aggfunc='mean'  # Se houver duplicatas, usar a média
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn_r',
            cbar_kws={'label': 'Distância Cosseno'},
            ax=ax
        )
        ax.set_title('Distâncias por Iteração e Candidato')
        ax.set_xlabel('Candidato')
        ax.set_ylabel('Iteração')
        
        st.pyplot(fig, clear_figure=True)
        
        st.markdown("""
        **Legenda:**
        -  **Verde**: Distâncias baixas (mais similar à referência)
        -  **Amarelo**: Distâncias médias
        -  **Vermelho**: Distâncias altas (menos similar à referência)
        - **Objetivo**: Ver a evolução das distâncias ao longo das iterações
        """)
    except Exception as e:
        st.error(f"Erro ao criar heatmap: {e}")
        st.info(" Isso pode acontecer se houver poucas iterações ou dados inconsistentes.")

st.markdown("---")

# ============================================================================
# 5. COMPARACAO MELHOR VS PIOR
# ============================================================================
if show_best_worst:
    st.subheader(" Comparação: Melhor vs Pior Ideia")
    
    # Encontrar melhor e pior
    best_row = df.loc[df['dist'].idxmin()]
    worst_row = df.loc[df['dist'].idxmax()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("###  Melhor Ideia")
        st.metric("Distância", f"{best_row['dist']:.4f}")
        st.metric("Iteração", int(best_row['iter']))
        st.metric("Candidato", int(best_row['cand_id']))
        
        best_file = Path(best_row['files'])
        if best_file.exists():
            with open(best_file, 'r', encoding='utf-8') as f:
                st.text_area("Texto:", f.read(), height=300, key="best")
    
    with col2:
        st.markdown("###  Pior Ideia")
        st.metric("Distância", f"{worst_row['dist']:.4f}")
        st.metric("Iteração", int(worst_row['iter']))
        st.metric("Candidato", int(worst_row['cand_id']))
        
        worst_file = Path(worst_row['files'])
        if worst_file.exists():
            with open(worst_file, 'r', encoding='utf-8') as f:
                st.text_area("Texto:", f.read(), height=300, key="worst")

st.markdown("---")

# ============================================================================
# 6. DENDROGRAMA HIERARQUICO
# ============================================================================
if show_dendrogram:
    st.subheader(" Dendrograma Hierárquico de Similaridade")
    
    with st.spinner("Calculando dendrograma..."):
        try:
            if len(texts) > 0 and 'embeddings_norm' in locals():
                # Usar embeddings normalizados
                # Calcular matriz de distâncias (cosseno)
                dist_matrix = pdist(embeddings_norm, metric='cosine')
                
                # Criar linkage (metodo: average)
                linkage_matrix = hierarchy.linkage(dist_matrix, method='average')
                
                # Criar figura
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Criar dendrograma
                dendrogram = hierarchy.dendrogram(
                    linkage_matrix,
                    labels=labels,
                    ax=ax,
                    orientation='right',
                    color_threshold=0.7 * max(linkage_matrix[:, 2]),
                    above_threshold_color='gray'
                )
                
                ax.set_xlabel('Distância (Cosseno)', fontsize=12)
                ax.set_ylabel('Ideias', fontsize=12)
                ax.set_title('Agrupamento Hierárquico de Ideias por Similaridade', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                
                # Colorir labels por iteração
                labels_order = dendrogram['ivl']
                colors_map = {}
                for i, label in enumerate(labels):
                    if label in labels_order:
                        # Extrair iteração do label
                        if 'Iter' in label:
                            iter_num = int(label.split('Iter')[1].split('-')[0])
                            colors_map[label] = plt.cm.viridis(iter_num / max(colors_list))
                        else:
                            colors_map[label] = 'red'  # Referencia
                
                # Aplicar cores aos labels
                ylbls = ax.get_ymajorticklabels()
                for lbl in ylbls:
                    label_text = lbl.get_text()
                    if label_text in colors_map:
                        lbl.set_color(colors_map[label_text])
                        if 'Ref' in label_text:
                            lbl.set_fontweight('bold')
                            lbl.set_fontsize(11)
                
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
                
                # Legenda
                st.markdown("""
                **Legenda:**
                -  **Árvore hierárquica**: Mostra como ideias se agrupam por similaridade
                -  **Eixo X (Distância)**: Quanto mais à esquerda o merge, mais similares as ideias
                -  **Cores**: Gradiente indica a iteração (azul escuro = inicial, amarelo = final)
                -  **Vermelho (negrito)**: Referência original
                
                **Interpretação:**
                - **Clusters baixos** (merge próximo a 0): Ideias muito similares
                - **Clusters altos** (merge longe de 0): Ideias diferentes
                - **Referência isolada**: Indicá que as ideias ainda não convergiram completamente
                - **Referência em cluster**: Indicá convergência bem-sucedida
                
                **O que procurar:**
                - Ideias de iterações finais (amarelo) devem estar próximas da referência (vermelho)
                - Múltiplos clusters podem indicar diferentes "caminhos" de convergência
                - Um único cluster grande indica convergência uniforme
                """)
            else:
                st.warning(" Carregue os embeddings primeiro (ative PCA ou outro embedding)")
        except Exception as e:
            st.error(f"Erro ao calcular dendrograma: {e}")

st.markdown("---")

# ============================================================================
# 7. GRAFO DE EVOLUCAO
# ============================================================================
if show_evolution_graph:
    st.subheader(" Grafo de Evolução Temporal")
    
    # Opções de filtro
    col1, col2, col3 = st.columns(3)
    with col1:
        max_iter_show = st.slider(
            "Mostrar até iteração:",
            min_value=1,
            max_value=max(colors_list) if colors_list else 10,
            value=min(10, max(colors_list) if colors_list else 10),
            help="Limite de iterações para visualizar (evita poluição visual)"
        )
    with col2:
        graph_height = st.slider("Altura do gráfico:", 400, 1200, 800, 100)
    with col3:
        show_all_edges = st.checkbox("Mostrar todas conexões", value=False, 
                                     help="Se desmarcado, mostra apenas a melhor conexão de cada ideia")
    
    with st.spinner("Construindo grafo..."):
        try:
            if len(texts) > 0 and 'embeddings_norm' in locals():
                # Criar grafo direcionado
                G = nx.DiGraph()
                
                # Filtrar por iteração máxima
                filtered_indices = []
                for i, label in enumerate(labels):
                    node_type = 'referencia' if types[i] == 'referencia' else 'gerada'
                    iter_num = 0 if node_type == 'referencia' else colors_list[i]
                    
                    # Incluir referência e ideias até max_iter_show
                    if node_type == 'referencia' or iter_num <= max_iter_show:
                        filtered_indices.append(i)
                
                # Adicionar nós filtrados
                for i in filtered_indices:
                    label = labels[i]
                    node_type = 'referencia' if types[i] == 'referencia' else 'gerada'
                    iter_num = 0 if node_type == 'referencia' else colors_list[i]
                    dist = 0.0 if node_type == 'referencia' else (cos_distances[i-1] if ref_text else 0.0)
                    
                    G.add_node(
                        label,
                        tipo=node_type,
                        iteração=iter_num,
                        distância=dist,
                        pos_x=0,  # Será calculado depois
                        pos_y=0
                    )
                
                # Adicionar arestas baseadas no fluxo REAL do experimento
                # Conectar TODAS as ideias de cada iteração à semente que as gerou
                
                if show_all_edges:
                    # Modo "todas conexões": mostrar similaridade de embeddings (top-3)
                    for iter_current in range(1, min(max_iter_show + 1, max(colors_list) + 1)):
                        current_labels = [labels[i] for i in filtered_indices
                                         if types[i] == 'gerada' and colors_list[i] == iter_current]
                        
                        if iter_current == 1:
                            prev_labels = [labels[i] for i in filtered_indices if types[i] == 'referencia']
                        else:
                            prev_labels = [labels[i] for i in filtered_indices
                                          if types[i] == 'gerada' and colors_list[i] == iter_current - 1]
                        
                        for curr_label in current_labels:
                            curr_idx = labels.index(curr_label)
                            curr_emb = embeddings_norm[curr_idx]
                            
                            similarities = []
                            for prev_label in prev_labels:
                                prev_idx = labels.index(prev_label)
                                prev_emb = embeddings_norm[prev_idx]
                                sim = np.dot(curr_emb, prev_emb)
                                similarities.append((prev_label, sim))
                            
                            similarities.sort(key=lambda x: x[1], reverse=True)
                            for prev_label, sim in similarities[:3]:
                                G.add_edge(prev_label, curr_label, weight=sim)
                else:
                    # Modo "fluxo de geração": mostrar quem gerou quem (baseado em chosen_A)
                    # TODAS as ideias de uma iteração vêm da ideia escolhida (chosen_A=1) da iteração anterior
                    for iter_current in range(1, min(max_iter_show + 1, max(colors_list) + 1)):
                        # Pegar TODAS as ideias da iteração atual
                        current_labels = [labels[i] for i in filtered_indices
                                         if types[i] == 'gerada' and colors_list[i] == iter_current]
                        
                        if iter_current == 1:
                            # Primeira iteração: TODAS vêm da referência
                            ref_label = [labels[i] for i in filtered_indices if types[i] == 'referencia']
                            if ref_label:
                                for curr_label in current_labels:
                                    if curr_label in G.nodes():
                                        G.add_edge(ref_label[0], curr_label, weight=1.0)
                        else:
                            # Iterações subsequentes: TODAS vêm da ideia escolhida (chosen_A=1) da iteração anterior
                            prev_iter_rows = df[df['iter'] == iter_current - 1]
                            prev_chosen = prev_iter_rows[prev_iter_rows['chosen_A'] == 1]
                            
                            if not prev_chosen.empty:
                                prev_cand_id = int(prev_chosen.iloc[0]['cand_id'])
                                prev_label = f"Iter{iter_current-1}-Cand{prev_cand_id}"
                                
                                # Conectar TODAS as ideias atuais à semente anterior
                                if prev_label in G.nodes():
                                    for curr_label in current_labels:
                                        if curr_label in G.nodes():
                                            G.add_edge(prev_label, curr_label, weight=1.0)
                
                # Layout hierárquico melhorado (por iteração, de cima para baixo)
                pos = {}
                y_offset = max_iter_show * 2  # Começar do topo
                
                for iter_num in range(max_iter_show + 1):
                    if iter_num == 0:
                        # Referência no topo (centralizada)
                        ref_nodes = [n for n in G.nodes() if G.nodes[n]['tipo'] == 'referencia']
                        for i, node in enumerate(ref_nodes):
                            pos[node] = (0, y_offset)
                    else:
                        # Ideias geradas (espaçadas horizontalmente)
                        iter_nodes = [n for n in G.nodes() if G.nodes[n]['iteração'] == iter_num]
                        if iter_nodes:
                            # Espaçamento horizontal baseado no número de nós
                            total_width = max(4.0, len(iter_nodes) * 1.5)
                            x_spacing = total_width / (len(iter_nodes) + 1)
                            for i, node in enumerate(iter_nodes):
                                x_pos = -total_width/2 + (i + 1) * x_spacing
                                pos[node] = (x_pos, y_offset)
                    
                    y_offset -= 2  # Espaçamento vertical entre iterações
                
                # Criar figura com plotly - SETAS DIRECIONADAS
                edge_trace = []
                annotations = []
                
                for edge in G.edges(data=True):
                    source, target, data = edge
                    x0, y0 = pos[source]
                    x1, y1 = pos[target]
                    
                    # Linha da aresta
                    edge_trace.append(
                        go.Scatter(
                            x=[x0, x1, None],
                            y=[y0, y1, None],
                            mode='lines',
                            line=dict(width=1.5, color='rgba(150,150,150,0.4)'),
                            hoverinfo='none',
                            showlegend=False
                        )
                    )
                    
                    # Seta direcionada (annotation)
                    annotations.append(
                        dict(
                            x=x1,
                            y=y1,
                            ax=x0,
                            ay=y0,
                            xref='x',
                            yref='y',
                            axref='x',
                            ayref='y',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1.5,
                            arrowcolor='rgba(150,150,150,0.4)',
                            standoff=8,
                        )
                    )
                
                # Adicionar labels de iteração no lado esquerdo
                for iter_num in range(max_iter_show + 1):
                    iter_nodes = [n for n in G.nodes() if G.nodes[n]['iteração'] == iter_num]
                    if iter_nodes:
                        # Pegar a posição Y da iteração
                        y_pos = pos[iter_nodes[0]][1]
                        label_text = "Referência" if iter_num == 0 else f"Iteração {iter_num}"
                        
                        annotations.append(
                            dict(
                                x=-6,  # Posição à esquerda
                                y=y_pos,
                                text=f"<b>{label_text}</b>",
                                showarrow=False,
                                xref='x',
                                yref='y',
                                xanchor='right',
                                font=dict(size=11, color='gray')
                            )
                        )
                
                # Nós
                node_trace_ref = go.Scatter(
                    x=[], y=[], mode='markers+text',
                    marker=dict(size=20, color='red', symbol='diamond', line=dict(width=2, color='darkred')),
                    text=[], textposition='top center', textfont=dict(size=10),
                    name='Referencia', hovertext=[], hoverinfo='text'
                )
                
                node_trace_gen = go.Scatter(
                    x=[], y=[], mode='markers+text',
                    marker=dict(size=[], color=[], colorscale='Viridis', showscale=True,
                               colorbar=dict(title="Iteração", x=1.15, xanchor='left', thickness=15, len=0.7)),
                    text=[], textposition='top center', textfont=dict(size=8),
                    name='Ideias Geradas', hovertext=[], hoverinfo='text'
                )
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_data = G.nodes[node]
                    
                    if node_data['tipo'] == 'referencia':
                        node_trace_ref['x'] += tuple([x])
                        node_trace_ref['y'] += tuple([y])
                        node_trace_ref['text'] += tuple([node.split('-')[0]])  # Apenas "Ref"
                        node_trace_ref['hovertext'] += tuple([f"{node}<br>Referencia Original"])
                    else:
                        node_trace_gen['x'] += tuple([x])
                        node_trace_gen['y'] += tuple([y])
                        node_trace_gen['text'] += tuple([f"I{node_data['iteração']}-{node.split('-')[-1]}"])
                        node_trace_gen['marker']['size'] += tuple([15])
                        node_trace_gen['marker']['color'] += tuple([node_data['iteração']])
                        node_trace_gen['hovertext'] += tuple([
                            f"{node}<br>Iteração: {node_data['iteração']}<br>Distância: {node_data['distância']:.4f}"
                        ])
                
                fig = go.Figure(data=edge_trace + [node_trace_ref, node_trace_gen])
                
                fig.update_layout(
                    title=f'Grafo de Evolução: Fluxo Temporal de Ideias (Iterações 1-{max_iter_show})',
                    showlegend=True,
                    hovermode='closest',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-8, 8]),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=graph_height,
                    plot_bgcolor='white',
                    annotations=annotations  # Adicionar setas e labels
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Legenda atualizada
                st.markdown(f"""
                **Legenda:**
                -  **Diamante vermelho**: Referência original (topo)
                -  **Círculos coloridos**: Ideias geradas (cor = iteração, azul escuro → amarelo)
                -  **Setas direcionadas**: Fluxo de evolução temporal
                -  **Labels laterais**: Indicam o número da iteração
                
                **Controles:**
                - **Slider "Mostrar até iteração"**: Filtra visualização (evita poluição com muitas iterações)
                - **Slider "Altura do gráfico"**: Ajusta espaçamento vertical
                - **Checkbox "Mostrar todas conexões"**:
                  -  **Desmarcado (RECOMENDADO)**: Mostra o **fluxo de GERAÇÃO** → cada iteração mostra TODAS as ideias geradas a partir da ideia escolhida (chosen_A=1) da iteração anterior. Visualiza claramente a "árvore de geração".
                  -  **Marcado**: Mostra top-3 conexões por **similaridade de embeddings** (análise exploratória)
                
                **Interpretação (modo desmarcado - FLUXO DE GERAÇÃO):**
                - **Estrutura em árvore**: Cada nó escolhido (chosen_A=1) gera 2 filhos na próxima iteração
                - **Exemplo Iter 1**: Referência → gera → Iter1-Cand1 e Iter1-Cand2 (ambos conectados à referência)
                - **Exemplo Iter 2**: Iter1-Cand2 (escolhido) → gera → Iter2-Cand1 e Iter2-Cand2 (ambos conectados ao Iter1-Cand2)
                - **Nós "folha"**: Ideias que foram geradas mas NÃO escolhidas (não geram filhos)
                - **Caminho principal**: Sequência de nós escolhidos (chosen_A=1) que continuam gerando
                - **Distância no hover**: Quão próxima cada ideia está da referência original
                
                **Interpretação (modo marcado - SIMILARIDADE):**
                - **Múltiplas setas**: Mostra as top-3 ideias mais similares (por embedding) da iteração anterior
                - **Útil para**: Análise semântica, não reflete o fluxo de geração real
                
                **O que procurar (modo desmarcado):**
                - **Ramificações**: Cada ideia escolhida deve ter exatamente 2 filhos (Cand1 e Cand2)
                - **Caminho principal**: Trace a sequência de ideias escolhidas (chosen_A=1)
                - **Ideias "mortas"**: Candidatos rejeitados aparecem como folhas (sem filhos)
                - **Convergência**: Distâncias devem diminuir ao longo do caminho principal
                """)
                
                # Estatísticas do grafo
                st.info(f"""
                 **Estatísticas do Grafo:**
                - **Nós**: {G.number_of_nodes()} ({G.number_of_nodes()-1} ideias + 1 referência)
                - **Conexões**: {G.number_of_edges()} setas
                - **Iterações mostradas**: 1 a {max_iter_show}
                - **Modo**: {'Top-3 conexões' if show_all_edges else 'Melhor conexão apenas'}
                """)
                
                # Debug: Mostrar conexões criadas
                if G.number_of_edges() > 0:
                    st.success(f" {G.number_of_edges()} conexões criadas!")
                    edges_list = list(G.edges())[:5]
                    with st.expander(" Ver exemplo de conexões"):
                        for src, tgt in edges_list:
                            st.write(f"  • {src} → {tgt}")
                else:
                    st.error(" NENHUMA conexão criada! Verifique o log.csv")
            else:
                st.warning(" Carregue os embeddings primeiro (ative PCA ou outro embedding)")
        except Exception as e:
            st.error(f"Erro ao construir grafo: {e}")
            import traceback
            st.code(traceback.format_exc())

st.markdown("---")

# ============================================================================
# 8. UMAP 3D - TODAS AS REFERÊNCIAS
# ============================================================================
if show_all_refs_umap and UMAP_AVAILABLE:
    st.header(" UMAP 3D - Contexto Global (Todas as Referências)")
    
    try:
        # Caminho para o arquivo de referências
        refs_file = st.sidebar.text_input(
            "Arquivo de referências:",
            value="/home/romulo/Documentos/MAI-DAI-USP/refs_combined.txt",
            key="refs_file_global"
        )
        
        if Path(refs_file).exists():
            # Carregar todas as referências
            with open(refs_file, 'r', encoding='utf-8') as f:
                content = f.read()
                all_refs = [r.strip() for r in content.split('---') if r.strip()]
            
            st.info(f" Carregadas {len(all_refs)} referências do arquivo")
            
            # Extrair ID da referência atual (ex: ref_001 -> 1)
            current_ref_id = int(selected_ref.split('_')[1])
            current_ref_text = all_refs[current_ref_id - 1] if current_ref_id <= len(all_refs) else ""
            
            # Carregar embeddings das iterações (já calculados anteriormente)
            if 'embeddings' in locals() or 'embeddings' in globals():
                # Usar embeddings já carregados
                iter_embeddings = embeddings
                iter_labels = labels
                iter_types = types
            else:
                # Carregar embeddings das iterações
                from sentence_transformers import SentenceTransformer
                
                with st.spinner(" Gerando embeddings..."):
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    # Embeddings das iterações
                    iter_texts = []
                    iter_labels = []
                    iter_types = []
                    
                    # Referência original
                    iter_texts.append(current_ref_text)
                    iter_labels.append("Referência")
                    iter_types.append("referencia")
                    
                    # Ideias geradas
                    for _, row in df.iterrows():
                        iter_num = int(row['iter'])
                        cand_id = int(row['cand_id'])
                        file_path = ref_path / f"iter_{iter_num:03d}" / f"{cand_id}.txt"
                        
                        if file_path.exists():
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                                iter_texts.append(text)
                                iter_labels.append(f"Iter{iter_num}-Cand{cand_id}")
                                iter_types.append("gerada")
                    
                    iter_embeddings = model.encode(iter_texts, normalize_embeddings=True)
            
            # Gerar embeddings de TODAS as referências
            with st.spinner(" Gerando embeddings de todas as referências..."):
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                all_refs_embeddings = model.encode(all_refs, normalize_embeddings=True)
            
            # Combinar todos os embeddings
            # Estrutura:
            # 1. Todas as referências (exceto a atual)
            # 2. Referência atual
            # 3. Iterações geradas
            
            combined_embeddings = []
            combined_labels = []
            combined_types = []
            combined_colors = []
            
            # 1. Outras referências (cinza claro)
            for i, ref_emb in enumerate(all_refs_embeddings):
                ref_num = i + 1
                if ref_num != current_ref_id:
                    combined_embeddings.append(ref_emb)
                    combined_labels.append(f"Ref{ref_num:03d}")
                    combined_types.append("outras_refs")
                    combined_colors.append(0)  # Cinza
            
            # 2. Referência atual (vermelho)
            combined_embeddings.append(all_refs_embeddings[current_ref_id - 1])
            combined_labels.append(f"Ref{current_ref_id:03d} (Atual)")
            combined_types.append("ref_atual")
            combined_colors.append(-1)  # Vermelho
            
            # 3. Iterações geradas (gradiente por iteração)
            for i, emb in enumerate(iter_embeddings):
                if iter_types[i] == "gerada":
                    combined_embeddings.append(emb)
                    combined_labels.append(iter_labels[i])
                    combined_types.append("gerada")
                    # Extrair número da iteração para colorir
                    iter_num = int(iter_labels[i].split('-')[0].replace('Iter', ''))
                    combined_colors.append(iter_num)
            
            combined_embeddings = np.array(combined_embeddings)
            
            # Aplicar UMAP 3D
            with st.spinner(" Aplicando UMAP 3D..."):
                umap_model = UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
                coords_3d = umap_model.fit_transform(combined_embeddings)
            
            # Criar DataFrame para Plotly
            plot_df = pd.DataFrame({
                'x': coords_3d[:, 0],
                'y': coords_3d[:, 1],
                'z': coords_3d[:, 2],
                'label': combined_labels,
                'type': combined_types,
                'color': combined_colors
            })
            
            # Criar figura Plotly
            fig = go.Figure()
            
            # 1. Outras referências (cruz azul claro)
            other_refs = plot_df[plot_df['type'] == 'outras_refs']
            if len(other_refs) > 0:
                fig.add_trace(go.Scatter3d(
                    x=other_refs['x'],
                    y=other_refs['y'],
                    z=other_refs['z'],
                    mode='markers',
                    name='Outras Referências',
                    marker=dict(
                        size=4,
                        color='cyan',
                        opacity=0.6,
                        symbol='x',
                        line=dict(color='darkblue', width=1)
                    ),
                    text=other_refs['label'],
                    hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
                    showlegend=True
                ))
            
            # 2. Referência atual (grande estrela vermelha)
            ref_atual = plot_df[plot_df['type'] == 'ref_atual']
            if len(ref_atual) > 0:
                fig.add_trace(go.Scatter3d(
                    x=ref_atual['x'],
                    y=ref_atual['y'],
                    z=ref_atual['z'],
                    mode='markers',
                    name=f'Referência {current_ref_id:03d} (Geradora)',
                    marker=dict(
                        size=20,
                        color='red',
                        symbol='diamond',
                        line=dict(color='darkred', width=2)
                    ),
                    text=ref_atual['label'],
                    hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
                    showlegend=True
                ))
            
            # 3. Iterações geradas (gradiente de cor)
            geradas = plot_df[plot_df['type'] == 'gerada']
            if len(geradas) > 0:
                fig.add_trace(go.Scatter3d(
                    x=geradas['x'],
                    y=geradas['y'],
                    z=geradas['z'],
                    mode='markers',
                    name='Ideias Geradas',
                    marker=dict(
                        size=8,
                        color=geradas['color'],
                        colorscale='Viridis',
                        showscale=True,
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
                    text=geradas['label'],
                    hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
                    showlegend=True
                ))
            
            # 4. Melhor ideia (menor distância da referência)
            if len(df) > 0:
                best_idx = df['dist'].idxmin()
                best_row = df.loc[best_idx]
                best_iter = int(best_row['iter'])
                best_cand = int(best_row['cand_id'])
                best_dist = float(best_row['dist'])
                best_label = f"Iter{best_iter}-Cand{best_cand}"
                
                # Encontrar coordenadas da melhor ideia no plot_df
                best_coords = plot_df[plot_df['label'] == best_label]
                if len(best_coords) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=best_coords['x'],
                        y=best_coords['y'],
                        z=best_coords['z'],
                        mode='markers',
                        name=f'Melhor Ideia (dist={best_dist:.4f})',
                        marker=dict(
                            size=18,
                            color='gold',
                            symbol='diamond',
                            line=dict(color='darkorange', width=3)
                        ),
                        text=[f"{best_label}<br>Distância: {best_dist:.4f}"],
                        hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
                        showlegend=True
                    ))
            
            # Layout
            fig.update_layout(
                title=f"UMAP 3D - Contexto Global: {len(all_refs)} Referências + {len(geradas)} Ideias Geradas",
                scene=dict(
                    xaxis_title="UMAP 1",
                    yaxis_title="UMAP 2",
                    zaxis_title="UMAP 3",
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.3)
                    )
                ),
                height=700,
                showlegend=True,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='gray',
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Legenda explicativa
            best_dist_val = df['dist'].min()
            best_row = df.loc[df['dist'].idxmin()]
            best_label_text = f"Iter{int(best_row['iter'])}-Cand{int(best_row['cand_id'])}"
            
            st.markdown("""
            **Como interpretar este gráfico:**
            
            - **Cruzes ciano (✖)**: Todas as outras referências do corpus ({} refs)
            - **Diamante vermelho grande**: Referência {} que gerou as iterações
            - **Pontos coloridos**: Ideias geradas nas iterações (cor = número da iteração)
            - **Diamante dourado grande**: Melhor ideia gerada ({} com distância {:.4f})
            
            **O que procurar:**
            - **Clusters**: Referências similares (cruzes ciano) ficam próximas no espaço 3D
            - **Trajetória**: As ideias geradas (coloridas) devem convergir em direção à referência vermelha
            - **Convergência**: O diamante dourado mostra o ponto de maior aproximação alcançado
            - **Contexto**: Veja como a referência atual se relaciona com todas as outras do corpus
            """.format(len(other_refs), current_ref_id, best_label_text, best_dist_val))
            
        else:
            st.error(f" Arquivo de referências não encontrado: {refs_file}")
    
    except Exception as e:
        st.error(f"Erro ao gerar UMAP 3D global: {e}")
        import traceback
        st.code(traceback.format_exc())

st.markdown("---")

# ============================================================================
# 9. TABELA DE DADOS
# ============================================================================
with st.expander(" Ver dados completos (log.csv)"):
    st.dataframe(df, use_container_width=True)
    
    # Download
    csv = df.to_csv(index=False)
    st.download_button(
        label=" Baixar CSV",
        data=csv,
        file_name=f"{selected_ref}_log.csv",
        mime="text/csv"
    )

# Rodape
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
     Visualização de Experimento Iterativo | Porta 8503
</div>
""", unsafe_allow_html=True)
