#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizacao grafica dos resultados do experimento iterativo.
Mostra distancias, embeddings e evolucao das ideias.
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
import warnings
warnings.filterwarnings('ignore')

# Import UMAP with fallback
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

st.set_page_config(
    page_title="Visualizacao Experimento",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Visualizacao do Experimento Iterativo")
st.markdown("Analise grafica das distancias e embeddings gerados")
st.markdown("---")

# Sidebar - Selecao de experimento
with st.sidebar:
    st.header("⚙️ Configuracoes")
    
    exp_dir = st.text_input(
        "Diretorio do experimento:",
        value="exp_out"
    )
    
    exp_path = Path(exp_dir)
    
    if not exp_path.exists():
        st.error(f"❌ Diretorio nao encontrado: {exp_dir}")
        st.stop()
    
    # Listar referencias disponiveis
    ref_dirs = sorted([d for d in exp_path.iterdir() if d.is_dir() and d.name.startswith("ref_")])
    
    if not ref_dirs:
        st.error("❌ Nenhuma referencia encontrada")
        st.stop()
    
    st.success(f"✅ {len(ref_dirs)} referencias encontradas")
    
    ref_names = [d.name for d in ref_dirs]
    selected_ref = st.selectbox(
        "Selecione a referencia:",
        ref_names,
        index=0
    )
    
    st.markdown("---")
    
    # Opcoes de visualizacao
    st.subheader("📈 Graficos")
    show_convergence = st.checkbox("Grafico de convergencia", value=True)
    show_embeddings_3d = st.checkbox("Embeddings 3D (PCA Interativo)", value=True)
    show_embeddings_tsne = st.checkbox("Embeddings 3D (t-SNE Interativo)", value=False)
    if UMAP_AVAILABLE:
        show_embeddings_umap = st.checkbox("Embeddings 3D (UMAP Interativo)", value=True)
    else:
        show_embeddings_umap = False
        st.info("ℹ️ UMAP nao disponivel. Instale com: pip install umap-learn")
    show_heatmap = st.checkbox("Heatmap de distancias", value=True)
    show_best_worst = st.checkbox("Comparacao melhor vs pior", value=True)

# Carregar dados da referencia selecionada
ref_path = exp_path / selected_ref
log_file = ref_path / "log.csv"

if not log_file.exists():
    st.error(f"❌ Arquivo log.csv nao encontrado em {ref_path}")
    st.stop()

# Ler log
df = pd.read_csv(log_file)

st.header(f"📄 Referencia: {selected_ref}")

# Metricas resumidas
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de iteracoes", df['iter'].max())

with col2:
    best_dist = df['dist'].min()
    st.metric("Melhor distancia", f"{best_dist:.4f}")

with col3:
    worst_dist = df['dist'].max()
    st.metric("Pior distancia", f"{worst_dist:.4f}")

with col4:
    improvement = ((worst_dist - best_dist) / worst_dist) * 100
    st.metric("Melhoria (%)", f"{improvement:.1f}%")

st.markdown("---")

# ============================================================================
# 1. GRAFICO DE CONVERGENCIA
# ============================================================================
if show_convergence:
    st.subheader("📈 Grafico de Convergencia")
    
    fig = go.Figure()
    
    # Agrupar por iteracao
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
                hovertemplate=f"<b>Iteracao {iter_num}</b><br>" +
                              f"Candidato: {cand_id}<br>" +
                              f"Distancia: {dist:.4f}<br>" +
                              f"Status: {'Escolhido (A)' if chosen else 'Rejeitado (B)'}<extra></extra>"
            ))
    
    # Linha da melhor distancia acumulada
    best_so_far = df.groupby('iter')['dmin_so_far'].first()
    fig.add_trace(go.Scatter(
        x=best_so_far.index,
        y=best_so_far.values,
        mode='lines+markers',
        line=dict(color='blue', width=2, dash='dash'),
        marker=dict(size=8, color='blue'),
        name='Melhor acumulada',
        hovertemplate="<b>Melhor ate iter %{x}</b><br>Distancia: %{y:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Evolucao das Distancias por Iteracao",
        xaxis_title="Iteracao",
        yaxis_title="Distancia Cosseno",
        hovermode='closest',
        height=500,
        showlegend=True,
        legend=dict(x=0.7, y=0.95)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Legenda
    st.markdown("""
    **Legenda:**
    - 🟢 **Estrela verde**: Ideia escolhida como A (melhor)
    - 🔴 **Circulo vermelho**: Ideia rejeitada como B (pior)
    - 🔵 **Linha azul tracejada**: Melhor distancia acumulada
    """)

st.markdown("---")

# ============================================================================
# 2. EMBEDDINGS 3D (PCA INTERATIVO)
# ============================================================================
if show_embeddings_3d:
    st.subheader("🎯 Visualizacao de Embeddings (PCA 3D Interativo)")
    
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
                ref_path / "referencia.txt",  # se houver arquivo especifico
            ]
            
            # Extrair numero da referencia (ex: ref_001 -> 1)
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
                        labels.append(f"Iter {iter_num} - Cand {cand_id}")
                        colors_list.append(iter_num)
                        types.append('gerada')
            
            if len(texts) > 0:
                # Gerar embeddings
                embeddings = embedder.encode(texts, show_progress_bar=False)
                
                # NORMALIZAR embeddings para que distancia euclidiana seja proporcional a cosseno
                # Apos normalizacao: dist_euclidiana^2 = 2 * (1 - cos_similarity)
                embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                # Calcular distancias da referencia (se existir)
                if ref_text:
                    ref_embedding = embeddings_norm[0]  # primeiro embedding e a referencia
                    gen_embeddings = embeddings_norm[1:]  # resto sao ideias geradas
                    
                    # Distancia cosseno (como no experimento)
                    cos_distances = 1 - np.dot(gen_embeddings, ref_embedding)
                    
                    # Distancia euclidiana (apos normalizacao)
                    eucl_distances = np.linalg.norm(gen_embeddings - ref_embedding, axis=1)
                    
                    # Adicionar ao DataFrame para exibir
                    st.info(f"📊 Distancias calculadas: {len(cos_distances)} ideias vs referencia")
                    st.write(f"**Distancia cosseno:** min={cos_distances.min():.4f}, max={cos_distances.max():.4f}, media={cos_distances.mean():.4f}")
                    st.write(f"**Distancia euclidiana:** min={eucl_distances.min():.4f}, max={eucl_distances.max():.4f}, media={eucl_distances.mean():.4f}")
                    st.write(f"**Relacao teorica:** dist_eucl = √(2 × dist_cos) → verificacao: {np.allclose(eucl_distances, np.sqrt(2 * cos_distances), atol=1e-5)}")
                
                # Reduzir para 3D com PCA (usando embeddings normalizados)
                pca = PCA(n_components=3)
                embeddings_3d = pca.fit_transform(embeddings_norm)
                
                # Criar DataFrame
                df_viz = pd.DataFrame({
                    'x': embeddings_3d[:, 0],
                    'y': embeddings_3d[:, 1],
                    'z': embeddings_3d[:, 2],
                    'label': labels,
                    'iteracao': colors_list,
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
                    # Encontrar a melhor ideia (menor distancia cosseno)
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
                                color=df_gen_normal['iteracao'],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Iteracao"),
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
                        hovertemplate="<b>🏆 MELHOR IDEIA</b><br>" +
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
                
                st.info(f"📊 Variancia explicada: {sum(pca.explained_variance_ratio_)*100:.1f}%")
                
                # Legenda
                st.markdown(f"""
                **Legenda:**
                - 💎 **Diamante vermelho grande**: Referencia original (alvo)
                - 💎 **Diamante dourado**: MELHOR ideia gerada (menor distancia)
                - 🔵 **Esferas coloridas**: Outras ideias geradas (cor = iteracao)
                - **Objetivo**: Ideias devem se aproximar do diamante vermelho!
                
                **🎮 Controles Interativos:**
                - **Rotacionar**: Clique e arraste
                - **Zoom**: Scroll do mouse ou pinch
                - **Pan**: Shift + arraste
                - **Reset**: Duplo clique
                
                **✨ Vantagens do 3D:**
                - Captura **{sum(pca.explained_variance_ratio_)*100:.1f}%** da variancia (vs ~34% em 2D)
                - Melhor percepcao de distancias e agrupamentos
                - Explore diferentes angulos para entender a convergencia
                
                **✨ Nota tecnica:** Embeddings foram **normalizados** antes do PCA, garantindo que a 
                distancia euclidiana no grafico seja **proporcional** a distancia cosseno usada no experimento.
                Use o hover para ver as distancias reais de cada ideia.
                """)
            else:
                st.warning("⚠️ Nenhum texto encontrado para visualizar")
                
        except Exception as e:
            st.error(f"❌ Erro ao gerar embeddings: {e}")

st.markdown("---")

# ============================================================================
# 3. t-SNE 3D INTERATIVO
# ============================================================================
if show_embeddings_tsne:
    st.subheader("🎯 Visualizacao de Embeddings (t-SNE 3D Interativo)")
    
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
                    'iteracao': colors_list,
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
                    # Encontrar a melhor ideia (menor distancia cosseno)
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
                                color=df_gen_normal['iteracao'],
                                colorscale='Plasma',
                                showscale=True,
                                colorbar=dict(title="Iteracao"),
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
                        hovertemplate="<b>🏆 MELHOR IDEIA</b><br>" +
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
                - 💎 **Diamante vermelho grande**: Referencia original (alvo)
                - 💎 **Diamante dourado**: MELHOR ideia gerada (menor distancia)
                - 🔵 **Esferas coloridas**: Outras ideias geradas (cor = iteracao)
                - **Objetivo**: Ideias devem se aproximar do diamante vermelho!
                
                **🎮 Controles Interativos:**
                - **Rotacionar**: Clique e arraste
                - **Zoom**: Scroll do mouse ou pinch
                - **Pan**: Shift + arraste
                - **Reset**: Duplo clique
                
                **✨ Vantagens do t-SNE 3D:**
                - Melhor visualização de **clusters** e agrupamentos
                - Estrutura local preservada em 3 dimensões
                - Identifica "famílias" de ideias similares
                - Explore diferentes ângulos para ver padrões
                
                **✨ Nota tecnica:** Embeddings foram **normalizados** antes do t-SNE, garantindo que a 
                distancia euclidiana no grafico seja **proporcional** a distancia cosseno usada no experimento.
                
                **⚠️ Importante:** t-SNE preserva estrutura **local** (clusters), não distâncias globais. 
                Ideias próximas no gráfico são semanticamente similares, mas distâncias absolutas podem 
                não refletir as distâncias reais. Use o hover para ver as distâncias verdadeiras!
                """)
            else:
                st.warning("⚠️ Carregue os embeddings primeiro (ative PCA)")
        except Exception as e:
            st.error(f"❌ Erro ao calcular t-SNE: {e}")

st.markdown("---")

# ============================================================================
# 3.5. VISUALIZACAO UMAP 3D INTERATIVO
# ============================================================================
if show_embeddings_umap and UMAP_AVAILABLE:
    st.subheader("🌐 Visualizacao de Embeddings (UMAP 3D Interativo)")
    
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
                    'iteracao': colors_list,
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
                    # Encontrar a melhor ideia (menor distancia cosseno)
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
                                color=df_gen_normal['iteracao'],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Iteracao"),
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
                        hovertemplate="<b>🏆 MELHOR IDEIA</b><br>" +
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
                - 💎 **Diamante vermelho grande**: Referencia original (alvo)
                - 💎 **Diamante dourado**: MELHOR ideia gerada (menor distancia)
                - 🔵 **Esferas coloridas**: Outras ideias geradas (cor = iteracao)
                - **Objetivo**: Ideias devem se aproximar do diamante vermelho!
                
                **🎮 Controles Interativos:**
                - **Rotacionar**: Clique e arraste
                - **Zoom**: Scroll do mouse ou pinch
                - **Pan**: Shift + arraste
                - **Reset**: Duplo clique
                
                **✨ Vantagens do UMAP 3D:**
                - 🏆 **MELHOR DOS DOIS MUNDOS**: Preserva estrutura **local E global**
                - ⚡ **Mais rapido** que t-SNE 3D
                - 🎯 **Mais consistente**: Mesmo resultado sempre (determinístico)
                - 📏 **Distancias confiaveis**: Usa metrica cosseno diretamente
                - 🌈 **Convergencia clara**: Veja o "caminho" completo das ideias
                - 🔍 **Mais informacao**: 3D preserva muito mais estrutura que 2D
                
                **📊 Interpretacao:** 
                UMAP 3D é a **melhor visualização** para entender convergência! Ele mostra:
                - **Estrutura local**: Clusters de ideias similares (como t-SNE)
                - **Estrutura global**: Distâncias reais entre grupos (como PCA)
                - **Caminho de convergência**: Ideias iniciais (azul) → finais (amarelo) → referência (vermelho)
                
                Explore diferentes ângulos para ver como as ideias se aproximam da referência!
                
                **✨ Nota tecnica:** Embeddings foram **normalizados** antes do UMAP, garantindo que a 
                distancia euclidiana no grafico seja **proporcional** a distancia cosseno usada no experimento.
                Use o hover para ver as distancias reais de cada ideia.
                """)
            else:
                st.warning("⚠️ Carregue os embeddings primeiro (ative PCA)")
        except Exception as e:
            st.error(f"❌ Erro ao calcular UMAP: {e}")

st.markdown("---")

# ============================================================================
# 4. HEATMAP DE DISTANCIAS
# ============================================================================
if show_heatmap:
    st.subheader("🔥 Heatmap de Distancias")
    
    # Criar matriz de distancias por iteracao e candidato
    pivot = df.pivot(index='iter', columns='cand_id', values='dist')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Distancia Cosseno'},
        ax=ax
    )
    ax.set_title('Distancias por Iteracao e Candidato')
    ax.set_xlabel('Candidato')
    ax.set_ylabel('Iteracao')
    
    st.pyplot(fig)

st.markdown("---")

# ============================================================================
# 5. COMPARACAO MELHOR VS PIOR
# ============================================================================
if show_best_worst:
    st.subheader("⚖️ Comparacao: Melhor vs Pior Ideia")
    
    # Encontrar melhor e pior
    best_row = df.loc[df['dist'].idxmin()]
    worst_row = df.loc[df['dist'].idxmax()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Melhor Ideia")
        st.metric("Distancia", f"{best_row['dist']:.4f}")
        st.metric("Iteracao", int(best_row['iter']))
        st.metric("Candidato", int(best_row['cand_id']))
        
        best_file = Path(best_row['files'])
        if best_file.exists():
            with open(best_file, 'r', encoding='utf-8') as f:
                st.text_area("Texto:", f.read(), height=300, key="best")
    
    with col2:
        st.markdown("### 💔 Pior Ideia")
        st.metric("Distancia", f"{worst_row['dist']:.4f}")
        st.metric("Iteracao", int(worst_row['iter']))
        st.metric("Candidato", int(worst_row['cand_id']))
        
        worst_file = Path(worst_row['files'])
        if worst_file.exists():
            with open(worst_file, 'r', encoding='utf-8') as f:
                st.text_area("Texto:", f.read(), height=300, key="worst")

st.markdown("---")

# ============================================================================
# 6. TABELA DE DADOS
# ============================================================================
with st.expander("📋 Ver dados completos (log.csv)"):
    st.dataframe(df, use_container_width=True)
    
    # Download
    csv = df.to_csv(index=False)
    st.download_button(
        label="💾 Baixar CSV",
        data=csv,
        file_name=f"{selected_ref}_log.csv",
        mime="text/csv"
    )

# Rodape
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    📊 Visualizacao de Experimento Iterativo | Porta 8503
</div>
""", unsafe_allow_html=True)
