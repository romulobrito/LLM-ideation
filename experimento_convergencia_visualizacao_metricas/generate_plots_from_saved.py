#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para gerar graficos a partir de experimentos ja salvos.

Uso:
    # Modo 1: Gerar para o experimento mais recente (padrao)
    python generate_plots_from_saved.py
    
    # Modo 2: Gerar para TODOS os experimentos
    python generate_plots_from_saved.py --all
    
    # Modo 3: Gerar para um experimento especifico
    python generate_plots_from_saved.py exp_refinement/20251031_162847
    python generate_plots_from_saved.py exp_refinement/20251031_162847/plots
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Optional

# Carregar variaveis de ambiente do .env
try:
    from dotenv import load_dotenv
    script_dir = Path(__file__).parent
    env_path = script_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print("[INFO] Arquivo .env carregado")
    else:
        print("[AVISO] Arquivo .env nao encontrado. Usando variaveis de ambiente do sistema.")
except ImportError:
    print("[AVISO] python-dotenv nao disponivel. Usando variaveis de ambiente do sistema.")

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("ERRO: Plotly nao disponivel. Instale com: pip install plotly")
    sys.exit(1)


def load_experiment(exp_dir: Path) -> dict:
    """Carrega dados do experimento salvo."""
    summary_file = exp_dir / "summary.json"
    
    if not summary_file.exists():
        raise FileNotFoundError(f"summary.json nao encontrado em: {exp_dir}")
    
    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)
    
    # Carregar iteracoes individuais para metricas completas
    iterations_data = []
    for iter_num in range(1, summary["total_iterations"] + 1):
        iter_file = exp_dir / f"iteration_{iter_num:02d}.json"
        if iter_file.exists():
            with open(iter_file, "r", encoding="utf-8") as f:
                iter_data = json.load(f)
                iterations_data.append(iter_data)
        else:
            print(f"[AVISO] Iteracao {iter_num} nao encontrada, pulando...")
    
    return {
        "summary": summary,
        "iterations": iterations_data
    }


def generate_convergence_plot(exp_data: dict, output_dir: Path) -> None:
    """Gera grafico de convergencia com multiplas metricas."""
    summary = exp_data["summary"]
    iterations = exp_data["iterations"]
    
    if not iterations:
        print("[AVISO] Nenhuma iteracao encontrada. Grafico de convergencia nao sera gerado.")
        return
    
    # Extrair metricas
    iter_nums = [item.get("iteration", i+1) for i, item in enumerate(iterations)]
    avg_distances = [item["avg_distance"] for item in iterations]
    min_distances = [item["min_distance"] for item in iterations]
    
    top3_distances = [item.get("top3_mean_distance") for item in iterations]
    top3_distances = [d for d in top3_distances if d is not None]
    
    centroid_distances = [item.get("centroid_distance") for item in iterations]
    centroid_distances = [d for d in centroid_distances if d is not None]
    
    c2c_distances = [item.get("centroid_to_centroid") for item in iterations]
    c2c_distances = [d for d in c2c_distances if d is not None]
    
    fig = go.Figure()
    
    # Media
    fig.add_trace(go.Scatter(
        x=iter_nums,
        y=avg_distances,
        mode="lines+markers",
        name="Media (avg)",
        line=dict(color="blue", width=2, dash="dot"),
        marker=dict(size=6)
    ))
    
    # Minima
    fig.add_trace(go.Scatter(
        x=iter_nums,
        y=min_distances,
        mode="lines+markers",
        name="Minima (min)",
        line=dict(color="green", width=2, dash="dot"),
        marker=dict(size=6)
    ))
    
    # Top-3 Media
    if top3_distances and len(top3_distances) == len(iter_nums):
        fig.add_trace(go.Scatter(
            x=iter_nums,
            y=top3_distances,
            mode="lines+markers",
            name="Top-3 Media (OTIMIZACAO)",
            line=dict(color="red", width=3),
            marker=dict(size=10)
        ))
    
    # Centroide
    if centroid_distances and len(centroid_distances) == len(iter_nums):
        fig.add_trace(go.Scatter(
            x=iter_nums,
            y=centroid_distances,
            mode="lines+markers",
            name="Centroide (media)",
            line=dict(color="purple", width=2, dash="dot"),
            marker=dict(size=6)
        ))
    
    # Centroid-to-Centroid
    if c2c_distances and len(c2c_distances) == len(iter_nums):
        fig.add_trace(go.Scatter(
            x=iter_nums,
            y=c2c_distances,
            mode="lines+markers",
            name="Centroid-to-Centroid (ESTAVEL)",
            line=dict(color="orange", width=3),
            marker=dict(size=10, symbol="diamond")
        ))
    
    fig.update_layout(
        xaxis_title="Iteracao",
        yaxis_title="Distancia Coseno",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        width=1200,
        title="Grafico de Convergencia - Multiplas Metricas"
    )
    
    # Salvar
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_dir / "convergencia.html"))
    print(f"[OK] Grafico de convergencia salvo: {output_dir / 'convergencia.html'}")
    
    try:
        fig.write_image(str(output_dir / "convergencia.png"), width=1200, height=600, scale=2)
        print(f"[OK] Imagem PNG salva: {output_dir / 'convergencia.png'}")
    except Exception as e:
        print(f"[AVISO] PNG nao gerado (kaleido nao disponivel): {e}")


def generate_trajectory_plot(exp_data: dict, output_dir: Path) -> None:
    """Gera grafico de trajetoria (distancia da iteracao 1)."""
    iterations = exp_data["iterations"]
    
    if not iterations:
        return
    
    iter1_distances = [item.get("distance_from_iter1") for item in iterations]
    iter1_distances = [d for d in iter1_distances if d is not None]
    
    if not iter1_distances or len(iter1_distances) != len(iterations):
        print("[AVISO] Metrica 'distance_from_iter1' nao disponivel. Grafico de trajetoria nao sera gerado.")
        return
    
    iter_nums = [item.get("iteration", i+1) for i, item in enumerate(iterations)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=iter_nums,
        y=iter1_distances,
        mode="lines+markers",
        name="Distancia da Iter 1",
        line=dict(color="teal", width=3),
        marker=dict(size=10, symbol="circle"),
        fill='tozeroy',
        fillcolor='rgba(0, 128, 128, 0.1)'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray",
                  annotation_text="Baseline (Iter 1)",
                  annotation_position="right")
    
    fig.update_layout(
        xaxis_title="Iteracao",
        yaxis_title="Distancia Coseno (Centroide)",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        width=1200,
        title="Trajetoria de Convergencia - Distancia da Iteracao 1"
    )
    
    # Salvar
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_dir / "trajetoria_iter1.html"))
    print(f"[OK] Grafico de trajetoria salvo: {output_dir / 'trajetoria_iter1.html'}")
    
    try:
        fig.write_image(str(output_dir / "trajetoria_iter1.png"), width=1200, height=600, scale=2)
        print(f"[OK] Imagem PNG salva: {output_dir / 'trajetoria_iter1.png'}")
    except Exception as e:
        print(f"[AVISO] PNG nao gerado (kaleido nao disponivel): {e}")


def generate_umap_plot(exp_data: dict, output_dir: Path) -> None:
    """Gera grafico UMAP 3D completo."""
    try:
        from umap import UMAP
        UMAP_AVAILABLE = True
    except ImportError:
        print("[AVISO] UMAP nao disponivel. Grafico UMAP 3D nao sera gerado.")
        return
    
    try:
        from experiment_iterativo import get_embedder, embed_texts, load_references_from_fs, cosine_distance
        import numpy as np
        import pandas as pd
    except ImportError as e:
        print(f"[AVISO] Dependencias faltando para UMAP: {e}")
        return
    
    summary = exp_data["summary"]
    iterations = exp_data["iterations"]
    
    if not iterations:
        print("[AVISO] Nenhuma iteracao encontrada. Grafico UMAP nao sera gerado.")
        return
    
    embedder_name = summary.get("config", {}).get("embedder", "all-MiniLM-L6-v2")
    
    print(f"[INFO] Carregando embedder: {embedder_name}")
    try:
        embedder = get_embedder(embedder_name)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar embedder: {e}")
        return
    
    # Carregar todas as ideias humanas disponiveis
    print("[INFO] Carregando ideias humanas...")
    try:
        human_ideas_all = load_references_from_fs("ideas-exp/human")
    except Exception:
        # Fallback: tentar usar as ideias do summary
        num_human = summary.get("config", {}).get("num_human_ideas", 0)
        human_ideas_all = []
        print(f"[AVISO] Nao foi possivel carregar de ideas-exp/human. Usando {num_human} ideias do summary.")
    
    human_embeddings_all = embed_texts(embedder, human_ideas_all) if human_ideas_all else []
    num_human_used = summary.get("config", {}).get("num_human_ideas", len(human_ideas_all)) if human_ideas_all else 0
    
    # Coletar todas as ideias geradas
    all_embeddings = []
    all_labels = []
    all_types = []
    all_iterations = []
    all_distances = []
    all_cluster_ids = []
    
    # Verificar se clustering foi usado
    clustering_info = summary.get("clustering", None)
    use_clustering_viz = clustering_info is not None
    cluster_labels = clustering_info.get("cluster_labels", []) if clustering_info else []
    selected_cluster = clustering_info.get("selected_cluster_id", None) if clustering_info else None
    
    # Adicionar ideias humanas (se disponiveis)
    if human_embeddings_all is not None and len(human_embeddings_all) > 0:
        for i, emb in enumerate(human_embeddings_all):
            all_embeddings.append(emb)
            all_labels.append(f"Humana {i+1}")
            
            cluster_id = cluster_labels[i] if i < len(cluster_labels) else -1
            all_cluster_ids.append(cluster_id)
            
            if use_clustering_viz:
                if cluster_id == selected_cluster:
                    all_types.append("humana_cluster_selecionado")
                elif cluster_id >= 0:
                    all_types.append("humana_outro_cluster")
                else:
                    all_types.append("humana_nao_usada")
            else:
                if i < num_human_used:
                    all_types.append("humana_usada")
                else:
                    all_types.append("humana_nao_usada")
            
            all_iterations.append(0)
            all_distances.append(0.0)
    
    # Adicionar ideias geradas
    print("[INFO] Processando ideias geradas...")
    for iter_data in iterations:
        iter_num = iter_data.get("iteration", 0)
        generated_ideas = iter_data.get("generated_ideas", [])
        
        if not generated_ideas:
            continue
        
        iter_embeddings = embed_texts(embedder, generated_ideas)
        iter_distances = iter_data.get("individual_distances", [])
        
        # Recalcular distancias se nao estiverem salvas
        if not iter_distances or len(iter_distances) != len(generated_ideas):
            human_emb_used = human_embeddings_all[:num_human_used] if (human_embeddings_all is not None and len(human_embeddings_all) > 0) else []
            iter_distances = []
            for emb in iter_embeddings:
                if human_emb_used is not None and len(human_emb_used) > 0:
                    dists_to_humans = [cosine_distance(emb, h_emb) for h_emb in human_emb_used]
                    min_dist = min(dists_to_humans)
                else:
                    min_dist = 0.0
                iter_distances.append(min_dist)
        
        for i, emb in enumerate(iter_embeddings):
            all_embeddings.append(emb)
            all_labels.append(f"Iter {iter_num} - Ideia {i+1}")
            all_types.append("gerada")
            all_cluster_ids.append(-1)
            all_iterations.append(iter_num)
            all_distances.append(iter_distances[i] if i < len(iter_distances) else 0.0)
    
    if len(all_embeddings) == 0:
        print("[AVISO] Nenhum embedding disponivel. Grafico UMAP nao sera gerado.")
        return
    
    # Aplicar UMAP
    print("[INFO] Aplicando UMAP 3D...")
    embeddings_array = np.array(all_embeddings)
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
        'cluster_id': all_cluster_ids,
        'iteracao': all_iterations,
        'distancia': all_distances
    })
    
    # Criar figura 3D
    print("[INFO] Criando grafico UMAP 3D...")
    fig_umap = go.Figure()
    
    # Ideias humanas
    if use_clustering_viz:
        # Cluster selecionado
        df_cluster_selected = df_umap[df_umap['tipo'] == 'humana_cluster_selecionado']
        if len(df_cluster_selected) > 0:
            fig_umap.add_trace(go.Scatter3d(
                x=df_cluster_selected['x'], y=df_cluster_selected['y'], z=df_cluster_selected['z'],
                mode='markers',
                marker=dict(size=14, color='darkred', symbol='diamond', line=dict(color='black', width=3)),
                name=f'Cluster {selected_cluster} (SELECIONADO)',
                text=df_cluster_selected['label'],
                hovertemplate="<b>%{text}</b><br>Cluster: %{customdata} (SELECIONADO)<br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<br>UMAP3: %{z:.3f}<extra></extra>",
                customdata=df_cluster_selected['cluster_id']
            ))
        
        # Outros clusters
        df_outros = df_umap[df_umap['tipo'] == 'humana_outro_cluster']
        if len(df_outros) > 0:
            cluster_colors = {
                0: 'lightblue', 1: 'lightgreen', 2: 'lightyellow', 3: 'lightpink',
                4: 'lightcoral', 5: 'lightcyan', 6: 'lavender', 7: 'lightsalmon',
            }
            
            for cluster_id in df_outros['cluster_id'].unique():
                df_cluster = df_outros[df_outros['cluster_id'] == cluster_id]
                color = cluster_colors.get(cluster_id, 'lightgray')
                
                fig_umap.add_trace(go.Scatter3d(
                    x=df_cluster['x'], y=df_cluster['y'], z=df_cluster['z'],
                    mode='markers',
                    marker=dict(size=10, color=color, symbol='diamond', line=dict(color='gray', width=1), opacity=0.4),
                    name=f'Cluster {cluster_id}',
                    text=df_cluster['label'],
                    hovertemplate="<b>%{text}</b><br>Cluster: %{customdata}<br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<br>UMAP3: %{z:.3f}<extra></extra>",
                    customdata=df_cluster['cluster_id']
                ))
    else:
        # Humanas USADAS
        df_human_used = df_umap[df_umap['tipo'] == 'humana_usada']
        if len(df_human_used) > 0:
            fig_umap.add_trace(go.Scatter3d(
                x=df_human_used['x'], y=df_human_used['y'], z=df_human_used['z'],
                mode='markers',
                marker=dict(size=12, color='darkred', symbol='diamond', line=dict(color='black', width=2)),
                name='Humanas (USADAS)',
                text=df_human_used['label'],
                hovertemplate="<b>%{text} - USADA</b><br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<br>UMAP3: %{z:.3f}<extra></extra>"
            ))
        
        # Humanas NAO USADAS
        df_human_not_used = df_umap[df_umap['tipo'] == 'humana_nao_usada']
        if len(df_human_not_used) > 0:
            fig_umap.add_trace(go.Scatter3d(
                x=df_human_not_used['x'], y=df_human_not_used['y'], z=df_human_not_used['z'],
                mode='markers',
                marker=dict(size=10, color='orange', symbol='diamond', line=dict(color='darkorange', width=1), opacity=0.5),
                name='Humanas (NAO USADAS)',
                text=df_human_not_used['label'],
                hovertemplate="<b>%{text} - NAO USADA</b><br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<br>UMAP3: %{z:.3f}<extra></extra>"
            ))
    
    # Ideias geradas
    df_generated = df_umap[df_umap['tipo'] == 'gerada']
    
    if len(df_generated) > 0:
        # IMPORTANTE: Quando EMA está ativo, usar distâncias suavizadas do summary
        # para ser consistente com o gráfico de convergência
        use_ema = summary.get('config', {}).get('use_ema', False)
        
        if use_ema:
            # Com EMA: usar min_distance suavizado de cada iteração
            print(f"[DEBUG UMAP] EMA ativo - usando distâncias suavizadas para consistência com gráfico")
            iter_summary = {item['iteration']: item['min_distance'] for item in summary['iterations']}
            
            # Encontrar melhor iteração (menor min_distance suavizado)
            best_iter = min(iter_summary, key=iter_summary.get)
            best_dist_smoothed = iter_summary[best_iter]
            
            # Encontrar a melhor ideia dessa iteração
            best_iter_ideas = df_generated[df_generated['iteracao'] == best_iter]
            best_idx = best_iter_ideas['distancia'].idxmin()
            best_row = df_generated.loc[best_idx]
            
            print(f"[DEBUG UMAP] Melhor iteração (EMA): Iter {best_iter}, min_dist_suavizado={best_dist_smoothed:.6f}")
            print(f"[DEBUG UMAP] Melhor ideia global: Iter {best_row['iteracao']}, dist={best_row['distancia']:.6f}")
        else:
            # Sem EMA: usar menor distância individual RAW
            best_idx = df_generated['distancia'].idxmin()
            best_row = df_generated.loc[best_idx]
            print(f"[DEBUG UMAP] Melhor ideia global (RAW): Iter {best_row['iteracao']}, dist={best_row['distancia']:.6f}")
        
        df_gen_normal = df_generated.drop(best_idx)
        
        # Ideias normais
        if len(df_gen_normal) > 0:
            fig_umap.add_trace(go.Scatter3d(
                x=df_gen_normal['x'], y=df_gen_normal['y'], z=df_gen_normal['z'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=df_gen_normal['iteracao'],
                    colorscale='Viridis',
                    colorbar=dict(title="Iteracao", x=1.15),
                    symbol='circle'
                ),
                name='Ideias Geradas',
                text=df_gen_normal['label'],
                customdata=df_gen_normal['distancia'],
                hovertemplate="<b>%{text}</b><br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<br>UMAP3: %{z:.3f}<br>Dist: %{customdata:.4f}<extra></extra>"
            ))
        
        # Melhor ideia
        fig_umap.add_trace(go.Scatter3d(
            x=[best_row['x']], y=[best_row['y']], z=[best_row['z']],
            mode='markers',
            marker=dict(size=20, color='gold', symbol='diamond', line=dict(color='orange', width=4)),
            name='⭐ Melhor Ideia',
            text=[best_row['label']],
            customdata=[best_row['distancia']],
            hovertemplate="<b>⭐ %{text}</b><br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<br>UMAP3: %{z:.3f}<br>Dist: %{customdata:.4f}<extra></extra>"
        ))
        
        # Trajetorias (se houver mais de 1 iteracao)
        if len(iterations) > 1:
            # Trajetoria dos centroides
            centroids_x, centroids_y, centroids_z = [], [], []
            for iter_num in range(1, len(iterations) + 1):
                df_iter = df_generated[df_generated['iteracao'] == iter_num]
                if len(df_iter) > 0:
                    centroids_x.append(df_iter['x'].mean())
                    centroids_y.append(df_iter['y'].mean())
                    centroids_z.append(df_iter['z'].mean())
            
            if len(centroids_x) > 1:
                fig_umap.add_trace(go.Scatter3d(
                    x=centroids_x, y=centroids_y, z=centroids_z,
                    mode='lines+markers+text',
                    line=dict(color='cyan', width=4, dash='solid'),
                    marker=dict(size=8, color='cyan', symbol='circle', line=dict(color='darkblue', width=2)),
                    text=[f"{i}" for i in range(1, len(centroids_x) + 1)],
                    textposition='top center',
                    textfont=dict(size=14, color='darkblue', family='Arial Black'),
                    name='Trajetoria (Centroides)',
                    hovertemplate="<b>Centroide Iter %{text}</b><br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<br>UMAP3: %{z:.3f}<extra></extra>"
                ))
            
            # Trajetoria das melhores ideias
            best_x, best_y, best_z = [], [], []
            best_iters = []
            best_dists = []
            for iter_num in range(1, len(iterations) + 1):
                df_iter = df_generated[df_generated['iteracao'] == iter_num]
                if len(df_iter) > 0:
                    best_iter_idx = df_iter['distancia'].idxmin()
                    best_iter_row = df_iter.loc[best_iter_idx]
                    best_x.append(best_iter_row['x'])
                    best_y.append(best_iter_row['y'])
                    best_z.append(best_iter_row['z'])
                    best_iters.append(iter_num)
                    best_dists.append(best_iter_row['distancia'])
            
            if len(best_x) > 1:
                fig_umap.add_trace(go.Scatter3d(
                    x=best_x, y=best_y, z=best_z,
                    mode='lines+markers+text',
                    line=dict(color='magenta', width=3, dash='solid'),
                    marker=dict(size=8, color='magenta', symbol='diamond', line=dict(color='purple', width=2)),
                    text=[f"{i}" for i in best_iters],  # Usar número real da iteração
                    textposition='bottom center',
                    textfont=dict(size=14, color='purple', family='Arial Black'),
                    name='Trajetoria (Melhores)',
                    customdata=best_dists,
                    hovertemplate="<b>Melhor Iter %{text}</b><br>Dist: %{customdata:.4f}<br>UMAP1: %{x:.3f}<br>UMAP2: %{y:.3f}<br>UMAP3: %{z:.3f}<extra></extra>"
                ))
    
    # Layout
    fig_umap.update_layout(
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
            bgcolor='rgba(240,240,240,0.9)',
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
        width=1200,
        hovermode='closest'
    )
    
    # Salvar
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_umap.write_html(str(output_dir / "umap_3d.html"))
    print(f"[OK] Grafico UMAP 3D salvo: {output_dir / 'umap_3d.html'}")
    
    try:
        # PNG para UMAP é mais complexo, tentar salvar
        fig_umap.write_image(str(output_dir / "umap_3d.png"), width=1200, height=700, scale=2)
        print(f"[OK] Imagem PNG salva: {output_dir / 'umap_3d.png'}")
    except Exception as e:
        print(f"[AVISO] PNG UMAP nao gerado (kaleido pode ter limitacoes com graficos 3D): {e}")


def generate_config_summary(exp_data: dict, output_dir: Path) -> None:
    """Gera arquivo de resumo de configuracoes."""
    summary = exp_data["summary"]
    config = summary.get("config", {})
    
    config_summary = {
        "configuracao_principal": {
            "model": config.get("model"),
            "embedder": config.get("embedder"),
            "max_iterations": config.get("max_iterations"),
            "patience": config.get("patience"),
            "delta_threshold": config.get("delta_threshold"),
            "num_ideas_per_iter": config.get("num_ideas_per_iter"),
            "num_human_ideas": config.get("num_human_ideas"),
        },
        "clustering": summary.get("clustering"),
        "resultados": {
            "total_iterations": summary.get("total_iterations"),
            "converged": summary.get("converged"),
            "convergence_reason": summary.get("convergence_reason"),
            "best_avg_distance": summary.get("best_avg_distance"),
            "best_min_distance": summary.get("best_min_distance"),
        }
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = output_dir / "configuracao.json"
    
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_summary, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Configuracao salva: {config_file}")


def find_all_experiments(base_dir: Path = Path("exp_refinement")) -> list[Path]:
    """Encontra todos os experimentos disponiveis."""
    if not base_dir.exists():
        return []
    
    experiments = []
    for exp_subdir in base_dir.iterdir():
        if exp_subdir.is_dir() and (exp_subdir / "summary.json").exists():
            experiments.append(exp_subdir)
    
    # Ordenar por nome (timestamp) - mais recente primeiro
    experiments.sort(reverse=True)
    return experiments


def find_latest_experiment(base_dir: Path = Path("exp_refinement")) -> Optional[Path]:
    """Encontra o experimento mais recente."""
    experiments = find_all_experiments(base_dir)
    return experiments[0] if experiments else None


def process_experiment(exp_path: Path) -> bool:
    """Processa um experimento e gera seus graficos.
    
    Args:
        exp_path: Caminho para o diretorio do experimento ou para plots/
    
    Returns:
        True se sucesso, False se falhou
    """
    # Se o caminho apontar para plots/, subir um nivel
    if exp_path.name == "plots":
        exp_path = exp_path.parent
    
    if not exp_path.exists():
        print(f"[ERRO] Diretorio nao existe: {exp_path}")
        return False
    
    # Verificar se e um experimento valido (tem summary.json)
    if not (exp_path / "summary.json").exists():
        print(f"[ERRO] Nao e um experimento valido (sem summary.json): {exp_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"[INFO] Processando: {exp_path.name}")
    print(f"{'='*60}")
    
    try:
        exp_data = load_experiment(exp_path)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar experimento: {e}")
        return False
    
    print(f"[INFO] Iteracoes encontradas: {exp_data['summary']['total_iterations']}")
    
    # Diretorio de saida
    plots_dir = exp_path / "plots"
    
    print(f"[INFO] Gerando graficos em: {plots_dir}")
    
    # Gerar graficos
    try:
        generate_convergence_plot(exp_data, plots_dir)
        generate_trajectory_plot(exp_data, plots_dir)
        generate_umap_plot(exp_data, plots_dir)
        generate_config_summary(exp_data, plots_dir)
        
        print(f"[OK] Graficos gerados com sucesso para: {exp_path.name}")
        return True
    except Exception as e:
        print(f"[ERRO] Falha ao gerar graficos para {exp_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Funcao principal.
    
    Uso:
        python generate_plots_from_saved.py                    # Processa apenas o mais recente
        python generate_plots_from_saved.py --all             # Processa TODOS os experimentos
        python generate_plots_from_saved.py <caminho>         # Processa experimento especifico
    """
    script_dir = Path(__file__).parent
    base_dir = script_dir / "exp_refinement"
    
    # Verificar argumentos
    if len(sys.argv) > 1:
        arg1 = sys.argv[1].strip().lower()
        
        # Modo: processar todos
        if arg1 in ['--all', '-a', '--todos', '--all-experiments']:
            print("[INFO] Modo: Processar TODOS os experimentos\n")
            experiments = find_all_experiments(base_dir)
            
            if not experiments:
                print("[ERRO] Nenhum experimento encontrado em exp_refinement/")
                sys.exit(1)
            
            print(f"[INFO] Encontrados {len(experiments)} experimentos para processar\n")
            
            # Processar cada experimento
            success_count = 0
            fail_count = 0
            
            for i, exp_path in enumerate(experiments, 1):
                print(f"\n{'='*60}")
                print(f"[{i}/{len(experiments)}] Processando: {exp_path.name}")
                print(f"{'='*60}")
                
                if process_experiment(exp_path):
                    success_count += 1
                else:
                    fail_count += 1
                
                print()  # Linha em branco entre experimentos
            
            # Resumo final
            print(f"\n{'='*60}")
            print(f"RESUMO FINAL")
            print(f"{'='*60}")
            print(f"[OK] Sucesso: {success_count}/{len(experiments)} experimentos")
            if fail_count > 0:
                print(f"[ERRO] Falhas: {fail_count}/{len(experiments)} experimentos")
            print()
            
            return
        
        # Modo: caminho especifico
        else:
            exp_path = Path(sys.argv[1])
            if not exp_path.is_absolute():
                exp_path = script_dir / exp_path
            
            print(f"[INFO] Modo: Processar experimento especifico: {exp_path.name}\n")
            
            if not process_experiment(exp_path):
                sys.exit(1)
            
            return
    
    # Modo padrao: apenas o mais recente
    else:
        print("[INFO] Modo: Processar apenas o experimento mais recente\n")
        exp_path = find_latest_experiment(base_dir)
        
        if not exp_path:
            print("[ERRO] Nenhum experimento encontrado em exp_refinement/")
            print("[INFO] Uso:")
            print("  python generate_plots_from_saved.py                    # Mais recente")
            print("  python generate_plots_from_saved.py --all             # Todos")
            print("  python generate_plots_from_saved.py <caminho>        # Especifico")
            sys.exit(1)
        
        print(f"[INFO] Usando experimento mais recente: {exp_path.name}\n")
        
        if not process_experiment(exp_path):
            sys.exit(1)


if __name__ == "__main__":
    main()

