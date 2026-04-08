"""
Pipeline Runner - Orquestracao completa de avaliacao de embeddings.

Executa o pipeline end-to-end:
1. Carrega dataset
2. Processa cada modelo configurado
3. Calcula metricas
4. Agrega resultados
5. Salva artefatos
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from embedding_models_eval.data import load_dataset
from embedding_models_eval.embeddings import get_provider
from embedding_models_eval.ranking import build_anchor_ranking
from embedding_models_eval.metrics import get_metric

from .optional_steps import (
    get_pipeline_extras,
    extras_has_enabled_work,
    validate_extras_for_visualizations,
    run_optional_pipeline_steps,
)


def build_comparison_table(resultados_por_modelo: Dict[str, Dict]) -> pd.DataFrame:
    """
    Constrói tabela comparativa macro com metricas de todos os modelos.
    
    Args:
        resultados_por_modelo: Dict {modelo_name: {ir_metrics, votes_metrics, df_scored}}
        
    Returns:
        DataFrame com uma linha por modelo e metricas nas colunas
    """
    rows = []
    
    for modelo_name, resultados in resultados_por_modelo.items():
        ir_macro = resultados.get("ir_metrics", {}).get("macro", {})
        votes_macro = resultados.get("votes_metrics", {}).get("macro", {})
        
        row = {"Modelo": modelo_name}
        
        # Adiciona metricas IR (MAP@k)
        for k in [1, 3, 5, 10]:
            map_key = f"MAP@{k}"
            if map_key in ir_macro:
                row[map_key] = ir_macro[map_key]
        
        # Adiciona metricas de votos
        for k in [1, 3, 5, 10]:
            mean_key = f"mean_votes@{k}_pred"
            norm_key = f"norm_mean_votes@{k}"
            if mean_key in votes_macro:
                row[mean_key] = votes_macro[mean_key]
            if norm_key in votes_macro:
                row[norm_key] = votes_macro[norm_key]
        
        rows.append(row)
    
    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows)


def save_artifacts(
    resultados_por_modelo: Dict[str, Dict],
    macro_metrics: pd.DataFrame,
    output_config: Dict
) -> Dict[str, Any]:
    """
    Salva todos os artefatos de saida.
    
    Args:
        resultados_por_modelo: Resultados por modelo
        macro_metrics: DataFrame com metricas macro
        output_config: Configuracao de saida do YAML
        
    Returns:
        Dict com paths: summary_csv, summary_excel, summary_json, detailed,
        detailed_json (mapa modelo -> path)
    """
    results_dir = Path(output_config.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    artifacts = {
        "summary_csv": None,
        "summary_excel": None,
        "summary_json": None,
        "detailed": {},
        "detailed_json": {},
    }
    
    # Salva tabela comparativa macro
    if not macro_metrics.empty and output_config.get("save_summary", True):
        # CSV
        csv_path = results_dir / "comparacao_modelos_macro.csv"
        macro_metrics.to_csv(csv_path, index=False)
        artifacts["summary_csv"] = str(csv_path)
        
        # JSON (lista de registros, ASCII)
        if output_config.get("save_summary_json", True):
            json_path = results_dir / "comparacao_modelos_macro.json"
            macro_metrics.to_json(
                json_path,
                orient="records",
                indent=2,
                force_ascii=True,
                date_format="iso",
            )
            artifacts["summary_json"] = str(json_path)
        
        # Excel (se disponivel e configurado)
        if output_config.get("save_summary_excel", True):
            try:
                excel_path = results_dir / "comparacao_modelos_macro.xlsx"
                macro_metrics.to_excel(excel_path, index=False, engine="openpyxl")
                artifacts["summary_excel"] = str(excel_path)
            except ImportError:
                # openpyxl nao instalado, continua sem erro
                pass
            except Exception as e:
                # Outro erro ao salvar Excel, continua sem erro
                pass
    
    # Salva DataFrames detalhados (por modelo)
    if output_config.get("save_detailed", True):
        # Observacao: JSON detalhado pode ser muito maior em disco que Parquet.
        # Para integracao via API costuma bastar comparacao_modelos_macro.json;
        # gere detalhe sob demanda ou defina save_detailed_json: false no YAML.
        save_json_detail = output_config.get("save_detailed_json", True)
        for modelo_name, resultados in resultados_por_modelo.items():
            df_scored = resultados.get("df_scored")
            if df_scored is not None and isinstance(df_scored, pd.DataFrame):
                parquet_path = results_dir / f"{modelo_name}_detalhado.parquet"
                df_scored.to_parquet(parquet_path, index=False)
                artifacts["detailed"][modelo_name] = str(parquet_path)
                if save_json_detail:
                    json_detail_path = results_dir / f"{modelo_name}_detalhado.json"
                    df_scored.to_json(
                        json_detail_path,
                        orient="records",
                        indent=2,
                        force_ascii=True,
                        date_format="iso",
                    )
                    artifacts["detailed_json"][modelo_name] = str(json_detail_path)
    
    return artifacts


def extract_per_prompt_metrics(resultados_por_modelo: Dict[str, Dict]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Extrai metricas por prompt de todos os modelos.
    
    Args:
        resultados_por_modelo: Resultados por modelo
        
    Returns:
        Dict {modelo_name: {metric_name: DataFrame}}
    """
    per_prompt = {}
    
    for modelo_name, resultados in resultados_por_modelo.items():
        per_prompt[modelo_name] = {}
        
        # IR metrics per prompt
        ir_per_prompt = resultados.get("ir_metrics", {}).get("per_prompt", pd.DataFrame())
        if not ir_per_prompt.empty:
            per_prompt[modelo_name]["ir"] = ir_per_prompt
        
        # Votes metrics per prompt
        votes_per_prompt = resultados.get("votes_metrics", {}).get("per_prompt", pd.DataFrame())
        if not votes_per_prompt.empty:
            per_prompt[modelo_name]["votes"] = votes_per_prompt
    
    return per_prompt


def run_experiment(
    config: Dict,
    verbose: bool = True,
    continue_on_error: bool = True
) -> Dict[str, Any]:
    """
    Executa pipeline completo de avaliacao de embeddings.
    
    Args:
        config: Configuracao carregada do YAML
        verbose: Se True, mostra progresso
        continue_on_error: Se True, continua processando outros modelos em caso de erro
        
    Returns:
        Dict com:
        - per_prompt_metrics: {modelo: {metrica: DataFrame}}
        - macro_metrics: DataFrame comparativo
        - artifacts: {summary_csv, summary_excel, summary_json, detailed,
          detailed_json: {modelo: path}}
        - pipeline_extras_report: relatorio das etapas opcionais (YAML pipeline_extras)
        
    Raises:
        ValueError: Se nenhum modelo foi processado com sucesso
    """
    if verbose:
        print("=" * 70)
        print("PIPELINE DE AVALIACAO DE EMBEDDINGS")
        print("=" * 70)
        print()

    extras_validation = validate_extras_for_visualizations(get_pipeline_extras(config))
    if extras_validation:
        raise ValueError(extras_validation)
    
    # 1. Carregar dataset
    if verbose:
        print("1. CARREGANDO DATASET")
        print("-" * 70)
    
    dataset_config = config["dataset"]
    df = load_dataset(
        dataset_config["path"],
        text_col=dataset_config.get("text_col", "extracted_idea_250"),
        truncate_content=dataset_config.get("truncate_content", 0)
    )
    
    if verbose:
        print(f"✓ Dataset carregado: {len(df)} linhas")
        prompts_count = df.groupby(config["ranking"]["group_cols"]).ngroups
        print(f"  Prompts unicos: {prompts_count}")
        print()
    
    # 2. Processar cada modelo
    if verbose:
        print("2. PROCESSANDO MODELOS")
        print("-" * 70)
        print()
    
    resultados_por_modelo = {}
    modelos_com_erro = []
    
    for i, model_config in enumerate(config["models"], 1):
        modelo_name = model_config["name"]
        
        if verbose:
            print(f"[{i}/{len(config['models'])}] Modelo: {modelo_name}")
            print("-" * 70)
        
        try:
            # 2.1. Criar provider
            if verbose:
                print("  Criando provider...")
            
            provider = get_provider(
                model_config["provider"],
                model_config["config"]
            )
            
            if verbose:
                print("  ✓ Provider criado")
            
            # 2.2. Calcular ranking
            if verbose:
                print("  Calculando ranking com ancora...")
            
            ranking_config = config["ranking"]
            df_scored = build_anchor_ranking(
                df,
                provider,
                text_col=dataset_config.get("text_col", "extracted_idea_250"),
                group_cols=ranking_config["group_cols"],
                rank_col=ranking_config["rank_col"],
                anchor_rank=ranking_config.get("anchor_rank", 1),
                show_progress=verbose
            )
            
            if verbose:
                print("  ✓ Ranking calculado")
            
            # 2.3. Calcular metricas
            metricas_resultados = {}
            
            for metric_name in config["metrics"]["metric_names"]:
                if verbose:
                    print(f"  Calculando metricas {metric_name}...")
                
                metric = get_metric(
                    metric_name,
                    k_values=config["metrics"]["k_values"]
                )
                resultados = metric.compute(df_scored)
                metricas_resultados[metric_name] = resultados
                
                if verbose:
                    print(f"  ✓ Metricas {metric_name} calculadas")
            
            # 2.4. Armazenar resultados
            resultados_por_modelo[modelo_name] = {
                "df_scored": df_scored,
                "ir_metrics": metricas_resultados.get("ir", {}),
                "votes_metrics": metricas_resultados.get("votes", {}),
            }
            
            if verbose:
                print(f"  ✓ Modelo {modelo_name} processado com sucesso!")
                print()
        
        except Exception as e:
            modelos_com_erro.append((modelo_name, str(e)))
            
            if continue_on_error:
                if verbose:
                    print(f"  ✗ ERRO ao processar {modelo_name}: {e}")
                    print("  Continuando com proximo modelo...")
                    print()
                continue
            else:
                raise
    
    # Verificar se algum modelo foi processado
    if not resultados_por_modelo:
        error_msg = "Nenhum modelo foi processado com sucesso."
        if modelos_com_erro:
            error_msg += f"\nErros encontrados: {modelos_com_erro}"
        raise ValueError(error_msg)
    
    # 3. Agregar resultados
    if verbose:
        print("3. AGREGANDO RESULTADOS")
        print("-" * 70)
        print()
    
    macro_metrics = build_comparison_table(resultados_por_modelo)
    
    if verbose and not macro_metrics.empty:
        print("TABELA COMPARATIVA (Macro Metrics)")
        print("-" * 70)
        print(macro_metrics.to_string(index=False))
        print()
    
    # 4. Salvar artefatos
    if verbose:
        print("4. SALVANDO ARTEFATOS")
        print("-" * 70)
        print()
    
    artifacts = save_artifacts(
        resultados_por_modelo,
        macro_metrics,
        config["output"]
    )
    
    if verbose:
        if artifacts["summary_csv"]:
            print(f"✓ Tabela comparativa CSV: {artifacts['summary_csv']}")
        if artifacts["summary_excel"]:
            print(f"✓ Tabela comparativa Excel: {artifacts['summary_excel']}")
        if artifacts["summary_json"]:
            print(f"✓ Tabela comparativa JSON: {artifacts['summary_json']}")
        if artifacts["detailed"]:
            print(f"✓ DataFrames detalhados: {len(artifacts['detailed'])} modelos")
        if artifacts["detailed_json"]:
            print(f"✓ Detalhado JSON: {len(artifacts['detailed_json'])} modelos")
        print()
    
    pipeline_extras_report: Dict[str, Any] = {}
    if extras_has_enabled_work(get_pipeline_extras(config)):
        try:
            pipeline_extras_report = run_optional_pipeline_steps(
                df=df,
                text_col=dataset_config.get("text_col", "extracted_idea_250"),
                config=config,
                resultados_por_modelo=resultados_por_modelo,
                verbose=verbose,
            )
        except Exception as e:
            if continue_on_error:
                if verbose:
                    print(f"AVISO pipeline_extras: {e}")
                pipeline_extras_report = {"error": str(e)}
            else:
                raise
    
    # 5. Extrair metricas por prompt
    per_prompt_metrics = extract_per_prompt_metrics(resultados_por_modelo)
    
    # Resumo final
    if verbose:
        print("=" * 70)
        print("PIPELINE CONCLUIDO")
        print("=" * 70)
        print(f"Modelos processados com sucesso: {len(resultados_por_modelo)}/{len(config['models'])}")
        if modelos_com_erro:
            print(f"Modelos com erro: {len(modelos_com_erro)}")
        print("=" * 70)
        print()
    
    return {
        "per_prompt_metrics": per_prompt_metrics,
        "macro_metrics": macro_metrics,
        "artifacts": artifacts,
        "pipeline_extras_report": pipeline_extras_report,
    }
