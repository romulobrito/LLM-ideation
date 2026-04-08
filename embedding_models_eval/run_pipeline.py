#!/usr/bin/env python3
"""
Entry point para executar o pipeline de avaliacao de embeddings.

Uso:
    python run_pipeline.py
    python run_pipeline.py --config configs/default.yaml
    python run_pipeline.py --config configs/default.yaml --output-dir results_custom
    python run_pipeline.py --config configs/default.yaml --verbose
    python run_pipeline.py --config configs/default.yaml --models minilm openai_small
"""

import argparse
import sys
from pathlib import Path

from pipeline import load_config, run_experiment


def main():
    """Funcao principal do entry point."""
    parser = argparse.ArgumentParser(
        description="Pipeline de Avaliacao de Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Usar config padrao
  python run_pipeline.py
  
  # Especificar config customizado
  python run_pipeline.py --config configs/custom.yaml
  
  # Especificar diretorio de saida
  python run_pipeline.py --output-dir results_experimento1
  
  # Processar apenas modelos especificos
  python run_pipeline.py --models minilm openai_small
  
  # Modo silencioso (sem output detalhado)
  python run_pipeline.py --quiet
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Caminho para arquivo de configuracao YAML (padrao: configs/default.yaml)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Diretorio para salvar resultados (sobrescreve config.output.results_dir)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Lista de modelos para processar (se nao especificado, processa todos)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Mostrar progresso detalhado (padrao: True)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Modo silencioso (sem output detalhado)"
    )
    
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Parar na primeira erro (padrao: continua mesmo com erros)"
    )
    
    args = parser.parse_args()
    
    # Ajustar verbose baseado em --quiet
    verbose = not args.quiet if args.quiet else args.verbose
    
    try:
        # Carregar configuracao
        if verbose:
            print("=" * 70)
            print("CARREGANDO CONFIGURACAO")
            print("=" * 70)
            print(f"Arquivo: {args.config}")
            print()
        
        config = load_config(args.config)
        
        # Ajustar output dir se especificado
        if args.output_dir:
            config["output"]["results_dir"] = args.output_dir
            if verbose:
                print(f"Diretorio de saida: {args.output_dir}")
                print()
        
        # Filtrar modelos se especificado
        if args.models:
            modelos_originais = config["models"]
            modelos_filtrados = [
                m for m in modelos_originais
                if m["name"] in args.models
            ]
            
            if not modelos_filtrados:
                print(f"ERRO: Nenhum dos modelos especificados encontrado: {args.models}")
                print(f"Modelos disponiveis: {[m['name'] for m in modelos_originais]}")
                sys.exit(1)
            
            modelos_nao_encontrados = set(args.models) - {m["name"] for m in modelos_filtrados}
            if modelos_nao_encontrados:
                print(f"AVISO: Modelos nao encontrados: {modelos_nao_encontrados}")
            
            config["models"] = modelos_filtrados
            
            if verbose:
                print(f"Modelos a processar: {[m['name'] for m in modelos_filtrados]}")
                print()
        
        # Executar pipeline
        resultados = run_experiment(
            config,
            verbose=verbose,
            continue_on_error=not args.fail_fast
        )
        
        # Resumo final
        if verbose:
            print()
            print("=" * 70)
            print("RESUMO FINAL")
            print("=" * 70)
            print(f"Modelos processados: {len(resultados['per_prompt_metrics'])}")
            print()
            print("Artefatos salvos:")
            if resultados["artifacts"]["summary_csv"]:
                print(f"  ✓ CSV: {resultados['artifacts']['summary_csv']}")
            if resultados["artifacts"]["summary_excel"]:
                print(f"  ✓ Excel: {resultados['artifacts']['summary_excel']}")
            if resultados["artifacts"]["summary_json"]:
                print(f"  ✓ JSON macro: {resultados['artifacts']['summary_json']}")
            if resultados["artifacts"]["detailed"]:
                print(f"  ✓ Parquet: {len(resultados['artifacts']['detailed'])} arquivos")
            if resultados["artifacts"]["detailed_json"]:
                n_json = len(resultados["artifacts"]["detailed_json"])
                print(f"  ✓ JSON detalhado: {n_json} arquivos")
            ext = resultados.get("pipeline_extras_report") or {}
            if ext.get("per_prompt_metrics_paths"):
                n_p = len(ext["per_prompt_metrics_paths"])
                print(f"  Extr opcional: {n_p} CSV per-prompt (metricas por prompt)")
            if ext.get("tfidf_output_dir"):
                print(f"  Extr opcional TF-IDF: {ext['tfidf_output_dir']}")
            if ext.get("visualizations_output_dir"):
                print(f"  Extr opcional visualizacoes: {ext['visualizations_output_dir']}")
            if ext.get("skipped_visualizations"):
                print(f"  AVISO visualizacoes: {ext['skipped_visualizations']}")
            if ext.get("error"):
                print(f"  AVISO pipeline_extras: {ext['error']}")
            print()
        
        # Mostrar tabela comparativa se nao estiver em modo quiet
        if verbose and not resultados["macro_metrics"].empty:
            print("TABELA COMPARATIVA (primeiras linhas):")
            print("-" * 70)
            print(resultados["macro_metrics"].head().to_string(index=False))
            print()
        
        print("=" * 70)
        print("PIPELINE EXECUTADO COM SUCESSO!")
        print("=" * 70)
        
        return 0
    
    except FileNotFoundError as e:
        print(f"ERRO: Arquivo nao encontrado: {e}")
        sys.exit(1)
    
    except ValueError as e:
        print(f"ERRO: {e}")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nPipeline interrompido pelo usuario.")
        sys.exit(130)
    
    except Exception as e:
        print(f"ERRO CRITICO: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
