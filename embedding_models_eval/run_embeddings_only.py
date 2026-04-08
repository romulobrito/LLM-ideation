"""
Processa dados reais ate gerar embeddings, sem ranking ou metricas.

Este script:
1. Carrega dados reais (saida_final.json)
2. Processa textos
3. Gera embeddings para todos os textos
4. Para por ai (sem ranking, sem metricas)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback: tqdm vira range
    def tqdm(iterable, desc=""):
        print(f"{desc}...")
        return iterable

def load_real_data(config_path="configs/default.yaml"):
    """Carrega dados reais usando o loader do pipeline."""
    print("=" * 70)
    print("1. CARREGANDO DADOS REAIS")
    print("=" * 70)
    print()
    
    try:
        from embedding_models_eval.pipeline.config_loader import load_config
        from embedding_models_eval.data import get_loader
        
        # Carrega configuracao (com substituicao de variaveis de ambiente)
        config = load_config(config_path)
        
        dataset_config = config.get("dataset", {})
        dataset_path = dataset_config.get("path", "saida_final.json")
        text_col = dataset_config.get("text_col", "extracted_idea_250")
        
        print(f"   Dataset: {dataset_path}")
        print(f"   Coluna de texto: {text_col}")
        print()
        
        # Carrega dados
        loader = get_loader("json", dataset_config)
        df = loader.load(dataset_path)
        
        print(f"    Dados carregados!")
        print(f"   Total de linhas: {len(df):,}")
        print(f"   Colunas: {list(df.columns)}")
        print()
        
        # Verifica se tem a coluna de texto
        if text_col not in df.columns:
            raise ValueError(f"Coluna '{text_col}' nao encontrada no dataset")
        
        # Remove textos vazios
        df = df[df[text_col].notna() & (df[text_col].str.strip() != "")]
        print(f"   Textos validos: {len(df):,}")
        print()
        
        return df, text_col, config
        
    except Exception as e:
        print(f"✗ Erro ao carregar dados: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def generate_embeddings(df, text_col, model_config, verbose=True):
    """Gera embeddings para todos os textos usando um modelo."""
    from embedding_models_eval.embeddings import get_provider
    
    model_name = model_config.get("name", "unknown")
    provider_name = model_config.get("provider")
    provider_config = model_config.get("config", {})
    
    if verbose:
        print(f"   Modelo: {model_name}")
        print(f"   Provider: {provider_name}")
        print(f"   Gerando embeddings...")
    
    # Cria provider
    provider = get_provider(provider_name, provider_config)
    
    # Pega textos
    textos = df[text_col].tolist()
    
    # Gera embeddings com barra de progresso
    if verbose:
        embeddings_list = []
        batch_size = provider_config.get("batch_size", 32)
        
        for i in tqdm(range(0, len(textos), batch_size), desc=f"   Processando {model_name}"):
            batch = textos[i:i + batch_size]
            batch_embeddings = provider.embed(batch)
            embeddings_list.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings_list)
    else:
        embeddings = provider.embed(textos)
    
    if verbose:
        print(f"    Embeddings gerados!")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Dimensao: {embeddings.shape[1]}D")
        print()
    
    return embeddings


def process_single_model(df, text_col, model_config, save_dir=None):
    """Processa um unico modelo."""
    model_name = model_config.get("name", "unknown")
    
    print("=" * 70)
    print(f"2. PROCESSANDO MODELO: {model_name}")
    print("=" * 70)
    print()
    
    try:
        # Gera embeddings
        embeddings = generate_embeddings(df, text_col, model_config, verbose=True)
        
        # Cria DataFrame com embeddings
        # Adiciona embeddings como colunas (ou pode salvar separado)
        df_with_embeddings = df.copy()
        
        # Opcao 1: Salvar embeddings em arquivo separado (mais eficiente)
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Salva embeddings como numpy array
            embeddings_path = save_dir / f"{model_name}_embeddings.npy"
            np.save(embeddings_path, embeddings)
            print(f"    Embeddings salvos: {embeddings_path}")
            
            # Salva metadados
            metadata_path = save_dir / f"{model_name}_metadata.parquet"
            df_with_embeddings.to_parquet(metadata_path)
            print(f"    Metadados salvos: {metadata_path}")
        
        print()
        return embeddings, df_with_embeddings
        
    except Exception as e:
        print(f"✗ Erro ao processar modelo {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Processa dados reais ate gerar embeddings."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Processa dados reais ate gerar embeddings (sem ranking/metricas)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Caminho para arquivo de configuracao YAML"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nome do modelo a processar (se nao especificado, processa todos)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/embeddings_only",
        help="Diretorio para salvar embeddings"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Modo verboso"
    )
    
    args = parser.parse_args()
    
    print()
    print("=" * 70)
    print("PROCESSAMENTO: DADOS REAIS ATE EMBEDDINGS")
    print("=" * 70)
    print()
    print("Este script processa os dados reais do pipeline")
    print("ate gerar embeddings, sem executar ranking ou metricas.")
    print()
    
    #  Carrega dados
    df, text_col, config = load_real_data(args.config)
    if df is None:
        return 1
    
    #  Pega modelos da configuracao
    models_config = config.get("models", [])
    
    if args.model:
        # Filtra apenas o modelo especificado
        models_config = [m for m in models_config if m.get("name") == args.model]
        if not models_config:
            print(f"✗ Modelo '{args.model}' nao encontrado na configuracao")
            return 1
    
    print("=" * 70)
    print(f"3. PROCESSANDO {len(models_config)} MODELO(S)")
    print("=" * 70)
    print()
    
    resultados = {}
    
    #  Processa cada modelo
    for i, model_config in enumerate(models_config, 1):
        model_name = model_config.get("name", f"model_{i}")
        
        print(f"[{i}/{len(models_config)}] Processando {model_name}...")
        print()
        
        embeddings, df_result = process_single_model(
            df, text_col, model_config, save_dir=args.output_dir
        )
        
        if embeddings is not None:
            resultados[model_name] = {
                "embeddings": embeddings,
                "df": df_result,
                "shape": embeddings.shape
            }
            print(f" {model_name} processado com sucesso!")
        else:
            print(f"✗ {model_name} falhou")
        
        print()
    
    #  Resumo
    print("=" * 70)
    print("RESUMO")
    print("=" * 70)
    print()
    print(f"Modelos processados: {len(resultados)}/{len(models_config)}")
    print()
    
    for model_name, dados in resultados.items():
        print(f"  {model_name}:")
        print(f"    Shape: {dados['shape']}")
        print(f"    Total de textos: {dados['shape'][0]:,}")
        print(f"    Dimensao: {dados['shape'][1]}D")
        print()
    
    if args.output_dir:
        print(f"Arquivos salvos em: {args.output_dir}")
        print()
    
    print("=" * 70)
    print("CONCLUIDO!")
    print("=" * 70)
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
