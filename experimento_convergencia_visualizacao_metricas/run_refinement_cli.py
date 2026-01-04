#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI para executar refinement loop via YAML config.

Todos os parametros sao carregados do YAML, incluindo invitation e directive.

Uso:
    python run_refinement_cli.py config_refinement.yaml
    python run_refinement_cli.py config_refinement.yaml --dry-run
"""

from __future__ import annotations

import yaml
import argparse
import sys
import os
from pathlib import Path
from typing import Tuple, Optional

# Carregar variaveis de ambiente do .env (mesmo comportamento do Streamlit)
try:
    from dotenv import load_dotenv
    # Tentar carregar .env de varios locais
    env_paths = [
        Path(__file__).parent / ".env",
        Path.cwd() / ".env",
        Path.home() / "Documentos" / "MAI-DAI-USP" / "experimento_convergencia_visualizacao_metricas" / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            print(f"[CLI] Arquivo .env carregado de: {env_path}")
            break
    else:
        print("[CLI] Arquivo .env nao encontrado. Usando variaveis de ambiente do sistema.")
except ImportError:
    print("[CLI] python-dotenv nao instalado. Usando apenas variaveis de ambiente do sistema.")

from refinement_loop import RefinementConfig, RefinementLoop
from experiment_iterativo import load_references_from_fs


def load_yaml_config(config_path: Path) -> dict:
    """
    Carrega config YAML.
    
    Args:
        config_path: Caminho para arquivo YAML
    
    Returns:
        Dict com configuracao
    
    Raises:
        FileNotFoundError: Se arquivo nao existe
        yaml.YAMLError: Se YAML invalido
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo nao existe: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_human_ideas(config: dict) -> Tuple[list[str], Optional[list[str]]]:
    """
    Carrega ideias humanas conforme config.
    
    Args:
        config: Dict com configuracao (secao human_ideas)
    
    Returns:
        Tupla (human_ideas, all_human_ideas)
        - human_ideas: Ideias a usar no experimento
        - all_human_ideas: Todas as ideias (para clustering) ou None
    
    Raises:
        FileNotFoundError: Se caminho nao existe
    """
    human_cfg = config['human_ideas']
    path = Path(human_cfg['path']).expanduser()
    
    if not path.exists():
        raise FileNotFoundError(f"Caminho nao existe: {path}")
    
    # Carregar todas as ideias
    all_ideas = load_references_from_fs(str(path))
    
    if not all_ideas:
        raise ValueError(f"Nenhuma ideia encontrada em: {path}")
    
    # Aplicar limite se especificado
    limit = human_cfg.get('limit')
    if limit:
        ideas = all_ideas[:limit]
    else:
        ideas = all_ideas
    
    # Para clustering, retornar todas tambem
    all_human_ideas = all_ideas if config['clustering']['enabled'] else None
    
    return ideas, all_human_ideas


def create_config(config_dict: dict) -> RefinementConfig:
    """
    Cria RefinementConfig a partir do dict YAML.
    
    Args:
        config_dict: Dict com configuracao completa
    
    Returns:
        RefinementConfig pronto para uso
    
    Raises:
        KeyError: Se campo obrigatorio faltando
        ValueError: Se valor invalido
    """
    # Carregar ideias humanas
    human_ideas, all_human_ideas = load_human_ideas(config_dict)
    
    # Normalizar paths
    output_dir = Path(config_dict['output']['dir']).expanduser().resolve()
    
    # Extrair invitation e directive (obrigatorios)
    invitation = config_dict.get('invitation', '').strip()
    directive = config_dict.get('directive', '').strip()
    
    if not invitation:
        raise ValueError("invitation e obrigatorio no YAML")
    if not directive:
        raise ValueError("directive e obrigatorio no YAML")
    
    # Extrair secoes
    model_cfg = config_dict['model']
    embedder_cfg = config_dict['embedder']
    cluster_cfg = config_dict['clustering']
    north_cfg = config_dict['north_star']
    consol_cfg = config_dict['consolidation']
    gen_cfg = config_dict['generation']
    conv_cfg = config_dict['convergence']
    div_cfg = config_dict['divergence_stop']
    
    # Converter reasoning_effort (None ou string)
    # YAML null -> Python None, string "None" ou "null" -> None
    reasoning_effort = model_cfg.get('reasoning_effort')
    if reasoning_effort is None or (isinstance(reasoning_effort, str) and reasoning_effort.lower() in ("none", "null", "")):
        reasoning_effort = None
    
    return RefinementConfig(
        # Carregados do YAML
        invitation=invitation,
        directive=directive,
        human_ideas=human_ideas,
        all_human_ideas=all_human_ideas,
        model=model_cfg['name'],
        embedder_name=embedder_cfg['name'],
        device=embedder_cfg['device'],
        max_iterations=conv_cfg['max_iterations'],
        patience=conv_cfg['patience'],
        delta_threshold=conv_cfg['delta_threshold'],
        num_ideas_per_iter=gen_cfg['num_ideas_per_iter'],
        temperature=gen_cfg['temperature'],
        max_tokens=model_cfg['max_tokens'],
        reasoning_effort=reasoning_effort,
        output_dir=output_dir,
        use_north_star=north_cfg['enabled'],
        north_star_model=north_cfg['model'],
        use_clustering=cluster_cfg['enabled'],
        clustering_method=cluster_cfg['method'],
        n_clusters=cluster_cfg.get('n_clusters', 4),
        distance_threshold=cluster_cfg.get('distance_threshold', 0.3),
        selected_cluster_id=cluster_cfg.get('selected_cluster_id'),
        min_cluster_size=cluster_cfg.get('min_cluster_size', 5),
        optimize_metric=conv_cfg['optimize_metric'],
        enable_divergence_stop=div_cfg['enabled'],
        divergence_threshold=div_cfg['threshold'],
        max_consecutive_worsening=div_cfg['max_consecutive_worsening'],
        max_distance_from_start=div_cfg['max_distance_from_start'],
        enable_consolidation=consol_cfg['enabled'],
        consolidation_threshold=consol_cfg['threshold'],
        consolidation_max_group_size=consol_cfg['max_group_size'],
        consolidation_model=consol_cfg['model'],
        consolidation_temperature=consol_cfg['temperature'],
    )


def validate_config(config_dict: dict) -> list[str]:
    """
    Valida config YAML e retorna lista de erros.
    
    Args:
        config_dict: Dict com configuracao
    
    Returns:
        Lista de mensagens de erro (vazia se valido)
    """
    errors = []
    
    # Campos obrigatorios (top-level)
    if 'invitation' not in config_dict or not config_dict['invitation']:
        errors.append("invitation e obrigatorio")
    if 'directive' not in config_dict or not config_dict['directive']:
        errors.append("directive e obrigatorio")
    
    # Campos obrigatorios (secoes)
    required_sections = ['human_ideas', 'model', 'embedder', 'clustering', 
                        'north_star', 'consolidation', 'generation', 
                        'convergence', 'divergence_stop', 'output']
    for section in required_sections:
        if section not in config_dict:
            errors.append(f"Secao obrigatoria faltando: {section}")
    
    # Validar human_ideas
    if 'human_ideas' in config_dict:
        if 'path' not in config_dict['human_ideas']:
            errors.append("human_ideas.path e obrigatorio")
    
    # Validar ranges
    if 'model' in config_dict:
        temp = config_dict['model'].get('temperature', 1.0)
        if not (0.0 <= temp <= 2.0):
            errors.append("model.temperature deve estar entre 0.0 e 2.0")
    
    if 'generation' in config_dict:
        temp = config_dict['generation'].get('temperature', 1.0)
        if not (0.0 <= temp <= 2.0):
            errors.append("generation.temperature deve estar entre 0.0 e 2.0")
    
    if 'convergence' in config_dict:
        max_iter = config_dict['convergence'].get('max_iterations', 20)
        if not isinstance(max_iter, int) or max_iter < 1:
            errors.append("convergence.max_iterations deve ser inteiro >= 1")
    
    return errors


def main() -> int:
    """Funcao principal do CLI."""
    parser = argparse.ArgumentParser(
        description='Executar refinement loop via YAML config (todos parametros via YAML)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Executar experimento
  python run_refinement_cli.py config_refinement.yaml
  
  # Validar config sem executar
  python run_refinement_cli.py config_refinement.yaml --dry-run
        """
    )
    parser.add_argument(
        'config',
        type=Path,
        help='Caminho para arquivo YAML de configuracao'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validar config sem executar o loop'
    )
    
    args = parser.parse_args()
    
    # Carregar config
    print(f"[CLI] Carregando config: {args.config}")
    try:
        config_dict = load_yaml_config(args.config)
    except FileNotFoundError as e:
        print(f"ERRO: {e}")
        return 1
    except yaml.YAMLError as e:
        print(f"ERRO: YAML invalido: {e}")
        return 1
    except Exception as e:
        print(f"ERRO ao carregar YAML: {e}")
        return 1
    
    # Validar config
    print("[CLI] Validando configuracao...")
    errors = validate_config(config_dict)
    if errors:
        print("ERRO: Configuracao invalida:")
        for error in errors:
            print(f"  - {error}")
        return 1
    
    # Criar RefinementConfig
    print("[CLI] Criando configuracao...")
    try:
        config = create_config(config_dict)
    except FileNotFoundError as e:
        print(f"ERRO: {e}")
        return 1
    except ValueError as e:
        print(f"ERRO: {e}")
        return 1
    except KeyError as e:
        print(f"ERRO: Campo obrigatorio faltando: {e}")
        return 1
    except Exception as e:
        print(f"ERRO ao criar config: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Dry run: apenas validar
    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN: Config validado com sucesso!")
        print("="*60)
        print(f"  Invitation: {config.invitation[:60]}...")
        print(f"  Directive: {config.directive}")
        print(f"  Ideias humanas: {len(config.human_ideas)}")
        if config.all_human_ideas:
            print(f"  Todas ideias (clustering): {len(config.all_human_ideas)}")
        print(f"  Modelo: {config.model}")
        print(f"  Embedder: {config.embedder_name}")
        print(f"  Max iteracoes: {config.max_iterations}")
        print(f"  Clustering: {'Sim' if config.use_clustering else 'Nao'}")
        print(f"  North Star: {'Sim' if config.use_north_star else 'Nao'}")
        print(f"  Consolidacao: {'Sim' if config.enable_consolidation else 'Nao'}")
        print(f"  Output dir: {config.output_dir}")
        print("="*60)
        return 0
    
    # Executar loop
    print("\n" + "="*60)
    print("INICIANDO REFINEMENT LOOP")
    print("="*60)
    print(f"Invitation: {config.invitation[:60]}...")
    print(f"Directive: {config.directive}")
    print(f"Modelo: {config.model}")
    print(f"Embedder: {config.embedder_name}")
    print(f"Max iteracoes: {config.max_iterations}")
    print(f"Output dir: {config.output_dir}")
    print("="*60 + "\n")
    
    try:
        loop = RefinementLoop(config)
        results = loop.run()
        
        print("\n" + "="*60)
        print(f"CONCLUIDO! {len(results)} iteracoes executadas.")
        print(f"Resultados salvos em: {config.output_dir}")
        print("="*60)
        return 0
        
    except KeyboardInterrupt:
        print("\n[CLI] Interrompido pelo usuario (Ctrl+C)")
        return 130
    except Exception as e:
        print(f"\nERRO durante execucao: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

