"""
Carregamento e validacao de configuracao YAML.

Responsabilidades:
- Carregar arquivo YAML
- Carregar variaveis de ambiente do .env
- Substituir variaveis de ambiente no config (${VAR})
- Validar estrutura basica
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_env_robust() -> None:
    """
    Carrega variaveis de ambiente do arquivo .env de varios locais comuns.
    
    Tenta carregar de:
    1. Diretorio atual (.env)
    2. Diretorio do projeto (experimento_convergencia_visualizacao_metricas/.env)
    3. Home do usuario
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        # python-dotenv nao instalado, usa apenas variaveis do sistema
        return
    
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent.parent / "experimento_convergencia_visualizacao_metricas" / ".env",
        Path.home() / "Documentos" / "MAI-DAI-USP" / "experimento_convergencia_visualizacao_metricas" / ".env",
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            return
    
    # Nenhum .env encontrado, continua sem erro


def substitute_env_vars(obj: Any) -> Any:
    """
    Substitui variaveis de ambiente no formato ${VAR} por seus valores.
    
    Recursivamente processa dicts, lists e strings.
    
    Args:
        obj: Objeto a processar (dict, list, str, ou outro tipo)
        
    Returns:
        Objeto com variaveis substituidas
    """
    if isinstance(obj, dict):
        return {k: substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Substitui ${VAR} por os.getenv("VAR")
        import re
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match):
            var_name = match.group(1)
            value = os.getenv(var_name)
            if value is None:
                # Se nao encontrado, retorna a string original
                return match.group(0)
            return value
        
        return re.sub(pattern, replace_var, obj)
    else:
        return obj


def validate_config(config: Dict) -> None:
    """
    Valida estrutura basica do config.
    
    Args:
        config: Dict com configuracao
        
    Raises:
        ValueError: Se estrutura invalida
    """
    required_sections = ["dataset", "models", "metrics", "ranking", "output"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Secao obrigatoria '{section}' nao encontrada no config")
    
    # Validar dataset
    if "path" not in config["dataset"]:
        raise ValueError("dataset.path e obrigatorio")
    
    # Validar models
    if not isinstance(config["models"], list) or len(config["models"]) == 0:
        raise ValueError("models deve ser uma lista nao vazia")
    
    for i, model in enumerate(config["models"]):
        if "name" not in model:
            raise ValueError(f"models[{i}].name e obrigatorio")
        if "provider" not in model:
            raise ValueError(f"models[{i}].provider e obrigatorio")
        if "config" not in model:
            raise ValueError(f"models[{i}].config e obrigatorio")
    
    # Validar metrics
    if "k_values" not in config["metrics"]:
        raise ValueError("metrics.k_values e obrigatorio")
    if "metric_names" not in config["metrics"]:
        raise ValueError("metrics.metric_names e obrigatorio")
    
    # Validar ranking
    if "group_cols" not in config["ranking"]:
        raise ValueError("ranking.group_cols e obrigatorio")
    if "rank_col" not in config["ranking"]:
        raise ValueError("ranking.rank_col e obrigatorio")
    
    # Validar output
    if "results_dir" not in config["output"]:
        raise ValueError("output.results_dir e obrigatorio")


def load_config(config_path: str) -> Dict:
    """
    Carrega configuracao YAML e prepara para uso.
    
    Processa:
    1. Carrega arquivo YAML
    2. Carrega variaveis de ambiente do .env
    3. Substitui variaveis de ambiente (${VAR})
    4. Valida estrutura
    
    Args:
        config_path: Caminho para arquivo YAML
        
    Returns:
        Dict com configuracao processada
        
    Raises:
        FileNotFoundError: Se arquivo nao existe
        yaml.YAMLError: Se YAML invalido
        ValueError: Se estrutura invalida
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Arquivo de configuracao nao encontrado: {config_path}")
    
    # Carrega .env primeiro
    load_env_robust()
    
    # Carrega YAML
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError("Arquivo YAML vazio ou invalido")
    
    # Substitui variaveis de ambiente
    config = substitute_env_vars(config)
    
    # Valida estrutura
    validate_config(config)
    
    return config
