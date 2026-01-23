# Guia de Uso: run_embeddings_only.py

## Descricao

O script `run_embeddings_only.py` processa dados reais do pipeline ate gerar embeddings, sem executar ranking ou metricas. Ele carrega os dados de `saida_final.json`, processa os textos e gera embeddings para todos os modelos configurados.

## Requisitos

### Dependencias Python

Instale as dependencias necessarias:

```bash
pip install -r requirements.txt
```

Principais dependencias:
- pandas
- numpy
- sentence-transformers
- openai (opcional, apenas se usar modelos OpenAI)
- pyyaml
- python-dotenv
- tqdm

### Arquivos Necessarios

- `saida_final.json` - Dataset com os dados
- `configs/default.yaml` - Configuracao dos modelos
- Modulos `embeddings/` e `data/` - Codigo do sistema

### Configuracao de API Keys (Opcional)

Se for usar modelos OpenAI, configure a API key:

1. Crie um arquivo `.env` no diretorio do projeto ou em `experimento_convergencia_visualizacao_metricas/.env`
2. Adicione:
   ```
   OPENAI_API_KEY=sk-sua-chave-aqui
   ```

O script carregara automaticamente a chave do arquivo `.env`.

## Uso Basico

### Processar Todos os Modelos

Para processar todos os modelos configurados no YAML:

```bash
python run_embeddings_only.py
```

O script ira:
1. Carregar dados de `saida_final.json`
2. Processar cada modelo configurado
3. Gerar embeddings para todos os textos
4. Salvar resultados em `results/embeddings_only/`

### Processar Apenas Um Modelo

Para processar apenas um modelo especifico:

```bash
python run_embeddings_only.py --model minilm
```

Substitua `minilm` pelo nome do modelo desejado (ex: `mpnet_base`, `openai_small`).

### Especificar Diretorio de Saida

Para salvar os resultados em um diretorio diferente:

```bash
python run_embeddings_only.py --output-dir meus_embeddings/
```

### Usar Configuracao Customizada

Para usar um arquivo de configuracao diferente:

```bash
python run_embeddings_only.py --config minha_config.yaml
```

## Opcoes de Linha de Comando

```
--config PATH        Caminho para arquivo de configuracao YAML (padrao: configs/default.yaml)
--model NOME         Nome do modelo a processar (se nao especificado, processa todos)
--output-dir PATH    Diretorio para salvar embeddings (padrao: results/embeddings_only)
--verbose            Modo verboso (mais informacoes)
```

## Exemplos de Uso

### Exemplo 1: Processar Todos os Modelos

```bash
python run_embeddings_only.py
```

Saida esperada:
- Processa todos os modelos configurados
- Gera embeddings para cada modelo
- Salva arquivos em `results/embeddings_only/`

### Exemplo 2: Processar Apenas Modelos Sentence-Transformers

```bash
# Processar minilm
python run_embeddings_only.py --model minilm

# Processar mpnet_base
python run_embeddings_only.py --model mpnet_base
```

### Exemplo 3: Processar com Saida Customizada

```bash
python run_embeddings_only.py --output-dir embeddings_2024/
```

### Exemplo 4: Processar com Configuracao Customizada

```bash
python run_embeddings_only.py --config configs/minha_config.yaml
```

## Estrutura de Saida

Para cada modelo processado, o script gera dois arquivos:

### 1. Arquivo de Embeddings (.npy)

Nome: `{modelo}_embeddings.npy`

Formato: Array numpy com shape `(n_textos, dimensao)`

Exemplo:
- `minilm_embeddings.npy` - Shape: (354, 384)
- `mpnet_base_embeddings.npy` - Shape: (354, 768)
- `openai_large_embeddings.npy` - Shape: (354, 3072)

Como carregar:
```python
import numpy as np
embeddings = np.load("results/embeddings_only/minilm_embeddings.npy")
print(embeddings.shape)  # (354, 384)
```

### 2. Arquivo de Metadados (.parquet)

Nome: `{modelo}_metadata.parquet`

Formato: DataFrame pandas com todas as colunas do dataset original

Colunas incluem:
- `extracted_idea_250` - Texto embedado
- `likes`, `comments` - Engajamento
- `rank_in_prompt` - Ranking gold
- `contest_number`, `context_prompt_url` - Chaves de agrupamento
- E todas as outras colunas do dataset

Como carregar:
```python
import pandas as pd
df = pd.read_parquet("results/embeddings_only/minilm_metadata.parquet")
print(df.columns)
```

## Modelos Configurados

O script processa os modelos definidos em `configs/default.yaml`:

### Sentence-Transformers (Locais, Gratuitos)

- `minilm` - all-MiniLM-L6-v2 (384D, rapido)
- `minilm_l12` - all-MiniLM-L12-v2 (384D, melhor qualidade)
- `mpnet_base` - all-mpnet-base-v2 (768D, alta qualidade)
- `multilingual` - paraphrase-multilingual-MiniLM-L12-v2 (384D)
- `paraphrase_minilm` - paraphrase-MiniLM-L6-v2 (384D)
- `multi_qa_minilm` - multi-qa-MiniLM-L6-cos-v1 (384D)

### OpenAI (API, Requer API Key)

- `openai_small` - text-embedding-3-small (1536D)
- `openai_large` - text-embedding-3-large (3072D)

## Interpretacao dos Resultados

### Shape dos Embeddings

O shape `(n_textos, dimensao)` indica:
- Primeira dimensao: numero de textos processados
- Segunda dimensao: dimensao do vetor de embedding

Exemplos:
- `(354, 384)` - 354 textos, cada um com vetor de 384 dimensoes
- `(354, 768)` - 354 textos, cada um com vetor de 768 dimensoes
- `(354, 3072)` - 354 textos, cada um com vetor de 3072 dimensoes

### Tempo de Processamento

Tempos tipicos (para 354 textos):
- Modelos 384D: 7-30 segundos
- Modelos 768D: 3-4 minutos
- Modelos OpenAI: 4-5 segundos (depende da API)

### Uso dos Embeddings

Os embeddings gerados podem ser usados para:
- Analise de similaridade entre textos
- Clustering de textos
- Visualizacao (t-SNE, UMAP)
- Continuar com ranking e metricas depois
- Treinamento de modelos de ML

## Solucao de Problemas

### Erro: "No module named 'sentence_transformers'"

Solucao: Instale as dependencias:
```bash
pip install -r requirements.txt
```

### Erro: "OpenAI API key nao fornecida"

Solucao: Configure a API key no arquivo `.env`:
```
OPENAI_API_KEY=sk-sua-chave-aqui
```

Ou defina como variavel de ambiente:
```bash
export OPENAI_API_KEY=sk-sua-chave-aqui
```

### Erro: "FileNotFoundError: saida_final.json"

Solucao: Certifique-se de que o arquivo `saida_final.json` esta no diretorio correto, ou ajuste o caminho no `configs/default.yaml`.

### Erro: "ModuleNotFoundError: No module named 'pipeline'"

Solucao: O script precisa do modulo `pipeline` para carregar configuracao. Certifique-se de que o arquivo `pipeline/config_loader.py` existe.

### Modelos OpenAI Falhando

Se os modelos OpenAI estiverem falhando com erro de autenticacao:
1. Verifique se a API key esta correta no `.env`
2. Verifique se o arquivo `.env` esta no local correto
3. Verifique se a substituicao de variaveis esta funcionando (o erro mostrara `${OPENAI_API_KEY}` se nao estiver substituindo)

## Diferenca do Pipeline Completo

Este script processa apenas ate a geracao de embeddings. Ele nao executa:
- Ranking baseado em similaridade
- Calculo de metricas (MAP@k, etc.)
- Comparacao entre modelos
- Geracao de graficos

Para executar o pipeline completo (com ranking e metricas), use:
```bash
python run_pipeline.py
```

## Praticas Recomendadas

1. **Processe um modelo por vez primeiro**: Teste com `--model minilm` antes de processar todos
2. **Verifique os resultados**: Carregue os arquivos `.npy` e `.parquet` para verificar se estao corretos
3. **Monitore o espaco em disco**: Embeddings podem ocupar bastante espaco (especialmente modelos grandes)
4. **Use GPU se disponivel**: Configure `device: "cuda"` no YAML para modelos Sentence-Transformers
5. **Salve em locais diferentes**: Use `--output-dir` para organizar resultados de diferentes execucoes

## Estrutura de Diretorios Esperada

```
embedding_models_eval/
├── run_embeddings_only.py
├── requirements.txt
├── configs/
│   └── default.yaml
├── embeddings/
│   ├── __init__.py
│   ├── base.py
│   ├── sentence_transformers.py
│   ├── openai.py
│   └── semdist.py
├── data/
│   ├── __init__.py
│   ├── base.py
│   └── json_loader.py
├── pipeline/
│   └── config_loader.py
├── saida_final.json
└── results/
    └── embeddings_only/  (criado automaticamente)
        ├── minilm_embeddings.npy
        ├── minilm_metadata.parquet
        └── ...
```

## Referencias

- Documentacao de embeddings: `STATUS_EMBEDDINGS.md`
- Arquivos necessarios: `ARQUIVOS_NECESSARIOS.md`
- Testes: `test_embeddings_structure.py` e `test_embeddings_only.py`
