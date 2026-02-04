# Run Ranking Only: Ranking por Ancora

## Objetivo

Este script executa o pipeline **ate o ranking por ancora**: apos carregar o dataset e gerar embeddings, calcula o **ranking por similaridade com ancora** (top-1 por prompt) e salva os outputs pertinentes.

## O que e executado

1. **Carregamento de dados**: mesmo loader e config que `run_embeddings_only.py` (dataset, text_col, rank_in_prompt).
2. **Por cada modelo** configurado no YAML:
   - Cria o provider de embeddings.
   - Chama `build_anchor_ranking(df, provider, ...)` (modulo `ranking/`).
   - Gera o DataFrame com colunas: `prompt_id`, `doc_id`, `score_to_anchor`, `rank_pred`, `rank_gold`.
3. **Saida**: um Parquet por modelo + um CSV de resumo (opcional).

## Requisitos

- Mesmas dependencias do projeto (pandas, numpy, sentence-transformers, openai, pyyaml, etc.).
- Dataset `saida_final.json` (ou path configurado no YAML).
- Para modelos OpenAI: `OPENAI_API_KEY` no ambiente ou em `.env`.

## Uso

### Ativar o ambiente virtual

Ative o ambiente onde as dependencias estao instaladas (venv na raiz do projeto ou no diretorio atual):

```bash
# Na raiz do projeto:
source .venv/bin/activate   # Linux/Mac
# ou
.venv\Scripts\activate      # Windows

# Depois entre no diretorio do script
cd embedding_models_eval
```

### Processar todos os modelos

```bash
python run_ranking_only.py
```

### Um so modelo

```bash
python run_ranking_only.py --model minilm
```

### Config e diretorio de saida

```bash
python run_ranking_only.py --config configs/default.yaml --output-dir results/ranking_only
```

### Opcoes

| Opcao          | Padrao               | Descricao                                      |
|----------------|----------------------|------------------------------------------------|
| `--config`     | `configs/default.yaml` | Arquivo YAML de configuracao                  |
| `--model`      | (todos)              | Nome do modelo a processar                     |
| `--output-dir` | `results/ranking_only` | Diretorio dos Parquets e do resumo           |
| `--verbose`    | True                 | Mostrar progresso                              |
| `--no-verbose` | -                   | Reduzir saida no terminal                      |

## Estrutura da saida

- **Diretorio**: por padrao `results/ranking_only/` (ou o indicado em `--output-dir`).

- **Por modelo**: um arquivo Parquet com o DataFrame completo apos ranking:
  - `{modelo}_scored.parquet`
  - Colunas minimas: `prompt_id`, `doc_id`, `score_to_anchor`, `rank_pred`, `rank_gold`.
  - Inclui tambem as colunas originais do dataset para rastreabilidade.

- **Resumo**: `ranking_summary.csv` com uma linha por modelo:
  - `modelo`, `n_linhas`, `n_prompts`, `score_to_anchor_min`, `score_to_anchor_mean`, `score_to_anchor_max`.

## Exemplo de saida no terminal

```
======================================================================
PIPELINE ATE RANKING POR ANCORA
======================================================================

1. CARREGANDO DADOS REAIS
   Dataset: saida_final.json
   Textos validos: 354

2. PROCESSANDO MODELO(S)
[1/8] minilm
   OK: 354 linhas
   Salvo: results/ranking_only/minilm_scored.parquet
...
RESUMO
Modelos processados: 8/8
Saida: results/ranking_only
   Resumo salvo: results/ranking_only/ranking_summary.csv
CONCLUIDO
```

## Relacao com outros scripts

- **run_embeddings_only.py**: para apos gerar embeddings; nao chama ranking.
- **run_ranking_only.py**: para apos ranking por ancora; nao chama metricas.
