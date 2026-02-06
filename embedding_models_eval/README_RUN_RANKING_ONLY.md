# Run Ranking Only: Ranking por Ancora + Metricas @k

## Objetivo

Este script executa o pipeline **completo ate metricas @k**: apos carregar o dataset, gera embeddings, calcula o **ranking por similaridade com ancora** (top-1 por prompt), e avalia o ranking previsto vs gold com **metricas IR @k** (MAP@k, P@k, R@k, F1@k).

## O que e executado

1. **Carregamento de dados**: mesmo loader e config que `run_embeddings_only.py` (dataset, text_col, rank_in_prompt).
2. **Por cada modelo** configurado no YAML:
   - Cria o provider de embeddings.
   - Chama `build_anchor_ranking(df, provider, ...)` (modulo `ranking/`).
   - Gera o DataFrame com colunas: `prompt_id`, `doc_id`, `score_to_anchor`, `rank_pred`, `rank_gold`.
3. **Calculo de metricas IR @k**:
   - Compara `rank_pred` vs `rank_gold` usando a biblioteca `ranx`.
   - Calcula P@k, R@k, F1@k, MAP@k para k em [1, 3, 5, 10] (configuravel).
4. **Saida**: Parquet por modelo + CSV de metricas por prompt + CSV de resumo com metricas macro.

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

- **Por modelo - Ranking**: arquivo Parquet com o DataFrame completo apos ranking:
  - `{modelo}_scored.parquet`
  - Colunas minimas: `prompt_id`, `doc_id`, `score_to_anchor`, `rank_pred`, `rank_gold`.
  - Inclui tambem as colunas originais do dataset para rastreabilidade.

- **Por modelo - Metricas**: arquivo CSV com metricas por prompt:
  - `{modelo}_metrics_per_prompt.csv`
  - Colunas: `prompt_id`, `P@1`, `R@1`, `F1@1`, `AP@1`, `P@3`, `R@3`, ... (para cada k).

- **Resumo macro**: `ranking_summary.csv` com uma linha por modelo:
  - `modelo`, `n_linhas`, `n_prompts`, `score_to_anchor_min`, `score_to_anchor_mean`, `score_to_anchor_max`
  - `MAP@1`, `MAP@3`, `MAP@5`, `MAP@10` (metricas agregadas).

## Exemplo de saida no terminal

```
======================================================================
PIPELINE COMPLETO: RANKING POR ANCORA + METRICAS @k
======================================================================

1. CARREGANDO DADOS REAIS
   Dataset: saida_final.json
   Textos validos: 354

2. PROCESSANDO MODELO(S)
[1/8] minilm
   OK: 354 linhas
   Salvo: results/ranking_only/minilm_scored.parquet
...

3. CALCULANDO METRICAS IR @k
   Valores de k: [1, 3, 5, 10]

   [minilm] Calculando metricas...
   [minilm] MAP@1=0.4523, MAP@3=0.5012, MAP@5=0.5234, MAP@10=0.5456
   [minilm] Metricas por prompt salvas: results/ranking_only/minilm_metrics_per_prompt.csv
...

RESUMO FINAL
Modelos processados: 8/8
Saida: results/ranking_only
   Resumo salvo: results/ranking_only/ranking_summary.csv
CONCLUIDO
```

## Relacao com outros scripts

- **run_embeddings_only.py**: para apos gerar embeddings; nao chama ranking nem metricas.
- **run_ranking_only.py**: pipeline completo ate metricas @k (ranking + avaliacao IR).
