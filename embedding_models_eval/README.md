# embedding-models-eval

Pipeline de avaliacao de embeddings: ranking por ancora, metricas IR e de votos, saidas tabulares/JSON, etapas opcionais (TF-IDF, visualizacoes com bootstrap) via YAML.

## Requisitos

- Python 3.10 ou superior
- Dados de entrada no formato esperado pelo loader JSON (ex.: `saida_final.json`)
- Para modelos OpenAI: variavel `OPENAI_API_KEY` (ou `.env` carregado pelo `config_loader`)

## Instalacao (recomendado)

Na pasta `embedding_models_eval` do repositorio:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

O comando acima instala **somente** as dependencias **obrigatorias** declaradas em `pyproject.toml` (pandas, torch, sentence-transformers, openai, ranx, etc.), suficientes para rodar o pipeline principal e gerar artefatos (CSV, Parquet, JSON, metricas).

Para incluir **extras**, veja a secao seguinte.

### Instalacao direto do Git

Ajuste URL, branch e use `#subdirectory=embedding_models_eval` se o pacote estiver nessa pasta no monorepo:

```bash
pip install "git+https://github.com/Labic-ICMC-USP/LLM-CreativityScore.git@BRANCH#subdirectory=embedding_models_eval"
```

Extras na URL (exemplo com visualizacoes e testes):

```bash
pip install "embedding-models-eval[viz,dev] @ git+https://github.com/Labic-ICMC-USP/LLM-CreativityScore.git@BRANCH#subdirectory=embedding_models_eval"
```

(A sintaxe exata pode variar com a versao do `pip`; em caso de duvida, clone e use `pip install -e ".[viz,dev]"` na pasta.)

## Dependencias opcionais (extras)

No `pyproject.toml`, o projeto define **grupos opcionais** (`[project.optional-dependencies]`). Eles existem para **nao** obrigar todo mundo a instalar pacotes que so fazem sentido em certos cenarios (plots, ou suite de testes).

### Por que separar?

| Cenario | O que voce precisa |
|---------|---------------------|
| So rodar experimentos (`embedding-eval`) | Dependencias base (`pip install -e .`) |
| Ativar `pipeline_extras.run_visualizations` no YAML | Bibliotecas de plotagem (`[viz]`) |
| Rodar `pytest` no repositorio | Ferramentas de teste (`[dev]`) |

Assim quem **consome** o pacote para producao ou estudos nao puxa `pytest` nem `matplotlib` sem necessidade; quem **desenvolve** ou valida o codigo instala o que falta com um unico sufixo.

### Extra `[viz]` (visualizacoes)

- **O que instala:** `matplotlib`, `seaborn` (versoes minimas no `pyproject.toml`).
- **Para que serve:** o pipeline pode, se `pipeline_extras.run_visualizations: true` no YAML, gerar boxplots, violins e graficos de intervalo de confianca (bootstrap) a partir dos CSVs per-prompt. Esse codigo importa `matplotlib` e `seaborn`; sem o extra, essa etapa pode falhar ao importar.
- **Comando:** `pip install -e ".[viz]"` ou combine com outros: `pip install -e ".[viz,dev]"`.

### Extra `[dev]` (desenvolvimento)

- **O que instala:** `pytest`, `pytest-cov` (testes e cobertura de codigo).
- **Para que serve:**
  - **`pytest`:** executa a suite em `tests/` (`pytest tests/` ou `pytest` com `pytest.ini`).
  - **`pytest-cov`:** opcional para relatorios de cobertura (ex.: `pytest --cov=embedding_models_eval`), util em CI ou revisao de qualidade.
- **Quem precisa:** mantenedores, integracao continua, ou qualquer pessoa que queira **verificar** que o pacote passa nos testes apos mudancas. **Nao** e necessario para apenas **rodar** `embedding-eval` com seus dados.
- **Comando:** `pip install -e ".[dev]"`.

### Combinando extras

Voce pode pedir varios de uma vez, separados por virgula **dentro das aspas**:

```bash
pip install -e ".[viz,dev]"
```

Ordem nao importa. Exemplos:

- Base + plots: `pip install -e ".[viz]"`
- Base + testes: `pip install -e ".[dev]"`
- Tudo para quem desenvolve e gera figuras integradas: `pip install -e ".[viz,dev]"`

### Onde isso esta definido

Abra `pyproject.toml` e procure `[project.optional-dependencies]`. Ali ficam os nomes exatos (`viz`, `dev`) e as versoes; se o grupo adicionar outro extra no futuro (por exemplo ferramentas de lint), a mesma ideia se aplica: `pip install -e ".[novo_extra]"`.

## Comandos (apos `pip install -e`)

| Comando | Descricao |
|---------|-----------|
| `embedding-eval` | Pipeline principal (embeddings + metricas + artefatos) |
| `embedding-viz` | Boxplots, violin e bootstrap (CSVs per-prompt) |
| `embedding-tfidf` | Baseline TF-IDF |

Exemplo:

```bash
embedding-eval --config configs/default.yaml
embedding-eval --config configs/default.yaml --models minilm --output-dir results/teste
```

## Sem instalar o pacote (clone apenas)

Defina `PYTHONPATH` para `src` e use os shims na raiz:

```bash
cd embedding_models_eval
PYTHONPATH=src python run_pipeline.py --config configs/default.yaml
```

## Parametrizacao e dados

1. **YAML**: copie e edite `configs/default.yaml` (`dataset.path`, `models`, `output`, `pipeline_extras`, etc.).
2. **JSON**: aponte `dataset.path` para o seu arquivo; o formato deve ser o mesmo consumido pelo loader em `embedding_models_eval.data`.
3. **Saida**: `output.results_dir` no YAML ou `--output-dir` na CLI.

## Testes

Requer o extra **`[dev]`** (inclui `pytest`). Opcionalmente use tambem `[viz]` se algum teste ou fluxo local depender de matplotlib (na suite atual o foco e `pytest`).

```bash
cd embedding_models_eval
pip install -e ".[dev]"
pytest tests/
```

Com visualizacoes e testes juntos:

```bash
pip install -e ".[viz,dev]"
pytest tests/
```

## Estrutura (resumo)

- `src/embedding_models_eval/` — codigo importavel (`pipeline`, `data`, `embeddings`, `metrics`, `ranking`, `cli.py`, …)
- `configs/` — YAML de referencia
- `tests/` — pytest
- `run_pipeline.py`, `run_visualizations.py`, `run_tfidf_baseline.py` — atalhos que delegam ao pacote

## Documentacao adicional

- `README_RUN_EMBEDDINGS.md` — guia focado em fluxos com `run_embeddings_only.py` e dependencias legadas (`requirements.txt`).
- `requirements.txt` — ainda util para ambientes sem instalar o projeto como pacote.
