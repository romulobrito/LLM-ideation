# Documentação Completa - Refinement Loop CLI

**Versão:** 1.0  
**Data:** 22/12/2024  
**Autor:** Sistema de Refinamento Iterativo

---

## Índice

1. [Instalação e Configuração do Ambiente](#instalação-e-configuração-do-ambiente)
2. [Estrutura do Arquivo YAML](#estrutura-do-arquivo-yaml)
3. [Uso do CLI](#uso-do-cli)
4. [Fluxo Macro do Sistema](#fluxo-macro-do-sistema)
5. [Detalhamento das Etapas](#detalhamento-das-etapas)
6. [Exemplos Práticos](#exemplos-práticos)
7. [Troubleshooting](#troubleshooting)

---

## 1. Instalação e Configuração do Ambiente

### 1.1. Requisitos do Sistema

- **Python:** 3.8 ou superior
- **Sistema Operacional:** Linux, macOS ou Windows (com WSL)
- **Memória RAM:** Mínimo 4GB (recomendado 8GB+)
- **Espaço em Disco:** ~2GB para dependências e modelos locais

### 1.2. Criação do Ambiente Virtual

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
# Linux/Mac:
source venv/bin/activate

# Windows (PowerShell):
venv\Scripts\Activate.ps1

# Windows (CMD):
venv\Scripts\activate.bat
```

### 1.3. Instalação de Dependências

```bash
# Instalar dependências básicas
pip install pyyaml python-dotenv

# Instalar dependências do projeto (se houver requirements.txt)
pip install -r requirements.txt

# Ou instalar manualmente as principais:
pip install numpy scikit-learn sentence-transformers openai
```

### 1.4. Configuração de API Keys

Crie um arquivo `.env` no diretório raiz do projeto:

```bash
# .env
OPENAI_API_KEY=sk-proj-sua-chave-openai-aqui
OPENROUTER_API_KEY=sk-or-v1-sua-chave-openrouter-aqui
```

**Importante:**
- O CLI carrega automaticamente o arquivo `.env` antes de executar
- O CLI procura o `.env` nos seguintes locais (em ordem):
  1. Diretório do script (`run_refinement_cli.py`)
  2. Diretório atual de trabalho
  3. Diretório padrão do projeto (`~/Documentos/MAI-DAI-USP/experimento_convergencia_visualizacao_metricas/`)

### 1.5. Preparação dos Dados

Organize suas ideias humanas em um diretório:

```
ideas-exp/
  human/
    idea_001.txt
    idea_002.txt
    idea_003.txt
    ...
```

Cada arquivo `.txt` deve conter uma ideia/historia completa.

---

## 2. Estrutura do Arquivo YAML

O arquivo YAML é a única fonte de configuração para o CLI. Todas as opções devem estar presentes.

### 2.1. Estrutura Completa

```yaml
# ============================================
# SEÇÃO 1: PROMPTS (Obrigatórios)
# ============================================

invitation: |
  Seu texto de convite aqui...
  Pode ter multiplas linhas.

directive: "Sua diretiva aqui (string simples)"

# ============================================
# SEÇÃO 2: IDEIAS HUMANAS
# ============================================

human_ideas:
  path: "ideas-exp/human"  # Caminho para diretório ou arquivo
  limit: null  # null = usar todas, ou número específico (ex: 8)

# ============================================
# SEÇÃO 3: MODELO LLM PRINCIPAL
# ============================================

model:
  name: "deepseek/deepseek-v3.2-exp"  # Modelo para critique, packing, generation
  temperature: 1.0  # 0.0 a 2.0
  max_tokens: 4000
  reasoning_effort: minimal  # null, "minimal", "low", "medium", "high"

# ============================================
# SEÇÃO 4: EMBEDDINGS
# ============================================

embedder:
  name: "text-embedding-3-large"  # Modelo para embeddings
  device: "auto"  # "auto", "cpu", "cuda"

# ============================================
# SEÇÃO 5: CLUSTERING (Opcional)
# ============================================

clustering:
  enabled: true  # true = usar clustering, false = usar todas as ideias
  method: "agglomerative"  # "kmeans" ou "agglomerative"
  n_clusters: 4  # Para kmeans
  distance_threshold: 0.4  # Para agglomerative
  min_cluster_size: 8
  selected_cluster_id: null  # null = auto-select melhor cluster

# ============================================
# SEÇÃO 6: NORTH STAR (Norte Fixo)
# ============================================

north_star:
  enabled: true  # true = gerar norte fixo, false = feedback 100% dinâmico
  model: "gpt-4o"  # Modelo para gerar norte fixo

# ============================================
# SEÇÃO 7: CONSOLIDAÇÃO SEMÂNTICA
# ============================================

consolidation:
  enabled: false  # true = consolidar ideias similares antes do critique
  threshold: 0.60  # Limiar de similaridade (0.0 a 1.0)
  max_group_size: 4  # Tamanho maximo de grupo para consolidar
  model: "gpt-4o-mini"  # Modelo para consolidar ideias
  temperature: 0.3  # Temperatura para consolidação

# ============================================
# SEÇÃO 8: GERAÇÃO DE IDEIAS
# ============================================

generation:
  num_ideas_per_iter: 16  # Número de ideias a gerar por iteração
  temperature: 1.0  # Temperatura para geração

# ============================================
# SEÇÃO 9: CONVERGÊNCIA
# ============================================

convergence:
  max_iterations: 20  # Numero maximo de iteracoes
  patience: 10  # Iteracoes sem melhoria antes de parar
  delta_threshold: 0.005  # Melhoria minima para considerar progresso
  optimize_metric: "min"  # Metrica para otimizar: "avg", "min", "top3_mean", "centroid", "centroid_to_centroid", "separability"

# ============================================
# SEÇÃO 10: PARADA POR DIVERGÊNCIA
# ============================================

divergence_stop:
  enabled: true  # Habilitar parada por divergência
  threshold: 0.08  # Piora máxima tolerada (0.08 = 8%)
  max_consecutive_worsening: 3  # Max iterações consecutivas piorando
  max_distance_from_start: 0.30  # Distância máxima da iteração 1

# ============================================
# SEÇÃO 11: OUTPUT
# ============================================

output:
  dir: "exp_refinement_test"  # Diretório de saída (relativo ao CWD ou absoluto)
```

### 2.2. Explicação Detalhada das Seções

#### 2.2.1. `invitation` e `directive` (Obrigatórios)

- **`invitation`**: Texto de contexto que introduz o tema/estilo desejado. Pode ter múltiplas linhas.
- **`directive`**: Instrução específica sobre o que gerar. Deve ser uma string simples.

**Exemplo:**
```yaml
invitation: |
  Strangers Again
  
  I've been thinking a lot lately about the need to feel connected...

directive: "Center your story around two characters who like each other but don't get a happily ever after."
```

#### 2.2.2. `human_ideas`

- **`path`**: Caminho para diretório contendo arquivos `.txt` com ideias humanas, ou caminho para um arquivo único.
- **`limit`**: Número máximo de ideias a usar (`null` = usar todas).

#### 2.2.3. `model`

- **`name`**: Modelo LLM principal usado em critique, packing e generation.
  - OpenAI: `gpt-4o`, `gpt-4o-mini`
  - DeepSeek (via OpenRouter): `deepseek/deepseek-v3.2-exp`, `deepseek/deepseek-chat`
  - O1: `o1-mini`
- **`temperature`**: Controla aleatoriedade (0.0 = determinista, 2.0 = muito aleatório).
- **`max_tokens`**: Número máximo de tokens na resposta.
- **`reasoning_effort`**: Nível de raciocínio para modelos DeepSeek (`null`, `"minimal"`, `"low"`, `"medium"`, `"high"`).

#### 2.2.4. `embedder`

- **`name`**: Modelo para gerar embeddings semanticos.
  - OpenAI: `text-embedding-3-large` (3072D, mais preciso), `text-embedding-3-small` (1536D)
  - Sentence Transformers: `all-MiniLM-L6-v2` (384D, local), `all-mpnet-base-v2` (768D)
- **`device`**: Dispositivo para modelos locais (`"auto"`, `"cpu"`, `"cuda"`).

#### 2.2.5. `clustering`

- **`enabled`**: Se `true`, agrupa ideias humanas por similaridade e seleciona um cluster.
- **`method`**: Algoritmo de clustering (`"kmeans"` ou `"agglomerative"`).
- **`n_clusters`**: Numero de clusters (para kmeans).
- **`distance_threshold`**: Threshold de distancia (para agglomerative).
- **`min_cluster_size`**: Tamanho mínimo do cluster (expande com vizinhos se menor).
- **`selected_cluster_id`**: ID do cluster a usar (`null` = auto-select melhor).

#### 2.2.6. `north_star`

- **`enabled`**: Se `true`, gera diretrizes CORE (norte fixo) baseadas em padrões das ideias humanas.
- **`model`**: Modelo usado para gerar o norte fixo (geralmente `gpt-4o` para maior precisão).

#### 2.2.7. `consolidation`

- **`enabled`**: Se `true`, consolida ideias similares antes do critique usando similaridade semântica.
- **`threshold`**: Limiar de similaridade coseno (0.0 a 1.0). Valores maiores = menos consolidação.
- **`max_group_size`**: Tamanho máximo de grupo para consolidar (grupos maiores são consolidados hierarquicamente).
- **`model`**: Modelo para consolidar ideias similares.
- **`temperature`**: Temperatura para consolidação (geralmente baixa, ex: 0.3).

#### 2.2.8. `generation`

- **`num_ideas_per_iter`**: Número de ideias a gerar em cada iteração.
- **`temperature`**: Temperatura para geração de ideias.

#### 2.2.9. `convergence`

- **`max_iterations`**: Número máximo de iterações a executar.
- **`patience`**: Número de iterações sem melhoria antes de parar (early stopping).
- **`delta_threshold`**: Melhoria mínima na métrica para considerar progresso.
- **`optimize_metric`**: Métrica a otimizar:
  - `"avg"`: Distância média
  - `"min"`: Distância mínima
  - `"top3_mean"`: Média das 3 melhores
  - `"centroid"`: Distância do centroide
  - `"centroid_to_centroid"`: Distância entre centroides
  - `"separability"`: Indistinguibilidade (quanto menor, melhor)

#### 2.2.10. `divergence_stop`

- **`enabled`**: Se `true`, para se o sistema estiver divergindo.
- **`threshold`**: Piora máxima tolerada (ex: 0.08 = 8%).
- **`max_consecutive_worsening`**: Max iterações consecutivas piorando antes de parar.
- **`max_distance_from_start`**: Distância máxima da iteração 1 antes de parar.

#### 2.2.11. `output`

- **`dir`**: Diretório onde os resultados serão salvos (criado automaticamente com timestamp).

---

## 3. Uso do CLI

### 3.1. Comando Básico

```bash
python3 run_refinement_cli.py config_refinement.yaml
```

### 3.2. Validação sem Executar (Dry Run)

```bash
python3 run_refinement_cli.py config_refinement.yaml --dry-run
```

Isso valida o YAML e mostra um resumo da configuração sem executar o loop.

### 3.3. Exemplo de Saída do Dry Run

```
[CLI] Arquivo .env carregado de: /path/to/.env
[CLI] Carregando config: config_refinement.yaml
[CLI] Validando configuração...
[CLI] Criando configuração...

============================================================
DRY RUN: Config validado com sucesso!
============================================================
  Invitation: Strangers Again...
  Directive: Center your story around two characters...
  Ideias humanas: 16
  Todas ideias (clustering): 16
  Modelo: deepseek/deepseek-v3.2-exp
  Embedder: text-embedding-3-large
  Max iteracoes: 20
  Clustering: Sim
  North Star: Sim
  Consolidação: Não
  Output dir: /path/to/exp_refinement_test
============================================================
```

### 3.4. Execução Completa

```bash
# Executar experimento completo
python3 run_refinement_cli.py config_refinement.yaml

# Executar em background (Linux/Mac)
nohup python3 run_refinement_cli.py config_refinement.yaml > experimento.log 2>&1 &

# Executar com redirecionamento de saída
python3 run_refinement_cli.py config_refinement.yaml 2>&1 | tee experimento.log
```

---

## 4. Fluxo Macro do Sistema

O sistema executa o seguinte fluxo completo:

```
┌─────────────────────────────────────────────────────────────┐
│                    INICIO (CLI)                              │
│  python3 run_refinement_cli.py config_refinement.yaml       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  FASE 0: INICIALIZAÇÃO                                       │
│  ├─ Carregar .env (API keys)                                │
│  ├─ Carregar YAML config                                     │
│  ├─ Validar configuração                                     │
│  └─ Criar RefinementConfig                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  FASE 1: SETUP DO REFINEMENT LOOP                            │
│  ├─ Inicializar embedder (text-embedding-3-large)           │
│  ├─ Carregar ideias humanas                                 │
│  ├─ Computar embeddings das ideias humanas                   │
│  └─ Criar diretório de output (com timestamp)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  FASE 2: CLUSTERING (Opcional)                               │
│  Se use_clustering=True:                                     │
│  ├─ Clusterizar todas as ideias humanas                     │
│  ├─ Analisar diversidade dos clusters                       │
│  ├─ Selecionar melhor cluster                               │
│  ├─ Expandir cluster se necessário                           │
│  └─ Selecionar representantes do cluster                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  FASE 3: GERAÇÃO DO NORTH STAR (Opcional)                    │
│  Se use_north_star=True:                                     │
│  ├─ Analisar ideias humanas para extrair padrões           │
│  ├─ Gerar diretrizes CORE (imutáveis)                       │
│  └─ Formatar North Star com bullets                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  FASE 4: GERAÇÃO DE IDEIAS INICIAIS (PURAS)                  │
│  ├─ Gerar ideias LLM sem critique (baseline)                │
│  ├─ Computar embeddings das ideias iniciais                  │
│  ├─ Calcular centroide das ideias iniciais                  │
│  └─ Calcular TODAS as métricas iniciais (baselines)         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  LOOP ITERATIVO (max_iterations vezes)                       │
│  │                                                           │
│  ├─ ITERAÇÃO 1                                              │
│  │   ├─ ETAPA 1: CRITIQUE                                    │
│  │   │   ├─ Selecionar ideias variadas (melhores +         │
│  │   │   │   medianas + piores + recentes)                 │
│  │   │   ├─ CONSOLIDAÇÃO SEMÂNTICA (opcional)               │
│  │   │   │   ├─ Agrupar ideias similares (cosine)           │
│  │   │   │   └─ Consolidar grupos com LLM                   │
│  │   │   ├─ Comparar ideias LLM vs humanas                  │
│  │   │   └─ Gerar feedback JSON (vibes + contrastes)        │
│  │   │                                                       │
│  │   ├─ ETAPA 2: PACKING                                    │
│  │   │   ├─ Consolidar JSON em bullets acionáveis          │
│  │   │   ├─ Sumarizar bullets anteriores                   │
│  │   │   ├─ Deduplicação semântica                         │
│  │   │   └─ Combinar North Star + bullets táticos           │
│  │   │                                                       │
│  │   ├─ ETAPA 3: GENERATION                                 │
│  │   │   ├─ Revisar directive com bullets                  │
│  │   │   └─ Gerar novas ideias (num_ideas_per_iter)        │
│  │   │                                                       │
│  │   ├─ CÁLCULO DE MÉTRICAS                                 │
│  │   │   ├─ Distâncias (avg, min, top3, centroid)          │
│  │   │   ├─ Separabilidade (AUC do classificador)          │
│  │   │   └─ Métricas normalizadas (vs baseline)            │
│  │   │                                                       │
│  │   ├─ VERIFICAÇÃO DE CONVERGÊNCIA                         │
│  │   │   ├─ EMA (Exponential Moving Average)               │
│  │   │   ├─ Early stopping (patience)                       │
│  │   │   └─ Parada por divergência                          │
│  │   │                                                       │
│  │   └─ SALVAR RESULTADOS                                   │
│  │       ├─ iteration_N.json                               │
│  │       └─ summary.json                                    │
│  │                                                           │
│  ├─ ITERAÇÃO 2                                              │
│  │   └─ (mesmo processo)                                    │
│  │                                                           │
│  └─ ... (até convergência ou max_iterations)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  FASE 5: FINALIZACAO                                         │
│  ├─ Gerar plots (se disponivel)                             │
│  ├─ Salvar summary final                                    │
│  └─ Exibir resumo de resultados                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Detalhamento das Etapas

### 5.1. Fase 0: Inicialização (CLI)

**Arquivo:** `run_refinement_cli.py`

1. **Carregar .env**: Procura arquivo `.env` em múltiplos locais e carrega API keys
2. **Carregar YAML**: Parse do arquivo YAML de configuração
3. **Validar Config**: Verifica campos obrigatórios e valores válidos
4. **Criar RefinementConfig**: Mapeia YAML para dataclass `RefinementConfig`
5. **Carregar Ideias Humanas**: Lê arquivos `.txt` do diretório especificado

**Saída:**
- Configuração validada
- `RefinementConfig` criada

---

### 5.2. Fase 1: Setup do Refinement Loop

**Arquivo:** `refinement_loop.py` (metodo `__init__`)

1. **Inicializar Embedder**:
   ```python
   self.embedder = get_embedder(self.config.embedder_name, device=device)
   ```
   - Carrega modelo de embeddings (OpenAI ou Sentence Transformers)
   - Determina device automaticamente (CUDA se disponivel)

2. **Embed Ideias Humanas**:
   ```python
   self.human_embeddings = embed_texts(self.embedder, self.config.human_ideas)
   ```
   - Gera embeddings de todas as ideias humanas
   - Shape: `(num_ideas, embedding_dim)`

3. **Criar Diretório de Output**:
   ```python
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   self.experiment_dir = self.config.output_dir / timestamp
   ```

**Saída:**
- Embedder inicializado
- Embeddings das ideias humanas computados
- Diretório de experimento criado

---

### 5.3. Fase 2: Clustering (Opcional)

**Arquivo:** `refinement_loop.py` (metodo `run`, linhas ~323-400)

**Se `use_clustering=True`:**

1. **Clusterizar Ideias Humanas**:
   ```python
   cluster_labels, clusters_dict = cluster_human_ideas(
       human_ideas=self.config.all_human_ideas,
       embedder=self.embedder,
       method=self.config.clustering_method,
       n_clusters=self.config.n_clusters,
       distance_threshold=self.config.distance_threshold,
   )
   ```
   - Gera embeddings de todas as ideias
   - Aplica algoritmo de clustering (K-means ou Agglomerative)
   - Retorna labels e dicionário de clusters

2. **Analisar Diversidade**:
   ```python
   cluster_stats = analyze_cluster_diversity(
       human_ideas=self.config.all_human_ideas,
       embedder=self.embedder,
       labels=cluster_labels,
       clusters_dict=clusters_dict,
   )
   ```
   - Calcula métricas de diversidade por cluster
   - Identifica cluster mais representativo

3. **Selecionar Cluster**:
   ```python
   self.selected_cluster_id = get_best_cluster(cluster_stats)
   ```
   - Seleciona melhor cluster baseado em métricas
   - Ou usa `selected_cluster_id` se especificado

4. **Expandir Cluster** (se necessário):
   - Se cluster menor que `min_cluster_size`, adiciona vizinhos proximos

5. **Selecionar Representantes**:
   ```python
   cluster_ideas = select_cluster_representatives(...)
   ```
   - Seleciona ideias representativas do cluster

**Saída:**
- Cluster selecionado
- Ideias do cluster (`self.config.human_ideas` atualizado)

---

### 5.4. Fase 3: Geração do North Star (Opcional)

**Arquivo:** `refinement_loop.py` (método `run`, linhas ~398-413)  
**Módulo:** `refinement_north.py`

**Se `use_north_star=True`:**

1. **Gerar North Star**:
   ```python
   self.north_star = generate_north_star(
       invitation=self.config.invitation,
       directive=self.config.directive,
       human_ideas=self.config.human_ideas,
       model=self.config.north_star_model,
       temperature=0.3,
       max_tokens=2000,
   )
   ```

2. **Processo Interno** (`refinement_north.py`):
   - Analisa ideias humanas para extrair padrões fundamentais
   - Gera diretrizes CORE (imutáveis) que caracterizam o estilo humano
   - Formata como bullets de diretrizes

**Exemplo de North Star:**
```
CORE DIRECTIVES (North Star):
- Stories focus on emotional connection between two characters
- Endings are bittersweet or unresolved, not happily ever after
- Characters have depth and internal conflict
- Dialogue feels natural and authentic
```

**Saída:**
- `self.north_star`: Texto com diretrizes CORE

---

### 5.5. Fase 4: Geração de Ideias Iniciais (PURAS)

**Arquivo:** `refinement_loop.py` (método `run`, linhas ~415-448)

1. **Gerar Ideias LLM sem Critique**:
   ```python
   current_llm_ideas = self._generate_initial_ideas()
   ```
   - Chama `generation_step` com directive original (sem bullets)
   - Gera `num_ideas_per_iter` ideias
   - Estas sao as ideias "PURAS" (baseline sem refinamento)

2. **Calcular Centroide Inicial**:
   ```python
   initial_embeddings = embed_texts(self.embedder, current_llm_ideas)
   self.initial_centroid = np.mean(initial_embeddings, axis=0)
   ```
   - Computa embeddings das ideias iniciais
   - Calcula centroide (media dos embeddings)
   - Normaliza o centroide

3. **Calcular Baselines de Todas as Métricas**:
   ```python
   (initial_avg_dist, initial_min_dist, ..., initial_separability, initial_auc) = 
       self._compute_distances(current_llm_ideas)
   ```
   - Calcula todas as métricas das ideias PURAS vs humanas
   - Salva como baselines para normalização futura

**Métricas Calculadas:**
- `avg_distance`: Distancia media
- `min_distance`: Distancia minima
- `top3_mean`: Media das 3 melhores
- `centroid_distance`: Distancia do centroide
- `centroid_to_centroid`: Distancia entre centroides
- `separability`: Indistinguibilidade (AUC do classificador)

**Saída:**
- Ideias iniciais PURAS
- Centroide inicial
- Baselines de todas as métricas

---

### 5.6. Loop Iterativo: Etapa 1 - CRITIQUE

**Arquivo:** `refinement_loop.py` (método `run`, linhas ~458-517)  
**Módulo:** `refinement_critique.py`

**Processo:**

1. **Selecionar Ideias Variadas**:
   ```python
   if len(all_generated_ideas) > 10:
       best_ideas = self._select_best_ideas(all_generated_ideas, k=3)
       median_ideas = self._select_median_ideas(all_generated_ideas, k=3)
       worst_ideas = self._select_worst_ideas(all_generated_ideas, k=2)
       recent_ideas = all_generated_ideas[-2:]
       critique_llm_ideas = list(dict.fromkeys(
           best_ideas + median_ideas + worst_ideas + recent_ideas
       ))[:10]
   ```
   - Seleciona 3 melhores + 3 medianas + 2 piores + 2 recentes
   - Garante feedback sobre problemas persistentes, nao so acertos

2. **Consolidação Semântica (Opcional)**:
   ```python
   if self.config.enable_consolidation:
       consolidated_ideas, consolidation_meta = consolidate_similar_ideas(
           ideas=critique_llm_ideas,
           embedder=self.embedder,
           model=self.config.consolidation_model,
           threshold=self.config.consolidation_threshold,
           max_group_size=self.config.consolidation_max_group_size,
           ...
       )
   ```
   - Agrupa ideias similares usando cosine similarity
   - Consolida cada grupo em uma unica ideia usando LLM
   - Reduz redundância antes do critique

3. **Gerar Critique JSON**:
   ```python
   critique_json = critique_step(
       invitation=self.config.invitation,
       directive=self.config.directive,
       human_ideas=self.config.human_ideas,
       llm_ideas=critique_llm_ideas,
       model=self.config.model,
       temperature=0.7,
       previous_feedbacks=previous_feedbacks,
   )
   ```

4. **Processo Interno** (`refinement_critique.py`):
   - Compara ideias LLM vs humanas
   - Identifica "vibes" (caracteristicas) das ideias humanas
   - Identifica "contrastes" (diferenças) entre LLM e humanas
   - Retorna JSON estruturado:
     ```json
     [
       {
         "vibe": "emotional depth",
         "human_examples": ["...", "..."],
         "llm_examples": ["...", "..."],
         "contrast": "LLM ideas lack internal conflict"
       },
       ...
     ]
     ```

**Saída:**
- `critique_json`: Lista de objetos com vibes, exemplos e contrastes

---

### 5.7. Loop Iterativo: Etapa 2 - PACKING

**Arquivo:** `refinement_loop.py` (método `run`, linhas ~519-549)  
**Módulo:** `refinement_packing.py`

**Processo:**

1. **Preparar Bullets Anteriores**:
   ```python
   previous_bullets = None
   if len(self.results) > 0:
       prev_bullets_list = []
       for prev_result in self.results:
           if hasattr(prev_result, 'tactical_bullets'):
               prev_bullets_list.append(prev_result.tactical_bullets)
       previous_bullets = "\n".join(prev_bullets_list)
   ```
   - Coleta bullets de todas as iterações anteriores
   - Usado para sumarização

2. **Gerar Bullets Táticos**:
   ```python
   tactical_bullets = packing_step(
       critique_json=critique_json,
       directive=self.config.directive,
       model=self.config.model,
       temperature=0.5,
       previous_bullets=previous_bullets,
       max_bullets=15,
       embedder=self.embedder,
       dedup_threshold=0.90,
   )
   ```

3. **Processo Interno** (`refinement_packing.py`):
   - Consolida JSON do critique em bullets acionáveis
   - Sumariza bullets anteriores (evita repetição)
   - Deduplicação semântica (remove bullets muito similares)
   - Retorna string com bullets formatados

**Exemplo de Bullets Táticos:**
```
CURRENT DIRECTIVES (Tactical):
- Add internal conflict to characters
- Use more natural dialogue
- Avoid happy endings
- Focus on emotional connection
```

4. **Combinar North Star + Bullets Taticos**:
   ```python
   if self.config.use_north_star:
       tactical_bullets = self._check_and_resolve_conflicts(
           self.north_star, tactical_bullets
       )
       bullets = format_north_with_tactical(self.north_star, tactical_bullets)
   else:
       bullets = tactical_bullets
   ```
   - Verifica conflitos entre CORE e CURRENT
   - Formata combinacao final

**Saída:**
- `bullets`: String com diretrizes combinadas (North Star + Tactical)

---

### 5.8. Loop Iterativo: Etapa 3 - GENERATION

**Arquivo:** `refinement_loop.py` (método `run`, linhas ~551-570)  
**Módulo:** `refinement_generation.py`

**Processo:**

1. **Revisar Directive com Bullets**:
   ```python
   revised_directive = f"{self.config.directive}\n\n{bullets}"
   ```

2. **Gerar Novas Ideias**:
   ```python
   new_ideas = generation_step(
       invitation=self.config.invitation,
       directive=revised_directive,
       num_ideas=self.config.num_ideas_per_iter,
       model=self.config.model,
       temperature=self.config.temperature,
       max_tokens=self.config.max_tokens,
       api_key_override=self.config.api_key_override,
       reasoning_effort=self.config.reasoning_effort,
   )
   ```

3. **Processo Interno** (`refinement_generation.py`):
   - Chama LLM com invitation + directive revisada
   - Gera `num_ideas_per_iter` ideias
   - Retorna lista de strings

**Saída:**
- `new_ideas`: Lista de novas ideias geradas

---

### 5.9. Loop Iterativo: Cálculo de Métricas

**Arquivo:** `refinement_loop.py` (método `_compute_distances`)

**Processo:**

1. **Computar Embeddings das Ideias LLM**:
   ```python
   llm_embeddings = embed_texts(self.embedder, llm_ideas)
   ```

2. **Calcular Distancias**:
   ```python
   distances = cosine_distance(llm_embeddings, self.human_embeddings)
   ```
   - Shape: `(num_llm_ideas, num_human_ideas)`

3. **Calcular Métricas**:
   - `avg_distance`: Média de todas as distâncias
   - `min_distance`: Menor distância encontrada
   - `top3_mean`: Média das 3 menores distâncias
   - `centroid_distance`: Distância do centroide das ideias LLM ao centroide das humanas
   - `centroid_to_centroid`: Distância entre centroides normalizados

4. **Calcular Separabilidade**:
   ```python
   separability_score, auc = self._compute_separability(llm_ideas)
   ```
   - Treina classificador (Logistic Regression) para distinguir LLM vs humanas
   - Usa validacao cruzada (StratifiedKFold) para evitar data leakage
   - Calcula ROC-AUC
   - `separability_score = abs(auc - 0.5)`: 0.0 = indistinguível, 0.5 = muito separável

5. **Normalizar Métricas**:
   ```python
   normalized_avg = avg_distance / self.initial_avg_distance
   normalized_min = min_distance / self.initial_min_distance
   ...
   ```
   - Divide cada métrica pelo seu baseline (ideias PURAS)
   - Valores < 1.0 = melhoria, > 1.0 = piora

**Saída:**
- Todas as métricas (raw e normalizadas)
- Separabilidade e AUC

---

### 5.10. Loop Iterativo: Verificação de Convergência

**Arquivo:** `refinement_loop.py` (método `_check_convergence`)

**Processo:**

1. **Calcular EMA (Exponential Moving Average)**:
   ```python
   self.ema_avg_dist = (1 - alpha) * self.ema_avg_dist + alpha * avg_distance
   ```
   - Suaviza métricas para detectar tendências
   - `alpha = 0.30` (fator de suavização)

2. **Verificar Early Stopping (Patience)**:
   ```python
   if current_metric < best_metric - delta_threshold:
       no_improvement_count = 0  # Reset contador
   else:
       no_improvement_count += 1
   
   if no_improvement_count >= patience:
       converged = True  # Parar
   ```
   - Se não houver melhoria por `patience` iterações, para

3. **Verificar Parada por Divergência**:
   ```python
   if enable_divergence_stop:
       if current_metric > best_metric * (1 + threshold):
           worsening_count += 1
       if worsening_count >= max_consecutive_worsening:
           converged = True  # Parar
   ```
   - Se piorar por `max_consecutive_worsening` iterações consecutivas, para

**Saída:**
- `converged`: Boolean indicando se convergiu
- `convergence_reason`: Razão da convergência

---

### 5.11. Loop Iterativo: Salvar Resultados

**Arquivo:** `refinement_loop.py` (método `_save_iteration_results`)

**Processo:**

1. **Criar IterationResult**:
   ```python
   result = IterationResult(
       iteration=iteration,
       llm_ideas=new_ideas,
       critique_json=critique_json,
       tactical_bullets=tactical_bullets,
       avg_distance=avg_distance,
       min_distance=min_distance,
       ...
   )
   ```

2. **Salvar JSON**:
   ```python
   json_path = self.experiment_dir / f"iteration_{iteration:02d}.json"
   with open(json_path, 'w', encoding='utf-8') as f:
       json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
   ```

3. **Atualizar Summary**:
   ```python
   summary = {
       "config": {...},
       "iterations": [...],
       "converged": self.converged,
       "convergence_reason": self.convergence_reason,
   }
   summary_path = self.experiment_dir / "summary.json"
   ```

**Saída:**
- `iteration_N.json`: Resultados de cada iteração
- `summary.json`: Resumo completo do experimento

---

## 6. Exemplos Praticos

### 6.1. Exemplo 1: Configuração Básica

```yaml
invitation: |
  Write a short story about friendship.

directive: "Create a story where two friends face a challenge together."

human_ideas:
  path: "ideas-exp/human"
  limit: 10

model:
  name: "gpt-4o-mini"
  temperature: 1.0
  max_tokens: 2000
  reasoning_effort: null

embedder:
  name: "text-embedding-3-large"
  device: "auto"

clustering:
  enabled: false
  method: "agglomerative"
  n_clusters: 4
  distance_threshold: 0.4
  min_cluster_size: 5
  selected_cluster_id: null

north_star:
  enabled: true
  model: "gpt-4o"

consolidation:
  enabled: false
  threshold: 0.60
  max_group_size: 4
  model: "gpt-4o-mini"
  temperature: 0.3

generation:
  num_ideas_per_iter: 8
  temperature: 1.0

convergence:
  max_iterations: 10
  patience: 5
  delta_threshold: 0.01
  optimize_metric: "min"

divergence_stop:
  enabled: true
  threshold: 0.10
  max_consecutive_worsening: 3
  max_distance_from_start: 0.35

output:
  dir: "exp_basico"
```

**Executar:**
```bash
python3 run_refinement_cli.py config_basico.yaml
```

### 6.2. Exemplo 2: Configuração Avançada com Clustering

```yaml
invitation: |
  Strangers Again
  
  I've been thinking a lot lately about the need to feel connected...

directive: "Center your story around two characters who like each other but don't get a happily ever after."

human_ideas:
  path: "ideas-exp/human"
  limit: null  # Usar todas

model:
  name: "deepseek/deepseek-v3.2-exp"
  temperature: 1.0
  max_tokens: 4000
  reasoning_effort: minimal

embedder:
  name: "text-embedding-3-large"
  device: "auto"

clustering:
  enabled: true  # ATIVADO
  method: "agglomerative"
  n_clusters: 4
  distance_threshold: 0.4
  min_cluster_size: 8
  selected_cluster_id: null  # Auto-select

north_star:
  enabled: true
  model: "gpt-4o"

consolidation:
  enabled: true  # ATIVADO
  threshold: 0.65
  max_group_size: 4
  model: "gpt-4o-mini"
  temperature: 0.3

generation:
  num_ideas_per_iter: 16
  temperature: 1.0

convergence:
  max_iterations: 20
  patience: 10
  delta_threshold: 0.005
  optimize_metric: "separability"  # Otimizar indistinguibilidade

divergence_stop:
  enabled: true
  threshold: 0.08
  max_consecutive_worsening: 3
  max_distance_from_start: 0.30

output:
  dir: "exp_avancado"
```

**Executar:**
```bash
python3 run_refinement_cli.py config_avancado.yaml --dry-run  # Validar primeiro
python3 run_refinement_cli.py config_avancado.yaml  # Executar
```

---

## 7. Troubleshooting

### 7.1. Erro: "OPENAI_API_KEY não encontrada"

**Causa:** Arquivo `.env` não encontrado ou chave ausente.

**Solução:**
1. Verifique se o arquivo `.env` existe em um dos locais:
   - Diretório do script
   - Diretório atual
   - `~/Documentos/MAI-DAI-USP/experimento_convergencia_visualizacao_metricas/`
2. Verifique se contém `OPENAI_API_KEY=...`
3. Se usar embeddings OpenAI, a chave é obrigatória

### 7.2. Erro: "Nenhuma chave OpenRouter encontrada"

**Causa:** `OPENROUTER_API_KEY` ausente no `.env`.

**Solução:**
1. Adicione `OPENROUTER_API_KEY=...` no `.env`
2. Necessaria para modelos DeepSeek (ex: `deepseek/deepseek-v3.2-exp`)

### 7.3. Erro: "Campo obrigatório faltando"

**Causa:** Seção ou campo obrigatório ausente no YAML.

**Solução:**
1. Use `--dry-run` para identificar o campo faltante
2. Verifique se todas as seções estão presentes
3. Verifique se campos obrigatórios dentro das seções estão preenchidos

### 7.4. Erro: "Caminho não existe" (human_ideas)

**Causa:** Caminho especificado em `human_ideas.path` não existe.

**Solução:**
1. Verifique se o caminho está correto (relativo ao CWD ou absoluto)
2. Verifique se o diretório contém arquivos `.txt`
3. Use caminho absoluto se necessário

### 7.5. Erro: "Nenhuma ideia encontrada"

**Causa:** Diretório vazio ou arquivos vazios.

**Solução:**
1. Verifique se o diretório tem arquivos `.txt`
2. Verifique se os arquivos não estão vazios
3. Verifique permissões de leitura

### 7.6. Erro: "YAML inválido"

**Causa:** Sintaxe YAML incorreta.

**Solução:**
1. Verifique indentação (YAML é sensível a espaços)
2. Verifique se strings com múltiplas linhas usam `|` ou `>`
3. Use um validador YAML online

### 7.7. Erro durante execução: "RuntimeError: ..."

**Causa:** Erro durante o loop (API, memória, etc.).

**Solução:**
1. Verifique logs completos no terminal
2. Verifique se API keys são válidas
3. Verifique se há espaço em disco suficiente
4. Verifique se a memória RAM é suficiente

### 7.8. Experimento não converge

**Causa:** Configuração de convergência muito restritiva ou métrica inadequada.

**Solução:**
1. Aumente `patience` (ex: de 5 para 10)
2. Diminua `delta_threshold` (ex: de 0.01 para 0.005)
3. Tente outra `optimize_metric` (ex: `"min"` em vez de `"avg"`)
4. Aumente `max_iterations`

### 7.9. Consolidação não reduz ideias

**Causa:** Threshold muito alto ou ideias muito diferentes.

**Solução:**
1. Diminua `consolidation.threshold` (ex: de 0.70 para 0.60)
2. Verifique se as ideias são realmente similares
3. Aumente `max_group_size` se grupos são grandes

---

## 8. Estrutura de Output

Após a execução, o diretório de output contém:

```
exp_refinement_test/
  20251222_160353/  # Timestamp da execucao
    iteration_01.json
    iteration_02.json
    ...
    iteration_20.json
    summary.json
    plots/  # Se gerado
      trajetoria_iter1.png
      convergencia_multiplas_metricas.png
      ...
```

### 8.1. Estrutura de `iteration_N.json`

```json
{
  "iteration": 1,
  "llm_ideas": ["...", "..."],
  "critique_json": [...],
  "tactical_bullets": "...",
  "avg_distance": 0.234,
  "min_distance": 0.156,
  "top3_mean": 0.178,
  "centroid_distance": 0.201,
  "centroid_to_centroid": 0.198,
  "separability_score": 0.045,
  "separability_auc": 0.545,
  "normalized_avg_distance": 0.987,
  "normalized_min_distance": 0.923,
  ...
}
```

### 8.2. Estrutura de `summary.json`

```json
{
  "config": {
    "model": "deepseek/deepseek-v3.2-exp",
    "embedder": "text-embedding-3-large",
    "max_iterations": 20,
    ...
  },
  "iterations": [
    {
      "iteration": 1,
      "avg_distance": 0.234,
      ...
    },
    ...
  ],
  "converged": false,
  "convergence_reason": "max_iterations",
  "final_metrics": {
    "avg_distance": 0.189,
    "min_distance": 0.134,
    ...
  }
}
```

---

## 9. Referências

- **CLI Script:** `run_refinement_cli.py`
- **Loop Principal:** `refinement_loop.py`
- **Critique:** `refinement_critique.py`
- **Packing:** `refinement_packing.py`
- **Generation:** `refinement_generation.py`
- **North Star:** `refinement_north.py`
- **Clustering:** `refinement_clustering.py`
- **Consolidation:** `refinement_consolidation.py`
- **Documentacao CLI:** `README_CLI.md`
- **Fluxo Detalhado:** `FLUXO_COMPLETO_DETALHADO.md`

---



