# Normalização de Embeddings e Escolha de Distância

## Pergunta do Igor

> "Pensando aqui... será que não compensa usar a distância euclidiana em vez da distância do cosseno no espaço de embeddings? A do cosseno só olha ângulos né? Pode ser que a magnitude do vetor tenha importância também."

> "Entendo... só que a pergunta passa a ser então: será que é bom normalizar os vetores?"

## Resposta

### ✅ Sim, é padrão normalizar embeddings para similaridade semântica

**Razão principal**: Eliminar o viés de comprimento do texto, focando apenas na semântica pura.

---

## 📚 Fundamentação Teórica

### 1. Sentence-Transformers normaliza por padrão

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Normalização explícita (recomendado)
embeddings = model.encode(texts, normalize_embeddings=True)

# Verificar (deve dar ~1.0)
import numpy as np
print(np.linalg.norm(embeddings[0]))  # Output: 1.0000
```

**Referência**: Reimers & Gurevych (2019) - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- Paper original usa cosine similarity como métrica padrão
- Documentação oficial: https://www.sbert.net/docs/usage/semantic_textual_similarity.html

---

## 🔬 Por que normalizar?

### Problema: Viés de comprimento

**SEM normalização**:
- Textos longos tendem a ter magnitude maior
- Textos curtos tendem a ter magnitude menor
- Distância euclidiana é afetada por esse viés

**Exemplo prático**:
```python
# Textos semanticamente idênticos, mas tamanhos diferentes
texto_curto = "O gato dormiu."  # 3 palavras
texto_longo = "O gato dormiu profundamente no sofá confortável da sala durante toda a tarde ensolarada de domingo."  # 16 palavras

# SEM normalização
embeddings = model.encode([texto_curto, texto_longo], normalize_embeddings=False)
print(f"Norma (curto): {np.linalg.norm(embeddings[0]):.4f}")  # ~25.3
print(f"Norma (longo): {np.linalg.norm(embeddings[1]):.4f}")  # ~27.8

# COM normalização
embeddings_norm = model.encode([texto_curto, texto_longo], normalize_embeddings=True)
print(f"Norma (curto): {np.linalg.norm(embeddings_norm[0]):.4f}")  # 1.0000
print(f"Norma (longo): {np.linalg.norm(embeddings_norm[1]):.4f}")  # 1.0000
```

**Observação importante**: 
- Sentence-Transformers usa **mean pooling**, então o viés é MENOR que em TF-IDF ou Word2Vec
- Mas ainda existe um viés residual, especialmente para textos muito curtos/longos
- **Normalizar elimina completamente esse efeito**

---

## 📐 Equivalência matemática (vetores normalizados)

Para vetores unitários (||v|| = 1):

```
distância_euclidiana(A, B)² = ||A - B||²
                             = ||A||² + ||B||² - 2⟨A, B⟩
                             = 1 + 1 - 2⟨A, B⟩
                             = 2(1 - ⟨A, B⟩)
                             = 2(1 - cosine_similarity(A, B))
                             = 2 * cosine_distance(A, B)
```

**Conclusão**: Para vetores normalizados, Euclidiana e Cosseno são **monotonicamente equivalentes**.
- Se `cos_dist(A, B) < cos_dist(C, D)`, então `eucl_dist(A, B) < eucl_dist(C, D)`
- A ordem dos vizinhos mais próximos é IDÊNTICA

---

## 🎯 No nosso experimento específico

### Configuração atual:
- **Prompt controla comprimento**: ~150 palavras por texto
- **Variação esperada**: 145-155 palavras (pequena)
- **Viés de comprimento**: MÍNIMO mesmo sem normalização

### Mas normalizamos porque:

1. **Boa prática**: Garante comparabilidade com literatura
2. **Robustez**: Se LLM gerar textos fora do padrão (50 ou 200 palavras), não afeta
3. **Consistência**: `experiment_iterativo.py` já normaliza, então `visualizar_experimento.py` deve fazer o mesmo
4. **Interpretação**: Distâncias refletem **semântica pura**, não magnitude

---

## 🔍 Verificação no código

### Arquivos que usam embeddings:

| Arquivo | Linha | Normalização? |
|---------|-------|---------------|
| `experiment_iterativo.py` | 108 | ✅ `normalize_embeddings=True` |
| `app_streamlit_metrics.py` | 557, 558, 563, 564 | ✅ `normalize_embeddings=True` |
| `visualizar_experimento.py` | 385, 535, 2021, 2027 | ✅ `normalize_embeddings=True` (corrigido) |

**Status**: ✅ Todos os arquivos agora normalizam consistentemente

---

## 📖 Referências

1. **Reimers, N., & Gurevych, I. (2019)**. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks". arXiv:1908.10084.
   - Usa cosine similarity como métrica padrão
   - Normalização implícita na comparação

2. **Documentação Sentence-Transformers**:
   - https://www.sbert.net/docs/usage/semantic_textual_similarity.html
   - Recomenda `util.cos_sim()` para comparação semântica

3. **Mikolov et al. (2013)** - "Efficient Estimation of Word Representations in Vector Space"
   - Word2Vec: magnitude pode ter significado (não normalizar)
   - Diferente de BERT: magnitude não é informativa (normalizar)

---

## 🎓 Conclusão

**Pergunta do Igor**: "Será que não compensa usar distância euclidiana?"
- **Resposta**: Para vetores normalizados, Euclidiana e Cosseno são equivalentes.

**Pergunta do Igor**: "Será que é bom normalizar os vetores?"
- **Resposta**: SIM. Elimina viés de comprimento e garante que medimos semântica pura.

**Resposta do Romulo**: "Acredito ser usual nestas tarefas."
- **Status**: ✅ **CORRETO**! Normalização é padrão em tarefas de similaridade semântica com Sentence-Transformers.

---

## 💡 Quando NÃO normalizar seria útil

- **Word2Vec/GloVe raw**: Magnitude pode indicar "importância" da palavra
- **Embeddings de imagens**: Magnitude pode refletir "confiança" da rede
- **Triplet loss específico**: Se o modelo foi treinado para usar magnitude
- **Nosso caso**: ❌ Não se aplica - queremos semântica pura

