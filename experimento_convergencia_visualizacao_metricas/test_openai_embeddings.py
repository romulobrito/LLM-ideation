#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testa embeddings OpenAI vs Sentence Transformers
"""

import os
import numpy as np
from dotenv import load_dotenv
from experiment_iterativo import get_embedder, embed_texts, cosine_distance

load_dotenv()

# Textos de teste
texts = [
    "A fisherman and a biologist fall in love on a boat, but she must leave for her research.",
    "An archivist and a photographer meet in a museum, drawn together by nostalgia.",
    "Two strangers meet on a train and share poetry, but their lives diverge at dawn.",
]

print("=" * 70)
print("TESTE: EMBEDDINGS OPENAI vs SENTENCE TRANSFORMERS")
print("=" * 70)

# Teste 1: Sentence Transformers (local, gratis)
print("\n🧪 TESTE 1: all-MiniLM-L6-v2 (384D, local)")
try:
    embedder_local = get_embedder("all-MiniLM-L6-v2")
    embeddings_local = embed_texts(embedder_local, texts)
    print(f"   ✅ Embeddings gerados: {embeddings_local.shape}")
    
    # Calcular distancias
    d01 = cosine_distance(embeddings_local[0], embeddings_local[1])
    d02 = cosine_distance(embeddings_local[0], embeddings_local[2])
    d12 = cosine_distance(embeddings_local[1], embeddings_local[2])
    
    print(f"   📊 Distancias:")
    print(f"      d(1,2) = {d01:.4f}")
    print(f"      d(1,3) = {d02:.4f}")
    print(f"      d(2,3) = {d12:.4f}")
except Exception as e:
    print(f"   ❌ ERRO: {e}")

# Teste 2: OpenAI Embeddings (API, pago)
print("\n🧪 TESTE 2: text-embedding-3-large (3072D, OpenAI)")

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    print("   ⚠️  PULADO: OPENAI_API_KEY nao encontrada")
else:
    try:
        embedder_openai = get_embedder("text-embedding-3-large")
        embeddings_openai = embed_texts(embedder_openai, texts)
        print(f"   ✅ Embeddings gerados: {embeddings_openai.shape}")
        
        # Calcular distancias
        d01 = cosine_distance(embeddings_openai[0], embeddings_openai[1])
        d02 = cosine_distance(embeddings_openai[0], embeddings_openai[2])
        d12 = cosine_distance(embeddings_openai[1], embeddings_openai[2])
        
        print(f"   📊 Distancias:")
        print(f"      d(1,2) = {d01:.4f}")
        print(f"      d(1,3) = {d02:.4f}")
        print(f"      d(2,3) = {d12:.4f}")
    except Exception as e:
        print(f"   ❌ ERRO: {e}")

# Teste 3: OpenAI text-embedding-3-small (mais barato)
print("\n🧪 TESTE 3: text-embedding-3-small (1536D, OpenAI)")

if not openai_key:
    print("   ⚠️  PULADO: OPENAI_API_KEY nao encontrada")
else:
    try:
        embedder_openai_small = get_embedder("text-embedding-3-small")
        embeddings_openai_small = embed_texts(embedder_openai_small, texts)
        print(f"   ✅ Embeddings gerados: {embeddings_openai_small.shape}")
        
        # Calcular distancias
        d01 = cosine_distance(embeddings_openai_small[0], embeddings_openai_small[1])
        d02 = cosine_distance(embeddings_openai_small[0], embeddings_openai_small[2])
        d12 = cosine_distance(embeddings_openai_small[1], embeddings_openai_small[2])
        
        print(f"   📊 Distancias:")
        print(f"      d(1,2) = {d01:.4f}")
        print(f"      d(1,3) = {d02:.4f}")
        print(f"      d(2,3) = {d12:.4f}")
    except Exception as e:
        print(f"   ❌ ERRO: {e}")

print("\n" + "=" * 70)
print("CONCLUSÃO")
print("=" * 70)
print("""
✅ Embeddings OpenAI:
   - Maior dimensionalidade (3072D vs 384D)
   - Captura nuances sutis
   - Melhor para convergencia

⚠️  Custo:
   - text-embedding-3-large: ~$0.13/1M tokens
   - text-embedding-3-small: ~$0.02/1M tokens
   - Para 55 ideias (~13K tokens): < $0.01

🎯 Recomendacao:
   - Teste com text-embedding-3-large primeiro
   - Se convergir, o problema era a metrica!
   - Se nao convergir, o problema e o feedback
""")

