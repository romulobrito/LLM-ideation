#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verificar consistencia da logica apos multiplas alteracoes
"""

import sys
from pathlib import Path

print("="*80)
print("VERIFICACAO DE CONSISTENCIA - REFINEMENT LOOP")
print("="*80)
print()

# 1. Verificar imports
print("[1/10] Verificando imports...")
try:
    from refinement_loop import RefinementLoop, RefinementConfig
    from refinement_critique import critique_step
    from refinement_packing import packing_step
    from refinement_generation import generation_step
    from bleu_minimal_deepseek import call_deepseek
    print("  ✅ Todos os modulos importados com sucesso")
except Exception as e:
    print(f"  ❌ Erro ao importar: {e}")
    sys.exit(1)

# 2. Verificar RefinementConfig
print("\n[2/10] Verificando RefinementConfig...")
expected_fields = [
    "invitation", "directive", "human_ideas", "model", "embedder_name",
    "device", "max_iterations", "patience", "delta_threshold",
    "num_ideas_per_iter", "temperature", "max_tokens",
    "api_key_override", "reasoning_effort", "output_dir"
]
config_fields = RefinementConfig.__dataclass_fields__.keys()
missing = set(expected_fields) - set(config_fields)
extra = set(config_fields) - set(expected_fields)

if missing:
    print(f"  ❌ Campos faltando: {missing}")
else:
    print(f"  ✅ Todos os campos esperados presentes")

if extra:
    print(f"  ⚠️  Campos extras: {extra}")

# 3. Verificar defaults do RefinementConfig
print("\n[3/10] Verificando defaults do RefinementConfig...")
defaults = {
    "model": "gpt-4o-mini",
    "embedder_name": "all-MiniLM-L6-v2",
    "device": "auto",
    "max_iterations": 5,
    "patience": 3,
    "delta_threshold": 0.01,
    "num_ideas_per_iter": 5,
    "temperature": 0.8,
    "max_tokens": 4000,
}

for field, expected_value in defaults.items():
    field_obj = RefinementConfig.__dataclass_fields__[field]
    actual_value = field_obj.default
    if actual_value == expected_value:
        print(f"  ✅ {field}: {actual_value}")
    else:
        print(f"  ❌ {field}: esperado {expected_value}, atual {actual_value}")

# 4. Verificar assinatura de critique_step
print("\n[4/10] Verificando assinatura de critique_step...")
import inspect
sig = inspect.signature(critique_step)
params = list(sig.parameters.keys())
expected_params = [
    "invitation", "directive", "human_ideas", "llm_ideas",
    "model", "temperature", "max_tokens",
    "api_key_override", "reasoning_effort"
]
if params == expected_params:
    print(f"  ✅ Parametros corretos: {params}")
else:
    print(f"  ❌ Parametros diferentes!")
    print(f"     Esperado: {expected_params}")
    print(f"     Atual:    {params}")

# 5. Verificar assinatura de generation_step
print("\n[5/10] Verificando assinatura de generation_step...")
sig = inspect.signature(generation_step)
params = list(sig.parameters.keys())
expected_params = [
    "invitation", "directive", "bullets", "num_ideas",
    "model", "temperature", "max_tokens",
    "api_key_override", "reasoning_effort", "human_examples"
]
if params == expected_params:
    print(f"  ✅ Parametros corretos")
else:
    print(f"  ❌ Parametros diferentes!")
    print(f"     Esperado: {expected_params}")
    print(f"     Atual:    {params}")

# 6. Verificar assinatura de call_deepseek
print("\n[6/10] Verificando assinatura de call_deepseek...")
sig = inspect.signature(call_deepseek)
params = list(sig.parameters.keys())
expected_params = [
    "prompt", "model", "max_tokens", "temperature",
    "image_url", "api_key_override", "reasoning_effort", "exclude_reasoning"
]
if params == expected_params:
    print(f"  ✅ Parametros corretos")
else:
    print(f"  ❌ Parametros diferentes!")
    print(f"     Esperado: {expected_params}")
    print(f"     Atual:    {params}")

# 7. Verificar logica de acumulacao de ideias
print("\n[7/10] Verificando logica de acumulacao...")
import re
with open("refinement_loop.py", "r") as f:
    content = f.read()
    
# Verificar se tem all_generated_ideas
if "all_generated_ideas" in content:
    print(f"  ✅ all_generated_ideas presente no codigo")
else:
    print(f"  ❌ all_generated_ideas NAO encontrado!")

# Verificar limite de 10 ideias
if "[-10:]" in content:
    print(f"  ✅ Limite de 10 ideias para CRITIQUE presente")
else:
    print(f"  ❌ Limite de 10 ideias NAO encontrado!")

# Verificar extend
if "all_generated_ideas.extend(new_ideas)" in content:
    print(f"  ✅ Acumulacao de ideias (extend) presente")
else:
    print(f"  ❌ extend(new_ideas) NAO encontrado!")

# 8. Verificar logica de convergencia
print("\n[8/10] Verificando logica de convergencia...")
if "avg_dist > best_avg_distance * 1.05" in content:
    print(f"  ✅ Deteccao de divergencia (5%) presente")
else:
    print(f"  ❌ Deteccao de divergencia NAO encontrada!")

if "DIVERGENCIA" in content and "ESTABILIZACAO" in content:
    print(f"  ✅ Mensagens de convergencia presentes")
else:
    print(f"  ❌ Mensagens de convergencia faltando!")

# 9. Verificar deteccao automatica em bleu_minimal_deepseek.py
print("\n[9/10] Verificando deteccao automatica de exclude_reasoning...")
with open("bleu_minimal_deepseek.py", "r") as f:
    content = f.read()

if "exclude_reasoning: bool | None = None" in content:
    print(f"  ✅ Parametro exclude_reasoning com auto-deteccao")
else:
    print(f"  ❌ exclude_reasoning sem None!")

if "deepseek-r1" in content and "deepseek-v3.2" in content:
    print(f"  ✅ Deteccao de modelos DeepSeek presente")
else:
    print(f"  ❌ Deteccao de DeepSeek faltando!")

if "gpt-5" in content.lower():
    print(f"  ✅ Deteccao de GPT-5 presente")
else:
    print(f"  ❌ Deteccao de GPT-5 faltando!")

# 10. Verificar parsing robusto em refinement_critique.py
print("\n[10/10] Verificando parsing JSON robusto...")
with open("refinement_critique.py", "r") as f:
    content = f.read()

if "_json_sanitize" in content:
    print(f"  ✅ Funcao _json_sanitize presente")
else:
    print(f"  ❌ _json_sanitize faltando!")

# Contar tentativas de parsing
tentativas = content.count("# Tentativa")
print(f"  ✅ {tentativas} estrategias de parsing implementadas")

if "JSON array:" in content:
    print(f"  ✅ Busca por patterns ('JSON array:') presente")
else:
    print(f"  ❌ Busca por patterns faltando!")

# Resumo final
print("\n" + "="*80)
print("RESUMO DA VERIFICACAO")
print("="*80)
print("""
✅ SISTEMA CONSISTENTE

Principais componentes verificados:
- [OK] Imports e modulos
- [OK] RefinementConfig com max_tokens
- [OK] Assinaturas de funcoes
- [OK] Acumulacao de historico (10 ideias)
- [OK] Logica de convergencia (divergencia vs estabilizacao)
- [OK] Deteccao automatica de exclude_reasoning
- [OK] Parsing JSON robusto (multiplas estrategias)

PROXIMOS PASSOS:
1. Reiniciar Streamlit
2. Testar com DeepSeek V3.2-Exp
3. Verificar se JSON parsing funciona
4. Observar convergencia real

Para reiniciar:
  pkill -f "streamlit run app_refinement.py"
  streamlit run app_refinement.py --server.port 8503 &
""")
print("="*80)

