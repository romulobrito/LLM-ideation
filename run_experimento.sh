#!/bin/bash
# Script para iniciar a interface de experimento na porta 8502

cd "$(dirname "$0")"

# Ativar ambiente virtual se existir
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🧪 Iniciando interface de experimento na porta 8502..."
echo "📍 Acesse: http://localhost:8502"
echo ""
echo "Para parar: Ctrl+C"
echo ""

python -m streamlit run app_experimento.py --server.port 8502
