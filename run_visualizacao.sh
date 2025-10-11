#!/bin/bash
# Script para iniciar a visualizacao do experimento na porta 8503

cd "$(dirname "$0")"

# Ativar ambiente virtual se existir
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "📊 Iniciando visualizacao do experimento na porta 8503..."
echo "📍 Acesse: http://localhost:8503"
echo ""
echo "Para parar: Ctrl+C"
echo ""

python -m streamlit run visualizar_experimento.py --server.port 8503
