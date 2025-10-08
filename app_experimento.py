#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface Streamlit dedicada para rodar experimentos iterativos.
Execute com: streamlit run app_experimento.py --server.port 8502
"""

import streamlit as st
import subprocess
import sys
import os
import tempfile
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="Experimento Iterativo LLM",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

def _load_env_robusto():
    """Carrega .env de forma robusta."""
    from pathlib import Path
    try_paths = [
        Path(__file__).parent / ".env",
        Path.cwd() / ".env",
        Path.home() / "Documentos" / "MAI-DAI-USP" / ".env",
    ]
    for p in try_paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and v:
                            os.environ[k] = v
            return True
    return False

# Carregar variáveis de ambiente
_load_env_robusto()

# Título principal
st.title("🧪 Experimento Iterativo - Interface Dedicada")
st.markdown("---")

# Sidebar com configurações
with st.sidebar:
    st.header("⚙️ Configurações do Experimento")
    
    st.subheader("📄 Referências")
    fonte_refs = st.radio(
        "Fonte das referências:",
        ["Arquivo", "Pasta", "Texto direto"],
        index=0
    )
    
    if fonte_refs == "Arquivo":
        refs_path = st.text_input(
            "Caminho do arquivo:",
            value="/home/romulo/Documentos/MAI-DAI-USP/refs_combined.txt"
        )
        refs_raw = None
    elif fonte_refs == "Pasta":
        refs_folder = st.text_input(
            "Caminho da pasta:",
            value="/home/romulo/Documentos/MAI-DAI-USP/ideas-exp"
        )
        refs_path = refs_folder
        refs_raw = None
    else:
        refs_raw = st.text_area(
            "Cole as referências (separadas por ---)",
            height=150
        )
        refs_path = None
    
    st.markdown("---")
    st.subheader("🤖 Modelo LLM")
    
    provider = st.selectbox(
        "Provedor:",
        ["OpenAI Direto", "DeepSeek (OpenRouter)", "GPT-5 (OpenRouter)", "Personalizado"],
        index=0
    )
    
    if provider == "OpenAI Direto":
        model_name = st.selectbox(
            "Modelo:",
            ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0
        )
    elif provider == "DeepSeek (OpenRouter)":
        model_name = "deepseek/deepseek-chat"
    elif provider == "GPT-5 (OpenRouter)":
        model_name = "openai/gpt-5"
    else:
        model_name = st.text_input("Nome do modelo:", value="gpt-4o-mini")
    
    temperature = st.slider("Temperatura:", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.number_input("Max tokens:", 100, 4000, 800, 50)
    
    reasoning_effort = st.selectbox(
        "Reasoning effort:",
        ["Nenhum", "low", "medium", "high"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("🧠 Embeddings")
    
    embedder_exp = st.selectbox(
        "Modelo de embedding:",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"],
        index=0
    )
    
    device_exp = st.selectbox(
        "Device:",
        ["auto", "cuda", "cpu"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("🔄 Iteração")
    
    max_iters_exp = st.number_input(
        "Máximo de iterações:",
        min_value=1,
        max_value=100,
        value=30,
        step=1
    )
    
    patience_exp = st.number_input(
        "Paciência (early stop):",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )
    
    delta_exp = st.number_input(
        "Delta mínimo:",
        min_value=0.0,
        max_value=0.1,
        value=0.005,
        step=0.001,
        format="%.4f"
    )
    
    st.markdown("---")
    st.subheader("📂 Saída")
    
    out_dir_exp = st.text_input(
        "Diretório de saída:",
        value="exp_out"
    )
    
    clean_before = st.checkbox(
        "🗑️ Limpar diretório antes de iniciar",
        value=False,
        help="Remove TODOS os arquivos do diretório de saída antes de rodar o experimento"
    )

# Área principal
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Status do Experimento")
    
    # Verificar se há chave API
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    has_openrouter_key = bool(os.getenv("OPENROUTER_API_KEY")) or bool(os.getenv("OPENROUTER_API_KEY_OPENAI")) or bool(os.getenv("OPENROUTER_API_KEY_DEEPSEEK"))
    
    if "/" in model_name:
        can_run = has_openrouter_key
        if not can_run:
            st.error("❌ Nenhuma chave OpenRouter encontrada no .env")
    else:
        can_run = has_openai_key
        if not can_run:
            st.error("❌ OPENAI_API_KEY não encontrada no .env")
    
    if can_run:
        st.success("✅ Chave API configurada corretamente")

with col2:
    st.header("🎯 Ações")
    
    # Botão principal
    run_button = st.button(
        "🚀 INICIAR EXPERIMENTO",
        type="primary",
        disabled=not can_run,
        use_container_width=True
    )
    
    if st.button("🔄 Limpar logs", use_container_width=True):
        st.rerun()

st.markdown("---")

# Área de logs
log_area = st.container()

# Executar experimento
if run_button:
    with log_area:
        st.info("🔧 Preparando experimento...")
        
        # Preparar argumentos
        tmp_file = None
        if refs_raw:
            # Criar arquivo temporário
            tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8")
            tmp.write(refs_raw or "")
            tmp.flush()
            tmp.close()
            refs_arg = tmp.name
            tmp_file = tmp.name
        else:
            refs_arg = refs_path or "/home/romulo/Documentos/MAI-DAI-USP/refs_combined.txt"
        
        reasoning_arg = reasoning_effort if reasoning_effort != "Nenhum" else "None"
        model_arg = model_name.strip() or "gpt-4o-mini"
        
        # Caminho para o script
        script_path = Path(__file__).parent / "experiment_iterativo.py"
        
        if not script_path.exists():
            st.error(f"❌ Script não encontrado: {script_path}")
            st.stop()
        
        # Comando
        cmd = [
            sys.executable,
            str(script_path),
            "--refs-path", refs_arg,
            "--out-dir", out_dir_exp,
            "--model", model_arg,
            "--reasoning", reasoning_arg,
            "--temperature", str(temperature),
            "--max-tokens", str(max_tokens),
            "--embedder", embedder_exp,
            "--device", device_exp,
            "--max-iters", str(int(max_iters_exp)),
            "--patience", str(int(patience_exp)),
            "--delta", str(float(delta_exp)),
        ]
        
        # Adicionar flag --clean se checkbox marcado
        if clean_before:
            cmd.append("--clean")
        
        st.info("🚀 Iniciando experimento... Acompanhe o progresso abaixo:")
        st.code(" ".join(cmd), language="bash")
        
        # Container para logs
        log_container = st.empty()
        status_container = st.empty()
        
        try:
            # Executar com streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=str(Path(__file__).parent)
            )
            
            output_lines = []
            
            # Ler saída em tempo real
            with status_container:
                with st.spinner("⏳ Executando experimento..."):
                    for line in process.stdout:
                        line_clean = line.rstrip()
                        if line_clean:
                            output_lines.append(line_clean)
                            # Mostrar últimas 40 linhas
                            display_text = "\n".join(output_lines[-40:])
                            log_container.code(display_text, language="text")
            
            # Aguardar conclusão
            process.wait()
            
            # Capturar stderr
            stderr_output = process.stderr.read()
            
            if process.returncode != 0:
                st.error(f"❌ Falha ao executar experimento (código {process.returncode})")
                if stderr_output:
                    with st.expander("📋 Erro detalhado"):
                        st.code(stderr_output, language="text")
            else:
                st.success(f"✅ Experimento concluído com sucesso!")
                st.success(f"📂 Resultados salvos em: `{out_dir_exp}/`")
                st.balloons()
                
                # Mostrar log completo
                if output_lines:
                    with st.expander("📊 Log Completo", expanded=False):
                        st.code("\n".join(output_lines), language="text")
                
                # Botão para baixar log
                log_text = "\n".join(output_lines)
                st.download_button(
                    label="💾 Baixar log completo",
                    data=log_text,
                    file_name=f"experimento_log_{out_dir_exp}.txt",
                    mime="text/plain"
                )
                        
        except Exception as e:
            st.error(f"❌ Erro ao iniciar experimento: {e}")
            import traceback
            with st.expander("📋 Traceback completo"):
                st.code(traceback.format_exc())
        finally:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    🧪 Interface de Experimento Iterativo | Porta 8502
</div>
""", unsafe_allow_html=True)
