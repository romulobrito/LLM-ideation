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
    page_icon="",
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
st.title(" Experimento Iterativo - Interface Dedicada")
st.markdown("---")

# Sidebar com configurações
with st.sidebar:
    st.header(" Configurações do Experimento")
    
    st.subheader(" Referências")
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
        
        # Contar número de referências no arquivo
        total_refs_available = 0
        if refs_path and Path(refs_path).exists():
            try:
                with open(refs_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    num_separators = content.count('---')
                    total_refs_available = num_separators + 1 if num_separators > 0 else 0
            except:
                total_refs_available = 0
        
        # Slider para limitar número de referências
        if total_refs_available > 0:
            max_refs_to_use = st.slider(
                "Número máximo de referências a processar:",
                min_value=1,
                max_value=total_refs_available,
                value=total_refs_available,
                help=f"Total disponível: {total_refs_available} referências"
            )
            st.caption(f" {max_refs_to_use} de {total_refs_available} referências serão processadas")
        else:
            max_refs_to_use = None
            
    elif fonte_refs == "Pasta":
        refs_folder = st.text_input(
            "Caminho da pasta:",
            value="/home/romulo/Documentos/MAI-DAI-USP/ideas-exp"
        )
        refs_path = refs_folder
        refs_raw = None
        max_refs_to_use = None  # Não aplicável para pasta
    else:
        refs_raw = st.text_area(
            "Cole as referências (separadas por ---)",
            height=150
        )
        refs_path = None
        max_refs_to_use = None  # Não aplicável para texto direto
    
    st.markdown("---")
    st.subheader(" Modelo LLM")
    
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
    st.subheader(" Embeddings")
    
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
    st.subheader(" Iteração")
    
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
    st.subheader(" Saída")
    
    out_dir_exp = st.text_input(
        "Diretório de saída:",
        value="exp_out"
    )
    
    clean_before = st.checkbox(
        " Limpar diretório antes de iniciar",
        value=False,
        help="Remove TODOS os arquivos do diretório de saída antes de rodar o experimento"
    )

# Área principal
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Status do Experimento")
    
    # Verificar se há chave API
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    has_openrouter_key = bool(os.getenv("OPENROUTER_API_KEY")) or bool(os.getenv("OPENROUTER_API_KEY_OPENAI")) or bool(os.getenv("OPENROUTER_API_KEY_DEEPSEEK"))
    
    if "/" in model_name:
        can_run = has_openrouter_key
        if not can_run:
            st.error(" Nenhuma chave OpenRouter encontrada no .env")
    else:
        can_run = has_openai_key
        if not can_run:
            st.error(" OPENAI_API_KEY não encontrada no .env")
    
    if can_run:
        st.success(" Chave API configurada corretamente")

# Inicializar session_state ANTES de usar
if 'process' not in st.session_state:
    st.session_state.process = None
if 'running' not in st.session_state:
    st.session_state.running = False
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False

with col2:
    st.header(" Ações")
    
    # Detectar progresso do experimento
    def analyze_experiment_progress(out_dir, refs_file):
        """Analisa o progresso do experimento e retorna estatísticas."""
        import pandas as pd
        
        # Contar total de referências no arquivo
        total_refs_in_file = 0
        if refs_file and Path(refs_file).exists():
            try:
                with open(refs_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Contar separadores "---"
                    # Número de referências = separadores + 1
                    num_separators = content.count('---')
                    total_refs_in_file = num_separators + 1 if num_separators > 0 else 0
            except:
                pass
        
        out_path = Path(out_dir)
        if not out_path.exists():
            return 1 if total_refs_in_file > 0 else None, 0, total_refs_in_file
        
        ref_folders = sorted(out_path.glob("ref_*"))
        if not ref_folders and total_refs_in_file > 0:
            return 1, 0, total_refs_in_file  # Nenhuma iniciada, começar da ref_001
        
        completed = 0
        first_incomplete = None
        max_ref_id = 0
        
        for ref_path in ref_folders:
            ref_id = int(ref_path.name.split("_")[1])
            max_ref_id = max(max_ref_id, ref_id)
            log_file = ref_path / "log.csv"
            
            is_complete = False
            if log_file.exists():
                try:
                    df = pd.read_csv(log_file)
                    # Considerar completa se tem >= 10 iterações (critério fixo)
                    if len(df) > 0 and df['iter'].max() >= 10:
                        is_complete = True
                        completed += 1
                except:
                    pass
            
            if not is_complete and first_incomplete is None:
                first_incomplete = ref_id
        
        # Se todas as existentes estão completas, mas ainda há refs no arquivo
        if first_incomplete is None and total_refs_in_file > max_ref_id:
            first_incomplete = max_ref_id + 1
        
        # Usar o total do arquivo se disponível, senão usar o número de pastas
        total_refs = total_refs_in_file if total_refs_in_file > 0 else len(ref_folders)
        
        return first_incomplete, completed, total_refs
    
    refs_file_path = refs_path if refs_path else "/home/romulo/Documentos/MAI-DAI-USP/refs_combined.txt"
    last_incomplete, num_completed, total_refs = analyze_experiment_progress(out_dir_exp, refs_file_path)
    
    # Mostrar progresso se houver experimento anterior
    if last_incomplete and num_completed is not None and total_refs is not None:
        progress_pct = (num_completed / total_refs) * 100 if total_refs > 0 else 0
        st.info(f" Progresso detectado: {num_completed}/{total_refs} refs completas ({progress_pct:.1f}%) | Próxima: ref_{last_incomplete:03d}")
    
    # Botões principais
    if last_incomplete:
        col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 1])
    else:
        col_btn1, col_btn2 = st.columns([2, 1])
    
    with col_btn1:
        run_button = st.button(
            " INICIAR DO ZERO",
            type="primary",
            disabled=not can_run or st.session_state.running,
            use_container_width=True
        )
    
    if last_incomplete:
        with col_btn2:
            continue_button = st.button(
                f" RETOMAR ({num_completed}/{total_refs})",
                type="primary",
                disabled=not can_run or st.session_state.running,
                use_container_width=True,
                help=f"Continuar de onde parou: ref_{last_incomplete:03d} ({num_completed} completas, {total_refs - num_completed} restantes)"
            )
    else:
        continue_button = False
    
    with (col_btn3 if last_incomplete else col_btn2):
        if st.session_state.running and st.session_state.process:
            if st.button(" PARAR", type="secondary", use_container_width=True, key="stop_btn"):
                try:
                    st.session_state.process.terminate()
                    st.session_state.process.wait(timeout=2)
                except:
                    st.session_state.process.kill()
                st.session_state.process = None
                st.session_state.running = False
                st.warning(" Experimento interrompido!")
                st.rerun()
        else:
            if st.button(" Limpar logs", use_container_width=True):
                st.rerun()

st.markdown("---")

# Área de logs
log_area = st.container()

# Executar experimento
if run_button or continue_button:
    with log_area:
        if continue_button:
            st.info(f" Continuando experimento a partir de ref_{last_incomplete:03d}...")
        else:
            st.info(" Preparando experimento...")
        
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
            st.error(f" Script não encontrado: {script_path}")
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
        
        # Adicionar --max-refs se disponível (apenas para fonte "Arquivo")
        if fonte_refs == "Arquivo" and max_refs_to_use:
            cmd.extend(["--max-refs", str(max_refs_to_use)])
        
        # Adicionar flag --clean se checkbox marcado (apenas para INICIAR DO ZERO)
        if clean_before and run_button:
            cmd.append("--clean")
        
        # Adicionar --start-from-ref se for CONTINUAR
        if continue_button and last_incomplete:
            cmd.extend(["--start-from-ref", str(last_incomplete)])
        
        if continue_button:
            st.info(f" Continuando de ref_{last_incomplete:03d}... Acompanhe o progresso abaixo:")
        else:
            st.info(" Iniciando experimento... Acompanhe o progresso abaixo:")
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
            
            # Armazenar processo no session_state
            st.session_state.process = process
            st.session_state.running = True
            st.session_state.stop_requested = False
            
            output_lines = []
            
            # Criar placeholder para botão de parar
            stop_placeholder = st.empty()
            
            # Ler saída em tempo real
            with status_container:
                with st.spinner(" Executando experimento..."):
                    # Mostrar botão de parar
                    with stop_placeholder.container():
                        if st.button(" PARAR EXPERIMENTO", type="secondary", key="stop_during_exec"):
                            st.session_state.stop_requested = True
                    
                    for line in process.stdout:
                        # Verificar se foi solicitada parada
                        if st.session_state.stop_requested:
                            st.warning(" Parando experimento...")
                            try:
                                process.terminate()
                                process.wait(timeout=3)
                            except:
                                process.kill()
                            break
                        
                        line_clean = line.rstrip()
                        if line_clean:
                            output_lines.append(line_clean)
                            # Mostrar últimas 40 linhas
                            display_text = "\n".join(output_lines[-40:])
                            log_container.code(display_text, language="text")
            
            # Limpar placeholder do botão
            stop_placeholder.empty()
            
            # Aguardar conclusão
            if not st.session_state.stop_requested:
                process.wait()
            
            # Limpar estado
            st.session_state.process = None
            st.session_state.running = False
            st.session_state.stop_requested = False
            
            # Capturar stderr
            stderr_output = process.stderr.read()
            
            if st.session_state.stop_requested or process.returncode == -15:  # SIGTERM
                st.warning(" Experimento interrompido pelo usuário!")
                if output_lines:
                    with st.expander("Log Parcial", expanded=False):
                        st.code("\n".join(output_lines), language="text")
            elif process.returncode != 0:
                st.error(f" Falha ao executar experimento (código {process.returncode})")
                if stderr_output:
                    with st.expander(" Erro detalhado"):
                        st.code(stderr_output, language="text")
            else:
                st.success(f" Experimento concluído com sucesso!")
                st.success(f" Resultados salvos em: `{out_dir_exp}/`")
                st.balloons()
                
                # Mostrar log completo
                if output_lines:
                    with st.expander("Log Completo", expanded=False):
                        st.code("\n".join(output_lines), language="text")
                
                # Botão para baixar log
                log_text = "\n".join(output_lines)
                st.download_button(
                    label=" Baixar log completo",
                    data=log_text,
                    file_name=f"experimento_log_{out_dir_exp}.txt",
                    mime="text/plain"
                )
                        
        except Exception as e:
            st.error(f" Erro ao iniciar experimento: {e}")
            import traceback
            with st.expander(" Traceback completo"):
                st.code(traceback.format_exc())
        finally:
            # Limpar estado
            st.session_state.process = None
            st.session_state.running = False
            
            # Remover arquivo temporário
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
     Interface de Experimento Iterativo | Porta 8502
</div>
""", unsafe_allow_html=True)
