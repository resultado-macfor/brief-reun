import os
import streamlit as st
import tempfile
from datetime import datetime
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import PyPDF2
import docx
import json
import re
import time

# ============================================================================
# CONFIGURAÇÃO INICIAL
# ============================================================================
st.set_page_config(
    page_title="Analisador de Reuniões IA",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURAÇÃO DA API GEMINI
# ============================================================================
gemini_api_key = os.getenv("GEM_API_KEY")

if not gemini_api_key:
    st.error("❌ API key do Gemini não encontrada. Configure a variável de ambiente GEM_API_KEY.")
    st.stop()

# ============================================================================
# AUTENTICAÇÃO SIMPLES
# ============================================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Se não estiver autenticado, mostra a tela de login
if not st.session_state.authenticated:
    # Layout centralizado para a tela de login
    st.set_page_config(layout="centered")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🔒 Analisador de Reuniões IA")
        st.markdown("---")
        
        # Card de login
        with st.container():
            st.markdown(
                """
                <style>
                .login-card {
                    background: white;
                    padding: 2rem;
                    border-radius: 10px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown('<div class="login-card">', unsafe_allow_html=True)
            
            senha_input = st.text_input(
                "**Digite a senha de acesso:**",
                type="password",
                placeholder="Digite a senha aqui...",
                key="senha_input"
            )
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button(
                    "🔓 Acessar Sistema",
                    type="primary",
                    use_container_width=True,
                    key="btn_login"
                ):
                    senha_correta = os.getenv('senha_per')
                    
                    if not senha_correta:
                        st.error("⚠️ Sistema não configurado. Verifique a variável de ambiente 'senha_per'.")
                        st.stop()
                    elif senha_input == senha_correta:
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("❌ Senha incorreta. Tente novamente.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Informações adicionais
            st.markdown("---")
            st.caption(
                """
                **Sistema de Análise de Reuniões com IA**  
                Para acessar, digite a senha fornecida pelo administrador.
                """
            )
    
    st.stop()

# ============================================================================
# CONFIGURAÇÃO DO GEMINI (só executa se autenticado)
# ============================================================================
try:
    genai.configure(api_key=gemini_api_key)
    # Gemini 1.5 Flash para vídeo/análise multimodal
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    st.success("✅ Gemini 1.5 Flash configurado com sucesso!")
except Exception as e:
    st.error(f"❌ Erro ao configurar Gemini: {str(e)}")
    st.stop()

# ============================================================================
# FUNÇÕES DE PROCESSAMENTO DE ARQUIVOS
# ============================================================================
def extract_text_from_pdf(file):
    """Extrai texto de PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Erro ao extrair texto do PDF: {str(e)}"

def extract_text_from_docx(file):
    """Extrai texto de DOCX"""
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        return f"Erro ao extrair texto do DOCX: {str(e)}"

def extract_text_from_txt(file):
    """Extrai texto de TXT"""
    try:
        return file.read().decode("utf-8")
    except:
        try:
            file.seek(0)
            return file.read().decode("latin-1")
        except Exception as e:
            return f"Erro ao extrair texto do TXT: {str(e)}"

def extract_text_from_file(file):
    """Extrai texto de qualquer arquivo suportado"""
    filename = file.name.lower()
    
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif filename.endswith('.docx'):
        return extract_text_from_docx(file)
    elif filename.endswith('.txt'):
        return extract_text_from_txt(file)
    else:
        return "Formato de arquivo não suportado"

# ============================================================================
# FUNÇÕES DE ANÁLISE DE VÍDEO COM GEMINI (CORRIGIDAS)
# ============================================================================
def upload_and_wait_for_processing(video_path, max_retries=10, delay_seconds=5):
    """Faz upload do vídeo e espera até estar processado"""
    try:
        # Fazer upload do arquivo
        st.info("📤 Fazendo upload do vídeo para o Gemini...")
        video_file = genai.upload_file(video_path)
        
        # Verificar estado do arquivo
        retries = 0
        while retries < max_retries:
            try:
                status_response = genai.get_file(video_file.name)
                status = status_response.state.name
                
                if status == "ACTIVE":
                    st.success("✅ Vídeo processado e pronto para análise!")
                    return video_file
                elif status == "FAILED":
                    st.error("❌ Falha no processamento do vídeo")
                    return None
                else:
                    st.info(f"⏳ Processando vídeo... ({status})")
                    time.sleep(delay_seconds)
                    retries += 1
                    
            except Exception as e:
                st.warning(f"⚠️ Aguardando processamento... (tentativa {retries + 1}/{max_retries})")
                time.sleep(delay_seconds)
                retries += 1
        
        st.error("❌ Tempo esgotado no processamento do vídeo")
        return None
        
    except Exception as e:
        st.error(f"❌ Erro no upload do vídeo: {str(e)}")
        return None

def analyze_video_with_gemini(video_path, meeting_info=None):
    """Analisa vídeo de reunião usando Gemini 1.5 Flash"""
    
    try:
        # Configurar safety settings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        
        # Prompt para análise de vídeo
        system_prompt = """Você é um especialista em análise de reuniões corporativas com background em psicologia organizacional, gestão de projetos e comunicação eficaz. 
        
        Analise este vídeo de reunião considerando:
        1. Conteúdo verbal (transcrição do que é dito)
        2. Tom de voz e entonação
        3. Dinâmica entre participantes quando identificável
        4. Estrutura da reunião
        5. Clareza das comunicações
        
        Forneça uma análise completa, prática e baseada em evidências observáveis."""
        
        # Upload e espera pelo processamento
        video_file = upload_and_wait_for_processing(video_path)
        
        if not video_file:
            return "❌ Falha no processamento do vídeo. Tente novamente com um vídeo menor ou formato diferente."
        
        # Preparar prompt
        prompt_parts = [
            system_prompt,
            "\nANALISE ESTE VÍDEO DE REUNIÃO CORPORATIVA:",
            video_file,
        ]
        
        if meeting_info:
            prompt_parts.append(f"\nINFORMAÇÕES ADICIONAIS DA REUNIÃO:\n{meeting_info}")
        
        prompt_parts.append("""
        
        FORNECER ANÁLISE NO SEGUINTE FORMATO DETALHADO:
        
        # 🎥 ANÁLISE DE VÍDEO DE REUNIÃO
        
        ## 📋 RESUMO EXECUTIVO
        [Resumo de 2-3 parágrafos com os pontos mais importantes da reunião]
        
        ## 🗣️ TRANSCRIÇÃO DOS PONTOS PRINCIPAIS
        [Transcrição dos momentos mais importantes discutidos - foco no conteúdo]
        
        ## 👥 PARTICIPANTES E DINÂMICA
        ### Participantes Identificados:
        - [Liste os participantes quando identificáveis]
        
        ### Análise da Dinâmica:
        - **Clima geral:** [positivo, tenso, neutro, colaborativo, etc.]
        - **Interações principais:** [como os participantes interagiram]
        - **Tom predominante:** [formal, informal, técnico, etc.]
        - **Ritmo da reunião:** [rápido, moderado, lento, bem distribuído]
        
        ## 🔊 ANÁLISE DE COMUNICAÇÃO
        - **Clareza geral:** [nível de entendimento das comunicações]
        - **Tom de voz observado:** [entonações, ênfases, variações]
        - **Momentos-chave pela comunicação:** [momentos importantes pela forma como foram comunicados]
        
        ## 🎯 CONTEÚDO E DECISÕES
        ### Tópicos Principais Discutidos:
        1. [Tópico 1]
        2. [Tópico 2]
        3. [Tópico 3]
        
        ### Decisões Tomadas:
        - [Decisão 1]
        - [Decisão 2]
        
        ### Ações Acordadas:
        - [Ação 1] (Responsável: [se identificado], Prazo: [se mencionado])
        - [Ação 2] (Responsável: [se identificado], Prazo: [se mencionado])
        
        ## 🚨 PONTOS DE ATENÇÃO
        - [Lista de pontos que merecem atenção ou melhorias]
        
        ## 💡 RECOMENDAÇÕES
        - [Sugestões práticas para melhorias em próximas reuniões]
        
        ## ⭐ AVALIAÇÃO FINAL
        **Eficácia da comunicação:** X/10
        **Clareza das decisões:** X/10  
        **Engajamento observado:** X/10
        **Média Geral:** X/10
        
        ### Observações Técnicas do Vídeo:
        - Qualidade do áudio: [boa, média, ruim]
        - Qualidade da imagem: [boa, média, ruim]
        - Recomendações técnicas: [sugestões para melhor qualidade]
        """)
        
        # Gerar análise
        with st.spinner("🔍 Gemini está analisando o vídeo... Isso pode levar alguns minutos"):
            response = gemini_model.generate_content(
                prompt_parts,
                safety_settings=safety_settings,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )
        
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        if "File" in error_msg and "not in an ACTIVE state" in error_msg:
            return "❌ O vídeo ainda está sendo processado. Aguarde alguns instantes e tente novamente, ou use um vídeo menor."
        elif "size" in error_msg.lower():
            return "❌ O vídeo é muito grande. Tente com um vídeo menor ou divida-o em partes."
        else:
            return f"❌ Erro na análise do vídeo: {error_msg}"

def analyze_transcript_with_gemini(transcript, meeting_info=None):
    """Analisa transcrição de reunião usando Gemini"""
    
    system_prompt = """Você é um especialista em análise de reuniões corporativas com background em psicologia organizacional, gestão de projetos e comunicação eficaz. 
    Sua análise deve ser profunda, prática e baseada em evidências científicas."""
    
    prompt = f"""
    ANALISE ESTA TRANSCRIÇÃO DE REUNIÃO:
    
    {transcript[:15000]}
    
    INSTRUÇÕES PARA ANÁLISE:
    
    1. IDENTIFIQUE OS PARTICIPANTES:
    - Liste todos os participantes mencionados
    - Estime seus cargos/funções baseado no contexto
    - Quantifique participação de cada um
    
    2. ANÁLISE DO CONTEÚDO:
    - Principais tópicos discutidos
    - Decisões tomadas
    - Ações acordadas (com responsáveis e prazos quando possível)
    - Pontos de conflito ou desacordo
    - Momentos-chave da reunião
    
    3. ANÁLISE COMPORTAMENTAL:
    - Nível de participação de cada um
    - Tom de voz e linguagem utilizada
    - Grau de colaboração
    - Clareza na comunicação
    - Demonstração de liderança
    
    4. RED FLAGS E PONTOS DE ATENÇÃO:
    - Comunicação ineficaz
    - Falta de clareza em responsabilidades
    - Conflitos não resolvidos
    - Desalinhamento de expectativas
    
    5. EFICÁCIA DA REUNIÃO:
    - Objetivos atingidos?
    - Tempo bem utilizado?
    - Participação equilibrada?
    - Decisões claras e acionáveis?
    
    6. RECOMENDAÇÕES:
    - Pontos fortes a serem mantidos
    - Melhorias sugeridas
    - Treinamentos recomendados
    
    FORMATO DA RESPOSTA:
    
    # 📊 RELATÓRIO DE ANÁLISE DE REUNIÃO
    
    ## 1. 🧑‍🤝‍🧑 PARTICIPANTES IDENTIFICADOS
    [Lista detalhada]
    
    ## 2. 📋 RESUMO DA REUNIÃO
    [Resumo executivo]
    
    ## 3. 🎯 DECISÕES E AÇÕES
    [Tabela com ações, responsáveis e prazos]
    
    ## 4. 👥 ANÁLISE INDIVIDUAL
    ### Participante 1: [Nome]
    - Participação: [X%]
    - Comportamento: [análise]
    - Comunicação: [análise]
    - Contribuição: [análise]
    
    ## 5. 🚨 PONTOS DE ATENÇÃO
    [Lista com explicação]
    
    ## 6. 💡 RECOMENDAÇÕES
    [Lista detalhada]
    
    ## 7. ⭐ NOTA FINAL
    **Eficiência:** X/10
    **Satisfação:** X/10  
    **Qualidade:** X/10
    **Média:** X/10
    """
    
    if meeting_info:
        prompt = f"INFORMAÇÕES ADICIONAIS:\n{meeting_info}\n\n{prompt}"
    
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 8192,
            }
        )
        return response.text
    except Exception as e:
        return f"❌ Erro na análise: {str(e)}"

def extract_meeting_metadata(text):
    """Extrai metadados básicos da reunião"""
    
    prompt = f"""
    Extraia informações básicas desta reunião:
    
    {text[:5000]}
    
    Procure por:
    1. Data da reunião
    2. Horário
    3. Participantes presentes
    4. Objetivo da reunião
    
    Responda em formato JSON:
    """
    
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config={"temperature": 0.1}
        )
        
        # Tentar extrair JSON
        text_response = response.text
        json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    return {
        "date": "Não identificada",
        "time": "Não identificado",
        "participants": ["Participantes não identificados"],
        "objective": "Não identificado"
    }

# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================
def main_app():
    """Interface principal do aplicativo"""
    
    # Sidebar
    with st.sidebar:
        st.title("🎯 Analisador de Reuniões")
        st.markdown("---")
        
        # Navegação
        page = st.radio(
            "📌 Navegação",
            ["📁 Nova Análise", "⚙️ Configurações", "ℹ️ Sobre"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Informações do sistema
        st.markdown("**ℹ️ Informações do Sistema**")
        st.write(f"**Modelo:** Gemini 2.5 Flash")
        st.write(f"**Data:** {datetime.now().strftime('%d/%m/%Y')}")
        
        st.markdown("---")
        
        # Botão de logout
        if st.button("🚪 Sair do Sistema", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()
    
    # Página: Nova Análise
    if page == "📁 Nova Análise":
        st.title("🎯 Análise de Reuniões com Gemini")
        st.markdown("Faça upload de vídeo ou transcrição para análise detalhada")
        st.markdown("---")
        
        # Abas para diferentes tipos de entrada
        tab1, tab2 = st.tabs(["🎥 Análise de Vídeo", "📄 Transcrição"])
        
        # Tab 1: Análise de Vídeo
        with tab1:
            st.subheader("🎥 Análise de Vídeo de Reunião")
            
            st.info("""
            **Funcionalidades disponíveis:**
            - Análise completa de vídeos de reuniões
            - Transcrição automática do áudio
            - Análise de tom de voz e entonação
            - Observações sobre dinâmica do grupo
            - Suporta vídeos até 2GB (Gemini 1.5 Flash)
            - Formatos suportados: MP4, MOV, AVI, WMV, FLV, WebM
            """)
            
            # Upload de vídeo
            video_file = st.file_uploader(
                "Selecione o vídeo da reunião:",
                type=['mp4', 'mov', 'avi', 'mkv', 'webm', 'wmv', 'flv'],
                key="video_uploader"
            )
            
            if video_file:
                # Mostrar informações do vídeo
                file_size_mb = video_file.size / (1024 * 1024)
                st.success(f"✅ Vídeo carregado: {video_file.name} ({file_size_mb:.1f} MB)")
                
                # Pré-visualização do vídeo
                col_vid1, col_vid2 = st.columns([2, 1])
                with col_vid1:
                    st.video(video_file)
                with col_vid2:
                    st.info(f"""
                    **📊 Informações:**
                    - Nome: {video_file.name}
                    - Tamanho: {file_size_mb:.1f} MB
                    - Tipo: {video_file.type}
                    """)
                
                # Formulário de informações
                with st.expander("✏️ Informações da Reunião", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        meeting_date = st.date_input(
                            "Data da reunião:",
                            value=datetime.now(),
                            key="video_date"
                        )
                        meeting_time = st.time_input(
                            "Horário:",
                            value=datetime.now().time(),
                            key="video_time"
                        )
                    
                    with col2:
                        meeting_type = st.selectbox(
                            "Tipo de reunião:",
                            ["Brainstorming", "Reunião de Equipe", "Apresentação", 
                             "Revisão de Projeto", "One-on-One", "Decisão", "Status", "Outro"],
                            key="video_type"
                        )
                        participants = st.text_area(
                            "Participantes (opcional, um por linha):",
                            height=80,
                            placeholder="João Silva\nMaria Santos\nPedro Oliveira",
                            help="Liste os participantes para melhor análise",
                            key="video_participants"
                        )
                        
                        meeting_objective = st.text_area(
                            "Objetivo da reunião (opcional):",
                            height=60,
                            placeholder="Ex: Decidir sobre o lançamento do novo produto...",
                            key="video_objective"
                        )
                
                # Avisos sobre tamanho
                if file_size_mb > 100:
                    st.warning("⚠️ Vídeos grandes podem demorar mais para processar. Recomendamos vídeos menores que 100MB para análise mais rápida.")
                
                # Botão de análise
                if st.button("🔍 Analisar Vídeo", type="primary", use_container_width=True):
                    # Salvar vídeo temporariamente
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(video_file.read())
                        video_path = tmp_file.name
                    
                    try:
                        # Preparar informações
                        meeting_info = f"""
                        DATA: {meeting_date.strftime('%d/%m/%Y')}
                        HORÁRIO: {meeting_time.strftime('%H:%M')}
                        TIPO: {meeting_type}
                        PARTICIPANTES: {participants if participants else 'Não informados'}
                        OBJETIVO: {meeting_objective if meeting_objective else 'Não informado'}
                        """
                        
                        # Realizar análise
                        analysis = analyze_video_with_gemini(video_path, meeting_info)
                        
                        # Mostrar resultados
                        st.markdown("---")
                        st.subheader("📊 Resultado da Análise")
                        
                        # Container para resultados
                        with st.container():
                            st.markdown(analysis)
                        
                        # Opções de download
                        st.markdown("---")
                        st.subheader("📥 Exportar Resultados")
                        
                        col_dl1, col_dl2, col_dl3 = st.columns(3)
                        
                        with col_dl1:
                            st.download_button(
                                "💾 Baixar como TXT",
                                data=analysis,
                                file_name=f"analise_video_{meeting_date.strftime('%Y%m%d')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col_dl2:
                            # Extrair resumo
                            if "## 📋 RESUMO EXECUTIVO" in analysis:
                                start_idx = analysis.find("## 📋 RESUMO EXECUTIVO")
                                end_idx = analysis.find("##", start_idx + 1)
                                summary = analysis[start_idx:end_idx] if end_idx != -1 else analysis[start_idx:]
                                
                                st.download_button(
                                    "📋 Resumo Executivo",
                                    data=summary,
                                    file_name=f"resumo_video_{meeting_date.strftime('%Y%m%d')}.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                            else:
                                st.info("Resumo não disponível")
                        
                        with col_dl3:
                            # Criar ações em CSV
                            csv_data = "Ação,Responsável,Prazo,Status\n"
                            if "### Ações Acordadas:" in analysis:
                                # Extrair ações da análise
                                start_idx = analysis.find("### Ações Acordadas:")
                                end_idx = analysis.find("##", start_idx + 1)
                                actions_text = analysis[start_idx:end_idx] if end_idx != -1 else analysis[start_idx:]
                                
                                # Processar ações
                                lines = actions_text.split('\n')
                                for line in lines:
                                    if '-' in line and '(' in line:
                                        action = line.split('-')[1].split('(')[0].strip()
                                        rest = line.split('(')[1].replace(')', '')
                                        responsible = ""
                                        deadline = ""
                                        if 'Responsável:' in rest:
                                            responsible = rest.split('Responsável:')[1].split(',')[0].strip()
                                        if 'Prazo:' in rest:
                                            deadline = rest.split('Prazo:')[1].strip()
                                        csv_data += f'"{action}","{responsible}","{deadline}","Pendente"\n'
                            
                            st.download_button(
                                "📊 Ações em CSV",
                                data=csv_data,
                                file_name=f"acoes_video_{meeting_date.strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        # Limpar arquivo temporário
                        try:
                            os.unlink(video_path)
                        except:
                            pass
                        
                    except Exception as e:
                        st.error(f"❌ Erro durante a análise: {str(e)}")
                        try:
                            os.unlink(video_path)
                        except:
                            pass
        
        # Tab 2: Transcrição
        with tab2:
            st.subheader("📄 Análise de Transcrição")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Selecione a transcrição:",
                    type=['pdf', 'docx', 'txt'],
                    key="transcript_uploader"
                )
            
            with col2:
                st.info("""
                **Formatos suportados:**
                - PDF (documentos, atas)
                - DOCX (Word)
                - TXT (texto puro)
                
                **Dica:** Para melhor análise, inclua nomes dos participantes no texto.
                """)
            
            if uploaded_file:
                with st.spinner("Processando arquivo..."):
                    text = extract_text_from_file(uploaded_file)
                    
                    if text and not text.startswith("Erro"):
                        st.success("✅ Arquivo processado com sucesso!")
                        
                        # Mostrar prévia
                        with st.expander("👁️ Prévia do texto", expanded=False):
                            st.text_area("", text[:1000], height=200, disabled=True)
                        
                        # Formulário
                        with st.expander("✏️ Informações da Reunião", expanded=True):
                            col_info1, col_info2 = st.columns(2)
                            
                            with col_info1:
                                meeting_date = st.date_input(
                                    "Data:",
                                    value=datetime.now(),
                                    key="transcript_date"
                                )
                                meeting_time = st.time_input(
                                    "Horário:",
                                    value=datetime.now().time(),
                                    key="transcript_time"
                                )
                            
                            with col_info2:
                                meeting_type = st.selectbox(
                                    "Tipo:",
                                    ["Brainstorming", "Reunião de Equipe", "Apresentação", 
                                     "Revisão de Projeto", "One-on-One", "Decisão", "Status", "Outro"],
                                    key="transcript_type"
                                )
                                participants = st.text_area(
                                    "Participantes (um por linha):",
                                    height=80,
                                    key="transcript_participants"
                                )
                        
                        # Botão de análise
                        if st.button("🔍 Analisar Transcrição", type="primary", use_container_width=True):
                            with st.spinner("Analisando... Isso pode levar alguns minutos"):
                                meeting_info = f"""
                                DATA: {meeting_date.strftime('%d/%m/%Y')}
                                HORÁRIO: {meeting_time.strftime('%H:%M')}
                                TIPO: {meeting_type}
                                PARTICIPANTES: {participants if participants else 'Não informados'}
                                """
                                
                                analysis = analyze_transcript_with_gemini(text, meeting_info)
                                
                                # Mostrar resultados
                                st.markdown("---")
                                st.subheader("📊 Resultado da Análise")
                                st.markdown(analysis)
                                
                                # Download
                                st.download_button(
                                    "💾 Baixar Análise",
                                    data=analysis,
                                    file_name=f"analise_{meeting_date.strftime('%Y%m%d')}.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                    else:
                        st.error(f"❌ Erro: {text}")
    
    # Página: Configurações
    elif page == "⚙️ Configurações":
        st.title("⚙️ Configurações do Sistema")
        st.markdown("---")
        
        # Configurações do Gemini
        st.subheader("🔧 Configurações do Gemini")
        
        st.write(f"**API Key:** {'✅ Configurada' if gemini_api_key else '❌ Não configurada'}")
        st.write(f"**Modelo:** Gemini 1.5 Flash")
        
        # Configurações de análise
        st.subheader("📊 Configurações de Análise")
        
        analysis_depth = st.select_slider(
            "Nível de detalhe:",
            options=["Básico", "Padrão", "Detalhado", "Completo"],
            value="Padrão"
        )
        
        include_tone_analysis = st.checkbox(
            "Incluir análise de tom e emoção",
            value=True
        )
        
        generate_actions = st.checkbox(
            "Gerar plano de ações automaticamente",
            value=True
        )
        
        if st.button("💾 Salvar Configurações", type="primary"):
            st.session_state.analysis_depth = analysis_depth
            st.session_state.include_tone = include_tone_analysis
            st.session_state.generate_actions = generate_actions
            st.success("Configurações salvas!")
    
    # Página: Sobre
    elif page == "ℹ️ Sobre":
        st.title("ℹ️ Sobre o Sistema")
        st.markdown("---")
        
        st.info("""
        ## 🎯 Analisador de Reuniões com Gemini 1.5
        
        **Versão:** 2.0  
        **Data:** Novembro 2024  
        **Tecnologia:** Google Gemini 1.5 Flash
        
        ### 🚀 Funcionalidades
        
        #### 🎥 Análise de Vídeo
        - Suporte para vídeos de reuniões
        - Transcrição automática do áudio
        - Análise de tom de voz e entonação
        - Observações sobre dinâmica do grupo
        - Suporte a múltiplos formatos (MP4, MOV, AVI, etc.)
        
        #### 📄 Análise de Transcrição
        - Processamento de PDF, DOCX e TXT
        - Identificação de participantes
        - Análise de decisões e ações
        - Detecção de pontos de atenção
        - Recomendações para melhorias
        
        #### 🔧 Recursos Técnicos
        - Gemini 1.5 Flash (multimodal)
        - Análise de vídeo + áudio
        - Processamento em português
        - Exportação de resultados
        
        ### 📋 Requisitos do Sistema
        
        1. **API Key do Gemini:** Configure a variável `GEM_API_KEY`
        2. **Senha de acesso:** Configure a variável `senha_per`
        3. **Python 3.8+:** Com bibliotecas necessárias
        
        ### ⚠️ Limitações Conhecidas
        
        - Vídeos muito grandes podem demorar para processar
        - Qualidade do áudio afeta a transcrição
        - Requer conexão com internet para API
        
        ### 🆘 Suporte
        
        Para problemas ou dúvidas, verifique:
        1. Configuração das variáveis de ambiente
        2. Qualidade do arquivo de entrada
        3. Conexão com a internet
        4. Limites da API do Gemini
        """)

# ============================================================================
# ESTILOS CSS
# ============================================================================
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    
    .video-analysis-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .warning-card {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-card {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    /* Estilo para o botão de análise */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .upload-info {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    main_app()
