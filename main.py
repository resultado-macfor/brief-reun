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
    gemini_model = genai.GenerativeModel('gemini-2.5-pro')
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
# FUNÇÕES DE ANÁLISE DE VÍDEO COM GEMINI
# ============================================================================
def analyze_video_with_gemini(video_path, meeting_info=None):
    """Analisa vídeo de reunião usando Gemini 1.5 Flash"""
    
    try:
        # Configurar safety settings para permitir análise de áudio/vídeo
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Prompt para análise de vídeo
        system_prompt = """Você é um especialista em análise de reuniões corporativas. 
        Analise este vídeo de reunião considerando:
        1. Conteúdo verbal (o que é dito)
        2. Tom de voz e entonação
        3. Dinâmica entre participantes
        4. Linguagem corporal quando visível
        5. Engajamento dos participantes
        
        Forneça uma análise completa e estruturada."""
        
        # Carregar o vídeo
        video_file = genai.upload_file(video_path)
        
        # Preparar prompt
        prompt_parts = [
            system_prompt,
            "\nANALISE ESTE VÍDEO DE REUNIÃO:",
            video_file,
        ]
        
        if meeting_info:
            prompt_parts.append(f"\nINFORMAÇÕES ADICIONAIS:\n{meeting_info}")
        
        prompt_parts.append("""
        FORNECE UMA ANÁLISE DETALHADA NO SEGUINTE FORMATO:
        
        # 🎥 ANÁLISE DE VÍDEO DE REUNIÃO
        
        ## 📋 RESUMO EXECUTIVO
        [Resumo de 3-4 parágrafos da reunião]
        
        ## 🗣️ TRANSCRIÇÃO PRINCIPAL
        [Transcrição dos pontos mais importantes discutidos]
        
        ## 👥 ANÁLISE DE PARTICIPANTES
        ### [Participante 1 - quando identificável]
        - **Participação:** [nível de participação]
        - **Tom de voz:** [análise do tom]
        - **Engajamento:** [análise do engajamento]
        - **Contribuições:** [principais contribuições]
        
        ## 🎭 DINÂMICA DA REUNIÃO
        - **Clima geral:** [positivo, tenso, neutro, etc.]
        - **Interações:** [como os participantes interagiram]
        - **Liderança:** [quem liderou a reunião]
        - **Conflitos:** [se houver conflitos observados]
        
        ## 🔊 ANÁLISE DE ÁUDIO
        - **Clareza da comunicação:** [nível de entendimento]
        - **Tom predominante:** [formal, informal, amigável, etc.]
        - **Momentos-chave:** [momentos importantes pela entonação]
        
        ## 👀 OBSERVAÇÕES VISUAIS (quando aplicável)
        - **Linguagem corporal:** [observações relevantes]
        - **Expressões faciais:** [expressões notáveis]
        - **Engajamento visual:** [nível de atenção]
        
        ## 🚨 PONTOS DE ATENÇÃO
        - [Lista de pontos que merecem atenção]
        
        ## 💡 RECOMENDAÇÕES
        - [Sugestões para melhorias]
        
        ## ⭐ AVALIAÇÃO FINAL
        **Eficácia da reunião:** X/10
        **Engajamento:** X/10
        **Produtividade:** X/10
        """)
        
        # Gerar análise
        with st.spinner("🔍 Gemini está analisando o vídeo... Isso pode levar alguns minutos"):
            response = gemini_model.generate_content(
                prompt_parts,
                safety_settings=safety_settings,
                generation_config={"temperature": 0.1}
            )
        
        return response.text
        
    except Exception as e:
        return f"❌ Erro na análise do vídeo: {str(e)}"

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
            generation_config={"temperature": 0.1}
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
        st.write(f"**Modelo:** Gemini 1.5 Flash")
        st.write(f"**Data:** {datetime.now().strftime('%d/%m/%Y')}")
        
        st.markdown("---")
        
        # Botão de logout
        if st.button("🚪 Sair do Sistema", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()
    
    # Página: Nova Análise
    if page == "📁 Nova Análise":
        st.title("🎯 Análise de Reuniões com Gemini")
        st.markdown("Faça upload de vídeo, áudio ou transcrição para análise detalhada")
        st.markdown("---")
        
        # Abas para diferentes tipos de entrada
        tab1, tab2, tab3 = st.tabs(["🎥 Análise de Vídeo", "📄 Transcrição", "🔊 Áudio (em breve)"])
        
        # Tab 1: Análise de Vídeo
        with tab1:
            st.subheader("🎥 Análise de Vídeo de Reunião")
            st.info("""
            **Funcionalidades disponíveis:**
            - Análise completa de vídeos de reuniões
            - Transcrição automática do áudio
            - Análise de tom de voz e entonação
            - Observações sobre dinâmica do grupo
            - Suporta vídeos até 1 hora (Gemini 1.5 Flash)
            """)
            
            # Upload de vídeo
            video_file = st.file_uploader(
                "Selecione o vídeo da reunião:",
                type=['mp4', 'mov', 'avi', 'mkv', 'webm'],
                key="video_uploader"
            )
            
            if video_file:
                # Mostrar informações do vídeo
                file_size_mb = video_file.size / (1024 * 1024)
                st.success(f"✅ Vídeo carregado: {video_file.name} ({file_size_mb:.1f} MB)")
                
                # Pré-visualização do vídeo
                st.video(video_file)
                
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
                             "Revisão de Projeto", "One-on-One", "Outro"],
                            key="video_type"
                        )
                        participants = st.text_area(
                            "Participantes (opcional, um por linha):",
                            height=80,
                            placeholder="João Silva\nMaria Santos\nPedro Oliveira",
                            key="video_participants"
                        )
                
                # Botão de análise
                if st.button("🔍 Analisar Vídeo", type="primary", use_container_width=True):
                    if file_size_mb > 100:  # Limite aproximado do Gemini
                        st.warning("⚠️ O vídeo é muito grande. Recomendamos vídeos menores que 100MB.")
                    
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
                        
                        col_dl1, col_dl2 = st.columns(2)
                        
                        with col_dl1:
                            st.download_button(
                                "💾 Baixar como TXT",
                                data=analysis,
                                file_name=f"analise_video_{meeting_date.strftime('%Y%m%d')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col_dl2:
                            # Resumo executivo
                            summary_prompt = f"Crie um resumo executivo de 1 parágrafo desta análise:\n\n{analysis}"
                            try:
                                response = gemini_model.generate_content(summary_prompt)
                                summary = response.text
                                st.download_button(
                                    "📋 Resumo Executivo",
                                    data=summary,
                                    file_name=f"resumo_video_{meeting_date.strftime('%Y%m%d')}.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                            except:
                                st.error("Erro ao criar resumo")
                        
                        # Limpar arquivo temporário
                        os.unlink(video_path)
                        
                    except Exception as e:
                        st.error(f"❌ Erro durante a análise: {str(e)}")
                        if os.path.exists(video_path):
                            os.unlink(video_path)
        
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
                - PDF
                - DOCX (Word)
                - TXT (texto puro)
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
                                     "Revisão de Projeto", "One-on-One", "Outro"],
                                    key="transcript_type"
                                )
                                participants = st.text_area(
                                    "Participantes:",
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
        
        # Tab 3: Áudio (placeholder)
        with tab3:
            st.subheader("🔊 Análise de Áudio")
            st.info("""
            **Em breve!**
            
            Estamos trabalhando na integração com:
            - Análise de áudio puro
            - Transcrição automática
            - Análise de tom de voz
            - Detecção de emoções
            
            **Por enquanto, use a opção de vídeo ou transcrição.**
            """)
    
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
        - Gemini 1.5 Flash (até 1 milhão de tokens)
        - Análise multimodal (vídeo + áudio)
        - Processamento em português
        - Exportação de resultados
        
        ### 📋 Requisitos do Sistema
        
        1. **API Key do Gemini:** Configure a variável `GEM_API_KEY`
        2. **Senha de acesso:** Configure a variável `senha_per`
        3. **Python 3.8+:** Com bibliotecas necessárias
        
        ### ⚠️ Limitações Conhecidas
        
        - Vídeos muito grandes podem demorar
        - Qualidade do áudio afeta a transcrição
        - Análise visual limitada pela qualidade do vídeo
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    main_app()
