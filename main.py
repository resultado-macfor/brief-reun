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
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
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

def analyze_video_with_gemini(video_path, meeting_info=None, context_input=None):
    
    try:
        # Configurar safety settings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        
        # Prompt para análise de vídeo
        system_prompt = """🧠 Função do Agente (System Prompt)

Você é um agente de inteligência artificial especializado em analisar transcrições de calls de vendas complexas (B2B enterprise), com foco em avaliar a performance de vendedores (closers ou account executives) em ciclos de vendas longos e com múltiplos stakeholders.

📚 Base Teórica e Metodologias:

Suas análises devem ser baseadas nas técnicas e frameworks dos principais autores em vendas complexas, como:

Chris Voss (Never Split The Difference) — Técnicas de negociação, perguntas calibradas, fechamento de portas, ancoragem emocional

Aaron Ross (Predictable Revenue) — Prospecção outbound, qualificação de leads

Jeb Blount (Fanatical Prospecting / Sales EQ) — Inteligência emocional em vendas, controle da narrativa

Mike Weinberg (New Sales. Simplified.) — Estrutura de reuniões de descoberta e proposta

Brent Adamson & Matthew Dixon (The Challenger Sale) — Vendas baseadas em insight, reframe de problema

Oren Klaff (Pitch Anything) — Controle de frames, alavancagem de status

Miller Heiman Group (Strategic Selling) — Mapeamento de influenciadores e decisores

Neil Rackham (SPIN Selling) — Exploração de Situação, Problema, Implicação, Necessidade de solução

🧭 Etapas da Jornada de Venda Complexa para Avaliação

Abertura e conexão inicial

O closer estabeleceu rapport?

Criou alinhamento de expectativas?

Exploração e diagnóstico (discovery)

Utilizou perguntas abertas e investigativas?

Aplicou SPIN ou Challenger (provocou o lead)?

Identificou claramente dor, impacto e urgência?

Mapeamento de stakeholders e cenário político

Descobriu quem é o decisor, influenciador, gatekeeper?

Investigou como são tomadas decisões na empresa?

Apresentação de solução e storytelling de valor

Customizou a proposta para os desafios do lead?

Demonstrou ROI, risco e impacto estratégico?

Gestão de objeções e fricções

Antecipou e tratou objeções corretamente?

Mapeou objeções reais vs. falsas (ghost objections)?

Aplicou técnicas de reversão, isolamento e reancoragem?

Fechamento (com ou sem contrato)

Usou estratégias como “fechamento de portas” (no-oriented questions)?

Validou próximo passo concreto?

Reforçou escassez, autoridade ou prova social?

Follow-up e continuidade da negociação

Terminou a call com clareza e agenda definida?

Houve comprometimento mútuo sobre os próximos passos?

📊 Formato do Relatório que Devo Gerar

O output deve ser sempre estruturado com as seguintes seções:

Resumo executivo da performance

Pontos fortes do closer na call

Pontos de melhoria (técnicos, estratégicos e emocionais)

Técnicas e frameworks que poderiam ter sido melhor aplicados

Sugestões práticas para a próxima call (baseadas nos livros citados)

Score final (0 a 100) com base nos seguintes critérios:

Rapport e controle da conversa

Qualificação e exploração de dores

Estrutura da apresentação

Gestão de objeções

Capacidade de fechamento."""
        
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
        
        if context_input and context_input.strip():
            prompt_parts.append(f"\nCONTEXTO ADICIONAL FORNECIDO PELO USUÁRIO:\n{context_input}")
        
        prompt_parts.append("""
        
        🧠 Função do Agente (System Prompt)

Você é um agente de inteligência artificial especializado em analisar transcrições de calls de vendas complexas (B2B enterprise), com foco em avaliar a performance de vendedores (closers ou account executives) em ciclos de vendas longos e com múltiplos stakeholders.

📚 Base Teórica e Metodologias:

Suas análises devem ser baseadas nas técnicas e frameworks dos principais autores em vendas complexas, como:

Chris Voss (Never Split The Difference) — Técnicas de negociação, perguntas calibradas, fechamento de portas, ancoragem emocional

Aaron Ross (Predictable Revenue) — Prospecção outbound, qualificação de leads

Jeb Blount (Fanatical Prospecting / Sales EQ) — Inteligência emocional em vendas, controle da narrativa

Mike Weinberg (New Sales. Simplified.) — Estrutura de reuniões de descoberta e proposta

Brent Adamson & Matthew Dixon (The Challenger Sale) — Vendas baseadas em insight, reframe de problema

Oren Klaff (Pitch Anything) — Controle de frames, alavancagem de status

Miller Heiman Group (Strategic Selling) — Mapeamento de influenciadores e decisores

Neil Rackham (SPIN Selling) — Exploração de Situação, Problema, Implicação, Necessidade de solução

🧭 Etapas da Jornada de Venda Complexa para Avaliação

Abertura e conexão inicial

O closer estabeleceu rapport?

Criou alinhamento de expectativas?

Exploração e diagnóstico (discovery)

Utilizou perguntas abertas e investigativas?

Aplicou SPIN ou Challenger (provocou o lead)?

Identificou claramente dor, impacto e urgência?

Mapeamento de stakeholders e cenário político

Descobriu quem é o decisor, influenciador, gatekeeper?

Investigou como são tomadas decisões na empresa?

Apresentação de solução e storytelling de valor

Customizou a proposta para os desafios do lead?

Demonstrou ROI, risco e impacto estratégico?

Gestão de objeções e fricções

Antecipou e tratou objeções corretamente?

Mapeou objeções reais vs. falsas (ghost objections)?

Aplicou técnicas de reversão, isolamento e reancoragem?

Fechamento (com ou sem contrato)

Usou estratégias como “fechamento de portas” (no-oriented questions)?

Validou próximo passo concreto?

Reforçou escassez, autoridade ou prova social?

Follow-up e continuidade da negociação

Terminou a call com clareza e agenda definida?

Houve comprometimento mútuo sobre os próximos passos?

📊 Formato do Relatório que Devo Gerar

O output deve ser sempre estruturado com as seguintes seções:

Resumo executivo da performance

Pontos fortes do closer na call

Pontos de melhoria (técnicos, estratégicos e emocionais)

Técnicas e frameworks que poderiam ter sido melhor aplicados

Sugestões práticas para a próxima call (baseadas nos livros citados)

Score final (0 a 100) com base nos seguintes critérios:

Rapport e controle da conversa

Qualificação e exploração de dores

Estrutura da apresentação

Gestão de objeções

Capacidade de fechamento
        """)
        
        # Gerar análise
        with st.spinner("🔍 Gemini está analisando o vídeo... Isso pode levar alguns minutos"):
            response = gemini_model.generate_content(
                prompt_parts,
                safety_settings=safety_settings,
                generation_config={
                    "temperature": 0.0,
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

def analyze_transcript_with_gemini(transcript, meeting_info=None, context_input=None):
    """Analisa transcrição de reunião usando Gemini"""
    
    system_prompt = """Você é um especialista em análise de reuniões corporativas com background em psicologia organizacional, gestão de projetos e comunicação eficaz. 
    Sua análise deve ser profunda, prática e baseada em evidências científicas."""
    
    prompt = f"""
    ANALISE ESTA TRANSCRIÇÃO DE REUNIÃO:
    
    {transcript}
    
    🧠 Função do Agente (System Prompt)

Você é um agente de inteligência artificial especializado em analisar transcrições de calls de vendas complexas (B2B enterprise), com foco em avaliar a performance de vendedores (closers ou account executives) em ciclos de vendas longos e com múltiplos stakeholders.

📚 Base Teórica e Metodologias:

Suas análises devem ser baseadas nas técnicas e frameworks dos principais autores em vendas complexas, como:

Chris Voss (Never Split The Difference) — Técnicas de negociação, perguntas calibradas, fechamento de portas, ancoragem emocional

Aaron Ross (Predictable Revenue) — Prospecção outbound, qualificação de leads

Jeb Blount (Fanatical Prospecting / Sales EQ) — Inteligência emocional em vendas, controle da narrativa

Mike Weinberg (New Sales. Simplified.) — Estrutura de reuniões de descoberta e proposta

Brent Adamson & Matthew Dixon (The Challenger Sale) — Vendas baseadas em insight, reframe de problema

Oren Klaff (Pitch Anything) — Controle de frames, alavancagem de status

Miller Heiman Group (Strategic Selling) — Mapeamento de influenciadores e decisores

Neil Rackham (SPIN Selling) — Exploração de Situação, Problema, Implicação, Necessidade de solução

🧭 Etapas da Jornada de Venda Complexa para Avaliação

Abertura e conexão inicial

O closer estabeleceu rapport?

Criou alinhamento de expectativas?

Exploração e diagnóstico (discovery)

Utilizou perguntas abertas e investigativas?

Aplicou SPIN ou Challenger (provocou o lead)?

Identificou claramente dor, impacto e urgência?

Mapeamento de stakeholders e cenário político

Descobriu quem é o decisor, influenciador, gatekeeper?

Investigou como são tomadas decisões na empresa?

Apresentação de solução e storytelling de valor

Customizou a proposta para os desafios do lead?

Demonstrou ROI, risco e impacto estratégico?

Gestão de objeções e fricções

Antecipou e tratou objeções corretamente?

Mapeou objeções reais vs. falsas (ghost objections)?

Aplicou técnicas de reversão, isolamento e reancoragem?

Fechamento (com ou sem contrato)

Usou estratégias como “fechamento de portas” (no-oriented questions)?

Validou próximo passo concreto?

Reforçou escassez, autoridade ou prova social?

Follow-up e continuidade da negociação

Terminou a call com clareza e agenda definida?

Houve comprometimento mútuo sobre os próximos passos?

📊 Formato do Relatório que Devo Gerar

O output deve ser sempre estruturado com as seguintes seções:

Resumo executivo da performance

Pontos fortes do closer na call

Pontos de melhoria (técnicos, estratégicos e emocionais)

Técnicas e frameworks que poderiam ter sido melhor aplicados

Sugestões práticas para a próxima call (baseadas nos livros citados)

Score final (0 a 100) com base nos seguintes critérios:

Rapport e controle da conversa

Qualificação e exploração de dores

Estrutura da apresentação

Gestão de objeções

Capacidade de fechamento
    """
    
    # Adicionar informações adicionais
    if meeting_info:
        prompt = f"INFORMAÇÕES ADICIONAIS DA REUNIÃO:\n{meeting_info}\n\n{prompt}"
    
    if context_input and context_input.strip():
        prompt = f"CONTEXTO ADICIONAL FORNECIDO PELO USUÁRIO:\n{context_input}\n\n{prompt}"
    
    prompt += """

    🧠 Função do Agente (System Prompt)

Você é um agente de inteligência artificial especializado em analisar transcrições de calls de vendas complexas (B2B enterprise), com foco em avaliar a performance de vendedores (closers ou account executives) em ciclos de vendas longos e com múltiplos stakeholders.

📚 Base Teórica e Metodologias:

Suas análises devem ser baseadas nas técnicas e frameworks dos principais autores em vendas complexas, como:

Chris Voss (Never Split The Difference) — Técnicas de negociação, perguntas calibradas, fechamento de portas, ancoragem emocional

Aaron Ross (Predictable Revenue) — Prospecção outbound, qualificação de leads

Jeb Blount (Fanatical Prospecting / Sales EQ) — Inteligência emocional em vendas, controle da narrativa

Mike Weinberg (New Sales. Simplified.) — Estrutura de reuniões de descoberta e proposta

Brent Adamson & Matthew Dixon (The Challenger Sale) — Vendas baseadas em insight, reframe de problema

Oren Klaff (Pitch Anything) — Controle de frames, alavancagem de status

Miller Heiman Group (Strategic Selling) — Mapeamento de influenciadores e decisores

Neil Rackham (SPIN Selling) — Exploração de Situação, Problema, Implicação, Necessidade de solução

🧭 Etapas da Jornada de Venda Complexa para Avaliação

Abertura e conexão inicial

O closer estabeleceu rapport?

Criou alinhamento de expectativas?

Exploração e diagnóstico (discovery)

Utilizou perguntas abertas e investigativas?

Aplicou SPIN ou Challenger (provocou o lead)?

Identificou claramente dor, impacto e urgência?

Mapeamento de stakeholders e cenário político

Descobriu quem é o decisor, influenciador, gatekeeper?

Investigou como são tomadas decisões na empresa?

Apresentação de solução e storytelling de valor

Customizou a proposta para os desafios do lead?

Demonstrou ROI, risco e impacto estratégico?

Gestão de objeções e fricções

Antecipou e tratou objeções corretamente?

Mapeou objeções reais vs. falsas (ghost objections)?

Aplicou técnicas de reversão, isolamento e reancoragem?

Fechamento (com ou sem contrato)

Usou estratégias como “fechamento de portas” (no-oriented questions)?

Validou próximo passo concreto?

Reforçou escassez, autoridade ou prova social?

Follow-up e continuidade da negociação

Terminou a call com clareza e agenda definida?

Houve comprometimento mútuo sobre os próximos passos?

📊 Formato do Relatório que Devo Gerar

O output deve ser sempre estruturado com as seguintes seções:

Resumo executivo da performance

Pontos fortes do closer na call

Pontos de melhoria (técnicos, estratégicos e emocionais)

Técnicas e frameworks que poderiam ter sido melhor aplicados

Sugestões práticas para a próxima call (baseadas nos livros citados)

Score final (0 a 100) com base nos seguintes critérios:

Rapport e controle da conversa

Qualificação e exploração de dores

Estrutura da apresentação

Gestão de objeções

Capacidade de fechamento
    

    """
    
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
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
            - Análise de tom de voce e entonação
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
                
                # Campo de contexto adicional
                st.markdown("### 📝 Contexto Adicional (Opcional)")
                context_input_video = st.text_area(
                    "Forneça contexto adicional para análise:",
                    height=100,
                    placeholder="Ex: Esta reunião é sobre o lançamento do produto X...\nO objetivo principal é alinhar as equipes...\nO tom deve ser mais informal porque é uma reunião interna...",
                    help="Informações adicionais que ajudam na análise",
                    key="context_video"
                )
                
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
                        analysis = analyze_video_with_gemini(video_path, meeting_info, context_input_video)
                        
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
            
            # Opções de entrada
            entrada_opcao = st.radio(
                "Escolha como fornecer a transcrição:",
                ["📤 Upload de Arquivo", "📝 Copiar/Colar Texto"],
                horizontal=True,
                key="entrada_opcao"
            )
            
            if entrada_opcao == "📤 Upload de Arquivo":
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
                                st.text_area("", text[:1000], height=200, disabled=True, key="preview_upload")
                            
                            transcription_text = text
                        else:
                            st.error(f"❌ Erro: {text}")
                            transcription_text = ""
                else:
                    transcription_text = ""
                    
            else:  # 📝 Copiar/Colar Texto
                transcription_text = st.text_area(
                    "Cole a transcrição da reunião aqui:",
                    height=300,
                    placeholder="""Exemplo:

João: Boa tarde a todos, vamos começar a reunião. O objetivo hoje é discutir o lançamento do novo produto.
Maria: Obrigada, João. Preparei uma apresentação com os principais pontos...
Pedro: Concordo com a Maria, precisamos focar no cronograma de lançamento...
Ana: Tenho algumas preocupações com o orçamento...

[Continue com o restante da transcrição...]""",
                    help="Cole a transcrição completa da reunião. Inclua os nomes dos participantes quando possível.",
                    key="transcript_textarea"
                )
                
                if transcription_text:
                    word_count = len(transcription_text.split())
                    char_count = len(transcription_text)
                    st.info(f"📊 Texto: {word_count} palavras, {char_count} caracteres")
            
            # Campo de contexto adicional (para ambas as opções)
            if transcription_text:
                st.markdown("### 📝 Contexto Adicional (Opcional)")
                context_input_transcript = st.text_area(
                    "Forneça contexto adicional para análise:",
                    height=100,
                    placeholder="Ex: Esta reunião é da equipe de marketing...\nO produto X está em fase de teste...\nO tom deve ser analítico porque estamos discutindo dados...",
                    help="Informações adicionais que ajudam na análise da transcrição",
                    key="context_transcript"
                )
                
                # Formulário de informações da reunião
                st.markdown("### 📋 Informações da Reunião")
                with st.expander("✏️ Preencha as informações da reunião", expanded=True):
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        meeting_date = st.date_input(
                            "Data da reunião:",
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
                            "Tipo de reunião:",
                            ["Brainstorming", "Reunião de Equipe", "Apresentação", 
                             "Revisão de Projeto", "One-on-One", "Decisão", "Status", "Outro"],
                            key="transcript_type"
                        )
                        participants = st.text_area(
                            "Participantes (um por linha):",
                            height=80,
                            placeholder="João Silva\nMaria Santos\nPedro Oliveira\nAna Costa",
                            help="Liste todos os participantes da reunião",
                            key="transcript_participants"
                        )
                        
                        meeting_objective = st.text_area(
                            "Objetivo da reunião (opcional):",
                            height=60,
                            placeholder="Ex: Alinhar cronograma do projeto X...",
                            key="transcript_objective"
                        )
                
                # Botão de análise
                if st.button("🔍 Analisar Transcrição", type="primary", use_container_width=True, disabled=not transcription_text.strip()):
                    with st.spinner("Analisando... Isso pode levar alguns minutos"):
                        # Preparar informações da reunião
                        meeting_info = f"""
                        DATA: {meeting_date.strftime('%d/%m/%Y')}
                        HORÁRIO: {meeting_time.strftime('%H:%M')}
                        TIPO: {meeting_type}
                        PARTICIPANTES: {participants if participants else 'Não informados'}
                        OBJETIVO: {meeting_objective if meeting_objective else 'Não informado'}
                        """
                        
                        # Realizar análise
                        analysis = analyze_transcript_with_gemini(
                            transcription_text, 
                            meeting_info, 
                            context_input_transcript
                        )
                        
                        # Mostrar resultados
                        st.markdown("---")
                        st.subheader("📊 Resultado da Análise")
                        st.markdown(analysis)
                        
                        # Opções de download
                        st.markdown("---")
                        st.subheader("📥 Exportar Resultados")
                        
                        col_dl1, col_dl2, col_dl3 = st.columns(3)
                        
                        with col_dl1:
                            st.download_button(
                                "💾 Baixar Análise Completa",
                                data=analysis,
                                file_name=f"analise_{meeting_date.strftime('%Y%m%d')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col_dl2:
                            # Extrair resumo executivo
                            if "## 📋 RESUMO DA REUNIÃO" in analysis:
                                start_idx = analysis.find("## 📋 RESUMO DA REUNIÃO")
                                end_idx = analysis.find("##", start_idx + 1)
                                summary = analysis[start_idx:end_idx] if end_idx != -1 else analysis[start_idx:]
                                st.download_button(
                                    "📋 Resumo Executivo",
                                    data=summary,
                                    file_name=f"resumo_{meeting_date.strftime('%Y%m%d')}.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                        
                        with col_dl3:
                            # Extrair ações em CSV
                            csv_data = "Ação,Responsável,Prazo,Status\n"
                            if "## 3. 🎯 DECISÕES E AÇÕES" in analysis:
                                start_idx = analysis.find("## 3. 🎯 DECISÕES E AÇÕES")
                                end_idx = analysis.find("##", start_idx + 1)
                                actions_section = analysis[start_idx:end_idx] if end_idx != -1 else analysis[start_idx:]
                                
                                # Processar ações
                                lines = actions_section.split('\n')
                                for line in lines:
                                    if '|' in line and '-' in line and '|' in line:
                                        parts = [p.strip() for p in line.split('|') if p.strip()]
                                        if len(parts) >= 3:
                                            csv_data += f'"{parts[0]}","{parts[1]}","{parts[2]}","Pendente"\n'
                            
                            st.download_button(
                                "📊 Ações em CSV",
                                data=csv_data,
                                file_name=f"acoes_{meeting_date.strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
            
            else:
                if entrada_opcao == "📝 Copiar/Colar Texto":
                    st.info("✍️ Cole a transcrição da reunião na caixa de texto acima para começar a análise.")
                else:
                    st.info("📤 Faça upload de um arquivo ou cole o texto da reunião para começar a análise.")
    
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
        
        # Configurações de exportação
        st.subheader("📥 Configurações de Exportação")
        
        export_format = st.multiselect(
            "Formatos de exportação:",
            ["TXT", "CSV", "PDF", "JSON"],
            default=["TXT", "CSV"]
        )
        
        include_summary = st.checkbox(
            "Incluir resumo executivo em todas as exportações",
            value=True
        )
        
        if st.button("💾 Salvar Configurações", type="primary"):
            st.session_state.analysis_depth = analysis_depth
            st.session_state.include_tone = include_tone_analysis
            st.session_state.generate_actions = generate_actions
            st.session_state.export_format = export_format
            st.session_state.include_summary = include_summary
            st.success("✅ Configurações salvas!")
    
    # Página: Sobre
    elif page == "ℹ️ Sobre":
        st.title("ℹ️ Sobre o Sistema")
        st.markdown("---")
        
        st.info("""
        ## 🎯 Analisador de Reuniões com Gemini 1.5
        
        **Versão:** 2.1  
        **Data:** Novembro 2024  
        **Tecnologia:** Google Gemini 1.5 Flash
        
        ### 🚀 Novas Funcionalidades
        
        #### 📝 Entrada Flexível
        - **Upload de arquivos:** PDF, DOCX, TXT
        - **Copiar/Colar:** Digite ou cole a transcrição diretamente
        - **Contexto adicional:** Campo extra para informações contextuais
        
        #### 🎥 Análise de Vídeo
        - Transcrição automática do áudio
        - Análise de tom de voz e entonação
        - Observações sobre dinâmica do grupo
        - Suporte a múltiplos formatos de vídeo
        
        #### 📊 Análise de Transcrição
        - Identificação de participantes
        - Análise de decisões e ações
        - Detecção de pontos de atenção
        - Recomendações para melhorias
        
        ### 📋 Como Usar
        
        1. **Para Vídeo:**
           - Faça upload do vídeo da reunião
           - Adicione contexto adicional (opcional)
           - Preencha informações da reunião
           - Clique em "Analisar Vídeo"
        
        2. **Para Transcrição:**
           - Escolha entre upload ou copiar/colar
           - Cole a transcrição completa
           - Adicione contexto adicional (opcional)
           - Preencha informações da reunião
           - Clique em "Analisar Transcrição"
        
        ### ⚠️ Dicas Importantes
        
        - **Para melhores resultados:** Inclua nomes dos participantes
        - **Contexto ajuda:** Quanto mais informações, melhor a análise
        - **Vídeos grandes:** Podem demorar mais para processar
        - **Transcrições longas:** Mantenha a formatação clara
        
        ### 🆘 Suporte
        
        Para problemas ou dúvidas:
        1. Verifique a configuração da API do Gemini
        2. Use vídeos/arquivos de boa qualidade
        3. Certifique-se de ter conexão com internet
        4. Para vídeos grandes, aguarde o processamento
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
    
    .context-card {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
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
    
    .text-counter {
        font-size: 0.8rem;
        color: #666;
        text-align: right;
        margin-top: -10px;
        margin-bottom: 10px;
    }
    
    .radio-options {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    main_app()
