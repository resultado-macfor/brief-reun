import os
import streamlit as st
import pandas as pd
from datetime import datetime
import hashlib
import json
from typing import List, Dict, Tuple
import PyPDF2
import docx
import io
import google.generativeai as genai
from anthropic import Anthropic
from openai import OpenAI
import requests
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
# CONFIGURAÇÃO DAS APIS
# ============================================================================
# Configurar APIs (coloque suas chaves nas variáveis de ambiente)
gemini_api_key = os.getenv("GEM_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
perplexity_api_key = os.getenv("PERP_API_KEY")

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
                        st.error("⚠️ Sistema não configurado. Verifique as variáveis de ambiente.")
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
# CONFIGURAÇÃO DOS CLIENTES DE IA (só executa se autenticado)
# ============================================================================
clients = {}

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    clients["gemini"] = genai.GenerativeModel("gemini-2.5-flash")
else:
    st.warning("API do Gemini não configurada")

if anthropic_api_key:
    clients["claude"] = Anthropic(api_key=anthropic_api_key)
else:
    st.warning("API do Claude não configurada")

if openai_api_key:
    clients["openai"] = OpenAI(api_key=openai_api_key)
else:
    st.warning("API do OpenAI não configurada")

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

def extract_text_from_md(file):
    """Extrai texto de Markdown"""
    try:
        return file.read().decode("utf-8")
    except:
        try:
            file.seek(0)
            return file.read().decode("latin-1")
        except Exception as e:
            return f"Erro ao extrair texto do Markdown: {str(e)}"

def extract_text_from_file(file):
    """Extrai texto de qualquer arquivo suportado"""
    filename = file.name.lower()
    
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif filename.endswith('.docx'):
        return extract_text_from_docx(file)
    elif filename.endswith('.txt'):
        return extract_text_from_txt(file)
    elif filename.endswith('.md'):
        return extract_text_from_md(file)
    else:
        return "Formato de arquivo não suportado"

# ============================================================================
# FUNÇÕES DE ANÁLISE COM LLM
# ============================================================================
def call_llm(prompt, model="gemini", system_prompt=None, temperature=0.1):
    """Chama diferentes modelos de LLM"""
    try:
        if model == "gemini" and "gemini" in clients:
            if system_prompt:
                prompt = f"{system_prompt}\n\n{prompt}"
            response = clients["gemini"].generate_content(prompt)
            return response.text
        
        elif model == "claude" and "claude" in clients:
            messages = [{"role": "user", "content": prompt}]
            if system_prompt:
                response = clients["claude"].messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=4000,
                    temperature=temperature,
                    system=system_prompt,
                    messages=messages
                )
            else:
                response = clients["claude"].messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=4000,
                    temperature=temperature,
                    messages=messages
                )
            return response.content[0].text
        
        elif model == "openai" and "openai" in clients:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = clients["openai"].chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=temperature,
                max_tokens=4000
            )
            return response.choices[0].message.content
        
        else:
            return "❌ Nenhum modelo de LLM configurado. Configure pelo menos uma API key."
            
    except Exception as e:
        return f"❌ Erro ao chamar LLM: {str(e)}"

def search_web_perplexity(query, max_results=3):
    """Busca informações na web usando Perplexity"""
    if not perplexity_api_key:
        return "API do Perplexity não configurada"
    
    try:
        headers = {
            "Authorization": f"Bearer {perplexity_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "sonar-medium-online",
            "messages": [
                {
                    "role": "user",
                    "content": f"""Pesquise informações sobre: {query}
                    
                    Forneça informações relevantes, atualizadas e confiáveis.
                    Inclua fontes quando possível.
                    Limite a {max_results} resultados principais."""
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Erro na busca: {response.status_code}"
            
    except Exception as e:
        return f"Erro na busca web: {str(e)}"

# ============================================================================
# FUNÇÕES ESPECÍFICAS PARA ANÁLISE DE REUNIÕES
# ============================================================================
def analyze_meeting_transcript(transcript, meeting_info=None):
    """Analisa transcrição de reunião"""
    
    system_prompt = """Você é um especialista em análise de reuniões corporativas com background em psicologia organizacional, gestão de projetos e comunicação eficaz. 
    Sua análise deve ser profunda, prática e baseada em evidências científicas."""
    
    prompt = f"""
    ANALISE ESTA TRANSCRIÇÃO DE REUNIÃO:
    
    {transcript[:15000]}  # Limita o tamanho para evitar token overflow
    
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
    
    3. ANÁLISE PSICOLÓGICA E COMPORTAMENTAL POR PARTICIPANTE:
    Para cada participante identificado, analise:
    - Nível de participação (ativo/passivo)
    - Tom de voz e linguagem utilizada
    - Grau de colaboração vs. competição
    - Clareza na comunicação
    - Demonstração de liderança ou followership
    - Sinais de estresse, frustração ou satisfação
    - Influência sobre decisões
    
    4. RED FLAGS E PONTOS DE ATENÇÃO:
    - Comunicação ineficaz
    - Falta de clareza em responsabilidades
    - Conflitos não resolvidos
    - Desalinhamento de expectativas
    - Falta de preparação
    - Dominação por parte de alguns participantes
    
    5. EFICÁCIA DA REUNIÃO:
    - Objetivos atingidos?
    - Tempo bem utilizado?
    - Participação equilibrada?
    - Decisões claras e acionáveis?
    - Próximos passos definidos?
    
    6. INSIGHTS E RECOMENDAÇÕES:
    - Pontos fortes a serem mantidos
    - Melhorias sugeridas para próximas reuniões
    - Treinamentos ou desenvolvimentos recomendados
    - Ajustes no formato da reunião
    
    7. NOTA FINAL (0-10):
    - Eficiência (0-10)
    - Satisfação dos participantes (estimada 0-10)
    - Qualidade das decisões (0-10)
    - Média final (0-10)
    
    FORMATO DA RESPOSTA:
    
    # 📊 RELATÓRIO DE ANÁLISE DE REUNIÃO
    
    ## 1. 🧑‍🤝‍🧑 PARTICIPANTES IDENTIFICADOS
    [Lista detalhada]
    
    ## 2. 📋 RESUMO DA REUNIÃO
    [Resumo executivo]
    
    ## 3. 🎯 DECISÕES E AÇÕES
    [Tabela com ações, responsáveis e prazos]
    
    ## 4. 👥 ANÁLISE INDIVIDUAL POR PARTICIPANTE
    ### Participante 1: [Nome]
    - Participação: [X%]
    - Comportamento: [análise]
    - Comunicação: [análise]
    - Contribuição: [análise]
    - Recomendações: [sugestões]
    
    [Repetir para cada participante]
    
    ## 5. 🚨 RED FLAGS IDENTIFICADAS
    [Lista com explicação e gravidade]
    
    ## 6. 💡 INSIGHTS E RECOMENDAÇÕES
    [Lista detalhada]
    
    ## 7. ⭐ NOTA FINAL DA REUNIÃO
    **Eficiência:** X/10
    **Satisfação:** X/10  
    **Qualidade:** X/10
    **Média Final:** X/10
    
    ## 8. 📌 PRÓXIMOS PASSOS
    [Resumo das ações acordadas]
    """
    
    if meeting_info:
        prompt = f"INFORMAÇÕES ADICIONAIS DA REUNIÃO:\n{meeting_info}\n\n{prompt}"
    
    return call_llm(prompt, model="gemini", system_prompt=system_prompt, temperature=0.1)

def extract_meeting_metadata(text):
    """Extrai metadados básicos da reunião do texto"""
    
    prompt = f"""
    Extraia informações básicas desta reunião:
    
    {text[:5000]}
    
    Procure por:
    1. Data da reunião
    2. Horário
    3. Participantes presentes
    4. Objetivo da reunião
    5. Tópicos principais
    
    Formato de resposta JSON:
    {{
        "date": "data encontrada ou desconhecida",
        "time": "horário encontrado ou desconhecido",
        "participants": ["lista de nomes"],
        "objective": "objetivo da reunião",
        "topics": ["lista de tópicos"]
    }}
    """
    
    response = call_llm(prompt, model="gemini", temperature=0.1)
    
    # Tentar extrair JSON da resposta
    try:
        # Procura por JSON na resposta
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    # Retorno padrão se não conseguir extrair JSON
    return {
        "date": "Não identificada",
        "time": "Não identificado",
        "participants": ["Participantes não identificados"],
        "objective": "Não identificado",
        "topics": ["Tópicos não identificados"]
    }

# ============================================================================
# INTERFACE PRINCIPAL (só aparece se autenticado)
# ============================================================================
def main_app():
    """Interface principal do aplicativo"""
    
    # Sidebar com navegação e logout
    with st.sidebar:
        st.title("🎯 Analisador de Reuniões IA")
        st.markdown("---")
        
        # Navegação
        st.subheader("📌 Navegação")
        page = st.radio(
            "Selecione a página:",
            ["📁 Nova Análise", "⚙️ Configurações"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Status do sistema
        st.subheader("ℹ️ Status do Sistema")
        
        # Verificar APIs configuradas
        apis_configuradas = []
        if gemini_api_key:
            apis_configuradas.append("✅ Gemini")
        if anthropic_api_key:
            apis_configuradas.append("✅ Claude")
        if openai_api_key:
            apis_configuradas.append("✅ OpenAI")
        if perplexity_api_key:
            apis_configuradas.append("✅ Perplexity")
        
        if apis_configuradas:
            st.write("**APIs Configuradas:**")
            for api in apis_configuradas:
                st.write(api)
        else:
            st.warning("⚠️ Nenhuma API configurada")
        
        st.markdown("---")
        
        # Botão de logout
        if st.button("🚪 Sair do Sistema", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()
    
    # Página: Nova Análise
    if page == "📁 Nova Análise":
        st.title("🎯 Análise de Reuniões")
        st.markdown("Faça upload da transcrição ou cole o texto para análise detalhada")
        st.markdown("---")
        
        # Abas para diferentes tipos de entrada
        tab1, tab2 = st.tabs(["📄 Upload de Documento", "📝 Colar Texto"])
        
        with tab1:
            st.subheader("Faça upload da transcrição da reunião")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Selecione o arquivo:",
                    type=['pdf', 'docx', 'txt', 'md'],
                    help="Formatos suportados: PDF, DOCX, TXT, MD"
                )
            
            with col2:
                st.info("""
                **Formatos suportados:**
                - PDF (atas, relatórios)
                - DOCX (documentos Word)
                - TXT (transcrições puras)
                - MD (Markdown)
                """)
            
            if uploaded_file:
                with st.spinner("Processando arquivo..."):
                    text = extract_text_from_file(uploaded_file)
                    
                    if text and not text.startswith("Erro"):
                        st.success("✅ Arquivo processado com sucesso!")
                        
                        # Extrair metadados
                        metadata = extract_meeting_metadata(text)
                        
                        # Formulário para informações adicionais
                        with st.expander("✏️ Adicionar informações da reunião", expanded=True):
                            col_info1, col_info2 = st.columns(2)
                            
                            with col_info1:
                                meeting_date = st.date_input(
                                    "Data da reunião:",
                                    value=datetime.now(),
                                    key="meeting_date"
                                )
                                meeting_time = st.time_input(
                                    "Horário:",
                                    value=datetime.now().time(),
                                    key="meeting_time"
                                )
                                meeting_type = st.selectbox(
                                    "Tipo de reunião:",
                                    ["Brainstorming", "Decisão", "Status", "Planejamento", "Retrospectiva", "Outro"]
                                )
                            
                            with col_info2:
                                participants_input = st.text_area(
                                    "Participantes (um por linha):",
                                    value="\n".join(metadata.get("participants", [])),
                                    height=100
                                )
                                meeting_objective = st.text_area(
                                    "Objetivo da reunião:",
                                    value=metadata.get("objective", ""),
                                    height=80
                                )
                        
                        # Botão para análise
                        if st.button("🔍 Analisar Reunião", type="primary", use_container_width=True):
                            with st.spinner("Analisando reunião... Isso pode levar alguns minutos"):
                                # Preparar informações adicionais
                                meeting_info = f"""
                                DATA: {meeting_date.strftime('%d/%m/%Y')}
                                HORÁRIO: {meeting_time.strftime('%H:%M')}
                                TIPO: {meeting_type}
                                PARTICIPANTES: {participants_input}
                                OBJETIVO: {meeting_objective}
                                """
                                
                                # Realizar análise
                                analysis = analyze_meeting_transcript(text, meeting_info)
                                
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
                                        "💾 Baixar como TXT",
                                        data=analysis,
                                        file_name=f"analise_reuniao_{meeting_date.strftime('%Y%m%d')}.txt",
                                        mime="text/plain"
                                    )
                                
                                with col_dl2:
                                    # Criar resumo executivo
                                    summary_prompt = f"Crie um resumo executivo de 1 parágrafo desta análise:\n\n{analysis}"
                                    summary = call_llm(summary_prompt, model="gemini")
                                    st.download_button(
                                        "📋 Resumo Executivo",
                                        data=summary,
                                        file_name=f"resumo_reuniao_{meeting_date.strftime('%Y%m%d')}.txt",
                                        mime="text/plain"
                                    )
                                
                                with col_dl3:
                                    # Criar ações em CSV
                                    csv_data = "Ação,Responsável,Prazo,Status\n"
                                    # Extrair ações da análise
                                    actions_prompt = f"Extraia as ações desta análise no formato CSV:\n\n{analysis}"
                                    actions = call_llm(actions_prompt, model="gemini")
                                    if "Ação" in actions:
                                        csv_data = actions
                                    st.download_button(
                                        "📊 Ações em CSV",
                                        data=csv_data,
                                        file_name=f"acoes_reuniao_{meeting_date.strftime('%Y%m%d')}.csv",
                                        mime="text/csv"
                                    )
                    else:
                        st.error(f"❌ Erro ao processar arquivo: {text}")
        
        with tab2:
            st.subheader("Cole a transcrição da reunião")
            
            manual_text = st.text_area(
                "Cole o texto da reunião aqui:",
                height=400,
                placeholder="Exemplo:\nJoão: Boa tarde a todos, vamos começar a reunião...\nMaria: O objetivo hoje é discutir...\nPedro: Concordo com a Maria, precisamos...",
                help="Formato livre. Inclua nomes dos participantes quando possível."
            )
            
            if manual_text:
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    meeting_date = st.date_input(
                        "Data da reunião:",
                        value=datetime.now(),
                        key="manual_date"
                    )
                    meeting_time = st.time_input(
                        "Horário:",
                        value=datetime.now().time(),
                        key="manual_time"
                    )
                
                with col_info2:
                    meeting_type = st.selectbox(
                        "Tipo de reunião:",
                        ["Brainstorming", "Decisão", "Status", "Planejamento", "Retrospectiva", "Outro"],
                        key="manual_type"
                    )
                    participants = st.text_area(
                        "Participantes (um por linha):",
                        height=100,
                        key="manual_participants"
                    )
                
                if st.button("🔍 Analisar Texto", type="primary", use_container_width=True):
                    with st.spinner("Analisando reunião..."):
                        meeting_info = f"""
                        DATA: {meeting_date.strftime('%d/%m/%Y')}
                        HORÁRIO: {meeting_time.strftime('%H:%M')}
                        TIPO: {meeting_type}
                        PARTICIPANTES: {participants}
                        """
                        
                        analysis = analyze_meeting_transcript(manual_text, meeting_info)
                        
                        st.markdown("---")
                        st.subheader("📊 Resultado da Análise")
                        st.markdown(analysis)
    
    # Página: Configurações
    elif page == "⚙️ Configurações":
        st.title("⚙️ Configurações do Sistema")
        st.markdown("---")
        
        st.subheader("Configurações de Análise")
        
        model_choice = st.selectbox(
            "Modelo de IA preferido:",
            ["Gemini", "Claude", "OpenAI"],
            index=0
        )
        
        analysis_depth = st.select_slider(
            "Profundidade da análise:",
            options=["Básica", "Padrão", "Detalhada", "Completa"],
            value="Padrão"
        )
        
        auto_extract = st.checkbox(
            "Extrair metadados automaticamente",
            value=True,
            help="Tenta extrair data, participantes e objetivos automaticamente"
        )
        
        include_web_search = st.checkbox(
            "Incluir pesquisa web para contexto",
            value=False,
            help="Busca informações adicionais na web (requer API do Perplexity)"
        )
        
        if st.button("💾 Salvar Configurações", type="primary"):
            st.success("Configurações salvas (em sessão temporária)!")
        
        st.markdown("---")
        st.subheader("Sobre o Sistema")
        
        st.info("""
        **Analisador de Reuniões IA**  
        Versão 1.0  
        
        Funcionalidades:
        - Análise detalhada de transcrições de reuniões
        - Identificação de participantes e análise comportamental
        - Detecção de decisões e ações
        - Identificação de red flags
        - Recomendações para melhorias
        
        APIs suportadas:
        - Google Gemini
        - Anthropic Claude
        - OpenAI GPT
        - Perplexity (para pesquisa web)
        """)

# ============================================================================
# ESTILOS CSS
# ============================================================================
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .analysis-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .participant-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #4CAF50;
    }
    
    .red-flag {
        background: #ffebee;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #f44336;
    }
    
    .insight-card {
        background: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #4CAF50;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    main_app()
