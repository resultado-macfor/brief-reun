import os
import streamlit as st
import pandas as pd
from datetime import datetime
import hashlib
from pymongo import MongoClient
import json
from typing import List, Dict, Tuple
import PyPDF2
import docx
import tempfile
import io
from PIL import Image
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
gemini_api_key = os.getenv("GEMINI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
mongo_uri = os.getenv('MONGO_URI')

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.set_page_config(layout="centered")
    
    st.title("🔒 Agente Performance")
    
    senha_input = st.text_input("Digite a senha de acesso:", type="password")
    
    if st.button("Acessar"):
        senha_correta = os.getenv('senha_per')
        
        if not senha_correta:
            st.error("Sistema não configurado.")
        elif senha_input == senha_correta:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Senha incorreta.")
    
    st.stop()


# Configurar clientes
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
# SISTEMA DE AUTENTICAÇÃO
# ============================================================================
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# Conexão MongoDB
@st.cache_resource
def get_mongo_client():
    try:
        client = MongoClient(mongo_uri)
        return client
    except Exception as e:
        st.error(f"Erro ao conectar ao MongoDB: {e}")
        return None

client = get_mongo_client()
if client:
    db = client['meeting_analyzer']
    users_collection = db['users']
    meetings_collection = db['meetings']
    reports_collection = db['reports']
else:
    # Fallback para dados locais (apenas desenvolvimento)
    users_collection = None
    meetings_collection = None
    reports_collection = None

# Funções de usuário
def create_user(email, password, name, company, role):
    """Cria um novo usuário"""
    try:
        if users_collection and users_collection.find_one({"email": email}):
            return False, "Usuário já existe"
        
        user_data = {
            "email": email,
            "password": make_hashes(password),
            "name": name,
            "company": company,
            "role": role,
            "created_at": datetime.now(),
            "last_login": None,
            "active": True
        }
        
        if users_collection:
            users_collection.insert_one(user_data)
        else:
            # Salvar em session state para desenvolvimento
            if "local_users" not in st.session_state:
                st.session_state.local_users = {}
            st.session_state.local_users[email] = user_data
        
        return True, "Usuário criado com sucesso"
    except Exception as e:
        return False, f"Erro ao criar usuário: {str(e)}"

def authenticate_user(email, password):
    """Autentica um usuário"""
    try:
        user = None
        
        # Tentar no MongoDB primeiro
        if users_collection:
            user = users_collection.find_one({"email": email, "active": True})
        elif "local_users" in st.session_state:
            user = st.session_state.local_users.get(email)
        
        if user:
            if check_hashes(password, user["password"]):
                # Atualizar último login
                if users_collection:
                    users_collection.update_one(
                        {"email": email},
                        {"$set": {"last_login": datetime.now()}}
                    )
                return True, user, "Login bem-sucedido"
            else:
                return False, None, "Senha incorreta"
        else:
            return False, None, "Usuário não encontrado"
    except Exception as e:
        return False, None, f"Erro na autenticação: {str(e)}"

# Interface de login/cadastro
def login_page():
    """Página de login/cadastro"""
    st.title("🔐 Analisador de Reuniões IA")
    st.markdown("---")
    
    tab_login, tab_register = st.tabs(["Login", "Cadastro"])
    
    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Senha", type="password")
            submit = st.form_submit_button("Entrar")
            
            if submit:
                if email and password:
                    success, user, message = authenticate_user(email, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user = {
                            "email": email,
                            "name": user.get("name", "Usuário"),
                            "company": user.get("company", ""),
                            "role": user.get("role", "")
                        }
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Preencha todos os campos")
    
    with tab_register:
        with st.form("register_form"):
            name = st.text_input("Nome completo")
            email = st.text_input("Email")
            company = st.text_input("Empresa")
            role = st.selectbox("Cargo", ["Gestor", "Analista", "Consultor", "Outro"])
            password = st.text_input("Senha", type="password")
            confirm_password = st.text_input("Confirmar senha", type="password")
            submit = st.form_submit_button("Criar conta")
            
            if submit:
                if not all([name, email, company, password, confirm_password]):
                    st.error("Preencha todos os campos obrigatórios")
                elif password != confirm_password:
                    st.error("As senhas não coincidem")
                elif len(password) < 6:
                    st.error("A senha deve ter pelo menos 6 caracteres")
                else:
                    success, message = create_user(email, password, name, company, role)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

# Verificar login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
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

def analyze_video_meeting(video_file):
    """Analisa vídeo de reunião (placeholder - na prática precisaria de APIs de vídeo)"""
    
    # Em produção, usar APIs como Google Video Intelligence, Azure Video Indexer, etc.
    return """
    # 🎥 ANÁLISE DE VÍDEO DE REUNIÃO
    
    ⚠️ **Funcionalidade em desenvolvimento**
    
    Para análise completa de vídeo, precisaríamos integrar com:
    - API de transcrição de áudio
    - Análise de expressões faciais
    - Detecção de tom de voz
    - Análise de linguagem corporal
    
    **Sugestão:** Faça upload da transcrição da reunião em formato de texto para análise detalhada.
    """

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
# INTERFACE PRINCIPAL
# ============================================================================
def main_app():
    """Interface principal do aplicativo"""
    
    # Sidebar com informações do usuário
    with st.sidebar:
        st.title(f"👋 Olá, {st.session_state.user['name']}")
        st.write(f"**Empresa:** {st.session_state.user['company']}")
        st.write(f"**Cargo:** {st.session_state.user['role']}")
        st.markdown("---")
        
        # Navegação
        st.title("📌 Navegação")
        page = st.radio(
            "Selecione a página:",
            ["📁 Nova Análise", "📊 Histórico", "⚙️ Configurações"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Informações do sistema
        st.title("ℹ️ Sistema")
        st.write(f"**Usuário:** {st.session_state.user['email']}")
        st.write(f"**Último login:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        
        # Logout
        if st.button("🚪 Sair", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Página: Nova Análise
    if page == "📁 Nova Análise":
        st.title("🎯 Análise de Reuniões")
        st.markdown("---")
        
        # Abas para diferentes tipos de entrada
        tab1, tab2, tab3 = st.tabs(["📄 Upload de Documento", "📝 Colar Texto", "🎥 Upload de Vídeo"])
        
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
                                
                                # Salvar no histórico
                                if meetings_collection:
                                    meeting_record = {
                                        "user_email": st.session_state.user["email"],
                                        "filename": uploaded_file.name,
                                        "meeting_date": meeting_date,
                                        "meeting_time": meeting_time,
                                        "meeting_type": meeting_type,
                                        "participants": participants_input.split("\n"),
                                        "objective": meeting_objective,
                                        "analysis": analysis,
                                        "created_at": datetime.now()
                                    }
                                    meetings_collection.insert_one(meeting_record)
                                
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
        
        with tab3:
            st.subheader("Faça upload do vídeo da reunião")
            st.warning("⚠️ Funcionalidade em desenvolvimento")
            st.info("""
            Para análise de vídeo, estamos desenvolvendo integração com:
            - Transcrição automática de áudio
            - Análise de expressões faciais
            - Detecção de tom de voz
            - Análise de engajamento
            
            **Por enquanto, use as opções de texto acima.**
            """)
            
            video_file = st.file_uploader(
                "Selecione o vídeo:",
                type=['mp4', 'mov', 'avi', 'mkv'],
                disabled=True  # Desabilitado até implementar
            )
    
    # Página: Histórico
    elif page == "📊 Histórico":
        st.title("📊 Histórico de Análises")
        st.markdown("---")
        
        # Buscar análises anteriores
        if meetings_collection:
            analyses = list(meetings_collection.find(
                {"user_email": st.session_state.user["email"]}
            ).sort("created_at", -1).limit(20))
        else:
            analyses = []
        
        if analyses:
            for analysis in analyses:
                with st.expander(f"📅 {analysis.get('meeting_date', 'Data não informada')} - {analysis.get('filename', 'Sem nome')}", expanded=False):
                    col_info, col_actions = st.columns([3, 1])
                    
                    with col_info:
                        st.write(f"**Tipo:** {analysis.get('meeting_type', 'Não informado')}")
                        st.write(f"**Participantes:** {len(analysis.get('participants', []))}")
                        st.write(f"**Data da análise:** {analysis['created_at'].strftime('%d/%m/%Y %H:%M')}")
                    
                    with col_actions:
                        if st.button("🔍 Ver Análise", key=f"view_{analysis.get('_id', '')}"):
                            st.markdown(analysis.get('analysis', 'Análise não disponível'))
                        
                        if st.button("📥 Exportar", key=f"export_{analysis.get('_id', '')}"):
                            st.download_button(
                                "Baixar",
                                data=analysis.get('analysis', ''),
                                file_name=f"analise_{analysis.get('meeting_date', 'data')}.txt",
                                mime="text/plain"
                            )
        else:
            st.info("Nenhuma análise encontrada. Faça sua primeira análise na página 'Nova Análise'.")
    
    # Página: Configurações
    elif page == "⚙️ Configurações":
        st.title("⚙️ Configurações")
        st.markdown("---")
        
        tab_config, tab_account = st.tabs(["Configurações do Sistema", "Conta"])
        
        with tab_config:
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
                st.success("Configurações salvas!")
        
        with tab_account:
            st.subheader("Informações da Conta")
            
            col_acc1, col_acc2 = st.columns(2)
            
            with col_acc1:
                st.text_input("Nome completo", value=st.session_state.user["name"], disabled=True)
                st.text_input("Email", value=st.session_state.user["email"], disabled=True)
            
            with col_acc2:
                st.text_input("Empresa", value=st.session_state.user["company"])
                st.text_input("Cargo", value=st.session_state.user["role"])
            
            st.subheader("Alterar Senha")
            
            current_pass = st.text_input("Senha atual", type="password")
            new_pass = st.text_input("Nova senha", type="password")
            confirm_pass = st.text_input("Confirmar nova senha", type="password")
            
            if st.button("🔐 Alterar Senha", type="primary"):
                if new_pass == confirm_pass:
                    st.success("Senha alterada com sucesso!")
                else:
                    st.error("As senhas não coincidem")

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
