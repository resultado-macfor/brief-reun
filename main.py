import streamlit as st
import io
import google.generativeai as genai
from PIL import Image
import requests
import datetime
import os
from pymongo import MongoClient
from bson import ObjectId
import json
import hashlib
from google.genai import types
import uuid
from typing import List, Dict
import openai
import pandas as pd
import csv

# Configure a API key do Perplexity (se ainda quiser manter)
try:
    from perplexity import Perplexity
    perp_api_key = os.getenv("PERP_API_KEY")
    if perp_api_key:
        perplexity_client = Perplexity(api_key=perp_api_key)
    else:
        perplexity_client = None
except:
    perplexity_client = None

# Configurações das credenciais - agora do .env fornecido
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASTRA_DB_API_ENDPOINT = os.getenv('ASTRA_DB_API_ENDPOINT')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_NAMESPACE = os.getenv('ASTRA_DB_NAMESPACE')
ASTRA_DB_COLLECTION = os.getenv('ASTRA_DB_COLLECTION')

# Outras configurações
mongo_uri = os.getenv('MONGO_URI')
senha_admin = os.getenv('SENHA_ADMIN')
senha_syn = os.getenv('SENHA_SYN')
senha_sme = os.getenv('SENHA_SME')
senha_ent = os.getenv('SENHA_ENT')
gemini_api_key = os.getenv("GEM_API_KEY")

class AstraDBClient:
    def __init__(self):
        self.base_url = f"{ASTRA_DB_API_ENDPOINT}/api/json/v1/{ASTRA_DB_NAMESPACE}"
        self.headers = {
            "Content-Type": "application/json",
            "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
            "Accept": "application/json"
        }
    
    def vector_search(self, collection: str, vector: List[float], limit: int = 10) -> List[Dict]:
        """Realiza busca por similaridade vetorial"""
        url = f"{self.base_url}/{collection}"
        payload = {
            "find": {
                "sort": {"$vector": vector},
                "options": {"limit": limit}
            }
        }
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("data", {}).get("documents", [])
        except Exception as e:
            st.error(f"Erro na busca vetorial: {str(e)}")
            return []
    
    def insert_document(self, collection: str, document: Dict) -> bool:
        """Insere um documento na coleção"""
        url = f"{self.base_url}/{collection}"
        try:
            response = requests.post(url, json=document, headers=self.headers, timeout=30)
            response.raise_for_status()
            return True
        except Exception as e:
            st.error(f"Erro ao inserir documento: {str(e)}")
            return False

# Inicializa o cliente AstraDB
astra_client = AstraDBClient()

def get_embedding(text: str) -> List[float]:
    """Obtém embedding do texto usando OpenAI"""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        st.warning(f"Embedding OpenAI não disponível: {str(e)}")
        # Fallback para embedding simples
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        vector = [float(int(text_hash[i:i+2], 16) / 255.0) for i in range(0, 32, 2)]
        # Preenche com valores aleatórios para ter 1536 dimensões
        while len(vector) < 1536:
            vector.append(0.0)
        return vector[:1536]

# Configuração inicial do Streamlit
st.set_page_config(
    layout="wide",
    page_title="Analisador de Reuniões - Sales Intelligence",
    page_icon="🎯"
)

# --- Sistema de Autenticação ---
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# Dados de usuário
users = {
    "admin": make_hashes(senha_admin),
    "SYN": make_hashes(senha_syn),
    "SME": make_hashes(senha_sme),
    "Enterprise": make_hashes(senha_ent)
}

def get_current_user():
    """Retorna o usuário atual da sessão"""
    return st.session_state.get('user', 'unknown')

def login():
    """Formulário de login"""
    st.title("🔐 Login - Analisador de Reuniões")
    with st.form("login_form"):
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if username in users and check_hashes(password, users[username]):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success("Login realizado com sucesso!")
                st.rerun()
            else:
                st.error("Usuário ou senha incorretos")

# Verificar se o usuário está logado
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# --- CONEXÃO MONGODB (após login) ---
client = MongoClient(mongo_uri)
db = client['sales_analytics']
collection_analises = db['analises_reunioes']
collection_vendedores = db['vendedores']

# Configuração da API do Gemini
if not gemini_api_key:
    st.error("GEMINI_API_KEY não encontrada nas variáveis de ambiente")
    st.stop()

genai.configure(api_key=gemini_api_key)
modelo_analise = genai.GenerativeModel("gemini-2.0-flash")
modelo_detalhado = genai.GenerativeModel("gemini-2.5-flash")

# --- SYSTEM PROMPT PARA ANÁLISE DE REUNIÕES ---
SYSTEM_PROMPT_ANALISE = """
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

Usou estratégias como "fechamento de portas" (no-oriented questions)?

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

# --- Função para transcrição de áudio/vídeo ---
def transcrever_audio_video(arquivo, tipo_arquivo):
    """Transcreve áudio ou vídeo usando a API do Gemini"""
    try:
        client = genai.Client(api_key=gemini_api_key)
        
        if tipo_arquivo == "audio":
            mime_type = f"audio/{arquivo.name.split('.')[-1]}"
        else:  # video
            mime_type = f"video/{arquivo.name.split('.')[-1]}"
        
        # Lê os bytes do arquivo
        arquivo_bytes = arquivo.read()
        
        # Para arquivos maiores, usa upload
        if len(arquivo_bytes) > 20 * 1024 * 1024:  # 20MB
            uploaded_file = client.files.upload(file=arquivo_bytes, mime_type=mime_type)
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=["Transcreva este arquivo em detalhes:", uploaded_file]
            )
        else:
            # Para arquivos menores, usa inline
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    "Transcreva este arquivo em detalhes:",
                    types.Part.from_bytes(data=arquivo_bytes, mime_type=mime_type)
                ]
            )
        
        return response.text
    except Exception as e:
        return f"Erro na transcrição: {str(e)}"

# --- Função para análise de reunião com RAG ---
def analisar_reuniao_com_rag(transcricao: str, contexto_vendedor: str = "") -> Dict:
    """Analisa uma transcrição de reunião usando RAG e o sistema de análise de vendas"""
    
    try:
        # Gera embedding para busca na base de conhecimento
        embedding = get_embedding(transcricao[:800])
        
        # Busca documentos relevantes no AstraDB
        relevant_docs = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=8)
        
        # Constrói contexto dos documentos
        rag_context = ""
        if relevant_docs:
            rag_context = "## 📚 CONHECIMENTO TÉCNICO DE VENDAS RELEVANTE:\n\n"
            for i, doc in enumerate(relevant_docs, 1):
                doc_content = str(doc)
                # Limpa e formata o documento
                doc_clean = doc_content.replace('{', '').replace('}', '').replace("'", "").replace('"', '')
                rag_context += f"--- Fonte {i} ---\n{doc_clean[:500]}...\n\n"
        else:
            rag_context = "Base de conhecimento não retornou resultados específicos para esta call."

        # Adiciona contexto do vendedor se disponível
        contexto_completo = ""
        if contexto_vendedor:
            contexto_completo += f"## 👤 CONTEXTO DO VENDEDOR:\n{contexto_vendedor}\n\n"
        
        contexto_completo += f"## 🎯 SISTEMA DE ANÁLISE DE VENDAS:\n{SYSTEM_PROMPT_ANALISE}\n\n"
        
        # Construir prompt final
        prompt_final = f"""
        {contexto_completo}
        
        {rag_context}
        
        ## 📝 TRANSCRIÇÃO DA REUNIÃO PARA ANÁLISE:
        {transcricao}
        
        ## 🎯 SUA TAREFA:
        
        Com base na transcrição acima, sua expertise em vendas complexas e o conhecimento técnico fornecido, gere uma análise completa seguindo EXATAMENTE este formato:
        
        ### 📊 RESUMO EXECUTIVO
        [Resuma em 3-4 parágrafos a performance geral do vendedor]
        
        ### ✅ PONTOS FORTES
        [Liste em bullet points os pontos fortes observados, citando técnicas específicas utilizadas]
        
        ### ⚠️ PONTOS DE MELHORIA
        [Liste em bullet points as oportunidades de melhoria, especificando onde técnicas poderiam ter sido aplicadas]
        
        ### 🛠️ TÉCNICAS QUE PODERIAM TER SIDO APLICADAS
        [Liste técnicas específicas dos autores mencionados que seriam apropriadas para esta situação]
        
        ### 🎯 SUGESTÕES PARA PRÓXIMA CALL
        [Dê 5-6 sugestões práticas e acionáveis para a próxima interação]
        
        ### 📈 SCORING (0-100)
        
        **Rapport e Controle da Conversa:** [0-20] - [Breve justificativa]
        
        **Qualificação e Exploração de Dores:** [0-20] - [Breve justificativa]
        
        **Estrutura da Apresentação:** [0-20] - [Breve justificativa]
        
        **Gestão de Objeções:** [0-20] - [Breve justificativa]
        
        **Capacidade de Fechamento:** [0-20] - [Breve justificativa]
        
        **SCORE FINAL:** [Soma dos scores acima]/100
        
        ### 🎬 CENAS CRÍTICAS
        [Identifique 3-4 momentos-chave da conversa e analise o que foi bem/ruim]
        
        ### 📚 REFERÊNCIAS BIBLIOGRÁFICAS APLICÁVEIS
        [Cite quais livros/técnicas são mais relevantes para este caso específico]
        
        **IMPORTANTE:** Seja específico, cite trechos da transcrição quando relevante, e dê feedback acionável.
        """
        
        # Gera análise
        response = modelo_detalhado.generate_content(prompt_final)
        
        # Extrair scoring
        score_final = 0
        lines = response.text.split('\n')
        for line in lines:
            if "SCORE FINAL:" in line:
                try:
                    score_part = line.split("SCORE FINAL:")[1].strip()
                    score_str = score_part.split('/')[0].strip()
                    score_final = int(score_str)
                except:
                    pass
        
        return {
            "analise_completa": response.text,
            "score": score_final,
            "documentos_relevantes": len(relevant_docs)
        }
        
    except Exception as e:
        st.error(f"Erro na análise com RAG: {str(e)}")
        return {
            "analise_completa": f"Erro na análise: {str(e)}",
            "score": 0,
            "documentos_relevantes": 0
        }

# --- Função para análise rápida (sem RAG) ---
def analise_rapida_reuniao(transcricao: str) -> str:
    """Análise rápida sem consulta à base de conhecimento"""
    prompt = f"""
    {SYSTEM_PROMPT_ANALISE}
    
    Analise esta transcrição rapidamente:
    
    {transcricao[:3000]}
    
    Forneça um resumo conciso dos pontos principais em 3 parágrafos.
    """
    
    response = modelo_analise.generate_content(prompt)
    return response.text

# --- Interface Principal ---
st.image('macLogo.png', width=300)
st.title("🎯 Analisador de Reuniões de Vendas")
st.markdown("Análise inteligente de calls com base em metodologias de vendas complexas")

# Botão de logout
if st.sidebar.button("🚪 Sair", key="logout_btn"):
    for key in ["logged_in", "user"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Mostrar usuário atual
st.sidebar.success(f"👤 Logado como: {get_current_user()}")

# Menu de abas
tab_analise, tab_vendedores, tab_historico, tab_config = st.tabs([
    "🔍 Analisar Reunião", 
    "👥 Gerenciar Vendedores", 
    "📊 Histórico de Análises",
    "⚙️ Configurações"
])

# ========== ABA: ANÁLISE DE REUNIÃO ==========
with tab_analise:
    st.header("🔍 Analisar Nova Reunião")
    
    # Seleção do vendedor
    vendedores = list(collection_vendedores.find({"ativo": True}))
    vendedor_options = {v['nome']: v['_id'] for v in vendedores}
    
    col1, col2 = st.columns(2)
    
    with col1:
        nome_vendedor = st.text_input("Nome do Vendedor*")
        empresa_cliente = st.text_input("Empresa do Cliente")
        tipo_venda = st.selectbox("Tipo de Venda", 
                                 ["Enterprise B2B", "Mid-Market", "SMB", "Renovação", "Upsell"])
        estagio_venda = st.selectbox("Estágio da Venda",
                                    ["Primeiro Contato", "Discovery", "Apresentação", 
                                     "Negociação", "Fechamento", "Follow-up"])
    
    with col2:
        duracao_call = st.number_input("Duração (minutos)", min_value=1, max_value=180, value=30)
        data_reuniao = st.date_input("Data da Reunião", value=datetime.datetime.now())
        resultado_esperado = st.selectbox("Resultado Esperado",
                                         ["Qualificação", "Demostração", "Proposta", 
                                          "Negociação", "Fechamento", "Outro"])
        canal = st.selectbox("Canal", ["Zoom", "Teams", "Google Meet", "Telefone", "Presencial"])
    
    # Área para transcrição
    st.subheader("📝 Transcrição da Reunião")
    
    # Opções de entrada
    metodo_entrada = st.radio("Como fornecer a transcrição:", 
                             ["Upload de Áudio/Video", "Texto Direto", "Gravação"])
    
    transcricao_texto = ""
    
    if metodo_entrada == "Upload de Áudio/Video":
        arquivo_midia = st.file_uploader("Selecione arquivo de áudio ou vídeo", 
                                        type=['mp3', 'wav', 'mp4', 'mov', 'avi'])
        
        if arquivo_midia:
            if st.button("🎬 Transcrever Áudio/Video"):
                with st.spinner("Transcrevendo..."):
                    tipo = "audio" if arquivo_midia.type.startswith('audio') else "video"
                    transcricao_texto = transcrever_audio_video(arquivo_midia, tipo)
                    st.success("Transcrição concluída!")
    
    elif metodo_entrada == "Texto Direto":
        transcricao_texto = st.text_area("Cole a transcrição completa da reunião:", 
                                        height=300,
                                        placeholder="Vendedor: Olá, como vai? Cliente: Bem, e você?...")
    
    # Contexto adicional
    with st.expander("🔧 Contexto Adicional (opcional)"):
        contexto_vendedor = st.text_area("Informações sobre o vendedor (estilo, experiência, etc.):")
        desafios_esperados = st.text_area("Desafios específicos esperados nesta venda:")
        produto_servico = st.text_area("Produto/Serviço sendo vendido:")
    
    # Botões de análise
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🚀 Análise Completa com RAG", type="primary", use_container_width=True):
            if transcricao_texto and nome_vendedor:
                with st.spinner("🔍 Analisando com base de conhecimento..."):
                    resultado = analisar_reuniao_com_rag(transcricao_texto, contexto_vendedor)
                    
                    # Salvar análise no banco de dados
                    analise_data = {
                        "vendedor": nome_vendedor,
                        "empresa_cliente": empresa_cliente,
                        "tipo_venda": tipo_venda,
                        "estagio_venda": estagio_venda,
                        "data_reuniao": datetime.datetime.combine(data_reuniao, datetime.datetime.min.time()),
                        "duracao_minutos": duracao_call,
                        "resultado_esperado": resultado_esperado,
                        "canal": canal,
                        "transcricao": transcricao_texto[:5000],  # Limitar tamanho
                        "contexto_vendedor": contexto_vendedor,
                        "analise_completa": resultado["analise_completa"],
                        "score_final": resultado["score"],
                        "documentos_utilizados": resultado["documentos_relevantes"],
                        "data_analise": datetime.datetime.now(),
                        "analista": get_current_user()
                    }
                    
                    collection_analises.insert_one(analise_data)
                    
                    # Mostrar resultados
                    st.success("✅ Análise concluída e salva!")
                    
                    # Exibir análise em abas
                    tab_resumo, tab_completa, tab_metricas = st.tabs(["📋 Resumo", "📊 Análise Completa", "📈 Métricas"])
                    
                    with tab_resumo:
                        # Extrair resumo executivo
                        analise_lines = resultado["analise_completa"].split('\n')
                        in_resumo = False
                        resumo_text = []
                        
                        for line in analise_lines:
                            if "### 📊 RESUMO EXECUTIVO" in line:
                                in_resumo = True
                                continue
                            elif in_resumo and line.startswith("### "):
                                break
                            elif in_resumo and line.strip():
                                resumo_text.append(line)
                        
                        st.markdown("\n".join(resumo_text))
                    
                    with tab_completa:
                        st.markdown(resultado["analise_completa"])
                    
                    with tab_metricas:
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("Score Final", f"{resultado['score']}/100")
                        with col_m2:
                            st.metric("Documentos Utilizados", resultado["documentos_relevantes"])
                        with col_m3:
                            st.metric("Duração", f"{duracao_call}min")
                        
                        # Gráfico de scores (simplificado)
                        scores_text = resultado["analise_completa"]
                        scores = {}
                        
                        for category in ["Rapport", "Qualificação", "Estrutura", "Gestão", "Fechamento"]:
                            for line in scores_text.split('\n'):
                                if category.lower() in line.lower() and "[" in line and "]" in line:
                                    try:
                                        score_part = line.split("[")[1].split("]")[0]
                                        score = int(score_part.split("-")[0].strip())
                                        scores[category] = score
                                    except:
                                        pass
                        
                        if scores:
                            df_scores = pd.DataFrame({
                                'Categoria': list(scores.keys()),
                                'Score': list(scores.values())
                            })
                            st.bar_chart(df_scores.set_index('Categoria'))
                    
                    # Botões de download
                    st.download_button(
                        "💾 Baixar Análise Completa",
                        data=resultado["analise_completa"],
                        file_name=f"analise_{nome_vendedor}_{data_reuniao.strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
            else:
                st.warning("Preencha pelo menos o nome do vendedor e forneça a transcrição")
    
    with col_btn2:
        if st.button("⚡ Análise Rápida", type="secondary", use_container_width=True):
            if transcricao_texto:
                with st.spinner("Analisando rapidamente..."):
                    resultado = analise_rapida_reuniao(transcricao_texto)
                    st.info("📋 Análise Rápida:")
                    st.write(resultado)
            else:
                st.warning("Forneça a transcrição primeiro")

# ========== ABA: GERENCIAR VENDEDORES ==========
with tab_vendedores:
    st.header("👥 Gerenciar Vendedores")
    
    # Subabas
    tab_criar, tab_editar, tab_listar = st.tabs(["➕ Criar", "✏️ Editar", "📋 Listar"])
    
    with tab_criar:
        with st.form("form_criar_vendedor"):
            nome = st.text_input("Nome Completo*")
            email = st.text_input("Email")
            experiencia = st.selectbox("Experiência", 
                                     ["Junior (<2 anos)", "Pleno (2-5 anos)", "Sênior (5+ anos)", "Líder (10+ anos)"])
            especialidades = st.multiselect("Especialidades",
                                          ["Enterprise Sales", "SMB", "Renovação", "Upsell/Cross-sell", 
                                           "Novos Negócios", "Contas Estratégicas", "Vendas Técnicas"])
            estilo_vendas = st.selectbox("Estilo de Vendas",
                                       ["Challenger", "Consultor", "Relacionamento", "Solution Seller", "Hunter"])
            metas = st.text_area("Metas e Objetivos")
            pontos_fortes = st.text_area("Pontos Fortes Conhecidos")
            areas_melhoria = st.text_area("Áreas para Melhoria")
            
            if st.form_submit_button("✅ Criar Vendedor"):
                if nome:
                    vendedor_data = {
                        "nome": nome,
                        "email": email,
                        "experiencia": experiencia,
                        "especialidades": especialidades,
                        "estilo_vendas": estilo_vendas,
                        "metas": metas,
                        "pontos_fortes": pontos_fortes,
                        "areas_melhoria": areas_melhoria,
                        "data_cadastro": datetime.datetime.now(),
                        "ativo": True
                    }
                    
                    collection_vendedores.insert_one(vendedor_data)
                    st.success(f"Vendedor {nome} criado com sucesso!")
                else:
                    st.error("Nome é obrigatório!")
    
    with tab_listar:
        vendedores = list(collection_vendedores.find({"ativo": True}))
        
        if vendedores:
            for vendedor in vendedores:
                with st.expander(f"👤 {vendedor['nome']} - {vendedor['experiencia']}", expanded=False):
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.write(f"**Email:** {vendedor.get('email', 'Não informado')}")
                        st.write(f"**Estilo:** {vendedor.get('estilo_vendas', 'Não definido')}")
                        st.write(f"**Especialidades:** {', '.join(vendedor.get('especialidades', []))}")
                    
                    with col_info2:
                        st.write(f"**Cadastro:** {vendedor['data_cadastro'].strftime('%d/%m/%Y')}")
                        st.write(f"**Status:** {'✅ Ativo' if vendedor.get('ativo', True) else '❌ Inativo'}")
                    
                    if st.button(f"Desativar {vendedor['nome']}", key=f"desativar_{vendedor['_id']}"):
                        collection_vendedores.update_one(
                            {"_id": vendedor["_id"]},
                            {"$set": {"ativo": False}}
                        )
                        st.success(f"Vendedor {vendedor['nome']} desativado!")
                        st.rerun()
        else:
            st.info("Nenhum vendedor cadastrado")

# ========== ABA: HISTÓRICO DE ANÁLISES ==========
with tab_historico:
    st.header("📊 Histórico de Análises")
    
    # Filtros
    col_filtro1, col_filtro2, col_filtro3 = st.columns(3)
    
    with col_filtro1:
        filtro_vendedor = st.selectbox("Filtrar por vendedor", 
                                      ["Todos"] + [v['nome'] for v in collection_vendedores.find({"ativo": True})])
    
    with col_filtro2:
        data_inicio = st.date_input("Data inicial", 
                                   value=datetime.datetime.now() - datetime.timedelta(days=30))
    
    with col_filtro3:
        data_fim = st.date_input("Data final", value=datetime.datetime.now())
    
    # Aplicar filtros
    query = {}
    
    if filtro_vendedor != "Todos":
        query["vendedor"] = filtro_vendedor
    
    query["data_reuniao"] = {
        "$gte": datetime.datetime.combine(data_inicio, datetime.datetime.min.time()),
        "$lte": datetime.datetime.combine(data_fim, datetime.datetime.max.time())
    }
    
    # Buscar análises
    analises = list(collection_analises.find(query).sort("data_reuniao", -1).limit(50))
    
    if analises:
        st.write(f"**{len(analises)} análises encontradas**")
        
        # Métricas gerais
        if analises:
            scores = [a.get('score_final', 0) for a in analises if 'score_final' in a]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            col_met1, col_met2, col_met3 = st.columns(3)
            with col_met1:
                st.metric("Média de Score", f"{avg_score:.1f}/100")
            with col_met2:
                st.metric("Total de Análises", len(analises))
            with col_met3:
                melhor_score = max(scores) if scores else 0
                st.metric("Melhor Score", f"{melhor_score}/100")
        
        # Lista de análises
        for analise in analises:
            with st.expander(f"📅 {analise['data_reuniao'].strftime('%d/%m/%Y')} - {analise['vendedor']} - {analise.get('empresa_cliente', 'N/A')}", expanded=False):
                col_det1, col_det2 = st.columns([2, 1])
                
                with col_det1:
                    st.write(f"**Vendedor:** {analise['vendedor']}")
                    st.write(f"**Cliente:** {analise.get('empresa_cliente', 'Não informado')}")
                    st.write(f"**Estágio:** {analise.get('estagio_venda', 'N/A')}")
                    st.write(f"**Duração:** {analise.get('duracao_minutos', 'N/A')}min")
                
                with col_det2:
                    score = analise.get('score_final', 0)
                    st.metric("Score", f"{score}/100")
                    st.write(f"**Analista:** {analise.get('analista', 'N/A')}")
                
                # Botão para ver análise completa
                if st.button("Ver Análise Completa", key=f"ver_{analise['_id']}"):
                    st.markdown(analise.get('analise_completa', 'Análise não disponível'))
                
                # Botão para deletar (apenas admin)
                if get_current_user() == "admin":
                    if st.button("🗑️ Deletar", key=f"del_{analise['_id']}"):
                        collection_analises.delete_one({"_id": analise["_id"]})
                        st.success("Análise deletada!")
                        st.rerun()
    else:
        st.info("Nenhuma análise encontrada com os filtros aplicados")

# ========== ABA: CONFIGURAÇÕES ==========
with tab_config:
    st.header("⚙️ Configurações")
    
    if get_current_user() != "admin":
        st.warning("Apenas administradores podem acessar esta seção")
    else:
        st.subheader("🔧 Configuração da Base de Conhecimento")
        
        # Upload de documentos para a base de conhecimento
        st.write("Adicionar novos documentos à base de conhecimento:")
        
        doc_texto = st.text_area("Cole o texto do documento técnico:", 
                                height=200,
                                placeholder="Ex: Técnicas de SPIN Selling...\nCapítulo 1...")
        
        doc_titulo = st.text_input("Título do documento:")
        doc_autor = st.text_input("Autor/Fonte:")
        doc_tipo = st.selectbox("Tipo de conteúdo", 
                               ["Livro", "Artigo", "Case Study", "Metodologia", "Framework"])
        
        if st.button("➕ Adicionar à Base de Conhecimento"):
            if doc_texto and doc_titulo:
                # Gerar embedding
                embedding = get_embedding(doc_texto)
                
                # Criar documento para AstraDB
                documento = {
                    "titulo": doc_titulo,
                    "autor": doc_autor,
                    "tipo": doc_tipo,
                    "conteudo": doc_texto[:2000],  # Limitar tamanho
                    "vector": embedding,
                    "data_adicao": datetime.datetime.now().isoformat(),
                    "adicionado_por": get_current_user()
                }
                
                # Inserir no AstraDB
                if astra_client.insert_document(ASTRA_DB_COLLECTION, documento):
                    st.success("✅ Documento adicionado à base de conhecimento!")
                else:
                    st.error("❌ Erro ao adicionar documento")
            else:
                st.warning("Preencha pelo menos o título e o conteúdo do documento")
        
        st.divider()
        
        # Estatísticas da base
        st.subheader("📊 Estatísticas da Base")
        
        # Testar conexão com AstraDB
        if st.button("🔄 Testar Conexão AstraDB"):
            try:
                test_docs = astra_client.vector_search(ASTRA_DB_COLLECTION, [0.1]*1536, limit=1)
                if test_docs is not None:
                    st.success(f"✅ Conexão bem sucedida! Base operacional.")
                else:
                    st.warning("⚠️ Conexão OK, mas sem documentos encontrados")
            except Exception as e:
                st.error(f"❌ Erro na conexão: {str(e)}")

# --- Estilos CSS ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    .score-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .score-medium {
        color: #FF9800;
        font-weight: bold;
    }
    .score-low {
        color: #F44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Rodapé ---
st.sidebar.markdown("---")
st.sidebar.markdown("**🎯 Sales Intelligence Suite**")
st.sidebar.caption(f"v1.0 • {datetime.datetime.now().year}")
