import streamlit as st
import google.generativeai as genai
import requests
import datetime
import os
import hashlib
from typing import List, Dict
import openai
import pandas as pd

# Configurações das credenciais
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASTRA_DB_API_ENDPOINT = os.getenv('ASTRA_DB_API_ENDPOINT')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_NAMESPACE = os.getenv('ASTRA_DB_NAMESPACE')
ASTRA_DB_COLLECTION = os.getenv('ASTRA_DB_COLLECTION')
gemini_api_key = os.getenv("GEM_API_KEY")

# Configuração inicial do Streamlit
st.set_page_config(
    layout="wide",
    page_title="Analisador de Reuniões - Sales Intelligence",
    page_icon="🎯"
)

class AstraDBClient:
    def __init__(self):
        self.base_url = f"{ASTRA_DB_API_ENDPOINT}/api/json/v1/{ASTRA_DB_NAMESPACE}"
        self.headers = {
            "Content-Type": "application/json",
            "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
            "Accept": "application/json"
        }
    
    def vector_search(self, collection: str, vector: List[float], limit: int = 8) -> List[Dict]:
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
        while len(vector) < 1536:
            vector.append(0.0)
        return vector[:1536]

# Configuração da API do Gemini
if not gemini_api_key:
    st.error("GEMINI_API_KEY não encontrada nas variáveis de ambiente")
    st.stop()

genai.configure(api_key=gemini_api_key)
modelo_analise = genai.GenerativeModel("gemini-2.5-flash")

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
        
        arquivo_bytes = arquivo.read()
        
        if len(arquivo_bytes) > 20 * 1024 * 1024:  # 20MB
            uploaded_file = client.files.upload(file=arquivo_bytes, mime_type=mime_type)
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=["Transcreva este arquivo em detalhes:", uploaded_file]
            )
        else:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    "Transcreva este arquivo em detalhes:",
                    genai.types.Part.from_bytes(data=arquivo_bytes, mime_type=mime_type)
                ]
            )
        
        return response.text
    except Exception as e:
        return f"Erro na transcrição: {str(e)}"

# --- Função para análise de reunião com RAG ---
def analisar_reuniao_com_rag(transcricao: str) -> Dict:
    """Analisa uma transcrição de reunião usando RAG e o sistema de análise de vendas"""
    
    try:
        # Gera embedding para busca na base de conhecimento
        embedding = get_embedding(transcricao)
        
        # Busca documentos relevantes no AstraDB
        relevant_docs = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=6)
        
        # Constrói contexto dos documentos
        rag_context = ""
        if relevant_docs:
            rag_context = "## 📚 CONHECIMENTO TÉCNICO DE VENDAS RELEVANTE:\n\n"
            for i, doc in enumerate(relevant_docs, 1):
                doc_content = str(doc)
                doc_clean = doc_content.replace('{', '').replace('}', '').replace("'", "").replace('"', '')
                rag_context += f"--- Fonte {i} ---\n{doc_clean[:400]}...\n\n"
        
        # Construir prompt final
        prompt_final = f"""
        {SYSTEM_PROMPT_ANALISE}
        
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
        
        **IMPORTANTE:** Seja específico, cite trechos da transcrição quando relevante, e dê feedback acionável.
        """
        
        # Gera análise
        response = modelo_analise.generate_content(prompt_final)
        
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
    
    {transcricao}
    
    Forneça um resumo conciso dos pontos principais em 3 parágrafos.
    """
    
    response = modelo_analise.generate_content(prompt)
    return response.text

# --- Interface Principal ---
st.title("🎯 Analisador de Reuniões de Vendas")
st.markdown("### Análise inteligente de calls com base em metodologias de vendas complexas")

# Menu de abas
tab_analise, tab_config = st.tabs(["🔍 Analisar Reunião", "⚙️ Configurações"])

# ========== ABA: ANÁLISE DE REUNIÃO ==========
with tab_analise:
    st.header("🔍 Analisar Reunião")
    
    # Área para transcrição
    st.subheader("📝 Forneça a Reunião para Análise")
    
    # Opções de entrada
    metodo_entrada = st.radio("Como fornecer a reunião:", 
                             ["Upload de Áudio/Video", "Texto Direto", "Gravação por Microfone"])
    
    transcricao_texto = ""
    
    if metodo_entrada == "Upload de Áudio/Video":
        arquivo_midia = st.file_uploader("Selecione arquivo de áudio ou vídeo", 
                                        type=['mp3', 'wav', 'mp4', 'mov', 'avi', 'm4a'])
        
        if arquivo_midia:
            if st.button("🎬 Transcrever e Analisar", type="primary"):
                with st.spinner("Transcrevendo e analisando..."):
                    tipo = "audio" if arquivo_midia.type.startswith('audio') else "video"
                    transcricao_texto = transcrever_audio_video(arquivo_midia, tipo)
                    
                    if "Erro" not in transcricao_texto:
                        resultado = analisar_reuniao_com_rag(transcricao_texto)
                        
                        # Mostrar resultados
                        st.success("✅ Análise concluída!")
                        
                        # Exibir análise em abas
                        tab_resumo, tab_completa, tab_metricas = st.tabs(["📋 Resumo", "📊 Análise Completa", "📈 Métricas"])
                        
                        with tab_resumo:
                            # Extrair resumo executivo
                            analise_lines = resultado["analise_completa"].split('\n')
                            in_resumo = False
                            resumo_text = []
                            
                            for line in analise_lines:
                                if "RESUMO EXECUTIVO" in line or "### 📊" in line:
                                    in_resumo = True
                                    continue
                                elif in_resumo and line.startswith("### "):
                                    break
                                elif in_resumo and line.strip():
                                    resumo_text.append(line)
                            
                            if resumo_text:
                                st.markdown("\n".join(resumo_text))
                            else:
                                st.info("Analisando conteúdo...")
                                st.write(resultado["analise_completa"][:1000] + "...")
                        
                        with tab_completa:
                            st.markdown(resultado["analise_completa"])
                        
                        with tab_metricas:
                            col_m1, col_m2, col_m3 = st.columns(3)
                            with col_m1:
                                st.metric("Score Final", f"{resultado['score']}/100")
                            with col_m2:
                                st.metric("Documentos Utilizados", resultado["documentos_relevantes"])
                            with col_m3:
                                st.metric("Status", "✅ Completo")
                            
                            # Score visual
                            score = resultado["score"]
                            if score >= 80:
                                st.success(f"🎉 Excelente performance!")
                            elif score >= 60:
                                st.info(f"👍 Boa performance")
                            else:
                                st.warning(f"⚠️ Precisa de melhorias")
                        
                        # Botões de download
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button(
                                "💾 Baixar Análise",
                                data=resultado["analise_completa"],
                                file_name=f"analise_reuniao_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                mime="text/plain"
                            )
                        with col_dl2:
                            st.download_button(
                                "💾 Baixar Transcrição",
                                data=transcricao_texto,
                                file_name=f"transcricao_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                mime="text/plain"
                            )
                    else:
                        st.error(f"Erro na transcrição: {transcricao_texto}")
    
    elif metodo_entrada == "Texto Direto":
        transcricao_texto = st.text_area("Cole a transcrição completa da reunião:", 
                                        height=300,
                                        placeholder="Vendedor: Olá, como vai? Cliente: Bem, e você?...\nVendedor: Gostaria de entender seus desafios atuais...\nCliente: Estamos com problemas de produtividade...")
        
        if transcricao_texto:
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("🚀 Análise Completa com RAG", type="primary", use_container_width=True):
                    with st.spinner("🔍 Analisando com base de conhecimento..."):
                        resultado = analisar_reuniao_com_rag(transcricao_texto)
                        
                        # Mostrar resultados
                        st.success("✅ Análise concluída!")
                        
                        # Exibir análise
                        st.markdown(resultado["analise_completa"])
                        
                        # Métricas
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.metric("Score Final", f"{resultado['score']}/100")
                        with col_m2:
                            st.metric("Documentos Utilizados", resultado["documentos_relevantes"])
                        
                        # Download
                        st.download_button(
                            "💾 Baixar Análise",
                            data=resultado["analise_completa"],
                            file_name=f"analise_reuniao_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain"
                        )
            
            with col_btn2:
                if st.button("⚡ Análise Rápida", type="secondary", use_container_width=True):
                    with st.spinner("Analisando rapidamente..."):
                        resultado = analise_rapida_reuniao(transcricao_texto)
                        st.info("📋 Análise Rápida:")
                        st.write(resultado)
    
    elif metodo_entrada == "Gravação por Microfone":
        st.info("🎤 Funcionalidade de gravação por microfone em desenvolvimento.")
        st.write("Por enquanto, use o upload de arquivo ou cole o texto diretamente.")

# ========== ABA: CONFIGURAÇÕES ==========
with tab_config:
    st.header("⚙️ Configurações")
    
    st.subheader("🔧 Status do Sistema")
    
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        # Testar Gemini
        try:
            test_response = modelo_analise.generate_content("Teste de conexão")
            st.success("✅ Gemini API: Conectado")
        except Exception as e:
            st.error(f"❌ Gemini API: {str(e)}")
    
    with col_stat2:
        # Testar AstraDB
        try:
            test_docs = astra_client.vector_search(ASTRA_DB_COLLECTION, [0.1]*1536, limit=1)
            if test_docs is not None:
                st.success(f"✅ AstraDB: Conectado")
            else:
                st.warning("⚠️ AstraDB: Conexão OK, sem documentos")
        except Exception as e:
            st.error(f"❌ AstraDB: {str(e)}")
    
    st.divider()
    
    st.subheader("📊 Informações Técnicas")
    
    st.write(f"**Modelo Gemini:** gemini-2.5-flash")
    st.write(f"**Base de Conhecimento:** {ASTRA_DB_COLLECTION}")
    st.write(f"**Última atualização:** {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}")

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
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        background-color: #f8f9fa;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Rodapé ---
st.sidebar.markdown("---")
st.sidebar.markdown("**🎯 Sales Intelligence Suite**")
st.sidebar.caption(f"v1.0 • {datetime.datetime.now().year}")
