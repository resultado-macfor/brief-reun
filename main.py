import streamlit as st
import google.generativeai as genai
import requests
import datetime
import os
from typing import List, Dict
import openai
import json

# Configurações das credenciais
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASTRA_DB_API_ENDPOINT = os.getenv('ASTRA_DB_API_ENDPOINT')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_NAMESPACE = os.getenv('ASTRA_DB_NAMESPACE')
ASTRA_DB_COLLECTION = os.getenv('ASTRA_DB_COLLECTION')
gemini_api_key = os.getenv("GEM_API_KEY")

# Configuração inicial do Streamlit
st.set_page_config(
    page_title="Analisador de Reuniões de Vendas",
    page_icon="🎯",
    layout="centered"
)

class AstraDBClient:
    def __init__(self):
        self.base_url = f"{ASTRA_DB_API_ENDPOINT}/api/json/v1/{ASTRA_DB_NAMESPACE}"
        self.headers = {
            "Content-Type": "application/json",
            "x-cassandra-token": ASTRA_DB_APPLICATION_TOKEN,
            "Accept": "application/json"
        }
    
    def vector_search(self, collection: str, vector: List[float], limit: int = 6) -> List[Dict]:
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
        except:
            return []

# Inicializa o cliente AstraDB
astra_client = AstraDBClient()

def get_embedding(texto: str) -> List[float]:
    """Obtém embedding do texto usando OpenAI"""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            input=texto,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except:
        # Fallback simples
        import hashlib
        text_hash = hashlib.md5(texto.encode()).hexdigest()
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

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT_ANALISE = """
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

Capacidade de fechamento.
"""

SYSTEM_PROMPT_OUTPUTS_ADICIONAIS = """
Com base na análise da transcrição da reunião de vendas fornecida, gere os seguintes outputs estruturados:

1. **ACORDOS E COMBINADOS**: Liste todos os acordos verbais, compromissos e combinações feitas durante a reunião entre vendedor e cliente. Seja específico sobre o que foi acordado.

2. **TASKS (TAREFAS)**: Para cada tarefa identificada, forneça:
   - Pessoa responsável (identificada pelo nome ou cargo)
   - Descrição clara da tarefa
   - Prazo (se mencionado ou sugerido)
   - Ferramentas necessárias para execução
   - Entrega final esperada
   - A quem reportar o resultado

3. **ENTREGÁVEIS**: Liste todos os materiais, documentos, propostas ou qualquer item que precise ser entregue por qualquer uma das partes, com especificações claras.

4. **PRÓXIMOS PASSOS E ATIVIDADES PARA PRÓXIMA REUNIÃO**: Descreva claramente o que deve acontecer após esta reunião, incluindo preparativos necessários, agenda sugerida para o próximo encontro e objetivos da próxima interação.

Formate a resposta com títulos claros para cada seção e use marcadores para facilitar a leitura.
"""

def analisar_reuniao_com_rag(transcricao: str) -> Dict[str, str]:
    """Analisa uma transcrição de reunião usando RAG e gera outputs adicionais"""
    
    try:
        # Gera embedding para busca na base de conhecimento
        embedding = get_embedding(transcricao)
        
        # Busca documentos relevantes no AstraDB
        relevant_docs = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=5)
        
        # Constrói contexto dos documentos
        rag_context = ""
        if relevant_docs:
            rag_context = "## CONHECIMENTO TÉCNICO RELEVANTE:\n\n"
            for i, doc in enumerate(relevant_docs, 1):
                doc_content = str(doc)
                doc_clean = doc_content.replace('{', '').replace('}', '').replace("'", "").replace('"', '')
                rag_context += f"--- Fonte {i} ---\n{doc_clean[:300]}...\n\n"
        
        # Construir prompt para análise principal
        prompt_analise = f"""
        {SYSTEM_PROMPT_ANALISE}
        
        {rag_context}
        
        ## TRANSCRIÇÃO DA REUNIÃO PARA ANÁLISE:
        {transcricao}
        
        ## SUA TAREFA:
        
        Com base na transcrição acima e no conhecimento técnico fornecido, gere uma análise completa seguindo EXATAMENTE o formato especificado.
        
        IMPORTANTE: Seja específico, cite trechos da transcrição quando relevante, e dê feedback acionável.
        """
        
        # Gera análise principal
        response_analise = modelo_analise.generate_content(prompt_analise)
        analise_principal = response_analise.text
        
        # Construir prompt para outputs adicionais baseados na análise
        prompt_outputs = f"""
        {SYSTEM_PROMPT_OUTPUTS_ADICIONAIS}
        
        ## ANÁLISE PRINCIPAL DA REUNIÃO:
        {analise_principal}
        
        ## TRANSCRIÇÃO ORIGINAL:
        {transcricao}
        
        ## BASE DE CONHECIMENTO UTILIZADA:
        {rag_context}
        
        ## SUA TAREFA:
        
        Com base na análise acima e na transcrição original, gere os outputs adicionais solicitados.
        Seja extremamente detalhista e específico. Para as tasks, sempre identifique a pessoa responsável pelo nome ou cargo mencionado na transcrição.
        Se alguma informação não estiver disponível na transcrição, indique como "Não especificado" ou sugira com base no contexto da análise.
        """
        
        # Gera outputs adicionais
        response_outputs = modelo_analise.generate_content(prompt_outputs)
        outputs_adicionais = response_outputs.text
        
        return {
            "analise_principal": analise_principal,
            "outputs_adicionais": outputs_adicionais
        }
        
    except Exception as e:
        return {
            "analise_principal": f"Erro na análise: {str(e)}",
            "outputs_adicionais": f"Erro ao gerar outputs adicionais: {str(e)}"
        }

# --- Interface Principal ---
st.title("🎯 Analisador de Reuniões de Vendas")
st.markdown("Cole a transcrição da reunião para receber uma análise completa com base em metodologias de vendas complexas.")

# Área para transcrição
transcricao_texto = st.text_area(
    "Transcrição da reunião:", 
    height=300,
    placeholder="""Vendedor: Bom dia! Como vai?
Cliente: Bem, obrigado!
Vendedor: Antes de começarmos, poderia me contar sobre seus principais desafios atuais?
Cliente: Temos problemas com produtividade da equipe...
[cole a transcrição completa aqui]""",
    help="Cole a transcrição completa da reunião de vendas."
)

if st.button("🔍 Analisar Reunião com RAG", type="primary", use_container_width=True):
    if transcricao_texto:
        with st.spinner("Analisando com base de conhecimento e gerando outputs estruturados..."):
            resultados = analisar_reuniao_com_rag(transcricao_texto)
            
            if "Erro" not in resultados["analise_principal"]:
                st.success("✅ Análise concluída!")
                
                # Criar abas para organizar os outputs
                tab1, tab2 = st.tabs(["📊 Análise Principal", "📋 Outputs Adicionais"])
                
                with tab1:
                    st.markdown("## Análise de Performance")
                    st.markdown(resultados["analise_principal"])
                
                with tab2:
                    st.markdown("## Acordos, Tasks e Próximos Passos")
                    st.markdown(resultados["outputs_adicionais"])
                
                # Preparar conteúdo completo para download
                conteudo_completo = f"""
===========================================
ANÁLISE DE REUNIÃO DE VENDAS
Data: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}
===========================================

===========================================
1. ANÁLISE PRINCIPAL
===========================================

{resultados["analise_principal"]}

===========================================
2. ACORDOS, TASKS E PRÓXIMOS PASSOS
===========================================

{resultados["outputs_adicionais"]}
                """
                
                # Botão de download
                st.download_button(
                    "💾 Baixar Análise Completa",
                    data=conteudo_completo,
                    file_name=f"analise_completa_reuniao_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.error(resultados["analise_principal"])
    else:
        st.warning("Por favor, cole a transcrição da reunião.")

# --- Rodapé ---
st.markdown("---")
st.caption(f"Analisador de Reuniões de Vendas • v2.0 com Outputs Estruturados • {datetime.datetime.now().year}")

# Sidebar com instruções
with st.sidebar:
    st.header("📋 Sobre o Analisador")
    st.markdown("""
    Esta ferramenta analisa transcrições de reuniões de vendas complexas utilizando:
    
    - **RAG (Retrieval-Augmented Generation)** com base de conhecimento especializada
    - **Metodologias** de Chris Voss, SPIN Selling, Challenger Sale e mais
    - **Outputs estruturados** para acionabilidade
    
    ### Outputs Gerados:
    1. **Análise Principal**: Performance do vendedor, pontos fortes/melhoria, score
    2. **Acordos e Combinados**: Compromissos estabelecidos
    3. **Tasks**: Tarefas com responsável, prazo e entregáveis
    4. **Entregáveis**: Materiais e documentos necessários
    5. **Próximos Passos**: Agenda para próxima reunião
    
    ### Como usar:
    1. Cole a transcrição completa
    2. Clique em "Analisar"
    3. Consulte as abas com os resultados
    4. Faça o download da análise completa
    """)
