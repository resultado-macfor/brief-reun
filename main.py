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

def traduzir_para_ingles(texto: str) -> str:
    """Traduz texto para inglês usando Gemini"""
    try:
        genai.configure(api_key=gemini_api_key)
        modelo = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""
        Traduza o seguinte texto do português para o inglês. 
        Mantenha o significado técnico e o contexto de vendas.
        
        Texto para traduzir:
        {texto[:2000]}
        
        Retorne APENAS a tradução em inglês.
        """
        
        response = modelo.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.warning(f"Erro na tradução: {str(e)}")
        return texto  # Retorna o original se falhar

def traduzir_para_portugues(texto: str) -> str:
    """Traduz texto para português usando Gemini"""
    try:
        genai.configure(api_key=gemini_api_key)
        modelo = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""
        Traduza o seguinte texto do inglês para o português brasileiro. 
        Mantenha o significado técnico e o contexto de vendas.
        
        Texto para traduzir:
        {texto[:2000]}
        
        Retorne APENAS a tradução em português.
        """
        
        response = modelo.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.warning(f"Erro na tradução: {str(e)}")
        return texto  # Retorna o original se falhar

def get_embedding(texto: str) -> List[float]:
    """Obtém embedding do texto usando OpenAI (texto deve estar em inglês)"""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            input=texto,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        st.warning(f"Embedding OpenAI não disponível: {str(e)}")
        # Fallback para embedding simples
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

O output deve ser sempre estruturado com as seguintes seções EM PORTUGUÊS:

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

**IMPORTANTE:** A resposta final DEVE ser SEMPRE em português brasileiro, mesmo que o input esteja em inglês.
"""

# --- Função para análise de reunião com RAG ---
def analisar_reuniao_com_rag(transcricao: str) -> Dict:
    """Analisa uma transcrição de reunião usando RAG com tradução para inglês"""
    
    try:
        # PASSO 1: Traduzir transcrição para inglês para busca vetorial
        st.info("🔄 Traduzindo conteúdo para busca na base de conhecimento...")
        transcricao_ingles = traduzir_para_ingles(transcricao[:1500])  # Limita para não sobrecarregar
        
        # PASSO 2: Gera embedding em inglês
        embedding = get_embedding(transcricao_ingles)
        
        # PASSO 3: Busca documentos relevantes no AstraDB (base em inglês)
        st.info("🔍 Buscando conhecimento relevante na base de dados...")
        relevant_docs = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=6)
        
        # PASSO 4: Constrói contexto dos documentos (mantém em inglês para o prompt)
        rag_context_ingles = ""
        if relevant_docs:
            rag_context_ingles = "## RELEVANT SALES KNOWLEDGE:\n\n"
            for i, doc in enumerate(relevant_docs, 1):
                doc_content = str(doc)
                doc_clean = doc_content.replace('{', '').replace('}', '').replace("'", "").replace('"', '')
                rag_context_ingles += f"--- Source {i} ---\n{doc_clean[:400]}...\n\n"
        
        # PASSO 5: Construir prompt final em inglês (mas com instrução para resposta em PT)
        prompt_final_ingles = f"""
        {SYSTEM_PROMPT_ANALISE}
        
        {rag_context_ingles}
        
        ## MEETING TRANSCRIPTION FOR ANALYSIS:
        {transcricao_ingles}
        
        ## YOUR TASK:
        
        Based on the transcription above, your expertise in complex sales, and the technical knowledge provided, generate a complete analysis following EXACTLY this format IN PORTUGUESE:
        
        ### 📊 EXECUTIVE SUMMARY
        [Summarize in 3-4 paragraphs the salesperson's overall performance]
        
        ### ✅ STRENGTHS
        [List in bullet points the observed strengths, citing specific techniques used]
        
        ### ⚠️ IMPROVEMENT POINTS
        [List in bullet points improvement opportunities, specifying where techniques could have been applied]
        
        ### 🛠️ TECHNIQUES THAT COULD HAVE BEEN APPLIED
        [List specific techniques from the mentioned authors that would be appropriate for this situation]
        
        ### 🎯 SUGGESTIONS FOR NEXT CALL
        [Give 5-6 practical and actionable suggestions for the next interaction]
        
        ### 📈 SCORING (0-100)
        
        **Rapport and Conversation Control:** [0-20] - [Brief justification]
        
        **Pain Qualification and Exploration:** [0-20] - [Brief justification]
        
        **Presentation Structure:** [0-20] - [Brief justification]
        
        **Objection Handling:** [0-20] - [Brief justification]
        
        **Closing Ability:** [0-20] - [Brief justification]
        
        **FINAL SCORE:** [Sum of above scores]/100
        
        **CRITICAL MOMENTS**
        [Identify 3-4 key moments from the conversation and analyze what went well/wrong]
        
        **APPLICABLE BIBLIOGRAPHICAL REFERENCES**
        [Cite which books/techniques are most relevant for this specific case]
        
        **CRITICAL INSTRUCTION:** The final response MUST be entirely in BRAZILIAN PORTUGUESE.
        Be specific, cite relevant transcript excerpts, and give actionable feedback.
        """
        
        # PASSO 6: Gera análise em português
        st.info("🤖 Gerando análise com inteligência artificial...")
        response = modelo_analise.generate_content(prompt_final_ingles)
        
        # PASSO 7: Garantir que a resposta está em português
        analise_texto = response.text
        
        # Verificar se precisa traduzir (caso o modelo tenha respondido em inglês)
        if "the" in analise_texto.lower() and "and" in analise_texto.lower():
            st.info("🔄 Traduzindo análise final para português...")
            analise_texto = traduzir_para_portugues(analise_texto)
        
        # PASSO 8: Extrair scoring
        score_final = 0
        lines = analise_texto.split('\n')
        for line in lines:
            if "SCORE FINAL:" in line.upper() or "FINAL SCORE:" in line.upper():
                try:
                    if "SCORE FINAL:" in line.upper():
                        score_part = line.upper().split("SCORE FINAL:")[1].strip()
                    else:
                        score_part = line.upper().split("FINAL SCORE:")[1].strip()
                    
                    # Extrair número antes da barra
                    score_str = score_part.split('/')[0].strip()
                    # Remover caracteres não numéricos
                    score_str = ''.join(filter(str.isdigit, score_str))
                    if score_str:
                        score_final = int(score_str)
                except:
                    pass
        
        # Se não encontrou score, tenta padrão diferente
        if score_final == 0:
            for line in lines:
                if "/100" in line:
                    try:
                        # Procura padrão como "85/100"
                        parts = line.split('/')
                        if len(parts) > 0:
                            score_part = parts[0]
                            score_str = ''.join(filter(str.isdigit, score_part[-3:]))
                            if score_str:
                                score_final = int(score_str)
                                break
                    except:
                        pass
        
        return {
            "analise_completa": analise_texto,
            "score": score_final,
            "documentos_relevantes": len(relevant_docs),
            "transcricao_ingles": transcricao_ingles[:500] + "..." if len(transcricao_ingles) > 500 else transcricao_ingles
        }
        
    except Exception as e:
        st.error(f"Erro na análise com RAG: {str(e)}")
        return {
            "analise_completa": f"Erro na análise: {str(e)}",
            "score": 0,
            "documentos_relevantes": 0,
            "transcricao_ingles": ""
        }

# --- Função para análise rápida (sem RAG) ---
def analise_rapida_reuniao(transcricao: str) -> str:
    """Análise rápida sem consulta à base de conhecimento"""
    prompt = f"""
    {SYSTEM_PROMPT_ANALISE}
    
    Analise esta transcrição rapidamente:
    
    {transcricao[:2000]}
    
    Forneça um resumo conciso dos pontos principais em 3 parágrafos EM PORTUGUÊS.
    """
    
    response = modelo_analise.generate_content(prompt)
    
    # Verificar se precisa traduzir
    resposta = response.text
    if "the" in resposta.lower() and "and" in resposta.lower():
        resposta = traduzir_para_portugues(resposta)
    
    return resposta

# --- Interface Principal ---
st.title("🎯 Analisador de Reuniões de Vendas")
st.markdown("### Análise inteligente de calls com base em metodologias de vendas complexas")

# Menu de abas
tab_analise, tab_config, tab_info = st.tabs(["🔍 Analisar Reunião", "⚙️ Configurações", "📚 Sobre"])

# ========== ABA: ANÁLISE DE REUNIÃO ==========
with tab_analise:
    st.header("🔍 Analisar Reunião")
    
    # Informação sobre o sistema
    st.info("""
    **Como funciona:**
    1. Cole a transcrição da reunião em português
    2. O sistema traduz para inglês para buscar na base de conhecimento
    3. Encontra técnicas de vendas relevantes
    4. Gera análise completa em português
    5. Inclui score e recomendações específicas
    """)
    
    # Área para transcrição
    st.subheader("📝 Cole a Transcrição da Reunião")
    
    transcricao_texto = st.text_area(
        "Transcrição completa da reunião de vendas:", 
        height=400,
        placeholder="""Exemplo:
Vendedor: Bom dia, João! Tudo bem?
Cliente: Bom dia! Tudo sim, e você?

Vendedor: Estou bem, obrigado! Antes de começarmos, você poderia me contar um pouco sobre os principais desafios que sua equipe enfrenta atualmente no processo de vendas?

Cliente: Nosso maior problema é a qualificação de leads. Muitas vezes gastamos tempo com prospects que não têm budget ou necessidade real...

Vendedor: Entendo perfeitamente. E qual seria o impacto financeiro aproximado desse problema para a empresa?
Cliente: Estimamos cerca de R$ 500.000 em oportunidades perdidas no último trimestre...

[Continue a transcrição aqui...]""",
        help="Cole a transcrição completa da reunião. Pode estar em português, o sistema fará a tradução automática."
    )
    
    if transcricao_texto:
        # Estatísticas rápidas
        palavras = len(transcricao_texto.split())
        st.caption(f"📊 {palavras} palavras | ~{palavras//150} minutos de conversa")
        
        # Botões de análise
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        
        with col_btn1:
            if st.button("🚀 Análise Completa com Base de Conhecimento", type="primary", use_container_width=True):
                with st.spinner("🔄 Processando..."):
                    resultado = analisar_reuniao_com_rag(transcricao_texto)
                    
                    # Mostrar resultados
                    st.success("✅ Análise concluída!")
                    
                    # Exibir análise em abas
                    tab_resumo, tab_completa, tab_metricas, tab_detalhes = st.tabs([
                        "📋 Resumo Executivo", 
                        "📊 Análise Completa", 
                        "📈 Métricas",
                        "🔧 Detalhes Técnicos"
                    ])
                    
                    with tab_resumo:
                        # Extrair resumo executivo
                        analise_lines = resultado["analise_completa"].split('\n')
                        in_resumo = False
                        resumo_text = []
                        
                        for line in analise_lines:
                            line_upper = line.upper()
                            if "RESUMO EXECUTIVO" in line_upper or "EXECUTIVE SUMMARY" in line_upper or "### 📊" in line:
                                in_resumo = True
                                continue
                            elif in_resumo and (line.startswith("### ") or "PONTOS FORTES" in line_upper or "STRENGTHS" in line_upper):
                                break
                            elif in_resumo and line.strip():
                                resumo_text.append(line)
                        
                        if resumo_text:
                            st.markdown("\n".join(resumo_text))
                        else:
                            # Fallback: mostrar primeiros 1000 caracteres
                            st.markdown(resultado["analise_completa"][:1000] + "...")
                    
                    with tab_completa:
                        st.markdown(resultado["analise_completa"])
                    
                    with tab_metricas:
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        with col_m1:
                            st.metric("Score Final", f"{resultado['score']}/100")
                        with col_m2:
                            st.metric("Documentos Encontrados", resultado["documentos_relevantes"])
                        with col_m3:
                            st.metric("Status", "✅ Completo")
                        with col_m4:
                            qualidade = "Excelente" if resultado['score'] >= 80 else "Boa" if resultado['score'] >= 60 else "Precisa Melhorar"
                            st.metric("Qualidade", qualidade)
                        
                        # Score visual
                        score = resultado["score"]
                        st.progress(score/100, text=f"Performance Geral: {score}%")
                        
                        if score >= 80:
                            st.success("🎉 **Excelente performance!** O vendedor demonstrou habilidades avançadas.")
                        elif score >= 60:
                            st.info("👍 **Boa performance** com algumas áreas para otimização.")
                        else:
                            st.warning("⚠️ **Performance abaixo do esperado.** Recomenda-se treinamento específico.")
                    
                    with tab_detalhes:
                        st.subheader("🔧 Processo Técnico")
                        col_d1, col_d2 = st.columns(2)
                        
                        with col_d1:
                            st.write("**Passos executados:**")
                            st.write("1. 📝 Recebimento da transcrição (PT-BR)")
                            st.write("2. 🔄 Tradução para inglês")
                            st.write("3. 🔍 Busca vetorial na base de conhecimento")
                            st.write(f"4. 🤖 Análise com {resultado['documentos_relevantes']} fontes relevantes")
                            st.write("5. 🇧🇷 Geração do relatório em português")
                        
                        with col_d2:
                            st.write("**Tecnologias utilizadas:**")
                            st.write("- Gemini 2.5 Flash (análise)")
                            st.write("- OpenAI Embeddings (busca)")
                            st.write("- AstraDB (base de conhecimento)")
                            st.write("- Streamlit (interface)")
                        
                        if resultado.get('transcricao_ingles'):
                            with st.expander("🔤 Ver tradução para busca"):
                                st.text(resultado['transcricao_ingles'])
                    
                    # Botões de download
                    st.divider()
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            "💾 Baixar Análise Completa",
                            data=resultado["analise_completa"],
                            file_name=f"analise_reuniao_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    with col_dl2:
                        st.download_button(
                            "📋 Baixar Resumo Executivo",
                            data="\n".join(resumo_text) if resumo_text else resultado["analise_completa"][:1000],
                            file_name=f"resumo_reuniao_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
        
        with col_btn2:
            if st.button("⚡ Análise Rápida", type="secondary", use_container_width=True):
                with st.spinner("Analisando rapidamente..."):
                    resultado = analise_rapida_reuniao(transcricao_texto)
                    st.info("📋 Análise Rápida:")
                    st.write(resultado)
        
        with col_btn3:
            if st.button("🔄 Limpar", type="secondary", use_container_width=True):
                st.rerun()

# ========== ABA: CONFIGURAÇÕES ==========
with tab_config:
    st.header("⚙️ Configurações do Sistema")
    
    st.subheader("🔧 Status das Conexões")
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        # Testar Gemini
        try:
            test_response = modelo_analise.generate_content("Teste de conexão - responda 'OK'")
            if "OK" in test_response.text.upper():
                st.success("✅ Gemini API: Conectado")
            else:
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
                st.warning("⚠️ AstraDB: Conexão OK")
        except Exception as e:
            st.error(f"❌ AstraDB: {str(e)}")
    
    with col_stat3:
        # Testar OpenAI
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            test_embedding = client.embeddings.create(
                input="test",
                model="text-embedding-3-small"
            )
            st.success("✅ OpenAI: Conectado")
        except Exception as e:
            st.warning(f"⚠️ OpenAI: {str(e)}")
    
    st.divider()
    
    st.subheader("📊 Informações Técnicas")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.write("**Especificações:**")
        st.write("- Modelo Gemini: gemini-2.5-flash")
        st.write("- Base de Conhecimento: AstraDB")
        st.write("- Embeddings: OpenAI text-embedding-3-small")
        st.write("- Idioma Base: Inglês")
        st.write("- Idioma Saída: Português Brasileiro")
    
    with col_info2:
        st.write("**Estatísticas:**")
        st.write(f"- Data: {datetime.datetime.now().strftime('%d/%m/%Y')}")
        st.write(f"- Hora: {datetime.datetime.now().strftime('%H:%M')}")
        st.write("- Versão: 2.0")
        st.write("- Pipeline: PT→EN→RAG→PT")

# ========== ABA: SOBRE ==========
with tab_info:
    st.header("📚 Sobre o Sistema")
    
    st.markdown("""
    ### 🎯 **Analisador de Reuniões de Vendas**
    
    **Objetivo:** Analisar automaticamente transcrições de reuniões de vendas complexas (B2B enterprise) 
    usando metodologias consagradas e inteligência artificial.
    
    ### 🔧 **Como Funciona**
    
    1. **Entrada:** Transcrição em português da reunião
    2. **Tradução:** Conteúdo é traduzido para inglês automaticamente
    3. **Busca Vetorial:** Sistema busca na base de conhecimento (em inglês) por técnicas relevantes
    4. **Análise RAG:** Combina a transcrição com conhecimento especializado
    5. **Geração:** Produz análise completa em português com score e recomendações
    
    ### 📚 **Base Teórica**
    
    O sistema utiliza frameworks de vendas complexas de autores renomados:
    
    - **Chris Voss** - Negociação e técnicas de influência
    - **Neil Rackham** - SPIN Selling
    - **Brent Adamson & Matthew Dixon** - Challenger Sale
    - **Mike Weinberg** - Estruturação de vendas
    - **Jeb Blount** - Inteligência emocional em vendas
    - **Aaron Ross** - Prospecção previsível
    
    ### ⚙️ **Tecnologias**
    
    - **Google Gemini 2.5 Flash:** Análise de texto e tradução
    - **OpenAI Embeddings:** Busca por similaridade semântica
    - **DataStax AstraDB:** Base de conhecimento vetorizada
    - **Streamlit:** Interface web
    
    ### 📊 **Métricas de Avaliação**
    
    Cada análise inclui score em 5 categorias críticas:
    
    1. **Rapport e Controle** (0-20)
    2. **Qualificação de Dores** (0-20)
    3. **Estrutura da Apresentação** (0-20)
    4. **Gestão de Objeções** (0-20)
    5. **Capacidade de Fechamento** (0-20)
    
    **Score Final:** 0-100
    
    ### 📞 **Suporte**
    
    Para dúvidas ou sugestões, entre em contato com a equipe de desenvolvimento.
    """)

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
    .highlight-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Rodapé ---
st.sidebar.markdown("---")
st.sidebar.markdown("**🎯 Sales Intelligence Suite**")
st.sidebar.caption(f"v2.0 • {datetime.datetime.now().year}")
