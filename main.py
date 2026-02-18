import streamlit as st
import google.generativeai as genai
import requests
import datetime
import os
from typing import List, Dict
import openai
import json
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

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
    layout="wide"
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

# --- SYSTEM PROMPTS ---
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
Com base na transcrição original da reunião de vendas E na análise RAG fornecida, extraia e estruture os seguintes outputs. É CRÍTICO que você siga o formato JSON especificado abaixo.

ATENÇÃO: A transcrição original contém informações factuais e específicas sobre acordos, tarefas, entregáveis e próximos passos. Use a análise RAG como contexto adicional, mas PRIORIZE a transcrição original para extrair informações concretas.

Formato JSON OBRIGATÓRIO:
{
    "acordos_combinados": [
        {
            "descricao": "Descrição clara do acordo verbal feito durante a reunião",
            "partes_envolvidas": ["nome/cargo da parte 1", "nome/cargo da parte 2"],
            "condicoes": "Condições específicas se houver (ex: 'sujeito a aprovação do VP')",
            "status": "pendente",
            "evidencia_transcricao": "Trecho da transcrição que comprova este acordo"
        }
    ],
    "tasks": [
        {
            "responsavel": {
                "nome": "Nome da pessoa responsável",
                "cargo": "Cargo/função identificado na transcrição",
                "contato": "Email se mencionado ou inferido do contexto"
            },
            "descricao": "Descrição clara da tarefa a ser executada",
            "prazo": "Data ou condição de prazo mencionada (ex: 'até sexta', 'semana que vem')",
            "ferramentas_necessarias": ["ferramentas mencionadas ou inferidas"],
            "entrega_final": "Descrição do que deve ser entregue ao final",
            "reportar_para": {
                "nome": "Nome de quem deve receber o reporte",
                "cargo": "Cargo dessa pessoa"
            },
            "prioridade": "alta/media/baixa (inferir do contexto)",
            "dependencias": ["descrição de tarefas que dependem desta"],
            "evidencia_transcricao": "Trecho da transcrição que menciona esta task"
        }
    ],
    "entregaveis": [
        {
            "nome": "Nome do entregável (ex: 'Proposta Comercial', 'Termo de POC')",
            "descricao": "Descrição detalhada do que deve conter",
            "responsavel_entrega": "Quem deve entregar (nome e cargo)",
            "formato_esperado": "Formato mencionado (PDF, documento, planilha, etc)",
            "prazo": "Prazo de entrega acordado",
            "destinatario": "Quem deve receber (nome e cargo)",
            "evidencia_transcricao": "Trecho da transcrição que menciona este entregável"
        }
    ],
    "proximos_passos": {
        "acoes_imediatas": ["ação1", "ação2"],
        "preparativos_proxima_reuniao": ["preparativos necessários antes da próxima reunião"],
        "agenda_sugerida": ["ponto1", "ponto2", "ponto3"],
        "objetivos_proxima_reuniao": ["objetivo1", "objetivo2"],
        "data_sugerida": "Data/horário sugerido para próxima reunião",
        "participantes_necessarios": ["participantes que devem estar presentes"]
    },
    "analise_quantitativa": {
        "participantes": [
            {
                "nome": "Nome do participante",
                "papel": "vendedor/cliente/outro",
                "metricas": {
                    "tempo_fala_segundos": 0,
                    "numero_falas": 0,
                    "palavras_por_fala": 0,
                    "perguntas_feitas": 0,
                    "objeções_levantadas": 0,
                    "acordos_propostos": 0
                },
                "qualidade_performance": {
                    "clareza_comunicacao": 0-10,
                    "escuta_ativa": 0-10,
                    "persuasao": 0-10,
                    "dominio_conteudo": 0-10,
                    "gestao_objeções": 0-10,
                    "fechamento": 0-10
                }
            }
        ],
        "estatisticas_gerais": {
            "duracao_total_segundos": 0,
            "total_falas": 0,
            "equilibrio_participacao": 0.0,
            "indice_colaboracao": 0.0,
            "densidade_informacao": 0.0
        }
    }
}

REGRAS IMPORTANTES:
1. SEMPRE inclua "evidencia_transcricao" para acordos, tasks e entregáveis, citando o trecho exato da transcrição
2. Use "não informado" apenas quando absolutamente nenhuma informação estiver disponível
3. Para tasks, identifique responsáveis mesmo que indiretamente (ex: "vou enviar" = responsável é quem fala)
4. Entregáveis são COMBINADOS na reunião - documentos, propostas, materiais que foram acordados
5. Seja extremamente fiel à transcrição original - não invente informações
6. Para análise quantitativa, estime métricas com base na transcrição (tempo de fala proporcional ao número de palavras)
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
                rag_context += f"--- Fonte {i} ---\n{doc_clean[:500]}...\n\n"
        
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
        
        # Construir prompt para outputs adicionais em formato JSON
        prompt_outputs = f"""
        {SYSTEM_PROMPT_OUTPUTS_ADICIONAIS}
        
        ## TRANSCRIÇÃO ORIGINAL DA REUNIÃO (FONTE PRIMÁRIA):
        {transcricao}
        
        ## ANÁLISE RAG DA REUNIÃO (CONTEXTO ADICIONAL):
        {analise_principal}
        
        ## BASE DE CONHECIMENTO UTILIZADA NO RAG:
        {rag_context}
        
        ## INSTRUÇÕES CRÍTICAS:
        
        1. A TRANSCRIÇÃO ORIGINAL é sua fonte primária - extraia dela todas as informações factuais
        2. Use a análise RAG apenas como contexto para entender melhor o que foi dito
        3. Para cada acordo, task e entregável, INCLUA O TRECHO EXATO da transcrição como evidência
        4. Seja extremamente detalhista - a transcrição contém muitas informações que precisam ser capturadas
        5. Identifique entregáveis como: propostas, documentos, termos, cases, budgets - tudo que foi COMBINADO entregar
        6. Para ANÁLISE QUANTITATIVA, identifique todos os participantes e atribua notas de qualidade
        
        Gere agora o JSON completo com todos os outputs estruturados baseados na transcrição original.
        """
        
        # Gera outputs adicionais
        response_outputs = modelo_analise.generate_content(prompt_outputs)
        
        # Tenta extrair JSON da resposta
        outputs_text = response_outputs.text
        json_match = re.search(r'\{.*\}', outputs_text, re.DOTALL)
        
        if json_match:
            try:
                outputs_json = json.loads(json_match.group())
                
                # Validação básica - verifica se tem os campos principais
                if not outputs_json.get("acordos_combinados"):
                    outputs_json["acordos_combinados"] = []
                if not outputs_json.get("tasks"):
                    outputs_json["tasks"] = []
                if not outputs_json.get("entregaveis"):
                    outputs_json["entregaveis"] = []
                if not outputs_json.get("proximos_passos"):
                    outputs_json["proximos_passos"] = {}
                if not outputs_json.get("analise_quantitativa"):
                    outputs_json["analise_quantitativa"] = {
                        "participantes": [],
                        "estatisticas_gerais": {}
                    }
                    
            except json.JSONDecodeError as e:
                outputs_json = {
                    "erro": f"Falha ao parsear JSON: {str(e)}", 
                    "texto_original": outputs_text[:1000] + "..."
                }
        else:
            outputs_json = {
                "erro": "JSON não encontrado na resposta", 
                "texto_original": outputs_text[:1000] + "..."
            }
        
        return {
            "analise_principal": analise_principal,
            "outputs_json": outputs_json,
            "outputs_raw": outputs_text
        }
        
    except Exception as e:
        return {
            "analise_principal": f"Erro na análise: {str(e)}",
            "outputs_json": {"erro": str(e)},
            "outputs_raw": ""
        }

def criar_dashboard_quantitativo(dados_quantitativos):
    """Cria dashboard com gráficos e análises quantitativas"""
    
    participantes = dados_quantitativos.get("participantes", [])
    estatisticas = dados_quantitativos.get("estatisticas_gerais", {})
    
    if not participantes:
        st.warning("Dados quantitativos não disponíveis para esta análise.")
        return
    
    # Métricas gerais em cards
    st.markdown("## 📊 Estatísticas Gerais da Reunião")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        duracao = estatisticas.get('duracao_total_segundos', 0)
        minutos = duracao // 60
        segundos = duracao % 60
        st.metric(
            "⏱️ Duração Total",
            f"{minutos}:{segundos:02d} min",
            help="Tempo total estimado da reunião"
        )
    
    with col2:
        st.metric(
            "💬 Total de Falas",
            estatisticas.get('total_falas', 0),
            help="Número total de intervenções na conversa"
        )
    
    with col3:
        equilibrio = estatisticas.get('equilibrio_participacao', 0)
        st.metric(
            "⚖️ Equilíbrio de Participação",
            f"{equilibrio:.1%}",
            delta=None if equilibrio > 0.3 else "Baixo equilíbrio",
            help="Quanto mais próximo de 50%, mais equilibrada a conversa"
        )
    
    with col4:
        densidade = estatisticas.get('densidade_informacao', 0)
        st.metric(
            "📈 Densidade de Informação",
            f"{densidade:.1f}",
            help="Quantidade de informação por minuto de conversa"
        )
    
    st.markdown("---")
    
    # Gráfico de tempo de fala por participante
    st.markdown("## 🎤 Distribuição de Tempo de Fala")
    
    df_tempo = pd.DataFrame([
        {
            "Participante": p["nome"],
            "Papel": p["papel"].capitalize(),
            "Tempo (minutos)": p["metricas"]["tempo_fala_segundos"] / 60,
            "Número de Falas": p["metricas"]["numero_falas"],
            "Média de Palavras por Fala": p["metricas"]["palavras_por_fala"]
        }
        for p in participantes
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_tempo = px.pie(
            df_tempo,
            values="Tempo (minutos)",
            names="Participante",
            title="Distribuição do Tempo de Fala",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        fig_tempo.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_tempo, use_container_width=True)
    
    with col2:
        fig_falas = px.bar(
            df_tempo,
            x="Participante",
            y="Número de Falas",
            color="Papel",
            title="Número de Intervenções por Participante",
            text_auto=True
        )
        fig_falas.update_layout(showlegend=True)
        st.plotly_chart(fig_falas, use_container_width=True)
    
    st.markdown("---")
    
    # Análise de qualidade por participante
    st.markdown("## ⭐ Análise de Qualidade por Participante")
    
    # Preparar dados para radar chart
    metricas_qualidade = [
        "clareza_comunicacao",
        "escuta_ativa",
        "persuasao",
        "dominio_conteudo",
        "gestao_objeções",
        "fechamento"
    ]
    
    nomes_metricas = [
        "Clareza",
        "Escuta Ativa",
        "Persuasão",
        "Domínio do Conteúdo",
        "Gestão de Objeções",
        "Fechamento"
    ]
    
    # Criar radar chart para cada participante
    tabs = st.tabs([p["nome"] for p in participantes])
    
    for idx, (tab, participante) in enumerate(zip(tabs, participantes)):
        with tab:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Radar chart
                valores = [
                    participante["qualidade_performance"].get(m, 0)
                    for m in metricas_qualidade
                ]
                
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=valores + [valores[0]],
                    theta=nomes_metricas + [nomes_metricas[0]],
                    fill='toself',
                    name=participante["nome"],
                    line_color='rgb(31, 119, 180)',
                    opacity=0.8
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10]
                        )),
                    showlegend=False,
                    title=f"Perfil de Performance - {participante['nome']}"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col2:
                st.markdown(f"### 📋 Detalhes")
                st.markdown(f"**Papel:** {participante['papel'].capitalize()}")
                st.markdown("**Métricas de Participação:**")
                st.markdown(f"- 🕐 Tempo de fala: {participante['metricas']['tempo_fala_segundos']//60}:{participante['metricas']['tempo_fala_segundos']%60:02d} min")
                st.markdown(f"- 💬 Falas: {participante['metricas']['numero_falas']}")
                st.markdown(f"- 📝 Média palavras/fala: {participante['metricas']['palavras_por_fala']:.0f}")
                st.markdown(f"- ❓ Perguntas feitas: {participante['metricas']['perguntas_feitas']}")
                st.markdown(f"- 🚫 Objeções levantadas: {participante['metricas']['objeções_levantadas']}")
                
                # Nota média
                media = sum(valores) / len(valores)
                st.markdown(f"### 🏆 Nota Média: {media:.1f}/10")
    
    st.markdown("---")
    
    # Comparativo de desempenho
    st.markdown("## 📈 Comparativo de Desempenho")
    
    # DataFrame para comparação
    df_comparativo = pd.DataFrame([
        {
            "Participante": p["nome"],
            **{nomes_metricas[i]: p["qualidade_performance"].get(m, 0) 
               for i, m in enumerate(metricas_qualidade)}
        }
        for p in participantes
    ])
    
    # Gráfico de barras agrupadas
    fig_comparativo = go.Figure()
    
    for metrica in nomes_metricas:
        fig_comparativo.add_trace(go.Bar(
            name=metrica,
            x=df_comparativo["Participante"],
            y=df_comparativo[metrica],
            text=df_comparativo[metrica],
            textposition='auto',
        ))
    
    fig_comparativo.update_layout(
        title="Comparação de Métricas por Participante",
        xaxis_title="Participante",
        yaxis_title="Nota (0-10)",
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1
    )
    
    st.plotly_chart(fig_comparativo, use_container_width=True)
    
    st.markdown("---")
    
    # Análise de interações
    st.markdown("## 🔍 Análise de Interações")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Perguntas vs Objeções
        df_interacoes = pd.DataFrame([
            {
                "Participante": p["nome"],
                "Perguntas": p["metricas"]["perguntas_feitas"],
                "Objeções": p["metricas"]["objeções_levantadas"],
                "Acordos": p["metricas"]["acordos_propostos"]
            }
            for p in participantes
        ])
        
        fig_interacoes = go.Figure()
        
        fig_interacoes.add_trace(go.Bar(
            name="Perguntas",
            x=df_interacoes["Participante"],
            y=df_interacoes["Perguntas"],
            marker_color='rgb(55, 83, 109)'
        ))
        
        fig_interacoes.add_trace(go.Bar(
            name="Objeções",
            x=df_interacoes["Participante"],
            y=df_interacoes["Objeções"],
            marker_color='rgb(219, 64, 82)'
        ))
        
        fig_interacoes.add_trace(go.Bar(
            name="Acordos",
            x=df_interacoes["Participante"],
            y=df_interacoes["Acordos"],
            marker_color='rgb(26, 118, 255)'
        ))
        
        fig_interacoes.update_layout(
            title="Tipos de Interação por Participante",
            xaxis_title="Participante",
            yaxis_title="Quantidade",
            barmode='group'
        )
        
        st.plotly_chart(fig_interacoes, use_container_width=True)
    
    with col2:
        # Scorecard resumo
        st.markdown("### 📊 Scorecard da Reunião")
        
        score_total = sum([
            p["qualidade_performance"].get("clareza_comunicacao", 0) * 0.2 +
            p["qualidade_performance"].get("escuta_ativa", 0) * 0.2 +
            p["qualidade_performance"].get("persuasao", 0) * 0.2 +
            p["qualidade_performance"].get("dominio_conteudo", 0) * 0.2 +
            p["qualidade_performance"].get("gestao_objeções", 0) * 0.1 +
            p["qualidade_performance"].get("fechamento", 0) * 0.1
            for p in participantes if p["papel"] == "vendedor"
        ])
        
        if score_total > 0:
            st.metric(
                "🎯 Efetividade do Vendedor",
                f"{score_total:.1f}/10",
                delta=None
            )
        
        # Insights automáticos
        st.markdown("### 💡 Insights Rápidos")
        
        insights = []
        
        # Verificar equilíbrio
        if estatisticas.get('equilibrio_participacao', 0) < 0.3:
            insights.append("⚠️ Conversa muito concentrada em poucos participantes")
        elif estatisticas.get('equilibrio_participacao', 0) > 0.45:
            insights.append("✅ Ótimo equilíbrio de participação")
        
        # Verificar engajamento do cliente
        for p in participantes:
            if p["papel"] == "cliente" and p["metricas"]["perguntas_feitas"] < 2:
                insights.append("⚠️ Cliente pouco questionador - pode indicar baixo engajamento")
            elif p["papel"] == "cliente" and p["metricas"]["perguntas_feitas"] > 5:
                insights.append("💪 Cliente altamente engajado - fez muitas perguntas")
        
        # Verificar objeções
        total_objeções = sum(p["metricas"]["objeções_levantadas"] for p in participantes)
        if total_objeções > 3:
            insights.append("🔄 Muitas objeções levantadas - reunião de alta complexidade")
        
        if not insights:
            insights.append("📊 Reunião dentro dos padrões esperados")
        
        for insight in insights:
            st.markdown(insight)

def display_task_card(task):
    """Exibe um card de task formatado"""
    responsavel = task.get('responsavel', {})
    reportar_para = task.get('reportar_para', {})
    evidencia = task.get('evidencia_transcricao', '')
    
    with st.container():
        with st.expander(f"✅ {task.get('descricao', 'Task sem descrição')}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                nome_resp = responsavel.get('nome', 'Não especificado')
                cargo_resp = responsavel.get('cargo', '')
                if cargo_resp:
                    st.markdown(f"👤 **Responsável:** {nome_resp} • {cargo_resp}")
                else:
                    st.markdown(f"👤 **Responsável:** {nome_resp}")
                
                ferramentas = task.get('ferramentas_necessarias', [])
                if ferramentas and ferramentas[0]:
                    st.markdown(f"🛠️ **Ferramentas:** {', '.join(ferramentas)}")
                
                entrega = task.get('entrega_final', '')
                if entrega:
                    st.markdown(f"📦 **Entrega:** {entrega}")
                
                if reportar_para and reportar_para.get('nome'):
                    nome_report = reportar_para.get('nome', '')
                    cargo_report = reportar_para.get('cargo', '')
                    if cargo_report:
                        st.markdown(f"📊 **Reportar para:** {nome_report} • {cargo_report}")
                    else:
                        st.markdown(f"📊 **Reportar para:** {nome_report}")
                
                dependencias = task.get('dependencias', [])
                if dependencias and dependencias[0]:
                    st.markdown(f"⛓️ **Depende de:** {', '.join(dependencias)}")
                
                if evidencia:
                    st.markdown("---")
                    st.markdown("📝 **Evidência na transcrição:**")
                    st.markdown(f"> *{evidencia}*")
            
            with col2:
                prazo = task.get('prazo', 'Não definido')
                st.markdown(f"**📅 Prazo**")
                st.markdown(f"**{prazo}**")
                
                prioridade = task.get('prioridade', 'media')
                if prioridade == 'alta':
                    st.markdown("🔴 **Alta Prioridade**")
                elif prioridade == 'media':
                    st.markdown("🟡 **Média Prioridade**")
                elif prioridade == 'baixa':
                    st.markdown("🟢 **Baixa Prioridade**")

def display_entregavel_card(entregavel):
    """Exibe um card de entregável formatado"""
    evidencia = entregavel.get('evidencia_transcricao', '')
    
    with st.container():
        with st.expander(f"📄 {entregavel.get('nome', 'Entregável')}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Descrição:** {entregavel.get('descricao', 'Não especificada')}")
                st.markdown(f"**Responsável:** {entregavel.get('responsavel_entrega', 'Não especificado')}")
                if evidencia:
                    st.markdown("---")
                    st.markdown("📝 **Evidência:**")
                    st.markdown(f"> *{evidencia}*")
            
            with col2:
                st.markdown(f"**Formato:** {entregavel.get('formato_esperado', 'Não especificado')}")
                st.markdown(f"**Prazo:** {entregavel.get('prazo', 'Não definido')}")
                st.markdown(f"**Destinatário:** {entregavel.get('destinatario', 'Não especificado')}")

def display_acordo_card(acordo):
    """Exibe um card de acordo formatado"""
    evidencia = acordo.get('evidencia_transcricao', '')
    
    with st.container():
        with st.expander(f"🤝 {acordo.get('descricao', 'Acordo')}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                partes = acordo.get('partes_envolvidas', [])
                if partes:
                    st.markdown(f"**Envolvidos:** {', '.join(partes)}")
                
                condicoes = acordo.get('condicoes', '')
                if condicoes:
                    st.markdown(f"**Condições:** {condicoes}")
            
            with col2:
                status = acordo.get('status', 'pendente')
                if status == 'pendente':
                    st.markdown("🟡 **Status:** Pendente")
                elif status == 'em_andamento':
                    st.markdown("🟠 **Status:** Em Andamento")
                elif status == 'concluido':
                    st.markdown("🟢 **Status:** Concluído")
            
            if evidencia:
                st.markdown("---")
                st.markdown("📝 **Evidência na transcrição:**")
                st.markdown(f"> *{evidencia}*")

# --- Interface Principal ---
st.title("🎯 Analisador de Reuniões de Vendas")
st.markdown("Cole a transcrição da reunião para receber uma análise completa com base em metodologias de vendas complexas.")

# Área para transcrição
transcricao_texto = st.text_area(
    "Transcrição da reunião:", 
    height=200,
    placeholder="""Vendedor: Bom dia! Como vai?
Cliente: Bem, obrigado!
Vendedor: Antes de começarmos, poderia me contar sobre seus principais desafios atuais?
Cliente: Temos problemas com produtividade da equipe...
[cole a transcrição completa aqui]""",
    help="Cole a transcrição completa da reunião de vendas."
)

if st.button("🔍 Analisar Reunião com RAG", type="primary", use_container_width=True):
    if transcricao_texto:
        with st.spinner("Analisando com base de conhecimento e extraindo outputs estruturados da transcrição..."):
            resultados = analisar_reuniao_com_rag(transcricao_texto)
            
            if "Erro" not in resultados["analise_principal"]:
                st.success("✅ Análise concluída!")
                
                # Criar abas para organizar os outputs
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📊 Análise Principal", 
                    "📈 Análise Quantitativa",
                    "🤝 Acordos", 
                    "✅ Tasks", 
                    "📦 Entregáveis",
                    "⏭️ Próximos Passos"
                ])
                
                with tab1:
                    st.markdown("## Análise de Performance")
                    st.markdown(resultados["analise_principal"])
                
                with tab2:
                    dados_quantitativos = resultados.get("outputs_json", {}).get("analise_quantitativa", {})
                    criar_dashboard_quantitativo(dados_quantitativos)
                
                with tab3:
                    st.markdown("## 🤝 Acordos e Combinados")
                    st.markdown("*Acordos verbais identificados na transcrição*")
                    acordos = resultados.get("outputs_json", {}).get("acordos_combinados", [])
                    
                    if acordos and len(acordos) > 0:
                        for acordo in acordos:
                            display_acordo_card(acordo)
                    else:
                        st.info("Nenhum acordo específico identificado na transcrição.")
                
                with tab4:
                    st.markdown("## ✅ Tasks e Responsáveis")
                    st.markdown("*Tarefas identificadas com responsáveis e prazos*")
                    tasks = resultados.get("outputs_json", {}).get("tasks", [])
                    
                    if tasks and len(tasks) > 0:
                        for task in tasks:
                            display_task_card(task)
                    else:
                        st.info("Nenhuma task específica identificada na transcrição.")
                
                with tab5:
                    st.markdown("## 📦 Entregáveis Combinados")
                    st.markdown("*Documentos, propostas e materiais acordados durante a reunião*")
                    entregaveis = resultados.get("outputs_json", {}).get("entregaveis", [])
                    
                    if entregaveis and len(entregaveis) > 0:
                        for entregavel in entregaveis:
                            display_entregavel_card(entregavel)
                    else:
                        st.info("Nenhum entregável específico identificado na transcrição.")
                
                with tab6:
                    st.markdown("## ⏭️ Próximos Passos")
                    st.markdown("*Encaminhamentos e agenda para continuidade*")
                    proximos_passos = resultados.get("outputs_json", {}).get("proximos_passos", {})
                    
                    if proximos_passos:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Ações Imediatas")
                            acoes = proximos_passos.get('acoes_imediatas', [])
                            if acoes:
                                for acao in acoes:
                                    st.markdown(f"- {acao}")
                            else:
                                st.markdown("*Nenhuma ação imediata especificada*")
                            
                            st.markdown("### Preparativos para Próxima Reunião")
                            preparativos = proximos_passos.get('preparativos_proxima_reuniao', [])
                            if preparativos:
                                for prep in preparativos:
                                    st.markdown(f"- {prep}")
                            else:
                                st.markdown("*Nenhum preparativo especificado*")
                        
                        with col2:
                            st.markdown("### Agenda Sugerida")
                            agenda = proximos_passos.get('agenda_sugerida', [])
                            if agenda:
                                for i, ponto in enumerate(agenda, 1):
                                    st.markdown(f"{i}. {ponto}")
                            else:
                                st.markdown("*Nenhuma agenda sugerida*")
                            
                            st.markdown("### Objetivos")
                            objetivos = proximos_passos.get('objetivos_proxima_reuniao', [])
                            if objetivos:
                                for obj in objetivos:
                                    st.markdown(f"🎯 {obj}")
                            else:
                                st.markdown("*Nenhum objetivo especificado*")
                        
                        st.markdown("---")
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            data_sugerida = proximos_passos.get('data_sugerida', '')
                            if data_sugerida:
                                st.markdown(f"**📅 Data sugerida:** {data_sugerida}")
                        
                        with col4:
                            participantes = proximos_passos.get('participantes_necessarios', [])
                            if participantes:
                                st.markdown(f"**👥 Participantes necessários:** {', '.join(participantes)}")
                    else:
                        st.info("Nenhum próximo passo específico identificado na transcrição.")
                
                # Preparar conteúdo completo para download
                conteudo_completo = f"""
===========================================
ANÁLISE DE REUNIÃO DE VENDAS
Data: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}
===========================================

===========================================
1. TRANSCRIÇÃO ORIGINAL
===========================================

{transcricao_texto}

===========================================
2. ANÁLISE PRINCIPAL (COM RAG)
===========================================

{resultados["analise_principal"]}

===========================================
3. ANÁLISE QUANTITATIVA
===========================================

{json.dumps(resultados.get("outputs_json", {}).get("analise_quantitativa", {}), indent=2, ensure_ascii=False)}

===========================================
4. OUTPUTS ESTRUTURADOS COMPLETOS
===========================================

{json.dumps(resultados.get("outputs_json", {}), indent=2, ensure_ascii=False)}
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
st.caption(f"Analisador de Reuniões de Vendas • v4.0 com Análise Quantitativa • {datetime.datetime.now().year}")

# Sidebar com instruções
with st.sidebar:
    st.header("📋 Sobre o Analisador")
    st.markdown("""
    Esta ferramenta analisa transcrições de reuniões de vendas complexas utilizando:
    
    - **RAG (Retrieval-Augmented Generation)** com base de conhecimento especializada
    - **Extração direta da transcrição** para outputs estruturados
    - **Análise quantitativa** com gráficos e métricas
    - **Evidências textuais** para cada item identificado
    
    ### Outputs Gerados:
    1. **Análise Principal**: Performance do vendedor com base em metodologias
    2. **Análise Quantitativa**: 
       - Distribuição de tempo de fala
       - Perfil de performance por participante (radar charts)
       - Comparativo de métricas
       - Insights automáticos
    3. **Acordos**: Compromissos verbais com evidências
    4. **Tasks**: Cards detalhados com responsável e prazo
    5. **Entregáveis**: Documentos e materiais COMBINADOS
    6. **Próximos Passos**: Encaminhamentos e agenda
    
    ### Diferenciais:
    - ✅ Dashboard interativo com gráficos Plotly
    - ✅ Análise comparativa entre participantes
    - ✅ Radar charts de performance individual
    - ✅ Métricas quantitativas de participação
    - ✅ Insights automáticos baseados em dados
    """)
