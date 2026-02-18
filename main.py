import streamlit as st
import google.generativeai as genai
import requests
import datetime
import os
from typing import List, Dict
import openai
import json
import re

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
Com base na análise da transcrição da reunião de vendas fornecida, gere os seguintes outputs estruturados. É CRÍTICO que você siga o formato JSON especificado abaixo para que possamos exibir corretamente na interface.

Formato JSON OBRIGATÓRIO:
{
    "acordos_combinados": [
        {
            "descricao": "Descrição clara do acordo",
            "partes_envolvidas": ["parte1", "parte2"],
            "condicoes": "Condições específicas se houver",
            "status": "pendente/em_andamento/concluido"
        }
    ],
    "tasks": [
        {
            "responsavel": {
                "nome": "Nome da pessoa",
                "cargo": "Cargo/função",
                "contato": "Email se mencionado ou 'não informado'"
            },
            "descricao": "Descrição clara da tarefa",
            "prazo": "Data ou condição de prazo",
            "ferramentas_necessarias": ["ferramenta1", "ferramenta2"],
            "entrega_final": "Descrição do que deve ser entregue",
            "reportar_para": {
                "nome": "Nome de quem deve receber o reporte",
                "cargo": "Cargo dessa pessoa"
            },
            "prioridade": "alta/media/baixa",
            "dependencias": ["dependência1"] ou []
        }
    ],
    "entregaveis": [
        {
            "nome": "Nome do entregável",
            "descricao": "Descrição detalhada",
            "responsavel_entrega": "Quem deve entregar",
            "formato_esperado": "PDF, documento, proposta, etc",
            "prazo": "Prazo de entrega",
            "destinatario": "Quem deve receber"
        }
    ],
    "proximos_passos": {
        "acoes_imediatas": ["ação1", "ação2"],
        "preparativos_proxima_reuniao": ["preparativo1", "preparativo2"],
        "agenda_sugerida": ["ponto1", "ponto2", "ponto3"],
        "objetivos_proxima_reuniao": ["objetivo1", "objetivo2"],
        "data_sugerida": "Data sugerida para próxima reunião",
        "participantes_necessarios": ["participante1", "participante2"]
    }
}

REGRAS IMPORTANTES:
1. Para cada task, SEMPRE identifique o responsável com nome e cargo sempre que possível
2. Use "não informado" quando dados não estiverem disponíveis na transcrição
3. Seja específico nas descrições
4. Priorize tasks identificadas explicitamente na conversa
5. Inclua dependências entre tasks quando relevante
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
        
        # Construir prompt para outputs adicionais em formato JSON
        prompt_outputs = f"""
        {SYSTEM_PROMPT_OUTPUTS_ADICIONAIS}
        
        ## ANÁLISE PRINCIPAL DA REUNIÃO:
        {analise_principal}
        
        ## TRANSCRIÇÃO ORIGINAL:
        {transcricao}
        
        ## BASE DE CONHECIMENTO UTILIZADA:
        {rag_context}
        
        ## SUA TAREFA:
        
        Com base na análise acima e na transcrição original, gere os outputs adicionais solicitados no formato JSON especificado.
        
        IMPORTANTE: 
        - Retorne APENAS o JSON válido, sem texto adicional antes ou depois
        - Certifique-se de que o JSON está bem formatado e pode ser parseado
        - Para tasks, SEMPRE inclua responsável com nome e cargo
        - Use sua inteligência para inferir cargos quando não explicitamente mencionados
        """
        
        # Gera outputs adicionais
        response_outputs = modelo_analise.generate_content(prompt_outputs)
        
        # Tenta extrair JSON da resposta
        outputs_text = response_outputs.text
        json_match = re.search(r'\{.*\}', outputs_text, re.DOTALL)
        
        if json_match:
            try:
                outputs_json = json.loads(json_match.group())
            except:
                # Se falhar o parse, retorna o texto original
                outputs_json = {"erro": "Falha ao parsear JSON", "texto_original": outputs_text}
        else:
            outputs_json = {"erro": "JSON não encontrado na resposta", "texto_original": outputs_text}
        
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

def display_task_card(task):
    """Exibe um card de task formatado"""
    responsavel = task.get('responsavel', {})
    reportar_para = task.get('reportar_para', {})
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{task.get('descricao', 'Task sem descrição')}**")
            
            # Responsável com ícone
            nome_resp = responsavel.get('nome', 'Não especificado')
            cargo_resp = responsavel.get('cargo', '')
            if cargo_resp:
                st.markdown(f"👤 **Responsável:** {nome_resp} • {cargo_resp}")
            else:
                st.markdown(f"👤 **Responsável:** {nome_resp}")
            
            # Ferramentas
            ferramentas = task.get('ferramentas_necessarias', [])
            if ferramentas:
                st.markdown(f"🛠️ **Ferramentas:** {', '.join(ferramentas)}")
            
            # Entrega final
            entrega = task.get('entrega_final', '')
            if entrega:
                st.markdown(f"📦 **Entrega:** {entrega}")
            
            # Reportar para
            if reportar_para:
                nome_report = reportar_para.get('nome', '')
                cargo_report = reportar_para.get('cargo', '')
                if cargo_report:
                    st.markdown(f"📊 **Reportar para:** {nome_report} • {cargo_report}")
                else:
                    st.markdown(f"📊 **Reportar para:** {nome_report}")
            
            # Dependências
            dependencias = task.get('dependencias', [])
            if dependencias and dependencias[0]:
                st.markdown(f"⛓️ **Depende de:** {', '.join(dependencias)}")
        
        with col2:
            # Prazo com destaque
            prazo = task.get('prazo', 'Não definido')
            st.markdown(f"**📅 Prazo**")
            st.markdown(f"**{prazo}**")
            
            # Prioridade com cor
            prioridade = task.get('prioridade', 'media')
            if prioridade == 'alta':
                st.markdown("🔴 **Alta Prioridade**")
            elif prioridade == 'media':
                st.markdown("🟡 **Média Prioridade**")
            elif prioridade == 'baixa':
                st.markdown("🟢 **Baixa Prioridade**")
        
        st.divider()

def display_entregavel_card(entregavel):
    """Exibe um card de entregável formatado"""
    with st.container():
        st.markdown(f"### 📄 {entregavel.get('nome', 'Entregável')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Descrição:** {entregavel.get('descricao', 'Não especificada')}")
            st.markdown(f"**Responsável:** {entregavel.get('responsavel_entrega', 'Não especificado')}")
        
        with col2:
            st.markdown(f"**Formato:** {entregavel.get('formato_esperado', 'Não especificado')}")
            st.markdown(f"**Prazo:** {entregavel.get('prazo', 'Não definido')}")
            st.markdown(f"**Destinatário:** {entregavel.get('destinatario', 'Não especificado')}")
        
        st.divider()

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
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Análise Principal", 
                    "🤝 Acordos", 
                    "✅ Tasks", 
                    "📦 Entregáveis",
                    "⏭️ Próximos Passos"
                ])
                
                with tab1:
                    st.markdown("## Análise de Performance")
                    st.markdown(resultados["analise_principal"])
                
                with tab2:
                    st.markdown("## 🤝 Acordos e Combinados")
                    acordos = resultados.get("outputs_json", {}).get("acordos_combinados", [])
                    
                    if acordos and len(acordos) > 0:
                        for acordo in acordos:
                            with st.container():
                                st.markdown(f"### 📝 {acordo.get('descricao', 'Acordo')}")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    partes = acordo.get('partes_envolvidas', [])
                                    if partes:
                                        st.markdown(f"**Envolvidos:** {', '.join(partes)}")
                                
                                with col2:
                                    status = acordo.get('status', 'pendente')
                                    if status == 'pendente':
                                        st.markdown("🟡 **Status:** Pendente")
                                    elif status == 'em_andamento':
                                        st.markdown("🟠 **Status:** Em Andamento")
                                    elif status == 'concluido':
                                        st.markdown("🟢 **Status:** Concluído")
                                
                                condicoes = acordo.get('condicoes', '')
                                if condicoes:
                                    st.markdown(f"**Condições:** {condicoes}")
                                
                                st.divider()
                    else:
                        st.info("Nenhum acordo específico identificado na transcrição.")
                        
                        # Mostrar raw se disponível
                        if "acordos" in resultados.get("outputs_raw", "").lower():
                            with st.expander("Ver análise raw de acordos"):
                                st.text(resultados["outputs_raw"])
                
                with tab3:
                    st.markdown("## ✅ Tasks e Responsáveis")
                    tasks = resultados.get("outputs_json", {}).get("tasks", [])
                    
                    if tasks and len(tasks) > 0:
                        for task in tasks:
                            display_task_card(task)
                    else:
                        st.info("Nenhuma task específica identificada na transcrição.")
                        
                        # Mostrar raw se disponível
                        if "task" in resultados.get("outputs_raw", "").lower():
                            with st.expander("Ver análise raw de tasks"):
                                st.text(resultados["outputs_raw"])
                
                with tab4:
                    st.markdown("## 📦 Entregáveis")
                    entregaveis = resultados.get("outputs_json", {}).get("entregaveis", [])
                    
                    if entregaveis and len(entregaveis) > 0:
                        for entregavel in entregaveis:
                            display_entregavel_card(entregavel)
                    else:
                        st.info("Nenhum entregável específico identificado na transcrição.")
                        
                        # Mostrar raw se disponível
                        if "entreg" in resultados.get("outputs_raw", "").lower():
                            with st.expander("Ver análise raw de entregáveis"):
                                st.text(resultados["outputs_raw"])
                
                with tab5:
                    st.markdown("## ⏭️ Próximos Passos")
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
                        
                        # Informações adicionais
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
                        
                        # Mostrar raw se disponível
                        if "próximos" in resultados.get("outputs_raw", "").lower() or "proximos" in resultados.get("outputs_raw", "").lower():
                            with st.expander("Ver análise raw de próximos passos"):
                                st.text(resultados["outputs_raw"])
                
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
2. OUTPUTS ESTRUTURADOS (RAW)
===========================================

{resultados["outputs_raw"]}

===========================================
3. OUTPUTS ESTRUTURADOS (JSON)
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
st.caption(f"Analisador de Reuniões de Vendas • v3.0 com Cards de Tasks • {datetime.datetime.now().year}")

# Sidebar com instruções
with st.sidebar:
    st.header("📋 Sobre o Analisador")
    st.markdown("""
    Esta ferramenta analisa transcrições de reuniões de vendas complexas utilizando:
    
    - **RAG (Retrieval-Augmented Generation)** com base de conhecimento especializada
    - **Metodologias** de Chris Voss, SPIN Selling, Challenger Sale e mais
    - **Outputs estruturados** em formato JSON para melhor visualização
    
    ### Outputs Gerados:
    1. **Análise Principal**: Performance do vendedor, pontos fortes/melhoria, score
    2. **Acordos e Combinados**: Compromissos estabelecidos com status
    3. **Tasks**: Cards detalhados com responsável (nome e cargo), prazo, ferramentas, entrega e reporte
    4. **Entregáveis**: Cards com especificações completas
    5. **Próximos Passos**: Ações, agenda e objetivos estruturados
    
    ### Como usar:
    1. Cole a transcrição completa
    2. Clique em "Analisar"
    3. Consulte as abas com os resultados organizados
    4. Faça o download da análise completa
    """)
