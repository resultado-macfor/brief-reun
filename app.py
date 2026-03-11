import datetime
import json
import streamlit as st

from agent.analyzer import analisar_reuniao_com_rag
from utils.dashboard import criar_dashboard_quantitativo
from utils.cards import display_task_card, display_entregavel_card, display_acordo_card

st.set_page_config(
    page_title="Analisador de Reuniões de Vendas",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Analisador de Reuniões de Vendas")
st.markdown("Cole a transcrição da reunião para receber uma análise completa com base em metodologias de vendas complexas.")

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

prompt_extra = st.text_input(
    "Instrução adicional (opcional):",
    placeholder="Ex: Foque na gestão de objeções e no fechamento",
    help="Direciona a análise para um aspecto específico da reunião."
)

if st.button("🔍 Analisar Reunião com RAG", type="primary", use_container_width=True):
    if not transcricao_texto:
        st.warning("Por favor, cole a transcrição da reunião.")
    else:
        with st.spinner("Analisando com base de conhecimento e extraindo outputs estruturados..."):
            resultados = analisar_reuniao_com_rag(transcricao_texto, prompt_extra)

        if "Erro" not in resultados["analise_principal"]:
            st.success("✅ Análise concluída!")

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
                if acordos:
                    for acordo in acordos:
                        display_acordo_card(acordo)
                else:
                    st.info("Nenhum acordo específico identificado na transcrição.")

            with tab4:
                st.markdown("## ✅ Tasks e Responsáveis")
                st.markdown("*Tarefas identificadas com responsáveis e prazos*")
                tasks = resultados.get("outputs_json", {}).get("tasks", [])
                if tasks:
                    for task in tasks:
                        display_task_card(task)
                else:
                    st.info("Nenhuma task específica identificada na transcrição.")

            with tab5:
                st.markdown("## 📦 Entregáveis Combinados")
                st.markdown("*Documentos, propostas e materiais acordados durante a reunião*")
                entregaveis = resultados.get("outputs_json", {}).get("entregaveis", [])
                if entregaveis:
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
                        for acao in acoes:
                            st.markdown(f"- {acao}")
                        if not acoes:
                            st.markdown("*Nenhuma ação imediata especificada*")

                        st.markdown("### Preparativos para Próxima Reunião")
                        preparativos = proximos_passos.get('preparativos_proxima_reuniao', [])
                        for prep in preparativos:
                            st.markdown(f"- {prep}")
                        if not preparativos:
                            st.markdown("*Nenhum preparativo especificado*")

                    with col2:
                        st.markdown("### Agenda Sugerida")
                        agenda = proximos_passos.get('agenda_sugerida', [])
                        for i, ponto in enumerate(agenda, 1):
                            st.markdown(f"{i}. {ponto}")
                        if not agenda:
                            st.markdown("*Nenhuma agenda sugerida*")

                        st.markdown("### Objetivos")
                        objetivos = proximos_passos.get('objetivos_proxima_reuniao', [])
                        for obj in objetivos:
                            st.markdown(f"🎯 {obj}")
                        if not objetivos:
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

            # Download
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

            st.download_button(
                "💾 Baixar Análise Completa",
                data=conteudo_completo,
                file_name=f"analise_completa_reuniao_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.error(resultados["analise_principal"])

# --- Rodapé ---
st.markdown("---")
st.caption(f"Analisador de Reuniões de Vendas • v4.0 com Análise Quantitativa • {datetime.datetime.now().year}")

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

    ### API REST (para integrações):
    - `POST /api/analisar` — análise completa
    - `POST /api/resumo` — análise + próximos passos (ideal para n8n)
    - `GET /health` — status do servidor
    - Docs: `http://localhost:5000/docs`
    """)
