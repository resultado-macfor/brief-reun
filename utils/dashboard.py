import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


METRICAS_QUALIDADE = [
    "clareza_comunicacao",
    "escuta_ativa",
    "persuasao",
    "dominio_conteudo",
    "gestao_objeções",
    "fechamento"
]

NOMES_METRICAS = [
    "Clareza",
    "Escuta Ativa",
    "Persuasão",
    "Domínio do Conteúdo",
    "Gestão de Objeções",
    "Fechamento"
]


def _render_estatisticas_gerais(estatisticas: dict):
    st.markdown("## 📊 Estatísticas Gerais da Reunião")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        duracao = estatisticas.get('duracao_total_segundos', 0)
        st.metric("⏱️ Duração Total", f"{duracao // 60}:{duracao % 60:02d} min",
                  help="Tempo total estimado da reunião")

    with col2:
        st.metric("💬 Total de Falas", estatisticas.get('total_falas', 0),
                  help="Número total de intervenções na conversa")

    with col3:
        equilibrio = estatisticas.get('equilibrio_participacao', 0)
        st.metric("⚖️ Equilíbrio de Participação", f"{equilibrio:.1%}",
                  delta=None if equilibrio > 0.3 else "Baixo equilíbrio",
                  help="Quanto mais próximo de 50%, mais equilibrada a conversa")

    with col4:
        densidade = estatisticas.get('densidade_informacao', 0)
        st.metric("📈 Densidade de Informação", f"{densidade:.1f}",
                  help="Quantidade de informação por minuto de conversa")


def _render_tempo_fala(participantes: list):
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
        fig = px.pie(df_tempo, values="Tempo (minutos)", names="Participante",
                     title="Distribuição do Tempo de Fala",
                     color_discrete_sequence=px.colors.qualitative.Set3, hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(df_tempo, x="Participante", y="Número de Falas", color="Papel",
                     title="Número de Intervenções por Participante", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)


def _render_qualidade_participantes(participantes: list):
    st.markdown("## ⭐ Análise de Qualidade por Participante")
    tabs = st.tabs([p["nome"] for p in participantes])

    for tab, participante in zip(tabs, participantes):
        with tab:
            col1, col2 = st.columns([2, 1])
            valores = [participante["qualidade_performance"].get(m, 0) for m in METRICAS_QUALIDADE]

            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=valores + [valores[0]],
                    theta=NOMES_METRICAS + [NOMES_METRICAS[0]],
                    fill='toself',
                    name=participante["nome"],
                    line_color='rgb(31, 119, 180)',
                    opacity=0.8
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                    showlegend=False,
                    title=f"Perfil de Performance - {participante['nome']}"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                m = participante["metricas"]
                st.markdown(f"### 📋 Detalhes")
                st.markdown(f"**Papel:** {participante['papel'].capitalize()}")
                st.markdown("**Métricas de Participação:**")
                st.markdown(f"- 🕐 Tempo de fala: {m['tempo_fala_segundos']//60}:{m['tempo_fala_segundos']%60:02d} min")
                st.markdown(f"- 💬 Falas: {m['numero_falas']}")
                st.markdown(f"- 📝 Média palavras/fala: {m['palavras_por_fala']:.0f}")
                st.markdown(f"- ❓ Perguntas feitas: {m['perguntas_feitas']}")
                st.markdown(f"- 🚫 Objeções levantadas: {m['objeções_levantadas']}")
                media = sum(valores) / len(valores)
                st.markdown(f"### 🏆 Nota Média: {media:.1f}/10")


def _render_comparativo(participantes: list):
    st.markdown("## 📈 Comparativo de Desempenho")

    df = pd.DataFrame([
        {"Participante": p["nome"],
         **{NOMES_METRICAS[i]: p["qualidade_performance"].get(m, 0)
            for i, m in enumerate(METRICAS_QUALIDADE)}}
        for p in participantes
    ])

    fig = go.Figure()
    for metrica in NOMES_METRICAS:
        fig.add_trace(go.Bar(name=metrica, x=df["Participante"], y=df[metrica],
                             text=df[metrica], textposition='auto'))
    fig.update_layout(title="Comparação de Métricas por Participante",
                      xaxis_title="Participante", yaxis_title="Nota (0-10)",
                      barmode='group', bargap=0.15, bargroupgap=0.1)
    st.plotly_chart(fig, use_container_width=True)


def _render_interacoes(participantes: list, estatisticas: dict):
    st.markdown("## 🔍 Análise de Interações")
    col1, col2 = st.columns(2)

    with col1:
        df = pd.DataFrame([
            {"Participante": p["nome"],
             "Perguntas": p["metricas"]["perguntas_feitas"],
             "Objeções": p["metricas"]["objeções_levantadas"],
             "Acordos": p["metricas"]["acordos_propostos"]}
            for p in participantes
        ])

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Perguntas", x=df["Participante"], y=df["Perguntas"],
                             marker_color='rgb(55, 83, 109)'))
        fig.add_trace(go.Bar(name="Objeções", x=df["Participante"], y=df["Objeções"],
                             marker_color='rgb(219, 64, 82)'))
        fig.add_trace(go.Bar(name="Acordos", x=df["Participante"], y=df["Acordos"],
                             marker_color='rgb(26, 118, 255)'))
        fig.update_layout(title="Tipos de Interação por Participante",
                          xaxis_title="Participante", yaxis_title="Quantidade",
                          barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📊 Scorecard da Reunião")

        score_total = sum(
            p["qualidade_performance"].get("clareza_comunicacao", 0) * 0.2 +
            p["qualidade_performance"].get("escuta_ativa", 0) * 0.2 +
            p["qualidade_performance"].get("persuasao", 0) * 0.2 +
            p["qualidade_performance"].get("dominio_conteudo", 0) * 0.2 +
            p["qualidade_performance"].get("gestao_objeções", 0) * 0.1 +
            p["qualidade_performance"].get("fechamento", 0) * 0.1
            for p in participantes if p["papel"] == "vendedor"
        )

        if score_total > 0:
            st.metric("🎯 Efetividade do Vendedor", f"{score_total:.1f}/10")

        st.markdown("### 💡 Insights Rápidos")
        insights = []

        equilibrio = estatisticas.get('equilibrio_participacao', 0)
        if equilibrio < 0.3:
            insights.append("⚠️ Conversa muito concentrada em poucos participantes")
        elif equilibrio > 0.45:
            insights.append("✅ Ótimo equilíbrio de participação")

        for p in participantes:
            if p["papel"] == "cliente":
                perguntas = p["metricas"]["perguntas_feitas"]
                if perguntas < 2:
                    insights.append("⚠️ Cliente pouco questionador - pode indicar baixo engajamento")
                elif perguntas > 5:
                    insights.append("💪 Cliente altamente engajado - fez muitas perguntas")

        total_objeções = sum(p["metricas"]["objeções_levantadas"] for p in participantes)
        if total_objeções > 3:
            insights.append("🔄 Muitas objeções levantadas - reunião de alta complexidade")

        if not insights:
            insights.append("📊 Reunião dentro dos padrões esperados")

        for insight in insights:
            st.markdown(insight)


def criar_dashboard_quantitativo(dados_quantitativos: dict):
    """Cria dashboard com gráficos e análises quantitativas"""
    participantes = dados_quantitativos.get("participantes", [])
    estatisticas = dados_quantitativos.get("estatisticas_gerais", {})

    if not participantes:
        st.warning("Dados quantitativos não disponíveis para esta análise.")
        return

    _render_estatisticas_gerais(estatisticas)
    st.markdown("---")
    _render_tempo_fala(participantes)
    st.markdown("---")
    _render_qualidade_participantes(participantes)
    st.markdown("---")
    _render_comparativo(participantes)
    st.markdown("---")
    _render_interacoes(participantes, estatisticas)
