import streamlit as st


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
                st.markdown("**📅 Prazo**")
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
