import json
import re
from typing import Dict

from config.config import ASTRA_DB_COLLECTION, configure_gemini
from services.embedding import get_embedding
from services.astra_client import AstraDBClient
from agent.prompts import SYSTEM_PROMPT_ANALISE, SYSTEM_PROMPT_OUTPUTS_ADICIONAIS

astra_client = AstraDBClient()
_modelo = None


def get_modelo():
    global _modelo
    if _modelo is None:
        _modelo = configure_gemini()
    return _modelo


def _build_rag_context(transcricao: str) -> str:
    embedding = get_embedding(transcricao)
    relevant_docs = astra_client.vector_search(ASTRA_DB_COLLECTION, embedding, limit=5)

    if not relevant_docs:
        return ""

    rag_context = "## CONHECIMENTO TÉCNICO RELEVANTE:\n\n"
    for i, doc in enumerate(relevant_docs, 1):
        doc_clean = str(doc).replace('{', '').replace('}', '').replace("'", "").replace('"', '')
        rag_context += f"--- Fonte {i} ---\n{doc_clean[:500]}...\n\n"
    return rag_context


def _parse_outputs_json(outputs_text: str) -> Dict:
    json_match = re.search(r'\{.*\}', outputs_text, re.DOTALL)
    if not json_match:
        return {"erro": "JSON não encontrado na resposta", "texto_original": outputs_text[:1000] + "..."}

    try:
        outputs_json = json.loads(json_match.group())
        for key, default in [
            ("acordos_combinados", []),
            ("tasks", []),
            ("entregaveis", []),
            ("proximos_passos", {}),
            ("analise_quantitativa", {"participantes": [], "estatisticas_gerais": {}}),
        ]:
            if not outputs_json.get(key):
                outputs_json[key] = default
        return outputs_json
    except json.JSONDecodeError as e:
        return {"erro": f"Falha ao parsear JSON: {str(e)}", "texto_original": outputs_text[:1000] + "..."}


def analisar_reuniao_com_rag(transcricao: str, prompt_customizado: str = "") -> Dict:
    """
    Analisa uma transcrição de reunião usando RAG.

    Args:
        transcricao: Texto da transcrição da reunião.
        prompt_customizado: Instrução adicional para personalizar a análise (ex: foco em objeções).

    Returns:
        Dict com analise_principal, outputs_json e outputs_raw.
    """
    modelo = get_modelo()

    try:
        rag_context = _build_rag_context(transcricao)

        instrucao_extra = f"\n\n## INSTRUÇÃO ADICIONAL:\n{prompt_customizado}" if prompt_customizado else ""

        prompt_analise = f"""
        {SYSTEM_PROMPT_ANALISE}

        {rag_context}

        ## TRANSCRIÇÃO DA REUNIÃO PARA ANÁLISE:
        {transcricao}
        {instrucao_extra}

        ## SUA TAREFA:

        Com base na transcrição acima e no conhecimento técnico fornecido, gere uma análise completa seguindo EXATAMENTE o formato especificado.

        IMPORTANTE: Seja específico, cite trechos da transcrição quando relevante, e dê feedback acionável.
        """

        response_analise = modelo.generate_content(prompt_analise)
        analise_principal = response_analise.text

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

        response_outputs = modelo.generate_content(prompt_outputs)
        outputs_text = response_outputs.text
        outputs_json = _parse_outputs_json(outputs_text)

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
