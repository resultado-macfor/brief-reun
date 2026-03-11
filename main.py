import datetime

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request
from flask_restx import Api, Resource, fields

from auth.jwt_handler import gerar_token
from auth.middleware import requer_token
from agent.analyzer import analisar_reuniao_com_rag
from config.config import API_USERS, JWT_EXPIRATION_HOURS

# App e Swagger
app = Flask(__name__)

authorizations = {
    "BearerAuth": {
        "type": "apiKey",
        "in": "header",
        "name": "Authorization",
        "description": "Informe: Bearer &lt;token&gt;"
    }
}

api = Api(
    app,
    version="1.0",
    title="API para Análise de Reuniões de Vendas",
    description=(
        "Análise de transcrições de reuniões de venda usando RAG + Gemini.\n\n"
        "**Autenticação:** obtenha um token em `POST /auth/token` e envie no header "
        "`Authorization: Bearer <token>` nas demais rotas."
    ),
    doc="/docs",
    authorizations=authorizations,
    security="BearerAuth"
)

ns_auth = api.namespace("auth", description="Autenticação JWT")
ns_api  = api.namespace("api",  description="Endpoints de análise")

login_model = api.model("Login", {
    "usuario": fields.String(required=True, example="admin"),
    "senha":   fields.String(required=True, example="admin123")
})

token_response = api.model("TokenResponse", {
    "token":     fields.String(description="JWT Bearer token"),
    "expira_em": fields.String(description="Data/hora UTC de expiração (ISO 8601)")
})

analise_request = api.model("AnaliseRequest", {
    "transcricao": fields.String(
        required=True,
        description="Texto completo da transcrição da reunião",
        example="Vendedor: Bom dia! Cliente: Olá, tudo bem?"
    ),
    "prompt": fields.String(
        required=False,
        description="Instrução adicional para direcionar a análise (opcional)",
        example="Foque nas objeções e no fechamento"
    )
})

resumo_response = api.model("ResumoResponse", {
    "analise":         fields.String(description="Análise textual da reunião"),
    "proximos_passos": fields.Raw(description="Próximos passos estruturados")
})

analise_completa_response = api.model("AnaliseCompletaResponse", {
    "analise_principal":    fields.String(description="Análise textual detalhada"),
    "acordos_combinados":   fields.List(fields.Raw(), description="Acordos verbais identificados"),
    "tasks":                fields.List(fields.Raw(), description="Tarefas com responsáveis e prazos"),
    "entregaveis":          fields.List(fields.Raw(), description="Documentos e materiais acordados"),
    "proximos_passos":      fields.Raw(description="Encaminhamentos e agenda"),
    "analise_quantitativa": fields.Raw(description="Métricas de participação e performance")
})

erro_response = api.model("Erro", {
    "erro": fields.String(description="Descrição do erro")
})

@api.route("/health")
class Health(Resource):
    @api.doc(security=None, description="Verifica se a API está no ar. Não requer autenticação.")
    @api.response(200, "API operacional")
    def get(self):
        """Status da API"""
        return {"status": "ok"}, 200

@ns_auth.route("/token")
class Token(Resource):
    @ns_auth.doc(
        security=None,
        description=(
            "Autentica com usuário e senha e retorna um JWT Bearer token.\n\n"
            "O token deve ser enviado no header `Authorization: Bearer <token>` "
            "em todas as rotas protegidas.\n\n"
            "**Credenciais padrão (dev):** `admin` / `admin123`  \n"
            "Configure usuários via variável de ambiente `API_USERS` (JSON):\n"
            '`API_USERS=\'{"usuario": "senha", "outro": "outrasenha"}\'`'
        )
    )
    @ns_auth.expect(login_model, validate=True)
    @ns_auth.response(200, "Token gerado com sucesso", token_response)
    @ns_auth.response(401, "Credenciais inválidas", erro_response)
    def post(self):
        """Gera token JWT"""
        data    = request.get_json(silent=True) or {}
        usuario = (data.get("usuario") or "").strip()
        senha   = (data.get("senha") or "").strip()

        senha_correta = API_USERS.get(usuario)
        if not senha_correta or senha != senha_correta:
            return {"erro": "Usuário ou senha inválidos"}, 401

        token  = gerar_token(usuario)
        expira = (
            datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRATION_HOURS)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        return {"token": token, "expira_em": expira}, 200

@ns_api.route("/analisar")
class Analisar(Resource):
    @ns_api.doc(
        security="BearerAuth",
        description=(
            "Análise completa da transcrição via RAG + Gemini.\n\n"
            "Retorna:\n"
            "- `analise_principal` — texto detalhado de performance do vendedor\n"
            "- `acordos_combinados` — acordos verbais com evidências da transcrição\n"
            "- `tasks` — tarefas com responsável, prazo e prioridade\n"
            "- `entregaveis` — documentos e materiais acordados\n"
            "- `proximos_passos` — encaminhamentos e agenda\n"
            "- `analise_quantitativa` — métricas de participação e radar de performance"
        )
    )
    @ns_api.expect(analise_request, validate=True)
    @ns_api.response(200, "Análise completa retornada", analise_completa_response)
    @ns_api.response(400, "Requisição inválida", erro_response)
    @ns_api.response(401, "Token ausente ou inválido", erro_response)
    @requer_token
    def post(self):
        """Análise completa da reunião (acordos, tasks, entregáveis, quantitativo)"""
        data        = request.get_json(silent=True) or {}
        transcricao = (data.get("transcricao") or "").strip()
        prompt      = (data.get("prompt") or "").strip()

        if not transcricao:
            return {"erro": "Campo 'transcricao' é obrigatório"}, 400

        resultado = analisar_reuniao_com_rag(transcricao, prompt)
        outputs   = resultado.get("outputs_json", {})

        return {
            "analise_principal":    resultado["analise_principal"],
            "acordos_combinados":   outputs.get("acordos_combinados", []),
            "tasks":                outputs.get("tasks", []),
            "entregaveis":          outputs.get("entregaveis", []),
            "proximos_passos":      outputs.get("proximos_passos", {}),
            "analise_quantitativa": outputs.get("analise_quantitativa", {})
        }, 200

@ns_api.route("/resumo")
class Resumo(Resource):
    @ns_api.doc(
        security="BearerAuth",
        description=(
            "Versão leve da análise — retorna apenas o texto da análise e os próximos passos.\n\n"
            "**Ideal para n8n, webhooks e automações** que precisam de resposta rápida "
            "sem os dados quantitativos completos.\n\n"
            "**Configuração no n8n:**\n"
            "1. Node *HTTP Request* → `POST http://<host>:5000/api/resumo`\n"
            "2. Header: `Authorization: Bearer <token>`\n"
            "3. Body (JSON): `{ \"transcricao\": \"...\", \"prompt\": \"...\" }`"
        )
    )
    @ns_api.expect(analise_request, validate=True)
    @ns_api.response(200, "Resumo retornado", resumo_response)
    @ns_api.response(400, "Requisição inválida", erro_response)
    @ns_api.response(401, "Token ausente ou inválido", erro_response)
    @requer_token
    def post(self):
        """Resumo da reunião + próximos passos (ideal para n8n)"""
        data        = request.get_json(silent=True) or {}
        transcricao = (data.get("transcricao") or "").strip()
        prompt      = (data.get("prompt") or "").strip()

        if not transcricao:
            return {"erro": "Campo 'transcricao' é obrigatório"}, 400

        resultado = analisar_reuniao_com_rag(transcricao, prompt)
        proximos  = resultado["outputs_json"].get("proximos_passos", {})

        return {
            "analise": resultado["analise_principal"],
            "proximos_passos": {
                "acoes_imediatas":           proximos.get("acoes_imediatas", []),
                "agenda_sugerida":           proximos.get("agenda_sugerida", []),
                "data_sugerida":             proximos.get("data_sugerida", ""),
                "participantes_necessarios": proximos.get("participantes_necessarios", [])
            }
        }, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
