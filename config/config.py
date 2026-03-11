import os
import json
import google.generativeai as genai

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASTRA_DB_API_ENDPOINT = os.getenv('ASTRA_DB_API_ENDPOINT')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_NAMESPACE = os.getenv('ASTRA_DB_NAMESPACE')
ASTRA_DB_COLLECTION = os.getenv('ASTRA_DB_COLLECTION')
GEMINI_API_KEY = os.getenv("GEM_API_KEY")

JWT_SECRET = os.getenv("JWT_SECRET", "troque-em-producao")
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
_users_env = os.getenv("API_USERS")
API_USERS: dict = json.loads(_users_env) if _users_env else {"admin": "admin123"}


def configure_gemini():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEM_API_KEY não encontrada nas variáveis de ambiente")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.5-flash")
