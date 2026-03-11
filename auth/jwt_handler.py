import datetime
import jwt
from config.config import JWT_SECRET, JWT_EXPIRATION_HOURS


def gerar_token(usuario: str) -> str:
    payload = {
        "sub": usuario,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def validar_token(token: str) -> dict:
    """
    Decodifica e valida o token JWT.
    Lança jwt.ExpiredSignatureError ou jwt.InvalidTokenError em caso de falha.
    """
    return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
