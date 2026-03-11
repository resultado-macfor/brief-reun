from functools import wraps
from flask import request
import jwt
from auth.jwt_handler import validar_token


def requer_token(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return {"erro": "Token não fornecido"}, 401

        token = auth_header.split(" ", 1)[1]

        try:
            payload = validar_token(token)
            request.usuario = payload.get("sub")
        except jwt.ExpiredSignatureError:
            return {"erro": "Token expirado"}, 401
        except jwt.InvalidTokenError:
            return {"erro": "Token inválido"}, 401

        return f(*args, **kwargs)

    return wrapper
