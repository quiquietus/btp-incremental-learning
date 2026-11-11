"""Authentication shared by the server and the client nodes.

Passwords: scrypt (memory-hard, Python standard library) with a random salt.
Tokens: JWT (HS256) signed with JWT_SECRET, which the server and the nodes share,
so a token issued by the server at login is accepted by the operator's node.
Node-to-server calls use a node token derived from the same secret.
"""
import base64, hashlib, hmac, os, secrets, time

import jwt
from fastapi import Depends, HTTPException, Request

JWT_SECRET = os.environ.get('JWT_SECRET', 'dev-secret-change-me')
TOKEN_HOURS = float(os.environ.get('TOKEN_HOURS', '12'))
ROLES = ('admin', 'operator', 'viewer')


def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return 'scrypt$' + base64.b64encode(salt).decode() + '$' + base64.b64encode(h).decode()


def verify_password(pw: str, stored: str) -> bool:
    try:
        _, s, h = stored.split('$')
        calc = hashlib.scrypt(pw.encode(), salt=base64.b64decode(s), n=2 ** 14, r=8, p=1, dklen=32)
        return hmac.compare_digest(calc, base64.b64decode(h))
    except Exception:
        return False


def make_token(username: str, role: str, node_id=None) -> str:
    now = int(time.time())
    return jwt.encode({'sub': username, 'role': role, 'node': node_id, 'iat': now, 'exp': now + int(TOKEN_HOURS * 3600)},
                      JWT_SECRET, algorithm='HS256')


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
    except jwt.PyJWTError:
        raise HTTPException(401, 'Invalid or expired token')


def node_token(node_id) -> str:
    return hmac.new(JWT_SECRET.encode(), f'node:{node_id}'.encode(), hashlib.sha256).hexdigest()


def check_node_token(node_id, token: str) -> bool:
    return hmac.compare_digest(node_token(node_id), token or '')


def _bearer(request: Request):
    h = request.headers.get('authorization', '')
    if h.lower().startswith('bearer '):
        return h[7:]
    return request.query_params.get('token')


def current_user(request: Request) -> dict:
    tok = _bearer(request)
    if not tok:
        raise HTTPException(401, 'Not logged in')
    return decode_token(tok)


def require(*roles):
    def dep(user=Depends(current_user)):
        if roles and user.get('role') not in roles:
            raise HTTPException(403, 'Your role cannot do this')
        return user
    return dep
