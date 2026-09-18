from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from passlib.context import CryptContext

from config import DEFAULT_CONFIG

# Configuration — read from centralized config
SECRET_KEY = DEFAULT_CONFIG.jwt_secret_key
ALGORITHM = DEFAULT_CONFIG.jwt_algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = DEFAULT_CONFIG.jwt_expire_minutes
REFRESH_TOKEN_EXPIRE_DAYS = 7

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError

# Explicit Argon2id Password Hasher (default type is argon2id)
ph = PasswordHasher()
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not plain_password or not hashed_password:
        return False
    try:
        if hashed_password.startswith("$argon2"):
            try:
                return ph.verify(hashed_password, plain_password)
            except VerifyMismatchError:
                return False
            except InvalidHashError:
                pass
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    return ph.hash(password)


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode = {**data, "exp": expire, "type": "access"}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta if expires_delta else timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
    to_encode = {**data, "exp": expire, "type": "refresh"}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str, expected_type: str = "access") -> dict[str, Any] | None:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], leeway=60)
        # Optional: check if the token type matches if it was provided
        if payload.get("type") and payload.get("type") != expected_type:
            return None
        return payload
    except jwt.PyJWTError:
        return None

