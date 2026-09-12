"""Authentication and security utilities."""
from datetime import datetime, timedelta
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
    try:
        return ph.verify(hashed_password, plain_password)
    except (VerifyMismatchError, InvalidHashError):
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            return False
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    return ph.hash(password)


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str, expected_type: str = "access") -> dict[str, Any] | None:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # Optional: check if the token type matches if it was provided
        if payload.get("type") and payload.get("type") != expected_type:
            return None
        return payload
    except jwt.PyJWTError:
        return None
