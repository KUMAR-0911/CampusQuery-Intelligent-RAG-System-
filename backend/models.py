"""User management and database models."""
import enum
from typing import Any, Optional
from datetime import datetime, timezone
from sqlalchemy import text
from sqlalchemy.engine import Engine


class UserRole(str, enum.Enum):
    USER = "USER"
    ADMIN = "ADMIN"


class UserStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    LOCKED = "LOCKED"
    PENDING_VERIFICATION = "PENDING_VERIFICATION"


class UserManager:
    def __init__(self, engine: Engine):
        self.engine = engine
        self._create_tables()

    def _create_tables(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255),
            nationality VARCHAR(255),
            email VARCHAR(255) UNIQUE NOT NULL,
            hashed_password VARCHAR(255) NOT NULL,
            role VARCHAR(50) NOT NULL DEFAULT 'USER',
            status VARCHAR(50) NOT NULL DEFAULT 'PENDING_VERIFICATION',
            otp_code VARCHAR(10),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """
        with self.engine.begin() as conn:
            conn.execute(text(ddl))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS name VARCHAR(255)"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS nationality VARCHAR(255)"))

    def create_user(
        self, email: str, hashed_password: str, name: Optional[str] = None, nationality: Optional[str] = None, role: str = UserRole.USER.value, otp_code: Optional[str] = None
    ) -> dict[str, Any]:
        with self.engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    INSERT INTO users (email, hashed_password, name, nationality, role, status, otp_code)
                    VALUES (:email, :hashed_password, :name, :nationality, :role, :status, :otp_code)
                    RETURNING id, email, name, nationality, role, status
                    """
                ),
                {
                    "email": email,
                    "hashed_password": hashed_password,
                    "name": name,
                    "nationality": nationality,
                    "role": role,
                    "status": UserStatus.PENDING_VERIFICATION.value,
                    "otp_code": otp_code,
                },
            )
            return dict(result.mappings().first())



    def get_user_by_email(self, email: str) -> Optional[dict[str, Any]]:
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM users WHERE email = :email"),
                {"email": email},
            )
            row = result.mappings().first()
            return dict(row) if row else None

    def get_user_by_id(self, user_id: int) -> Optional[dict[str, Any]]:
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM users WHERE id = :id"),
                {"id": user_id},
            )
            row = result.mappings().first()
            return dict(row) if row else None

    def update_user_name(self, email: str, name: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE users SET name = :name WHERE email = :email"),
                {"name": name, "email": email},
            )

    def update_user_profile(self, email: str, name: str, nationality: Optional[str] = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE users SET name = :name, nationality = :nationality WHERE email = :email"),
                {"name": name, "nationality": nationality, "email": email},
            )


    def update_user_status(self, email: str, status: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE users SET status = :status WHERE email = :email"),
                {"status": status, "email": email},
            )

    def update_otp(self, email: str, otp_code: Optional[str]) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE users SET otp_code = :otp_code WHERE email = :email"),
                {"otp_code": otp_code, "email": email},
            )

    def update_password(self, email: str, hashed_password: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE users SET hashed_password = :hashed_password WHERE email = :email"),
                {"hashed_password": hashed_password, "email": email},
            )

    def get_all_users(self) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT id, name, nationality, email, role, status, created_at FROM users ORDER BY id ASC")


            )
            return [dict(row) for row in result.mappings().all()]
