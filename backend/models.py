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
        ddl_users = """
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
        );
        """
        ddl_metrics = """
        CREATE TABLE IF NOT EXISTS api_metrics (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(50),
            user_email VARCHAR(255),
            endpoint VARCHAR(255) NOT NULL,
            method VARCHAR(10) NOT NULL,
            status_code INTEGER NOT NULL,
            duration_ms DOUBLE PRECISION NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_api_metrics_created_at ON api_metrics(created_at);
        CREATE INDEX IF NOT EXISTS idx_api_metrics_endpoint ON api_metrics(endpoint);
        CREATE INDEX IF NOT EXISTS idx_api_metrics_user_email ON api_metrics(user_email);
        """
        with self.engine.begin() as conn:
            conn.execute(text(ddl_users))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS name VARCHAR(255)"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS nationality VARCHAR(255)"))
            conn.execute(text(ddl_metrics))

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

    def log_api_metric(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
    ) -> None:
        """Record an API request latency measurement."""
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO api_metrics (user_id, user_email, endpoint, method, status_code, duration_ms)
                        VALUES (:user_id, :user_email, :endpoint, :method, :status_code, :duration_ms)
                        """
                    ),
                    {
                        "user_id": str(user_id) if user_id else None,
                        "user_email": user_email,
                        "endpoint": endpoint,
                        "method": method,
                        "status_code": status_code,
                        "duration_ms": round(duration_ms, 2),
                    },
                )
        except Exception as exc:
            # Metrics logging should never break the request pipeline
            print(f"[metrics] Failed to record API metric: {exc}")

    def get_latency_metrics(self, hours: int = 24) -> dict[str, Any]:
        """Compute P50, P95, P99 percentiles, endpoint breakdown, and user stats."""
        with self.engine.connect() as conn:
            # Global summary metrics including P50/P95/P99, traffic load, error rate & active users
            global_res = conn.execute(
                text(
                    """
                    SELECT 
                        COUNT(*) as total_requests,
                        COALESCE(ROUND(COUNT(*)::numeric / GREATEST((:hours * 60.0), 1.0), 2), 0) as requests_per_minute,
                        COALESCE(ROUND(100.0 * COUNT(*) FILTER (WHERE status_code < 400)::numeric / GREATEST(COUNT(*), 1), 2), 100.0) as success_rate_percent,
                        COALESCE(ROUND(100.0 * COUNT(*) FILTER (WHERE status_code >= 400)::numeric / GREATEST(COUNT(*), 1), 2), 0.0) as error_rate_percent,
                        COUNT(DISTINCT user_email) FILTER (WHERE user_email IS NOT NULL AND user_email != 'anonymous') as active_users,
                        COALESCE(ROUND(AVG(duration_ms)::numeric, 2), 0) as avg_latency_ms,
                        COALESCE(ROUND(MIN(duration_ms)::numeric, 2), 0) as min_latency_ms,
                        COALESCE(ROUND(MAX(duration_ms)::numeric, 2), 0) as max_latency_ms,
                        COALESCE(ROUND(percentile_cont(0.50) WITHIN GROUP (ORDER BY duration_ms)::numeric, 2), 0) as p50_latency_ms,
                        COALESCE(ROUND(percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms)::numeric, 2), 0) as p95_latency_ms,
                        COALESCE(ROUND(percentile_cont(0.99) WITHIN GROUP (ORDER BY duration_ms)::numeric, 2), 0) as p99_latency_ms
                    FROM api_metrics
                    WHERE created_at >= NOW() - (:hours || ' hour')::interval
                    """
                ),
                {"hours": hours},
            ).mappings().first()

            # Breakdown by endpoint
            endpoints_res = conn.execute(
                text(
                    """
                    SELECT 
                        endpoint,
                        method,
                        COUNT(*) as request_count,
                        COALESCE(ROUND(AVG(duration_ms)::numeric, 2), 0) as avg_ms,
                        COALESCE(ROUND(percentile_cont(0.50) WITHIN GROUP (ORDER BY duration_ms)::numeric, 2), 0) as p50_ms,
                        COALESCE(ROUND(percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms)::numeric, 2), 0) as p95_ms,
                        COALESCE(ROUND(percentile_cont(0.99) WITHIN GROUP (ORDER BY duration_ms)::numeric, 2), 0) as p99_ms
                    FROM api_metrics
                    WHERE created_at >= NOW() - (:hours || ' hour')::interval
                    GROUP BY endpoint, method
                    ORDER BY request_count DESC
                    """
                ),
                {"hours": hours},
            ).mappings().all()

            # Breakdown by user (slowest users or highest consumption)
            users_res = conn.execute(
                text(
                    """
                    SELECT 
                        COALESCE(user_email, 'anonymous') as user_email,
                        COUNT(*) as request_count,
                        COALESCE(ROUND(AVG(duration_ms)::numeric, 2), 0) as avg_ms,
                        COALESCE(ROUND(percentile_cont(0.50) WITHIN GROUP (ORDER BY duration_ms)::numeric, 2), 0) as p50_ms,
                        COALESCE(ROUND(percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms)::numeric, 2), 0) as p95_ms,
                        COALESCE(ROUND(MAX(duration_ms)::numeric, 2), 0) as max_ms
                    FROM api_metrics
                    WHERE created_at >= NOW() - (:hours || ' hour')::interval
                    GROUP BY user_email
                    ORDER BY avg_ms DESC
                    LIMIT 20
                    """
                ),
                {"hours": hours},
            ).mappings().all()

            # Slowest recent requests
            slowest_res = conn.execute(
                text(
                    """
                    SELECT 
                        COALESCE(user_email, 'anonymous') as user_email,
                        endpoint,
                        method,
                        status_code,
                        duration_ms,
                        created_at
                    FROM api_metrics
                    WHERE created_at >= NOW() - (:hours || ' hour')::interval
                    ORDER BY duration_ms DESC
                    LIMIT 15
                    """
                ),
                {"hours": hours},
            ).mappings().all()

            summary_dict = dict(global_res) if global_res else {}
            kpis = {
                "total_requests": summary_dict.get("total_requests", 0),
                "requests_per_min": summary_dict.get("requests_per_minute", 0.0),
                "success_rate": f"{summary_dict.get('success_rate_percent', 100.0)}%",
                "error_rate": f"{summary_dict.get('error_rate_percent', 0.0)}%",
                "p50_latency_ms": summary_dict.get("p50_latency_ms", 0.0),
                "p95_latency_ms": summary_dict.get("p95_latency_ms", 0.0),
                "p99_latency_ms": summary_dict.get("p99_latency_ms", 0.0),
                "active_users": summary_dict.get("active_users", 0),
            }

            return {
                "time_window_hours": hours,
                "kpi_metrics": kpis,
                "global_summary": summary_dict,
                "by_endpoint": [dict(r) for r in endpoints_res],
                "by_user": [dict(r) for r in users_res],
                "slowest_recent_requests": [
                    {**dict(r), "created_at": r["created_at"].isoformat() if r.get("created_at") else None}
                    for r in slowest_res
                ],
            }

