"""Database connection manager."""
from functools import lru_cache
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from config import DEFAULT_CONFIG

@lru_cache(maxsize=1)
def get_db_engine() -> Engine:
    """Return a singleton database engine for connection pooling."""
    if not DEFAULT_CONFIG.postgres_url:
        raise ValueError("POSTGRES_URL is not set.")
        
    # Normalise the URI scheme
    url = DEFAULT_CONFIG.postgres_url
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
        
    connect_args = {
        # Faster failure on network issues (was 10s)
        "connect_timeout": 8,
        "application_name": "campusquery",
        # TCP keepalives to detect stale connections early
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
        # Kill runaway queries after 25s (prevents blocking the pool forever)
        "options": "-c statement_timeout=25000",
    }
    engine = create_engine(
        url,
        connect_args=connect_args,
        pool_pre_ping=True,
        # Recycle connections every 30 min instead of 5 min — avoids thrashing
        # healthy connections on Render's managed PostgreSQL
        pool_recycle=1800,
        # 5 idle connections is sufficient for Render Free Tier (512MB RAM).
        # 10 idle connections waste ~50MB in overhead with no concurrency benefit.
        pool_size=5,
        # Allow short bursts to 15 total connections under load
        max_overflow=10,
        # Fail faster if pool is exhausted — avoids request pileup
        pool_timeout=10,
    )
    return engine


