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
        "connect_timeout": 10,
        "application_name": "campusquery",
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    }
    engine = create_engine(
        url,
        connect_args=connect_args,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=10,
        max_overflow=20,
        pool_timeout=15,
    )
    return engine


