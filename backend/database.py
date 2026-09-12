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
        
    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=10,
        max_overflow=20,
    )
    return engine

