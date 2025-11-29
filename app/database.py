import os
import time
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError, DisconnectionError, TimeoutError as SQLTimeoutError
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from project root
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# Add connect_timeout to PostgreSQL URLs if not already present
# This prevents hanging when database is unreachable
if DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://"):
    if "connect_timeout" not in DATABASE_URL:
        separator = "&" if "?" in DATABASE_URL else "?"
        DATABASE_URL = f"{DATABASE_URL}{separator}connect_timeout=3"

# Create SQLAlchemy engine with improved connection pool settings and connection timeout
# Using pool_pre_ping=False to avoid blocking on import
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=False,  # Disable pre-ping to avoid blocking on import
    pool_size=5,  # Number of connections to maintain
    max_overflow=10,  # Additional connections beyond pool_size
    pool_recycle=3600,  # Recycle connections after 1 hour
    pool_timeout=3,  # Timeout for getting connection from pool (3 seconds)
    connect_args={
        "connect_timeout": 3,  # Connection timeout in seconds for psycopg2
    },
    echo=False,  # Set to True for SQL query logging
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()


def get_db():
    """
    Dependency function to get database session.
    Use this in FastAPI route dependencies.
    The pool_timeout on the engine will prevent indefinite blocking.
    """
    try:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    except (OperationalError, DisconnectionError, SQLTimeoutError) as e:
        # Let FastAPI handle this - it will be caught by the exception handler
        raise


def check_database_connection(max_retries: int = 2, retry_delay: float = 0.5, timeout: float = 3.0) -> bool:
    """
    Check if database connection is available with timeout.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        timeout: Maximum time to wait for connection in seconds
        
    Returns:
        True if connection is available, False otherwise
    """
    for attempt in range(max_retries):
        try:
            # Use a timeout to prevent hanging
            start_time = time.time()
            with engine.connect() as conn:
                # Set a statement timeout
                conn.execute(text("SET statement_timeout = '3s'"))
                conn.execute(text("SELECT 1"))
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"Database connection check took too long: {elapsed:.2f}s")
                return False
            return True
        except (OperationalError, DisconnectionError, SQLTimeoutError) as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            print(f"Database connection check failed: {str(e)}")
            return False
        except TimeoutError as e:
            print(f"Database connection check timed out: {str(e)}")
            return False
        except Exception as e:
            print(f"Unexpected error checking database connection: {str(e)}")
            return False
    return False

