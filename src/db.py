"""Shared Postgres connection helpers.

Both apply_schema.py and ingest_nasa_power.py use these so there is exactly one
place that knows how to read credentials and open a connection.
"""

import os
import time

import psycopg2
from dotenv import load_dotenv

# The container always listens on 5432 internally; docker-compose.yml publishes it
# on POSTGRES_PORT (5433 by default, to avoid a PostgreSQL already on the host).
DB_HOST = "localhost"
DEFAULT_DB_PORT = 5433

# Connection errors that will never resolve by waiting — retrying these just hides
# the real problem behind a misleading "is the database running?" message.
FATAL_ERROR_MARKERS = (
    "does not exist",
    "password authentication failed",
    "no pg_hba.conf entry",
)


def load_db_config() -> dict:
    """Read Postgres credentials and port from .env into a psycopg2-ready dict."""
    load_dotenv()
    return {
        "host": DB_HOST,
        "port": int(os.environ.get("POSTGRES_PORT", DEFAULT_DB_PORT)),
        "dbname": os.environ["POSTGRES_DB"],
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
    }


def connect_with_retry(config: dict, max_attempts: int = 5, delay_s: int = 2):
    """Connect to Postgres, retrying while the container finishes starting up.

    `docker compose up -d db` returns before Postgres accepts connections, so a
    few short retries smooth over that race. After max_attempts, raise with a
    message that points at the fix rather than a raw connection stack trace.
    """
    port = config["port"]
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            return psycopg2.connect(**config)
        except psycopg2.OperationalError as exc:
            message = str(exc)

            # Credentials/role problems never fix themselves — fail immediately and
            # surface the real reason instead of retrying into a generic timeout.
            if any(marker in message for marker in FATAL_ERROR_MARKERS):
                raise RuntimeError(
                    f"Postgres at {DB_HOST}:{port} rejected the connection: {message.strip()}\n"
                    "The server is reachable, so this is a credentials/database mismatch, "
                    "not a startup delay. Check that .env matches the running container "
                    f"(and that nothing else is listening on port {port})."
                ) from exc

            last_error = exc
            if attempt < max_attempts:
                log(f"Postgres not ready (attempt {attempt}/{max_attempts}), retrying in {delay_s}s...")
                time.sleep(delay_s)

    raise RuntimeError(
        f"Could not connect to Postgres at {DB_HOST}:{port} after {max_attempts} attempts. "
        "Is the database running? Start it with:  docker compose up -d db\n"
        f"Last error: {last_error}"
    )


def log(message: str) -> None:
    """Print a timestamped status line."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")
