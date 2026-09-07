"""Postgres connection helpers shared by the other scripts."""

import os
import time
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

# Resolved from this file rather than the working directory, so the scripts work
# no matter where they're run from.
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# The container always listens on 5432 internally; compose publishes it on 5433
# so it doesn't collide with a Postgres already installed on the host.
DB_HOST = "localhost"
DEFAULT_DB_PORT = 5433

# Worth failing on immediately — waiting won't fix a wrong password or a missing role.
FATAL_ERROR_MARKERS = (
    "does not exist",
    "password authentication failed",
    "no pg_hba.conf entry",
)


def load_db_config() -> dict:
    """Read credentials from .env into a psycopg2-ready dict."""
    if not ENV_PATH.exists():
        raise RuntimeError(
            f"No .env file found at {ENV_PATH}. Copy .env.example to .env and fill "
            "in the database credentials."
        )

    # override=True so a variable left exported in the shell can't quietly win over .env.
    load_dotenv(ENV_PATH, override=True)

    missing = [
        key
        for key in ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD")
        if not os.environ.get(key)
    ]
    if missing:
        raise RuntimeError(f"{ENV_PATH} is missing required key(s): {', '.join(missing)}")

    return {
        "host": os.environ.get("POSTGRES_HOST", DB_HOST),
        "port": int(os.environ.get("POSTGRES_PORT", DEFAULT_DB_PORT)),
        "dbname": os.environ["POSTGRES_DB"],
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
    }


def connect_with_retry(config: dict, max_attempts: int = 5, delay_s: int = 2):
    """Connect to Postgres, retrying while the container finishes starting up.

    `docker compose up -d db` returns before Postgres is actually accepting
    connections, so the first attempt or two can legitimately fail.
    """
    port = config["port"]
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            return psycopg2.connect(**config)
        except psycopg2.OperationalError as exc:
            message = str(exc)

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
