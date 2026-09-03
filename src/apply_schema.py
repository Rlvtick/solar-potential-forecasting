"""Apply the database schema and seed the three forecast locations.

Safe to re-run: the schema uses CREATE TABLE IF NOT EXISTS and the seed uses
ON CONFLICT (name) DO NOTHING.

Usage:
    docker compose up -d db
    .venv/bin/python3 src/apply_schema.py
"""

from pathlib import Path

from db import connect_with_retry, load_db_config, log

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"
SCHEMA_FILE = SQL_DIR / "schema.sql"
SEED_FILE = SQL_DIR / "seed_locations.sql"


def run_sql_file(conn, path: Path) -> None:
    """Execute every statement in a .sql file as one transaction."""
    sql = path.read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    log(f"Applied {path.name}")


def main() -> None:
    conn = connect_with_retry(load_db_config())
    try:
        run_sql_file(conn, SCHEMA_FILE)
        run_sql_file(conn, SEED_FILE)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name"
            )
            tables = [row[0] for row in cur.fetchall()]
            cur.execute("SELECT count(*) FROM locations")
            location_count = cur.fetchone()[0]

        log(f"Tables present ({len(tables)}): {', '.join(tables)}")
        log(f"locations seeded: {location_count} rows")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
