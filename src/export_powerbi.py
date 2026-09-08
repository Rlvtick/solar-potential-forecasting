"""Dump the reporting views to CSV for Power BI.

The Windows VM can read Postgres directly, but that depends on VM networking and
the Npgsql driver behaving. These files are the fallback if it doesn't.

Usage:
    .venv/bin/python3 src/export_powerbi.py
"""

from decimal import Decimal
from pathlib import Path

import pandas as pd

from db import connect_with_retry, load_db_config, log

EXPORT_DIR = Path(__file__).resolve().parent.parent / "data" / "powerbi"

VIEWS = {
    "forecast": "SELECT * FROM powerbi_forecast ORDER BY location_name, obs_date",
    "scores": "SELECT * FROM powerbi_scores ORDER BY location_name, rmse",
}


def export_view(conn, name: str, query: str) -> int:
    """Write one view out as CSV and return the row count."""
    with conn.cursor() as cur:
        cur.execute(query)
        columns = [col.name for col in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=columns)

    # Postgres NUMERIC arrives as Decimal and serialises to 20-odd decimal places.
    # GHI is in kWh/m^2/day, so 4 is already past anything meaningful. Only real
    # numbers get rounded — bool is a subclass of int in Python, so an unguarded
    # numeric check turns True into 1.0 and ids into floats.
    def is_measurement(value):
        return isinstance(value, (float, Decimal)) and not isinstance(value, bool)

    for column in df.columns:
        # dropna first: picp is NULL for the baselines, and a column with nulls
        # would otherwise never qualify.
        present = df[column].dropna()
        if len(present) and present.map(is_measurement).all():
            df[column] = df[column].astype(float).round(4)

    path = EXPORT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    log(f"{path.name}: {len(df)} rows, {len(columns)} columns")
    return len(df)


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect_with_retry(load_db_config())
    try:
        for name, query in VIEWS.items():
            export_view(conn, name, query)
    finally:
        conn.close()

    log(f"Written to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
