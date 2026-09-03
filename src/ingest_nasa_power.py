"""Pull daily solar/weather data from NASA POWER into daily_observations.

One API call per location covers the full date range (confirmed: 730 days come
back in a single response, so no chunking is needed). Safe to re-run — rows are
upserted on (location_id, obs_date).

Usage:
    .venv/bin/python3 src/apply_schema.py     # must run first
    .venv/bin/python3 src/ingest_nasa_power.py
"""

import time

import pandas as pd
import requests
from psycopg2.extras import execute_values

from db import connect_with_retry, load_db_config, log

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# NASA POWER parameter code -> our column name. Codes confirmed against a live
# test call; ghi is returned in kWh/m^2/day.
NASA_PARAM_MAP = {
    "ALLSKY_SFC_SW_DWN": "ghi",
    "T2M": "temperature_c",
    "WS2M": "wind_speed_ms",  # 2m to match T2M's measurement height
    "CLOUD_AMT": "cloud_cover_pct",
}

# NASA POWER uses -999.0 for missing readings (declared in the response header).
FILL_VALUE = -999.0

# Locked project parameters. Names must match sql/seed_locations.sql exactly.
LOCATIONS = [
    {"name": "Phoenix, AZ, USA", "lat": 33.4484, "lon": -112.0740},
    {"name": "Melbourne, AU", "lat": -37.8136, "lon": 144.9631},
    {"name": "Jakarta, ID", "lat": -6.2088, "lon": 106.8456},
]
START_DATE = "2024-06-01"
END_DATE = "2026-05-31"

VALUE_COLUMNS = list(NASA_PARAM_MAP.values())


def fetch_nasa_power_data(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    backoff_base_s: int = 2,
) -> pd.DataFrame:
    """Fetch one location's daily data and return a tidy DataFrame.

    Retries with exponential backoff on network errors or non-2xx responses.
    Missing readings (-999) become None so they land in Postgres as NULL.
    """
    params = {
        "parameters": ",".join(NASA_PARAM_MAP),
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
        "format": "JSON",
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(NASA_POWER_URL, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            break
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                wait_s = backoff_base_s * (2**attempt)
                log(f"  request failed ({exc}); retrying in {wait_s}s...")
                time.sleep(wait_s)
    else:
        raise RuntimeError(
            f"NASA POWER request failed after {max_retries} attempts. Last error: {last_error}"
        )

    parameter_block = payload["properties"]["parameter"]

    df = pd.DataFrame(parameter_block).rename(columns=NASA_PARAM_MAP)
    df.index = pd.to_datetime(df.index, format="%Y%m%d").date
    df.index.name = "obs_date"
    df = df.reset_index()

    # Convert the -999 sentinel to NULL. Gap-filling belongs to Day 2 wrangling,
    # not here — this only marks what is genuinely missing.
    for column in VALUE_COLUMNS:
        df.loc[df[column].sub(FILL_VALUE).abs() < 0.01, column] = None

    return df[["obs_date"] + VALUE_COLUMNS]


def get_location_id_map(conn) -> dict:
    """Return {location name: id} for the seeded locations."""
    with conn.cursor() as cur:
        cur.execute("SELECT name, id FROM locations")
        return dict(cur.fetchall())


def _to_sql_null(value):
    """Convert pandas' NaN to None so it lands in Postgres as NULL.

    Assigning None into a float column leaves NaN, not None, and psycopg2 adapts
    NaN into a literal NUMERIC 'NaN' — which IS NOT NULL, and poisons AVG()/model
    input silently. This is the boundary where that has to be corrected.
    """
    return None if value is None or pd.isna(value) else value


def upsert_observations(df: pd.DataFrame, location_id: int, conn) -> int:
    """Insert or update this location's rows, keyed on (location_id, obs_date)."""
    rows = [
        (
            location_id,
            record["obs_date"],
            _to_sql_null(record["ghi"]),
            _to_sql_null(record["temperature_c"]),
            _to_sql_null(record["wind_speed_ms"]),
            _to_sql_null(record["cloud_cover_pct"]),
        )
        for record in df.to_dict("records")
    ]

    sql = """
        INSERT INTO daily_observations
            (location_id, obs_date, ghi, temperature_c, wind_speed_ms, cloud_cover_pct)
        VALUES %s
        ON CONFLICT (location_id, obs_date) DO UPDATE SET
            ghi = EXCLUDED.ghi,
            temperature_c = EXCLUDED.temperature_c,
            wind_speed_ms = EXCLUDED.wind_speed_ms,
            cloud_cover_pct = EXCLUDED.cloud_cover_pct,
            retrieved_at = now()
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
    return len(rows)


def main() -> None:
    started = time.time()
    conn = connect_with_retry(load_db_config())
    try:
        location_ids = get_location_id_map(conn)

        for location in LOCATIONS:
            name = location["name"]
            if name not in location_ids:
                raise RuntimeError(
                    f"Location '{name}' is not in the locations table. "
                    "Run src/apply_schema.py first."
                )

            log(f"{name}: fetching {START_DATE} to {END_DATE}...")
            df = fetch_nasa_power_data(location["lat"], location["lon"], START_DATE, END_DATE)
            written = upsert_observations(df, location_ids[name], conn)
            missing = int(df[VALUE_COLUMNS].isna().any(axis=1).sum())
            log(f"{name}: {written} rows upserted ({missing} rows with missing values)")
    finally:
        conn.close()

    log(f"Done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
