"""Pull daily solar/weather data from NASA POWER into daily_observations.

One API call per location covers the full date range (confirmed: 730 days come
back in a single response, so no chunking is needed). Safe to re-run — rows are
upserted on (location_id, obs_date).

Usage:
    .venv/bin/python3 src/apply_schema.py     # must run first
    .venv/bin/python3 src/ingest_nasa_power.py
"""

import json
import re
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
from psycopg2.extras import execute_values

from db import connect_with_retry, load_db_config, log

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Raw API responses are cached here before anything is written to the database, so
# a degraded NASA response can never leave us with neither good rows nor a copy of
# what was actually returned.
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

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


def _to_api_date(value: str) -> str:
    """Convert YYYY-MM-DD to the YYYYMMDD the API expects, rejecting bad input."""
    return datetime.strptime(value, "%Y-%m-%d").strftime("%Y%m%d")


def expected_day_count(start_date: str, end_date: str) -> int:
    """Inclusive day count for the range, so a short response can be detected."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    return (end - start).days + 1


def _cache_raw_response(payload: dict, name: str) -> None:
    """Write the raw API response to data/raw/ before anything touches the DB."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    path = RAW_DIR / f"{slug}.json"
    path.write_text(json.dumps(payload, indent=2))


def fetch_nasa_power_data(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    backoff_base_s: int = 2,
) -> tuple[pd.DataFrame, dict]:
    """Fetch one location's daily data; return (tidy DataFrame, raw payload).

    Retries with exponential backoff on network errors or non-2xx responses.
    Missing readings (-999) become None so they land in Postgres as NULL.
    """
    params = {
        "parameters": ",".join(NASA_PARAM_MAP),
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": _to_api_date(start_date),
        "end": _to_api_date(end_date),
        "format": "JSON",
        # Pinned rather than left to the API default: LST means obs_date is a local
        # solar day at each site. A UTC day would straddle two local daylight
        # periods (Phoenix is UTC-7) and quietly corrupt the target variable.
        "time-standard": "LST",
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

    # NASA reports parameter-level problems here even on an HTTP 200.
    if payload.get("messages"):
        log(f"  NASA POWER returned messages: {payload['messages']}")

    parameter_block = payload["properties"]["parameter"]

    missing_params = [code for code in NASA_PARAM_MAP if code not in parameter_block]
    if missing_params:
        raise RuntimeError(
            f"NASA POWER response is missing requested parameter(s): {missing_params}. "
            f"Returned: {sorted(parameter_block)}"
        )

    df = pd.DataFrame(parameter_block).rename(columns=NASA_PARAM_MAP)
    df.index = pd.to_datetime(df.index, format="%Y%m%d").date
    df.index.name = "obs_date"
    df = df.reset_index()

    expected = expected_day_count(start_date, end_date)
    if len(df) != expected:
        raise RuntimeError(
            f"NASA POWER returned {len(df)} days for {start_date}..{end_date}, "
            f"expected {expected}. Refusing to write a partial pull over existing rows."
        )

    # Convert the -999 sentinel to NULL. Gap-filling belongs to Day 2 wrangling,
    # not here — this only marks what is genuinely missing.
    for column in VALUE_COLUMNS:
        df.loc[df[column].sub(FILL_VALUE).abs() < 0.01, column] = None

    return df[["obs_date"] + VALUE_COLUMNS], payload


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
            df, payload = fetch_nasa_power_data(
                location["lat"], location["lon"], START_DATE, END_DATE
            )

            # Cache before writing: the upsert overwrites existing values, so this
            # is the only copy of what NASA actually returned this run.
            _cache_raw_response(payload, name)

            written = upsert_observations(df, location_ids[name], conn)
            missing = int(df[VALUE_COLUMNS].isna().any(axis=1).sum())
            log(f"{name}: {written} rows upserted ({missing} rows with missing values)")
    finally:
        conn.close()

    log(f"Done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
