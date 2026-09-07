"""Loading features and storing model output. Shared by the baselines and GPR."""

import pandas as pd
from psycopg2.extras import execute_values

# What the models actually train on. Weather comes from the previous day; the
# cyclical date terms describe the day being predicted.
FEATURE_COLUMNS = [
    "prev_ghi",
    "prev_temperature_c",
    "prev_wind_speed_ms",
    "prev_cloud_cover_pct",
    "doy_sin",
    "doy_cos",
]
TARGET_COLUMN = "target_ghi"


def load_features(conn) -> pd.DataFrame:
    """Pull daily_features into a DataFrame, numeric columns cast to float.

    Built straight off a cursor rather than pd.read_sql, which wants a SQLAlchemy
    connection and warns otherwise. Postgres NUMERIC also arrives as Decimal,
    which sklearn won't accept.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT location_id, location_name, obs_date, target_ghi, "
            "prev_ghi, prev_temperature_c, prev_wind_speed_ms, prev_cloud_cover_pct, "
            "day_of_year, month, doy_sin, doy_cos, split "
            "FROM daily_features ORDER BY location_id, obs_date"
        )
        columns = [col.name for col in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=columns)

    numeric = FEATURE_COLUMNS + [TARGET_COLUMN]
    df[numeric] = df[numeric].astype(float)
    return df


def to_sql_null(value):
    """Swap NaN for None — psycopg2 writes NaN as NUMERIC 'NaN', which is not NULL
    and slips past both the NOT NULL constraint and any IS NULL audit query."""
    return None if value is None or pd.isna(value) else value


def save_predictions(conn, model_name: str, rows: list) -> int:
    """Store predictions as (location_id, obs_date, predicted_ghi, std, lower, upper).

    std and the bounds are None for the baselines — only GPR fills them in.
    """
    sql = """
        INSERT INTO model_predictions
            (location_id, obs_date, model_name, predicted_ghi,
             predicted_std, lower_bound, upper_bound)
        VALUES %s
        ON CONFLICT (location_id, obs_date, model_name) DO UPDATE SET
            predicted_ghi = EXCLUDED.predicted_ghi,
            predicted_std = EXCLUDED.predicted_std,
            lower_bound   = EXCLUDED.lower_bound,
            upper_bound   = EXCLUDED.upper_bound,
            created_at    = now()
    """
    values = [
        (
            location_id,
            obs_date,
            model_name,
            to_sql_null(predicted),
            to_sql_null(std),
            to_sql_null(lower),
            to_sql_null(upper),
        )
        for location_id, obs_date, predicted, std, lower, upper in rows
    ]
    if any(row[3] is None for row in values):
        raise ValueError(f"{model_name}: refusing to store a NULL/NaN prediction")
    with conn.cursor() as cur:
        execute_values(cur, sql, values)
    conn.commit()
    return len(values)


def save_evaluation(conn, model_name: str, location_id: int, scores: dict) -> None:
    """Store one model's scores for one location, replacing any previous run."""
    sql = """
        INSERT INTO model_evaluation (model_name, location_id, rmse, mae, picp)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (model_name, location_id) DO UPDATE SET
            rmse = EXCLUDED.rmse,
            mae  = EXCLUDED.mae,
            picp = EXCLUDED.picp,
            evaluated_at = now()
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                model_name,
                location_id,
                to_sql_null(scores.get("rmse")),
                to_sql_null(scores.get("mae")),
                to_sql_null(scores.get("picp")),
            ),
        )
    conn.commit()
