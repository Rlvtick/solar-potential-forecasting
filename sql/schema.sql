-- The three sites we forecast for.
CREATE TABLE IF NOT EXISTS locations (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,   -- UNIQUE gives the seed something to conflict on
    latitude    NUMERIC(8,5) NOT NULL,
    longitude   NUMERIC(8,5) NOT NULL,
    notes       TEXT
);

-- Raw daily readings from NASA POWER, untouched apart from -999 becoming NULL.
CREATE TABLE IF NOT EXISTS daily_observations (
    id                SERIAL PRIMARY KEY,
    location_id       INT REFERENCES locations(id),
    obs_date          DATE NOT NULL,
    ghi               NUMERIC,      -- global horizontal irradiance (ALLSKY_SFC_SW_DWN), kWh/m^2/day
    temperature_c     NUMERIC,
    wind_speed_ms     NUMERIC,
    cloud_cover_pct   NUMERIC,
    source            TEXT DEFAULT 'NASA_POWER',
    retrieved_at      TIMESTAMP DEFAULT now(),
    UNIQUE (location_id, obs_date)
);

-- Both the baselines and GPR write their predictions here.
CREATE TABLE IF NOT EXISTS model_predictions (
    id              SERIAL PRIMARY KEY,
    location_id     INT REFERENCES locations(id),
    obs_date        DATE NOT NULL,
    model_name      TEXT NOT NULL,     -- 'baseline_persistence' | 'baseline_linear' | 'gpr'
    predicted_ghi   NUMERIC NOT NULL,
    predicted_std   NUMERIC,           -- null for baselines, populated for GPR
    lower_bound     NUMERIC,
    upper_bound     NUMERIC,
    created_at      TIMESTAMP DEFAULT now(),
    -- Keeps a re-run from stacking a second set of predictions on top of the first,
    -- which would quietly skew every metric computed off this table.
    UNIQUE (location_id, obs_date, model_name)
);

-- Summary scores per model and location — this is what Power BI reads.
CREATE TABLE IF NOT EXISTS model_evaluation (
    id            SERIAL PRIMARY KEY,
    model_name    TEXT NOT NULL,
    location_id   INT REFERENCES locations(id),
    rmse          NUMERIC,
    mae           NUMERIC,
    picp          NUMERIC,   -- prediction interval coverage probability, GPR only
    evaluated_at  TIMESTAMP DEFAULT now(),
    -- One score per model per location, so re-evaluating overwrites rather than appends.
    UNIQUE (model_name, location_id)
);

-- CREATE TABLE IF NOT EXISTS won't touch a table that already exists, so the two
-- constraints above need adding separately on any database built before them.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'model_predictions_location_id_obs_date_model_name_key'
    ) THEN
        ALTER TABLE model_predictions
            ADD CONSTRAINT model_predictions_location_id_obs_date_model_name_key
            UNIQUE (location_id, obs_date, model_name);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'model_evaluation_model_name_location_id_key'
    ) THEN
        ALTER TABLE model_evaluation
            ADD CONSTRAINT model_evaluation_model_name_location_id_key
            UNIQUE (model_name, location_id);
    END IF;
END $$;
