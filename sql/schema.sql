-- Locations being forecast
CREATE TABLE IF NOT EXISTS locations (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,   -- deviation from DESIGN.md draft: UNIQUE added for idempotent seeding
    latitude    NUMERIC(8,5) NOT NULL,
    longitude   NUMERIC(8,5) NOT NULL,
    notes       TEXT
);

-- Raw daily observations pulled from NASA POWER
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

-- Model predictions — baseline and GPR both write here
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
    -- Model training gets re-run often while tuning. Without this, each re-run
    -- appends a second set of predictions and every downstream RMSE/AVG silently
    -- averages across runs.
    UNIQUE (location_id, obs_date, model_name)
);

-- Evaluation summary — feeds the Power BI comparison view
CREATE TABLE IF NOT EXISTS model_evaluation (
    id            SERIAL PRIMARY KEY,
    model_name    TEXT NOT NULL,
    location_id   INT REFERENCES locations(id),
    rmse          NUMERIC,
    mae           NUMERIC,
    picp          NUMERIC,   -- prediction interval coverage probability, GPR only
    evaluated_at  TIMESTAMP DEFAULT now(),
    -- One current score per model per location; re-evaluating replaces it.
    UNIQUE (model_name, location_id)
);

-- Retrofit the uniqueness constraints above onto a database created before they
-- were added. CREATE TABLE IF NOT EXISTS skips existing tables entirely, so the
-- constraints would otherwise never appear on an already-provisioned database.
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
