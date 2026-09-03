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
    created_at      TIMESTAMP DEFAULT now()
);

-- Evaluation summary — feeds the Power BI comparison view
CREATE TABLE IF NOT EXISTS model_evaluation (
    id            SERIAL PRIMARY KEY,
    model_name    TEXT NOT NULL,
    location_id   INT REFERENCES locations(id),
    rmse          NUMERIC,
    mae           NUMERIC,
    picp          NUMERIC,   -- prediction interval coverage probability, GPR only
    evaluated_at  TIMESTAMP DEFAULT now()
);
