-- Model-ready features for the one-day-ahead forecast. Weather features are all
-- taken from the previous day; the calendar ones use the target day, since a date
-- is known in advance either way.

CREATE OR REPLACE VIEW daily_features AS
WITH lagged AS (
    SELECT
        o.location_id,
        l.name AS location_name,
        o.obs_date,
        o.ghi AS target_ghi,
        -- prev_obs_date is exposed so the gap-free assumption can actually be checked:
        -- LAG() returns the previous row, which is only the previous day while the
        -- series has no holes in it.
        LAG(o.obs_date)        OVER w AS prev_obs_date,
        LAG(o.ghi)             OVER w AS prev_ghi,
        LAG(o.temperature_c)   OVER w AS prev_temperature_c,
        LAG(o.wind_speed_ms)   OVER w AS prev_wind_speed_ms,
        LAG(o.cloud_cover_pct) OVER w AS prev_cloud_cover_pct
    FROM daily_observations o
    JOIN locations l ON l.id = o.location_id
    WINDOW w AS (PARTITION BY o.location_id ORDER BY o.obs_date)
)
SELECT
    location_id,
    location_name,
    obs_date,
    target_ghi,
    prev_obs_date,
    prev_ghi,
    prev_temperature_c,
    prev_wind_speed_ms,
    prev_cloud_cover_pct,

    EXTRACT(DOY   FROM obs_date)::int AS day_of_year,
    EXTRACT(MONTH FROM obs_date)::int AS month,

    -- Encoded cyclically so 31 Dec sits next to 1 Jan instead of at the opposite
    -- end of the scale. 365.25 to absorb leap years.
    sin(2 * pi() * EXTRACT(DOY FROM obs_date) / 365.25) AS doy_sin,
    cos(2 * pi() * EXTRACT(DOY FROM obs_date) / 365.25) AS doy_cos,

    -- Defining the split here rather than in Python means the model and Power BI
    -- can't end up disagreeing on where the test period starts.
    CASE WHEN obs_date >= DATE '2026-03-01' THEN 'test' ELSE 'train' END AS split
FROM lagged
-- Drops each location's first day, which has no previous day to draw features from.
WHERE prev_ghi IS NOT NULL;
