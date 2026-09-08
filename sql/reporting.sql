-- Views Power BI reads from, so the dashboard never has to join anything itself.

-- One row per location per test day, models side by side. Wide rather than long
-- because the confidence band needs lower and upper on the same row to plot.
CREATE OR REPLACE VIEW powerbi_forecast AS
SELECT
    f.location_id,
    f.location_name,
    f.obs_date,
    f.target_ghi                     AS actual_ghi,
    gpr.predicted_ghi                AS gpr_ghi,
    gpr.lower_bound                  AS gpr_lower,
    gpr.upper_bound                  AS gpr_upper,
    persistence.predicted_ghi        AS persistence_ghi,
    linear.predicted_ghi             AS linear_ghi,
    -- Signed errors, so the dashboard can show bias without recomputing anything.
    gpr.predicted_ghi - f.target_ghi         AS gpr_error,
    persistence.predicted_ghi - f.target_ghi AS persistence_error,
    linear.predicted_ghi - f.target_ghi      AS linear_error,
    -- Whether the actual landed inside GPR's band, for a coverage visual.
    (f.target_ghi BETWEEN gpr.lower_bound AND gpr.upper_bound) AS inside_interval
FROM daily_features f
LEFT JOIN model_predictions gpr
       ON gpr.location_id = f.location_id AND gpr.obs_date = f.obs_date
      AND gpr.model_name = 'gpr'
LEFT JOIN model_predictions persistence
       ON persistence.location_id = f.location_id AND persistence.obs_date = f.obs_date
      AND persistence.model_name = 'baseline_persistence'
LEFT JOIN model_predictions linear
       ON linear.location_id = f.location_id AND linear.obs_date = f.obs_date
      AND linear.model_name = 'baseline_linear'
WHERE f.split = 'test';

-- Scores per model per location, with readable names for the comparison table.
CREATE OR REPLACE VIEW powerbi_scores AS
SELECT
    l.name AS location_name,
    e.model_name,
    CASE e.model_name
        WHEN 'gpr'                  THEN 'Gaussian Process'
        WHEN 'baseline_linear'      THEN 'Linear Regression'
        WHEN 'baseline_persistence' THEN 'Persistence'
        ELSE e.model_name
    END AS model_label,
    e.rmse,
    e.mae,
    e.picp,
    -- RMSE as a share of mean GHI. Raw RMSE can't be compared between sites,
    -- since a site with flatter irradiance scores lower without forecasting better.
    e.rmse / NULLIF(site.mean_ghi, 0) AS nrmse
FROM model_evaluation e
JOIN locations l ON l.id = e.location_id
JOIN (
    SELECT location_id, avg(target_ghi) AS mean_ghi
    FROM daily_features WHERE split = 'test'
    GROUP BY location_id
) site ON site.location_id = e.location_id;
