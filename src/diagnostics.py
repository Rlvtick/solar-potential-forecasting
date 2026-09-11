"""Reproduce the analysis behind the project's claims, and write it to disk.

Everything here was previously either ad-hoc or a stdout print, which meant the
headline result — that the accuracy differences aren't statistically
distinguishable — had no artefact anyone could re-run. This computes nothing new;
it just makes the reported numbers reproducible.

Read-only: refits the GPR in memory to recover kernel hyperparameters, but never
writes to the database.

Usage:
    .venv/bin/python3 src/diagnostics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from data import FEATURE_COLUMNS, TARGET_COLUMN, load_features
from db import connect_with_retry, load_db_config, log
from gpr import Z, fit_predict

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "diagnostics"

BASELINES = {"persistence": "persistence_ghi", "linear": "linear_ghi"}

# sklearn's normalize_y divides by np.std(y) with ddof=0, so the spread figures
# here use ddof=0 too — otherwise the learned-vs-observed comparison is off by
# sqrt(n/(n-1)) and the ratios don't line up with what the model actually saw.
DDOF = 0


def _kernel_part(kernel, kind):
    """Find a sub-kernel by type rather than by position.

    `kernel_.k1.k2` works but silently depends on how the expression was written;
    reorder the kernel and it points somewhere else. Walking by type doesn't care.
    """
    if isinstance(kernel, kind):
        return kernel
    for attr in ("k1", "k2"):
        part = getattr(kernel, attr, None)
        if part is not None:
            found = _kernel_part(part, kind)
            if found is not None:
                return found
    return None


def load_forecast(conn) -> pd.DataFrame:
    """Pull powerbi_forecast — actuals, all three models, and GPR's interval."""
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM powerbi_forecast ORDER BY location_name, obs_date")
        columns = [col.name for col in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=columns)

    numeric = [c for c in columns if c not in ("location_id", "location_name", "obs_date", "inside_interval")]
    df[numeric] = df[numeric].astype(float)
    return df


def accuracy_tests(forecast: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank, GPR against the better baseline, per location.

    Paired on day — same dates, same actuals — so this asks whether GPR's per-day
    errors are systematically smaller, not whether two RMSE numbers differ.
    """
    rows = []
    for location, site in forecast.groupby("location_name"):
        gpr_err = (site.actual_ghi - site.gpr_ghi).abs()

        # "Better" = lower RMSE, which is the comparison the write-up makes.
        rmses = {
            name: float(np.sqrt(np.mean((site.actual_ghi - site[column]) ** 2)))
            for name, column in BASELINES.items()
        }
        best = min(rmses, key=rmses.get)
        base_err = (site.actual_ghi - site[BASELINES[best]]).abs()

        stat, p_value = wilcoxon(gpr_err, base_err)
        gpr_rmse = float(np.sqrt(np.mean((site.actual_ghi - site.gpr_ghi) ** 2)))

        rows.append(
            {
                "location_name": location,
                "n_days": len(site),
                "best_baseline": best,
                "baseline_rmse": rmses[best],
                "gpr_rmse": gpr_rmse,
                "rmse_margin_pct": 100 * (gpr_rmse - rmses[best]) / rmses[best],
                "wilcoxon_stat": float(stat),
                "p_value": float(p_value),
                "distinguishable_at_05": bool(p_value < 0.05),
            }
        )
    return pd.DataFrame(rows)


def interval_widths(forecast: pd.DataFrame) -> pd.DataFrame:
    """How much the GPR band actually varies day to day.

    The answer is "barely" — the coefficient of variation is under 1% everywhere.
    Worth reporting, because a near-constant band is a real qualification on what
    "calibrated uncertainty" is buying, and it isn't visible from PICP alone.
    """
    width = forecast.gpr_upper - forecast.gpr_lower
    grouped = width.groupby(forecast.location_name)

    return pd.DataFrame(
        {
            "min_width": grouped.min(),
            "max_width": grouped.max(),
            "mean_width": grouped.mean(),
            "cv_pct": 100 * grouped.std(ddof=DDOF) / grouped.mean(),
            "picp": forecast.groupby("location_name").inside_interval.mean(),
            "n_lower_below_zero": forecast[forecast.gpr_lower < 0].groupby("location_name").size(),
        }
    ).fillna({"n_lower_below_zero": 0}).reset_index()


def calibration(features: pd.DataFrame, forecast: pd.DataFrame, fits: dict) -> pd.DataFrame:
    """Learned noise against the spread actually observed on test.

    One ratio explains all three PICP results. The model fixes its noise level at
    training time; where the test window turned out calmer the band is too wide,
    where it turned out choppier it's too narrow.
    """
    rows = []
    for location, site in forecast.groupby("location_name"):
        model = fits[location]
        site_features = features[features.location_name == location]
        train = site_features[site_features.split == "train"]

        # noise_level is a VARIANCE, and normalize_y standardised the target, so
        # getting back to GHI units is sqrt() first, then rescale. Dropping the
        # sqrt here is the easy mistake and inflates the number ~2x.
        noise_level = _kernel_part(model.kernel_, WhiteKernel).noise_level
        y_train_std = float(np.std(train[TARGET_COLUMN].to_numpy(float)))
        learned_noise_sd = float(np.sqrt(noise_level) * y_train_std)

        residual_sd = float(np.std(site.actual_ghi - site.gpr_ghi, ddof=DDOF))

        # Mean day-to-day change, train vs test. prev_ghi is always the day before,
        # so this difference is the actual overnight swing, not a row-order artefact.
        # Each split drops its own first row: on the test side that row's previous
        # day is 2026-02-28, a training day, and letting it in would put a training
        # observation inside a test-period statistic. Test is 91 swings, not 92.
        swing = (site_features[TARGET_COLUMN] - site_features.prev_ghi).abs()
        by_split = swing.groupby(site_features.split).apply(lambda s: s.iloc[1:].mean())

        rows.append(
            {
                "location_name": location,
                "noise_level": float(noise_level),
                "y_train_std": y_train_std,
                "learned_noise_sd": learned_noise_sd,
                "test_residual_sd": residual_sd,
                "ratio_learned_over_actual": learned_noise_sd / residual_sd,
                "train_mean_swing": float(by_split.get("train", np.nan)),
                "test_mean_swing": float(by_split.get("test", np.nan)),
                "picp": float(site.inside_interval.mean()),
            }
        )
    return pd.DataFrame(rows)


def length_scales(fits: dict) -> pd.DataFrame:
    """Fitted ARD length scales, one row per location per feature.

    Short means the feature is being used; very long means it was ignored. Note
    the flag: past ~1e4 the likelihood is flat, so the optimiser barely moves off
    its starting draw and the number is an artefact. Read those as "ignored", and
    don't quote the value.
    """
    rows = []
    for location, model in fits.items():
        scales = np.atleast_1d(_kernel_part(model.kernel_, Matern).length_scale)
        for feature, scale in zip(FEATURE_COLUMNS, scales):
            rows.append(
                {
                    "location_name": location,
                    "feature": feature,
                    "length_scale": float(scale),
                    "effectively_ignored": bool(scale > 1e4),
                }
            )
    return pd.DataFrame(rows).sort_values(["location_name", "length_scale"])


def refit_all(features: pd.DataFrame) -> dict:
    """Refit per location to recover kernel hyperparameters.

    Deterministic (random_state=42), so this reproduces the committed run rather
    than being a fresh one — but it deliberately reuses gpr.fit_predict, so if
    that changes these diagnostics follow it instead of drifting.
    """
    fits = {}
    for _, site in features.groupby("location_id", sort=True):
        name = site.location_name.iloc[0]
        log(f"{name}: refitting to read hyperparameters...")
        _, _, model = fit_predict(site[site.split == "train"], site[site.split == "test"])
        fits[name] = model
    return fits


def write_summary(tables: dict, path: Path) -> None:
    """One readable file, so the numbers can be checked without running anything."""
    lines = [
        "# Diagnostics",
        "",
        f"Generated by `src/diagnostics.py`. Nominal interval {Z:.4f} SD either side "
        "of the mean (95% two-sided).",
        "",
    ]
    for title, table in tables.items():
        # Fenced to_string rather than to_markdown — the latter wants tabulate, and
        # a dependency for table borders isn't worth it.
        lines += [f"## {title}", "", "```", table.round(4).to_string(index=False), "```", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect_with_retry(load_db_config())
    try:
        features = load_features(conn)
        forecast = load_forecast(conn)
    finally:
        conn.close()

    if len(forecast) != 276:
        raise ValueError(f"expected 276 test rows in powerbi_forecast, got {len(forecast)}")

    fits = refit_all(features)

    tables = {
        "Accuracy — GPR vs best baseline (Wilcoxon signed-rank, paired by day)": accuracy_tests(forecast),
        "Calibration — learned noise vs observed test spread": calibration(features, forecast, fits),
        "Interval width": interval_widths(forecast),
        "Fitted ARD length scales": length_scales(fits),
    }

    for title, table in tables.items():
        name = title.split(" —")[0].split(" (")[0].lower().replace(" ", "_")
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        log(f"{name}.csv: {len(table)} rows")

    write_summary(tables, OUTPUT_DIR / "summary.md")
    log(f"Written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
