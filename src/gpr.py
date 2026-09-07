"""Fit a Gaussian Process per location and store predictions with uncertainty.

The point of GPR here is the interval, not just the point forecast — it should say
how confident it is, and PICP checks whether that confidence is honest.

Usage:
    .venv/bin/python3 src/gpr.py
"""

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from data import FEATURE_COLUMNS, TARGET_COLUMN, load_features, save_evaluation, save_predictions
from db import connect_with_retry, load_db_config, log
from metrics import mae, picp, rmse

MODEL_NAME = "gpr"

# 95% interval, so 1.96 standard deviations either side of the mean.
INTERVAL = 0.95
Z = norm.ppf(0.5 + INTERVAL / 2)

RANDOM_SEED = 42
N_RESTARTS = 5


def build_kernel(n_features: int):
    """Matern 5/2 with a per-feature length scale, plus a noise term.

    One length scale per feature lets the model decide what actually matters —
    those values are worth reading afterwards. The WhiteKernel matters just as
    much: without it GPR treats the readings as exact and returns intervals far
    tighter than the data justifies.
    """
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        # Upper bound is deliberately generous: features are standardised, so a
        # length scale in the thousands already means "ignore this one", and a
        # tighter bound just pins the optimiser against it and warns.
        * Matern(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e5), nu=2.5)
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    )


def fit_predict(train_df, test_df):
    """Scale on train only, fit, then predict the test window with std devs."""
    scaler = StandardScaler().fit(train_df[FEATURE_COLUMNS])
    x_train = scaler.transform(train_df[FEATURE_COLUMNS])
    x_test = scaler.transform(test_df[FEATURE_COLUMNS])

    # ravel keeps y 1-D, so predict() returns (n,) rather than (n,1).
    y_train = train_df[TARGET_COLUMN].to_numpy(float).ravel()

    model = GaussianProcessRegressor(
        kernel=build_kernel(len(FEATURE_COLUMNS)),
        normalize_y=True,
        n_restarts_optimizer=N_RESTARTS,
        random_state=RANDOM_SEED,
    )
    model.fit(x_train, y_train)

    mean, std = model.predict(x_test, return_std=True)
    return np.ravel(mean), np.ravel(std), model


def to_prediction_rows(test_df, mean, std, lower, upper):
    """Shape GPR output for save_predictions, uncertainty columns included."""
    if not (len(mean) == len(std) == len(lower) == len(upper) == len(test_df)):
        raise ValueError("prediction arrays and test rows are different lengths")

    return [
        (int(row.location_id), row.obs_date, float(m), float(s), float(lo), float(hi))
        for row, m, s, lo, hi in zip(test_df.itertuples(), mean, std, lower, upper)
    ]


def describe_length_scales(model) -> str:
    """Report which features the fitted kernel leaned on.

    A short length scale means the target changes quickly along that feature, so
    the model is using it; a very long one means it was effectively ignored.
    """
    scales = model.kernel_.k1.k2.length_scale
    pairs = sorted(zip(FEATURE_COLUMNS, np.atleast_1d(scales)), key=lambda p: p[1])
    return ", ".join(f"{name}={value:.2f}" for name, value in pairs)


def main() -> None:
    conn = connect_with_retry(load_db_config())
    try:
        features = load_features(conn)

        for location_id, location_df in features.groupby("location_id", sort=True):
            name = location_df["location_name"].iloc[0]
            train_df = location_df[location_df["split"] == "train"]
            test_df = location_df[location_df["split"] == "test"]

            log(f"{name}: fitting GPR on {len(train_df)} rows...")
            mean, std, model = fit_predict(train_df, test_df)

            lower, upper = mean - Z * std, mean + Z * std
            actual = test_df[TARGET_COLUMN].to_numpy(float)

            save_predictions(
                conn, MODEL_NAME, to_prediction_rows(test_df, mean, std, lower, upper)
            )
            scores = {
                "rmse": rmse(actual, mean),
                "mae": mae(actual, mean),
                "picp": picp(actual, lower, upper),
            }
            save_evaluation(conn, MODEL_NAME, int(location_id), scores)

            log(f"  RMSE {scores['rmse']:.3f}  MAE {scores['mae']:.3f}  PICP {scores['picp']:.1%}")
            log(f"  mean interval width: {np.mean(upper - lower):.2f}")
            log(f"  length scales: {describe_length_scales(model)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
