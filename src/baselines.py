"""Run the two baselines and score them, so GPR has something to beat.

Persistence needs no fitting — it just carries yesterday's GHI forward, which is
already sitting in the feature view as prev_ghi. Linear regression is fit per
location on the training split only.

Usage:
    .venv/bin/python3 src/baselines.py
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from data import FEATURE_COLUMNS, TARGET_COLUMN, load_features, save_evaluation, save_predictions
from db import connect_with_retry, load_db_config, log
from metrics import mae, rmse

PERSISTENCE = "baseline_persistence"
LINEAR = "baseline_linear"


def predict_persistence(test_df):
    """Yesterday's GHI, carried forward unchanged."""
    return test_df["prev_ghi"].to_numpy(float)


def predict_linear(train_df, test_df):
    """Fit on the training split only, then predict the test period."""
    model = LinearRegression()
    model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
    return model.predict(test_df[FEATURE_COLUMNS])


def to_prediction_rows(test_df, predicted):
    """Shape predictions for save_predictions. Baselines carry no uncertainty.

    The length check matters because zip() would quietly drop the tail if the two
    ever fell out of step, mis-attributing predictions with nothing to show for it.
    """
    predicted = np.ravel(predicted)
    if len(predicted) != len(test_df):
        raise ValueError(f"got {len(predicted)} predictions for {len(test_df)} rows")

    return [
        (int(row.location_id), row.obs_date, float(value), None, None, None)
        for row, value in zip(test_df.itertuples(), predicted)
    ]


def main() -> None:
    conn = connect_with_retry(load_db_config())
    try:
        features = load_features(conn)

        for location_id, location_df in features.groupby("location_id", sort=True):
            name = location_df["location_name"].iloc[0]
            train_df = location_df[location_df["split"] == "train"]
            test_df = location_df[location_df["split"] == "test"]

            log(f"{name}: {len(train_df)} train / {len(test_df)} test rows")

            actual = test_df[TARGET_COLUMN].to_numpy(float)

            for model_name, predicted in (
                (PERSISTENCE, predict_persistence(test_df)),
                (LINEAR, predict_linear(train_df, test_df)),
            ):
                save_predictions(conn, model_name, to_prediction_rows(test_df, predicted))
                scores = {"rmse": rmse(actual, predicted), "mae": mae(actual, predicted)}
                save_evaluation(conn, model_name, int(location_id), scores)
                log(f"  {model_name}: RMSE {scores['rmse']:.3f}  MAE {scores['mae']:.3f}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
