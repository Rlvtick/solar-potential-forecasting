"""Scoring functions shared across models."""

import numpy as np


def _as_pair(actual, predicted):
    """Flatten both inputs and check they line up.

    Without the ravel, a (n,) actual against a (n,1) prediction broadcasts into an
    n x n matrix instead of failing, and the metric comes back plausible but wrong
    — sklearn hands back a column vector if the model was fit on a 1-column frame.
    NaN is rejected here too, since it would otherwise sail through the metric and
    get written to Postgres as a NUMERIC 'NaN'.
    """
    actual = np.asarray(actual, float).ravel()
    predicted = np.asarray(predicted, float).ravel()

    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: actual {actual.shape}, predicted {predicted.shape}")
    if actual.size == 0:
        raise ValueError("no values to score")
    if np.isnan(actual).any() or np.isnan(predicted).any():
        raise ValueError("NaN in actual or predicted values")

    return actual, predicted


def rmse(actual, predicted) -> float:
    """Root mean squared error."""
    actual, predicted = _as_pair(actual, predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual, predicted) -> float:
    """Mean absolute error."""
    actual, predicted = _as_pair(actual, predicted)
    return float(np.mean(np.abs(actual - predicted)))
