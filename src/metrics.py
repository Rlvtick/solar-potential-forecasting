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


def picp(actual, lower, upper) -> float:
    """Share of actuals falling inside the prediction interval.

    For a nominal 95% interval this should come out near 0.95 — much lower means
    the model is claiming more certainty than it has earned.
    """
    actual, lower = _as_pair(actual, lower)
    _, upper = _as_pair(actual, upper)

    if np.any(upper < lower):
        raise ValueError("upper bound below lower bound")

    return float(np.mean((actual >= lower) & (actual <= upper)))
