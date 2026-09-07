"""Scoring functions shared across models."""

import numpy as np


def rmse(actual, predicted) -> float:
    """Root mean squared error."""
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual, predicted) -> float:
    """Mean absolute error."""
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.mean(np.abs(actual - predicted)))
