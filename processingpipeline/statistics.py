from __future__ import annotations

import numpy as np


def mean(image: np.ndarray) -> float:
    return float(np.mean(image))


def variance(image: np.ndarray) -> float:
    return float(np.var(image))


def stddev(image: np.ndarray) -> float:
    return float(np.std(image))


def minimum(image: np.ndarray) -> int:
    return int(np.min(image))


def maximum(image: np.ndarray) -> int:
    return int(np.max(image))
