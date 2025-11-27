from __future__ import annotations

import numpy as np


def image_histogram(image: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 255))
    return hist


def otsu_threshold(image: np.ndarray) -> int:
    histogram = image_histogram(image)
    total = image.size

    sum_total = np.dot(np.arange(256), histogram)
    sum_background = 0.0
    weight_background = 0
    max_variance = 0.0
    threshold = 0

    for i in range(256):
        weight_background += histogram[i]
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += i * histogram[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        between_variance = (
            weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        )

        if between_variance > max_variance:
            max_variance = between_variance
            threshold = i

    return threshold


def binarize(image: np.ndarray, threshold: int) -> np.ndarray:
    binary = np.where(image > threshold, 255, 0)
    return binary.astype(np.uint8)
