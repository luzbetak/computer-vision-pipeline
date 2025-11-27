from __future__ import annotations

import numpy as np


def apply_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a median filter with the given kernel size."""

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    pad = kernel_size // 2
    padded = np.pad(image, pad, mode="edge")

    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i : i + kernel_size, j : j + kernel_size]
            output[i, j] = np.median(window)
    return output
