from __future__ import annotations

import numpy as np


def gaussian_kernel(sigma: float, size: int) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel.astype(np.float64)


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve an image with a kernel using boundary clamping."""

    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")

    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i : i + k_h, j : j + k_w]
            output[i, j] = np.sum(window * kernel)
    return output
