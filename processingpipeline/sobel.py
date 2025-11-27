from __future__ import annotations

import numpy as np

GX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
GY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)


def sobel_filters(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Sobel magnitude and direction images."""

    padded = np.pad(image, 1, mode="edge")
    gx = np.zeros_like(image, dtype=np.float64)
    gy = np.zeros_like(image, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i : i + 3, j : j + 3]
            gx[i, j] = np.sum(window * GX)
            gy[i, j] = np.sum(window * GY)

    magnitude = np.hypot(gx, gy)
    direction = np.arctan2(gy, gx)

    scaled_magnitude = (magnitude * 255) / np.hypot(1024, 1024)
    scaled_direction = ((direction + np.pi) / (2 * np.pi)) * 255
    return scaled_magnitude, scaled_direction
