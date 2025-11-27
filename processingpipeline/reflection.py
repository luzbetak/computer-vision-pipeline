from __future__ import annotations

import numpy as np


def apply_reflection_padding(image: np.ndarray, x_size: int, y_size: int) -> np.ndarray:
    """Pad the image using reflection, matching the Java implementation."""

    half_x = x_size // 2
    half_y = y_size // 2
    return np.pad(image, ((half_y, half_y), (half_x, half_x)), mode="reflect")
