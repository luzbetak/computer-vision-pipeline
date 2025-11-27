from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def read_grayscale_image(path: str | Path) -> np.ndarray:
    """Return the green channel of the input image as a grayscale matrix.

    The legacy Java version focused on the green channel, so we mirror that
    behavior here for consistency.
    """

    image = Image.open(path).convert("RGB")
    green_channel = np.asarray(image)[:, :, 1]
    return green_channel.astype(np.float64)


def write_grayscale_image(path: str | Path, data: np.ndarray) -> None:
    """Persist a 2D grayscale matrix to disk as an 8-bit PNG."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    clipped = np.clip(data, 0, 255).astype(np.uint8)
    image = Image.fromarray(clipped, mode="L")
    image.save(path)


def scale_down(padded: np.ndarray, x_size: int, y_size: int) -> np.ndarray:
    """Remove reflection padding to restore the original image dimensions."""

    half_x = x_size // 2
    half_y = y_size // 2
    return padded[half_y : padded.shape[0] - half_y, half_x : padded.shape[1] - half_x]


def ensure_image_dimensions(match: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Trim or pad images so both share the same dimensions."""

    height = min(match.shape[0], target.shape[0])
    width = min(match.shape[1], target.shape[1])
    return match[:height, :width], target[:height, :width]
