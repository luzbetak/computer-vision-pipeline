from __future__ import annotations

import numpy as np


def histogram(image: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 255))
    return hist


def set_cutoff(hist: np.ndarray, total_pixels: int, percent: float) -> tuple[np.ndarray, int, int]:
    percent = percent / 100.0
    cutoff = int(total_pixels * percent)
    cutoff_high = total_pixels - cutoff

    cumulative = 0
    trimmed_hist = np.zeros_like(hist)
    first = 0
    last = 0

    for key, value in enumerate(hist):
        cumulative += value
        if cumulative < cutoff:
            trimmed_hist[key] = 0
            first = key + 1
        elif cumulative > cutoff_high:
            if last == 0:
                last = key - 1
            trimmed_hist[key] = 255
        else:
            trimmed_hist[key] = value

    return trimmed_hist, first, last


def stretch_map(first: int, last: int) -> np.ndarray:
    stretched = np.zeros(256, dtype=np.int32)
    if last == first:
        return stretched

    increment = 253 / (last - first)
    for old_key in range(first, last + 1):
        stretched[old_key] = int(((old_key - first) * increment) + 1)
    return stretched


def apply_stretch(image: np.ndarray, first: int, last: int, lookup: np.ndarray | None = None) -> np.ndarray:
    stretched = image.copy()
    below_mask = stretched < first
    above_mask = stretched > last

    stretched[below_mask] = 0
    stretched[above_mask] = 255

    if lookup is not None and lookup.size > 0:
        middle_mask = ~below_mask & ~above_mask
        stretched[middle_mask] = lookup[stretched[middle_mask].astype(int)]

    return stretched
