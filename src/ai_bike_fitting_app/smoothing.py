"""Offline interpolation and Savitzky–Golay smoothing of keypoint positions."""

import numpy as np
from scipy.signal import savgol_filter

SAVGOL_WINDOW = 7
SAVGOL_ORDER = 3


def smooth_positions(points: np.ndarray) -> np.ndarray:
    """Fill bounded 1–2 frame gaps, then filter each continuous XY segment."""
    data = np.array(points, dtype=float, copy=True)
    data[~np.isfinite(data).all(axis=1)] = np.nan
    index = 0
    while index < len(data):
        if np.isfinite(data[index]).all():
            index += 1
            continue
        start = index
        while index < len(data) and not np.isfinite(data[index]).all():
            index += 1
        if start > 0 and index < len(data) and index - start < 3:
            data[start:index] = np.linspace(data[start - 1], data[index], index - start + 2)[1:-1]
    index = 0
    while index < len(data):
        if not np.isfinite(data[index]).all():
            index += 1
            continue
        start = index
        while index < len(data) and np.isfinite(data[index]).all():
            index += 1
        length = index - start
        window = min(SAVGOL_WINDOW, length if length % 2 else length - 1)
        if window >= 5:
            data[start:index] = savgol_filter(data[start:index], window, SAVGOL_ORDER, axis=0, mode="interp")
    return data
