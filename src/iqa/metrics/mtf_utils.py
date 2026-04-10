"""
Low-level MTF/SFR utility functions (ISO 12233 pipeline).

Processing chain:
  slanted-edge ROI  →  edge_detect  →  ESF (super-sampled)
  →  LSF (smoothed)  →  windowed FFT  →  MTF curve  →  MTF50
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Edge detection & angle estimation
# ---------------------------------------------------------------------------

class EdgeAngleError(ValueError):
    """Raised when the detected edge angle is outside the ISO 12233 range."""


def detect_edge_angle(roi: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Estimate slanted-edge angle and per-row edge positions via linear regression.

    Algorithm:
    1. For each row, locate the sub-pixel midpoint of the dark-to-bright
       transition (centroid of the gradient).
    2. Fit a line through (row_index, edge_position) to get angle.

    Args:
        roi: 2-D float32 grayscale ROI [H, W] containing a single slanted edge.

    Returns:
        Tuple of (angle_degrees, edge_x_per_row) where *edge_x_per_row* is a
        1-D array of estimated edge x-positions for each row.

    Raises:
        EdgeAngleError: If the detected angle is outside [4°, 15°] (configurable
                        by the caller).
    """
    H, W = roi.shape
    rows = np.arange(H, dtype=np.float64)
    edge_positions = np.empty(H, dtype=np.float64)

    for r in range(H):
        line = roi[r].astype(np.float64)
        grad = np.abs(np.gradient(line))
        total = grad.sum()
        if total < 1e-9:
            edge_positions[r] = W / 2.0
        else:
            edge_positions[r] = np.sum(grad * np.arange(W)) / total

    # Linear regression: x = a*row + b
    A = np.vstack([rows, np.ones(H)]).T
    result = np.linalg.lstsq(A, edge_positions, rcond=None)
    a, _ = result[0]
    angle_deg = float(np.degrees(np.arctan(a)))
    return angle_deg, edge_positions


# ---------------------------------------------------------------------------
# ESF (Edge Spread Function) — super-sampled
# ---------------------------------------------------------------------------

def compute_esf(
    roi: np.ndarray,
    edge_positions: np.ndarray,
    oversample: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a super-sampled Edge Spread Function from a slanted-edge ROI.

    Each pixel is mapped to a sub-pixel distance from the edge line.
    Samples are binned and their median is taken per bin to suppress noise.

    Args:
        roi:            2-D grayscale float32 ROI [H, W].
        edge_positions: Per-row fractional edge x-positions (length H).
        oversample:     Super-sampling factor (default 4).

    Returns:
        Tuple of (distances, esf_values) — 1-D arrays of equal length, sorted
        by distance.
    """
    H, W = roi.shape
    distances: list[float] = []
    values: list[float] = []

    for r in range(H):
        ex = edge_positions[r]
        for c in range(W):
            distances.append(float(c) - ex)
            values.append(float(roi[r, c]))

    distances = np.array(distances)
    values = np.array(values)

    # Sort by distance
    order = np.argsort(distances)
    distances = distances[order]
    values = values[order]

    # Bin into oversample bins per pixel
    bin_width = 1.0 / oversample
    d_min = distances[0]
    d_max = distances[-1]
    bins = np.arange(d_min, d_max + bin_width, bin_width)
    indices = np.digitize(distances, bins)

    esf_d: list[float] = []
    esf_v: list[float] = []
    for b in range(1, len(bins)):
        mask = indices == b
        if mask.sum() == 0:
            continue
        esf_d.append(float(bins[b - 1] + bin_width / 2))
        esf_v.append(float(np.median(values[mask])))

    return np.array(esf_d, dtype=np.float64), np.array(esf_v, dtype=np.float64)


# ---------------------------------------------------------------------------
# LSF (Line Spread Function)
# ---------------------------------------------------------------------------

def esf_to_lsf(
    esf: np.ndarray,
    *,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Differentiate the ESF to obtain the LSF.

    Savitzky-Golay smoothing is applied before differentiation to reduce
    noise amplification.

    Args:
        esf:           1-D ESF values.
        window_length: Savitzky-Golay window (must be odd, ≥ polyorder+2).
        polyorder:     Polynomial order for Savitzky-Golay filter.

    Returns:
        1-D LSF array (same length as *esf*).
    """
    wl = window_length
    # Ensure window length is odd and at least polyorder + 2
    if wl % 2 == 0:
        wl += 1
    wl = max(wl, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)
    wl = min(wl, len(esf) if len(esf) % 2 == 1 else len(esf) - 1)

    smoothed = savgol_filter(esf, window_length=wl, polyorder=polyorder)
    lsf = np.gradient(smoothed)
    return lsf


# ---------------------------------------------------------------------------
# MTF computation
# ---------------------------------------------------------------------------

def lsf_to_mtf(
    lsf: np.ndarray,
    *,
    windowing: str = "hamming",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the MTF from the LSF via windowed FFT.

    Args:
        lsf:       1-D LSF array.
        windowing: Window function name: ``"hamming"``, ``"hann"``, or ``"none"``.

    Returns:
        Tuple of (freq_cy_per_px, mtf) — positive-frequency MTF, normalised
        so that MTF[0] = 1.0 (DC = unity).
    """
    N = len(lsf)

    if windowing == "hamming":
        win = np.hamming(N)
    elif windowing in ("hann", "hanning"):
        win = np.hanning(N)
    else:
        win = np.ones(N)

    lsf_windowed = lsf * win
    spectrum = np.fft.fft(lsf_windowed)
    magnitude = np.abs(spectrum)

    dc = magnitude[0]
    if dc < 1e-12:
        dc = 1.0
    mtf = magnitude / dc

    freq = np.fft.fftfreq(N, d=1.0)  # cycles per sample (pixel @ oversample=1)

    # Return positive frequencies only (Nyquist included)
    half = N // 2 + 1
    return freq[:half], mtf[:half]


# ---------------------------------------------------------------------------
# MTF50
# ---------------------------------------------------------------------------

def compute_mtf50(
    freq: np.ndarray,
    mtf: np.ndarray,
    oversample: int = 1,
) -> float:
    """
    Find the spatial frequency at which MTF = 0.5 (MTF50).

    Linear interpolation is used between the two MTF samples straddling 0.5.

    Args:
        freq:       Frequency axis (cycles per sample or per pixel).
        mtf:        Corresponding MTF values, DC-normalised.
        oversample: Super-sampling factor used during ESF computation.
                    The returned MTF50 is divided by *oversample* to convert
                    to cycles/pixel.

    Returns:
        MTF50 in cycles per pixel, or ``float('nan')`` if MTF never crosses 0.5.
    """
    # Work on indices after the DC peak (skip index 0)
    freq_pos = freq[1:]
    mtf_pos  = mtf[1:]

    # Find first crossing below 0.5
    below = np.where(mtf_pos < 0.5)[0]
    if len(below) == 0:
        return float("nan")
    idx = below[0]
    if idx == 0:
        return float(freq_pos[0] / oversample)

    # Linear interpolation
    f0, m0 = freq_pos[idx - 1], mtf_pos[idx - 1]
    f1, m1 = freq_pos[idx],     mtf_pos[idx]
    if abs(m1 - m0) < 1e-12:
        return float((f0 + f1) / 2 / oversample)
    f50 = f0 + (0.5 - m0) * (f1 - f0) / (m1 - m0)
    return float(f50 / oversample)
