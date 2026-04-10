"""
Shared pytest fixtures for the IQA test suite.

Fixtures provide synthetic ground-truth data so tests can verify correctness
without requiring real sensor images.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqa.utils.config_loader import (
    AWBConfig,
    BLCConfig,
    CCMConfig,
    DemosaicConfig,
    PipelineConfig,
    SFRConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bayer(
    h: int, w: int,
    pattern: str = "RGGB",
    r_val: float = 800.0,
    g_val: float = 512.0,
    b_val: float = 300.0,
) -> np.ndarray:
    """Create a synthetic flat-field Bayer image with known channel values."""
    from iqa.pipeline.blc import BAYER_OFFSETS
    img = np.zeros((h, w), dtype=np.float32)
    off = BAYER_OFFSETS[pattern]
    img[off["R"][0]::2,  off["R"][1]::2]  = r_val
    img[off["Gr"][0]::2, off["Gr"][1]::2] = g_val
    img[off["Gb"][0]::2, off["Gb"][1]::2] = g_val
    img[off["B"][0]::2,  off["B"][1]::2]  = b_val
    return img


def _slanted_edge(
    h: int = 64,
    w: int = 64,
    angle_deg: float = 7.0,
    dark: float = 0.0,
    bright: float = 1.0,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """
    Generate a synthetic slanted-edge image.

    The edge is placed at the horizontal midpoint and tilted by *angle_deg*.
    Optionally Gaussian-blurred to simulate defocus.
    """
    cols = np.arange(w, dtype=np.float64)
    rows = np.arange(h, dtype=np.float64)
    C, R = np.meshgrid(cols, rows)

    slope = np.tan(np.radians(angle_deg))
    edge_x = w / 2 + slope * (R - h / 2)

    img = np.where(C < edge_x, dark, bright).astype(np.float32)

    if blur_sigma > 0:
        import cv2
        ksize = int(6 * blur_sigma + 1) | 1  # odd kernel
        img = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)

    return img


# ---------------------------------------------------------------------------
# Fixtures — ISP Pipeline
# ---------------------------------------------------------------------------

@pytest.fixture
def bayer_rggb_64():
    """64×64 RGGB Bayer, R=800 Gr/Gb=512 B=300, float32."""
    return _make_bayer(64, 64, pattern="RGGB")


@pytest.fixture
def bayer_bggr_64():
    """64×64 BGGR Bayer, same channel values as rggb_64."""
    return _make_bayer(64, 64, pattern="BGGR")


@pytest.fixture
def flat_rgb_neutral():
    """128×128 neutral-grey RGB image, all channels = 0.5, float32."""
    return np.full((128, 128, 3), 0.5, dtype=np.float32)


@pytest.fixture
def default_isp_config():
    """PipelineConfig with default values (no YAML needed)."""
    return PipelineConfig()


@pytest.fixture
def debug_isp_config():
    """PipelineConfig with debug_mode=True."""
    return PipelineConfig(debug_mode=True)


# ---------------------------------------------------------------------------
# Fixtures — SFR / MTF
# ---------------------------------------------------------------------------

@pytest.fixture
def ideal_slanted_edge():
    """64×64 ideal slanted edge at 7° — MTF50 should be ≈ 0.5 cy/px."""
    return _slanted_edge(h=64, w=64, angle_deg=7.0)


@pytest.fixture
def blurred_slanted_edge():
    """64×64 Gaussian-blurred (σ=1.0) slanted edge — MTF50 < 0.5 cy/px."""
    return _slanted_edge(h=64, w=64, angle_deg=7.0, blur_sigma=1.0)


@pytest.fixture
def sfr_config_default():
    """Default SFRConfig."""
    return SFRConfig()


# ---------------------------------------------------------------------------
# Fixtures — Boresight / Calibration
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_cameras():
    """
    Three cameras arranged as: front (identity), left (90° Y-rotation, 300mm offset),
    rear (180° Y-rotation, 1500mm offset).  Reprojection error should be exactly 0
    when using the correct world-to-pixel projection.
    """
    from iqa.calibration.extrinsic import CameraExtrinsic

    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)

    R_front = np.eye(3, dtype=np.float64)
    t_front = np.zeros((3, 1), dtype=np.float64)

    # Left camera: rotated 90° around Y, translated 300 mm in X
    angle = np.radians(90.0)
    R_left = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0,             1, 0            ],
        [-np.sin(angle),0, np.cos(angle)],
    ])
    t_left = np.array([[300.0], [0.0], [0.0]])

    # Rear camera: rotated 180° around Y, translated 1500 mm in X
    R_rear = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    t_rear = np.array([[1500.0], [0.0], [0.0]])

    return [
        CameraExtrinsic("front", R_front, t_front, K),
        CameraExtrinsic("left",  R_left,  t_left,  K),
        CameraExtrinsic("rear",  R_rear,  t_rear,  K),
    ]


@pytest.fixture
def world_points_grid():
    """5×5 grid of world points on the Z=5000 mm plane."""
    xs = np.linspace(-500, 500, 5)
    ys = np.linspace(-400, 400, 5)
    X, Y = np.meshgrid(xs, ys)
    Z = np.full_like(X, 5000.0)
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    return pts.astype(np.float64)
