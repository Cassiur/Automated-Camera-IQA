"""Tests for the AWB stage."""

import numpy as np
import pytest

from iqa.pipeline.awb import AutoWhiteBalance
from iqa.utils.config_loader import AWBConfig


def _stage(method="gray_world", **kw):
    cfg = AWBConfig(method=method, **kw)
    return AutoWhiteBalance(cfg)


def _meta():
    return {}


def _rgb(h, w, r, g, b):
    """Uniform colour RGB float32 image."""
    img = np.empty((h, w, 3), dtype=np.float32)
    img[..., 0] = r
    img[..., 1] = g
    img[..., 2] = b
    return img


# ---------------------------------------------------------------------------
# Gray-world: neutral scene → gains ≈ 1
# ---------------------------------------------------------------------------

def test_gray_world_neutral_no_change():
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    out, meta = _stage()(img, _meta())
    gains = meta["awb_gains"]
    assert abs(gains[0] - 1.0) < 0.05, f"R gain should be ≈1.0, got {gains[0]}"
    assert abs(gains[2] - 1.0) < 0.05, f"B gain should be ≈1.0, got {gains[2]}"


# ---------------------------------------------------------------------------
# Gray-world: red scene → B gain > 1, R gain < 1
# ---------------------------------------------------------------------------

def test_gray_world_red_scene():
    img = _rgb(32, 32, r=0.8, g=0.5, b=0.2)
    _, meta = _stage()(img, _meta())
    gains = meta["awb_gains"]
    assert gains[0] < 1.0, "R gain should decrease for red scene"
    assert gains[2] > 1.0, "B gain should increase for red scene"


# ---------------------------------------------------------------------------
# Gains stored in meta
# ---------------------------------------------------------------------------

def test_gains_in_meta():
    img = np.ones((8, 8, 3), dtype=np.float32) * 0.5
    _, meta = _stage()(img, _meta())
    assert "awb_gains" in meta
    assert len(meta["awb_gains"]) == 3


# ---------------------------------------------------------------------------
# Manual gains applied correctly
# ---------------------------------------------------------------------------

def test_manual_gains():
    img = np.ones((4, 4, 3), dtype=np.float32) * 0.5
    stage = _stage(method="manual", manual_gains={"R": 2.0, "G": 1.0, "B": 0.5})
    out, meta = stage(img, _meta())
    assert np.allclose(out[0, 0, 0], 1.0, atol=0.01)   # 0.5 * 2.0 = 1.0 (clipped)
    assert np.allclose(out[0, 0, 1], 0.5, atol=0.01)   # unchanged
    assert np.allclose(out[0, 0, 2], 0.25, atol=0.01)  # 0.5 * 0.5


# ---------------------------------------------------------------------------
# Perfect reflector: result similar to gray-world for uniform scene
# ---------------------------------------------------------------------------

def test_perfect_reflector_uniform():
    img = np.full((32, 32, 3), 0.6, dtype=np.float32)
    out, meta = _stage(method="perfect_reflector")(img, _meta())
    gains = meta["awb_gains"]
    # Uniform image → gains still ≈ 1.0
    assert abs(gains[0] - 1.0) < 0.1
    assert abs(gains[2] - 1.0) < 0.1


# ---------------------------------------------------------------------------
# Output not exceeds 1.0 for [0,1] range input
# ---------------------------------------------------------------------------

def test_output_clipped():
    # Heavily blue image; R gain will be very high → would exceed 1 without clip
    img = _rgb(32, 32, r=0.05, g=0.5, b=0.95)
    out, _ = _stage()(img, _meta())
    assert out.max() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# Disabled stage passthrough
# ---------------------------------------------------------------------------

def test_disabled_passthrough():
    img = _rgb(8, 8, r=0.2, g=0.5, b=0.8)
    stage = AutoWhiteBalance(AWBConfig(enabled=False))
    out, _ = stage(img, _meta())
    assert np.array_equal(out, img)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_wrong_shape_raises():
    stage = _stage()
    with pytest.raises(ValueError):
        stage(np.ones((8, 8), dtype=np.float32), _meta())
