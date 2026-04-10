"""Tests for BLC (Black Level Correction) stage."""

import numpy as np
import pytest

from iqa.pipeline.blc import BAYER_OFFSETS, BlackLevelCorrection
from iqa.utils.config_loader import BLCConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stage(bl=64, enabled=True):
    cfg = BLCConfig(
        enabled=enabled,
        black_levels={"R": bl, "Gr": bl, "Gb": bl, "B": bl},
    )
    return BlackLevelCorrection(cfg)


def _meta(pattern="RGGB", bit_depth=12):
    return {"bayer_pattern": pattern, "bit_depth": bit_depth}


# ---------------------------------------------------------------------------
# Basic subtraction
# ---------------------------------------------------------------------------

def test_basic_subtraction():
    """Pixel value 128 minus BL 64 should give 64 in all channels."""
    img = np.full((4, 4), 128.0, dtype=np.float32)
    stage = _make_stage(bl=64)
    out, _ = stage(img, _meta())
    assert np.allclose(out, 64.0), f"Expected 64.0, got {out}"


def test_clip_negative_to_zero():
    """Values below black level must be clipped to 0."""
    img = np.full((4, 4), 32.0, dtype=np.float32)
    stage = _make_stage(bl=64)
    out, _ = stage(img, _meta())
    assert np.all(out == 0.0)


def test_no_clip_at_maximum():
    """Max value (4095 for 12-bit) should remain unchanged when BL=0."""
    img = np.full((4, 4), 4095.0, dtype=np.float32)
    stage = _make_stage(bl=0)
    out, _ = stage(img, _meta(bit_depth=12))
    assert np.allclose(out, 4095.0)


def test_per_channel_independence():
    """Each Bayer channel uses its own black level."""
    cfg = BLCConfig(
        enabled=True,
        black_levels={"R": 100, "Gr": 50, "Gb": 50, "B": 200},
    )
    stage = BlackLevelCorrection(cfg)
    img = np.full((8, 8), 300.0, dtype=np.float32)
    meta = _meta("RGGB")
    out, _ = stage(img, meta)

    off = BAYER_OFFSETS["RGGB"]
    assert np.allclose(out[off["R"][0]::2,  off["R"][1]::2],  200.0)  # 300-100
    assert np.allclose(out[off["Gr"][0]::2, off["Gr"][1]::2], 250.0)  # 300-50
    assert np.allclose(out[off["B"][0]::2,  off["B"][1]::2],  100.0)  # 300-200


# ---------------------------------------------------------------------------
# All four Bayer patterns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_all_bayer_patterns_run(pattern):
    """BLC should complete without error for every supported pattern."""
    img = np.full((8, 8), 200.0, dtype=np.float32)
    stage = _make_stage(bl=64)
    out, meta = stage(img, _meta(pattern))
    assert out.shape == img.shape
    assert meta.get("blc_applied") is True


def test_invalid_pattern_raises():
    img = np.full((8, 8), 200.0, dtype=np.float32)
    stage = _make_stage()
    with pytest.raises(ValueError, match="Unknown bayer_pattern"):
        stage(img, {"bayer_pattern": "XYZW", "bit_depth": 12})


# ---------------------------------------------------------------------------
# Disabled stage
# ---------------------------------------------------------------------------

def test_disabled_stage_passthrough():
    img = np.full((4, 4), 200.0, dtype=np.float32)
    stage = _make_stage(enabled=False)
    out, _ = stage(img, _meta())
    assert np.array_equal(out, img)


# ---------------------------------------------------------------------------
# Meta update
# ---------------------------------------------------------------------------

def test_meta_updated():
    img = np.full((4, 4), 200.0, dtype=np.float32)
    stage = _make_stage(bl=64)
    _, meta = stage(img, _meta())
    assert meta.get("blc_applied") is True
    assert "black_levels" in meta
