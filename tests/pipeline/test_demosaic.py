"""Tests for the Demosaic stage."""

import numpy as np
import pytest

from iqa.pipeline.blc import BAYER_OFFSETS
from iqa.pipeline.demosaic import Demosaic, _demosaic_malvar
from iqa.utils.config_loader import DemosaicConfig


def _make_stage(algo="malvar"):
    return Demosaic(DemosaicConfig(algorithm=algo))


def _meta(pattern="RGGB", bit_depth=12):
    return {"bayer_pattern": pattern, "bit_depth": bit_depth}


def _pure_channel_bayer(h, w, pattern, channel, value=800.0):
    """Bayer image with only one channel lit; all others = 0."""
    img = np.zeros((h, w), dtype=np.float32)
    off = BAYER_OFFSETS[pattern][channel]
    img[off[0]::2, off[1]::2] = value
    return img


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_output_shape_malvar(bayer_rggb_64):
    out, _ = _make_stage("malvar")(bayer_rggb_64, _meta())
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.float32


def test_output_shape_bilinear(bayer_rggb_64):
    out, _ = _make_stage("bilinear")(bayer_rggb_64, _meta())
    assert out.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# Neutral grey → balanced channels
# ---------------------------------------------------------------------------

def test_neutral_bayer_balanced_channels():
    """If R=G=B in every Bayer position, demosaiced channels should be equal."""
    img = np.full((64, 64), 512.0, dtype=np.float32)
    out, _ = _make_stage()(img, _meta())
    # Allow small rounding from kernel edge effects
    assert np.allclose(out[..., 0], out[..., 1], atol=1.0)
    assert np.allclose(out[..., 1], out[..., 2], atol=1.0)


# ---------------------------------------------------------------------------
# Pure channel
# ---------------------------------------------------------------------------

def test_pure_red_channel():
    """Only R pixels lit → after demosaic, R >> G,B."""
    bayer = _pure_channel_bayer(64, 64, "RGGB", "R", value=4000.0)
    out, _ = _make_stage()(bayer, _meta(bit_depth=12))
    r_mean = out[..., 0].mean()
    g_mean = out[..., 1].mean()
    b_mean = out[..., 2].mean()
    assert r_mean > g_mean * 2, f"R mean {r_mean:.1f} should dominate G {g_mean:.1f}"
    assert r_mean > b_mean * 2


# ---------------------------------------------------------------------------
# All patterns produce valid 3-channel output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_all_bayer_patterns(pattern):
    bayer = np.random.default_rng(0).uniform(0, 4095, (64, 64)).astype(np.float32)
    out, meta = _make_stage()(bayer, _meta(pattern))
    assert out.shape == (64, 64, 3)
    assert meta.get("demosaiced") is True


# ---------------------------------------------------------------------------
# Unknown algorithm
# ---------------------------------------------------------------------------

def test_unknown_algorithm_raises():
    stage = Demosaic(DemosaicConfig(algorithm="nonexistent"))
    img = np.ones((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown demosaic algorithm"):
        stage(img, _meta())


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_wrong_ndim_raises():
    stage = _make_stage()
    # 3-D input should be rejected by validate_input
    with pytest.raises(ValueError):
        stage(np.ones((8, 8, 3), dtype=np.float32), _meta())


def test_wrong_dtype_raises():
    stage = _make_stage()
    with pytest.raises(ValueError):
        stage(np.ones((8, 8), dtype=np.uint16), _meta())
