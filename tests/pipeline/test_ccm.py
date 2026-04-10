"""Tests for the CCM (Colour Correction Matrix) stage."""

import numpy as np
import pytest

from iqa.pipeline.ccm import ColourCorrectionMatrix
from iqa.utils.config_loader import CCMConfig


IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

D65_CCM = [
    [ 1.732, -0.512, -0.220],
    [-0.234,  1.571, -0.337],
    [ 0.012, -0.488,  1.476],
]


def _stage(matrix=None, clip=True, enabled=True):
    m = matrix or IDENTITY
    cfg = CCMConfig(matrix=m, clip=clip, enabled=enabled)
    return ColourCorrectionMatrix(cfg)


def _rgb(r, g, b, h=4, w=4):
    img = np.empty((h, w, 3), dtype=np.float32)
    img[..., 0] = r
    img[..., 1] = g
    img[..., 2] = b
    return img


# ---------------------------------------------------------------------------
# Identity matrix → no change
# ---------------------------------------------------------------------------

def test_identity_matrix_no_change():
    img = _rgb(0.3, 0.5, 0.7)
    out, _ = _stage(IDENTITY)(img, {})
    assert np.allclose(out, img, atol=1e-5)


# ---------------------------------------------------------------------------
# Shape preserved
# ---------------------------------------------------------------------------

def test_output_shape():
    img = np.random.rand(64, 64, 3).astype(np.float32)
    out, _ = _stage()(img, {})
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Clip enforced
# ---------------------------------------------------------------------------

def test_clip_enforced():
    # Boost matrix that could produce values > 1
    boost = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    img = _rgb(0.9, 0.9, 0.9)
    out, _ = _stage(boost, clip=True)(img, {})
    assert out.max() <= 1.0 + 1e-5


def test_no_clip():
    boost = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    img = _rgb(0.9, 0.9, 0.9)
    out, _ = _stage(boost, clip=False)(img, {})
    assert out.max() > 1.0  # not clipped


# ---------------------------------------------------------------------------
# Known transform
# ---------------------------------------------------------------------------

def test_known_transform():
    """Apply a permutation matrix: R→G→B→R; verify channels cycle."""
    perm = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]  # R=G_in, G=B_in, B=R_in
    img = _rgb(r=0.1, g=0.5, b=0.9)
    out, _ = _stage(perm, clip=False)(img, {})
    assert np.allclose(out[..., 0], 0.5, atol=1e-4)   # R = G_in
    assert np.allclose(out[..., 1], 0.9, atol=1e-4)   # G = B_in
    assert np.allclose(out[..., 2], 0.1, atol=1e-4)   # B = R_in


# ---------------------------------------------------------------------------
# Meta update
# ---------------------------------------------------------------------------

def test_meta_updated():
    img = _rgb(0.3, 0.5, 0.7)
    _, meta = _stage(D65_CCM)(img, {})
    assert meta.get("ccm_applied") is True
    assert "ccm_matrix" in meta


# ---------------------------------------------------------------------------
# Bad matrix shape raises
# ---------------------------------------------------------------------------

def test_bad_matrix_shape():
    with pytest.raises(ValueError, match="shape"):
        ColourCorrectionMatrix(CCMConfig(matrix=[[1, 0], [0, 1]]))


# ---------------------------------------------------------------------------
# Disabled passthrough
# ---------------------------------------------------------------------------

def test_disabled_passthrough():
    img = _rgb(0.3, 0.5, 0.7)
    stage = ColourCorrectionMatrix(CCMConfig(matrix=IDENTITY, enabled=False))
    out, _ = stage(img, {})
    assert np.array_equal(out, img)
