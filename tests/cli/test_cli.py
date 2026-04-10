"""CLI smoke tests using Click's CliRunner."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from iqa.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_bayer_png(tmp_path):
    """Write a 64×64 grayscale PNG that looks like a flat Bayer image."""
    img = np.full((64, 64), 128, dtype=np.uint8)
    p = tmp_path / "test_raw.png"
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture
def sample_edge_png(tmp_path):
    """Write a slanted-edge greyscale PNG for SFR testing."""
    cols = np.arange(64, dtype=np.float32)
    rows = np.arange(64, dtype=np.float32)
    C, R = np.meshgrid(cols, rows)
    slope = np.tan(np.radians(7.0))
    edge_x = 32 + slope * (R - 32)
    img = np.where(C < edge_x, 0, 255).astype(np.uint8)
    p = tmp_path / "edge.png"
    cv2.imwrite(str(p), img)
    return p


# ---------------------------------------------------------------------------
# --help on all commands
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cmd", ["isp", "sfr", "boresight", ""])
def test_help(runner, cmd):
    args = [cmd, "--help"] if cmd else ["--help"]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output


# ---------------------------------------------------------------------------
# iqa isp
# ---------------------------------------------------------------------------

def test_isp_smoke(runner, sample_bayer_png, tmp_path):
    """iqa isp should process one PNG without crashing."""
    out_dir = tmp_path / "out"
    result = runner.invoke(cli, [
        "isp",
        "--input", str(sample_bayer_png),
        "--output", str(out_dir),
        "--bayer", "RGGB",
        "--bit-depth", "8",
    ])
    assert result.exit_code == 0, result.output
    assert out_dir.exists()


def test_isp_missing_input(runner, tmp_path):
    """Non-existent glob should exit with error code."""
    result = runner.invoke(cli, [
        "isp",
        "--input", str(tmp_path / "no_match_*.raw"),
        "--output", str(tmp_path / "out"),
    ])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# iqa sfr
# ---------------------------------------------------------------------------

def test_sfr_smoke(runner, sample_edge_png, tmp_path):
    out_dir = tmp_path / "sfr_out"
    result = runner.invoke(cli, [
        "sfr",
        "--input", str(sample_edge_png),
        "--output", str(out_dir),
        "--format", "json",
    ])
    assert result.exit_code == 0, result.output
    # Report file should be created
    assert (out_dir / "sfr_report.json").exists()


def test_sfr_missing_input(runner, tmp_path):
    result = runner.invoke(cli, [
        "sfr",
        "--input", str(tmp_path / "no_match_*.png"),
        "--output", str(tmp_path / "out"),
    ])
    assert result.exit_code != 0
