"""YAML configuration loader with dataclass-based schema validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from iqa.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# ISP config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BLCConfig:
    enabled: bool = True
    black_levels: dict[str, int] = field(default_factory=lambda: {"R": 64, "Gr": 64, "Gb": 64, "B": 64})


@dataclass
class DemosaicConfig:
    enabled: bool = True
    algorithm: str = "malvar"  # malvar | bilinear


@dataclass
class AWBConfig:
    enabled: bool = True
    method: str = "gray_world"  # gray_world | perfect_reflector | manual
    saturation_threshold: float = 0.98
    perfect_reflector_percentile: float = 1.0
    manual_gains: dict[str, float] = field(default_factory=lambda: {"R": 1.0, "G": 1.0, "B": 1.0})


@dataclass
class CCMConfig:
    enabled: bool = True
    matrix: list[list[float]] = field(
        default_factory=lambda: [
            [1.732, -0.512, -0.220],
            [-0.234, 1.571, -0.337],
            [0.012, -0.488, 1.476],
        ]
    )
    clip: bool = True


@dataclass
class PipelineConfig:
    bayer_pattern: str = "RGGB"
    bit_depth: int = 12
    output_bit_depth: int = 8
    debug_mode: bool = False

    blc: BLCConfig = field(default_factory=BLCConfig)
    demosaic: DemosaicConfig = field(default_factory=DemosaicConfig)
    awb: AWBConfig = field(default_factory=AWBConfig)
    ccm: CCMConfig = field(default_factory=CCMConfig)


# ---------------------------------------------------------------------------
# SFR config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SFRConfig:
    oversample_factor: int = 4
    edge_angle_min: float = 4.0
    edge_angle_max: float = 15.0
    smoothing_window: int = 11
    smoothing_polyorder: int = 3
    windowing: str = "hamming"
    pixel_size_um: float = 0.0   # 0 = skip lp/mm conversion


# ---------------------------------------------------------------------------
# Boresight config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BoresightThresholds:
    reprojection_rms_px: float = 0.5
    baseline_dist_mm: float = 1.0
    baseline_angle_deg: float = 0.1


@dataclass
class BoresightConfig:
    thresholds: BoresightThresholds = field(default_factory=BoresightThresholds)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _deep_get(d: dict, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dict."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def load_isp_config(path: str | Path) -> PipelineConfig:
    """Load and parse an ISP YAML config file into :class:`PipelineConfig`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    log.debug("Loaded ISP config from %s", path)

    p = raw.get("pipeline", {})
    stages = raw.get("stages", {})

    blc_raw = stages.get("blc", {})
    bl = blc_raw.get("black_levels", {})
    if isinstance(bl, int):
        bl = {"R": bl, "Gr": bl, "Gb": bl, "B": bl}

    awb_raw = stages.get("awb", {})
    pr = awb_raw.get("perfect_reflector", {})
    mg = awb_raw.get("manual_gains", {})

    ccm_raw = stages.get("ccm", {})

    return PipelineConfig(
        bayer_pattern=p.get("bayer_pattern", "RGGB"),
        bit_depth=p.get("bit_depth", 12),
        output_bit_depth=p.get("output_bit_depth", 8),
        debug_mode=p.get("debug_mode", False),
        blc=BLCConfig(
            enabled=blc_raw.get("enabled", True),
            black_levels=bl or {"R": 64, "Gr": 64, "Gb": 64, "B": 64},
        ),
        demosaic=DemosaicConfig(
            enabled=stages.get("demosaic", {}).get("enabled", True),
            algorithm=stages.get("demosaic", {}).get("algorithm", "malvar"),
        ),
        awb=AWBConfig(
            enabled=awb_raw.get("enabled", True),
            method=awb_raw.get("method", "gray_world"),
            saturation_threshold=awb_raw.get("saturation_threshold", 0.98),
            perfect_reflector_percentile=pr.get("top_percentile", 1.0),
            manual_gains=mg or {"R": 1.0, "G": 1.0, "B": 1.0},
        ),
        ccm=CCMConfig(
            enabled=ccm_raw.get("enabled", True),
            matrix=ccm_raw.get("matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            clip=ccm_raw.get("clip", True),
        ),
    )


def load_sfr_config(path: str | Path) -> SFRConfig:
    """Load SFR YAML config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    s = raw.get("sfr", {})
    cam = raw.get("camera", {})
    sm = s.get("smoothing", {})
    return SFRConfig(
        oversample_factor=s.get("oversample_factor", 4),
        edge_angle_min=s.get("edge_angle_min", 4.0),
        edge_angle_max=s.get("edge_angle_max", 15.0),
        smoothing_window=sm.get("window_length", 11),
        smoothing_polyorder=sm.get("polyorder", 3),
        windowing=s.get("windowing", "hamming"),
        pixel_size_um=cam.get("pixel_size_um", 0.0),
    )


def load_yaml(path: str | Path) -> dict:
    """Generic YAML loader — returns raw dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
