"""
SFR / MTF analyser — ISO 12233 compliant slanted-edge method.

Public API::

    analyzer = SFRAnalyzer(config)
    result   = analyzer.analyze(roi_image)        # single image
    report   = analyzer.run_batch(path_list)      # batch mode
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from iqa.metrics.mtf_utils import (
    EdgeAngleError,
    compute_esf,
    compute_mtf50,
    detect_edge_angle,
    esf_to_lsf,
    lsf_to_mtf,
)
from iqa.utils.config_loader import SFRConfig
from iqa.utils.image_io import load_image
from iqa.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SFRResult:
    """MTF/SFR result for a single ROI."""

    path: Optional[Path] = None
    label: str = ""
    status: str = "ok"           # "ok" | "error"
    error: str = ""

    edge_angle_deg: float = float("nan")
    mtf50_cy_px: float = float("nan")
    mtf50_lp_mm: float = float("nan")   # nan when pixel_size_um not set

    freq_axis: list[float] = field(default_factory=list)
    mtf_curve: list[float] = field(default_factory=list)
    esf: list[float] = field(default_factory=list)
    lsf: list[float] = field(default_factory=list)


@dataclass
class BatchSFRReport:
    """Container for a batch SFR run."""

    results: list[SFRResult] = field(default_factory=list)

    @property
    def passed(self) -> list[SFRResult]:
        return [r for r in self.results if r.status == "ok"]

    @property
    def failed(self) -> list[SFRResult]:
        return [r for r in self.results if r.status == "error"]

    def summary(self) -> dict:
        valid = [r for r in self.passed if not np.isnan(r.mtf50_cy_px)]
        mtf50s = [r.mtf50_cy_px for r in valid]
        return {
            "n_total": len(self.results),
            "n_ok": len(self.passed),
            "n_error": len(self.failed),
            "mtf50_mean_cy_px": float(np.mean(mtf50s)) if mtf50s else float("nan"),
            "mtf50_min_cy_px":  float(np.min(mtf50s))  if mtf50s else float("nan"),
            "mtf50_max_cy_px":  float(np.max(mtf50s))  if mtf50s else float("nan"),
        }


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class SFRAnalyzer:
    """
    ISO 12233 SFR / MTF analyser.

    Args:
        config: :class:`~iqa.utils.config_loader.SFRConfig` instance.
    """

    def __init__(self, config: Optional[SFRConfig] = None) -> None:
        self.config = config or SFRConfig()

    def analyze(self, roi: np.ndarray) -> SFRResult:
        """
        Compute MTF50 for a single slanted-edge ROI.

        Args:
            roi: 2-D float32 grayscale ROI [H, W].

        Returns:
            :class:`SFRResult` populated with MTF curve and MTF50.
        """
        cfg = self.config
        result = SFRResult()

        # --- Ensure grayscale ---
        if roi.ndim == 3:
            roi = 0.299 * roi[..., 0] + 0.587 * roi[..., 1] + 0.114 * roi[..., 2]
        roi = roi.astype(np.float32)

        # --- Edge detection ---
        try:
            angle, edge_pos = detect_edge_angle(roi)
        except Exception as exc:
            result.status = "error"
            result.error = f"Edge detection failed: {exc}"
            return result

        result.edge_angle_deg = angle
        abs_angle = abs(angle)
        if not (cfg.edge_angle_min <= abs_angle <= cfg.edge_angle_max):
            result.status = "error"
            result.error = (
                f"Edge angle {angle:.2f}° outside valid range "
                f"[{cfg.edge_angle_min}°, {cfg.edge_angle_max}°]"
            )
            log.warning(result.error)
            return result

        # --- ESF ---
        esf_d, esf_v = compute_esf(roi, edge_pos, oversample=cfg.oversample_factor)
        result.esf = esf_v.tolist()

        # --- LSF ---
        lsf = esf_to_lsf(
            esf_v,
            window_length=cfg.smoothing_window,
            polyorder=cfg.smoothing_polyorder,
        )
        result.lsf = lsf.tolist()

        # --- MTF ---
        freq, mtf = lsf_to_mtf(lsf, windowing=cfg.windowing)
        result.freq_axis = freq.tolist()
        result.mtf_curve = mtf.tolist()

        # --- MTF50 ---
        mtf50_cy_px = compute_mtf50(freq, mtf, oversample=cfg.oversample_factor)
        result.mtf50_cy_px = mtf50_cy_px

        if cfg.pixel_size_um > 0 and not np.isnan(mtf50_cy_px):
            # cy/px ÷ (pixel_size_um × 1e-3 mm) = lp/mm
            result.mtf50_lp_mm = mtf50_cy_px / (cfg.pixel_size_um * 1e-3)

        log.info(
            "SFR  angle=%.2f°  MTF50=%.4f cy/px",
            angle, mtf50_cy_px,
        )
        return result

    def run_batch(
        self,
        input_paths: list[Path | str],
        *,
        roi_box: Optional[tuple[int, int, int, int]] = None,
        labels: Optional[list[str]] = None,
    ) -> BatchSFRReport:
        """
        Analyse a list of image files and collect results.

        Args:
            input_paths: List of paths to single-channel or RGB images.
            roi_box:     Optional ``(x1, y1, x2, y2)`` crop applied to every image.
            labels:      Optional human-readable label per file (for reporting).

        Returns:
            :class:`BatchSFRReport` with per-image results and summary statistics.
        """
        results: list[SFRResult] = []
        for i, fpath in enumerate(input_paths):
            fpath = Path(fpath)
            label = (labels[i] if labels and i < len(labels) else fpath.stem)
            try:
                img = load_image(fpath, grayscale=True)
                if roi_box is not None:
                    x1, y1, x2, y2 = roi_box
                    img = img[y1:y2, x1:x2]
                res = self.analyze(img)
                res.path = fpath
                res.label = label
            except Exception as exc:
                log.error("SFR batch error on %s: %s", fpath, exc)
                res = SFRResult(path=fpath, label=label, status="error", error=str(exc))
            results.append(res)

        report = BatchSFRReport(results=results)
        s = report.summary()
        log.info(
            "Batch SFR done  ok=%d/%d  MTF50 mean=%.4f cy/px",
            s["n_ok"], s["n_total"], s["mtf50_mean_cy_px"],
        )
        return report
