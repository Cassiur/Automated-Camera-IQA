"""Report generation: JSON / CSV output for SFR batch results."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from iqa.utils.logger import get_logger

if TYPE_CHECKING:
    from iqa.metrics.sfr import BatchSFRReport, SFRResult

log = get_logger(__name__)


def _nan_safe(v: float) -> object:
    """Replace NaN / Inf with None for JSON serialisability."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _result_to_dict(r: "SFRResult") -> dict:
    return {
        "label":          r.label,
        "path":           str(r.path) if r.path else "",
        "status":         r.status,
        "error":          r.error,
        "edge_angle_deg": _nan_safe(r.edge_angle_deg),
        "mtf50_cy_px":    _nan_safe(r.mtf50_cy_px),
        "mtf50_lp_mm":    _nan_safe(r.mtf50_lp_mm),
        # Compact MTF curve representation: list of [freq, mtf] pairs
        "mtf_curve":      [
            [round(f, 6), round(m, 6)]
            for f, m in zip(r.freq_axis, r.mtf_curve)
        ],
    }


def save_json(report: "BatchSFRReport", output_dir: Path | str) -> Path:
    """
    Write a full JSON report to *output_dir*/sfr_report.json.

    Returns the path of the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sfr_report.json"

    payload = {
        "summary": report.summary(),
        "results": [_result_to_dict(r) for r in report.results],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    log.info("JSON report → %s", out_path)
    return out_path


def save_csv(report: "BatchSFRReport", output_dir: Path | str) -> Path:
    """
    Write a flat CSV report to *output_dir*/sfr_report.csv.

    Columns: label, path, status, error, edge_angle_deg, mtf50_cy_px, mtf50_lp_mm.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sfr_report.csv"

    fieldnames = [
        "label", "path", "status", "error",
        "edge_angle_deg", "mtf50_cy_px", "mtf50_lp_mm",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in report.results:
            row = _result_to_dict(r)
            row.pop("mtf_curve", None)
            writer.writerow(row)

    log.info("CSV report  → %s", out_path)
    return out_path


def save_report(
    report: "BatchSFRReport",
    output_dir: Path | str,
    formats: list[str] | None = None,
) -> list[Path]:
    """
    Convenience wrapper — write report in the requested *formats*.

    Args:
        report:     :class:`~iqa.metrics.sfr.BatchSFRReport`.
        output_dir: Destination directory.
        formats:    List of ``"json"``, ``"csv"``; defaults to both.

    Returns:
        List of written file paths.
    """
    if formats is None:
        formats = ["json", "csv"]
    paths: list[Path] = []
    for fmt in formats:
        if fmt == "json":
            paths.append(save_json(report, output_dir))
        elif fmt == "csv":
            paths.append(save_csv(report, output_dir))
        else:
            log.warning("Unknown report format '%s' — skipping.", fmt)
    return paths
