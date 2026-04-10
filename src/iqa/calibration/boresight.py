"""
Boresight Alignment checker.

Validates multi-camera extrinsic consistency by:
  1. Computing per-camera reprojection RMS error.
  2. Checking camera-to-camera baseline distances and rotation angles against
     nominal (design-value) specifications.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from iqa.calibration.extrinsic import (
    CameraExtrinsic,
    make_transform,
    relative_transform,
    rotation_angle_deg,
)
from iqa.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CameraReprojResult:
    camera_id: str
    rms_px: float
    n_points: int
    status: str    # "PASS" | "FAIL" | "NO_DATA"


@dataclass
class BaselineResult:
    pair: tuple[str, str]
    dist_mm: float
    angle_deg: float
    nominal_dist_mm: Optional[float]
    nominal_angle_deg: Optional[float]
    dist_deviation_mm: float
    angle_deviation_deg: float
    status: str    # "PASS" | "FAIL" | "N/A"


@dataclass
class BoresightReport:
    overall_rms_px: float
    overall_status: str
    per_camera: list[CameraReprojResult] = field(default_factory=list)
    baselines: list[BaselineResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "overall_rms_px": self.overall_rms_px,
                "status": self.overall_status,
            },
            "per_camera": [
                {
                    "id":       c.camera_id,
                    "rms_px":   round(c.rms_px, 4),
                    "n_points": c.n_points,
                    "status":   c.status,
                }
                for c in self.per_camera
            ],
            "baselines": [
                {
                    "pair":                 list(b.pair),
                    "dist_mm":              round(b.dist_mm, 3),
                    "angle_deg":            round(b.angle_deg, 4),
                    "nominal_dist_mm":      b.nominal_dist_mm,
                    "nominal_angle_deg":    b.nominal_angle_deg,
                    "dist_deviation_mm":    round(b.dist_deviation_mm, 3),
                    "angle_deviation_deg":  round(b.angle_deviation_deg, 4),
                    "status":               b.status,
                }
                for b in self.baselines
            ],
        }

    def to_text(self) -> str:
        lines = [
            "=" * 60,
            f"Boresight Alignment Report",
            f"  Overall RMS : {self.overall_rms_px:.4f} px  [{self.overall_status}]",
            "-" * 60,
            "Per-Camera Reprojection:",
        ]
        for c in self.per_camera:
            lines.append(
                f"  {c.camera_id:<12s}  RMS={c.rms_px:.4f} px  "
                f"pts={c.n_points}  [{c.status}]"
            )
        lines += ["-" * 60, "Camera-to-Camera Baselines:"]
        for b in self.baselines:
            lines.append(
                f"  {b.pair[0]} ↔ {b.pair[1]:<10s}  "
                f"dist={b.dist_mm:.2f} mm (Δ{b.dist_deviation_mm:+.3f})  "
                f"angle={b.angle_deg:.3f}° (Δ{b.angle_deviation_deg:+.4f})  "
                f"[{b.status}]"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BoresightChecker
# ---------------------------------------------------------------------------

class BoresightChecker:
    """
    Multi-camera boresight alignment validator.

    Args:
        cameras:         List of :class:`~iqa.calibration.extrinsic.CameraExtrinsic`.
        rms_threshold:   Maximum allowed per-camera reprojection RMS (pixels).
        dist_threshold:  Maximum allowed baseline distance deviation (mm).
        angle_threshold: Maximum allowed baseline rotation deviation (degrees).
    """

    def __init__(
        self,
        cameras: list[CameraExtrinsic],
        *,
        rms_threshold: float = 0.5,
        dist_threshold: float = 1.0,
        angle_threshold: float = 0.1,
    ) -> None:
        self.cameras = cameras
        self.rms_threshold   = rms_threshold
        self.dist_threshold  = dist_threshold
        self.angle_threshold = angle_threshold

    # ------------------------------------------------------------------
    # Reprojection error
    # ------------------------------------------------------------------

    def reprojection_rms(
        self,
        world_points: np.ndarray,
        observations: dict[str, np.ndarray],
    ) -> list[CameraReprojResult]:
        """
        Compute per-camera reprojection RMS.

        Args:
            world_points:  [N, 3] float64 world coordinates.
            observations:  Dict mapping camera_id → [N, 2] observed pixel coords.

        Returns:
            List of :class:`CameraReprojResult`.
        """
        results: list[CameraReprojResult] = []
        for cam in self.cameras:
            obs = observations.get(cam.camera_id)
            if obs is None:
                results.append(CameraReprojResult(
                    camera_id=cam.camera_id, rms_px=float("nan"),
                    n_points=0, status="NO_DATA",
                ))
                continue

            N = len(world_points)
            errors = np.empty(N)
            for i in range(N):
                P_w = world_points[i].reshape(3, 1)
                P_c = cam.R @ P_w + cam.t          # [3, 1]
                if P_c[2, 0] <= 0:
                    errors[i] = float("nan")
                    continue
                # Perspective projection (pinhole, no distortion correction here)
                p = cam.K @ P_c
                p_px = (p[:2] / p[2]).ravel()      # [2]
                errors[i] = float(np.linalg.norm(p_px - obs[i]))

            valid = errors[~np.isnan(errors)]
            rms = float(np.sqrt(np.mean(valid ** 2))) if len(valid) else float("nan")
            status = "PASS" if rms <= self.rms_threshold else "FAIL"
            results.append(CameraReprojResult(
                camera_id=cam.camera_id,
                rms_px=rms,
                n_points=len(valid),
                status=status,
            ))
            log.debug("Camera %s  RMS=%.4f px  [%s]", cam.camera_id, rms, status)

        return results

    # ------------------------------------------------------------------
    # Baseline consistency
    # ------------------------------------------------------------------

    def check_baselines(
        self,
        nominal: Optional[list[dict]] = None,
    ) -> list[BaselineResult]:
        """
        Compute camera-to-camera baselines and compare against nominal specs.

        Args:
            nominal: List of dicts with keys ``pair``, ``distance_mm``,
                     ``angle_deg``.  May be None (all results will be "N/A").

        Returns:
            List of :class:`BaselineResult`.
        """
        nominal = nominal or []
        nominal_map: dict[frozenset, dict] = {
            frozenset(entry["pair"]): entry for entry in nominal
        }

        cam_map = {c.camera_id: c for c in self.cameras}
        ids = list(cam_map.keys())
        results: list[BaselineResult] = []

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_i, id_j = ids[i], ids[j]
                ci, cj = cam_map[id_i], cam_map[id_j]

                T_i = make_transform(ci.R, ci.t)
                T_j = make_transform(cj.R, cj.t)
                T_ij = relative_transform(T_i, T_j)

                dist = float(np.linalg.norm(T_ij[:3, 3])) * 1000.0  # m → mm if units match
                # If translation magnitude looks like it's already in mm (>1), keep as-is
                # heuristic: if raw value < 10 assume metres
                if dist < 10.0:
                    dist_mm = dist * 1000.0   # likely in metres
                else:
                    dist_mm = dist            # already mm-scale

                angle = rotation_angle_deg(T_ij[:3, :3])

                key = frozenset([id_i, id_j])
                nom = nominal_map.get(key)
                if nom:
                    nom_dist  = float(nom["distance_mm"])
                    nom_angle = float(nom["angle_deg"])
                    d_dev = abs(dist_mm  - nom_dist)
                    a_dev = abs(angle    - nom_angle)
                    status = "PASS" if (d_dev <= self.dist_threshold
                                        and a_dev <= self.angle_threshold) else "FAIL"
                else:
                    nom_dist = nom_angle = None
                    d_dev = a_dev = float("nan")
                    status = "N/A"

                results.append(BaselineResult(
                    pair=(id_i, id_j),
                    dist_mm=dist_mm,
                    angle_deg=angle,
                    nominal_dist_mm=nom_dist,
                    nominal_angle_deg=nom_angle,
                    dist_deviation_mm=d_dev,
                    angle_deviation_deg=a_dev,
                    status=status,
                ))
                log.debug(
                    "Baseline %s↔%s  dist=%.2f mm  angle=%.3f°  [%s]",
                    id_i, id_j, dist_mm, angle, status,
                )

        return results

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def run(
        self,
        world_points: Optional[np.ndarray] = None,
        observations: Optional[dict[str, np.ndarray]] = None,
        nominal_baselines: Optional[list[dict]] = None,
    ) -> BoresightReport:
        """
        Execute the full boresight check.

        Args:
            world_points:       [N, 3] world points (optional; skipped if None).
            observations:       Per-camera 2-D observed points (optional).
            nominal_baselines:  Nominal baseline list (optional).

        Returns:
            :class:`BoresightReport`.
        """
        # Reprojection
        if world_points is not None and observations is not None:
            cam_results = self.reprojection_rms(world_points, observations)
        else:
            cam_results = [
                CameraReprojResult(c.camera_id, float("nan"), 0, "NO_DATA")
                for c in self.cameras
            ]

        valid_rms = [r.rms_px for r in cam_results
                     if r.status not in ("NO_DATA",) and not np.isnan(r.rms_px)]
        overall_rms = float(np.sqrt(np.mean(np.array(valid_rms) ** 2))) \
            if valid_rms else float("nan")

        all_pass = all(
            r.status in ("PASS", "NO_DATA") for r in cam_results
        )

        # Baselines
        baseline_results = self.check_baselines(nominal_baselines)
        bl_pass = all(b.status in ("PASS", "N/A") for b in baseline_results)

        overall_status = "PASS" if (all_pass and bl_pass) else "FAIL"
        if np.isnan(overall_rms):
            overall_status = "NO_DATA"

        report = BoresightReport(
            overall_rms_px=overall_rms,
            overall_status=overall_status,
            per_camera=cam_results,
            baselines=baseline_results,
        )
        log.info(
            "Boresight check complete  overall_rms=%.4f px  [%s]",
            overall_rms, overall_status,
        )
        return report

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_report(
        report: BoresightReport,
        output_dir: Path | str,
        formats: list[str] | None = None,
    ) -> list[Path]:
        """Write the report to *output_dir* in the requested *formats*."""
        if formats is None:
            formats = ["json", "txt"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        if "json" in formats:
            p = output_dir / "boresight_report.json"
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(report.to_dict(), fh, indent=2)
            log.info("Boresight JSON → %s", p)
            written.append(p)

        if "txt" in formats:
            p = output_dir / "boresight_report.txt"
            p.write_text(report.to_text(), encoding="utf-8")
            log.info("Boresight TXT  → %s", p)
            written.append(p)

        return written
