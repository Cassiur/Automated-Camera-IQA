"""``iqa boresight`` sub-command — multi-camera alignment checker."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from iqa.calibration.boresight import BoresightChecker
from iqa.calibration.extrinsic import CameraExtrinsic, load_extrinsic_npz
from iqa.utils.config_loader import load_yaml
from iqa.utils.logger import get_logger

log = get_logger(__name__)


@click.command("boresight")
@click.option(
    "--config", "-c", "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Boresight YAML config file.",
)
@click.option(
    "--output", "-o", "output_dir",
    required=True,
    type=click.Path(),
    help="Output directory for the alignment report.",
)
@click.option("--format", "output_format",
              default="both", type=click.Choice(["json", "txt", "both"]),
              help="Report format.")
@click.option("--threshold-dist", default=None, type=float,
              help="Baseline distance deviation threshold in mm (overrides config).")
@click.option("--threshold-angle", default=None, type=float,
              help="Baseline rotation deviation threshold in degrees (overrides config).")
def boresight(
    config_path: str,
    output_dir: str,
    output_format: str,
    threshold_dist: float | None,
    threshold_angle: float | None,
) -> None:
    """Validate multi-camera boresight alignment from extrinsic calibration data."""

    cfg = load_yaml(config_path)

    # --- Thresholds ---
    thr = cfg.get("thresholds", {})
    rms_thr   = float(thr.get("reprojection_rms_px", 0.5))
    dist_thr  = float(threshold_dist  or thr.get("baseline_dist_mm",   1.0))
    angle_thr = float(threshold_angle or thr.get("baseline_angle_deg", 0.1))

    # --- Load cameras ---
    cameras: list[CameraExtrinsic] = []
    for cam_cfg in cfg.get("cameras", []):
        cid = cam_cfg["camera_id"] if "camera_id" in cam_cfg else cam_cfg["id"]
        ext_path = Path(cam_cfg["extrinsic_file"])
        intr_path = cam_cfg.get("intrinsic_file")
        if not ext_path.exists():
            click.echo(f"[WARN] Extrinsic file not found: {ext_path} — skipping.")
            continue
        cam = load_extrinsic_npz(
            cid, ext_path,
            intrinsic_path=Path(intr_path) if intr_path else None,
        )
        cameras.append(cam)
        log.info("Loaded camera '%s'", cid)

    if not cameras:
        click.echo("[ERROR] No camera data loaded.", err=True)
        raise SystemExit(1)

    # --- Load world points & observations (optional) ---
    wp_path  = cfg.get("world_points_file")
    obs_path = cfg.get("observations_file")

    world_points = None
    observations = None

    if wp_path and Path(wp_path).exists():
        world_points = np.load(wp_path).astype(np.float64)
        log.info("World points: %s  shape=%s", wp_path, world_points.shape)

    if obs_path and Path(obs_path).exists():
        npz = np.load(obs_path)
        observations = {k: npz[k].astype(np.float64) for k in npz.files}
        log.info("Observations loaded for cameras: %s", list(observations.keys()))

    # --- Run check ---
    checker = BoresightChecker(
        cameras,
        rms_threshold=rms_thr,
        dist_threshold=dist_thr,
        angle_threshold=angle_thr,
    )
    report = checker.run(
        world_points=world_points,
        observations=observations,
        nominal_baselines=cfg.get("nominal_baselines"),
    )

    # --- Save ---
    fmts = ["json", "txt"] if output_format == "both" else [output_format]
    BoresightChecker.save_report(report, output_dir, formats=fmts)

    # Print summary to stdout
    click.echo(report.to_text())
