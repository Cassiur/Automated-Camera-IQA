"""Tests for Boresight alignment checker."""

import numpy as np
import pytest

from iqa.calibration.boresight import BoresightChecker, BoresightReport
from iqa.calibration.extrinsic import (
    CameraExtrinsic,
    make_transform,
    relative_transform,
    rotation_angle_deg,
)


# ---------------------------------------------------------------------------
# Extrinsic utilities
# ---------------------------------------------------------------------------

class TestExtrinsicUtils:
    def test_make_transform_shape(self, identity_cameras):
        T = make_transform(identity_cameras[0].R, identity_cameras[0].t)
        assert T.shape == (4, 4)
        assert T[3, 3] == 1.0

    def test_identity_rotation_angle_zero(self):
        angle = rotation_angle_deg(np.eye(3))
        assert abs(angle) < 1e-6

    def test_rotation_angle_90(self):
        R90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
        angle = rotation_angle_deg(R90)
        assert abs(angle - 90.0) < 0.01

    def test_relative_transform_identity(self, identity_cameras):
        cam = identity_cameras[0]
        T = make_transform(cam.R, cam.t)
        T_rel = relative_transform(T, T)
        assert np.allclose(T_rel, np.eye(4), atol=1e-9)


# ---------------------------------------------------------------------------
# Reprojection error — perfect calibration
# ---------------------------------------------------------------------------

class TestReprojection:
    def test_zero_error_perfect_calibration(self, identity_cameras, world_points_grid):
        """With exact extrinsics, reprojection error should be ≈ 0."""
        # Build observations by projecting world points through each camera
        observations = {}
        for cam in identity_cameras:
            pts = []
            for P in world_points_grid:
                P_c = cam.R @ P.reshape(3, 1) + cam.t
                if P_c[2, 0] <= 0:
                    pts.append([0.0, 0.0])
                    continue
                p = cam.K @ P_c
                pts.append((p[:2] / p[2]).ravel().tolist())
            observations[cam.camera_id] = np.array(pts, dtype=np.float64)

        checker = BoresightChecker(identity_cameras, rms_threshold=0.01)
        results = checker.reprojection_rms(world_points_grid, observations)

        for r in results:
            assert r.status in ("PASS", "NO_DATA"), \
                f"Camera {r.camera_id} failed: rms={r.rms_px:.4f}"
            if r.status == "PASS":
                assert r.rms_px < 0.01

    def test_no_data_camera_status(self, identity_cameras, world_points_grid):
        """Camera with no observations should get status NO_DATA."""
        checker = BoresightChecker(identity_cameras)
        results = checker.reprojection_rms(world_points_grid, observations={})
        assert all(r.status == "NO_DATA" for r in results)


# ---------------------------------------------------------------------------
# Baseline check
# ---------------------------------------------------------------------------

class TestBaselines:
    def test_correct_baseline_passes(self, identity_cameras):
        checker = BoresightChecker(identity_cameras, dist_threshold=5.0, angle_threshold=1.0)
        nominal = [
            {"pair": ["front", "left"],  "distance_mm": 300.0,  "angle_deg": 90.0},
            {"pair": ["front", "rear"],  "distance_mm": 1500.0, "angle_deg": 180.0},
        ]
        results = checker.check_baselines(nominal)
        for b in results:
            if b.status != "N/A":
                assert b.status == "PASS", \
                    f"Baseline {b.pair} FAILED: dist={b.dist_mm:.2f}, angle={b.angle_deg:.2f}"

    def test_no_nominal_gives_na(self, identity_cameras):
        checker = BoresightChecker(identity_cameras)
        results = checker.check_baselines(nominal=None)
        assert all(b.status == "N/A" for b in results)


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

class TestBoresightRun:
    def test_run_returns_report(self, identity_cameras):
        checker = BoresightChecker(identity_cameras)
        report = checker.run()
        assert isinstance(report, BoresightReport)
        assert len(report.per_camera) == len(identity_cameras)

    def test_save_json(self, identity_cameras, tmp_path):
        checker = BoresightChecker(identity_cameras)
        report = checker.run()
        paths = BoresightChecker.save_report(report, tmp_path, formats=["json"])
        assert len(paths) == 1
        assert paths[0].exists()

    def test_save_txt(self, identity_cameras, tmp_path):
        checker = BoresightChecker(identity_cameras)
        report = checker.run()
        paths = BoresightChecker.save_report(report, tmp_path, formats=["txt"])
        assert len(paths) == 1
        text = paths[0].read_text()
        assert "Boresight" in text
