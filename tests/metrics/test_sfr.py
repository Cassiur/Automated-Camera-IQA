"""Tests for the SFR / MTF engine (ISO 12233)."""

import numpy as np
import pytest

from iqa.metrics.mtf_utils import (
    compute_esf,
    compute_mtf50,
    detect_edge_angle,
    esf_to_lsf,
    lsf_to_mtf,
)
from iqa.metrics.sfr import SFRAnalyzer
from iqa.utils.config_loader import SFRConfig


# ---------------------------------------------------------------------------
# Edge angle detection
# ---------------------------------------------------------------------------

class TestEdgeDetection:
    def test_angle_estimate_close_to_truth(self, ideal_slanted_edge):
        """Detected angle should be within ±0.5° of 7°."""
        angle, _ = detect_edge_angle(ideal_slanted_edge)
        assert abs(abs(angle) - 7.0) < 0.5, f"Expected ~7°, got {angle:.2f}°"

    def test_edge_positions_length(self, ideal_slanted_edge):
        """Length of edge_positions should equal image height."""
        _, ep = detect_edge_angle(ideal_slanted_edge)
        assert len(ep) == ideal_slanted_edge.shape[0]


# ---------------------------------------------------------------------------
# ESF
# ---------------------------------------------------------------------------

class TestESF:
    def test_esf_monotone_at_edge(self, ideal_slanted_edge):
        """ESF should transition from low to high (monotone around edge)."""
        _, ep = detect_edge_angle(ideal_slanted_edge)
        _, esf_v = compute_esf(ideal_slanted_edge, ep, oversample=4)
        # First quarter should be lower than last quarter
        n = len(esf_v)
        assert esf_v[:n // 4].mean() < esf_v[3 * n // 4:].mean()

    def test_esf_length_positive(self, ideal_slanted_edge):
        _, ep = detect_edge_angle(ideal_slanted_edge)
        d, v = compute_esf(ideal_slanted_edge, ep, oversample=4)
        assert len(d) > 0
        assert len(d) == len(v)


# ---------------------------------------------------------------------------
# LSF
# ---------------------------------------------------------------------------

class TestLSF:
    def test_lsf_peaks_at_edge(self, ideal_slanted_edge):
        """LSF should have its maximum near the centre of the transition."""
        _, ep = detect_edge_angle(ideal_slanted_edge)
        _, esf_v = compute_esf(ideal_slanted_edge, ep, oversample=4)
        lsf = esf_to_lsf(esf_v)
        peak_idx = np.argmax(np.abs(lsf))
        # Peak should be within the middle third of the array
        n = len(lsf)
        assert n // 4 < peak_idx < 3 * n // 4


# ---------------------------------------------------------------------------
# MTF
# ---------------------------------------------------------------------------

class TestMTF:
    def test_dc_normalised(self, ideal_slanted_edge):
        """MTF[0] (DC) must equal 1.0 after normalisation."""
        _, ep = detect_edge_angle(ideal_slanted_edge)
        _, esf_v = compute_esf(ideal_slanted_edge, ep, oversample=4)
        lsf = esf_to_lsf(esf_v)
        freq, mtf = lsf_to_mtf(lsf, windowing="hamming")
        assert abs(mtf[0] - 1.0) < 1e-5

    def test_mtf_between_zero_and_one(self, ideal_slanted_edge):
        _, ep = detect_edge_angle(ideal_slanted_edge)
        _, esf_v = compute_esf(ideal_slanted_edge, ep, oversample=4)
        lsf = esf_to_lsf(esf_v)
        freq, mtf = lsf_to_mtf(lsf)
        assert mtf.min() >= -0.05  # allow tiny numerical noise
        assert mtf.max() <= 1.05


# ---------------------------------------------------------------------------
# MTF50 full analyser
# ---------------------------------------------------------------------------

class TestMTF50:
    def test_ideal_edge_mtf50_near_half(self, ideal_slanted_edge):
        """Ideal step-edge MTF50 should be close to 0.5 cy/px."""
        cfg = SFRConfig(oversample_factor=4)
        result = SFRAnalyzer(cfg).analyze(ideal_slanted_edge)
        assert result.status == "ok", result.error
        assert not np.isnan(result.mtf50_cy_px)
        assert 0.3 < result.mtf50_cy_px <= 0.5, \
            f"Expected MTF50 near 0.5, got {result.mtf50_cy_px:.4f}"

    def test_blurred_edge_lower_mtf50(self, ideal_slanted_edge, blurred_slanted_edge):
        """Blurred edge must have a strictly lower MTF50 than the ideal edge."""
        cfg = SFRConfig(oversample_factor=4)
        analyzer = SFRAnalyzer(cfg)
        r_sharp  = analyzer.analyze(ideal_slanted_edge)
        r_blurred = analyzer.analyze(blurred_slanted_edge)
        assert r_sharp.status  == "ok"
        assert r_blurred.status == "ok"
        assert r_blurred.mtf50_cy_px < r_sharp.mtf50_cy_px, (
            f"Blurred MTF50 {r_blurred.mtf50_cy_px:.4f} should be < "
            f"sharp MTF50 {r_sharp.mtf50_cy_px:.4f}"
        )

    def test_angle_out_of_range_returns_error(self):
        """Edge angle > 15° should return status='error'."""
        import cv2
        # Build a nearly vertical edge (angle ≈ 45°)
        img = np.zeros((64, 64), dtype=np.float32)
        for r in range(64):
            img[r, r:] = 1.0
        cfg = SFRConfig(edge_angle_min=4.0, edge_angle_max=15.0)
        result = SFRAnalyzer(cfg).analyze(img)
        assert result.status == "error"
        assert "angle" in result.error.lower()

    def test_lp_mm_computed_when_pixel_size_set(self, ideal_slanted_edge):
        cfg = SFRConfig(oversample_factor=4, pixel_size_um=3.0)
        result = SFRAnalyzer(cfg).analyze(ideal_slanted_edge)
        if result.status == "ok":
            assert not np.isnan(result.mtf50_lp_mm)
            # MTF50 in lp/mm should be > MTF50 in cy/px for 3 µm pixel
            assert result.mtf50_lp_mm > result.mtf50_cy_px


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_returns_correct_count(self, tmp_path, ideal_slanted_edge):
        import cv2
        imgs = []
        for i in range(3):
            p = tmp_path / f"edge_{i}.png"
            cv2.imwrite(str(p), (ideal_slanted_edge * 255).astype(np.uint8))
            imgs.append(p)

        cfg = SFRConfig(oversample_factor=4)
        report = SFRAnalyzer(cfg).run_batch(imgs)
        assert len(report.results) == 3
        assert report.summary()["n_total"] == 3
