"""Tests for SFR report generation (JSON / CSV)."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from iqa.metrics.report import save_csv, save_json, save_report
from iqa.metrics.sfr import BatchSFRReport, SFRResult


def _make_report():
    r1 = SFRResult(
        path=Path("img1.png"), label="center",
        status="ok", mtf50_cy_px=0.42, mtf50_lp_mm=140.0,
        edge_angle_deg=7.1, freq_axis=[0.0, 0.1, 0.2], mtf_curve=[1.0, 0.8, 0.5],
    )
    r2 = SFRResult(
        path=Path("img2.png"), label="corner",
        status="error", error="angle out of range",
    )
    return BatchSFRReport(results=[r1, r2])


def test_json_report_valid(tmp_path):
    report = _make_report()
    p = save_json(report, tmp_path)
    assert p.exists()
    with open(p) as f:
        data = json.load(f)
    assert "summary" in data
    assert "results" in data
    assert len(data["results"]) == 2


def test_json_summary_counts(tmp_path):
    report = _make_report()
    p = save_json(report, tmp_path)
    with open(p) as f:
        data = json.load(f)
    s = data["summary"]
    assert s["n_total"] == 2
    assert s["n_ok"] == 1
    assert s["n_error"] == 1


def test_csv_columns(tmp_path):
    report = _make_report()
    p = save_csv(report, tmp_path)
    assert p.exists()
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert "mtf50_cy_px" in rows[0]
    assert "status" in rows[0]


def test_save_report_both_formats(tmp_path):
    report = _make_report()
    paths = save_report(report, tmp_path, formats=["json", "csv"])
    assert len(paths) == 2
    exts = {p.suffix for p in paths}
    assert ".json" in exts
    assert ".csv" in exts


def test_nan_values_serialisable(tmp_path):
    """JSON should not contain NaN literals (replaced with null)."""
    r = SFRResult(status="ok", mtf50_cy_px=float("nan"))
    report = BatchSFRReport(results=[r])
    p = save_json(report, tmp_path)
    text = p.read_text()
    assert "NaN" not in text
    assert "Infinity" not in text
