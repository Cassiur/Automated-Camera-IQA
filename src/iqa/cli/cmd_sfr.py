"""``iqa sfr`` sub-command — ISO 12233 SFR/MTF batch evaluation."""

from __future__ import annotations

import glob as _glob
from pathlib import Path
from typing import Optional

import click

from iqa.metrics.report import save_report
from iqa.metrics.sfr import SFRAnalyzer
from iqa.utils.config_loader import SFRConfig, load_sfr_config
from iqa.utils.logger import get_logger

log = get_logger(__name__)


def _parse_roi(roi_str: str) -> tuple[int, int, int, int]:
    parts = [int(x.strip()) for x in roi_str.split(",")]
    if len(parts) != 4:
        raise click.BadParameter("ROI must be 'x1,y1,x2,y2'")
    return tuple(parts)  # type: ignore[return-value]


@click.command("sfr")
@click.option(
    "--input", "-i", "input_glob",
    required=True,
    help="Input image file or glob pattern (e.g. 'data/*.png').",
)
@click.option(
    "--output", "-o", "output_dir",
    required=True,
    type=click.Path(),
    help="Directory to save JSON/CSV reports and optional plots.",
)
@click.option("--config", "-c", "config_path",
              default=None, type=click.Path(exists=True),
              help="SFR YAML config file.")
@click.option("--roi", "roi_str", default=None,
              help="ROI crop: 'x1,y1,x2,y2' (overrides config).")
@click.option("--oversample", default=None, type=int,
              help="ESF super-sampling factor (overrides config).")
@click.option("--pixel-size", "pixel_size_um", default=None, type=float,
              help="Pixel physical size in μm for lp/mm conversion.")
@click.option("--format", "output_format",
              default="both", type=click.Choice(["json", "csv", "both"]),
              help="Report output format.")
@click.option("--plot", is_flag=True, default=False,
              help="Generate MTF curve PNG plots.")
def sfr(
    input_glob: str,
    output_dir: str,
    config_path: str | None,
    roi_str: str | None,
    oversample: int | None,
    pixel_size_um: float | None,
    output_format: str,
    plot: bool,
) -> None:
    """Evaluate SFR/MTF50 using the ISO 12233 slanted-edge method."""

    # --- Config ---
    if config_path:
        cfg = load_sfr_config(config_path)
    else:
        cfg = SFRConfig()

    if oversample:
        cfg.oversample_factor = oversample
    if pixel_size_um:
        cfg.pixel_size_um = pixel_size_um

    # --- Files ---
    files = sorted(Path(p) for p in _glob.glob(input_glob))
    if not files:
        click.echo(f"[ERROR] No files matched: {input_glob}", err=True)
        raise SystemExit(1)

    roi_box: Optional[tuple[int, int, int, int]] = None
    if roi_str:
        roi_box = _parse_roi(roi_str)

    click.echo(f"Analysing {len(files)} image(s)…")
    analyzer = SFRAnalyzer(cfg)
    report = analyzer.run_batch(files, roi_box=roi_box)

    # --- Reports ---
    fmts = ["json", "csv"] if output_format == "both" else [output_format]
    save_report(report, output_dir, formats=fmts)

    # --- Optional plots ---
    if plot:
        _save_plots(report, output_dir)

    s = report.summary()
    click.echo(
        f"Done.  ok={s['n_ok']}/{s['n_total']}  "
        f"MTF50 mean={s['mtf50_mean_cy_px']:.4f} cy/px"
    )


def _save_plots(report, output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        click.echo("[WARN] matplotlib not installed — skipping plots.")
        return

    out = Path(output_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)

    for r in report.passed:
        if not r.freq_axis:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(r.freq_axis, r.mtf_curve, linewidth=1.5, label=r.label)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="MTF50")
        ax.set_xlabel("Frequency (cy/px)")
        ax.set_ylabel("MTF")
        ax.set_title(f"MTF — {r.label}  (MTF50={r.mtf50_cy_px:.4f})")
        ax.legend()
        ax.set_xlim(0, max(r.freq_axis) if r.freq_axis else 0.5)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out / f"{r.label}_mtf.png"
        fig.savefig(str(p), dpi=150)
        plt.close(fig)
