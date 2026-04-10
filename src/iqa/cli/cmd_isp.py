"""``iqa isp`` sub-command — run the ISP pipeline on RAW images."""

from __future__ import annotations

from pathlib import Path

import click

from iqa.pipeline.pipeline import ISPPipeline
from iqa.utils.config_loader import PipelineConfig, load_isp_config
from iqa.utils.logger import get_logger

log = get_logger(__name__)


@click.command("isp")
@click.option(
    "--input", "-i", "input_path",
    required=True,
    help="Input RAW file or glob pattern (e.g. data/*.raw).",
)
@click.option(
    "--output", "-o", "output_dir",
    required=True,
    type=click.Path(),
    help="Output directory for processed images.",
)
@click.option(
    "--config", "-c", "config_path",
    default=None,
    type=click.Path(exists=True),
    help="ISP YAML config file (defaults to built-in default_isp.yaml).",
)
@click.option(
    "--bayer",
    default=None,
    type=click.Choice(["RGGB", "BGGR", "GRBG", "GBRG"]),
    help="Bayer pattern (overrides config).",
)
@click.option("--bit-depth", default=None, type=int,
              help="RAW bit depth (overrides config).")
@click.option("--height", default=None, type=int,
              help="RAW image height in pixels (required for .raw files).")
@click.option("--width", default=None, type=int,
              help="RAW image width in pixels (required for .raw files).")
@click.option("--debug", is_flag=True, default=False,
              help="Save per-stage intermediate images to output/debug/.")
@click.option("--format", "output_format",
              default="png", type=click.Choice(["png", "tiff"]),
              help="Output image format.")
def isp(
    input_path: str,
    output_dir: str,
    config_path: str | None,
    bayer: str | None,
    bit_depth: int | None,
    height: int | None,
    width: int | None,
    debug: bool,
    output_format: str,
) -> None:
    """Run the ISP pipeline (BLC → Demosaic → AWB → CCM) on RAW image(s)."""

    # --- Load config ---
    if config_path:
        cfg = load_isp_config(config_path)
    else:
        default_cfg = Path(__file__).parents[3] / "configs" / "default_isp.yaml"
        cfg = load_isp_config(default_cfg) if default_cfg.exists() else PipelineConfig()

    # CLI overrides
    if bayer:
        cfg.bayer_pattern = bayer
    if bit_depth:
        cfg.bit_depth = bit_depth
    if debug:
        cfg.debug_mode = True

    pipeline = ISPPipeline(cfg)

    # --- Resolve input files ---
    import glob as _glob
    files = sorted(Path(p) for p in _glob.glob(input_path))
    if not files:
        click.echo(f"[ERROR] No files matched: {input_path}", err=True)
        raise SystemExit(1)

    click.echo(f"Processing {len(files)} file(s) → {output_dir}")

    results = pipeline.process_batch(
        files,
        Path(output_dir),
        raw_height=height,
        raw_width=width,
        output_format=output_format,
    )

    ok = sum(1 for r in results if r.source_path is not None)
    click.echo(f"Done. {ok}/{len(files)} files processed successfully.")
