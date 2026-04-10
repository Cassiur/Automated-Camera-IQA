"""ISP Pipeline orchestrator: BLC → Demosaic → AWB → CCM."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from iqa.pipeline.awb import AutoWhiteBalance
from iqa.pipeline.base import BaseStage
from iqa.pipeline.blc import BlackLevelCorrection
from iqa.pipeline.ccm import ColourCorrectionMatrix
from iqa.pipeline.demosaic import Demosaic
from iqa.utils.config_loader import PipelineConfig, load_isp_config
from iqa.utils.image_io import load_image, save_image
from iqa.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PipelineResult:
    """Output container from a single ISP pipeline run."""

    image: np.ndarray
    """Final processed RGB image [H, W, 3] float32."""

    meta: dict
    """Pipeline context / provenance metadata."""

    intermediates: dict = field(default_factory=dict)
    """stage_name → intermediate image (populated in debug mode)."""

    elapsed_ms: float = 0.0
    """Wall-clock processing time in milliseconds."""

    source_path: Optional[Path] = None
    """Input file path, if processed from disk."""


class ISPPipeline:
    """
    Full ISP processing chain: BLC → Demosaic → AWB → CCM.

    The pipeline is constructed from a :class:`~iqa.utils.config_loader.PipelineConfig`
    (loaded from YAML or instantiated programmatically) and exposes two main
    entry points:

    * :meth:`run` — process a single NumPy array.
    * :meth:`process_batch` — process a list of files and write results to disk.

    Example::

        cfg = load_isp_config("configs/default_isp.yaml")
        isp = ISPPipeline(cfg)
        result = isp.run(raw_array, meta={"bayer_pattern": "RGGB", "bit_depth": 12})

    Args:
        config: Parsed :class:`PipelineConfig`.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.stages: list[BaseStage] = self._build_stages()

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ISPPipeline":
        """Instantiate a pipeline directly from a YAML config file."""
        cfg = load_isp_config(path)
        return cls(cfg)

    def _build_stages(self) -> list[BaseStage]:
        cfg = self.config
        return [
            BlackLevelCorrection(cfg.blc),
            Demosaic(cfg.demosaic),
            AutoWhiteBalance(cfg.awb),
            ColourCorrectionMatrix(cfg.ccm),
        ]

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def run(
        self,
        raw: np.ndarray,
        meta: Optional[dict] = None,
        source_path: Optional[Path] = None,
    ) -> PipelineResult:
        """
        Run the full ISP chain on a single raw image.

        Args:
            raw:         Input RAW array (will be cast to float32).
            meta:        Pipeline context.  Defaults inject ``bayer_pattern``
                         and ``bit_depth`` from the config if absent.
            source_path: Optional provenance path stored in the result.

        Returns:
            :class:`PipelineResult` with processed image and metadata.
        """
        if meta is None:
            meta = {}
        meta.setdefault("bayer_pattern", self.config.bayer_pattern)
        meta.setdefault("bit_depth", self.config.bit_depth)

        image = raw.astype(np.float32)
        intermediates: dict[str, np.ndarray] = {}
        debug = self.config.debug_mode

        t0 = time.perf_counter()
        for stage in self.stages:
            image, meta = stage(image, meta)
            if debug:
                intermediates[stage.name] = image.copy()
            log.debug("Stage [%s] done  shape=%s", stage.name, image.shape)
        elapsed = (time.perf_counter() - t0) * 1000.0

        log.info(
            "ISP done  stages=%d  elapsed=%.1f ms  out_shape=%s",
            len(self.stages), elapsed, image.shape,
        )
        return PipelineResult(
            image=image,
            meta=meta,
            intermediates=intermediates,
            elapsed_ms=elapsed,
            source_path=source_path,
        )

    def process_batch(
        self,
        file_list: list[Path | str],
        output_dir: Path | str,
        *,
        raw_height: Optional[int] = None,
        raw_width: Optional[int] = None,
        output_format: str = "png",
    ) -> list[PipelineResult]:
        """
        Process multiple files and write results to *output_dir*.

        Each output file keeps the source stem with an ``_isp`` suffix.
        Intermediate images (debug mode) are written to a ``debug/`` sub-folder.

        Args:
            file_list:    List of input file paths.
            output_dir:   Destination directory (created if absent).
            raw_height:   Required for ``.raw`` / ``.bin`` files.
            raw_width:    Required for ``.raw`` / ``.bin`` files.
            output_format: ``"png"`` or ``"tiff"``.

        Returns:
            List of :class:`PipelineResult` (one per input file).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.debug_mode:
            (output_dir / "debug").mkdir(exist_ok=True)

        results: list[PipelineResult] = []
        for fpath in file_list:
            fpath = Path(fpath)
            try:
                raw = load_image(fpath, height=raw_height, width=raw_width)
                result = self.run(raw, source_path=fpath)

                # Normalise to [0,1] for saving
                norm = result.image
                if norm.max() > 1.0 + 1e-6:
                    norm = norm / float(2 ** self.config.bit_depth - 1)

                out_path = output_dir / f"{fpath.stem}_isp.{output_format}"
                save_image(
                    out_path,
                    norm,
                    normalize=True,
                    bit_depth=self.config.output_bit_depth,
                )
                log.info("Saved → %s", out_path)

                if self.config.debug_mode:
                    for stage_name, intermed in result.intermediates.items():
                        dbg_path = output_dir / "debug" / f"{fpath.stem}_{stage_name}.{output_format}"
                        dbg_norm = intermed
                        if intermed.max() > 1.0 + 1e-6:
                            dbg_norm = intermed / float(2 ** self.config.bit_depth - 1)
                        save_image(dbg_path, dbg_norm, normalize=True,
                                   bit_depth=self.config.output_bit_depth)

                results.append(result)

            except Exception as exc:
                log.error("Failed to process %s: %s", fpath, exc)

        return results
