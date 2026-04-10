"""Black Level Correction (BLC) stage."""

from __future__ import annotations

import numpy as np

from iqa.pipeline.base import BaseStage
from iqa.utils.config_loader import BLCConfig

# Bayer pattern → (row_offset, col_offset) for each channel
BAYER_OFFSETS: dict[str, dict[str, tuple[int, int]]] = {
    "RGGB": {"R": (0, 0), "Gr": (0, 1), "Gb": (1, 0), "B": (1, 1)},
    "BGGR": {"B": (0, 0), "Gb": (0, 1), "Gr": (1, 0), "R": (1, 1)},
    "GRBG": {"Gr": (0, 0), "R": (0, 1), "B": (1, 0), "Gb": (1, 1)},
    "GBRG": {"Gb": (0, 0), "B": (0, 1), "R": (1, 0), "Gr": (1, 1)},
}


class BlackLevelCorrection(BaseStage):
    """
    Per-channel Black Level Correction for Bayer RAW images.

    Each of the four Bayer sub-channels (R, Gr, Gb, B) has its own black
    pedestal subtracted, and the result is clipped to [0, 2**bit_depth - 1].

    Args:
        config: :class:`~iqa.utils.config_loader.BLCConfig` instance.
    """

    name = "blc"

    def __init__(self, config: BLCConfig) -> None:
        self.config = config

    def process(
        self, image: np.ndarray, meta: dict
    ) -> tuple[np.ndarray, dict]:
        """
        Apply black-level subtraction.

        Args:
            image: 2-D float32 Bayer array [H, W].
            meta:  Must contain ``bayer_pattern`` (str) and ``bit_depth`` (int).

        Returns:
            BLC-corrected image and updated meta.
        """
        if not self.config.enabled:
            self._log.debug("BLC disabled — skipping.")
            return image, meta

        pattern = meta.get("bayer_pattern", "RGGB").upper()
        bit_depth = meta.get("bit_depth", 12)
        max_val = float(2 ** bit_depth - 1)

        if pattern not in BAYER_OFFSETS:
            raise ValueError(
                f"Unknown bayer_pattern '{pattern}'. "
                f"Expected one of {list(BAYER_OFFSETS)}"
            )

        offsets = BAYER_OFFSETS[pattern]
        bl = self.config.black_levels

        # Normalise scalar BL to per-channel dict
        if isinstance(bl, (int, float)):
            bl = {ch: int(bl) for ch in ("R", "Gr", "Gb", "B")}

        out = image.copy()
        for channel, (row, col) in offsets.items():
            level = float(bl.get(channel, 0))
            out[row::2, col::2] -= level

        np.clip(out, 0.0, max_val, out=out)

        meta = {**meta, "blc_applied": True, "black_levels": dict(bl)}
        self._log.debug(
            "BLC applied  pattern=%s  bit_depth=%d  BL=%s",
            pattern, bit_depth, bl,
        )
        return out, meta
