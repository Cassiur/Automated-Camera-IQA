"""Auto White Balance (AWB) stage — Gray-World and Perfect Reflector."""

from __future__ import annotations

import numpy as np

from iqa.pipeline.base import BaseStage
from iqa.utils.config_loader import AWBConfig


class AutoWhiteBalance(BaseStage):
    """
    AWB stage supporting three modes:

    * **gray_world** — assumes scene average ≈ neutral grey.
    * **perfect_reflector** — uses the brightest *top-N%* pixels as white reference.
    * **manual** — applies user-specified R/G/B gains directly.

    Args:
        config: :class:`~iqa.utils.config_loader.AWBConfig` instance.
    """

    name = "awb"

    def __init__(self, config: AWBConfig) -> None:
        self.config = config

    def validate_input(self, image: np.ndarray, meta: dict) -> None:
        super().validate_input(image, meta)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"[awb] Expected 3-channel image [H,W,3], got shape {image.shape}. "
                "Ensure Demosaic runs before AWB."
            )

    def process(
        self, image: np.ndarray, meta: dict
    ) -> tuple[np.ndarray, dict]:
        """
        Apply white balance to a float32 RGB image.

        The image is expected to have been demosaiced and normalised to the
        sensor bit-depth range (or [0, 1]).  Gains are written back into *meta*.

        Args:
            image: Float32 RGB array [H, W, 3].
            meta:  Pipeline context.

        Returns:
            White-balanced image and updated meta.
        """
        if not self.config.enabled:
            self._log.debug("AWB disabled — skipping.")
            return image, meta

        method = self.config.method.lower()

        if method == "gray_world":
            gains = self._gray_world(image)
        elif method == "perfect_reflector":
            gains = self._perfect_reflector(image)
        elif method == "manual":
            mg = self.config.manual_gains
            gains = np.array([mg.get("R", 1.0), mg.get("G", 1.0), mg.get("B", 1.0)],
                             dtype=np.float32)
        else:
            raise ValueError(f"Unknown AWB method '{method}'")

        out = image * gains[np.newaxis, np.newaxis, :]

        bit_depth = meta.get("bit_depth", 12)
        max_val = float(2 ** bit_depth - 1) if meta.get("demosaiced") else 1.0
        # Tolerate images already in [0,1]
        if image.max() <= 1.0 + 1e-6:
            max_val = 1.0
        np.clip(out, 0.0, max_val, out=out)

        meta = {**meta, "awb_gains": gains.tolist(), "awb_method": method}
        self._log.debug("AWB  method=%s  gains=R%.3f G%.3f B%.3f",
                        method, *gains)
        return out, meta

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _saturation_mask(self, image: np.ndarray) -> np.ndarray:
        """Boolean mask True where ALL channels are below saturation threshold."""
        thresh = self.config.saturation_threshold
        # Use relative scale: compare to max observed value
        max_obs = image.max()
        if max_obs <= 0:
            return np.ones(image.shape[:2], dtype=bool)
        return (image / max_obs).max(axis=-1) < thresh

    def _gray_world(self, image: np.ndarray) -> np.ndarray:
        mask = self._saturation_mask(image)
        r_mean = image[mask, 0].mean()
        g_mean = image[mask, 1].mean()
        b_mean = image[mask, 2].mean()

        ref = g_mean if g_mean > 0 else 1.0
        r_gain = ref / r_mean if r_mean > 0 else 1.0
        b_gain = ref / b_mean if b_mean > 0 else 1.0
        return np.array([r_gain, 1.0, b_gain], dtype=np.float32)

    def _perfect_reflector(self, image: np.ndarray) -> np.ndarray:
        top_pct = self.config.perfect_reflector_percentile / 100.0
        # Luminance approximation
        lum = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        threshold = np.percentile(lum, (1.0 - top_pct) * 100.0)
        bright_mask = lum >= threshold

        r_mean = image[bright_mask, 0].mean()
        g_mean = image[bright_mask, 1].mean()
        b_mean = image[bright_mask, 2].mean()

        ref = g_mean if g_mean > 0 else 1.0
        r_gain = ref / r_mean if r_mean > 0 else 1.0
        b_gain = ref / b_mean if b_mean > 0 else 1.0
        return np.array([r_gain, 1.0, b_gain], dtype=np.float32)
