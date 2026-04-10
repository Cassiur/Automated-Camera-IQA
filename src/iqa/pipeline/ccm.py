"""Colour Correction Matrix (CCM) stage — 3×3 linear transform."""

from __future__ import annotations

import numpy as np

from iqa.pipeline.base import BaseStage
from iqa.utils.config_loader import CCMConfig
from iqa.utils.logger import get_logger

_log = get_logger(__name__)


class ColourCorrectionMatrix(BaseStage):
    """
    Apply a 3×3 colour-correction matrix to a demosaiced RGB image.

    The transform is:  out[i,j] = CCM @ in[i,j]  (per-pixel, row vector form:
    ``pixels @ CCM.T``).

    After the transform, values are optionally clipped to [0, max_val].

    Args:
        config: :class:`~iqa.utils.config_loader.CCMConfig` instance.
    """

    name = "ccm"

    def __init__(self, config: CCMConfig) -> None:
        self.config = config
        self._matrix = np.array(config.matrix, dtype=np.float32)
        self._validate_matrix()

    def _validate_matrix(self) -> None:
        if self._matrix.shape != (3, 3):
            raise ValueError(
                f"CCM matrix must be shape (3, 3), got {self._matrix.shape}"
            )
        row_sums = self._matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=0.15):
            _log.warning(
                "CCM row sums %s deviate from 1.0 — "
                "check matrix for correctness.",
                row_sums.tolist(),
            )

    def validate_input(self, image: np.ndarray, meta: dict) -> None:
        super().validate_input(image, meta)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"[ccm] Expected 3-channel image [H,W,3], got shape {image.shape}"
            )

    def process(
        self, image: np.ndarray, meta: dict
    ) -> tuple[np.ndarray, dict]:
        """
        Apply the CCM.

        The image is treated as floating-point in whatever range it is in
        (ADU or [0, 1]).  The matrix is applied in-place on a copy, then
        clipped if ``config.clip`` is True.

        Args:
            image: Float32 RGB array [H, W, 3].
            meta:  Pipeline context.

        Returns:
            CCM-corrected image and updated meta.
        """
        if not self.config.enabled:
            self._log.debug("CCM disabled — skipping.")
            return image, meta

        H, W, _ = image.shape
        flat = image.reshape(-1, 3)          # [H*W, 3]
        out_flat = flat @ self._matrix.T     # [H*W, 3]
        out = out_flat.reshape(H, W, 3)

        if self.config.clip:
            # Determine clipping range from image statistics
            max_val = 1.0 if image.max() <= 1.0 + 1e-6 else float(image.max())
            np.clip(out, 0.0, max_val, out=out)

        meta = {**meta, "ccm_applied": True, "ccm_matrix": self._matrix.tolist()}
        self._log.debug("CCM applied  shape=%s", out.shape)
        return out, meta
