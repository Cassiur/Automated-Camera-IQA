"""Demosaic stage — Malvar-He-Cutler (2004) and fallback bilinear."""

from __future__ import annotations

import cv2
import numpy as np

from iqa.pipeline.base import BaseStage
from iqa.pipeline.blc import BAYER_OFFSETS
from iqa.utils.config_loader import DemosaicConfig

# ---------------------------------------------------------------------------
# Malvar-He-Cutler 5×5 convolution kernels
# Reference: "High-Quality Linear Interpolation for Demosaicing of
#             Bayer-Patterned Color Images", ICASSP 2004
# Kernel weights are multiplied by 8 in the original paper; here we store
# the actual floating-point values directly.
# ---------------------------------------------------------------------------

_K_G_AT_R_B = np.array([
    [ 0,  0, -1,  0,  0],
    [ 0,  0,  2,  0,  0],
    [-1,  2,  4,  2, -1],
    [ 0,  0,  2,  0,  0],
    [ 0,  0, -1,  0,  0],
], dtype=np.float32) / 8.0

_K_R_AT_G_RB = np.array([    # R at G in R-rows, B-columns
    [ 0,  0,  0.5,  0,  0],
    [ 0, -1,  0,   -1,  0],
    [-1,  4,  5,    4, -1],
    [ 0, -1,  0,   -1,  0],
    [ 0,  0,  0.5,  0,  0],
], dtype=np.float32) / 8.0

_K_R_AT_G_BR = np.array([    # R at G in B-rows, R-columns  (transpose of above)
    [ 0,  0, -1,  0,  0],
    [ 0, -1,  4, -1,  0],
    [0.5, 0,  5,  0, 0.5],
    [ 0, -1,  4, -1,  0],
    [ 0,  0, -1,  0,  0],
], dtype=np.float32) / 8.0

_K_R_AT_B = np.array([       # R at B
    [ 0,  0, -1.5, 0,  0],
    [ 0,  2,  0,   2,  0],
    [-1.5,0,  6,   0, -1.5],
    [ 0,  2,  0,   2,  0],
    [ 0,  0, -1.5, 0,  0],
], dtype=np.float32) / 8.0

# B kernels are symmetric to R kernels (swap R↔B)
_K_B_AT_G_BR = _K_R_AT_G_RB.copy()
_K_B_AT_G_RB = _K_R_AT_G_BR.copy()
_K_B_AT_R    = _K_R_AT_B.copy()


def _conv(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D convolution with reflection padding (border-safe)."""
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def _demosaic_malvar(bayer: np.ndarray, pattern: str) -> np.ndarray:
    """
    Malvar-He-Cutler interpolation.

    Args:
        bayer:   2-D float32 Bayer array [H, W].
        pattern: Bayer CFA pattern string (RGGB / BGGR / GRBG / GBRG).

    Returns:
        3-D float32 RGB array [H, W, 3].
    """
    offsets = BAYER_OFFSETS[pattern]
    r_row, r_col = offsets["R"]
    b_row, b_col = offsets["B"]
    # Gr / Gb offsets
    gr_row, gr_col = offsets["Gr"]
    gb_row, gb_col = offsets["Gb"]

    H, W = bayer.shape
    R = np.zeros((H, W), dtype=np.float32)
    G = np.zeros((H, W), dtype=np.float32)
    B = np.zeros((H, W), dtype=np.float32)

    # --- Fill known positions from raw Bayer data ---
    R[r_row::2, r_col::2]   = bayer[r_row::2, r_col::2]
    G[gr_row::2, gr_col::2] = bayer[gr_row::2, gr_col::2]
    G[gb_row::2, gb_col::2] = bayer[gb_row::2, gb_col::2]
    B[b_row::2, b_col::2]   = bayer[b_row::2, b_col::2]

    # --- G at R & B locations ---
    G_interp = _conv(bayer, _K_G_AT_R_B)
    G[r_row::2, r_col::2] = G_interp[r_row::2, r_col::2]
    G[b_row::2, b_col::2] = G_interp[b_row::2, b_col::2]

    # --- R at G in R-rows (Gr positions) ---
    R_at_G_Rrow = _conv(bayer, _K_R_AT_G_RB)
    R[gr_row::2, gr_col::2] = R_at_G_Rrow[gr_row::2, gr_col::2]

    # --- R at G in B-rows (Gb positions) ---
    R_at_G_Brow = _conv(bayer, _K_R_AT_G_BR)
    R[gb_row::2, gb_col::2] = R_at_G_Brow[gb_row::2, gb_col::2]

    # --- R at B ---
    R_at_B = _conv(bayer, _K_R_AT_B)
    R[b_row::2, b_col::2] = R_at_B[b_row::2, b_col::2]

    # --- B at G in B-rows (Gb positions) ---
    B_at_G_Brow = _conv(bayer, _K_B_AT_G_BR)
    B[gb_row::2, gb_col::2] = B_at_G_Brow[gb_row::2, gb_col::2]

    # --- B at G in R-rows (Gr positions) ---
    B_at_G_Rrow = _conv(bayer, _K_B_AT_G_RB)
    B[gr_row::2, gr_col::2] = B_at_G_Rrow[gr_row::2, gr_col::2]

    # --- B at R ---
    B_at_R = _conv(bayer, _K_B_AT_R)
    B[r_row::2, r_col::2] = B_at_R[r_row::2, r_col::2]

    return np.stack([R, G, B], axis=-1)


def _demosaic_bilinear(bayer: np.ndarray, pattern: str) -> np.ndarray:
    """Simple bilinear demosaic via OpenCV (fallback)."""
    cv_code = {
        "RGGB": cv2.COLOR_BayerRG2RGB,
        "BGGR": cv2.COLOR_BayerBG2RGB,
        "GRBG": cv2.COLOR_BayerGR2RGB,
        "GBRG": cv2.COLOR_BayerGB2RGB,
    }[pattern]
    u16 = np.clip(bayer, 0, 65535).astype(np.uint16)
    rgb = cv2.cvtColor(u16, cv_code)
    return rgb.astype(np.float32)


class Demosaic(BaseStage):
    """
    Bayer → RGB demosaic stage.

    Supports Malvar-He-Cutler (default, ~5 dB PSNR over bilinear) and a
    bilinear fallback via OpenCV.

    Args:
        config: :class:`~iqa.utils.config_loader.DemosaicConfig` instance.
    """

    name = "demosaic"

    def __init__(self, config: DemosaicConfig) -> None:
        self.config = config

    def validate_input(self, image: np.ndarray, meta: dict) -> None:
        super().validate_input(image, meta)
        if image.ndim != 2:
            raise ValueError(
                f"[demosaic] Expected 2-D Bayer image, got shape {image.shape}"
            )

    def process(
        self, image: np.ndarray, meta: dict
    ) -> tuple[np.ndarray, dict]:
        """
        Demosaic a 2-D Bayer image into 3-channel RGB.

        Args:
            image: 2-D float32 Bayer array [H, W].
            meta:  Must contain ``bayer_pattern``.

        Returns:
            3-D float32 RGB array [H, W, 3] and updated meta.
        """
        if not self.config.enabled:
            self._log.debug("Demosaic disabled — skipping.")
            # Return 3-ch duplicate so downstream stages see consistent shape
            return np.stack([image, image, image], axis=-1), meta

        pattern = meta.get("bayer_pattern", "RGGB").upper()
        if pattern not in BAYER_OFFSETS:
            raise ValueError(f"Unknown bayer_pattern '{pattern}'")

        algo = self.config.algorithm.lower()
        if algo == "malvar":
            rgb = _demosaic_malvar(image, pattern)
        elif algo == "bilinear":
            rgb = _demosaic_bilinear(image, pattern)
        else:
            raise ValueError(f"Unknown demosaic algorithm '{algo}'")

        bit_depth = meta.get("bit_depth", 12)
        max_val = float(2 ** bit_depth - 1)
        np.clip(rgb, 0.0, max_val, out=rgb)

        meta = {**meta, "demosaiced": True, "demosaic_algo": algo}
        self._log.debug(
            "Demosaic done  algo=%s  pattern=%s  out_shape=%s",
            algo, pattern, rgb.shape,
        )
        return rgb, meta
