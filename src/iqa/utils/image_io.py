"""Image I/O utilities supporting RAW, PNG, TIFF (8/16-bit)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from iqa.utils.logger import get_logger

log = get_logger(__name__)

# Supported extensions for colour images
_COLOR_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# Raw extensions treated as packed uint16 binary blobs
_RAW_EXT = {".raw", ".bin"}


def load_image(
    path: str | Path,
    *,
    grayscale: bool = False,
    height: int | None = None,
    width: int | None = None,
    dtype: str = "float32",
) -> np.ndarray:
    """
    Load an image from *path* and return a float32 ndarray.

    For .raw/.bin files supply *height* and *width*; the file is interpreted
    as a packed uint16 array in row-major order (suitable for Bayer RAW).

    Args:
        path:       File path.
        grayscale:  If True, convert to single-channel.
        height:     Required for RAW files.
        width:      Required for RAW files.
        dtype:      Output NumPy dtype string (default "float32").

    Returns:
        ndarray, shape [H, W] (RAW / grayscale) or [H, W, 3] (colour).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    ext = path.suffix.lower()

    if ext in _RAW_EXT:
        if height is None or width is None:
            raise ValueError("height and width are required for RAW files.")
        data = np.fromfile(path, dtype=np.uint16).reshape(height, width)
        log.debug("Loaded RAW %s  shape=%s", path.name, data.shape)
        return data.astype(dtype)

    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
    img = cv2.imread(str(path), flags)
    if img is None:
        raise IOError(f"cv2.imread failed for: {path}")

    # Convert BGR → RGB for colour images
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    log.debug("Loaded image %s  shape=%s  dtype=%s", path.name, img.shape, img.dtype)
    return img.astype(dtype)


def save_image(
    path: str | Path,
    image: np.ndarray,
    *,
    normalize: bool = True,
    bit_depth: int = 8,
) -> None:
    """
    Save *image* to *path*.

    Args:
        path:       Destination file path (.png or .tiff).
        image:      Float32 array [H,W] or [H,W,3], values in [0, 1] or raw ADU.
        normalize:  If True, scale to full output bit-depth range.
        bit_depth:  8 or 16 for PNG; 16 for TIFF.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = image.copy().astype(np.float32)

    if normalize:
        max_val = float(2 ** bit_depth - 1)
        # If image already in [0,1], scale; otherwise assume already in ADU range
        if img.max() <= 1.0 + 1e-6:
            img = img * max_val
        img = np.clip(img, 0, max_val)

    if bit_depth <= 8:
        out = img.astype(np.uint8)
    else:
        out = img.astype(np.uint16)

    # Convert RGB → BGR for OpenCV
    if out.ndim == 3 and out.shape[2] == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    ok = cv2.imwrite(str(path), out)
    if not ok:
        raise IOError(f"cv2.imwrite failed for: {path}")
    log.debug("Saved image → %s  shape=%s", path, out.shape)
