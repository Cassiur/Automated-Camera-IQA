"""Extrinsic matrix utilities: loading, composing, and decomposing transforms."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np


class CameraExtrinsic(NamedTuple):
    """Per-camera extrinsic data container."""

    camera_id: str
    R: np.ndarray    # [3, 3] rotation matrix
    t: np.ndarray    # [3, 1] translation vector

    K: np.ndarray = np.eye(3, dtype=np.float64)
    """Intrinsic matrix [3, 3] (default identity)."""

    dist: np.ndarray = np.zeros(5, dtype=np.float64)
    """Distortion coefficients (default zeros)."""


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Build a 4×4 homogeneous transformation matrix from R and t.

    Args:
        R: 3×3 rotation matrix.
        t: 3-element or (3,1) translation vector.

    Returns:
        4×4 float64 transformation matrix.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.astype(np.float64)
    T[:3, 3]  = np.asarray(t, dtype=np.float64).ravel()
    return T


def relative_transform(T_i: np.ndarray, T_j: np.ndarray) -> np.ndarray:
    """
    Compute T_ij = T_j @ inv(T_i) — relative transform from camera i to camera j.

    Args:
        T_i: 4×4 transform of camera i.
        T_j: 4×4 transform of camera j.

    Returns:
        4×4 relative transform.
    """
    return T_j @ np.linalg.inv(T_i)


def rotation_angle_deg(R: np.ndarray) -> float:
    """
    Compute the rotation angle (in degrees) from a 3×3 rotation matrix.

    Uses the Rodrigues angle formula: θ = arccos((trace(R) - 1) / 2).

    Args:
        R: 3×3 rotation matrix.

    Returns:
        Rotation angle in degrees.
    """
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def load_extrinsic_npz(
    camera_id: str,
    extrinsic_path: str | Path,
    intrinsic_path: str | Path | None = None,
) -> CameraExtrinsic:
    """
    Load extrinsic (and optionally intrinsic) parameters from .npz files.

    Expected keys in *extrinsic_path*: ``"R"`` (3×3) and ``"t"`` (3 or 3×1).
    Expected keys in *intrinsic_path*: ``"K"`` (3×3), ``"dist"`` (N,).

    Args:
        camera_id:       Human-readable camera identifier.
        extrinsic_path:  Path to the extrinsics .npz file.
        intrinsic_path:  Path to the intrinsics .npz file (optional).

    Returns:
        :class:`CameraExtrinsic` named tuple.
    """
    ext = np.load(str(extrinsic_path))
    R = ext["R"].astype(np.float64)
    t = ext["t"].astype(np.float64).reshape(3, 1)

    K    = np.eye(3, dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    if intrinsic_path is not None:
        intr = np.load(str(intrinsic_path))
        K    = intr["K"].astype(np.float64)
        dist = intr.get("dist", np.zeros(5)).astype(np.float64)

    return CameraExtrinsic(camera_id=camera_id, R=R, t=t, K=K, dist=dist)
