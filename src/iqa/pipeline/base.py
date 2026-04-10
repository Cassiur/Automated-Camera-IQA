"""Abstract base class for ISP pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from iqa.utils.logger import get_logger


class BaseStage(ABC):
    """
    Contract for every ISP processing stage.

    Each stage receives a *float32* image and a *meta* dictionary that carries
    pipeline-wide context (bayer_pattern, bit_depth, AWB gains, etc.) and
    returns the transformed image together with the (possibly updated) meta.

    Subclasses must implement :meth:`process`.
    """

    #: Short identifier used in log messages and intermediate-result keys.
    name: str = "base"

    def __call__(
        self, image: np.ndarray, meta: dict
    ) -> tuple[np.ndarray, dict]:
        self.validate_input(image, meta)
        return self.process(image, meta)

    @abstractmethod
    def process(
        self, image: np.ndarray, meta: dict
    ) -> tuple[np.ndarray, dict]:
        """
        Execute the stage.

        Args:
            image: Current float32 image array.
            meta:  Pipeline context dictionary.

        Returns:
            Tuple of (processed image, updated meta).
        """

    def validate_input(self, image: np.ndarray, meta: dict) -> None:
        """
        Lightweight sanity checks executed before :meth:`process`.

        Raises:
            TypeError:  If *image* is not a NumPy ndarray.
            ValueError: If *image* dtype is not float32.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"[{self.name}] Expected np.ndarray, got {type(image)}")
        if image.dtype != np.float32:
            raise ValueError(
                f"[{self.name}] Expected float32 image, got {image.dtype}"
            )

    @property
    def _log(self):
        return get_logger(f"iqa.pipeline.{self.name}")
