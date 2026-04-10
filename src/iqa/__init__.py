"""
Automated-Camera-IQA
====================
Automotive camera ISP simulation and image quality assessment toolkit.

Modules:
    pipeline   - ISP processing stages (BLC, Demosaic, AWB, CCM)
    metrics    - SFR/MTF quality evaluation (ISO 12233)
    calibration - Boresight alignment and extrinsic validation
    cli        - Command-line interface
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from iqa.pipeline.pipeline import ISPPipeline, PipelineResult
from iqa.metrics.sfr import SFRAnalyzer
from iqa.calibration.boresight import BoresightChecker

__all__ = ["ISPPipeline", "PipelineResult", "SFRAnalyzer", "BoresightChecker"]
