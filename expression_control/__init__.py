"""
Expression Control Package

Vision-driven robot expression control system using MediaPipe and LNN-S4.
"""

__version__ = "0.1.0"

from expression_control.protocol import ServoCommandProtocol
from expression_control.serial_manager import SerialManager
from expression_control.smoother import TemporalSmoother
from expression_control.inference import (
    InferenceConfig,
    InferenceEngine,
    LNNS4Inference,
    FallbackMapper,
)
from expression_control.config import load_config, save_config

__all__ = [
    "ServoCommandProtocol",
    "SerialManager",
    "TemporalSmoother",
    "InferenceConfig",
    "InferenceEngine",
    "LNNS4Inference",
    "FallbackMapper",
    "load_config",
    "save_config",
]
