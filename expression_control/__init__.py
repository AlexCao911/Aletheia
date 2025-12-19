"""
Expression Control Package

Vision-driven robot expression control system using MediaPipe and LNN-S4.
"""

__version__ = "0.1.0"

from expression_control.protocol import ServoCommandProtocol
from expression_control.serial_manager import SerialManager

__all__ = [
    "ServoCommandProtocol",
    "SerialManager",
]
