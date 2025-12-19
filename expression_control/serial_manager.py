"""
Serial communication manager for USB connection to MouthMaster Pico.

This module handles the USB serial communication between the Raspberry Pi 5
and the MouthMaster Pico, including connection management and command sending.
"""

import logging
import time
from typing import Dict, Optional

try:
    import serial
    from serial import SerialException
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    SerialException = Exception  # type: ignore

from expression_control.protocol import ServoCommandProtocol


logger = logging.getLogger(__name__)


class SerialManager:
    """USB serial communication manager for MouthMaster Pico."""

    DEFAULT_PORT = "/dev/ttyACM0"
    DEFAULT_BAUDRATE = 115200
    DEFAULT_TIMEOUT = 1.0  # seconds
    COMMAND_TERMINATOR = "\n"

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout: float = DEFAULT_TIMEOUT
    ):
        """
        Initialize the serial manager.

        Args:
            port: Serial port path (e.g., "/dev/ttyACM0" on Linux).
            baudrate: Communication baud rate.
            timeout: Read timeout in seconds.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial: Optional["serial.Serial"] = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if serial connection is active."""
        return self._connected and self._serial is not None and self._serial.is_open

    def connect(self) -> bool:
        """
        Establish serial connection to MouthMaster Pico.

        Returns:
            True if connection successful, False otherwise.
        """
        if not SERIAL_AVAILABLE:
            logger.error("pyserial not installed. Install with: pip install pyserial")
            return False

        if self.is_connected:
            return True

        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            self._connected = True
            logger.info(f"Connected to {self.port} at {self.baudrate} baud")
            # Allow time for connection to stabilize
            time.sleep(0.1)
            return True
        except SerialException as e:
            logger.error(f"Failed to connect to {self.port}: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Close the serial connection."""
        if self._serial is not None:
            try:
                self._serial.close()
            except SerialException as e:
                logger.warning(f"Error closing serial port: {e}")
            finally:
                self._serial = None
                self._connected = False
                logger.info("Serial connection closed")

    def send_command(self, command: str) -> bool:
        """
        Send a raw command string to MouthMaster Pico.

        Args:
            command: Command string to send.

        Returns:
            True if command sent successfully, False otherwise.
        """
        if not self.is_connected:
            if not self.connect():
                return False

        try:
            full_command = command + self.COMMAND_TERMINATOR
            self._serial.write(full_command.encode('utf-8'))  # type: ignore
            self._serial.flush()  # type: ignore
            logger.debug(f"Sent command: {command}")
            return True
        except SerialException as e:
            logger.error(f"Failed to send command: {e}")
            self._connected = False
            return False

    def send_angles(self, angles: Dict[str, int]) -> bool:
        """
        Send servo angle command to MouthMaster Pico.

        Args:
            angles: Dictionary mapping servo names to angle values.
                    Must contain all 21 servos with angles in range [0, 180].

        Returns:
            True if command sent successfully, False otherwise.

        Raises:
            ValueError: If angles dict is invalid (propagated from protocol encoder).
        """
        command = ServoCommandProtocol.encode(angles)
        return self.send_command(command)

    def read_response(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Read a response line from MouthMaster Pico.

        Args:
            timeout: Read timeout in seconds. Uses default if None.

        Returns:
            Response string (without terminator), or None if no response.
        """
        if not self.is_connected:
            return None

        try:
            if timeout is not None:
                old_timeout = self._serial.timeout  # type: ignore
                self._serial.timeout = timeout  # type: ignore

            response = self._serial.readline()  # type: ignore

            if timeout is not None:
                self._serial.timeout = old_timeout  # type: ignore

            if response:
                return response.decode('utf-8').strip()
            return None
        except SerialException as e:
            logger.error(f"Failed to read response: {e}")
            return None

    def __enter__(self) -> "SerialManager":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()
