"""
Integration tests for RPi5-Pico communication.

These tests verify the communication between Raspberry Pi 5 and MouthMaster Pico
over USB serial. Tests can run in two modes:
1. Mock mode (default): Uses a mock serial port for CI/automated testing
2. Hardware mode: Requires actual Pico connected (set PICO_SERIAL_PORT env var)

**Validates: Requirements 7.2**
- Test command sending and response
- Test sustained communication at 30 Hz
"""

import os
import time
import threading
import queue
from typing import Dict, Optional, List
from unittest.mock import MagicMock, patch
from io import BytesIO

import pytest

from expression_control.protocol import ServoCommandProtocol
from expression_control.serial_manager import SerialManager


# Check if hardware testing is enabled
PICO_SERIAL_PORT = os.environ.get("PICO_SERIAL_PORT")
HARDWARE_AVAILABLE = PICO_SERIAL_PORT is not None


class MockSerial:
    """
    Mock serial port that simulates MouthMaster Pico behavior.
    
    Simulates the Pico's command parsing and response generation
    for testing without actual hardware.
    """
    
    def __init__(self, port: str = "", baudrate: int = 115200, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._is_open = True
        self._read_buffer = BytesIO()
        self._write_buffer = BytesIO()
        self._current_angles: Dict[str, int] = {
            name: 90 for name in ServoCommandProtocol.SERVO_ORDER
        }
        self._response_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        
    @property
    def is_open(self) -> bool:
        return self._is_open
    
    def close(self) -> None:
        self._is_open = False
        
    def write(self, data: bytes) -> int:
        """Process incoming command and generate response."""
        command = data.decode('utf-8').strip()
        response = self._process_command(command)
        if response:
            self._response_queue.put(response)
        return len(data)
    
    def flush(self) -> None:
        pass
    
    def readline(self) -> bytes:
        """Return the next response from the queue."""
        try:
            response = self._response_queue.get(timeout=self.timeout)
            return (response + "\n").encode('utf-8')
        except queue.Empty:
            return b""
    
    def _process_command(self, command: str) -> Optional[str]:
        """
        Simulate MouthMaster Pico command processing.
        
        Mirrors the behavior in MouthMaster.py main loop.
        """
        with self._lock:
            if command.startswith("angles:"):
                angles = self._parse_angles_command(command)
                if angles is not None:
                    # Update internal state
                    for i, name in enumerate(ServoCommandProtocol.SERVO_ORDER):
                        self._current_angles[name] = angles[i]
                    return "OK"
                else:
                    return "Error: Invalid angles command format"
            elif command == "get_angles":
                angle_values = [
                    str(self._current_angles[name]) 
                    for name in ServoCommandProtocol.SERVO_ORDER
                ]
                return "angles:" + ",".join(angle_values)
            elif command in ("on", "off"):
                return None  # LED commands don't respond
            elif command in ("eyes_move", "eyes_open", "eyes_close",
                           "brows_up", "brows_down", "brows_happy", "brows_angry",
                           "mouth_closed", "mouth_open", "lips_down", "lips_up",
                           "tongue_down", "tongue_up", "smile", "frown", "wide", "narrow"):
                return None  # Pose commands don't respond
            else:
                return "Unknown command"
    
    def _parse_angles_command(self, command: str) -> Optional[List[int]]:
        """Parse angles command, mirroring MouthMaster.py logic."""
        if not command.startswith("angles:"):
            return None
        
        angle_str = command[7:]
        if not angle_str:
            return None
        
        parts = angle_str.split(",")
        if len(parts) != 21:
            return None
        
        angles = []
        for part in parts:
            try:
                angle = int(part.strip())
                if angle < 0 or angle > 180:
                    return None
                angles.append(angle)
            except ValueError:
                return None
        
        return angles


class TestCommandSendingAndResponse:
    """
    Test command sending and response handling.
    
    **Validates: Requirements 7.2** - Integration tests for RPi5-Pico communication
    """
    
    @pytest.fixture
    def mock_serial_manager(self):
        """Create a SerialManager with mocked serial port."""
        with patch('expression_control.serial_manager.serial') as mock_serial_module:
            mock_serial_module.Serial = MockSerial
            manager = SerialManager(port="/dev/mock", baudrate=115200)
            manager.connect()
            yield manager
            manager.disconnect()
    
    def test_send_angles_command_receives_ok_response(self, mock_serial_manager):
        """Test that sending valid angles receives OK response."""
        angles = {name: 90 for name in ServoCommandProtocol.SERVO_ORDER}
        
        # Send angles
        success = mock_serial_manager.send_angles(angles)
        assert success, "send_angles should return True on success"
        
        # Read response
        response = mock_serial_manager.read_response()
        assert response == "OK", f"Expected 'OK' response, got: {response}"
    
    def test_send_angles_updates_pico_state(self, mock_serial_manager):
        """Test that sent angles are stored in Pico state."""
        # Set specific angles
        angles = {name: i * 8 for i, name in enumerate(ServoCommandProtocol.SERVO_ORDER)}
        # Clamp to valid range
        angles = {name: min(180, angle) for name, angle in angles.items()}
        
        # Send angles
        mock_serial_manager.send_angles(angles)
        mock_serial_manager.read_response()  # Consume OK
        
        # Request readback
        mock_serial_manager.send_command("get_angles")
        response = mock_serial_manager.read_response()
        
        # Parse response and verify
        assert response is not None
        decoded = ServoCommandProtocol.decode(response)
        assert decoded is not None, f"Failed to decode response: {response}"
        assert decoded == angles, f"Angles mismatch: expected {angles}, got {decoded}"
    
    def test_send_invalid_angles_receives_error(self, mock_serial_manager):
        """Test that invalid angle format receives error response."""
        # Send malformed command directly
        mock_serial_manager.send_command("angles:invalid")
        response = mock_serial_manager.read_response()
        
        assert response is not None
        assert "Error" in response or "error" in response.lower()
    
    def test_send_raw_command_compatibility(self, mock_serial_manager):
        """Test backward compatibility with existing commands."""
        # Test pose commands (these don't generate responses in real Pico)
        success = mock_serial_manager.send_command("mouth_open")
        assert success, "send_command should return True"
        
        success = mock_serial_manager.send_command("eyes_open")
        assert success, "send_command should return True"
    
    def test_get_angles_readback(self, mock_serial_manager):
        """Test angle readback command for data collection."""
        # First set some angles
        angles = {name: 45 for name in ServoCommandProtocol.SERVO_ORDER}
        mock_serial_manager.send_angles(angles)
        mock_serial_manager.read_response()  # Consume OK
        
        # Request readback
        mock_serial_manager.send_command("get_angles")
        response = mock_serial_manager.read_response()
        
        assert response is not None
        assert response.startswith("angles:")
        decoded = ServoCommandProtocol.decode(response)
        assert decoded == angles


class TestSustainedCommunication:
    """
    Test sustained communication at 30 Hz.
    
    **Validates: Requirements 7.2** - Test sustained communication at 30 Hz
    """
    
    @pytest.fixture
    def mock_serial_manager(self):
        """Create a SerialManager with mocked serial port."""
        with patch('expression_control.serial_manager.serial') as mock_serial_module:
            mock_serial_module.Serial = MockSerial
            manager = SerialManager(port="/dev/mock", baudrate=115200)
            manager.connect()
            yield manager
            manager.disconnect()
    
    def test_sustained_30hz_communication_1_second(self, mock_serial_manager):
        """
        Test sustained communication at 30 Hz for 1 second.
        
        Sends 30 angle commands in 1 second and verifies all succeed.
        """
        target_hz = 30
        duration_seconds = 1.0
        target_commands = int(target_hz * duration_seconds)
        interval = 1.0 / target_hz
        
        successful_sends = 0
        successful_responses = 0
        start_time = time.time()
        
        for i in range(target_commands):
            # Generate varying angles to simulate real usage
            angles = {
                name: 90 + (i % 20) - 10  # Oscillate around 90
                for name in ServoCommandProtocol.SERVO_ORDER
            }
            
            # Send command
            if mock_serial_manager.send_angles(angles):
                successful_sends += 1
            
            # Read response (non-blocking check)
            response = mock_serial_manager.read_response(timeout=0.01)
            if response == "OK":
                successful_responses += 1
            
            # Maintain timing
            elapsed = time.time() - start_time
            expected_elapsed = (i + 1) * interval
            if elapsed < expected_elapsed:
                time.sleep(expected_elapsed - elapsed)
        
        total_time = time.time() - start_time
        actual_hz = target_commands / total_time
        
        # Verify results
        assert successful_sends == target_commands, \
            f"Expected {target_commands} successful sends, got {successful_sends}"
        assert successful_responses >= target_commands * 0.95, \
            f"Expected at least 95% responses, got {successful_responses}/{target_commands}"
        assert actual_hz >= target_hz * 0.9, \
            f"Expected at least 27 Hz, achieved {actual_hz:.1f} Hz"
    
    def test_sustained_30hz_communication_5_seconds(self, mock_serial_manager):
        """
        Test sustained communication at 30 Hz for 5 seconds.
        
        Longer duration test to verify stability.
        """
        target_hz = 30
        duration_seconds = 5.0
        target_commands = int(target_hz * duration_seconds)
        interval = 1.0 / target_hz
        
        successful_sends = 0
        errors = []
        start_time = time.time()
        
        for i in range(target_commands):
            # Generate varying angles
            phase = (i / target_commands) * 2 * 3.14159
            import math
            offset = int(45 * math.sin(phase))
            angles = {
                name: 90 + offset
                for name in ServoCommandProtocol.SERVO_ORDER
            }
            
            try:
                if mock_serial_manager.send_angles(angles):
                    successful_sends += 1
                    # Consume response
                    mock_serial_manager.read_response(timeout=0.005)
            except Exception as e:
                errors.append(str(e))
            
            # Maintain timing
            elapsed = time.time() - start_time
            expected_elapsed = (i + 1) * interval
            if elapsed < expected_elapsed:
                time.sleep(expected_elapsed - elapsed)
        
        total_time = time.time() - start_time
        actual_hz = target_commands / total_time
        success_rate = successful_sends / target_commands
        
        # Verify results
        assert success_rate >= 0.99, \
            f"Expected 99% success rate, got {success_rate*100:.1f}%"
        assert len(errors) == 0, f"Encountered errors: {errors[:5]}"
        assert actual_hz >= target_hz * 0.9, \
            f"Expected at least 27 Hz, achieved {actual_hz:.1f} Hz"
    
    def test_communication_latency(self, mock_serial_manager):
        """
        Test round-trip latency for command-response cycle.
        
        Measures time from send to response received.
        """
        latencies = []
        num_samples = 100
        
        angles = {name: 90 for name in ServoCommandProtocol.SERVO_ORDER}
        
        for _ in range(num_samples):
            start = time.time()
            mock_serial_manager.send_angles(angles)
            response = mock_serial_manager.read_response(timeout=0.1)
            end = time.time()
            
            if response == "OK":
                latencies.append((end - start) * 1000)  # Convert to ms
        
        assert len(latencies) >= num_samples * 0.95, \
            f"Too many failed responses: {num_samples - len(latencies)}"
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        # For mock, latency should be very low
        # For real hardware, expect < 33ms (one frame at 30 Hz)
        assert avg_latency < 10, f"Average latency too high: {avg_latency:.2f}ms"
        assert max_latency < 50, f"Max latency too high: {max_latency:.2f}ms"




@pytest.mark.skipif(not HARDWARE_AVAILABLE, reason="Hardware not available")
class TestHardwareCommunication:
    """
    Hardware integration tests - only run when PICO_SERIAL_PORT is set.
    
    To run these tests:
    1. Connect MouthMaster Pico via USB
    2. Set environment variable: export PICO_SERIAL_PORT=/dev/ttyACM0
    3. Run: pytest tests/test_integration_rpi5_pico.py -v
    
    **Validates: Requirements 7.2**
    """
    
    @pytest.fixture
    def hardware_serial_manager(self):
        """Create a SerialManager connected to real hardware."""
        manager = SerialManager(port=PICO_SERIAL_PORT, baudrate=115200)
        connected = manager.connect()
        if not connected:
            pytest.skip(f"Could not connect to Pico at {PICO_SERIAL_PORT}")
        yield manager
        manager.disconnect()
    
    def test_hardware_connection(self, hardware_serial_manager):
        """Test basic connection to hardware."""
        assert hardware_serial_manager.is_connected
    
    def test_hardware_send_angles(self, hardware_serial_manager):
        """Test sending angles to real hardware."""
        # Use neutral angles for safety
        angles = {name: 90 for name in ServoCommandProtocol.SERVO_ORDER}
        
        success = hardware_serial_manager.send_angles(angles)
        assert success, "Failed to send angles to hardware"
        
        # Give hardware time to process
        time.sleep(0.1)
        
        response = hardware_serial_manager.read_response(timeout=1.0)
        assert response == "OK", f"Unexpected response: {response}"
    
    def test_hardware_angle_readback(self, hardware_serial_manager):
        """Test angle readback from real hardware."""
        # Set known angles
        test_angles = {name: 100 for name in ServoCommandProtocol.SERVO_ORDER}
        hardware_serial_manager.send_angles(test_angles)
        hardware_serial_manager.read_response(timeout=1.0)
        
        # Request readback
        hardware_serial_manager.send_command("get_angles")
        response = hardware_serial_manager.read_response(timeout=1.0)
        
        assert response is not None, "No response from get_angles"
        decoded = ServoCommandProtocol.decode(response)
        assert decoded is not None, f"Failed to decode: {response}"
        assert decoded == test_angles
    
    def test_hardware_sustained_30hz(self, hardware_serial_manager):
        """
        Test sustained 30 Hz communication with real hardware.
        
        Runs for 3 seconds to verify stable communication.
        """
        target_hz = 30
        duration_seconds = 3.0
        target_commands = int(target_hz * duration_seconds)
        interval = 1.0 / target_hz
        
        successful_sends = 0
        start_time = time.time()
        
        for i in range(target_commands):
            # Small oscillation around neutral for safety
            offset = (i % 10) - 5
            angles = {
                name: 90 + offset
                for name in ServoCommandProtocol.SERVO_ORDER
            }
            
            if hardware_serial_manager.send_angles(angles):
                successful_sends += 1
            
            # Maintain timing
            elapsed = time.time() - start_time
            expected_elapsed = (i + 1) * interval
            if elapsed < expected_elapsed:
                time.sleep(expected_elapsed - elapsed)
        
        total_time = time.time() - start_time
        actual_hz = target_commands / total_time
        success_rate = successful_sends / target_commands
        
        # Return to neutral
        neutral = {name: 90 for name in ServoCommandProtocol.SERVO_ORDER}
        hardware_serial_manager.send_angles(neutral)
        
        assert success_rate >= 0.95, \
            f"Expected 95% success rate, got {success_rate*100:.1f}%"
        assert actual_hz >= target_hz * 0.9, \
            f"Expected at least 27 Hz, achieved {actual_hz:.1f} Hz"


class TestConnectionResilience:
    """
    Test connection resilience and error handling.
    
    **Validates: Requirements 7.2**
    """
    
    @pytest.fixture
    def mock_serial_manager(self):
        """Create a SerialManager with mocked serial port."""
        with patch('expression_control.serial_manager.serial') as mock_serial_module:
            mock_serial_module.Serial = MockSerial
            manager = SerialManager(port="/dev/mock", baudrate=115200)
            yield manager
    
    def test_auto_connect_on_send(self, mock_serial_manager):
        """Test that send_angles auto-connects if not connected."""
        assert not mock_serial_manager.is_connected
        
        angles = {name: 90 for name in ServoCommandProtocol.SERVO_ORDER}
        success = mock_serial_manager.send_angles(angles)
        
        assert success, "send_angles should succeed with auto-connect"
        assert mock_serial_manager.is_connected
    
    def test_reconnect_after_disconnect(self, mock_serial_manager):
        """Test reconnection after explicit disconnect."""
        # Connect
        mock_serial_manager.connect()
        assert mock_serial_manager.is_connected
        
        # Disconnect
        mock_serial_manager.disconnect()
        assert not mock_serial_manager.is_connected
        
        # Reconnect via send
        angles = {name: 90 for name in ServoCommandProtocol.SERVO_ORDER}
        success = mock_serial_manager.send_angles(angles)
        
        assert success
        assert mock_serial_manager.is_connected
    
    def test_context_manager_usage(self):
        """Test SerialManager as context manager."""
        with patch('expression_control.serial_manager.serial') as mock_serial_module:
            mock_serial_module.Serial = MockSerial
            
            with SerialManager(port="/dev/mock") as manager:
                assert manager.is_connected
                angles = {name: 90 for name in ServoCommandProtocol.SERVO_ORDER}
                success = manager.send_angles(angles)
                assert success
            
            # After context exit, should be disconnected
            assert not manager.is_connected
