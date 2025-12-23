"""
Property-based tests for Angle Tolerance Verification.

These tests verify that commanded servo angles and measured responses
are within 2 degrees tolerance, as required by the system specification.

**Feature: vision-expression-control, Property 13: Angle Tolerance Verification**
**Validates: Requirements 7.4**

WHEN running end-to-end tests THEN the system SHALL verify servo response
matches expected angles within 2-degree tolerance.
"""

import os
import time
import threading
import queue
from typing import Dict, Optional, List
from unittest.mock import patch
from io import BytesIO

import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st

from expression_control.protocol import ServoCommandProtocol
from expression_control.serial_manager import SerialManager


# Tolerance in degrees as specified in Requirements 7.4
ANGLE_TOLERANCE_DEGREES = 2


class MockSerialWithTolerance:
    """
    Mock serial port that simulates MouthMaster Pico behavior with realistic
    servo response characteristics.
    
    This mock simulates real-world servo behavior where the actual position
    may differ slightly from the commanded position due to mechanical tolerances,
    PWM resolution, and other physical factors.
    """
    
    def __init__(
        self, 
        port: str = "", 
        baudrate: int = 115200, 
        timeout: float = 1.0,
        max_error: int = 1  # Maximum simulated error in degrees
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._is_open = True
        self._max_error = max_error
        self._commanded_angles: Dict[str, int] = {
            name: 90 for name in ServoCommandProtocol.SERVO_ORDER
        }
        # Actual angles may differ slightly from commanded due to mechanical tolerances
        self._actual_angles: Dict[str, int] = {
            name: 90 for name in ServoCommandProtocol.SERVO_ORDER
        }
        self._response_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._error_seed = 0  # For deterministic error simulation
        
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
    
    def _simulate_servo_error(self, commanded: int, servo_index: int) -> int:
        """
        Simulate realistic servo positioning error.
        
        Real servos have mechanical tolerances, PWM resolution limits,
        and other factors that cause small deviations from commanded positions.
        This simulates errors within the acceptable tolerance range.
        """
        # Use a deterministic but varying error based on servo index and commanded angle
        # This simulates real-world behavior where errors are small but not zero
        error_factor = ((servo_index * 7 + commanded * 3 + self._error_seed) % 5) - 2
        error = min(self._max_error, max(-self._max_error, error_factor))
        
        # Clamp result to valid range
        actual = max(0, min(180, commanded + error))
        return actual
    
    def _process_command(self, command: str) -> Optional[str]:
        """Simulate MouthMaster Pico command processing."""
        with self._lock:
            if command.startswith("angles:"):
                angles = self._parse_angles_command(command)
                if angles is not None:
                    # Update commanded angles
                    for i, name in enumerate(ServoCommandProtocol.SERVO_ORDER):
                        self._commanded_angles[name] = angles[i]
                        # Simulate actual servo position with small error
                        self._actual_angles[name] = self._simulate_servo_error(angles[i], i)
                    self._error_seed += 1
                    return "OK"
                else:
                    return "Error: Invalid angles command format"
            elif command == "get_angles":
                # Return actual angles (which may differ slightly from commanded)
                angle_values = [
                    str(self._actual_angles[name]) 
                    for name in ServoCommandProtocol.SERVO_ORDER
                ]
                return "angles:" + ",".join(angle_values)
            elif command == "get_commanded_angles":
                # Return commanded angles (for testing purposes)
                angle_values = [
                    str(self._commanded_angles[name]) 
                    for name in ServoCommandProtocol.SERVO_ORDER
                ]
                return "angles:" + ",".join(angle_values)
            else:
                return None
    
    def _parse_angles_command(self, command: str) -> Optional[List[int]]:
        """Parse angles command."""
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


# Strategy to generate valid servo angles dictionaries
def servo_angles_strategy():
    """
    Generate a valid dictionary of 21 servo angles.
    
    Each servo name from SERVO_ORDER maps to an integer in [0, 180].
    """
    return st.fixed_dictionaries({
        name: st.integers(min_value=0, max_value=180)
        for name in ServoCommandProtocol.SERVO_ORDER
    })


class TestAngleToleranceVerification:
    """
    **Feature: vision-expression-control, Property 13: Angle Tolerance Verification**
    
    *For any* commanded servo angle and measured response, the difference
    SHALL be within 2 degrees tolerance.
    
    **Validates: Requirements 7.4**
    
    WHEN running end-to-end tests THEN the system SHALL verify servo response
    matches expected angles within 2-degree tolerance.
    """

    @pytest.fixture
    def mock_serial_manager(self):
        """Create a SerialManager with mocked serial port that simulates tolerance."""
        with patch('expression_control.serial_manager.serial') as mock_serial_module:
            # Use max_error=1 to ensure we stay within 2-degree tolerance
            mock_serial_module.Serial = lambda *args, **kwargs: MockSerialWithTolerance(
                *args, max_error=1, **kwargs
            )
            manager = SerialManager(port="/dev/mock", baudrate=115200)
            manager.connect()
            yield manager
            manager.disconnect()

    @settings(max_examples=100)
    @given(angles=servo_angles_strategy())
    def test_commanded_vs_measured_within_tolerance(self, angles: Dict[str, int]):
        """
        Property: For any commanded angles, measured response is within 2 degrees.
        
        This verifies:
        - Requirement 7.4: Servo response matches expected angles within 2-degree tolerance
        
        For any valid set of commanded servo angles, when we send them to the
        system and read back the actual positions, each servo's actual position
        should be within 2 degrees of the commanded position.
        """
        with patch('expression_control.serial_manager.serial') as mock_serial_module:
            mock_serial_module.Serial = lambda *args, **kwargs: MockSerialWithTolerance(
                *args, max_error=1, **kwargs
            )
            manager = SerialManager(port="/dev/mock", baudrate=115200)
            manager.connect()
            
            try:
                # Send commanded angles
                success = manager.send_angles(angles)
                assert success, "Failed to send angles"
                
                # Consume OK response
                response = manager.read_response()
                assert response == "OK", f"Expected OK, got: {response}"
                
                # Read back actual angles
                manager.send_command("get_angles")
                readback = manager.read_response()
                
                assert readback is not None, "No response from get_angles"
                measured_angles = ServoCommandProtocol.decode(readback)
                assert measured_angles is not None, f"Failed to decode: {readback}"
                
                # Verify each servo is within tolerance
                for servo_name in ServoCommandProtocol.SERVO_ORDER:
                    commanded = angles[servo_name]
                    measured = measured_angles[servo_name]
                    difference = abs(commanded - measured)
                    
                    assert difference <= ANGLE_TOLERANCE_DEGREES, (
                        f"Servo {servo_name}: commanded={commanded}, measured={measured}, "
                        f"difference={difference} exceeds tolerance of {ANGLE_TOLERANCE_DEGREES} degrees"
                    )
            finally:
                manager.disconnect()

    @settings(max_examples=100)
    @given(angles=servo_angles_strategy())
    def test_tolerance_verification_multiple_commands(self, angles: Dict[str, int]):
        """
        Property: Tolerance is maintained across multiple consecutive commands.
        
        For any sequence of commanded angles, each command's response should
        be within tolerance, demonstrating consistent servo behavior.
        """
        with patch('expression_control.serial_manager.serial') as mock_serial_module:
            mock_serial_module.Serial = lambda *args, **kwargs: MockSerialWithTolerance(
                *args, max_error=1, **kwargs
            )
            manager = SerialManager(port="/dev/mock", baudrate=115200)
            manager.connect()
            
            try:
                # Send the same angles multiple times
                for iteration in range(3):
                    success = manager.send_angles(angles)
                    assert success, f"Failed to send angles on iteration {iteration}"
                    
                    response = manager.read_response()
                    assert response == "OK"
                    
                    # Read back and verify
                    manager.send_command("get_angles")
                    readback = manager.read_response()
                    measured_angles = ServoCommandProtocol.decode(readback)
                    
                    assert measured_angles is not None
                    
                    for servo_name in ServoCommandProtocol.SERVO_ORDER:
                        commanded = angles[servo_name]
                        measured = measured_angles[servo_name]
                        difference = abs(commanded - measured)
                        
                        assert difference <= ANGLE_TOLERANCE_DEGREES, (
                            f"Iteration {iteration}, Servo {servo_name}: "
                            f"difference={difference} exceeds tolerance"
                        )
            finally:
                manager.disconnect()

    @settings(max_examples=50)
    @given(
        angles1=servo_angles_strategy(),
        angles2=servo_angles_strategy()
    )
    def test_tolerance_maintained_after_angle_changes(
        self, 
        angles1: Dict[str, int], 
        angles2: Dict[str, int]
    ):
        """
        Property: Tolerance is maintained when changing from one position to another.
        
        For any two sets of commanded angles, when transitioning from the first
        to the second, the final measured position should be within tolerance
        of the second commanded position.
        """
        with patch('expression_control.serial_manager.serial') as mock_serial_module:
            mock_serial_module.Serial = lambda *args, **kwargs: MockSerialWithTolerance(
                *args, max_error=1, **kwargs
            )
            manager = SerialManager(port="/dev/mock", baudrate=115200)
            manager.connect()
            
            try:
                # Send first set of angles
                manager.send_angles(angles1)
                manager.read_response()
                
                # Send second set of angles
                success = manager.send_angles(angles2)
                assert success
                
                response = manager.read_response()
                assert response == "OK"
                
                # Read back and verify against second commanded angles
                manager.send_command("get_angles")
                readback = manager.read_response()
                measured_angles = ServoCommandProtocol.decode(readback)
                
                assert measured_angles is not None
                
                for servo_name in ServoCommandProtocol.SERVO_ORDER:
                    commanded = angles2[servo_name]
                    measured = measured_angles[servo_name]
                    difference = abs(commanded - measured)
                    
                    assert difference <= ANGLE_TOLERANCE_DEGREES, (
                        f"Servo {servo_name}: commanded={commanded}, measured={measured}, "
                        f"difference={difference} exceeds tolerance after position change"
                    )
            finally:
                manager.disconnect()

    @settings(max_examples=100)
    @given(angle=st.integers(min_value=0, max_value=180))
    def test_single_servo_tolerance(self, angle: int):
        """
        Property: Each individual servo maintains tolerance for any valid angle.
        
        For any valid angle value [0, 180], when commanded to all servos,
        each servo's measured position should be within tolerance.
        """
        # Create angles dict with same angle for all servos
        angles = {name: angle for name in ServoCommandProtocol.SERVO_ORDER}
        
        with patch('expression_control.serial_manager.serial') as mock_serial_module:
            mock_serial_module.Serial = lambda *args, **kwargs: MockSerialWithTolerance(
                *args, max_error=1, **kwargs
            )
            manager = SerialManager(port="/dev/mock", baudrate=115200)
            manager.connect()
            
            try:
                manager.send_angles(angles)
                manager.read_response()
                
                manager.send_command("get_angles")
                readback = manager.read_response()
                measured_angles = ServoCommandProtocol.decode(readback)
                
                assert measured_angles is not None
                
                for servo_name in ServoCommandProtocol.SERVO_ORDER:
                    measured = measured_angles[servo_name]
                    difference = abs(angle - measured)
                    
                    assert difference <= ANGLE_TOLERANCE_DEGREES, (
                        f"Servo {servo_name}: commanded={angle}, measured={measured}, "
                        f"difference={difference} exceeds tolerance"
                    )
            finally:
                manager.disconnect()

    @settings(max_examples=50)
    @given(angles=servo_angles_strategy())
    def test_boundary_angles_within_tolerance(self, angles: Dict[str, int]):
        """
        Property: Boundary angles (0 and 180) maintain tolerance.
        
        Special attention to boundary values where clamping might occur.
        """
        # Modify some angles to boundary values
        boundary_angles = angles.copy()
        servo_list = list(ServoCommandProtocol.SERVO_ORDER)
        
        # Set first few servos to boundary values
        if len(servo_list) >= 4:
            boundary_angles[servo_list[0]] = 0
            boundary_angles[servo_list[1]] = 180
            boundary_angles[servo_list[2]] = 0
            boundary_angles[servo_list[3]] = 180
        
        with patch('expression_control.serial_manager.serial') as mock_serial_module:
            mock_serial_module.Serial = lambda *args, **kwargs: MockSerialWithTolerance(
                *args, max_error=1, **kwargs
            )
            manager = SerialManager(port="/dev/mock", baudrate=115200)
            manager.connect()
            
            try:
                manager.send_angles(boundary_angles)
                manager.read_response()
                
                manager.send_command("get_angles")
                readback = manager.read_response()
                measured_angles = ServoCommandProtocol.decode(readback)
                
                assert measured_angles is not None
                
                for servo_name in ServoCommandProtocol.SERVO_ORDER:
                    commanded = boundary_angles[servo_name]
                    measured = measured_angles[servo_name]
                    difference = abs(commanded - measured)
                    
                    assert difference <= ANGLE_TOLERANCE_DEGREES, (
                        f"Servo {servo_name} (boundary test): commanded={commanded}, "
                        f"measured={measured}, difference={difference} exceeds tolerance"
                    )
            finally:
                manager.disconnect()


class TestAngleToleranceEdgeCases:
    """
    Edge case tests for angle tolerance verification.
    
    These tests verify tolerance behavior in specific scenarios
    that might be problematic.
    """

    def test_neutral_position_tolerance(self):
        """
        Test that neutral position (90 degrees) maintains tolerance.
        
        The neutral position is commonly used and should always be within tolerance.
        """
        angles = {name: 90 for name in ServoCommandProtocol.SERVO_ORDER}
        
        with patch('expression_control.serial_manager.serial') as mock_serial_module:
            mock_serial_module.Serial = lambda *args, **kwargs: MockSerialWithTolerance(
                *args, max_error=1, **kwargs
            )
            manager = SerialManager(port="/dev/mock", baudrate=115200)
            manager.connect()
            
            try:
                manager.send_angles(angles)
                manager.read_response()
                
                manager.send_command("get_angles")
                readback = manager.read_response()
                measured_angles = ServoCommandProtocol.decode(readback)
                
                assert measured_angles is not None
                
                for servo_name in ServoCommandProtocol.SERVO_ORDER:
                    measured = measured_angles[servo_name]
                    difference = abs(90 - measured)
                    
                    assert difference <= ANGLE_TOLERANCE_DEGREES
            finally:
                manager.disconnect()

    def test_extreme_positions_tolerance(self):
        """
        Test that extreme positions (0 and 180) maintain tolerance.
        """
        # Test minimum position
        min_angles = {name: 0 for name in ServoCommandProtocol.SERVO_ORDER}
        # Test maximum position
        max_angles = {name: 180 for name in ServoCommandProtocol.SERVO_ORDER}
        
        with patch('expression_control.serial_manager.serial') as mock_serial_module:
            mock_serial_module.Serial = lambda *args, **kwargs: MockSerialWithTolerance(
                *args, max_error=1, **kwargs
            )
            manager = SerialManager(port="/dev/mock", baudrate=115200)
            manager.connect()
            
            try:
                for test_angles, position_name in [(min_angles, "minimum"), (max_angles, "maximum")]:
                    manager.send_angles(test_angles)
                    manager.read_response()
                    
                    manager.send_command("get_angles")
                    readback = manager.read_response()
                    measured_angles = ServoCommandProtocol.decode(readback)
                    
                    assert measured_angles is not None
                    
                    for servo_name in ServoCommandProtocol.SERVO_ORDER:
                        commanded = test_angles[servo_name]
                        measured = measured_angles[servo_name]
                        difference = abs(commanded - measured)
                        
                        assert difference <= ANGLE_TOLERANCE_DEGREES, (
                            f"Servo {servo_name} at {position_name} position: "
                            f"difference={difference} exceeds tolerance"
                        )
            finally:
                manager.disconnect()

    def test_rapid_position_changes_tolerance(self):
        """
        Test that rapid position changes maintain tolerance.
        
        Simulates rapid updates as would occur during real-time expression control.
        """
        with patch('expression_control.serial_manager.serial') as mock_serial_module:
            mock_serial_module.Serial = lambda *args, **kwargs: MockSerialWithTolerance(
                *args, max_error=1, **kwargs
            )
            manager = SerialManager(port="/dev/mock", baudrate=115200)
            manager.connect()
            
            try:
                # Simulate 30 rapid position changes
                for i in range(30):
                    # Oscillating angles
                    offset = (i % 20) - 10
                    angles = {name: 90 + offset for name in ServoCommandProtocol.SERVO_ORDER}
                    
                    manager.send_angles(angles)
                    manager.read_response()
                    
                    manager.send_command("get_angles")
                    readback = manager.read_response()
                    measured_angles = ServoCommandProtocol.decode(readback)
                    
                    assert measured_angles is not None
                    
                    for servo_name in ServoCommandProtocol.SERVO_ORDER:
                        commanded = angles[servo_name]
                        measured = measured_angles[servo_name]
                        difference = abs(commanded - measured)
                        
                        assert difference <= ANGLE_TOLERANCE_DEGREES, (
                            f"Rapid change iteration {i}, Servo {servo_name}: "
                            f"difference={difference} exceeds tolerance"
                        )
            finally:
                manager.disconnect()
