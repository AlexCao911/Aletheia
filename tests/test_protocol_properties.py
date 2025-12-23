"""
Property-based tests for ServoCommandProtocol.

These tests verify correctness properties of the protocol encoder/decoder
using Hypothesis for property-based testing.
"""

import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st

from expression_control.protocol import ServoCommandProtocol


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


# Strategy to generate invalid angles (outside [0, 180])
def invalid_angle_strategy():
    """
    Generate an integer angle value outside the valid range [0, 180].
    
    This includes negative values and values greater than 180.
    """
    return st.one_of(
        st.integers(max_value=-1),  # Negative angles
        st.integers(min_value=181)  # Angles > 180
    )


class TestProtocolRoundTrip:
    """
    **Feature: vision-expression-control, Property 1: Protocol Round-Trip Consistency**
    
    *For any* valid dictionary of 21 servo angles (each in range 0-180),
    encoding to command string and then decoding back SHALL produce an
    identical dictionary of angles.
    
    **Validates: Requirements 2.3, 2.7, 2.8**
    """

    @settings(max_examples=100)
    @given(angles=servo_angles_strategy())
    def test_encode_decode_round_trip(self, angles: dict):
        """
        Property: encode(angles) -> decode -> should equal original angles.
        
        This verifies:
        - Requirement 2.3: Command format "angles:A1,A2,...,A21"
        - Requirement 2.7: Parsing extracts exactly 21 angle values
        - Requirement 2.8: Serialized strings can be parsed back to identical values
        """
        # Encode the angles to command string
        encoded = ServoCommandProtocol.encode(angles)
        
        # Verify command format
        assert encoded.startswith("angles:")
        
        # Decode back to dictionary
        decoded = ServoCommandProtocol.decode(encoded)
        
        # Round-trip should produce identical result
        assert decoded is not None, "Decode should not return None for valid encoded string"
        assert decoded == angles, f"Round-trip failed: {angles} -> {encoded} -> {decoded}"

    @settings(max_examples=100)
    @given(angles=servo_angles_strategy())
    def test_encode_produces_valid_format(self, angles: dict):
        """
        Property: encode always produces a string with correct format.
        
        The encoded string should:
        - Start with "angles:" prefix
        - Contain exactly 21 comma-separated values
        - Each value should be a valid integer string
        """
        encoded = ServoCommandProtocol.encode(angles)
        
        # Check prefix
        assert encoded.startswith(ServoCommandProtocol.COMMAND_PREFIX)
        
        # Extract and check values
        values_str = encoded[len(ServoCommandProtocol.COMMAND_PREFIX):]
        parts = values_str.split(",")
        
        # Should have exactly 21 values
        assert len(parts) == 21, f"Expected 21 values, got {len(parts)}"
        
        # Each part should be a valid integer
        for part in parts:
            int(part)  # Should not raise

    @settings(max_examples=100)
    @given(angles=servo_angles_strategy())
    def test_decode_preserves_servo_order(self, angles: dict):
        """
        Property: decode preserves the mapping between servo names and angles.
        
        Each servo should map to its correct angle value after round-trip.
        """
        encoded = ServoCommandProtocol.encode(angles)
        decoded = ServoCommandProtocol.decode(encoded)
        
        assert decoded is not None
        
        # Check each servo individually
        for servo_name in ServoCommandProtocol.SERVO_ORDER:
            assert servo_name in decoded, f"Missing servo: {servo_name}"
            assert decoded[servo_name] == angles[servo_name], \
                f"Servo {servo_name}: expected {angles[servo_name]}, got {decoded[servo_name]}"


class TestAngleRangeValidation:
    """
    **Feature: vision-expression-control, Property 2: Angle Range Validation**
    
    *For any* angle value outside the range [0, 180], the protocol encoder
    SHALL reject the input and raise a validation error.
    
    **Validates: Requirements 2.6**
    """

    @settings(max_examples=100)
    @given(
        invalid_angle=invalid_angle_strategy(),
        servo_index=st.integers(min_value=0, max_value=20)
    )
    def test_encoder_rejects_out_of_range_angles(self, invalid_angle: int, servo_index: int):
        """
        Property: encode() raises ValueError for any angle outside [0, 180].
        
        This verifies:
        - Requirement 2.6: Servo angle range is 0-180 degrees
        
        For any servo position and any invalid angle value (< 0 or > 180),
        the encoder must reject the input with a ValueError.
        """
        # Create a valid angles dict first
        angles = {name: 90 for name in ServoCommandProtocol.SERVO_ORDER}
        
        # Replace one servo's angle with the invalid value
        servo_name = ServoCommandProtocol.SERVO_ORDER[servo_index]
        angles[servo_name] = invalid_angle
        
        # Encoding should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            ServoCommandProtocol.encode(angles)
        
        # Verify error message mentions the angle is out of range
        assert "out of range" in str(exc_info.value).lower() or \
               str(invalid_angle) in str(exc_info.value)

    @settings(max_examples=100)
    @given(invalid_angle=invalid_angle_strategy())
    def test_validate_angle_rejects_invalid(self, invalid_angle: int):
        """
        Property: validate_angle() raises ValueError for any angle outside [0, 180].
        
        Direct test of the validation function to ensure it properly
        rejects all invalid angle values.
        """
        with pytest.raises(ValueError) as exc_info:
            ServoCommandProtocol.validate_angle(invalid_angle)
        
        assert "out of range" in str(exc_info.value).lower()

    @settings(max_examples=100)
    @given(valid_angle=st.integers(min_value=0, max_value=180))
    def test_validate_angle_accepts_valid(self, valid_angle: int):
        """
        Property: validate_angle() accepts all angles in [0, 180].
        
        Complementary test to ensure valid angles are accepted.
        """
        # Should not raise any exception
        ServoCommandProtocol.validate_angle(valid_angle)

    @settings(max_examples=100)
    @given(invalid_angle=invalid_angle_strategy())
    def test_decoder_rejects_out_of_range_angles(self, invalid_angle: int):
        """
        Property: decode() returns None for command strings with out-of-range angles.
        
        When a command string contains an angle outside [0, 180],
        the decoder should return None to indicate invalid input.
        """
        # Create a command string with an invalid angle
        valid_angles = [90] * 21
        valid_angles[0] = invalid_angle  # Replace first angle with invalid
        command = f"angles:{','.join(str(a) for a in valid_angles)}"
        
        # Decoding should return None for invalid angles
        result = ServoCommandProtocol.decode(command)
        assert result is None, \
            f"decode() should return None for out-of-range angle {invalid_angle}"

    @settings(max_examples=100)
    @given(
        num_invalid=st.integers(min_value=1, max_value=21),
        invalid_angle=invalid_angle_strategy()
    )
    def test_encoder_rejects_multiple_invalid_angles(self, num_invalid: int, invalid_angle: int):
        """
        Property: encode() rejects input even when multiple angles are invalid.
        
        The encoder should fail fast on the first invalid angle encountered.
        """
        # Create angles dict with multiple invalid values
        angles = {}
        for i, name in enumerate(ServoCommandProtocol.SERVO_ORDER):
            if i < num_invalid:
                angles[name] = invalid_angle
            else:
                angles[name] = 90
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            ServoCommandProtocol.encode(angles)
