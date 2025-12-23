"""
Property-based tests for TemporalSmoother.

These tests verify correctness properties of the EMA temporal smoother
using Hypothesis for property-based testing.
"""

import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st
import numpy as np

from expression_control.smoother import TemporalSmoother


class TestEMASmootherBounds:
    """
    **Feature: vision-expression-control, Property 11: EMA Smoothing Bounds**
    
    *For any* sequence of input angles, the EMA smoother output SHALL always
    be bounded between the minimum and maximum of the input history within
    the smoothing window.
    
    **Validates: Requirements 6.2**
    """

    @settings(max_examples=100)
    @given(
        angle_sequence=st.lists(
            st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=100
        ),
        alpha=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        num_servos=st.integers(min_value=1, max_value=21)
    )
    def test_ema_output_bounded_by_input_history(self, angle_sequence, alpha, num_servos):
        """
        Property: EMA smoother output is always bounded by min/max of input history.
        
        This verifies:
        - Requirement 6.2: Temporal smoothing using exponential moving average
        
        For any sequence of input angles and any valid alpha value, the smoothed
        output at each timestep should be bounded by the minimum and maximum
        of all inputs seen so far (including the current input).
        
        This is a fundamental property of EMA: the smoothed value is a weighted
        average of current and past inputs, so it cannot exceed their bounds.
        """
        smoother = TemporalSmoother(alpha=alpha, num_servos=num_servos)
        
        # Create angle arrays for each timestep (replicate single value across servos)
        for i, angle_value in enumerate(angle_sequence):
            angles = np.full(num_servos, angle_value, dtype=np.float64)
            smoothed = smoother.smooth(angles)
            
            # Get the history of inputs up to and including current timestep
            input_history = angle_sequence[:i+1]
            min_input = min(input_history)
            max_input = max(input_history)
            
            # All smoothed values should be within [min_input, max_input]
            # Use tolerance for floating-point comparison
            for servo_idx in range(num_servos):
                # Allow small floating-point tolerance
                tolerance = 1e-10
                assert (min_input - tolerance) <= smoothed[servo_idx] <= (max_input + tolerance), \
                    f"At timestep {i}, servo {servo_idx}: smoothed value {smoothed[servo_idx]} " \
                    f"is outside bounds [{min_input}, {max_input}]. " \
                    f"Input history: {input_history[:min(10, len(input_history))]}"

    @settings(max_examples=100)
    @given(
        angle_sequences=st.lists(
            st.lists(
                st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False),
                min_size=21,
                max_size=21
            ),
            min_size=1,
            max_size=50
        ),
        alpha=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    def test_ema_per_servo_bounds(self, angle_sequences, alpha):
        """
        Property: Each servo's smoothed output is bounded by its own input history.
        
        When smoothing multiple servos independently, each servo's smoothed
        value should be bounded by the min/max of that specific servo's
        input history, not the global min/max across all servos.
        """
        smoother = TemporalSmoother(alpha=alpha, num_servos=21)
        
        # Track history for each servo
        servo_histories = [[] for _ in range(21)]
        
        for timestep, angles_list in enumerate(angle_sequences):
            angles = np.array(angles_list, dtype=np.float64)
            smoothed = smoother.smooth(angles)
            
            # Update histories and check bounds for each servo
            for servo_idx in range(21):
                servo_histories[servo_idx].append(angles_list[servo_idx])
                
                min_servo = min(servo_histories[servo_idx])
                max_servo = max(servo_histories[servo_idx])
                
                # Allow small floating-point tolerance
                tolerance = 1e-10
                assert (min_servo - tolerance) <= smoothed[servo_idx] <= (max_servo + tolerance), \
                    f"Timestep {timestep}, servo {servo_idx}: " \
                    f"smoothed {smoothed[servo_idx]:.2f} outside bounds " \
                    f"[{min_servo:.2f}, {max_servo:.2f}]"

    @settings(max_examples=100)
    @given(
        constant_value=st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        sequence_length=st.integers(min_value=1, max_value=50),
        alpha=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    def test_ema_converges_to_constant_input(self, constant_value, sequence_length, alpha):
        """
        Property: EMA converges to constant input value.
        
        When the input is constant, the EMA output should converge to that
        constant value. This is a special case of the bounds property where
        min = max = constant.
        """
        smoother = TemporalSmoother(alpha=alpha, num_servos=1)
        
        angles = np.array([constant_value], dtype=np.float64)
        
        for _ in range(sequence_length):
            smoothed = smoother.smooth(angles)
            
            # Smoothed value should equal the constant (within floating point tolerance)
            # After enough iterations, it should converge
            assert np.isclose(smoothed[0], constant_value, rtol=1e-10, atol=1e-10) or \
                   abs(smoothed[0] - constant_value) < abs(angles[0] - constant_value), \
                   f"EMA not converging: smoothed={smoothed[0]}, constant={constant_value}"

    @settings(max_examples=100)
    @given(
        initial_value=st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        alpha=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    def test_ema_first_output_equals_first_input(self, initial_value, alpha):
        """
        Property: First EMA output equals first input.
        
        On the first call to smooth(), the output should equal the input
        since there's no previous state to blend with.
        """
        smoother = TemporalSmoother(alpha=alpha, num_servos=1)
        
        angles = np.array([initial_value], dtype=np.float64)
        smoothed = smoother.smooth(angles)
        
        assert np.isclose(smoothed[0], initial_value, rtol=1e-10), \
            f"First output {smoothed[0]} should equal first input {initial_value}"

    @settings(max_examples=100)
    @given(
        angle_sequence=st.lists(
            st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=50
        ),
        alpha=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)
    )
    def test_ema_monotonic_approach(self, angle_sequence, alpha):
        """
        Property: EMA monotonically approaches target when input is constant.
        
        If the input changes to a new constant value, the EMA should
        monotonically approach that value (no oscillation).
        """
        # Assume we have at least 2 different values to test transition
        assume(len(set(angle_sequence)) >= 2)
        
        smoother = TemporalSmoother(alpha=alpha, num_servos=1)
        
        # Initialize with first value
        first_angles = np.array([angle_sequence[0]], dtype=np.float64)
        smoother.smooth(first_angles)
        
        # Now switch to second value and track convergence
        target_value = angle_sequence[1]
        target_angles = np.array([target_value], dtype=np.float64)
        
        previous_smoothed = smoother.current_state[0]
        
        # Apply the same target value multiple times
        for _ in range(20):
            smoothed = smoother.smooth(target_angles)
            current_smoothed = smoothed[0]
            
            # Check monotonic approach
            if target_value > previous_smoothed:
                # Should be increasing towards target
                assert current_smoothed >= previous_smoothed or \
                       np.isclose(current_smoothed, target_value, rtol=1e-10), \
                       f"Not monotonically increasing: {previous_smoothed} -> {current_smoothed} (target: {target_value})"
            elif target_value < previous_smoothed:
                # Should be decreasing towards target
                assert current_smoothed <= previous_smoothed or \
                       np.isclose(current_smoothed, target_value, rtol=1e-10), \
                       f"Not monotonically decreasing: {previous_smoothed} -> {current_smoothed} (target: {target_value})"
            
            # Should always be bounded
            min_val = min(angle_sequence[0], target_value)
            max_val = max(angle_sequence[0], target_value)
            tolerance = 1e-10
            assert (min_val - tolerance) <= current_smoothed <= (max_val + tolerance), \
                f"Smoothed value {current_smoothed} outside bounds [{min_val}, {max_val}]"
            
            previous_smoothed = current_smoothed
            
            # Stop if converged
            if np.isclose(current_smoothed, target_value, rtol=1e-6):
                break

    @settings(max_examples=100)
    @given(
        angle_sequence=st.lists(
            st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=50
        ),
        alpha=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    def test_ema_reset_clears_history(self, angle_sequence, alpha):
        """
        Property: Reset clears smoother state and next output equals input.
        
        After calling reset(), the smoother should behave as if it's
        processing the first frame again.
        """
        smoother = TemporalSmoother(alpha=alpha, num_servos=1)
        
        # Process some angles
        for angle_value in angle_sequence[:3]:
            angles = np.array([angle_value], dtype=np.float64)
            smoother.smooth(angles)
        
        # Reset
        smoother.reset()
        
        # Next output should equal input (like first frame)
        new_angle = angle_sequence[-1]
        new_angles = np.array([new_angle], dtype=np.float64)
        smoothed = smoother.smooth(new_angles)
        
        assert np.isclose(smoothed[0], new_angle, rtol=1e-10), \
            f"After reset, output {smoothed[0]} should equal input {new_angle}"
