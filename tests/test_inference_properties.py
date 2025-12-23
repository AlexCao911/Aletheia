"""
Property-based tests for InferenceEngine face detection fallback.

These tests verify correctness properties of the inference engine's
face detection fallback behavior using Hypothesis for property-based testing.
"""

import pytest
import numpy as np
from hypothesis import given, settings, assume
import hypothesis.strategies as st
from unittest.mock import Mock, patch
import time

from expression_control.features import FaceFeatures
from expression_control.inference import InferenceEngine, InferenceConfig, FallbackMapper
from expression_control.protocol import ServoCommandProtocol


# Strategy to generate valid FaceFeatures objects
def face_features_strategy():
    """
    Generate valid FaceFeatures objects with all 14 required fields.
    """
    return st.builds(
        FaceFeatures,
        left_eye_aspect_ratio=st.floats(min_value=0.0, max_value=1.0),
        right_eye_aspect_ratio=st.floats(min_value=0.0, max_value=1.0),
        eye_gaze_horizontal=st.floats(min_value=-1.0, max_value=1.0),
        eye_gaze_vertical=st.floats(min_value=-1.0, max_value=1.0),
        left_eyebrow_height=st.floats(min_value=0.0, max_value=1.0),
        right_eyebrow_height=st.floats(min_value=0.0, max_value=1.0),
        eyebrow_furrow=st.floats(min_value=0.0, max_value=1.0),
        mouth_open_ratio=st.floats(min_value=0.0, max_value=1.0),
        mouth_width_ratio=st.floats(min_value=0.0, max_value=1.0),
        lip_pucker=st.floats(min_value=0.0, max_value=1.0),
        smile_intensity=st.floats(min_value=0.0, max_value=1.0),
        head_pitch=st.floats(min_value=-90.0, max_value=90.0),
        head_yaw=st.floats(min_value=-90.0, max_value=90.0),
        head_roll=st.floats(min_value=-90.0, max_value=90.0),
        landmarks=st.none(),
        timestamp=st.floats(min_value=0.0, max_value=1e10),
    )


# Strategy to generate valid servo angle arrays
def servo_angles_strategy():
    """Generate valid servo angle arrays (21 values in [0, 180])."""
    return st.lists(
        st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        min_size=21,
        max_size=21
    ).map(lambda x: np.array(x, dtype=np.float64))


def create_mock_inference_engine():
    """Create an InferenceEngine with mocked components for testing."""
    config = InferenceConfig(
        model_path=None,  # No model, use fallback mode
        camera_id=0,
        serial_port="/dev/null",
        smoothing_alpha=1.0,  # No smoothing for deterministic testing
        face_timeout_ms=500.0,
        fallback_enabled=True,
    )
    engine = InferenceEngine(config)
    
    # Mock the extractor to control face detection results
    engine._extractor = Mock()
    
    # Mock the smoother to pass through values unchanged
    engine._smoother = Mock()
    engine._smoother.smooth = lambda x: x
    engine._smoother.reset = Mock()
    
    # Initialize fallback mapper
    engine._fallback_mapper = FallbackMapper(sensitivity=1.0)
    
    # Initialize last valid angles to neutral
    engine._last_valid_angles = np.array([
        config.neutral_angles[name]
        for name in ServoCommandProtocol.SERVO_ORDER
    ], dtype=np.float64)
    
    # Set initial face time to now
    engine._last_face_time = time.time()
    
    engine._is_initialized = True
    
    return engine


class TestFaceDetectionFallback:
    """
    **Feature: vision-expression-control, Property 6: Face Detection Fallback**
    
    *For any* sequence of frames where face detection fails, the inference system
    SHALL output the same servo angles as the last successful detection.
    
    **Validates: Requirements 4.8**
    
    This property ensures that:
    1. When face detection fails, the system maintains the last valid servo positions
    2. The fallback behavior is consistent across multiple failed frames
    3. The system correctly tracks the last successful detection
    """
    
    @settings(max_examples=100)
    @given(
        initial_features=face_features_strategy(),
        num_failed_frames=st.integers(min_value=1, max_value=10)
    )
    def test_fallback_maintains_last_valid_angles(
        self, initial_features, num_failed_frames
    ):
        """
        Property: When face detection fails, output equals last successful detection.
        
        For any valid face features followed by any number of failed detections
        (within timeout), the system should output the same angles as the last
        successful detection.
        """
        engine = create_mock_inference_engine()
        
        # First, process a frame with successful face detection
        engine._extractor.extract = Mock(return_value=initial_features)
        
        # Create a dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process frame with face detected
        angles_with_face, info_with_face = engine.process_frame(dummy_frame)
        
        assert info_with_face['face_detected'] is True
        
        # Store the angles from successful detection
        last_valid_angles = angles_with_face.copy()
        
        # Now simulate face detection failures (within timeout)
        engine._extractor.extract = Mock(return_value=None)
        
        # Keep the face time recent to avoid timeout
        engine._last_face_time = time.time()
        
        for i in range(num_failed_frames):
            # Process frame with no face detected
            angles_no_face, info_no_face = engine.process_frame(dummy_frame)
            
            assert info_no_face['face_detected'] is False
            assert info_no_face['using_neutral'] is False, \
                f"Should not use neutral within timeout (frame {i})"
            
            # The output should match the last valid angles
            np.testing.assert_array_almost_equal(
                angles_no_face, last_valid_angles,
                decimal=5,
                err_msg=f"Fallback angles differ from last valid at frame {i}"
            )
    
    @settings(max_examples=100)
    @given(
        features_sequence=st.lists(
            face_features_strategy(),
            min_size=2,
            max_size=5
        )
    )
    def test_fallback_tracks_most_recent_detection(
        self, features_sequence
    ):
        """
        Property: Fallback always uses the most recent successful detection.
        
        When processing a sequence of successful detections followed by failures,
        the fallback should use the angles from the most recent success.
        """
        engine = create_mock_inference_engine()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process each successful detection
        last_angles = None
        for features in features_sequence:
            engine._extractor.extract = Mock(return_value=features)
            angles, info = engine.process_frame(dummy_frame)
            assert info['face_detected'] is True
            last_angles = angles.copy()
        
        # Now fail detection
        engine._extractor.extract = Mock(return_value=None)
        engine._last_face_time = time.time()  # Keep within timeout
        
        # Process failed frame
        fallback_angles, info = engine.process_frame(dummy_frame)
        
        assert info['face_detected'] is False
        
        # Should match the most recent successful detection
        np.testing.assert_array_almost_equal(
            fallback_angles, last_angles,
            decimal=5,
            err_msg="Fallback should use most recent successful detection"
        )
    
    @settings(max_examples=100)
    @given(initial_features=face_features_strategy())
    def test_fallback_consistent_across_multiple_failures(
        self, initial_features
    ):
        """
        Property: Fallback output is consistent across consecutive failures.
        
        When face detection fails multiple times in a row (within timeout),
        each failure should produce the same output angles.
        """
        engine = create_mock_inference_engine()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # First, get a successful detection
        engine._extractor.extract = Mock(return_value=initial_features)
        success_angles, _ = engine.process_frame(dummy_frame)
        
        # Now fail detection multiple times
        engine._extractor.extract = Mock(return_value=None)
        
        fallback_outputs = []
        for _ in range(5):
            engine._last_face_time = time.time()  # Keep within timeout
            angles, info = engine.process_frame(dummy_frame)
            assert info['face_detected'] is False
            fallback_outputs.append(angles.copy())
        
        # All fallback outputs should be identical
        for i, output in enumerate(fallback_outputs[1:], start=1):
            np.testing.assert_array_almost_equal(
                output, fallback_outputs[0],
                decimal=5,
                err_msg=f"Fallback output {i} differs from first fallback"
            )
    
    @settings(max_examples=100)
    @given(
        initial_features=face_features_strategy(),
        second_features=face_features_strategy()
    )
    def test_fallback_updates_after_new_detection(
        self, initial_features, second_features
    ):
        """
        Property: Fallback updates when a new face is detected.
        
        After a period of failed detections, when a new face is detected,
        subsequent failures should use the new detection's angles.
        """
        engine = create_mock_inference_engine()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # First successful detection
        engine._extractor.extract = Mock(return_value=initial_features)
        first_angles, _ = engine.process_frame(dummy_frame)
        
        # Fail detection
        engine._extractor.extract = Mock(return_value=None)
        engine._last_face_time = time.time()
        fallback_angles_1, _ = engine.process_frame(dummy_frame)
        
        # Verify fallback uses first detection
        np.testing.assert_array_almost_equal(
            fallback_angles_1, first_angles,
            decimal=5
        )
        
        # New successful detection
        engine._extractor.extract = Mock(return_value=second_features)
        second_angles, _ = engine.process_frame(dummy_frame)
        
        # Fail detection again
        engine._extractor.extract = Mock(return_value=None)
        engine._last_face_time = time.time()
        fallback_angles_2, _ = engine.process_frame(dummy_frame)
        
        # Verify fallback now uses second detection
        np.testing.assert_array_almost_equal(
            fallback_angles_2, second_angles,
            decimal=5,
            err_msg="Fallback should update to use new detection"
        )
    
    def test_fallback_transitions_to_neutral_after_timeout(self):
        """
        Unit test: After timeout, system transitions to neutral position.
        
        This verifies Requirement 4.8 combined with 6.3: when face detection
        fails for more than 500ms, transition to neutral.
        """
        engine = create_mock_inference_engine()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Get initial successful detection
        initial_features = FaceFeatures.neutral()
        initial_features.smile_intensity = 0.8  # Non-neutral expression
        engine._extractor.extract = Mock(return_value=initial_features)
        success_angles, _ = engine.process_frame(dummy_frame)
        
        # Fail detection with timeout exceeded
        engine._extractor.extract = Mock(return_value=None)
        engine._last_face_time = time.time() - 1.0  # 1 second ago (> 500ms timeout)
        
        timeout_angles, info = engine.process_frame(dummy_frame)
        
        assert info['face_detected'] is False
        assert info['using_neutral'] is True
        
        # Should be neutral angles, not the last valid angles
        neutral_angles = np.array([
            engine.config.neutral_angles[name]
            for name in ServoCommandProtocol.SERVO_ORDER
        ], dtype=np.float64)
        
        np.testing.assert_array_almost_equal(
            timeout_angles, neutral_angles,
            decimal=5,
            err_msg="Should transition to neutral after timeout"
        )
    
    @settings(max_examples=100)
    @given(initial_angles=servo_angles_strategy())
    def test_fallback_preserves_exact_angle_values(
        self, initial_angles
    ):
        """
        Property: Fallback preserves exact angle values without modification.
        
        The fallback mechanism should return the exact same angle values
        as the last successful detection, not modified or interpolated values.
        """
        engine = create_mock_inference_engine()
        
        # Directly set the last valid angles
        engine._last_valid_angles = initial_angles.copy()
        engine._last_face_time = time.time()
        
        # Fail detection
        engine._extractor.extract = Mock(return_value=None)
        
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fallback_angles, info = engine.process_frame(dummy_frame)
        
        assert info['face_detected'] is False
        
        # Should be exactly the same values
        np.testing.assert_array_almost_equal(
            fallback_angles, initial_angles,
            decimal=10,  # High precision to verify exact preservation
            err_msg="Fallback should preserve exact angle values"
        )
