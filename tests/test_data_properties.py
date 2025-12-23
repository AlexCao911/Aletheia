"""
Property-based tests for training data serialization.

These tests verify correctness properties of TrainingDataSample and
TrainingDataset serialization/deserialization using Hypothesis for
property-based testing.
"""

import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st
import numpy as np

from expression_control.data import TrainingDataSample, TrainingDataset, EXPRESSION_LABELS
from expression_control.features import FaceFeatures
from expression_control.protocol import ServoCommandProtocol


# Strategy to generate valid FaceFeatures
def face_features_strategy():
    """
    Generate a valid FaceFeatures instance.
    
    All ratio values are in [0, 1].
    Gaze values are in [-1, 1].
    Head pose angles are in [-180, 180] degrees.
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
        head_pitch=st.floats(min_value=-180.0, max_value=180.0),
        head_yaw=st.floats(min_value=-180.0, max_value=180.0),
        head_roll=st.floats(min_value=-180.0, max_value=180.0),
        timestamp=st.floats(min_value=0.0, max_value=1000.0),
        landmarks=st.none(),  # Don't include landmarks in serialization tests
    )


# Strategy to generate valid servo angles dictionary
def servo_angles_strategy():
    """
    Generate a valid dictionary of 21 servo angles.
    
    Each servo name from SERVO_ORDER maps to an integer in [0, 180].
    """
    return st.fixed_dictionaries({
        name: st.integers(min_value=0, max_value=180)
        for name in ServoCommandProtocol.SERVO_ORDER
    })


# Strategy to generate valid TrainingDataSample
def training_data_sample_strategy():
    """
    Generate a valid TrainingDataSample instance.
    
    Includes face features, servo angles, optional expression label,
    and optional video frame path.
    """
    return st.builds(
        TrainingDataSample,
        timestamp=st.floats(min_value=0.0, max_value=1000.0),
        face_features=face_features_strategy(),
        servo_angles=servo_angles_strategy(),
        expression_label=st.one_of(
            st.none(),
            st.sampled_from(list(EXPRESSION_LABELS))
        ),
        video_frame_path=st.one_of(
            st.none(),
            st.text(min_size=1, max_size=100).map(lambda s: f"frames/{s}.jpg")
        ),
    )


class TestTrainingDataRoundTrip:
    """
    **Feature: vision-expression-control, Property 3: Training Data Round-Trip Consistency**
    
    *For any* valid TrainingDataSample, serializing to JSON and then deserializing
    SHALL produce an equivalent TrainingDataSample with identical feature values
    and servo angles.
    
    **Validates: Requirements 3.6, 3.7**
    """

    @settings(max_examples=100)
    @given(sample=training_data_sample_strategy())
    def test_sample_json_round_trip(self, sample: TrainingDataSample):
        """
        Property: to_json() -> from_json() should equal original sample.
        
        This verifies:
        - Requirement 3.6: Data integrity validation against schema
        - Requirement 3.7: JSON schema for metadata serialization
        
        For any valid TrainingDataSample, serializing to JSON and deserializing
        back should produce an equivalent sample with identical values.
        """
        # Serialize to JSON
        json_str = sample.to_json()
        
        # Deserialize back
        loaded = TrainingDataSample.from_json(json_str)
        
        # Verify timestamp
        assert np.isclose(loaded.timestamp, sample.timestamp), \
            f"Timestamp mismatch: {loaded.timestamp} != {sample.timestamp}"
        
        # Verify face features (use numpy allclose for float comparison)
        original_features = sample.face_features.to_array()
        loaded_features = loaded.face_features.to_array()
        assert np.allclose(original_features, loaded_features, rtol=1e-6), \
            f"Face features mismatch:\nOriginal: {original_features}\nLoaded: {loaded_features}"
        
        # Verify servo angles (exact match for integers)
        assert loaded.servo_angles == sample.servo_angles, \
            f"Servo angles mismatch:\nOriginal: {sample.servo_angles}\nLoaded: {loaded.servo_angles}"
        
        # Verify expression label
        assert loaded.expression_label == sample.expression_label, \
            f"Expression label mismatch: {loaded.expression_label} != {sample.expression_label}"
        
        # Verify video frame path
        assert loaded.video_frame_path == sample.video_frame_path, \
            f"Video frame path mismatch: {loaded.video_frame_path} != {sample.video_frame_path}"

    @settings(max_examples=100)
    @given(sample=training_data_sample_strategy())
    def test_sample_dict_round_trip(self, sample: TrainingDataSample):
        """
        Property: to_dict() -> from_dict() should equal original sample.
        
        This verifies the dictionary serialization path, which is used
        internally by JSON serialization.
        """
        # Serialize to dict
        data_dict = sample.to_dict()
        
        # Deserialize back
        loaded = TrainingDataSample.from_dict(data_dict)
        
        # Verify all fields match
        assert np.isclose(loaded.timestamp, sample.timestamp)
        assert np.allclose(loaded.face_features.to_array(), sample.face_features.to_array(), rtol=1e-6)
        assert loaded.servo_angles == sample.servo_angles
        assert loaded.expression_label == sample.expression_label
        assert loaded.video_frame_path == sample.video_frame_path

    @settings(max_examples=100)
    @given(
        samples=st.lists(training_data_sample_strategy(), min_size=1, max_size=20),
        fps=st.floats(min_value=1.0, max_value=120.0),
    )
    def test_dataset_json_round_trip(self, samples, fps):
        """
        Property: TrainingDataset JSON round-trip preserves all samples.
        
        For any list of valid samples, creating a dataset, serializing to JSON,
        and deserializing back should preserve all samples with identical values.
        """
        # Create dataset
        dataset = TrainingDataset(samples=samples, fps=fps)
        
        # Serialize to JSON
        json_str = dataset.to_json()
        
        # Deserialize back
        loaded = TrainingDataset.from_json(json_str)
        
        # Verify metadata
        assert loaded.version == dataset.version
        assert np.isclose(loaded.fps, dataset.fps)
        assert loaded.total_samples == dataset.total_samples
        assert len(loaded.samples) == len(dataset.samples)
        
        # Verify each sample
        for i, (original, loaded_sample) in enumerate(zip(dataset.samples, loaded.samples)):
            assert np.isclose(loaded_sample.timestamp, original.timestamp), \
                f"Sample {i}: timestamp mismatch"
            assert np.allclose(
                loaded_sample.face_features.to_array(),
                original.face_features.to_array(),
                rtol=1e-6
            ), f"Sample {i}: face features mismatch"
            assert loaded_sample.servo_angles == original.servo_angles, \
                f"Sample {i}: servo angles mismatch"
            assert loaded_sample.expression_label == original.expression_label, \
                f"Sample {i}: expression label mismatch"

    @settings(max_examples=100)
    @given(sample=training_data_sample_strategy())
    def test_sample_preserves_servo_order(self, sample: TrainingDataSample):
        """
        Property: Serialization preserves the correct mapping of servo names to angles.
        
        After round-trip, each servo should map to its original angle value.
        """
        json_str = sample.to_json()
        loaded = TrainingDataSample.from_json(json_str)
        
        # Check each servo individually
        for servo_name in ServoCommandProtocol.SERVO_ORDER:
            assert servo_name in loaded.servo_angles, \
                f"Missing servo: {servo_name}"
            assert loaded.servo_angles[servo_name] == sample.servo_angles[servo_name], \
                f"Servo {servo_name}: expected {sample.servo_angles[servo_name]}, " \
                f"got {loaded.servo_angles[servo_name]}"

    @settings(max_examples=100)
    @given(sample=training_data_sample_strategy())
    def test_sample_preserves_feature_order(self, sample: TrainingDataSample):
        """
        Property: Serialization preserves the order and values of all 14 face features.
        
        After round-trip, each feature field should have its original value.
        """
        json_str = sample.to_json()
        loaded = TrainingDataSample.from_json(json_str)
        
        # Check each feature field individually
        original_ff = sample.face_features
        loaded_ff = loaded.face_features
        
        assert np.isclose(loaded_ff.left_eye_aspect_ratio, original_ff.left_eye_aspect_ratio)
        assert np.isclose(loaded_ff.right_eye_aspect_ratio, original_ff.right_eye_aspect_ratio)
        assert np.isclose(loaded_ff.eye_gaze_horizontal, original_ff.eye_gaze_horizontal)
        assert np.isclose(loaded_ff.eye_gaze_vertical, original_ff.eye_gaze_vertical)
        assert np.isclose(loaded_ff.left_eyebrow_height, original_ff.left_eyebrow_height)
        assert np.isclose(loaded_ff.right_eyebrow_height, original_ff.right_eyebrow_height)
        assert np.isclose(loaded_ff.eyebrow_furrow, original_ff.eyebrow_furrow)
        assert np.isclose(loaded_ff.mouth_open_ratio, original_ff.mouth_open_ratio)
        assert np.isclose(loaded_ff.mouth_width_ratio, original_ff.mouth_width_ratio)
        assert np.isclose(loaded_ff.lip_pucker, original_ff.lip_pucker)
        assert np.isclose(loaded_ff.smile_intensity, original_ff.smile_intensity)
        assert np.isclose(loaded_ff.head_pitch, original_ff.head_pitch)
        assert np.isclose(loaded_ff.head_yaw, original_ff.head_yaw)
        assert np.isclose(loaded_ff.head_roll, original_ff.head_roll)
