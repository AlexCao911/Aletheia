"""
Property-based tests for LiquidS4Model.

These tests verify correctness properties of the model output
using Hypothesis for property-based testing.
"""

import pytest
import torch
import numpy as np
from hypothesis import given, settings, assume
import hypothesis.strategies as st

from expression_control.models.liquid_s4 import LiquidS4Model
from expression_control.models.config import LNNS4Config


# Strategy to generate valid input feature vectors
def feature_vector_strategy():
    """
    Generate a valid 14-dimensional feature vector.
    
    Feature ranges based on FaceFeatures specification:
    - Eye ratios, eyebrow heights, mouth features: [0, 1]
    - Gaze: [-1, 1]
    - Head pose angles: [-90, 90] degrees
    
    For property testing, we allow a wider range to test model robustness.
    """
    return st.lists(
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        min_size=14,
        max_size=14
    )


# Strategy to generate batch sizes
def batch_size_strategy():
    """Generate valid batch sizes for testing."""
    return st.integers(min_value=1, max_value=8)


# Strategy to generate sequence lengths
def sequence_length_strategy():
    """Generate valid sequence lengths for testing."""
    return st.integers(min_value=1, max_value=32)


class TestModelOutputValidity:
    """
    **Feature: vision-expression-control, Property 5: Model Output Validity**
    
    *For any* valid input feature vector, the LNN-S4 model SHALL output
    exactly 21 values, each within the range [0, 180].
    
    **Validates: Requirements 4.4**
    
    This property ensures that:
    1. The model always outputs exactly 21 servo angles
    2. All output angles are within the valid servo range [0, 180]
    3. The output shape is correct for both single frame and sequence inputs
    4. The model produces finite values (no NaN or inf)
    """
    
    @pytest.fixture(scope="class")
    def model(self):
        """Create a model instance for testing."""
        config = LNNS4Config(
            input_dim=14,
            output_dim=21,
            hidden_dim=32,  # Smaller for faster testing
            state_dim=16,
            num_layers=2,
            liquid_units=32,  # Must be >= output_dim + 2 for ncps AutoNCP
            dropout=0.0,  # No dropout for deterministic testing
            min_angle=0.0,
            max_angle=180.0,
        )
        model = LiquidS4Model(config)
        model.eval()  # Set to evaluation mode
        return model
    
    @settings(max_examples=100)
    @given(features=feature_vector_strategy())
    def test_single_frame_output_shape_and_range(self, model, features):
        """
        Property: For any single feature vector, model outputs exactly 21 angles in [0, 180].
        
        This verifies:
        - Requirement 4.4: Model outputs 21 servo angle values in range 0-180
        
        For any valid 14-dimensional feature vector, the model should produce
        exactly 21 output values, each representing a servo angle in [0, 180].
        """
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, 14)
        
        # Forward pass
        with torch.no_grad():
            angles, s4_states, ltc_state = model(features_tensor)
        
        # Convert to numpy for assertions
        angles_np = angles.cpu().numpy()
        
        # Check shape: should be (batch=1, output_dim=21)
        assert angles_np.shape == (1, 21), \
            f"Expected shape (1, 21), got {angles_np.shape}"
        
        # Check all angles are in valid range [0, 180]
        assert np.all(angles_np >= 0.0), \
            f"Found angles < 0: {angles_np[angles_np < 0.0]}"
        assert np.all(angles_np <= 180.0), \
            f"Found angles > 180: {angles_np[angles_np > 180.0]}"
        
        # Check all values are finite (not NaN or inf)
        assert np.all(np.isfinite(angles_np)), \
            f"Found non-finite values in output: {angles_np}"
    
    @settings(max_examples=100)
    @given(
        features=feature_vector_strategy(),
        batch_size=batch_size_strategy()
    )
    def test_batched_single_frame_output(self, model, features, batch_size):
        """
        Property: For any batch of feature vectors, model outputs correct shape and range.
        
        When processing multiple frames in a batch, each frame should produce
        exactly 21 angles in [0, 180].
        """
        # Create batch by repeating features
        features_batch = torch.tensor([features] * batch_size, dtype=torch.float32)  # (batch, 14)
        
        # Forward pass
        with torch.no_grad():
            angles, s4_states, ltc_state = model(features_batch)
        
        # Convert to numpy
        angles_np = angles.cpu().numpy()
        
        # Check shape: should be (batch_size, 21)
        assert angles_np.shape == (batch_size, 21), \
            f"Expected shape ({batch_size}, 21), got {angles_np.shape}"
        
        # Check all angles are in valid range
        assert np.all(angles_np >= 0.0) and np.all(angles_np <= 180.0), \
            f"Found angles outside [0, 180] range"
        
        # Check all values are finite
        assert np.all(np.isfinite(angles_np)), \
            f"Found non-finite values in output"
    
    @settings(max_examples=100)
    @given(
        features=feature_vector_strategy(),
        batch_size=batch_size_strategy(),
        seq_len=sequence_length_strategy()
    )
    def test_sequence_output_shape_and_range(self, model, features, batch_size, seq_len):
        """
        Property: For any sequence of feature vectors, model outputs correct shape and range.
        
        When processing temporal sequences, the model should output angles for
        each timestep, all within [0, 180].
        """
        # Create sequence batch
        features_seq = torch.tensor(
            [[[features[i % 14] for i in range(14)] for _ in range(seq_len)] for _ in range(batch_size)],
            dtype=torch.float32
        )  # (batch, seq_len, 14)
        
        # Forward pass
        with torch.no_grad():
            angles, s4_states, ltc_state = model(features_seq)
        
        # Convert to numpy
        angles_np = angles.cpu().numpy()
        
        # Check shape: should be (batch_size, seq_len, 21)
        assert angles_np.shape == (batch_size, seq_len, 21), \
            f"Expected shape ({batch_size}, {seq_len}, 21), got {angles_np.shape}"
        
        # Check all angles are in valid range
        assert np.all(angles_np >= 0.0) and np.all(angles_np <= 180.0), \
            f"Found angles outside [0, 180] range"
        
        # Check all values are finite
        assert np.all(np.isfinite(angles_np)), \
            f"Found non-finite values in output"
    
    @settings(max_examples=100)
    @given(features=feature_vector_strategy())
    def test_step_method_output_validity(self, model, features):
        """
        Property: The step() method for single-step inference produces valid output.
        
        The step() method is used for real-time inference and should produce
        the same output validity guarantees as forward().
        """
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, 14)
        
        # Single step forward
        with torch.no_grad():
            angles, s4_states, ltc_state = model.step(features_tensor)
        
        # Convert to numpy
        angles_np = angles.cpu().numpy()
        
        # Check shape: should be (batch=1, 21)
        assert angles_np.shape == (1, 21), \
            f"Expected shape (1, 21), got {angles_np.shape}"
        
        # Check all angles are in valid range
        assert np.all(angles_np >= 0.0) and np.all(angles_np <= 180.0), \
            f"Found angles outside [0, 180] range"
        
        # Check all values are finite
        assert np.all(np.isfinite(angles_np)), \
            f"Found non-finite values in output"
    
    @settings(max_examples=100)
    @given(
        features_list=st.lists(feature_vector_strategy(), min_size=2, max_size=10)
    )
    def test_stateful_inference_output_validity(self, model, features_list):
        """
        Property: Stateful inference maintains output validity across multiple steps.
        
        When processing a sequence of frames with state management, all outputs
        should remain valid.
        """
        s4_states = None
        ltc_state = None
        
        for features in features_list:
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                angles, s4_states, ltc_state = model.step(features_tensor, s4_states, ltc_state)
            
            angles_np = angles.cpu().numpy()
            
            # Check shape
            assert angles_np.shape == (1, 21)
            
            # Check range
            assert np.all(angles_np >= 0.0) and np.all(angles_np <= 180.0), \
                f"Found angles outside [0, 180] range at step"
            
            # Check finite
            assert np.all(np.isfinite(angles_np))
    
    @settings(max_examples=100)
    @given(features=feature_vector_strategy())
    def test_output_exactly_21_servos(self, model, features):
        """
        Property: Model always outputs exactly 21 servo angles, no more, no less.
        
        This is critical for the servo control system which expects exactly
        21 angles corresponding to the 21 servos (11 mouth + 6 eyes + 4 brows).
        """
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            angles, _, _ = model(features_tensor)
        
        # Check last dimension is exactly 21
        assert angles.shape[-1] == 21, \
            f"Expected exactly 21 servo outputs, got {angles.shape[-1]}"
    
    @settings(max_examples=100)
    @given(
        features=feature_vector_strategy(),
        batch_size=batch_size_strategy()
    )
    def test_output_deterministic_in_eval_mode(self, model, features, batch_size):
        """
        Property: In eval mode, same input produces same output (deterministic).
        
        This ensures reproducibility and predictable behavior during inference.
        """
        # Create batch
        features_batch = torch.tensor([features] * batch_size, dtype=torch.float32)
        
        # Run twice
        with torch.no_grad():
            angles1, _, _ = model(features_batch)
            angles2, _, _ = model(features_batch)
        
        # Should be identical (within floating point tolerance)
        assert torch.allclose(angles1, angles2, rtol=1e-5, atol=1e-6), \
            f"Model output is not deterministic in eval mode"
    
    @settings(max_examples=100)
    @given(features=feature_vector_strategy())
    def test_output_respects_config_angle_bounds(self, features):
        """
        Property: Model respects the min_angle and max_angle from config.
        
        The output should be scaled according to the configured angle range.
        """
        # Test with custom angle range
        config = LNNS4Config(
            input_dim=14,
            output_dim=21,
            hidden_dim=32,
            state_dim=16,
            num_layers=2,
            liquid_units=32,  # Must be >= output_dim + 2 for ncps AutoNCP
            dropout=0.0,
            min_angle=10.0,  # Custom min
            max_angle=170.0,  # Custom max
        )
        model = LiquidS4Model(config)
        model.eval()
        
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            angles, _, _ = model(features_tensor)
        
        angles_np = angles.cpu().numpy()
        
        # Check angles are within custom range
        assert np.all(angles_np >= config.min_angle), \
            f"Found angles < {config.min_angle}: {angles_np[angles_np < config.min_angle]}"
        assert np.all(angles_np <= config.max_angle), \
            f"Found angles > {config.max_angle}: {angles_np[angles_np > config.max_angle]}"
    
    def test_output_validity_with_neutral_features(self, model):
        """
        Unit test: Model produces valid output for neutral face features.
        
        This is a specific example test to complement the property tests.
        """
        # Neutral features: all zeros except head pose at center
        neutral_features = torch.zeros(1, 14, dtype=torch.float32)
        
        with torch.no_grad():
            angles, _, _ = model(neutral_features)
        
        angles_np = angles.cpu().numpy()
        
        # Check shape
        assert angles_np.shape == (1, 21)
        
        # Check range
        assert np.all(angles_np >= 0.0) and np.all(angles_np <= 180.0)
        
        # Check finite
        assert np.all(np.isfinite(angles_np))
    
    def test_output_validity_with_extreme_features(self, model):
        """
        Unit test: Model produces valid output for extreme feature values.
        
        Tests edge cases with maximum and minimum feature values.
        """
        # Test with all maximum values
        max_features = torch.ones(1, 14, dtype=torch.float32)
        
        with torch.no_grad():
            angles_max, _, _ = model(max_features)
        
        angles_max_np = angles_max.cpu().numpy()
        assert np.all(angles_max_np >= 0.0) and np.all(angles_max_np <= 180.0)
        assert np.all(np.isfinite(angles_max_np))
        
        # Test with all minimum values
        min_features = torch.full((1, 14), -1.0, dtype=torch.float32)
        
        with torch.no_grad():
            angles_min, _, _ = model(min_features)
        
        angles_min_np = angles_min.cpu().numpy()
        assert np.all(angles_min_np >= 0.0) and np.all(angles_min_np <= 180.0)
        assert np.all(np.isfinite(angles_min_np))
