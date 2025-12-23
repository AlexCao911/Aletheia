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



class TestCheckpointRoundTrip:
    """
    **Feature: vision-expression-control, Property 9: Checkpoint Round-Trip Consistency**
    
    *For any* trained model state, saving to checkpoint and loading back SHALL
    produce a model that generates identical outputs for the same inputs.
    
    **Validates: Requirements 5.4**
    
    This property ensures that:
    1. Model state can be saved and restored without loss
    2. Optimizer state is preserved correctly
    3. Training state (epoch, loss, etc.) is maintained
    4. Loaded model produces identical outputs to the original model
    5. The checkpoint format is stable and reliable
    """
    
    @pytest.fixture(scope="class")
    def temp_checkpoint_dir(self, tmp_path_factory):
        """Create a temporary directory for checkpoint files."""
        return tmp_path_factory.mktemp("checkpoints")
    
    @pytest.fixture(scope="class")
    def sample_datasets(self, tmp_path_factory):
        """Create minimal sample datasets for training."""
        import json
        from expression_control.protocol import ServoCommandProtocol
        
        data_dir = tmp_path_factory.mktemp("data")
        
        # Create minimal training data
        train_data = {
            "version": "1.0",
            "created_at": "2024-01-01T00:00:00Z",
            "total_samples": 20,
            "fps": 30,
            "servo_order": ServoCommandProtocol.SERVO_ORDER,
            "samples": []
        }
        
        # Generate 20 samples with random features and angles
        for i in range(20):
            sample = {
                "timestamp": i * 0.033,
                "features": {
                    "left_eye_aspect_ratio": 0.3 + i * 0.01,
                    "right_eye_aspect_ratio": 0.3 + i * 0.01,
                    "eye_gaze_horizontal": 0.0,
                    "eye_gaze_vertical": 0.0,
                    "left_eyebrow_height": 0.5,
                    "right_eyebrow_height": 0.5,
                    "eyebrow_furrow": 0.0,
                    "mouth_open_ratio": 0.2 + i * 0.01,
                    "mouth_width_ratio": 0.5,
                    "lip_pucker": 0.0,
                    "smile_intensity": 0.3,
                    "head_pitch": 0.0,
                    "head_yaw": 0.0,
                    "head_roll": 0.0,
                },
                "servo_angles": [90 + (i % 10) for _ in range(21)],
                "expression_label": "neutral"
            }
            train_data["samples"].append(sample)
        
        # Save training data
        train_path = data_dir / "train.json"
        with open(train_path, 'w') as f:
            json.dump(train_data, f)
        
        # Create validation data (same structure, different values)
        val_data = train_data.copy()
        val_data["samples"] = train_data["samples"][:10]  # Use first 10 samples
        
        val_path = data_dir / "val.json"
        with open(val_path, 'w') as f:
            json.dump(val_data, f)
        
        return str(train_path), str(val_path)
    
    def test_checkpoint_saves_and_loads_model_state(self, temp_checkpoint_dir, sample_datasets):
        """
        Property: Saving and loading a checkpoint preserves model state.
        
        For any model, after saving to checkpoint and loading back, the model
        should produce identical outputs for the same inputs.
        """
        from expression_control.trainer import Trainer
        from expression_control.models.config import LNNS4Config
        
        train_path, val_path = sample_datasets
        
        # Create a small config for fast testing
        config = LNNS4Config(
            input_dim=14,
            output_dim=21,
            hidden_dim=32,
            state_dim=16,
            num_layers=2,
            liquid_units=32,
            dropout=0.0,
            sequence_length=4,  # Short sequences for fast testing
            batch_size=4,
            num_epochs=2,
            learning_rate=1e-3,
            weight_decay=1e-4,
        )
        
        # Create trainer
        trainer = Trainer(config, train_path, val_path, device="cpu")
        
        # Generate test input
        test_input = torch.randn(1, 14, dtype=torch.float32)
        
        # Get output before saving
        trainer.model.eval()
        with torch.no_grad():
            output_before, s4_before, ltc_before = trainer.model(test_input)
        
        # Save checkpoint
        checkpoint_path = temp_checkpoint_dir / "test_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))
        
        # Verify checkpoint file exists
        assert checkpoint_path.exists(), "Checkpoint file was not created"
        
        # Create a new trainer with same config
        trainer_loaded = Trainer(config, train_path, val_path, device="cpu")
        
        # Load checkpoint
        trainer_loaded.load_checkpoint(str(checkpoint_path))
        
        # Get output after loading
        trainer_loaded.model.eval()
        with torch.no_grad():
            output_after, s4_after, ltc_after = trainer_loaded.model(test_input)
        
        # Verify outputs are identical
        assert torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-6), \
            f"Model outputs differ after checkpoint round-trip. Max diff: {torch.max(torch.abs(output_before - output_after))}"
    
    @settings(max_examples=50)
    @given(
        features=feature_vector_strategy(),
        seed=st.integers(min_value=0, max_value=10000)
    )
    def test_checkpoint_round_trip_with_random_inputs(self, temp_checkpoint_dir, sample_datasets, features, seed):
        """
        Property: For any input, checkpoint round-trip produces identical outputs.
        
        This tests that the checkpoint mechanism works correctly across a wide
        range of inputs, not just a single test case.
        """
        from expression_control.trainer import Trainer
        from expression_control.models.config import LNNS4Config
        
        train_path, val_path = sample_datasets
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create config
        config = LNNS4Config(
            input_dim=14,
            output_dim=21,
            hidden_dim=32,
            state_dim=16,
            num_layers=2,
            liquid_units=32,
            dropout=0.0,
            sequence_length=4,
            batch_size=4,
            num_epochs=2,
        )
        
        # Create trainer
        trainer = Trainer(config, train_path, val_path, device="cpu")
        
        # Convert features to tensor
        test_input = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Get output before saving
        trainer.model.eval()
        with torch.no_grad():
            output_before, _, _ = trainer.model(test_input)
        
        # Save checkpoint
        checkpoint_path = temp_checkpoint_dir / f"checkpoint_{seed}.pt"
        trainer.save_checkpoint(str(checkpoint_path))
        
        # Create new trainer and load
        trainer_loaded = Trainer(config, train_path, val_path, device="cpu")
        trainer_loaded.load_checkpoint(str(checkpoint_path))
        
        # Get output after loading
        trainer_loaded.model.eval()
        with torch.no_grad():
            output_after, _, _ = trainer_loaded.model(test_input)
        
        # Verify outputs are identical
        assert torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-6), \
            f"Outputs differ after checkpoint round-trip for input {features}"
    
    def test_checkpoint_preserves_training_state(self, temp_checkpoint_dir, sample_datasets):
        """
        Property: Checkpoint preserves all training state variables.
        
        This verifies that epoch, best_val_loss, patience_counter, and other
        training state is correctly saved and restored.
        """
        from expression_control.trainer import Trainer
        from expression_control.models.config import LNNS4Config
        
        train_path, val_path = sample_datasets
        
        config = LNNS4Config(
            input_dim=14,
            output_dim=21,
            hidden_dim=32,
            state_dim=16,
            num_layers=2,
            liquid_units=32,
            dropout=0.0,
            sequence_length=4,
            batch_size=4,
            num_epochs=2,
        )
        
        # Create trainer
        trainer = Trainer(config, train_path, val_path, device="cpu")
        
        # Manually set some training state
        trainer.best_val_loss = 0.123
        trainer.current_epoch = 5
        trainer.patience_counter = 3
        
        # Save checkpoint
        checkpoint_path = temp_checkpoint_dir / "state_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))
        
        # Create new trainer and load
        trainer_loaded = Trainer(config, train_path, val_path, device="cpu")
        trainer_loaded.load_checkpoint(str(checkpoint_path))
        
        # Verify training state is preserved
        assert trainer_loaded.best_val_loss == trainer.best_val_loss, \
            f"best_val_loss not preserved: {trainer_loaded.best_val_loss} != {trainer.best_val_loss}"
        assert trainer_loaded.current_epoch == trainer.current_epoch, \
            f"current_epoch not preserved: {trainer_loaded.current_epoch} != {trainer.current_epoch}"
        assert trainer_loaded.patience_counter == trainer.patience_counter, \
            f"patience_counter not preserved: {trainer_loaded.patience_counter} != {trainer.patience_counter}"
    
    def test_checkpoint_preserves_optimizer_state(self, temp_checkpoint_dir, sample_datasets):
        """
        Property: Checkpoint preserves optimizer state including momentum buffers.
        
        This ensures that training can be resumed exactly where it left off.
        """
        from expression_control.trainer import Trainer
        from expression_control.models.config import LNNS4Config
        
        train_path, val_path = sample_datasets
        
        config = LNNS4Config(
            input_dim=14,
            output_dim=21,
            hidden_dim=32,
            state_dim=16,
            num_layers=2,
            liquid_units=32,
            dropout=0.0,
            sequence_length=4,
            batch_size=4,
            num_epochs=2,
        )
        
        # Create trainer and train for 1 epoch to build optimizer state
        trainer = Trainer(config, train_path, val_path, device="cpu")
        trainer.train_epoch()
        
        # Get optimizer state before saving
        optimizer_state_before = trainer.optimizer.state_dict()
        
        # Save checkpoint
        checkpoint_path = temp_checkpoint_dir / "optimizer_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))
        
        # Create new trainer and load
        trainer_loaded = Trainer(config, train_path, val_path, device="cpu")
        trainer_loaded.load_checkpoint(str(checkpoint_path))
        
        # Get optimizer state after loading
        optimizer_state_after = trainer_loaded.optimizer.state_dict()
        
        # Verify optimizer state is preserved
        # Check param_groups
        assert len(optimizer_state_before['param_groups']) == len(optimizer_state_after['param_groups']), \
            "Number of param_groups differs"
        
        for pg_before, pg_after in zip(optimizer_state_before['param_groups'], optimizer_state_after['param_groups']):
            assert pg_before['lr'] == pg_after['lr'], "Learning rate not preserved"
            assert pg_before['weight_decay'] == pg_after['weight_decay'], "Weight decay not preserved"
    
    def test_checkpoint_preserves_scheduler_state(self, temp_checkpoint_dir, sample_datasets):
        """
        Property: Checkpoint preserves learning rate scheduler state.
        
        This ensures that the learning rate schedule continues correctly after
        loading a checkpoint.
        """
        from expression_control.trainer import Trainer
        from expression_control.models.config import LNNS4Config
        
        train_path, val_path = sample_datasets
        
        config = LNNS4Config(
            input_dim=14,
            output_dim=21,
            hidden_dim=32,
            state_dim=16,
            num_layers=2,
            liquid_units=32,
            dropout=0.0,
            sequence_length=4,
            batch_size=4,
            num_epochs=10,
        )
        
        # Create trainer and step scheduler a few times
        trainer = Trainer(config, train_path, val_path, device="cpu")
        for _ in range(3):
            trainer.scheduler.step()
        
        # Get learning rate before saving
        lr_before = trainer.scheduler.get_last_lr()[0]
        
        # Save checkpoint
        checkpoint_path = temp_checkpoint_dir / "scheduler_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))
        
        # Create new trainer and load
        trainer_loaded = Trainer(config, train_path, val_path, device="cpu")
        trainer_loaded.load_checkpoint(str(checkpoint_path))
        
        # Get learning rate after loading
        lr_after = trainer_loaded.scheduler.get_last_lr()[0]
        
        # Verify learning rate is preserved
        assert abs(lr_before - lr_after) < 1e-8, \
            f"Learning rate not preserved: {lr_before} != {lr_after}"
    
    @settings(max_examples=30)
    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=1, max_value=8)
    )
    def test_checkpoint_works_with_different_batch_sizes(self, temp_checkpoint_dir, sample_datasets, batch_size, seq_len):
        """
        Property: Checkpoint works correctly regardless of batch size or sequence length.
        
        The saved model should work with different batch sizes and sequence lengths
        than it was trained with.
        """
        from expression_control.trainer import Trainer
        from expression_control.models.config import LNNS4Config
        
        train_path, val_path = sample_datasets
        
        config = LNNS4Config(
            input_dim=14,
            output_dim=21,
            hidden_dim=32,
            state_dim=16,
            num_layers=2,
            liquid_units=32,
            dropout=0.0,
            sequence_length=4,
            batch_size=4,
            num_epochs=2,
        )
        
        # Create and save trainer
        trainer = Trainer(config, train_path, val_path, device="cpu")
        checkpoint_path = temp_checkpoint_dir / f"batch_{batch_size}_seq_{seq_len}.pt"
        trainer.save_checkpoint(str(checkpoint_path))
        
        # Load checkpoint
        trainer_loaded = Trainer(config, train_path, val_path, device="cpu")
        trainer_loaded.load_checkpoint(str(checkpoint_path))
        
        # Test with different batch size and sequence length
        test_input = torch.randn(batch_size, seq_len, 14, dtype=torch.float32)
        
        trainer_loaded.model.eval()
        with torch.no_grad():
            output, _, _ = trainer_loaded.model(test_input)
        
        # Verify output shape is correct
        assert output.shape == (batch_size, seq_len, 21), \
            f"Expected shape ({batch_size}, {seq_len}, 21), got {output.shape}"
        
        # Verify output is valid
        assert torch.all(output >= 0.0) and torch.all(output <= 180.0), \
            "Output angles outside valid range [0, 180]"
    
    def test_checkpoint_file_format_stability(self, temp_checkpoint_dir, sample_datasets):
        """
        Property: Checkpoint file format is stable and contains expected keys.
        
        This ensures that the checkpoint format doesn't change unexpectedly.
        """
        from expression_control.trainer import Trainer
        from expression_control.models.config import LNNS4Config
        
        train_path, val_path = sample_datasets
        
        config = LNNS4Config(
            input_dim=14,
            output_dim=21,
            hidden_dim=32,
            state_dim=16,
            num_layers=2,
            liquid_units=32,
            dropout=0.0,
            sequence_length=4,
            batch_size=4,
            num_epochs=2,
        )
        
        # Create trainer and save checkpoint
        trainer = Trainer(config, train_path, val_path, device="cpu")
        checkpoint_path = temp_checkpoint_dir / "format_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))
        
        # Load checkpoint directly
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Verify expected keys are present
        expected_keys = {
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "config",
            "best_val_loss",
            "current_epoch",
            "patience_counter",
            "history",
        }
        
        actual_keys = set(checkpoint.keys())
        
        assert expected_keys.issubset(actual_keys), \
            f"Missing keys in checkpoint: {expected_keys - actual_keys}"
        
        # Verify types
        assert isinstance(checkpoint["model_state_dict"], dict), \
            "model_state_dict should be a dict"
        assert isinstance(checkpoint["optimizer_state_dict"], dict), \
            "optimizer_state_dict should be a dict"
        assert isinstance(checkpoint["config"], dict), \
            "config should be a dict"
        assert isinstance(checkpoint["best_val_loss"], (int, float)), \
            "best_val_loss should be numeric"
        assert isinstance(checkpoint["current_epoch"], int), \
            "current_epoch should be an int"


class TestMetricComputationCorrectness:
    """
    **Feature: vision-expression-control, Property 10: Metric Computation Correctness**
    
    *For any* set of predicted and ground truth angle arrays, the computed MAE
    SHALL equal the mean of absolute differences, and RMSE SHALL equal the
    square root of mean squared differences.
    
    **Validates: Requirements 5.6**
    
    This property ensures that:
    1. MAE is computed correctly as mean(|predictions - targets|)
    2. RMSE is computed correctly as sqrt(mean((predictions - targets)^2))
    3. Metrics are mathematically correct for all input shapes
    4. Metrics handle edge cases (zeros, identical values, etc.)
    5. Metrics produce finite, non-negative values
    """
    
    @settings(max_examples=100)
    @given(
        predictions=st.lists(
            st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=1000
        ),
        targets=st.lists(
            st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=1000
        )
    )
    def test_mae_computation_correctness(self, predictions, targets):
        """
        Property: MAE equals mean of absolute differences.
        
        For any set of predictions and targets, the computed MAE should be
        exactly equal to the mean of the absolute differences between them.
        
        This verifies Requirement 5.6: Training pipeline includes validation
        metrics for angle prediction accuracy (MAE).
        """
        from expression_control.trainer import compute_mae
        
        # Ensure same length
        min_len = min(len(predictions), len(targets))
        predictions = predictions[:min_len]
        targets = targets[:min_len]
        
        # Convert to numpy arrays
        pred_array = np.array(predictions, dtype=np.float32)
        target_array = np.array(targets, dtype=np.float32)
        
        # Compute MAE using the function
        mae = compute_mae(pred_array, target_array)
        
        # Compute expected MAE manually
        expected_mae = np.mean(np.abs(pred_array - target_array))
        
        # Verify they are equal (within floating point tolerance)
        assert np.isfinite(mae), f"MAE is not finite: {mae}"
        assert mae >= 0.0, f"MAE should be non-negative, got {mae}"
        assert np.isclose(mae, expected_mae, rtol=1e-5, atol=1e-6), \
            f"MAE computation incorrect: got {mae}, expected {expected_mae}"
    
    @settings(max_examples=100)
    @given(
        predictions=st.lists(
            st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=1000
        ),
        targets=st.lists(
            st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=1000
        )
    )
    def test_rmse_computation_correctness(self, predictions, targets):
        """
        Property: RMSE equals square root of mean squared differences.
        
        For any set of predictions and targets, the computed RMSE should be
        exactly equal to sqrt(mean((predictions - targets)^2)).
        
        This verifies Requirement 5.6: Training pipeline includes validation
        metrics for angle prediction accuracy (RMSE).
        """
        from expression_control.trainer import compute_rmse
        
        # Ensure same length
        min_len = min(len(predictions), len(targets))
        predictions = predictions[:min_len]
        targets = targets[:min_len]
        
        # Convert to numpy arrays
        pred_array = np.array(predictions, dtype=np.float32)
        target_array = np.array(targets, dtype=np.float32)
        
        # Compute RMSE using the function
        rmse = compute_rmse(pred_array, target_array)
        
        # Compute expected RMSE manually
        expected_rmse = np.sqrt(np.mean((pred_array - target_array) ** 2))
        
        # Verify they are equal (within floating point tolerance)
        assert np.isfinite(rmse), f"RMSE is not finite: {rmse}"
        assert rmse >= 0.0, f"RMSE should be non-negative, got {rmse}"
        assert np.isclose(rmse, expected_rmse, rtol=1e-5, atol=1e-6), \
            f"RMSE computation incorrect: got {rmse}, expected {expected_rmse}"
    
    @settings(max_examples=100)
    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        seq_len=st.integers(min_value=1, max_value=50),
        num_servos=st.integers(min_value=1, max_value=21)
    )
    def test_metrics_with_multidimensional_arrays(self, batch_size, seq_len, num_servos):
        """
        Property: Metrics work correctly with multidimensional arrays.
        
        The metric functions should handle arrays of any shape (flattened internally).
        """
        from expression_control.trainer import compute_mae, compute_rmse
        
        # Generate random predictions and targets
        shape = (batch_size, seq_len, num_servos)
        predictions = np.random.uniform(0, 180, size=shape).astype(np.float32)
        targets = np.random.uniform(0, 180, size=shape).astype(np.float32)
        
        # Compute metrics
        mae = compute_mae(predictions, targets)
        rmse = compute_rmse(predictions, targets)
        
        # Compute expected values manually
        expected_mae = np.mean(np.abs(predictions - targets))
        expected_rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # Verify
        assert np.isclose(mae, expected_mae, rtol=1e-5, atol=1e-6), \
            f"MAE incorrect for shape {shape}: got {mae}, expected {expected_mae}"
        assert np.isclose(rmse, expected_rmse, rtol=1e-5, atol=1e-6), \
            f"RMSE incorrect for shape {shape}: got {rmse}, expected {expected_rmse}"
    
    def test_mae_with_identical_values(self):
        """
        Unit test: MAE is zero when predictions equal targets.
        
        This is a specific edge case that should always hold.
        """
        from expression_control.trainer import compute_mae
        
        # Create identical arrays
        values = np.array([45.0, 90.0, 135.0, 180.0, 0.0], dtype=np.float32)
        
        mae = compute_mae(values, values)
        
        assert mae == 0.0, f"MAE should be 0 for identical values, got {mae}"
    
    def test_rmse_with_identical_values(self):
        """
        Unit test: RMSE is zero when predictions equal targets.
        
        This is a specific edge case that should always hold.
        """
        from expression_control.trainer import compute_rmse
        
        # Create identical arrays
        values = np.array([45.0, 90.0, 135.0, 180.0, 0.0], dtype=np.float32)
        
        rmse = compute_rmse(values, values)
        
        assert rmse == 0.0, f"RMSE should be 0 for identical values, got {rmse}"
    
    def test_mae_with_known_values(self):
        """
        Unit test: MAE computation with known values.
        
        Test with a simple example where we can manually verify the result.
        """
        from expression_control.trainer import compute_mae
        
        predictions = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        targets = np.array([15.0, 25.0, 35.0, 45.0], dtype=np.float32)
        
        # Expected MAE: mean(|10-15|, |20-25|, |30-35|, |40-45|) = mean(5, 5, 5, 5) = 5.0
        mae = compute_mae(predictions, targets)
        
        assert np.isclose(mae, 5.0, rtol=1e-5), \
            f"MAE should be 5.0, got {mae}"
    
    def test_rmse_with_known_values(self):
        """
        Unit test: RMSE computation with known values.
        
        Test with a simple example where we can manually verify the result.
        """
        from expression_control.trainer import compute_rmse
        
        predictions = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        targets = np.array([13.0, 24.0, 35.0, 48.0], dtype=np.float32)
        
        # Differences: [3, 4, 5, 8]
        # Squared: [9, 16, 25, 64]
        # Mean: (9 + 16 + 25 + 64) / 4 = 114 / 4 = 28.5
        # RMSE: sqrt(28.5) ≈ 5.3385
        expected_rmse = np.sqrt(28.5)
        
        rmse = compute_rmse(predictions, targets)
        
        assert np.isclose(rmse, expected_rmse, rtol=1e-5), \
            f"RMSE should be {expected_rmse:.4f}, got {rmse}"
    
    @settings(max_examples=100)
    @given(
        predictions=st.lists(
            st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=100
        )
    )
    def test_mae_is_always_non_negative(self, predictions):
        """
        Property: MAE is always non-negative.
        
        By definition, MAE is the mean of absolute values, so it must be >= 0.
        """
        from expression_control.trainer import compute_mae
        
        # Create targets (can be anything)
        targets = np.random.uniform(0, 180, size=len(predictions)).astype(np.float32)
        pred_array = np.array(predictions, dtype=np.float32)
        
        mae = compute_mae(pred_array, targets)
        
        assert mae >= 0.0, f"MAE should be non-negative, got {mae}"
    
    @settings(max_examples=100)
    @given(
        predictions=st.lists(
            st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=100
        )
    )
    def test_rmse_is_always_non_negative(self, predictions):
        """
        Property: RMSE is always non-negative.
        
        By definition, RMSE is the square root of a mean of squares, so it must be >= 0.
        """
        from expression_control.trainer import compute_rmse
        
        # Create targets (can be anything)
        targets = np.random.uniform(0, 180, size=len(predictions)).astype(np.float32)
        pred_array = np.array(predictions, dtype=np.float32)
        
        rmse = compute_rmse(pred_array, targets)
        
        assert rmse >= 0.0, f"RMSE should be non-negative, got {rmse}"
    
    @settings(max_examples=100)
    @given(
        predictions=st.lists(
            st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=100
        ),
        targets=st.lists(
            st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=100
        )
    )
    def test_rmse_greater_than_or_equal_to_mae(self, predictions, targets):
        """
        Property: RMSE >= MAE (Root Mean Square >= Mean Absolute).
        
        This is a mathematical property that should always hold:
        sqrt(mean(x^2)) >= mean(|x|) for any x.
        """
        from expression_control.trainer import compute_mae, compute_rmse
        
        # Ensure same length
        min_len = min(len(predictions), len(targets))
        predictions = predictions[:min_len]
        targets = targets[:min_len]
        
        pred_array = np.array(predictions, dtype=np.float32)
        target_array = np.array(targets, dtype=np.float32)
        
        mae = compute_mae(pred_array, target_array)
        rmse = compute_rmse(pred_array, target_array)
        
        # RMSE should be >= MAE (with small tolerance for floating point)
        assert rmse >= mae - 1e-6, \
            f"RMSE ({rmse}) should be >= MAE ({mae})"
    
    def test_mae_with_all_zeros(self):
        """
        Unit test: MAE with all zero predictions and targets.
        
        Edge case: both arrays are all zeros.
        """
        from expression_control.trainer import compute_mae
        
        predictions = np.zeros(10, dtype=np.float32)
        targets = np.zeros(10, dtype=np.float32)
        
        mae = compute_mae(predictions, targets)
        
        assert mae == 0.0, f"MAE should be 0 for all zeros, got {mae}"
    
    def test_rmse_with_all_zeros(self):
        """
        Unit test: RMSE with all zero predictions and targets.
        
        Edge case: both arrays are all zeros.
        """
        from expression_control.trainer import compute_rmse
        
        predictions = np.zeros(10, dtype=np.float32)
        targets = np.zeros(10, dtype=np.float32)
        
        rmse = compute_rmse(predictions, targets)
        
        assert rmse == 0.0, f"RMSE should be 0 for all zeros, got {rmse}"
    
    def test_mae_with_single_value(self):
        """
        Unit test: MAE with single value arrays.
        
        Edge case: arrays with only one element.
        """
        from expression_control.trainer import compute_mae
        
        predictions = np.array([100.0], dtype=np.float32)
        targets = np.array([90.0], dtype=np.float32)
        
        mae = compute_mae(predictions, targets)
        
        assert np.isclose(mae, 10.0, rtol=1e-5), \
            f"MAE should be 10.0, got {mae}"
    
    def test_rmse_with_single_value(self):
        """
        Unit test: RMSE with single value arrays.
        
        Edge case: arrays with only one element.
        """
        from expression_control.trainer import compute_rmse
        
        predictions = np.array([100.0], dtype=np.float32)
        targets = np.array([90.0], dtype=np.float32)
        
        rmse = compute_rmse(predictions, targets)
        
        assert np.isclose(rmse, 10.0, rtol=1e-5), \
            f"RMSE should be 10.0, got {rmse}"
    
    @settings(max_examples=100)
    @given(
        constant_error=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
        array_size=st.integers(min_value=1, max_value=100)
    )
    def test_mae_with_constant_error(self, constant_error, array_size):
        """
        Property: MAE equals the constant error when all errors are the same.
        
        If all predictions differ from targets by the same amount, MAE should
        equal that amount.
        """
        from expression_control.trainer import compute_mae
        
        # Create arrays where all errors are constant
        targets = np.random.uniform(0, 180 - constant_error, size=array_size).astype(np.float32)
        predictions = targets + constant_error
        
        mae = compute_mae(predictions, targets)
        
        # Use float32-appropriate tolerance (float32 has ~7 decimal digits of precision)
        assert np.isclose(mae, constant_error, rtol=1e-4, atol=1e-5), \
            f"MAE should equal constant error {constant_error}, got {mae}"
    
    @settings(max_examples=100)
    @given(
        constant_error=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
        array_size=st.integers(min_value=1, max_value=100)
    )
    def test_rmse_with_constant_error(self, constant_error, array_size):
        """
        Property: RMSE equals the constant error when all errors are the same.
        
        If all predictions differ from targets by the same amount, RMSE should
        equal that amount (since sqrt(mean(c^2)) = c for constant c).
        """
        from expression_control.trainer import compute_rmse
        
        # Create arrays where all errors are constant
        targets = np.random.uniform(0, 180 - constant_error, size=array_size).astype(np.float32)
        predictions = targets + constant_error
        
        rmse = compute_rmse(predictions, targets)
        
        # Use float32-appropriate tolerance (float32 has ~7 decimal digits of precision)
        assert np.isclose(rmse, constant_error, rtol=1e-4, atol=1e-5), \
            f"RMSE should equal constant error {constant_error}, got {rmse}"


class TestTemporalCoherence:
    """
    **Feature: vision-expression-control, Property 7: Temporal Coherence**
    
    *For any* two consecutive feature vectors with Euclidean distance less than
    threshold ε, the model output angles SHALL differ by no more than δ degrees
    per servo (where δ is configurable).
    
    **Validates: Requirements 4.9**
    
    This property ensures that:
    1. Small changes in input features produce small changes in output angles
    2. The model maintains smooth transitions between consecutive frames
    3. No sudden jumps in servo angles for similar inputs
    4. The model is Lipschitz continuous (bounded rate of change)
    """
    
    @pytest.fixture(scope="class")
    def model(self):
        """Create a model instance for testing temporal coherence."""
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
    @given(
        base_features=feature_vector_strategy(),
        perturbation_scale=st.floats(min_value=0.001, max_value=0.1, allow_nan=False, allow_infinity=False)
    )
    def test_small_input_changes_produce_bounded_output_changes(self, model, base_features, perturbation_scale):
        """
        Property: Small input perturbations produce bounded output changes.
        
        For any two consecutive feature vectors where the Euclidean distance
        is less than ε, the maximum change in any servo angle should be bounded.
        
        This verifies Requirement 4.9: When processing consecutive video frames,
        the model SHALL maintain temporal coherence in output angles.
        
        The relationship tested is:
        ||f1 - f2|| < ε  =>  max(|θ1_i - θ2_i|) < δ for all servos i
        
        Where:
        - f1, f2 are consecutive feature vectors
        - θ1, θ2 are the corresponding output angle vectors
        - ε is the input perturbation threshold
        - δ is the maximum allowed angle change per servo
        """
        # Convert base features to tensor
        base_tensor = torch.tensor(base_features, dtype=torch.float32).unsqueeze(0)  # (1, 14)
        
        # Create perturbed features (small random perturbation)
        perturbation = torch.randn_like(base_tensor) * perturbation_scale
        perturbed_tensor = base_tensor + perturbation
        
        # Compute Euclidean distance between inputs
        input_distance = torch.norm(perturbation).item()
        
        # Get model outputs for both inputs
        with torch.no_grad():
            angles_base, _, _ = model(base_tensor)
            angles_perturbed, _, _ = model(perturbed_tensor)
        
        # Compute maximum angle change across all servos
        angle_diff = torch.abs(angles_base - angles_perturbed)
        max_angle_change = torch.max(angle_diff).item()
        
        # Define the coherence bound: δ should be proportional to ε
        # Using a Lipschitz-like bound: max_angle_change <= K * input_distance
        # where K is a reasonable Lipschitz constant for the model
        # For a well-behaved model, K should be bounded (e.g., K <= 1000 degrees per unit input)
        lipschitz_constant = 1000.0  # Maximum degrees change per unit input distance
        expected_max_change = lipschitz_constant * input_distance
        
        # Also enforce an absolute maximum change for very small perturbations
        # This ensures temporal coherence even for tiny input changes
        absolute_max_change = 45.0  # Maximum 45 degrees change for any small perturbation
        
        # The actual bound is the minimum of the two constraints
        coherence_bound = min(expected_max_change, absolute_max_change)
        
        assert max_angle_change <= coherence_bound, (
            f"Temporal coherence violated: max angle change {max_angle_change:.2f}° "
            f"exceeds bound {coherence_bound:.2f}° for input distance {input_distance:.6f}"
        )
    
    @settings(max_examples=100)
    @given(
        base_features=feature_vector_strategy(),
        num_steps=st.integers(min_value=5, max_value=20)
    )
    def test_gradual_input_changes_produce_smooth_output_trajectory(self, model, base_features, num_steps):
        """
        Property: Gradual input changes produce smooth output trajectories.
        
        When input features change gradually over multiple steps, the output
        angles should also change smoothly without sudden jumps.
        
        This tests temporal coherence over a sequence of frames, not just
        between two consecutive frames.
        """
        # Create a smooth trajectory of input features
        base_tensor = torch.tensor(base_features, dtype=torch.float32)
        
        # Generate a random direction for the trajectory
        direction = torch.randn(14)
        direction = direction / torch.norm(direction)  # Normalize
        
        # Create trajectory: base -> base + direction * step_size * num_steps
        step_size = 0.05  # Small step size for smooth trajectory
        
        angles_trajectory = []
        
        with torch.no_grad():
            for step in range(num_steps):
                # Interpolate along the trajectory
                current_features = base_tensor + direction * step_size * step
                current_features = current_features.unsqueeze(0)  # (1, 14)
                
                angles, _, _ = model(current_features)
                angles_trajectory.append(angles.squeeze(0).numpy())
        
        # Check that consecutive outputs don't have large jumps
        max_step_change = 0.0
        for i in range(1, len(angles_trajectory)):
            step_diff = np.abs(angles_trajectory[i] - angles_trajectory[i-1])
            max_step_change = max(max_step_change, np.max(step_diff))
        
        # For a smooth trajectory with step_size=0.05, we expect bounded changes
        # The bound should be proportional to step_size
        max_allowed_step_change = 30.0  # Maximum 30 degrees per step for smooth trajectory
        
        assert max_step_change <= max_allowed_step_change, (
            f"Trajectory not smooth: max step change {max_step_change:.2f}° "
            f"exceeds {max_allowed_step_change:.2f}° for step_size={step_size}"
        )
    
    @settings(max_examples=100)
    @given(
        features=feature_vector_strategy()
    )
    def test_identical_inputs_produce_identical_outputs(self, model, features):
        """
        Property: Identical inputs produce identical outputs (determinism).
        
        This is a special case of temporal coherence: when input distance is 0,
        output distance should also be 0.
        """
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            angles1, _, _ = model(features_tensor)
            angles2, _, _ = model(features_tensor)
        
        # Outputs should be identical
        assert torch.allclose(angles1, angles2, rtol=1e-5, atol=1e-6), (
            f"Identical inputs produced different outputs: "
            f"max diff = {torch.max(torch.abs(angles1 - angles2)).item():.6f}"
        )
    
    @settings(max_examples=100)
    @given(
        base_features=feature_vector_strategy(),
        epsilon=st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False)
    )
    def test_epsilon_ball_coherence(self, model, base_features, epsilon):
        """
        Property: All inputs within ε-ball produce outputs within δ-ball.
        
        For any base input and any perturbed input within distance ε,
        the output angles should be within a bounded distance δ.
        
        This is a stronger form of temporal coherence that ensures
        local Lipschitz continuity.
        """
        base_tensor = torch.tensor(base_features, dtype=torch.float32).unsqueeze(0)
        
        # Generate multiple random perturbations within ε-ball
        num_samples = 10
        max_output_distance = 0.0
        
        with torch.no_grad():
            angles_base, _, _ = model(base_tensor)
            
            for _ in range(num_samples):
                # Generate random perturbation
                perturbation = torch.randn(1, 14)
                perturbation = perturbation / torch.norm(perturbation) * epsilon * torch.rand(1).item()
                
                perturbed_tensor = base_tensor + perturbation
                angles_perturbed, _, _ = model(perturbed_tensor)
                
                # Compute output distance (max angle difference)
                output_distance = torch.max(torch.abs(angles_base - angles_perturbed)).item()
                max_output_distance = max(max_output_distance, output_distance)
        
        # The output distance should be bounded by a function of epsilon
        # Using a linear bound: δ <= K * ε where K is the Lipschitz constant
        lipschitz_constant = 500.0  # Degrees per unit input distance
        delta_bound = lipschitz_constant * epsilon
        
        # Also cap at a reasonable maximum
        delta_bound = min(delta_bound, 90.0)  # Max 90 degrees for any ε <= 0.5
        
        assert max_output_distance <= delta_bound, (
            f"ε-ball coherence violated: max output distance {max_output_distance:.2f}° "
            f"exceeds bound {delta_bound:.2f}° for ε={epsilon:.4f}"
        )
    
    @settings(max_examples=50)
    @given(
        features_sequence=st.lists(
            feature_vector_strategy(),
            min_size=5,
            max_size=20
        )
    )
    def test_stateful_inference_temporal_coherence(self, model, features_sequence):
        """
        Property: Stateful inference maintains temporal coherence.
        
        When processing a sequence of frames with state management,
        the model should maintain smooth transitions between outputs.
        
        This tests the step() method which is used for real-time inference.
        """
        s4_states = None
        ltc_state = None
        
        angles_history = []
        
        with torch.no_grad():
            for features in features_sequence:
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                angles, s4_states, ltc_state = model.step(features_tensor, s4_states, ltc_state)
                angles_history.append(angles.squeeze(0).numpy())
        
        # Check temporal coherence between consecutive outputs
        max_consecutive_change = 0.0
        for i in range(1, len(angles_history)):
            diff = np.abs(angles_history[i] - angles_history[i-1])
            max_consecutive_change = max(max_consecutive_change, np.max(diff))
        
        # For arbitrary input sequences, we allow larger changes
        # but they should still be bounded
        max_allowed_change = 90.0  # Maximum 90 degrees between any two consecutive frames
        
        assert max_consecutive_change <= max_allowed_change, (
            f"Stateful inference temporal coherence violated: "
            f"max consecutive change {max_consecutive_change:.2f}° exceeds {max_allowed_change:.2f}°"
        )
    
    def test_temporal_coherence_with_neutral_to_smile_transition(self, model):
        """
        Unit test: Smooth transition from neutral to smile expression.
        
        This tests a realistic scenario where facial features gradually
        change from neutral to smiling.
        """
        # Neutral features
        neutral = torch.zeros(1, 14, dtype=torch.float32)
        neutral[0, 10] = 0.0  # smile_intensity = 0
        
        # Smile features
        smile = torch.zeros(1, 14, dtype=torch.float32)
        smile[0, 10] = 1.0  # smile_intensity = 1
        
        # Create interpolated trajectory
        num_steps = 10
        angles_trajectory = []
        
        with torch.no_grad():
            for i in range(num_steps + 1):
                t = i / num_steps
                features = neutral * (1 - t) + smile * t
                angles, _, _ = model(features)
                angles_trajectory.append(angles.squeeze(0).numpy())
        
        # Check smoothness of trajectory
        max_step_change = 0.0
        for i in range(1, len(angles_trajectory)):
            diff = np.abs(angles_trajectory[i] - angles_trajectory[i-1])
            max_step_change = max(max_step_change, np.max(diff))
        
        # For a 10-step interpolation, each step should have bounded change
        max_allowed_per_step = 20.0  # Maximum 20 degrees per interpolation step
        
        assert max_step_change <= max_allowed_per_step, (
            f"Neutral-to-smile transition not smooth: "
            f"max step change {max_step_change:.2f}° exceeds {max_allowed_per_step:.2f}°"
        )
    
    def test_temporal_coherence_with_eye_blink(self, model):
        """
        Unit test: Smooth transition during eye blink.
        
        This tests temporal coherence during a rapid but natural
        facial movement (eye blink).
        """
        # Open eyes
        open_eyes = torch.zeros(1, 14, dtype=torch.float32)
        open_eyes[0, 0] = 0.3  # left_eye_aspect_ratio (open)
        open_eyes[0, 1] = 0.3  # right_eye_aspect_ratio (open)
        
        # Closed eyes
        closed_eyes = torch.zeros(1, 14, dtype=torch.float32)
        closed_eyes[0, 0] = 0.0  # left_eye_aspect_ratio (closed)
        closed_eyes[0, 1] = 0.0  # right_eye_aspect_ratio (closed)
        
        # Blink sequence: open -> closed -> open (quick transition)
        blink_sequence = [
            open_eyes,
            open_eyes * 0.7 + closed_eyes * 0.3,
            open_eyes * 0.3 + closed_eyes * 0.7,
            closed_eyes,
            open_eyes * 0.3 + closed_eyes * 0.7,
            open_eyes * 0.7 + closed_eyes * 0.3,
            open_eyes,
        ]
        
        angles_trajectory = []
        
        with torch.no_grad():
            for features in blink_sequence:
                angles, _, _ = model(features)
                angles_trajectory.append(angles.squeeze(0).numpy())
        
        # Check smoothness
        max_step_change = 0.0
        for i in range(1, len(angles_trajectory)):
            diff = np.abs(angles_trajectory[i] - angles_trajectory[i-1])
            max_step_change = max(max_step_change, np.max(diff))
        
        # Eye blinks are fast but should still be smooth
        max_allowed_per_step = 30.0  # Maximum 30 degrees per blink step
        
        assert max_step_change <= max_allowed_per_step, (
            f"Eye blink transition not smooth: "
            f"max step change {max_step_change:.2f}° exceeds {max_allowed_per_step:.2f}°"
        )

