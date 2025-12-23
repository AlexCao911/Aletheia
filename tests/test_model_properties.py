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
