"""
PyTorch Dataset for expression control training.

This module provides the ExpressionDataset class for loading and preprocessing
training data for the LNN-S4 model, including sequence creation and data
augmentation.

Requirements: 3.5, 5.1, 5.2
"""

import json
import random
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # type: ignore

from expression_control.data import TrainingDataset, TrainingDataSample
from expression_control.protocol import ServoCommandProtocol


class ExpressionDataset(Dataset):
    """
    PyTorch Dataset for expression control training.
    
    Loads training data from JSON files and creates fixed-length sequences
    for training the LNN-S4 model. Supports data augmentation including
    temporal jittering and feature noise injection.
    
    Each sample consists of:
    - Input: Sequence of facial feature vectors (seq_len, 14)
    - Target: Sequence of servo angle vectors (seq_len, 21)
    
    Usage:
        dataset = ExpressionDataset("data/train.json", sequence_length=16)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for features, targets in dataloader:
            # features: (batch, seq_len, 14)
            # targets: (batch, seq_len, 21)
            ...
    """
    
    # Feature dimension (14 facial features)
    FEATURE_DIM = 14
    
    # Output dimension (21 servo angles)
    OUTPUT_DIM = 21
    
    def __init__(
        self,
        json_path: str,
        sequence_length: int = 16,
        augment: bool = False,
        temporal_jitter_prob: float = 0.3,
        noise_std: float = 0.02,
        normalize_angles: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            json_path: Path to the JSON dataset file.
            sequence_length: Length of sequences to create.
            augment: Whether to apply data augmentation.
            temporal_jitter_prob: Probability of applying temporal jittering.
            noise_std: Standard deviation of Gaussian noise for feature augmentation.
            normalize_angles: Whether to normalize servo angles to [0, 1].
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Install with: pip install torch"
            )
        
        self.sequence_length = sequence_length
        self.augment = augment
        self.temporal_jitter_prob = temporal_jitter_prob
        self.noise_std = noise_std
        self.normalize_angles = normalize_angles
        
        # Load dataset
        self.dataset = TrainingDataset.load(json_path)
        self.samples = self.dataset.samples
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _create_sequences(self) -> List[List[TrainingDataSample]]:
        """
        Create fixed-length sequences from samples.
        
        Uses a sliding window approach to create overlapping sequences.
        
        Returns:
            List of sample sequences, each of length sequence_length.
        """
        sequences = []
        
        if len(self.samples) < self.sequence_length:
            # If not enough samples, pad with repetition
            if len(self.samples) > 0:
                padded = self.samples * (self.sequence_length // len(self.samples) + 1)
                sequences.append(padded[:self.sequence_length])
        else:
            # Sliding window
            for i in range(len(self.samples) - self.sequence_length + 1):
                seq = self.samples[i:i + self.sequence_length]
                sequences.append(seq)
        
        return sequences
    
    def _extract_features(self, sample: TrainingDataSample) -> np.ndarray:
        """
        Extract feature vector from a sample.
        
        Returns:
            Shape (14,) feature vector.
        """
        return sample.face_features.to_array()
    
    def _extract_targets(self, sample: TrainingDataSample) -> np.ndarray:
        """
        Extract target servo angles from a sample.
        
        Returns:
            Shape (21,) angle vector, optionally normalized to [0, 1].
        """
        angles = np.array([
            sample.servo_angles[name] 
            for name in ServoCommandProtocol.SERVO_ORDER
        ], dtype=np.float32)
        
        if self.normalize_angles:
            angles = angles / 180.0  # Normalize to [0, 1]
        
        return angles
    
    def _augment_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to feature sequence.
        
        Augmentation includes:
        - Temporal jittering: randomly skip frames and interpolate
        - Feature noise: add Gaussian noise to features
        
        Args:
            features: Shape (seq_len, 14) feature sequence.
            
        Returns:
            Augmented feature sequence of same shape.
        """
        seq_len = features.shape[0]
        
        # Temporal jittering: randomly skip frames
        if random.random() < self.temporal_jitter_prob and seq_len > 2:
            skip = random.randint(1, min(2, seq_len - 1))
            subsampled = features[::skip]
            
            # Interpolate back to original length
            if len(subsampled) < seq_len:
                old_indices = np.linspace(0, len(subsampled) - 1, len(subsampled))
                new_indices = np.linspace(0, len(subsampled) - 1, seq_len)
                
                # Interpolate each feature dimension
                interpolated = np.zeros((seq_len, features.shape[1]), dtype=np.float32)
                for dim in range(features.shape[1]):
                    interpolated[:, dim] = np.interp(
                        new_indices, old_indices, subsampled[:, dim]
                    )
                features = interpolated
        
        # Feature noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, features.shape)
            features = features + noise.astype(np.float32)
        
        return features.astype(np.float32)
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence by index.
        
        Args:
            idx: Sequence index.
            
        Returns:
            Tuple of (features, targets) tensors:
            - features: Shape (seq_len, 14)
            - targets: Shape (seq_len, 21)
        """
        seq = self.sequences[idx]
        
        # Extract features and targets
        features = np.array([self._extract_features(s) for s in seq], dtype=np.float32)
        targets = np.array([self._extract_targets(s) for s in seq], dtype=np.float32)
        
        # Apply augmentation
        if self.augment:
            features = self._augment_features(features)
        
        return torch.from_numpy(features), torch.from_numpy(targets)
    
    @property
    def feature_dim(self) -> int:
        """Return the feature dimension."""
        return self.FEATURE_DIM
    
    @property
    def output_dim(self) -> int:
        """Return the output dimension."""
        return self.OUTPUT_DIM
    
    @classmethod
    def from_multiple_files(
        cls,
        json_paths: List[str],
        sequence_length: int = 16,
        augment: bool = False,
        **kwargs
    ) -> "ExpressionDataset":
        """
        Create a dataset from multiple JSON files.
        
        Args:
            json_paths: List of paths to JSON dataset files.
            sequence_length: Length of sequences to create.
            augment: Whether to apply data augmentation.
            **kwargs: Additional arguments passed to __init__.
            
        Returns:
            Combined ExpressionDataset.
        """
        if not json_paths:
            raise ValueError("At least one JSON path is required")
        
        # Load first dataset
        combined = TrainingDataset.load(json_paths[0])
        
        # Merge additional datasets
        for path in json_paths[1:]:
            additional = TrainingDataset.load(path)
            combined.samples.extend(additional.samples)
        
        # Create temporary file for combined dataset
        # (This is a workaround since __init__ expects a file path)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(combined.to_json())
            temp_path = f.name
        
        try:
            dataset = cls(
                temp_path,
                sequence_length=sequence_length,
                augment=augment,
                **kwargs
            )
        finally:
            import os
            os.unlink(temp_path)
        
        return dataset


def split_dataset(
    json_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    output_dir: str = ".",
    seed: int = 42,
) -> Tuple[str, str, str]:
    """
    Split a dataset into train/val/test sets.
    
    Args:
        json_path: Path to the source JSON dataset.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        output_dir: Directory to save split datasets.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_path, val_path, test_path).
    """
    import os
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # Load dataset
    dataset = TrainingDataset.load(json_path)
    samples = dataset.samples.copy()
    
    # Shuffle samples
    random.seed(seed)
    random.shuffle(samples)
    
    # Calculate split indices
    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split samples
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    # Create split datasets
    train_dataset = TrainingDataset(samples=train_samples, fps=dataset.fps)
    val_dataset = TrainingDataset(samples=val_samples, fps=dataset.fps)
    test_dataset = TrainingDataset(samples=test_samples, fps=dataset.fps)
    
    # Save split datasets
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    train_path = os.path.join(output_dir, f"{base_name}_train.json")
    val_path = os.path.join(output_dir, f"{base_name}_val.json")
    test_path = os.path.join(output_dir, f"{base_name}_test.json")
    
    train_dataset.save(train_path)
    val_dataset.save(val_path)
    test_dataset.save(test_path)
    
    return train_path, val_path, test_path
