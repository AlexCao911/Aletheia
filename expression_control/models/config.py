"""
Configuration dataclass for LNN-S4 model.

This module defines the LNNS4Config dataclass containing all model hyperparameters
for the Liquid Neural Network with S4 layers used in expression control.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LNNS4Config:
    """
    LNN-S4 模型配置
    
    Configuration for the Liquid-S4 Expression Model that processes
    facial features and outputs servo angles for robot expression control.
    
    Attributes:
        input_dim: Dimension of input FaceFeatures vector (14 features)
        output_dim: Number of servo outputs (21 servos)
        hidden_dim: Hidden layer dimension for S4 blocks
        state_dim: S4 state space dimension
        num_layers: Number of S4 blocks
        liquid_units: Number of neurons in LTC layer
        sequence_length: Temporal window length for training
        dropout: Dropout rate for regularization
        learning_rate: Initial learning rate for optimizer
        weight_decay: L2 regularization weight
        batch_size: Training batch size
        num_epochs: Maximum training epochs
        early_stopping_patience: Epochs to wait before early stopping
        min_angle: Minimum servo angle (degrees)
        max_angle: Maximum servo angle (degrees)
    """
    
    # Input/Output dimensions
    input_dim: int = 14               # FaceFeatures 特征维度
    output_dim: int = 21              # 舵机数量
    
    # S4 layer configuration
    hidden_dim: int = 64              # 隐藏层维度
    state_dim: int = 32               # S4 状态维度
    num_layers: int = 2               # S4 层数
    
    # Liquid Time-Constant (LTC) configuration
    liquid_units: int = 32            # Liquid 神经元数量
    
    # Training configuration
    sequence_length: int = 16         # 时序窗口长度
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Servo angle constraints
    min_angle: float = 0.0
    max_angle: float = 180.0
    
    # Model size constraint (bytes) - 20MB max for edge deployment
    max_model_size: int = 20 * 1024 * 1024
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {self.state_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.liquid_units <= 0:
            raise ValueError(f"liquid_units must be positive, got {self.liquid_units}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.min_angle >= self.max_angle:
            raise ValueError(f"min_angle ({self.min_angle}) must be less than max_angle ({self.max_angle})")
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'state_dim': self.state_dim,
            'num_layers': self.num_layers,
            'liquid_units': self.liquid_units,
            'sequence_length': self.sequence_length,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'min_angle': self.min_angle,
            'max_angle': self.max_angle,
            'max_model_size': self.max_model_size,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'LNNS4Config':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
