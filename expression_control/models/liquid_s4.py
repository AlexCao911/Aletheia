"""
Liquid-S4 Model for Expression Control.

This module implements the LiquidS4Model that combines S4 layers for temporal
sequence modeling with Liquid Time-Constant (LTC) neurons for smooth,
adaptive expression control.

Reference:
- S4: https://github.com/raminmh/liquid-s4
- ncps: https://github.com/mlech26l/ncps
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .config import LNNS4Config
from .s4 import S4Block


class LiquidS4Model(nn.Module):
    """
    Liquid-S4 Model for Expression Control.
    
    This model processes facial feature sequences and outputs servo angles
    for robot expression control. It combines:
    
    1. Feature embedding layer
    2. Multiple S4 blocks for temporal modeling
    3. Liquid Time-Constant (LTC) layer for smooth dynamics
    4. Output head with sigmoid activation scaled to [0, 180]
    
    Architecture:
        Input (14-dim) -> Embedding (64-dim) -> S4 Blocks x2 -> LTC -> Output (21-dim)
    
    Attributes:
        config: Model configuration
        embedding: Feature embedding layer
        s4_blocks: List of S4 blocks
        ltc: Liquid Time-Constant layer (from ncps)
        output_head: Final projection to servo angles
    """
    
    def __init__(self, config: LNNS4Config):
        """
        Initialize Liquid-S4 model.
        
        Args:
            config: Model configuration dataclass
        """
        super().__init__()
        self.config = config
        
        # Feature embedding: input_dim -> hidden_dim
        self.embedding = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        
        # S4 blocks for temporal modeling
        self.s4_blocks = nn.ModuleList([
            S4Block(
                d_model=config.hidden_dim,
                d_state=config.state_dim,
                dropout=config.dropout,
                ff_mult=4,
            )
            for _ in range(config.num_layers)
        ])
        
        # Liquid Time-Constant (LTC) layer from ncps
        # We use a try/except to handle cases where ncps is not installed
        self._init_ltc_layer(config)
        
        # Output head: _ltc_output_dim -> output_dim with sigmoid
        # Note: _ltc_output_dim is set by _init_ltc_layer based on whether ncps is available
        self.output_head = nn.Sequential(
            nn.Linear(self._ltc_output_dim, config.output_dim),
            nn.Sigmoid(),  # Output in [0, 1], will be scaled to [0, 180]
        )
        
        # Store angle range for scaling
        self.min_angle = config.min_angle
        self.max_angle = config.max_angle
    
    def _init_ltc_layer(self, config: LNNS4Config):
        """
        Initialize the LTC layer from ncps library.
        
        Falls back to a simple RNN if ncps is not available.
        """
        try:
            from ncps.torch import LTC
            from ncps.wirings import AutoNCP
            
            # Create wiring for LTC
            # AutoNCP outputs output_dim directly
            wiring = AutoNCP(config.liquid_units, config.output_dim)
            self.ltc = LTC(config.hidden_dim, wiring, batch_first=True)
            self.use_ncps = True
            # LTC with AutoNCP outputs output_dim (21)
            self._ltc_output_dim = config.output_dim
        except ImportError:
            # Fallback to simple GRU if ncps not available
            self.ltc = nn.GRU(
                input_size=config.hidden_dim,
                hidden_size=config.liquid_units,
                batch_first=True,
            )
            self.use_ncps = False
            # GRU outputs liquid_units (32)
            self._ltc_output_dim = config.liquid_units
    
    def forward(
        self,
        x: torch.Tensor,
        s4_states: Optional[List[torch.Tensor]] = None,
        ltc_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim) or (batch, input_dim)
            s4_states: List of previous S4 states, one per S4 block
            ltc_state: Previous LTC hidden state
            
        Returns:
            Tuple of:
                - angles: Servo angles of shape (batch, seq_len, 21) or (batch, 21)
                - new_s4_states: Updated S4 states
                - new_ltc_state: Updated LTC state
        """
        # Handle single frame input (add sequence dimension)
        single_frame = x.dim() == 2
        if single_frame:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        batch, seq_len, _ = x.shape
        
        # Initialize S4 states if not provided
        if s4_states is None:
            s4_states = [None] * self.config.num_layers
        
        # Feature embedding
        x = self.embedding(x)  # (batch, seq_len, hidden_dim)
        
        # Process through S4 blocks
        new_s4_states = []
        for i, s4_block in enumerate(self.s4_blocks):
            x, new_state = s4_block(x, s4_states[i])
            new_s4_states.append(new_state)
        
        # Process through LTC layer
        if self.use_ncps:
            x, new_ltc_state = self.ltc(x, ltc_state)
        else:
            # Fallback GRU path
            if ltc_state is not None:
                ltc_state = ltc_state.unsqueeze(0)  # (1, batch, hidden)
            x, new_ltc_state = self.ltc(x, ltc_state)
            if new_ltc_state is not None:
                new_ltc_state = new_ltc_state.squeeze(0)  # (batch, hidden)
        
        # Output head: project to servo angles
        angles = self.output_head(x)  # (batch, seq_len, output_dim) in [0, 1]
        
        # Scale to angle range [min_angle, max_angle]
        angles = angles * (self.max_angle - self.min_angle) + self.min_angle
        
        # Remove sequence dimension if single frame input
        if single_frame:
            angles = angles.squeeze(1)  # (batch, output_dim)
        
        return angles, new_s4_states, new_ltc_state
    
    def step(
        self,
        x: torch.Tensor,
        s4_states: Optional[List[torch.Tensor]] = None,
        ltc_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Single step forward pass for real-time inference.
        
        More efficient than forward() for single frame processing.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            s4_states: List of previous S4 states
            ltc_state: Previous LTC hidden state
            
        Returns:
            Tuple of (angles, new_s4_states, new_ltc_state)
        """
        batch = x.shape[0]
        
        # Initialize S4 states if not provided
        if s4_states is None:
            s4_states = [None] * self.config.num_layers
        
        # Feature embedding
        x = self.embedding(x)  # (batch, hidden_dim)
        
        # Process through S4 blocks (single step mode)
        new_s4_states = []
        for i, s4_block in enumerate(self.s4_blocks):
            x, new_state = s4_block.step(x, s4_states[i])
            new_s4_states.append(new_state)
        
        # Process through LTC layer (need to add/remove seq dim)
        x = x.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        if self.use_ncps:
            x, new_ltc_state = self.ltc(x, ltc_state)
        else:
            if ltc_state is not None:
                ltc_state = ltc_state.unsqueeze(0)
            x, new_ltc_state = self.ltc(x, ltc_state)
            if new_ltc_state is not None:
                new_ltc_state = new_ltc_state.squeeze(0)
        
        x = x.squeeze(1)  # (batch, hidden_dim or liquid_units)
        
        # Output head
        angles = self.output_head(x)  # (batch, output_dim) in [0, 1]
        
        # Scale to angle range
        angles = angles * (self.max_angle - self.min_angle) + self.min_angle
        
        return angles, new_s4_states, new_ltc_state
    
    def reset_states(self) -> Tuple[None, None]:
        """
        Reset all hidden states for new sequence.
        
        Returns:
            Tuple of (None, None) representing reset s4_states and ltc_state
        """
        return None, None
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_bytes(self) -> int:
        """Get approximate model size in bytes."""
        param_size = sum(
            p.numel() * p.element_size() for p in self.parameters()
        )
        buffer_size = sum(
            b.numel() * b.element_size() for b in self.buffers()
        )
        return param_size + buffer_size


def create_model(config: Optional[LNNS4Config] = None) -> LiquidS4Model:
    """
    Factory function to create a LiquidS4Model.
    
    Args:
        config: Model configuration. If None, uses default config.
        
    Returns:
        Initialized LiquidS4Model
    """
    if config is None:
        config = LNNS4Config()
    return LiquidS4Model(config)
