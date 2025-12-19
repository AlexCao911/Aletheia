"""
Structured State Space (S4) Layer implementation.

This module implements the S4 layer with HiPPO initialization for temporal
sequence modeling in the expression control system.

Reference: https://github.com/raminmh/liquid-s4
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_hippo_matrix(n: int) -> torch.Tensor:
    """
    Create HiPPO (High-order Polynomial Projection Operators) matrix.
    
    The HiPPO matrix provides optimal initialization for capturing
    long-range dependencies in sequential data.
    
    Args:
        n: State dimension
        
    Returns:
        HiPPO-LegS matrix of shape (n, n)
    """
    # HiPPO-LegS (Legendre) matrix
    P = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i > j:
                P[i, j] = math.sqrt(2 * i + 1) * math.sqrt(2 * j + 1)
            elif i == j:
                P[i, j] = i + 1
    return -P


class S4Layer(nn.Module):
    """
    Structured State Space Sequence Layer.
    
    Implements the S4 layer which uses structured state space models
    for efficient sequence modeling with HiPPO initialization.
    
    The state space model is defined as:
        x'(t) = A x(t) + B u(t)
        y(t)  = C x(t) + D u(t)
    
    where:
        - x is the hidden state
        - u is the input
        - y is the output
        - A, B, C, D are learnable parameters
    
    Attributes:
        d_model: Input/output dimension
        d_state: State space dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 32,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        """
        Initialize S4 layer.
        
        Args:
            d_model: Input and output dimension
            d_state: State space dimension (hidden state size)
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional processing
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidirectional = bidirectional
        
        # Initialize A matrix with HiPPO
        hippo = make_hippo_matrix(d_state)
        self.A = nn.Parameter(hippo)
        
        # Initialize B, C with scaled random values
        self.B = nn.Parameter(
            torch.randn(d_state, d_model) / math.sqrt(d_model)
        )
        self.C = nn.Parameter(
            torch.randn(d_model, d_state) / math.sqrt(d_state)
        )
        
        # D is a skip connection (initialized to 1)
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Discretization time step (learnable)
        self.log_dt = nn.Parameter(torch.zeros(d_model))
        
        # Layer normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
    
    def _discretize(self, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discretize continuous state space to discrete form using bilinear transform.
        
        Args:
            dt: Time step tensor
            
        Returns:
            Tuple of (A_bar, B_bar) discretized matrices
        """
        # Bilinear (Tustin) discretization
        # A_bar = (I + dt/2 * A) @ inv(I - dt/2 * A)
        # B_bar = dt * inv(I - dt/2 * A) @ B
        
        I = torch.eye(self.d_state, device=self.A.device)
        dt_A = dt.mean() * self.A  # Use mean dt for simplicity
        
        # Compute (I - dt/2 * A)^(-1)
        inv_term = torch.linalg.inv(I - 0.5 * dt_A)
        
        A_bar = (I + 0.5 * dt_A) @ inv_term
        B_bar = dt.mean() * inv_term @ self.B
        
        return A_bar, B_bar
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through S4 layer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            state: Previous hidden state of shape (batch, d_state), or None
            
        Returns:
            Tuple of:
                - Output tensor of shape (batch, seq_len, d_model)
                - New hidden state of shape (batch, d_state)
        """
        batch, seq_len, _ = x.shape
        device = x.device
        
        # Initialize state if not provided
        if state is None:
            state = torch.zeros(batch, self.d_state, device=device)
        
        # Get discretization time step
        dt = torch.exp(self.log_dt).clamp(min=1e-4, max=1.0)
        
        # Discretize state space matrices
        A_bar, B_bar = self._discretize(dt)
        
        # Process sequence step by step (recurrent mode)
        outputs = []
        for t in range(seq_len):
            u_t = x[:, t, :]  # (batch, d_model)
            
            # State update: x_{t+1} = A_bar @ x_t + B_bar @ u_t
            state = torch.tanh(state @ A_bar.T + u_t @ B_bar.T)
            
            # Output: y_t = C @ x_t + D * u_t
            y_t = state @ self.C.T + u_t * self.D
            outputs.append(y_t)
        
        # Stack outputs
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        
        # Apply output projection, dropout, and normalization
        y = self.output_proj(y)
        y = self.dropout(y)
        
        return y, state
    
    def step(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step forward pass (for inference).
        
        Args:
            x: Input tensor of shape (batch, d_model)
            state: Previous hidden state of shape (batch, d_state), or None
            
        Returns:
            Tuple of:
                - Output tensor of shape (batch, d_model)
                - New hidden state of shape (batch, d_state)
        """
        batch = x.shape[0]
        device = x.device
        
        # Initialize state if not provided
        if state is None:
            state = torch.zeros(batch, self.d_state, device=device)
        
        # Get discretization time step
        dt = torch.exp(self.log_dt).clamp(min=1e-4, max=1.0)
        
        # Discretize state space matrices
        A_bar, B_bar = self._discretize(dt)
        
        # State update
        state = torch.tanh(state @ A_bar.T + x @ B_bar.T)
        
        # Output
        y = state @ self.C.T + x * self.D
        
        # Apply output projection and dropout
        y = self.output_proj(y)
        y = self.dropout(y)
        
        return y, state


class S4Block(nn.Module):
    """
    S4 Block with residual connection and feed-forward network.
    
    Architecture:
        x -> S4Layer -> + -> FFN -> + -> LayerNorm -> output
            |__________|   |________|
              residual       residual
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 32,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        """
        Initialize S4 block.
        
        Args:
            d_model: Model dimension
            d_state: State space dimension
            dropout: Dropout probability
            ff_mult: Feed-forward expansion multiplier
        """
        super().__init__()
        
        self.s4_layer = S4Layer(d_model, d_state, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through S4 block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            state: Previous S4 state
            
        Returns:
            Tuple of (output, new_state)
        """
        # S4 layer with residual
        s4_out, new_state = self.s4_layer(x, state)
        x = self.norm1(x + s4_out)
        
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        
        return x, new_state
    
    def step(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step forward pass (for inference).
        
        Args:
            x: Input tensor of shape (batch, d_model)
            state: Previous S4 state
            
        Returns:
            Tuple of (output, new_state)
        """
        # S4 layer with residual
        s4_out, new_state = self.s4_layer.step(x, state)
        x = self.norm1(x + s4_out)
        
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        
        return x, new_state
