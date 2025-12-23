"""Temporal smoothing module for servo angle outputs.

This module provides exponential moving average (EMA) smoothing to prevent
servo jitter and ensure smooth transitions between predicted angles.
"""

import numpy as np
from typing import Optional


class TemporalSmoother:
    """指数移动平均时序平滑器 (Exponential Moving Average Temporal Smoother)
    
    Applies EMA smoothing to servo angle predictions to prevent jitter
    and ensure smooth transitions. The smoothing formula is:
    
        smoothed[t] = alpha * input[t] + (1 - alpha) * smoothed[t-1]
    
    A smaller alpha value results in smoother (but slower) transitions,
    while a larger alpha makes the output more responsive to changes.
    
    Attributes:
        alpha: EMA smoothing coefficient in range (0, 1]. Smaller = smoother.
        num_servos: Number of servo channels to smooth.
    """
    
    def __init__(self, alpha: float = 0.3, num_servos: int = 21):
        """Initialize the temporal smoother.
        
        Args:
            alpha: EMA smoothing coefficient. Must be in range (0, 1].
                   Smaller values produce smoother output but slower response.
                   Default is 0.3 which provides good balance.
            num_servos: Number of servo channels to smooth. Default is 21
                        for the full expression control system.
        
        Raises:
            ValueError: If alpha is not in range (0, 1] or num_servos < 1.
        """
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in range (0, 1], got {alpha}")
        if num_servos < 1:
            raise ValueError(f"num_servos must be >= 1, got {num_servos}")
        
        self.alpha = alpha
        self.num_servos = num_servos
        self._smoothed: Optional[np.ndarray] = None
    
    def smooth(self, angles: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to angle predictions.
        
        Uses exponential moving average to smooth the input angles.
        On the first call (or after reset), the input is returned unchanged
        to initialize the smoother state.
        
        Args:
            angles: Current frame's raw angle predictions. Shape should be
                    (num_servos,) or compatible with num_servos.
        
        Returns:
            Smoothed angles with the same shape as input. Values are
            guaranteed to be within the range of previously seen values.
        
        Raises:
            ValueError: If angles shape doesn't match num_servos.
        """
        angles = np.asarray(angles, dtype=np.float64)
        
        if angles.shape != (self.num_servos,):
            raise ValueError(
                f"Expected angles shape ({self.num_servos},), got {angles.shape}"
            )
        
        if self._smoothed is None:
            # First frame: initialize with input values
            self._smoothed = angles.copy()
        else:
            # Apply EMA: smoothed = alpha * input + (1 - alpha) * previous
            self._smoothed = self.alpha * angles + (1 - self.alpha) * self._smoothed
        
        return self._smoothed.copy()
    
    def reset(self):
        """Reset the smoother state for a new sequence.
        
        Call this method when starting a new video sequence or when
        the face detection is lost and regained. This ensures the
        smoother doesn't carry over state from previous sequences.
        """
        self._smoothed = None
    
    @property
    def is_initialized(self) -> bool:
        """Check if the smoother has been initialized with data.
        
        Returns:
            True if smooth() has been called at least once since
            initialization or last reset().
        """
        return self._smoothed is not None
    
    @property
    def current_state(self) -> Optional[np.ndarray]:
        """Get the current smoothed state.
        
        Returns:
            Copy of the current smoothed values, or None if not initialized.
        """
        if self._smoothed is None:
            return None
        return self._smoothed.copy()
