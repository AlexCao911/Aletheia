"""
LiquidS4-based mapper from facial features to servo angles.

Uses ONNX Runtime for inference - suitable for edge deployment (Raspberry Pi).
"""

from typing import Optional

import numpy as np

from expression_control.features import FaceFeatures
from expression_control.mappers.base import FeatureToAngleMapper


class LiquidS4Mapper(FeatureToAngleMapper):
    """
    Neural network mapper using trained LNN-S4 model (ONNX format).
    
    This mapper uses a trained and exported ONNX model for inference.
    Designed for edge deployment on devices like Raspberry Pi.
    
    Features:
    - Learned non-linear mappings from training data
    - Built-in temporal modeling from S4/LTC layers
    - Lightweight ONNX runtime (no PyTorch needed at inference)
    
    Usage:
        mapper = LiquidS4Mapper("model.onnx")
        angles = mapper.map(features)
    """
    
    def __init__(
        self,
        onnx_path: str,
        neutral_angles: Optional[np.ndarray] = None,
    ):
        """
        Initialize with ONNX model.
        
        Args:
            onnx_path: Path to exported .onnx model file.
            neutral_angles: Optional neutral angles. Defaults to 90 for all.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required: pip install onnxruntime"
            )
        
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # Neutral angles for fallback
        self._neutral = neutral_angles if neutral_angles is not None else \
            np.full(self.NUM_SERVOS, 90.0, dtype=np.float64)
    
    def map(self, features: FaceFeatures) -> np.ndarray:
        """Map facial features to servo angles using ONNX model."""
        # Prepare input: (batch=1, seq_len=1, features=14)
        feature_array = features.to_array().astype(np.float32)
        x = feature_array.reshape(1, 1, -1)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: x})
        angles = outputs[0].squeeze()  # Shape: (21,)
        
        # Clip to valid range
        angles = np.clip(angles, 0.0, 180.0)
        
        return angles.astype(np.float64)
    
    def reset(self) -> None:
        """
        Reset internal state.
        
        Note: ONNX models are stateless per-call. For stateful inference
        with S4/LTC, the model should be exported with state handling,
        or use a sequence of frames as input.
        """
        pass
    
    def get_neutral_angles(self) -> np.ndarray:
        """Get neutral angles."""
        return self._neutral.copy()
