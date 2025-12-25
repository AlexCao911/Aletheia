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
        Load an ONNX model and initialize mapper state.
        
        Creates an ONNX Runtime InferenceSession from `onnx_path` and records the model's first input name for inference. Initializes the mapper's neutral angles to `neutral_angles` if provided, otherwise to 90.0 degrees for each of `NUM_SERVOS`. Raises ImportError if `onnxruntime` is not installed.
        
        Parameters:
            onnx_path (str): Filesystem path to the exported ONNX model.
            neutral_angles (Optional[np.ndarray]): Optional array of neutral servo angles; used as fallback when mapping.
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
        """
        Map facial features to servo angles using the loaded ONNX model.
        
        Converts the provided FaceFeatures into a 21-element array of servo angles in degrees; values are clipped to the valid range 0.0–180.0.
        
        Parameters:
            features (FaceFeatures): Facial feature values used as model input.
        
        Returns:
            np.ndarray: Shape (21,) array of servo angles in degrees, dtype float64, clipped to [0.0, 180.0].
        """
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
        Perform a reset of any mapper-held runtime state.
        
        For ONNX-backed models this is a no-op because the exported model does not expose persistent inferencer state per instance. For stateful S4/LTC inference, embed state handling into the exported model or provide input sequences that carry state; this method does not modify or clear any external model state.
        """
        pass
    
    def get_neutral_angles(self) -> np.ndarray:
        """
        Return the mapper's neutral servo angles.
        
        Returns:
            np.ndarray: A copy of the internal neutral angles array (shape (NUM_SERVOS,), degrees).
        """
        return self._neutral.copy()