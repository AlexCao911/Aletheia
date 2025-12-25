"""
Abstract base class for feature-to-angle mappers.

All mappers must implement this interface to be used with ExpressionController.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

from expression_control.features import FaceFeatures
from expression_control.protocol import ServoCommandProtocol


class FeatureToAngleMapper(ABC):
    """
    Abstract base class for mapping facial features to servo angles.
    
    All concrete mappers (RuleMapper, LiquidS4Mapper, etc.) must implement
    this interface. This allows the ExpressionController to work with any
    mapper implementation interchangeably.
    
    The mapper is responsible for:
    1. Converting 14-dim facial features to 21-dim servo angles
    2. Managing any internal state (e.g., for temporal smoothing)
    3. Providing neutral/default angles when needed
    """
    
    NUM_FEATURES = 14
    NUM_SERVOS = 21
    SERVO_ORDER = ServoCommandProtocol.SERVO_ORDER
    
    @abstractmethod
    def map(self, features: FaceFeatures) -> np.ndarray:
        """
        Map facial features to servo angles.
        
        Args:
            features: FaceFeatures object from MediaPipe extractor.
            
        Returns:
            numpy array of shape (21,) with servo angles in range [0, 180].
            Order follows ServoCommandProtocol.SERVO_ORDER.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset internal state.
        
        Call this when:
        - Face detection is lost and regained
        - Starting a new video sequence
        - Switching between different faces
        """
        pass
    
    @abstractmethod
    def get_neutral_angles(self) -> np.ndarray:
        """
        Get neutral position angles for all servos.
        
        Returns:
            numpy array of shape (21,) with neutral angles.
        """
        pass
    
    def map_to_dict(self, features: FaceFeatures) -> Dict[str, int]:
        """
        Map facial features to servo angles as a dictionary.
        
        Convenience method that converts array output to named dictionary.
        
        Args:
            features: FaceFeatures object from MediaPipe extractor.
            
        Returns:
            Dictionary mapping servo names to integer angles.
        """
        angles = self.map(features)
        return {
            name: int(round(angles[i]))
            for i, name in enumerate(self.SERVO_ORDER)
        }
    
    def get_neutral_dict(self) -> Dict[str, int]:
        """Get neutral angles as a dictionary."""
        angles = self.get_neutral_angles()
        return {
            name: int(round(angles[i]))
            for i, name in enumerate(self.SERVO_ORDER)
        }
