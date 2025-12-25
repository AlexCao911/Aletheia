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
        Convert facial features into a 21-element array of servo angles.
        
        Parameters:
            features (FaceFeatures): Facial keypoints/features produced by the MediaPipe extractor.
        
        Returns:
            numpy.ndarray: Array of shape (21,) containing servo angles in degrees (0–180). Values are ordered according to SERVO_ORDER.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the mapper's internal temporal and stateful data to their defaults.
        
        Call when face tracking is lost or regained, when starting a new video sequence, or when switching between different faces.
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
        Map facial features to a dictionary of servo angles.
        
        Parameters:
            features (FaceFeatures): Facial feature measurements (14-dim) from the MediaPipe extractor.
        
        Returns:
            Dict[str, int]: Mapping from servo name to its angle in degrees, with each angle rounded to the nearest integer.
        """
        angles = self.map(features)
        return {
            name: int(round(angles[i]))
            for i, name in enumerate(self.SERVO_ORDER)
        }
    
    def get_neutral_dict(self) -> Dict[str, int]:
        """
        Provide neutral servo angles as a mapping from servo name to integer angle.
        
        The mapping keys follow SERVO_ORDER; each value is the neutral angle in degrees rounded to the nearest integer (expected range 0–180).
        
        Returns:
            Dict[str, int]: Dictionary mapping servo name to rounded neutral angle.
        """
        angles = self.get_neutral_angles()
        return {
            name: int(round(angles[i]))
            for i, name in enumerate(self.SERVO_ORDER)
        }