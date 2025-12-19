"""
FaceFeatures dataclass for MediaPipe extracted facial features.

This module defines the data structure for storing facial features
extracted from MediaPipe Face Mesh, including eye ratios, eyebrow
positions, mouth features, and head pose.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class FaceFeatures:
    """MediaPipe 提取的面部特征
    
    Contains 14 feature fields extracted from MediaPipe Face Mesh:
    - Eye features (4): aspect ratios and gaze direction
    - Eyebrow features (3): heights and furrow
    - Mouth features (4): openness, width, pucker, smile
    - Head pose (3): pitch, yaw, roll
    
    All ratio values are normalized to [0, 1].
    Gaze values are normalized to [-1, 1].
    Head pose angles are in degrees.
    """
    
    # Eye features
    left_eye_aspect_ratio: float      # 左眼开合度 [0, 1]
    right_eye_aspect_ratio: float     # 右眼开合度 [0, 1]
    eye_gaze_horizontal: float        # 水平视线方向 [-1, 1]
    eye_gaze_vertical: float          # 垂直视线方向 [-1, 1]
    
    # Eyebrow features
    left_eyebrow_height: float        # 左眉高度 [0, 1]
    right_eyebrow_height: float       # 右眉高度 [0, 1]
    eyebrow_furrow: float             # 眉头皱起程度 [0, 1]
    
    # Mouth features
    mouth_open_ratio: float           # 嘴巴张开度 [0, 1]
    mouth_width_ratio: float          # 嘴巴宽度 [0, 1]
    lip_pucker: float                 # 嘴唇噘起程度 [0, 1]
    smile_intensity: float            # 微笑强度 [0, 1]
    
    # Head pose (degrees)
    head_pitch: float                 # 俯仰角 (degrees)
    head_yaw: float                   # 偏航角 (degrees)
    head_roll: float                  # 翻滚角 (degrees)
    
    # Optional raw landmarks for debugging
    landmarks: Optional[np.ndarray] = field(default=None, repr=False)  # (478, 3)
    
    # Timestamp
    timestamp: float = 0.0

    # Number of feature fields (excluding landmarks and timestamp)
    NUM_FEATURES: int = field(default=14, init=False, repr=False)
    
    def to_array(self) -> np.ndarray:
        """
        Convert FaceFeatures to a numpy array for model input.
        
        Returns:
            np.ndarray: Shape (14,) containing all feature values in order:
                [left_eye_aspect_ratio, right_eye_aspect_ratio, 
                 eye_gaze_horizontal, eye_gaze_vertical,
                 left_eyebrow_height, right_eyebrow_height, eyebrow_furrow,
                 mouth_open_ratio, mouth_width_ratio, lip_pucker, smile_intensity,
                 head_pitch, head_yaw, head_roll]
        """
        return np.array([
            self.left_eye_aspect_ratio,
            self.right_eye_aspect_ratio,
            self.eye_gaze_horizontal,
            self.eye_gaze_vertical,
            self.left_eyebrow_height,
            self.right_eyebrow_height,
            self.eyebrow_furrow,
            self.mouth_open_ratio,
            self.mouth_width_ratio,
            self.lip_pucker,
            self.smile_intensity,
            self.head_pitch,
            self.head_yaw,
            self.head_roll,
        ], dtype=np.float32)
    
    @classmethod
    def from_array(cls, arr: np.ndarray, landmarks: Optional[np.ndarray] = None, 
                   timestamp: float = 0.0) -> 'FaceFeatures':
        """
        Create FaceFeatures from a numpy array.
        
        Args:
            arr: Shape (14,) array containing feature values in the same order
                 as to_array() output.
            landmarks: Optional raw landmarks array (478, 3)
            timestamp: Optional timestamp value
            
        Returns:
            FaceFeatures: New instance with values from the array
            
        Raises:
            ValueError: If array does not have exactly 14 elements
        """
        if len(arr) != 14:
            raise ValueError(f"Expected array of length 14, got {len(arr)}")
        
        return cls(
            left_eye_aspect_ratio=float(arr[0]),
            right_eye_aspect_ratio=float(arr[1]),
            eye_gaze_horizontal=float(arr[2]),
            eye_gaze_vertical=float(arr[3]),
            left_eyebrow_height=float(arr[4]),
            right_eyebrow_height=float(arr[5]),
            eyebrow_furrow=float(arr[6]),
            mouth_open_ratio=float(arr[7]),
            mouth_width_ratio=float(arr[8]),
            lip_pucker=float(arr[9]),
            smile_intensity=float(arr[10]),
            head_pitch=float(arr[11]),
            head_yaw=float(arr[12]),
            head_roll=float(arr[13]),
            landmarks=landmarks,
            timestamp=timestamp,
        )
    
    @classmethod
    def neutral(cls, timestamp: float = 0.0) -> 'FaceFeatures':
        """
        Create a neutral face features instance with default values.
        
        Useful for fallback when face detection fails.
        
        Returns:
            FaceFeatures: Instance with neutral expression values
        """
        return cls(
            left_eye_aspect_ratio=0.3,   # Partially open eyes
            right_eye_aspect_ratio=0.3,
            eye_gaze_horizontal=0.0,      # Looking straight
            eye_gaze_vertical=0.0,
            left_eyebrow_height=0.5,      # Neutral eyebrow position
            right_eyebrow_height=0.5,
            eyebrow_furrow=0.0,           # No furrow
            mouth_open_ratio=0.0,         # Closed mouth
            mouth_width_ratio=0.5,        # Neutral width
            lip_pucker=0.0,               # No pucker
            smile_intensity=0.0,          # No smile
            head_pitch=0.0,               # Looking straight
            head_yaw=0.0,
            head_roll=0.0,
            timestamp=timestamp,
        )
