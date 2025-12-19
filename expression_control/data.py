"""
Training data structures and JSON serialization for expression control.

This module defines the data structures for storing training samples
that pair facial features with servo angles, along with JSON serialization
for dataset persistence.

Requirements: 3.2, 3.7
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any

from expression_control.features import FaceFeatures
from expression_control.protocol import ServoCommandProtocol


# Valid expression labels for annotation
EXPRESSION_LABELS = frozenset(["happy", "sad", "angry", "surprised", "neutral"])


@dataclass
class TrainingDataSample:
    """
    Training data sample pairing facial features with servo angles.
    
    Attributes:
        timestamp: Time offset from session start in seconds.
        face_features: Extracted MediaPipe facial features.
        servo_angles: Dictionary mapping servo names to angle values [0, 180].
        expression_label: Optional expression annotation (happy, sad, angry, surprised, neutral).
        video_frame_path: Optional path to the raw video frame for debugging.
    """
    
    timestamp: float
    face_features: FaceFeatures
    servo_angles: Dict[str, int]
    expression_label: Optional[str] = None
    video_frame_path: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate the sample data after initialization."""
        if self.expression_label is not None and self.expression_label not in EXPRESSION_LABELS:
            raise ValueError(
                f"Invalid expression label '{self.expression_label}'. "
                f"Must be one of: {sorted(EXPRESSION_LABELS)}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the sample to a dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the sample.
        """
        return {
            "timestamp": self.timestamp,
            "features": {
                "left_eye_aspect_ratio": self.face_features.left_eye_aspect_ratio,
                "right_eye_aspect_ratio": self.face_features.right_eye_aspect_ratio,
                "eye_gaze_horizontal": self.face_features.eye_gaze_horizontal,
                "eye_gaze_vertical": self.face_features.eye_gaze_vertical,
                "left_eyebrow_height": self.face_features.left_eyebrow_height,
                "right_eyebrow_height": self.face_features.right_eyebrow_height,
                "eyebrow_furrow": self.face_features.eyebrow_furrow,
                "mouth_open_ratio": self.face_features.mouth_open_ratio,
                "mouth_width_ratio": self.face_features.mouth_width_ratio,
                "lip_pucker": self.face_features.lip_pucker,
                "smile_intensity": self.face_features.smile_intensity,
                "head_pitch": self.face_features.head_pitch,
                "head_yaw": self.face_features.head_yaw,
                "head_roll": self.face_features.head_roll,
            },
            "servo_angles": [self.servo_angles[name] for name in ServoCommandProtocol.SERVO_ORDER],
            "expression_label": self.expression_label,
            "video_frame_path": self.video_frame_path,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingDataSample":
        """
        Create a TrainingDataSample from a dictionary.
        
        Args:
            data: Dictionary with sample data (as produced by to_dict()).
            
        Returns:
            New TrainingDataSample instance.
            
        Raises:
            KeyError: If required fields are missing.
            ValueError: If data is invalid.
        """
        features_dict = data["features"]
        face_features = FaceFeatures(
            left_eye_aspect_ratio=float(features_dict["left_eye_aspect_ratio"]),
            right_eye_aspect_ratio=float(features_dict["right_eye_aspect_ratio"]),
            eye_gaze_horizontal=float(features_dict["eye_gaze_horizontal"]),
            eye_gaze_vertical=float(features_dict["eye_gaze_vertical"]),
            left_eyebrow_height=float(features_dict["left_eyebrow_height"]),
            right_eyebrow_height=float(features_dict["right_eyebrow_height"]),
            eyebrow_furrow=float(features_dict["eyebrow_furrow"]),
            mouth_open_ratio=float(features_dict["mouth_open_ratio"]),
            mouth_width_ratio=float(features_dict["mouth_width_ratio"]),
            lip_pucker=float(features_dict["lip_pucker"]),
            smile_intensity=float(features_dict["smile_intensity"]),
            head_pitch=float(features_dict["head_pitch"]),
            head_yaw=float(features_dict["head_yaw"]),
            head_roll=float(features_dict["head_roll"]),
            timestamp=float(data["timestamp"]),
        )
        
        # Convert servo angles list back to dict
        servo_angles_list = data["servo_angles"]
        if len(servo_angles_list) != len(ServoCommandProtocol.SERVO_ORDER):
            raise ValueError(
                f"Expected {len(ServoCommandProtocol.SERVO_ORDER)} servo angles, "
                f"got {len(servo_angles_list)}"
            )
        servo_angles = {
            name: int(angle) 
            for name, angle in zip(ServoCommandProtocol.SERVO_ORDER, servo_angles_list)
        }
        
        return cls(
            timestamp=float(data["timestamp"]),
            face_features=face_features,
            servo_angles=servo_angles,
            expression_label=data.get("expression_label"),
            video_frame_path=data.get("video_frame_path"),
        )
    
    def to_json(self) -> str:
        """
        Serialize the sample to a JSON string.
        
        Returns:
            JSON string representation of the sample.
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "TrainingDataSample":
        """
        Deserialize a sample from a JSON string.
        
        Args:
            json_str: JSON string (as produced by to_json()).
            
        Returns:
            New TrainingDataSample instance.
        """
        return cls.from_dict(json.loads(json_str))



@dataclass
class TrainingDataset:
    """
    Collection of training data samples with metadata.
    
    JSON Schema:
    {
        "version": "1.0",
        "created_at": "ISO8601 timestamp",
        "total_samples": int,
        "fps": float,
        "servo_order": ["JL", "JR", ...],
        "samples": [TrainingDataSample.to_dict(), ...]
    }
    
    Attributes:
        version: Schema version string.
        created_at: ISO8601 timestamp of dataset creation.
        fps: Target frames per second during recording.
        samples: List of training data samples.
    """
    
    samples: List[TrainingDataSample] = field(default_factory=list)
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    fps: float = 30.0
    
    @property
    def total_samples(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def add_sample(self, sample: TrainingDataSample) -> None:
        """Add a sample to the dataset."""
        self.samples.append(sample)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the dataset to a dictionary for JSON serialization.
        
        Returns:
            Dictionary representation following the JSON schema.
        """
        return {
            "version": self.version,
            "created_at": self.created_at,
            "total_samples": self.total_samples,
            "fps": self.fps,
            "servo_order": list(ServoCommandProtocol.SERVO_ORDER),
            "samples": [sample.to_dict() for sample in self.samples],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingDataset":
        """
        Create a TrainingDataset from a dictionary.
        
        Args:
            data: Dictionary with dataset data (as produced by to_dict()).
            
        Returns:
            New TrainingDataset instance.
            
        Raises:
            KeyError: If required fields are missing.
            ValueError: If data is invalid or version is unsupported.
        """
        version = data.get("version", "1.0")
        if version != "1.0":
            raise ValueError(f"Unsupported dataset version: {version}")
        
        samples = [TrainingDataSample.from_dict(s) for s in data.get("samples", [])]
        
        return cls(
            samples=samples,
            version=version,
            created_at=data.get("created_at", datetime.now().isoformat()),
            fps=float(data.get("fps", 30.0)),
        )
    
    def to_json(self, indent: int = 2) -> str:
        """
        Serialize the dataset to a JSON string.
        
        Args:
            indent: JSON indentation level for pretty printing.
            
        Returns:
            JSON string representation of the dataset.
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> "TrainingDataset":
        """
        Deserialize a dataset from a JSON string.
        
        Args:
            json_str: JSON string (as produced by to_json()).
            
        Returns:
            New TrainingDataset instance.
        """
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: str) -> None:
        """
        Save the dataset to a JSON file.
        
        Args:
            path: File path to save to.
        """
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, path: str) -> "TrainingDataset":
        """
        Load a dataset from a JSON file.
        
        Args:
            path: File path to load from.
            
        Returns:
            New TrainingDataset instance.
        """
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())
