"""
Data collection tool for synchronized capture of facial features and servo angles.

This module provides the DataCollector class for recording training data
that pairs MediaPipe facial features with servo angle states.

Requirements: 3.1, 3.4, 3.8
"""

import logging
import os
import time
from typing import Dict, Optional, List

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from expression_control.data import (
    TrainingDataSample, 
    TrainingDataset, 
    EXPRESSION_LABELS
)
from expression_control.extractor import FaceFeatureExtractor
from expression_control.features import FaceFeatures
from expression_control.serial_manager import SerialManager
from expression_control.protocol import ServoCommandProtocol


logger = logging.getLogger(__name__)


class DataCollector:
    """
    Data collection tool for recording synchronized facial features and servo angles.
    
    Records paired data of:
    - MediaPipe facial landmarks and extracted features
    - Current servo angle states from MouthMaster Pico
    - Optional expression labels for supervised learning
    - Optional raw video frames for debugging
    
    Usage:
        collector = DataCollector(camera_id=0, serial_port="/dev/ttyACM0")
        collector.start_session(expression_label="happy")
        collector.record_session(duration_seconds=60)
        collector.save_dataset("data/happy_session.json")
    """
    
    DEFAULT_FPS = 30.0
    ANGLE_READBACK_COMMAND = "readangles"
    
    def __init__(
        self,
        camera_id: int = 0,
        serial_port: Optional[str] = None,
        save_frames: bool = False,
        frames_dir: str = "frames",
        min_detection_confidence: float = 0.5,
    ):
        """
        Initialize the data collector.
        
        Args:
            camera_id: Camera device ID for OpenCV VideoCapture.
            serial_port: Serial port for MouthMaster Pico. If None, servo angles
                        will not be recorded (useful for testing).
            save_frames: Whether to save raw video frames for debugging.
            frames_dir: Directory to save video frames.
            min_detection_confidence: MediaPipe face detection confidence threshold.
        """
        if not CV2_AVAILABLE:
            raise ImportError(
                "OpenCV is not installed. Install with: pip install opencv-python"
            )
        
        self.camera_id = camera_id
        self.serial_port = serial_port
        self.save_frames = save_frames
        self.frames_dir = frames_dir
        self.min_detection_confidence = min_detection_confidence
        
        # Components (initialized lazily)
        self._camera: Optional["cv2.VideoCapture"] = None
        self._extractor: Optional[FaceFeatureExtractor] = None
        self._serial: Optional[SerialManager] = None
        
        # Session state
        self._dataset: Optional[TrainingDataset] = None
        self._session_active = False
        self._current_expression: Optional[str] = None
        self._frame_count = 0
        
        # Default servo angles (neutral position)
        self._default_angles: Dict[str, int] = {
            name: 90 for name in ServoCommandProtocol.SERVO_ORDER
        }
        self._last_known_angles: Dict[str, int] = self._default_angles.copy()
    
    def _init_camera(self) -> bool:
        """Initialize camera capture."""
        if self._camera is not None and self._camera.isOpened():
            return True
        
        self._camera = cv2.VideoCapture(self.camera_id)
        if not self._camera.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            return False
        
        # Set camera properties for consistent capture
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._camera.set(cv2.CAP_PROP_FPS, self.DEFAULT_FPS)
        
        logger.info(f"Camera {self.camera_id} initialized")
        return True
    
    def _init_extractor(self) -> bool:
        """Initialize MediaPipe face feature extractor."""
        if self._extractor is not None:
            return True
        
        try:
            self._extractor = FaceFeatureExtractor(
                min_detection_confidence=self.min_detection_confidence
            )
            logger.info("Face feature extractor initialized")
            return True
        except ImportError as e:
            logger.error(f"Failed to initialize extractor: {e}")
            return False
    
    def _init_serial(self) -> bool:
        """Initialize serial connection to MouthMaster Pico."""
        if self.serial_port is None:
            logger.warning("No serial port specified, servo angles will use defaults")
            return True
        
        if self._serial is not None and self._serial.is_connected:
            return True
        
        self._serial = SerialManager(port=self.serial_port)
        if not self._serial.connect():
            logger.warning(
                f"Failed to connect to {self.serial_port}, "
                "servo angles will use defaults"
            )
            return True  # Continue without serial
        
        logger.info(f"Serial connection to {self.serial_port} established")
        return True
    
    def _read_servo_angles(self) -> Dict[str, int]:
        """
        Read current servo angles from MouthMaster Pico.
        
        Returns:
            Dictionary mapping servo names to current angles.
            Returns last known angles if read fails.
        """
        if self._serial is None or not self._serial.is_connected:
            return self._last_known_angles.copy()
        
        # Send readback command
        if not self._serial.send_command(self.ANGLE_READBACK_COMMAND):
            return self._last_known_angles.copy()
        
        # Read response
        response = self._serial.read_response(timeout=0.1)
        if response is None:
            return self._last_known_angles.copy()
        
        # Parse response (expected format: "angles:A1,A2,...,A21")
        angles = ServoCommandProtocol.decode(response)
        if angles is not None:
            self._last_known_angles = angles
            return angles
        
        return self._last_known_angles.copy()
    
    def start_session(
        self,
        expression_label: Optional[str] = None,
        fps: float = DEFAULT_FPS,
    ) -> bool:
        """
        Start a new recording session.
        
        Args:
            expression_label: Optional expression label for all samples in session.
            fps: Target frames per second.
            
        Returns:
            True if session started successfully.
        """
        if expression_label is not None and expression_label not in EXPRESSION_LABELS:
            raise ValueError(
                f"Invalid expression label '{expression_label}'. "
                f"Must be one of: {sorted(EXPRESSION_LABELS)}"
            )
        
        # Initialize components
        if not self._init_camera():
            return False
        if not self._init_extractor():
            return False
        if not self._init_serial():
            return False
        
        # Create frames directory if saving frames
        if self.save_frames:
            os.makedirs(self.frames_dir, exist_ok=True)
        
        # Initialize dataset
        self._dataset = TrainingDataset(fps=fps)
        self._current_expression = expression_label
        self._frame_count = 0
        self._session_active = True
        
        logger.info(
            f"Recording session started"
            + (f" with label '{expression_label}'" if expression_label else "")
        )
        return True
    
    def capture_sample(self) -> Optional[TrainingDataSample]:
        """
        Capture a single synchronized sample.
        
        Returns:
            TrainingDataSample if capture successful, None otherwise.
        """
        if not self._session_active:
            logger.warning("No active session, call start_session() first")
            return None
        
        if self._camera is None or self._extractor is None:
            return None
        
        # Capture frame
        ret, frame = self._camera.read()
        if not ret or frame is None:
            logger.warning("Failed to capture frame")
            return None
        
        # Extract facial features
        features = self._extractor.extract(frame)
        if features is None:
            logger.debug("No face detected in frame")
            return None
        
        # Read servo angles
        servo_angles = self._read_servo_angles()
        
        # Save frame if requested
        frame_path: Optional[str] = None
        if self.save_frames:
            frame_path = os.path.join(
                self.frames_dir, 
                f"frame_{self._frame_count:06d}.jpg"
            )
            cv2.imwrite(frame_path, frame)
        
        # Create sample
        sample = TrainingDataSample(
            timestamp=features.timestamp,
            face_features=features,
            servo_angles=servo_angles,
            expression_label=self._current_expression,
            video_frame_path=frame_path,
        )
        
        self._frame_count += 1
        return sample
    
    def record_session(
        self,
        duration_seconds: float,
        expression_label: Optional[str] = None,
    ) -> int:
        """
        Record a training session for the specified duration.
        
        Args:
            duration_seconds: Recording duration in seconds.
            expression_label: Optional expression label (overrides session label).
            
        Returns:
            Number of samples recorded.
        """
        # Start session if not already active
        if not self._session_active:
            if not self.start_session(expression_label=expression_label):
                return 0
        elif expression_label is not None:
            self._current_expression = expression_label
        
        if self._dataset is None:
            return 0
        
        start_time = time.time()
        target_interval = 1.0 / self._dataset.fps
        samples_recorded = 0
        
        logger.info(f"Recording for {duration_seconds} seconds...")
        
        while time.time() - start_time < duration_seconds:
            frame_start = time.time()
            
            # Capture and add sample
            sample = self.capture_sample()
            if sample is not None:
                # Adjust timestamp relative to session start
                sample.timestamp = time.time() - start_time
                self._dataset.add_sample(sample)
                samples_recorded += 1
            
            # Maintain target frame rate
            elapsed = time.time() - frame_start
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
        
        logger.info(f"Recorded {samples_recorded} samples")
        return samples_recorded
    
    def add_sample(self, sample: TrainingDataSample) -> None:
        """
        Add a pre-created sample to the current dataset.
        
        Args:
            sample: TrainingDataSample to add.
        """
        if self._dataset is None:
            self._dataset = TrainingDataset()
        self._dataset.add_sample(sample)
    
    def set_servo_angles(self, angles: Dict[str, int]) -> bool:
        """
        Set servo angles (for manual data collection mode).
        
        Args:
            angles: Dictionary mapping servo names to angle values.
            
        Returns:
            True if angles were sent successfully.
        """
        if self._serial is None:
            self._last_known_angles = angles.copy()
            return True
        
        if self._serial.send_angles(angles):
            self._last_known_angles = angles.copy()
            return True
        return False
    
    def get_dataset(self) -> Optional[TrainingDataset]:
        """Get the current dataset."""
        return self._dataset
    
    def save_dataset(self, output_path: str) -> bool:
        """
        Save the recorded dataset to a JSON file.
        
        Args:
            output_path: File path to save the dataset.
            
        Returns:
            True if save successful.
        """
        if self._dataset is None or self._dataset.total_samples == 0:
            logger.warning("No samples to save")
            return False
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            self._dataset.save(output_path)
            logger.info(
                f"Saved {self._dataset.total_samples} samples to {output_path}"
            )
            return True
        except IOError as e:
            logger.error(f"Failed to save dataset: {e}")
            return False
    
    def stop_session(self) -> None:
        """Stop the current recording session."""
        self._session_active = False
        logger.info("Recording session stopped")
    
    def close(self) -> None:
        """Release all resources."""
        self.stop_session()
        
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        
        if self._extractor is not None:
            self._extractor.close()
            self._extractor = None
        
        if self._serial is not None:
            self._serial.disconnect()
            self._serial = None
        
        logger.info("DataCollector resources released")
    
    def __enter__(self) -> "DataCollector":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
