"""
Unified expression controller that coordinates all components.

This is the main entry point for the expression control system.
It handles camera input, feature extraction, mapping, and servo output.
"""

import time
from dataclasses import dataclass
from typing import Optional, Dict, Callable

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from expression_control.extractor import FaceFeatureExtractor
from expression_control.features import FaceFeatures
from expression_control.mappers.base import FeatureToAngleMapper
from expression_control.mappers.rule_mapper import RuleMapper
from expression_control.protocol import ServoCommandProtocol
from expression_control.serial_manager import SerialManager


@dataclass
class ControllerConfig:
    """Configuration for ExpressionController."""
    
    # Camera settings
    camera_id: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    
    # Serial settings
    serial_port: Optional[str] = None
    serial_baudrate: int = 115200
    
    # Behavior settings
    face_lost_timeout: float = 0.5  # Seconds before resetting to neutral
    send_neutral_on_lost: bool = True


class ExpressionController:
    """
    Unified controller for expression control system.
    
    Coordinates:
    - Camera capture
    - MediaPipe feature extraction
    - Feature-to-angle mapping (pluggable)
    - Serial communication to robot
    
    Usage:
        # With rule-based mapper (default)
        controller = ExpressionController()
        controller.run()
        
        # With neural network mapper
        from expression_control.mappers import LiquidS4Mapper
        mapper = LiquidS4Mapper.from_checkpoint("model.pt")
        controller = ExpressionController(mapper=mapper)
        controller.run()
    """
    
    def __init__(
        self,
        mapper: Optional[FeatureToAngleMapper] = None,
        config: Optional[ControllerConfig] = None,
    ):
        """
        Initialize the controller.
        
        Args:
            mapper: Feature-to-angle mapper. Defaults to RuleMapper.
            config: Controller configuration. Uses defaults if None.
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required: pip install opencv-python")
        
        self.config = config or ControllerConfig()
        self.mapper = mapper or RuleMapper()
        
        # Components (initialized lazily)
        self._camera: Optional[cv2.VideoCapture] = None
        self._extractor: Optional[FaceFeatureExtractor] = None
        self._serial: Optional[SerialManager] = None
        
        # State
        self._running = False
        self._last_face_time = 0.0
        self._frame_count = 0
        self._start_time = 0.0
        
        # Callbacks
        self._on_frame: Optional[Callable] = None
        self._on_angles: Optional[Callable] = None
    
    def _init_camera(self) -> bool:
        """Initialize camera capture."""
        if self._camera is not None:
            return True
        
        self._camera = cv2.VideoCapture(self.config.camera_id)
        if not self._camera.isOpened():
            return False
        
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self._camera.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        
        return True
    
    def _init_extractor(self) -> bool:
        """Initialize feature extractor."""
        if self._extractor is not None:
            return True
        
        try:
            self._extractor = FaceFeatureExtractor()
            return True
        except Exception:
            return False
    
    def _init_serial(self) -> bool:
        """Initialize serial connection."""
        if self.config.serial_port is None:
            return True  # Serial disabled
        
        if self._serial is not None:
            return True
        
        self._serial = SerialManager(
            port=self.config.serial_port,
            baudrate=self.config.serial_baudrate,
        )
        return self._serial.connect()
    
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if all components initialized successfully.
        """
        if not self._init_camera():
            print(f"Failed to open camera {self.config.camera_id}")
            return False
        
        if not self._init_extractor():
            print("Failed to initialize feature extractor")
            return False
        
        if not self._init_serial():
            print(f"Warning: Could not connect to {self.config.serial_port}")
            # Continue without serial
        
        return True
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, int]]:
        """
        Process a single frame.
        
        Args:
            frame: BGR image from camera.
            
        Returns:
            Dictionary of servo angles, or None if no face detected.
        """
        if self._extractor is None:
            return None
        
        # Extract features
        features = self._extractor.extract(frame)
        
        if features is None:
            # No face detected
            elapsed = time.time() - self._last_face_time
            
            if elapsed > self.config.face_lost_timeout:
                self.mapper.reset()
                
                if self.config.send_neutral_on_lost:
                    return self.mapper.get_neutral_dict()
            
            return None
        
        # Update face detection time
        self._last_face_time = time.time()
        
        # Map features to angles
        angles = self.mapper.map_to_dict(features)
        
        return angles
    
    def send_angles(self, angles: Dict[str, int]) -> bool:
        """
        Send angles to robot via serial.
        
        Args:
            angles: Dictionary mapping servo names to angles.
            
        Returns:
            True if sent successfully (or serial disabled).
        """
        if self._serial is None:
            return True
        
        command = ServoCommandProtocol.encode(angles)
        return self._serial.send_command(command)
    
    def step(self) -> Optional[Dict[str, int]]:
        """
        Run one iteration of the control loop.
        
        Returns:
            Servo angles if face detected, None otherwise.
        """
        if self._camera is None:
            return None
        
        ret, frame = self._camera.read()
        if not ret:
            return None
        
        # Process frame
        angles = self.process_frame(frame)
        
        # Callback
        if self._on_frame is not None:
            self._on_frame(frame, angles)
        
        # Send to robot
        if angles is not None:
            self.send_angles(angles)
            
            if self._on_angles is not None:
                self._on_angles(angles)
        
        self._frame_count += 1
        
        return angles
    
    def run(
        self,
        show_video: bool = False,
        on_frame: Optional[Callable] = None,
        on_angles: Optional[Callable] = None,
    ) -> None:
        """
        Run the main control loop.
        
        Args:
            show_video: Whether to display video window.
            on_frame: Callback(frame, angles) called each frame.
            on_angles: Callback(angles) called when angles are computed.
        """
        if not self.initialize():
            return
        
        self._on_frame = on_frame
        self._on_angles = on_angles
        self._running = True
        self._start_time = time.time()
        self._frame_count = 0
        self._last_face_time = time.time()
        
        target_interval = 1.0 / self.config.target_fps
        
        print("Expression controller started. Press 'q' to quit.")
        
        try:
            while self._running:
                loop_start = time.time()
                
                # Process one frame
                ret, frame = self._camera.read()
                if not ret:
                    continue
                
                angles = self.process_frame(frame)
                
                # Send to robot
                if angles is not None:
                    self.send_angles(angles)
                    
                    if self._on_angles is not None:
                        self._on_angles(angles)
                
                # Show video
                if show_video:
                    if angles is not None:
                        self._draw_debug(frame, angles)
                    else:
                        cv2.putText(frame, "No face", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    cv2.imshow("Expression Control", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Frame callback
                if self._on_frame is not None:
                    self._on_frame(frame, angles)
                
                self._frame_count += 1
                
                # Maintain frame rate
                elapsed = time.time() - loop_start
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
        
        except KeyboardInterrupt:
            print("\nInterrupted")
        
        finally:
            self.stop()
    
    def _draw_debug(self, frame: np.ndarray, angles: Dict[str, int]) -> None:
        """Draw debug info on frame."""
        fps = self._frame_count / (time.time() - self._start_time + 0.001)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Key angles
        y = 60
        texts = [
            f"Jaw: {angles['JL']}",
            f"Smile: {angles['CUL']}",
            f"Eye: {angles['LR']}/{angles['UD']}",
        ]
        for text in texts:
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y += 20
    
    def stop(self) -> None:
        """Stop the controller and release resources."""
        self._running = False
        
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        
        if self._extractor is not None:
            self._extractor.close()
            self._extractor = None
        
        if self._serial is not None:
            self._serial.disconnect()
            self._serial = None
        
        cv2.destroyAllWindows()
        
        if self._frame_count > 0:
            elapsed = time.time() - self._start_time
            print(f"Processed {self._frame_count} frames in {elapsed:.1f}s "
                  f"({self._frame_count/elapsed:.1f} FPS)")
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
