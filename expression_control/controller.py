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
        Create a new ExpressionController with an optional feature-to-angle mapper and configuration.
        
        Parameters:
            mapper (Optional[FeatureToAngleMapper]): Mapper that converts extracted face features into servo angles. If omitted, a default RuleMapper is used.
            config (Optional[ControllerConfig]): Controller configuration. If omitted, a default ControllerConfig() is created.
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
        """
        Ensure the configured camera is opened and configured for capture.
        
        If the camera is not already initialized, opens OpenCV VideoCapture with the configured camera_id
        and applies frame width, height, and target FPS settings.
        
        Returns:
            True if the camera is opened and configured successfully, False otherwise.
        """
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
        """
        Lazily create and store the face feature extractor on the controller.
        
        If an extractor is already present this is a no-op. Otherwise attempts to construct a
        FaceFeatureExtractor and assign it to self._extractor.
        
        Returns:
            True if the extractor is available after the call, False if initialization failed.
        """
        if self._extractor is not None:
            return True
        
        try:
            self._extractor = FaceFeatureExtractor()
            return True
        except Exception:
            return False
    
    def _init_serial(self) -> bool:
        """
        Initialize and connect the serial manager if a serial port is configured.
        
        If no serial port is configured, this function treats serial as disabled and returns True.
        When a serial port is configured and no serial manager exists yet, it creates a SerialManager,
        assigns it to self._serial, and attempts to connect.
        
        Returns:
            True if serial is disabled, already initialized, or the connection succeeded; False if connecting failed.
        """
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
        Initialize the camera, feature extractor, and optional serial connection.
        
        Attempts to initialize the camera and the face feature extractor; also tries to establish a serial connection if configured. A failure to initialize the camera or extractor causes initialization to fail; a failure to connect serial is tolerated (the controller continues with serial disabled).
        
        Returns:
            True if the camera and feature extractor were initialized successfully (serial may be unavailable), False otherwise.
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
        Compute servo angle commands from a captured BGR camera frame.
        
        Processes the provided BGR image to extract facial features and map them to a dictionary of servo angles. If the feature extractor is not initialized or no face is detected, this returns None unless the time since the last detected face exceeds the configured face_lost_timeout and send_neutral_on_lost is True, in which case the mapper's neutral angles are returned.
        
        Parameters:
            frame (np.ndarray): BGR image from the camera (HxWx3).
        
        Returns:
            dict[str, int] | None: Mapping of servo identifiers to angle values, or `None` when no angles should be produced.
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
        Transmit servo angles to the robot over serial if a serial connection is configured.
        
        Parameters:
            angles (Dict[str, int]): Mapping from servo identifiers to target angles (integer degrees).
        
        Returns:
            bool: `True` if the command was sent (or serial is disabled), `False` if sending failed.
        """
        if self._serial is None:
            return True
        
        command = ServoCommandProtocol.encode(angles)
        return self._serial.send_command(command)
    
    def step(self) -> Optional[Dict[str, int]]:
        """
        Run a single control-loop iteration: capture one camera frame, process it into servo angles, invoke callbacks, and send angles to the robot if available.
        
        Returns:
            Optional[Dict[str, int]]: Mapping of servo identifiers to angle degrees when a face is detected; `None` if no frame was read or no angles were produced.
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
        Run the controller main loop that captures camera frames, processes facial features into servo angles, sends angles to the robot, and optionally displays video.
        
        Parameters:
            show_video (bool): If True, display a live video window with debug overlay and allow quitting with 'q'.
            on_frame (Optional[Callable]): Optional callback invoked every frame as on_frame(frame, angles).
            on_angles (Optional[Callable]): Optional callback invoked when angles are produced as on_angles(angles).
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
        """
        Overlay runtime debug information and key servo angles onto the provided video frame.
        
        Parameters:
            frame (np.ndarray): BGR image to draw overlays on; modified in place.
            angles (Dict[str, int]): Mapping of servo angle values. Expected keys:
                - 'JL' : jaw angle value
                - 'CUL': smile (mouth) angle value
                - 'LR' : left-right eye angle value
                - 'UD' : up-down eye angle value
        """
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
        """
        Stop the controller and release all acquired resources.
        
        Closes and clears the camera, feature extractor, and serial connection (if present), destroys any OpenCV windows, and stops the run loop. If frames were processed, prints a brief summary with total frames and average FPS.
        """
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
        """
        Enter context by initializing the controller and returning the controller instance.
        
        Initializes internal components (camera, extractor, optional serial) via initialize() and returns self for use as a context manager.
        
        Returns:
            self: The initialized ExpressionController instance.
        """
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager, stop the controller, and release resources.
        
        Parameters:
            exc_type (Optional[type]): Exception type if the block raised, otherwise None. Ignored.
            exc_val (Optional[BaseException]): Exception instance if raised, otherwise None. Ignored.
            exc_tb (Optional[types.TracebackType]): Traceback if an exception was raised, otherwise None. Ignored.
        """
        self.stop()