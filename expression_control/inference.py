"""
Real-time inference engine for LNN-S4 expression control.

This module provides the LNNS4Inference class for ONNX runtime inference
and the InferenceEngine class that integrates the full pipeline:
camera -> MediaPipe -> model -> smoother -> serial.

Requirements: 4.3, 4.4, 6.1, 6.3, 6.4, 6.7, 6.8
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .features import FaceFeatures
from .protocol import ServoCommandProtocol

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for the inference engine.
    
    Attributes:
        model_path: Path to ONNX model file
        camera_id: Camera device ID (default 0)
        serial_port: Serial port for Pico communication
        smoothing_alpha: EMA smoothing coefficient (0, 1]
        face_timeout_ms: Timeout before transitioning to neutral (ms)
        fallback_enabled: Whether to use direct MediaPipe mapping as fallback
        sensitivity: Global sensitivity multiplier for features
        log_performance: Whether to log inference latency statistics
        target_fps: Target frames per second for inference loop
    """
    model_path: Optional[str] = None
    camera_id: int = 0
    serial_port: str = "/dev/ttyACM0"
    smoothing_alpha: float = 0.3
    face_timeout_ms: float = 500.0
    fallback_enabled: bool = True
    sensitivity: float = 1.0
    log_performance: bool = True
    target_fps: float = 30.0
    
    # Neutral servo positions (used when face detection fails)
    neutral_angles: Dict[str, int] = field(default_factory=lambda: {
        name: 90 for name in ServoCommandProtocol.SERVO_ORDER
    })


class LNNS4Inference:
    """ONNX runtime inference wrapper for LNN-S4 model.
    
    Handles model loading, state management, and single-frame prediction
    using ONNX runtime for efficient edge deployment.
    
    Requirements: 4.3, 4.4
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize ONNX inference session.
        
        Args:
            model_path: Path to ONNX model file
            device: Execution device ("cpu" or "cuda")
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ImportError: If onnxruntime is not installed
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed. Install with: pip install onnxruntime"
            )
        
        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Select execution provider based on device
        if device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers,
        )
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # Initialize states
        self._s4_states: Optional[np.ndarray] = None
        self._ltc_state: Optional[np.ndarray] = None
        
        logger.info(f"Loaded ONNX model from {model_path}")
        logger.debug(f"Input names: {self.input_names}")
        logger.debug(f"Output names: {self.output_names}")
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Single frame inference.
        
        Args:
            features: Face feature vector of shape (14,)
            
        Returns:
            Tuple of:
                - angles: Servo angles of shape (21,) in range [0, 180]
                - success: Whether inference succeeded
        """
        try:
            # Prepare input: (1, 1, 14) for batch=1, seq_len=1
            features_input = features.reshape(1, 1, -1).astype(np.float32)
            
            # Build input dict
            inputs = {'features': features_input}
            
            # Add states if available and model expects them
            if 's4_states' in self.input_names and self._s4_states is not None:
                inputs['s4_states'] = self._s4_states
            if 'ltc_state' in self.input_names and self._ltc_state is not None:
                inputs['ltc_state'] = self._ltc_state
            
            # Run inference
            outputs = self.session.run(self.output_names, inputs)
            
            # Parse outputs
            angles = outputs[0].squeeze()  # Remove batch and seq dims
            
            # Update states if model outputs them
            if len(outputs) > 1:
                self._s4_states = outputs[1]
            if len(outputs) > 2:
                self._ltc_state = outputs[2]
            
            # Clamp angles to valid range
            angles = np.clip(angles, 0, 180)
            
            return angles, True
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return np.full(21, 90.0), False
    
    def reset(self):
        """Reset hidden states for new sequence."""
        self._s4_states = None
        self._ltc_state = None
        logger.debug("Reset inference states")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.session is not None


class FallbackMapper:
    """Direct MediaPipe-to-servo mapping for fallback mode.
    
    Provides a simple rule-based mapping from facial features to servo
    angles when the LNN-S4 model is unavailable.
    
    Requirements: 6.8
    """

    # Servo name to feature mapping
    # Maps each servo to the facial feature(s) that control it
    SERVO_MAPPINGS = {
        # Mouth servos (11)
        "JL": ("mouth_open_ratio", 0.5, 90, 45),   # (feature, scale, center, range)
        "JR": ("mouth_open_ratio", 0.5, 90, 45),
        "LUL": ("smile_intensity", 1.0, 90, 30),
        "LUR": ("smile_intensity", 1.0, 90, 30),
        "LLL": ("smile_intensity", 1.0, 90, 30),
        "LLR": ("smile_intensity", 1.0, 90, 30),
        "CUL": ("mouth_width_ratio", 1.0, 90, 20),
        "CUR": ("mouth_width_ratio", 1.0, 90, 20),
        "CLL": ("lip_pucker", 1.0, 90, 25),
        "CLR": ("lip_pucker", 1.0, 90, 25),
        "TON": ("mouth_open_ratio", 0.3, 90, 20),
        # Eye servos (6)
        "LR": ("eye_gaze_horizontal", 1.0, 90, 30),
        "UD": ("eye_gaze_vertical", 1.0, 90, 25),
        "TL": ("left_eye_aspect_ratio", 1.0, 90, 35),
        "BL": ("left_eye_aspect_ratio", -1.0, 90, 35),
        "TR": ("right_eye_aspect_ratio", 1.0, 90, 35),
        "BR": ("right_eye_aspect_ratio", -1.0, 90, 35),
        # Brow servos (4)
        "LO": ("left_eyebrow_height", 1.0, 90, 30),
        "LI": ("eyebrow_furrow", -1.0, 90, 25),
        "RI": ("eyebrow_furrow", -1.0, 90, 25),
        "RO": ("right_eyebrow_height", 1.0, 90, 30),
    }
    
    def __init__(self, sensitivity: float = 1.0):
        """
        Initialize fallback mapper.
        
        Args:
            sensitivity: Global sensitivity multiplier for all mappings
        """
        self.sensitivity = sensitivity
    
    def map_features(self, features: FaceFeatures) -> Dict[str, int]:
        """
        Map facial features to servo angles using rule-based mapping.
        
        Args:
            features: Extracted facial features
            
        Returns:
            Dictionary mapping servo names to angles [0, 180]
        """
        angles = {}
        
        for servo_name, mapping in self.SERVO_MAPPINGS.items():
            feature_name, scale, center, range_val = mapping
            
            # Get feature value
            feature_value = getattr(features, feature_name, 0.0)
            
            # Apply mapping: angle = center + (feature * scale * sensitivity * range)
            # Features are typically in [-1, 1] or [0, 1]
            offset = feature_value * scale * self.sensitivity * range_val
            angle = center + offset
            
            # Clamp to valid range
            angles[servo_name] = int(np.clip(angle, 0, 180))
        
        return angles
    
    def map_array(self, features: FaceFeatures) -> np.ndarray:
        """
        Map facial features to servo angle array.
        
        Args:
            features: Extracted facial features
            
        Returns:
            Array of shape (21,) with servo angles in protocol order
        """
        angles_dict = self.map_features(features)
        return np.array([
            angles_dict[name] for name in ServoCommandProtocol.SERVO_ORDER
        ], dtype=np.float64)


class InferenceEngine:
    """Full inference pipeline integrating all components.
    
    Integrates camera, MediaPipe feature extraction, LNN-S4 model (or fallback),
    temporal smoothing, and serial communication for real-time expression control.
    
    Requirements: 6.1, 6.3, 6.4, 6.7, 6.8
    """
    
    def __init__(self, config: InferenceConfig):
        """
        Initialize inference engine with all components.
        
        Args:
            config: Inference configuration
        """
        self.config = config
        
        # Initialize components (lazy loading)
        self._camera = None
        self._extractor = None
        self._model: Optional[LNNS4Inference] = None
        self._smoother = None
        self._serial = None
        self._fallback_mapper = None
        
        # State tracking
        self._last_face_time: float = 0.0
        self._last_valid_angles: Optional[np.ndarray] = None
        self._using_fallback: bool = False
        self._is_initialized: bool = False
        
        # Performance tracking
        self._latencies: List[float] = []
        self._frame_count: int = 0
    
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if initialization successful
            
        Requirements: 6.5 (initialization within 10 seconds)
        """
        start_time = time.time()
        
        try:
            # Initialize MediaPipe extractor
            from .extractor import FaceFeatureExtractor
            self._extractor = FaceFeatureExtractor()
            logger.info("MediaPipe Face Mesh initialized")
            
            # Initialize temporal smoother
            from .smoother import TemporalSmoother
            self._smoother = TemporalSmoother(
                alpha=self.config.smoothing_alpha,
                num_servos=21,
            )
            logger.info("Temporal smoother initialized")
            
            # Initialize fallback mapper
            self._fallback_mapper = FallbackMapper(
                sensitivity=self.config.sensitivity
            )
            
            # Try to load ONNX model
            if self.config.model_path and Path(self.config.model_path).exists():
                try:
                    self._model = LNNS4Inference(self.config.model_path)
                    self._using_fallback = False
                    logger.info(f"LNN-S4 model loaded from {self.config.model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load model: {e}")
                    if self.config.fallback_enabled:
                        self._using_fallback = True
                        logger.info("Using fallback mode (direct MediaPipe mapping)")
                    else:
                        raise
            else:
                if self.config.fallback_enabled:
                    self._using_fallback = True
                    logger.info("No model specified, using fallback mode")
                else:
                    raise FileNotFoundError("No model specified and fallback disabled")
            
            # Initialize camera
            try:
                import cv2
                self._camera = cv2.VideoCapture(self.config.camera_id)
                if not self._camera.isOpened():
                    raise RuntimeError(f"Failed to open camera {self.config.camera_id}")
                logger.info(f"Camera {self.config.camera_id} initialized")
            except ImportError:
                logger.warning("OpenCV not available, camera disabled")
                self._camera = None
            
            # Initialize serial connection
            try:
                from .serial_manager import SerialManager
                self._serial = SerialManager(port=self.config.serial_port)
                if self._serial.connect():
                    logger.info(f"Serial connected to {self.config.serial_port}")
                else:
                    logger.warning("Serial connection failed, running without hardware")
            except Exception as e:
                logger.warning(f"Serial initialization failed: {e}")
                self._serial = None
            
            # Initialize neutral angles
            self._last_valid_angles = np.array([
                self.config.neutral_angles[name]
                for name in ServoCommandProtocol.SERVO_ORDER
            ], dtype=np.float64)
            
            self._is_initialized = True
            
            elapsed = time.time() - start_time
            logger.info(f"Initialization complete in {elapsed:.2f}s")
            
            return elapsed < 10.0  # Requirement: within 10 seconds
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Process a single video frame through the full pipeline.
        
        Args:
            frame: BGR video frame
            
        Returns:
            Tuple of:
                - angles: Servo angles array (21,)
                - info: Dictionary with processing info
        """
        start_time = time.perf_counter()
        info = {
            'face_detected': False,
            'using_fallback': self._using_fallback,
            'using_neutral': False,
            'latency_ms': 0.0,
        }
        
        # Extract facial features
        features = self._extractor.extract(frame) if self._extractor else None
        
        if features is not None:
            info['face_detected'] = True
            self._last_face_time = time.time()
            
            # Get angles from model or fallback
            if self._using_fallback or self._model is None:
                angles = self._fallback_mapper.map_array(features)
            else:
                feature_array = features.to_array()
                angles, success = self._model.predict(feature_array)
                if not success:
                    angles = self._fallback_mapper.map_array(features)
                    info['using_fallback'] = True
            
            self._last_valid_angles = angles.copy()
            
        else:
            # Face not detected - check timeout
            time_since_face = (time.time() - self._last_face_time) * 1000
            
            if time_since_face > self.config.face_timeout_ms:
                # Transition to neutral position
                angles = np.array([
                    self.config.neutral_angles[name]
                    for name in ServoCommandProtocol.SERVO_ORDER
                ], dtype=np.float64)
                info['using_neutral'] = True
                
                # Reset model states for clean restart
                if self._model:
                    self._model.reset()
                if self._smoother:
                    self._smoother.reset()
            else:
                # Use last valid angles
                angles = self._last_valid_angles.copy()
        
        # Apply temporal smoothing
        if self._smoother:
            angles = self._smoother.smooth(angles)
        
        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        info['latency_ms'] = latency_ms
        
        # Track performance
        if self.config.log_performance:
            self._latencies.append(latency_ms)
            self._frame_count += 1
        
        return angles, info
    
    def send_angles(self, angles: np.ndarray) -> bool:
        """
        Send servo angles to hardware.
        
        Args:
            angles: Servo angles array (21,)
            
        Returns:
            True if sent successfully
        """
        if self._serial is None:
            return False
        
        # Convert to dict
        angles_dict = {
            name: int(np.clip(angles[i], 0, 180))
            for i, name in enumerate(ServoCommandProtocol.SERVO_ORDER)
        }
        
        return self._serial.send_angles(angles_dict)
    
    def step(self) -> Tuple[Optional[np.ndarray], Dict[str, any]]:
        """
        Execute one inference step: capture -> process -> send.
        
        Returns:
            Tuple of (angles, info) or (None, info) if camera unavailable
        """
        if self._camera is None:
            return None, {'error': 'Camera not available'}
        
        # Capture frame
        ret, frame = self._camera.read()
        if not ret:
            return None, {'error': 'Failed to capture frame'}
        
        # Process frame
        angles, info = self.process_frame(frame)
        
        # Send to hardware
        info['sent'] = self.send_angles(angles)
        
        return angles, info
    
    def switch_model(self, model_path: str) -> bool:
        """
        Switch to a different model at runtime.
        
        Args:
            model_path: Path to new ONNX model
            
        Returns:
            True if switch successful
            
        Requirements: 6.7
        """
        try:
            new_model = LNNS4Inference(model_path)
            self._model = new_model
            self._using_fallback = False
            self.config.model_path = model_path
            
            # Reset smoother for clean transition
            if self._smoother:
                self._smoother.reset()
            
            logger.info(f"Switched to model: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False
    
    def set_smoothing(self, alpha: float) -> None:
        """
        Update smoothing parameter at runtime.
        
        Args:
            alpha: New EMA smoothing coefficient (0, 1]
            
        Requirements: 6.4
        """
        if self._smoother:
            from .smoother import TemporalSmoother
            self._smoother = TemporalSmoother(alpha=alpha, num_servos=21)
            self.config.smoothing_alpha = alpha
            logger.info(f"Smoothing alpha set to {alpha}")
    
    def set_sensitivity(self, sensitivity: float) -> None:
        """
        Update sensitivity parameter at runtime.
        
        Args:
            sensitivity: New sensitivity multiplier
            
        Requirements: 6.4
        """
        self.config.sensitivity = sensitivity
        if self._fallback_mapper:
            self._fallback_mapper.sensitivity = sensitivity
        logger.info(f"Sensitivity set to {sensitivity}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get inference performance statistics.
        
        Returns:
            Dictionary with latency statistics
            
        Requirements: 6.6
        """
        if not self._latencies:
            return {'mean_ms': 0, 'p95_ms': 0, 'max_ms': 0, 'fps': 0}
        
        latencies = np.array(self._latencies[-1000:])  # Last 1000 frames
        return {
            'mean_ms': float(np.mean(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'max_ms': float(np.max(latencies)),
            'fps': 1000.0 / np.mean(latencies) if np.mean(latencies) > 0 else 0,
            'frame_count': self._frame_count,
        }
    
    def cleanup(self) -> None:
        """Release all resources."""
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        
        if self._extractor is not None:
            self._extractor.close()
            self._extractor = None
        
        if self._serial is not None:
            self._serial.disconnect()
            self._serial = None
        
        self._is_initialized = False
        logger.info("Inference engine cleanup complete")
    
    def __enter__(self) -> 'InferenceEngine':
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.cleanup()
    
    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized."""
        return self._is_initialized
    
    @property
    def is_using_fallback(self) -> bool:
        """Check if using fallback mode."""
        return self._using_fallback
