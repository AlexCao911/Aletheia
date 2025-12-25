"""
Rule-based mapper from facial features to servo angles.

This mapper uses hand-crafted rules with configurable parameters.
No training required - works out of the box.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from expression_control.features import FaceFeatures
from expression_control.mappers.base import FeatureToAngleMapper
from expression_control.smoother import TemporalSmoother


@dataclass
class ServoConfig:
    """Configuration for a single servo."""
    neutral: float = 90.0
    min_angle: float = 0.0
    max_angle: float = 180.0
    inverted: bool = False


@dataclass 
class RuleMapperConfig:
    """Configuration for RuleMapper."""
    
    # Smoothing
    smooth_alpha: float = 0.3
    use_smoothing: bool = True
    
    # Sigmoid steepness (higher = sharper transition)
    sigmoid_k: float = 5.0
    
    # Per-servo configurations
    servo_configs: Dict[str, ServoConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        # Set defaults if not provided
        """
        Ensure per-servo configurations are populated with defaults when none were provided.
        
        If `servo_configs` is empty or falsy after initialization, replaces it with the module's default servo configuration mapping produced by `_default_servo_configs()`.
        """
        if not self.servo_configs:
            self.servo_configs = self._default_servo_configs()
    
    @staticmethod
    def _default_servo_configs() -> Dict[str, ServoConfig]:
        """
        Return a dictionary of default ServoConfig instances for all supported servos.
        
        The mapping provides per-servo default neutral, min_angle, max_angle, and inversion settings for mouth, eyes, eyelids, and brows. Keys included: JL, JR, LUL, LUR, LLL, LLR, CUL, CUR, CLL, CLR, TON, LR, UD, TL, BL, TR, BR, LO, LI, RI, RO.
        
        Returns:
            Dict[str, ServoConfig]: Mapping from servo name to its default ServoConfig.
        """
        return {
            # Mouth servos
            "JL": ServoConfig(neutral=90, min_angle=60, max_angle=120),
            "JR": ServoConfig(neutral=90, min_angle=60, max_angle=120),
            "LUL": ServoConfig(neutral=90, min_angle=70, max_angle=110),
            "LUR": ServoConfig(neutral=90, min_angle=70, max_angle=110),
            "LLL": ServoConfig(neutral=90, min_angle=70, max_angle=110),
            "LLR": ServoConfig(neutral=90, min_angle=70, max_angle=110),
            "CUL": ServoConfig(neutral=90, min_angle=60, max_angle=120),
            "CUR": ServoConfig(neutral=90, min_angle=60, max_angle=120),
            "CLL": ServoConfig(neutral=90, min_angle=60, max_angle=120),
            "CLR": ServoConfig(neutral=90, min_angle=60, max_angle=120),
            "TON": ServoConfig(neutral=90, min_angle=70, max_angle=110),
            # Eye servos
            "LR": ServoConfig(neutral=90, min_angle=60, max_angle=120),
            "UD": ServoConfig(neutral=90, min_angle=70, max_angle=110),
            "TL": ServoConfig(neutral=90, min_angle=60, max_angle=120),
            "BL": ServoConfig(neutral=90, min_angle=60, max_angle=120, inverted=True),
            "TR": ServoConfig(neutral=90, min_angle=60, max_angle=120),
            "BR": ServoConfig(neutral=90, min_angle=60, max_angle=120, inverted=True),
            # Brow servos
            "LO": ServoConfig(neutral=90, min_angle=60, max_angle=120),
            "LI": ServoConfig(neutral=90, min_angle=60, max_angle=120),
            "RI": ServoConfig(neutral=90, min_angle=60, max_angle=120),
            "RO": ServoConfig(neutral=90, min_angle=60, max_angle=120),
        }


class RuleMapper(FeatureToAngleMapper):
    """
    Rule-based mapper using hand-crafted feature-to-angle mappings.
    
    Features:
    - Configurable per-servo parameters
    - Non-linear sigmoid transformation
    - Built-in temporal smoothing (optional)
    - No training required
    
    Usage:
        mapper = RuleMapper()
        angles = mapper.map(features)  # Returns np.ndarray of shape (21,)
    """
    
    def __init__(self, config: Optional[RuleMapperConfig] = None):
        """
        Create a RuleMapper configured for converting facial features to servo angles.
        
        Parameters:
            config (Optional[RuleMapperConfig]): Configuration to customize mapping behavior; when `None`, a default `RuleMapperConfig` is used. The initializer also creates an internal TemporalSmoother using the config's `smooth_alpha` and the mapper's `NUM_SERVOS`.
        """
        self.config = config or RuleMapperConfig()
        self.smoother = TemporalSmoother(
            alpha=self.config.smooth_alpha,
            num_servos=self.NUM_SERVOS,
        )
    
    def _sigmoid(self, x: float) -> float:
        """
        Apply a configurable sigmoid transform that maps an input in [0,1] to a non-linear value emphasizing deviations from 0.5.
        
        Parameters:
            x (float): Input value (expected in the range 0.0 to 1.0).
        
        Returns:
            float: Transformed value between 0 and 1 (exclusive) using the configured sigmoid steepness.
        """
        k = self.config.sigmoid_k
        return 1.0 / (1.0 + np.exp(-k * (x - 0.5)))
    
    def _map_value(
        self,
        value: float,
        servo_name: str,
        input_min: float = 0.0,
        input_max: float = 1.0,
    ) -> float:
        """
        Map a normalized input value to a servo angle using the servo's configured range and optional inversion.
        
        Parameters:
            value (float): Input value to map; expected within or near the range [input_min, input_max].
            servo_name (str): Key identifying the servo in the mapper's `servo_configs`.
            input_min (float): Minimum of the input range used for normalization (default 0.0).
            input_max (float): Maximum of the input range used for normalization (default 1.0).
        
        Returns:
            float: Servo angle clamped to the servo's [min_angle, max_angle].
        """
        config = self.config.servo_configs[servo_name]
        
        # Normalize to [0, 1]
        normalized = (value - input_min) / (input_max - input_min + 1e-6)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Non-linear transform
        normalized = self._sigmoid(normalized)
        
        # Invert if needed
        if config.inverted:
            normalized = 1.0 - normalized
        
        # Map to angle range
        angle = config.min_angle + normalized * (config.max_angle - config.min_angle)
        return float(np.clip(angle, config.min_angle, config.max_angle))
    
    def map(self, features: FaceFeatures) -> np.ndarray:
        """
        Convert FaceFeatures into an array of servo angles for every configured servo.
        
        The mapping applies per-servo rules (jaw, upper/lower lips, mouth corners, tongue, eye gaze,
        eyelids, and eyebrows), optionally applies a sigmoid transform and per-servo inversion,
        and then applies temporal smoothing if enabled in the mapper config.
        
        Parameters:
            features (FaceFeatures): Facial measurements (e.g., mouth_open_ratio, smile_intensity,
                mouth_width_ratio, eye_gaze_horizontal/vertical, left/right_eye_aspect_ratio,
                left/right_eyebrow_height, eyebrow_furrow) used to compute servo targets.
        
        Returns:
            np.ndarray: Array of angles (degrees) with shape (NUM_SERVOS,), ordered according to
                the mapper's SERVO_ORDER. Each element is clamped to the corresponding servo's
                configured [min_angle, max_angle].
        """
        angles = np.zeros(self.NUM_SERVOS, dtype=np.float64)
        
        # === Jaw (indices 0-1) ===
        jaw = self._map_value(features.mouth_open_ratio, "JL")
        angles[0] = jaw  # JL
        angles[1] = jaw  # JR
        
        # === Upper Lip (indices 2-3) ===
        upper_lip = self._map_value(features.mouth_open_ratio * 0.5, "LUL")
        angles[2] = upper_lip  # LUL
        angles[3] = upper_lip  # LUR
        
        # === Lower Lip (indices 4-5) ===
        lower_lip = self._map_value(features.mouth_open_ratio * 0.7, "LLL")
        angles[4] = lower_lip  # LLL
        angles[5] = lower_lip  # LLR
        
        # === Mouth Corners (indices 6-9) ===
        smile = features.smile_intensity
        width = features.mouth_width_ratio
        corner_factor = smile * 0.8 + width * 0.2
        
        corner_up = self._map_value(corner_factor, "CUL")
        angles[6] = corner_up  # CUL
        angles[7] = corner_up  # CUR
        
        corner_low = self._map_value(smile * 0.6, "CLL")
        angles[8] = corner_low  # CLL
        angles[9] = corner_low  # CLR
        
        # === Tongue (index 10) ===
        angles[10] = self.config.servo_configs["TON"].neutral  # TON
        
        # === Eye Gaze (indices 11-12) ===
        angles[11] = self._map_value(
            features.eye_gaze_horizontal, "LR", input_min=-1.0, input_max=1.0
        )  # LR
        angles[12] = self._map_value(
            features.eye_gaze_vertical, "UD", input_min=-1.0, input_max=1.0
        )  # UD
        
        # === Eyelids (indices 13-16) ===
        angles[13] = self._map_value(features.left_eye_aspect_ratio, "TL")   # TL
        angles[14] = self._map_value(features.left_eye_aspect_ratio, "BL")   # BL (inverted in config)
        angles[15] = self._map_value(features.right_eye_aspect_ratio, "TR")  # TR
        angles[16] = self._map_value(features.right_eye_aspect_ratio, "BR")  # BR (inverted in config)
        
        # === Eyebrows (indices 17-20) ===
        left_brow = features.left_eyebrow_height
        right_brow = features.right_eyebrow_height
        furrow = features.eyebrow_furrow
        
        angles[17] = self._map_value(left_brow, "LO")   # LO
        angles[18] = self._map_value(left_brow * (1 - furrow * 0.3), "LI")   # LI
        angles[19] = self._map_value(right_brow * (1 - furrow * 0.3), "RI")  # RI
        angles[20] = self._map_value(right_brow, "RO")  # RO
        
        # === Apply Smoothing ===
        if self.config.use_smoothing:
            angles = self.smoother.smooth(angles)
        
        return angles
    
    def reset(self) -> None:
        """
        Reset the internal temporal smoother to its initial state.
        """
        self.smoother.reset()
    
    def get_neutral_angles(self) -> np.ndarray:
        """
        Return an array of neutral angles for every servo in the mapper's defined order.
        
        Returns:
            neutral_angles (np.ndarray): 1-D array of neutral angles (dtype float64) ordered according to `SERVO_ORDER`.
        """
        return np.array([
            self.config.servo_configs[name].neutral
            for name in self.SERVO_ORDER
        ], dtype=np.float64)