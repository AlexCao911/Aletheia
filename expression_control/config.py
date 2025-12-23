"""
Configuration management for expression control inference.

This module provides configuration file loading and saving for the
inference engine, supporting YAML and JSON formats.

Requirements: 6.4, 6.7
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .inference import InferenceConfig
from .protocol import ServoCommandProtocol

logger = logging.getLogger(__name__)

# Default config file locations
DEFAULT_CONFIG_PATHS = [
    Path("expression_config.yaml"),
    Path("expression_config.json"),
    Path.home() / ".config" / "expression_control" / "config.yaml",
    Path.home() / ".config" / "expression_control" / "config.json",
]


def load_config(
    config_path: Optional[Union[str, Path]] = None
) -> InferenceConfig:
    """
    Load inference configuration from file.
    
    Supports YAML and JSON formats. If no path is specified, searches
    default locations.
    
    Args:
        config_path: Path to config file, or None to search defaults
        
    Returns:
        InferenceConfig instance
        
    Raises:
        FileNotFoundError: If no config file found
        ValueError: If config file is invalid
    """
    # Find config file
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
    else:
        path = None
        for default_path in DEFAULT_CONFIG_PATHS:
            if default_path.exists():
                path = default_path
                break
        
        if path is None:
            logger.info("No config file found, using defaults")
            return InferenceConfig()
    
    # Load config
    logger.info(f"Loading config from {path}")
    
    try:
        with open(path, 'r') as f:
            if path.suffix in ('.yaml', '.yml'):
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError(
                        "PyYAML not installed. Install with: pip install pyyaml"
                    )
            else:
                data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse config file: {e}")
    
    return _dict_to_config(data)


def save_config(
    config: InferenceConfig,
    config_path: Union[str, Path],
    format: str = "auto"
) -> None:
    """
    Save inference configuration to file.
    
    Args:
        config: Configuration to save
        config_path: Output file path
        format: "yaml", "json", or "auto" (detect from extension)
    """
    path = Path(config_path)
    
    # Determine format
    if format == "auto":
        if path.suffix in ('.yaml', '.yml'):
            format = "yaml"
        else:
            format = "json"
    
    # Convert to dict
    data = _config_to_dict(config)
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write file
    with open(path, 'w') as f:
        if format == "yaml":
            try:
                import yaml
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
            except ImportError:
                raise ImportError(
                    "PyYAML not installed. Install with: pip install pyyaml"
                )
        else:
            json.dump(data, f, indent=2)
    
    logger.info(f"Saved config to {path}")


def _dict_to_config(data: Dict[str, Any]) -> InferenceConfig:
    """Convert dictionary to InferenceConfig."""
    # Handle nested neutral_angles
    neutral_angles = data.get('neutral_angles', None)
    if neutral_angles is None:
        neutral_angles = {name: 90 for name in ServoCommandProtocol.SERVO_ORDER}
    
    return InferenceConfig(
        model_path=data.get('model_path'),
        camera_id=data.get('camera_id', 0),
        serial_port=data.get('serial_port', '/dev/ttyACM0'),
        smoothing_alpha=data.get('smoothing_alpha', 0.3),
        face_timeout_ms=data.get('face_timeout_ms', 500.0),
        fallback_enabled=data.get('fallback_enabled', True),
        sensitivity=data.get('sensitivity', 1.0),
        log_performance=data.get('log_performance', True),
        target_fps=data.get('target_fps', 30.0),
        neutral_angles=neutral_angles,
    )


def _config_to_dict(config: InferenceConfig) -> Dict[str, Any]:
    """Convert InferenceConfig to dictionary."""
    return {
        'model_path': config.model_path,
        'camera_id': config.camera_id,
        'serial_port': config.serial_port,
        'smoothing_alpha': config.smoothing_alpha,
        'face_timeout_ms': config.face_timeout_ms,
        'fallback_enabled': config.fallback_enabled,
        'sensitivity': config.sensitivity,
        'log_performance': config.log_performance,
        'target_fps': config.target_fps,
        'neutral_angles': config.neutral_angles,
    }


def create_default_config(output_path: Union[str, Path]) -> None:
    """
    Create a default configuration file with comments.
    
    Args:
        output_path: Path to write the config file
    """
    path = Path(output_path)
    
    if path.suffix in ('.yaml', '.yml'):
        content = """# Expression Control Configuration
# ================================

# Path to ONNX model file (null for fallback mode)
model_path: null

# Camera device ID (usually 0 for built-in camera)
camera_id: 0

# Serial port for Pico communication
serial_port: /dev/ttyACM0

# EMA smoothing coefficient (0, 1]
# Lower values = smoother but slower response
# Higher values = more responsive but potentially jittery
smoothing_alpha: 0.3

# Face detection timeout in milliseconds
# After this time without face detection, servos move to neutral
face_timeout_ms: 500.0

# Enable fallback mode (direct MediaPipe mapping) when model unavailable
fallback_enabled: true

# Global sensitivity multiplier for feature-to-servo mapping
sensitivity: 1.0

# Log inference performance statistics
log_performance: true

# Target frames per second for inference loop
target_fps: 30.0

# Neutral servo positions (used when face not detected)
neutral_angles:
  JL: 90
  JR: 90
  LUL: 90
  LUR: 90
  LLL: 90
  LLR: 90
  CUL: 90
  CUR: 90
  CLL: 90
  CLR: 90
  TON: 90
  LR: 90
  UD: 90
  TL: 90
  BL: 90
  TR: 90
  BR: 90
  LO: 90
  LI: 90
  RI: 90
  RO: 90
"""
    else:
        config = InferenceConfig()
        content = json.dumps(_config_to_dict(config), indent=2)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    
    logger.info(f"Created default config at {path}")
