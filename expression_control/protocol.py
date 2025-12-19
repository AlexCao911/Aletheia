"""
Servo command protocol encoder and decoder.

This module implements the communication protocol for sending servo angle commands
to the MouthMaster Pico. The protocol uses a text-based format compatible with
USB serial communication.

Command format: "angles:A1,A2,...,A21"
Where A1-A21 are integer angle values in the range [0, 180].
"""

from typing import Dict, Optional, List


class ServoCommandProtocol:
    """Servo command protocol encoder/decoder for MouthMaster communication."""

    # Servo order: Mouth (11) + Eyes (6) + Brows (4) = 21 total
    SERVO_ORDER: List[str] = [
        # Mouth (11): MouthMaster Pico
        "JL", "JR", "LUL", "LUR", "LLL", "LLR",
        "CUL", "CUR", "CLL", "CLR", "TON",
        # Eyes (6): Eyes Pico
        "LR", "UD", "TL", "BL", "TR", "BR",
        # Brows (4): Brows Pico
        "LO", "LI", "RI", "RO"
    ]

    MIN_ANGLE: int = 0
    MAX_ANGLE: int = 180
    COMMAND_PREFIX: str = "angles:"

    @classmethod
    def validate_angle(cls, angle: int, servo_name: str = "") -> None:
        """
        Validate that an angle is within the valid range [0, 180].

        Args:
            angle: The angle value to validate.
            servo_name: Optional servo name for error messages.

        Raises:
            ValueError: If angle is outside the valid range.
        """
        if not isinstance(angle, int):
            raise ValueError(
                f"Angle must be an integer, got {type(angle).__name__}"
                + (f" for servo '{servo_name}'" if servo_name else "")
            )
        if angle < cls.MIN_ANGLE or angle > cls.MAX_ANGLE:
            raise ValueError(
                f"Angle {angle} is out of range [{cls.MIN_ANGLE}, {cls.MAX_ANGLE}]"
                + (f" for servo '{servo_name}'" if servo_name else "")
            )

    @classmethod
    def encode(cls, angles: Dict[str, int]) -> str:
        """
        Encode a dictionary of servo angles to a command string.

        Args:
            angles: Dictionary mapping servo names to angle values.
                    Must contain all 21 servos with angles in range [0, 180].

        Returns:
            Command string in format "angles:A1,A2,...,A21"

        Raises:
            ValueError: If angles dict is missing servos, has invalid servos,
                       or contains out-of-range angle values.
        """
        # Validate that all required servos are present
        missing_servos = set(cls.SERVO_ORDER) - set(angles.keys())
        if missing_servos:
            raise ValueError(f"Missing servo angles: {sorted(missing_servos)}")

        # Validate that no extra servos are present
        extra_servos = set(angles.keys()) - set(cls.SERVO_ORDER)
        if extra_servos:
            raise ValueError(f"Unknown servo names: {sorted(extra_servos)}")

        # Validate and collect angles in the correct order
        angle_values: List[str] = []
        for servo_name in cls.SERVO_ORDER:
            angle = angles[servo_name]
            cls.validate_angle(angle, servo_name)
            angle_values.append(str(angle))

        return cls.COMMAND_PREFIX + ",".join(angle_values)

    @classmethod
    def decode(cls, command: str) -> Optional[Dict[str, int]]:
        """
        Decode a command string to a dictionary of servo angles.

        Args:
            command: Command string in format "angles:A1,A2,...,A21"

        Returns:
            Dictionary mapping servo names to angle values,
            or None if the command format is invalid.
        """
        # Check command prefix
        if not command.startswith(cls.COMMAND_PREFIX):
            return None

        # Extract angle values
        angle_str = command[len(cls.COMMAND_PREFIX):]
        if not angle_str:
            return None

        parts = angle_str.split(",")

        # Validate correct number of angles
        if len(parts) != len(cls.SERVO_ORDER):
            return None

        # Parse and validate each angle
        angles: Dict[str, int] = {}
        for i, (servo_name, value_str) in enumerate(zip(cls.SERVO_ORDER, parts)):
            try:
                angle = int(value_str.strip())
            except ValueError:
                return None

            # Validate angle range
            if angle < cls.MIN_ANGLE or angle > cls.MAX_ANGLE:
                return None

            angles[servo_name] = angle

        return angles
