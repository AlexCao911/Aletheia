#!/usr/bin/env python3
"""
Real-time expression control using rule-based mapping.

Usage:
    python run_rule_mapper.py --camera 0 --serial /dev/ttyACM0
    python run_rule_mapper.py --camera 0 --no-serial --show-video
"""

import argparse

from expression_control.controller import ExpressionController, ControllerConfig
from expression_control.mappers.rule_mapper import RuleMapper, RuleMapperConfig


def parse_args():
    """
    Parse command-line arguments for running the real-time expression control pipeline.
    
    The recognized options configure camera device, serial connection, smoothing, video display, and target frame rate.
    
    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            - camera (int): camera device ID
            - serial (str): serial port path
            - no_serial (bool): True when running without a serial connection
            - smooth (float): smoothing alpha in [0, 1]
            - show_video (bool): True to display the video window
            - fps (int): target frames per second
    """
    parser = argparse.ArgumentParser(description="Real-time expression control")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--serial", type=str, default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--no-serial", action="store_true", help="Run without serial")
    parser.add_argument("--smooth", type=float, default=0.3, help="Smoothing alpha (0-1)")
    parser.add_argument("--show-video", action="store_true", help="Show video window")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    return parser.parse_args()


def main():
    """
    Run the rule-based expression mapper and start the real-time controller using command-line options.
    
    Configures a RuleMapper with the smoothing alpha from CLI, constructs an ExpressionController with camera, serial (or no serial), and target FPS from CLI, then starts the controller loop. Optionally displays video and invokes a callback that logs selected facial angles every 30 frames.
    """
    args = parse_args()
    
    # Configure mapper
    mapper_config = RuleMapperConfig(smooth_alpha=args.smooth)
    mapper = RuleMapper(config=mapper_config)
    
    # Configure controller
    controller_config = ControllerConfig(
        camera_id=args.camera,
        serial_port=None if args.no_serial else args.serial,
        target_fps=args.fps,
    )
    
    # Create and run controller
    controller = ExpressionController(
        mapper=mapper,
        config=controller_config,
    )
    
    # Optional: log angles periodically
    frame_count = [0]
    def on_angles(angles):
        """
        Log selected facial angles every 30 frames.
        
        Parameters:
            angles (dict): Mapping of angle names to integer values. Required keys:
                - 'JL': jaw angle
                - 'CUL': smile (corner up-left) angle
                - 'LR': eye left/right angle
                - 'UD': eye up/down angle
        
        Description:
            Increments an external frame counter and, every 30 frames, prints a single-line summary
            showing the jaw, smile, and eye angles formatted as: "Jaw:{JL} Smile:{CUL} Eye:{LR}/{UD}".
        """
        frame_count[0] += 1
        if frame_count[0] % 30 == 0:
            print(f"Jaw:{angles['JL']:3d} Smile:{angles['CUL']:3d} "
                  f"Eye:{angles['LR']:3d}/{angles['UD']:3d}")
    
    controller.run(show_video=args.show_video, on_angles=on_angles)


if __name__ == "__main__":
    main()