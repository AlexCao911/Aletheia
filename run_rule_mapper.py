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
    parser = argparse.ArgumentParser(description="Real-time expression control")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--serial", type=str, default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--no-serial", action="store_true", help="Run without serial")
    parser.add_argument("--smooth", type=float, default=0.3, help="Smoothing alpha (0-1)")
    parser.add_argument("--show-video", action="store_true", help="Show video window")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    return parser.parse_args()


def main():
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
        frame_count[0] += 1
        if frame_count[0] % 30 == 0:
            print(f"Jaw:{angles['JL']:3d} Smile:{angles['CUL']:3d} "
                  f"Eye:{angles['LR']:3d}/{angles['UD']:3d}")
    
    controller.run(show_video=args.show_video, on_angles=on_angles)


if __name__ == "__main__":
    main()
