#!/usr/bin/env python3
"""
Real-time inference CLI for LNN-S4 expression control.

Usage:
    python -m expression_control.cli.run \
        --model models/expression_model.onnx \
        --camera 0 \
        --port /dev/ttyACM0

Requirements: 6.5, 6.6
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run real-time expression control inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to ONNX model file (uses fallback if not specified)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (YAML or JSON)",
    )
    
    # Hardware arguments
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for Pico communication",
    )
    parser.add_argument(
        "--no-serial",
        action="store_true",
        help="Run without serial communication (for testing)",
    )
    
    # Processing arguments
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.3,
        help="EMA smoothing alpha (0, 1]",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help="Feature sensitivity multiplier",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=500.0,
        help="Face detection timeout in ms before neutral position",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target frames per second",
    )
    
    # Output arguments
    parser.add_argument(
        "--show-video",
        action="store_true",
        help="Display video feed with overlay",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        help="Log performance statistics periodically",
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=5.0,
        help="Interval in seconds for logging stats",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    # Misc arguments
    parser.add_argument(
        "--create-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Create a default config file and exit",
    )
    
    return parser.parse_args()


class InferenceRunner:
    """Main inference loop runner."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.engine = None
        self.running = False
        self._last_stats_time = 0.0
    
    def setup(self) -> bool:
        """Initialize the inference engine."""
        from expression_control.inference import InferenceConfig, InferenceEngine
        from expression_control.config import load_config
        
        # Load config from file or create from args
        if self.args.config:
            config = load_config(self.args.config)
            # Override with command line args
            if self.args.model:
                config.model_path = self.args.model
            config.camera_id = self.args.camera
            if not self.args.no_serial:
                config.serial_port = self.args.port
        else:
            config = InferenceConfig(
                model_path=self.args.model,
                camera_id=self.args.camera,
                serial_port=self.args.port if not self.args.no_serial else None,
                smoothing_alpha=self.args.smoothing,
                face_timeout_ms=self.args.timeout,
                sensitivity=self.args.sensitivity,
                log_performance=self.args.log_stats,
                target_fps=self.args.fps,
            )
        
        # Create engine
        self.engine = InferenceEngine(config)
        
        # Initialize
        if not self.engine.initialize():
            logger.error("Failed to initialize inference engine")
            return False
        
        return True
    
    def run(self) -> int:
        """Run the main inference loop."""
        if self.engine is None:
            return 1
        
        self.running = True
        frame_interval = 1.0 / self.args.fps
        
        # Setup video display if requested
        cv2 = None
        if self.args.show_video:
            try:
                import cv2 as cv2_module
                cv2 = cv2_module
            except ImportError:
                logger.warning("OpenCV not available, video display disabled")
        
        logger.info("Starting inference loop (Ctrl+C to stop)")
        logger.info(f"Mode: {'Model' if not self.engine.is_using_fallback else 'Fallback'}")
        
        try:
            while self.running:
                loop_start = time.perf_counter()
                
                # Execute one inference step
                angles, info = self.engine.step()
                
                if angles is None:
                    if 'error' in info:
                        logger.warning(f"Step error: {info['error']}")
                    continue
                
                # Log frame info if not quiet
                if not self.args.quiet and self.args.debug:
                    status = []
                    if info.get('face_detected'):
                        status.append("face")
                    if info.get('using_fallback'):
                        status.append("fallback")
                    if info.get('using_neutral'):
                        status.append("neutral")
                    logger.debug(
                        f"Frame: {info.get('latency_ms', 0):.1f}ms, "
                        f"status=[{', '.join(status)}]"
                    )
                
                # Log periodic stats
                if self.args.log_stats:
                    now = time.time()
                    if now - self._last_stats_time >= self.args.stats_interval:
                        self._log_stats()
                        self._last_stats_time = now
                
                # Display video if requested
                if cv2 is not None and self.args.show_video:
                    # Get frame from camera
                    if self.engine._camera is not None:
                        ret, frame = self.engine._camera.read()
                        if ret:
                            # Add overlay
                            frame = self._add_overlay(cv2, frame, angles, info)
                            cv2.imshow('Expression Control', frame)
                            
                            # Check for quit key
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                logger.info("Quit requested via keyboard")
                                break
                
                # Rate limiting
                elapsed = time.perf_counter() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.running = False
            if cv2 is not None:
                cv2.destroyAllWindows()
        
        return 0
    
    def _add_overlay(self, cv2, frame, angles, info):
        """Add status overlay to video frame."""
        h, w = frame.shape[:2]
        
        # Status text
        status_lines = [
            f"FPS: {1000/info.get('latency_ms', 33.3):.1f}",
            f"Latency: {info.get('latency_ms', 0):.1f}ms",
            f"Face: {'Yes' if info.get('face_detected') else 'No'}",
            f"Mode: {'Fallback' if info.get('using_fallback') else 'Model'}",
        ]
        
        if info.get('using_neutral'):
            status_lines.append("NEUTRAL")
        
        # Draw status
        y = 30
        for line in status_lines:
            cv2.putText(
                frame, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            y += 25
        
        return frame
    
    def _log_stats(self):
        """Log performance statistics."""
        if self.engine is None:
            return
        
        stats = self.engine.get_performance_stats()
        logger.info(
            f"Performance: mean={stats['mean_ms']:.1f}ms, "
            f"p95={stats['p95_ms']:.1f}ms, "
            f"fps={stats['fps']:.1f}, "
            f"frames={stats['frame_count']}"
        )
    
    def stop(self):
        """Stop the inference loop."""
        self.running = False
    
    def cleanup(self):
        """Clean up resources."""
        if self.engine is not None:
            self.engine.cleanup()
            self.engine = None


def main() -> int:
    """Main entry point for inference CLI."""
    args = parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Handle create-config option
    if args.create_config:
        from expression_control.config import create_default_config
        create_default_config(args.create_config)
        print(f"Created default config at: {args.create_config}")
        return 0
    
    # Print banner
    if not args.quiet:
        print("=" * 60)
        print("LNN-S4 Expression Control - Real-time Inference")
        print("=" * 60)
        print(f"Model: {args.model or 'Fallback mode'}")
        print(f"Camera: {args.camera}")
        print(f"Serial: {args.port if not args.no_serial else 'Disabled'}")
        print(f"Target FPS: {args.fps}")
        print("=" * 60)
        print()
    
    # Create runner
    runner = InferenceRunner(args)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        runner.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup and run
    try:
        if not runner.setup():
            return 1
        
        return runner.run()
    
    finally:
        runner.cleanup()
        if not args.quiet:
            print()
            print("Inference stopped")


if __name__ == "__main__":
    sys.exit(main())
