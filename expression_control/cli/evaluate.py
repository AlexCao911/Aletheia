#!/usr/bin/env python3
"""
Evaluation CLI for LNN-S4 expression control model.

Usage:
    python -m expression_control.cli.evaluate \
        --model models/expression_model.onnx \
        --test data/test.json

    # Or with PyTorch checkpoint:
    python -m expression_control.cli.evaluate \
        --checkpoint checkpoints/best_model.pt \
        --test data/test.json

Requirements: 5.1, 5.6
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate LNN-S4 expression control model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model arguments (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        type=str,
        help="Path to ONNX model file",
    )
    model_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to PyTorch checkpoint file",
    )
    
    # Data arguments
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to test dataset JSON file",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=16,
        help="Sequence length for evaluation",
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save detailed results (JSON)",
    )
    parser.add_argument(
        "--per-servo",
        action="store_true",
        help="Show per-servo metrics",
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device to use for evaluation",
    )
    
    return parser.parse_args()


def evaluate_onnx(
    model_path: str,
    test_path: str,
    seq_length: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluate ONNX model on test dataset.
    
    Returns:
        Tuple of (predictions, targets, inference_time_ms)
    """
    import onnxruntime as ort
    from expression_control.dataset import ExpressionDataset
    from expression_control.protocol import ServoCommandProtocol
    
    # Load dataset
    dataset = ExpressionDataset(
        test_path,
        sequence_length=seq_length,
        augment=False,
        normalize_angles=False,  # Keep angles in degrees
    )
    
    # Create ONNX session
    session = ort.InferenceSession(model_path)
    
    all_preds = []
    all_targets = []
    total_time = 0.0
    
    # Process in batches
    for i in range(0, len(dataset), batch_size):
        batch_features = []
        batch_targets = []
        
        for j in range(i, min(i + batch_size, len(dataset))):
            features, targets = dataset[j]
            batch_features.append(features.numpy())
            batch_targets.append(targets.numpy())
        
        features_batch = np.stack(batch_features)
        targets_batch = np.stack(batch_targets)
        
        # Run inference
        start_time = time.perf_counter()
        outputs = session.run(None, {"features": features_batch})
        total_time += time.perf_counter() - start_time
        
        preds = outputs[0]
        all_preds.append(preds)
        all_targets.append(targets_batch)
    
    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    # Average inference time per sample
    avg_time_ms = (total_time / len(dataset)) * 1000
    
    return predictions, targets, avg_time_ms


def evaluate_pytorch(
    checkpoint_path: str,
    test_path: str,
    seq_length: int,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluate PyTorch model on test dataset.
    
    Returns:
        Tuple of (predictions, targets, inference_time_ms)
    """
    import torch
    from torch.utils.data import DataLoader
    from expression_control.models.config import LNNS4Config
    from expression_control.models.liquid_s4 import create_model
    from expression_control.dataset import ExpressionDataset
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint.get("config", {})
    config = LNNS4Config.from_dict(config_dict) if config_dict else LNNS4Config()
    config.sequence_length = seq_length
    
    # Create model
    model = create_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load dataset
    dataset = ExpressionDataset(
        test_path,
        sequence_length=seq_length,
        augment=False,
        normalize_angles=False,
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_targets = []
    total_time = 0.0
    
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            
            start_time = time.perf_counter()
            preds, _, _ = model(features)
            total_time += time.perf_counter() - start_time
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
    
    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    avg_time_ms = (total_time / len(dataset)) * 1000
    
    return predictions, targets, avg_time_ms


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    per_servo: bool = False,
) -> dict:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Shape (N, seq_len, 21) or (N, 21)
        targets: Same shape as predictions
        per_servo: Whether to compute per-servo metrics
        
    Returns:
        Dictionary of metrics
    """
    from expression_control.protocol import ServoCommandProtocol
    
    # Flatten to (N*seq_len, 21) if needed
    if predictions.ndim == 3:
        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])
    
    # Overall metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    max_error = np.max(np.abs(predictions - targets))
    
    # Percentage within tolerance
    within_2deg = np.mean(np.abs(predictions - targets) <= 2.0) * 100
    within_5deg = np.mean(np.abs(predictions - targets) <= 5.0) * 100
    
    metrics = {
        "overall": {
            "mae": float(mae),
            "rmse": float(rmse),
            "max_error": float(max_error),
            "within_2deg_pct": float(within_2deg),
            "within_5deg_pct": float(within_5deg),
        }
    }
    
    # Per-servo metrics
    if per_servo:
        servo_metrics = {}
        for i, name in enumerate(ServoCommandProtocol.SERVO_ORDER):
            servo_preds = predictions[:, i]
            servo_targets = targets[:, i]
            
            servo_metrics[name] = {
                "mae": float(np.mean(np.abs(servo_preds - servo_targets))),
                "rmse": float(np.sqrt(np.mean((servo_preds - servo_targets) ** 2))),
                "max_error": float(np.max(np.abs(servo_preds - servo_targets))),
            }
        
        metrics["per_servo"] = servo_metrics
    
    return metrics


def main() -> int:
    """Main entry point for evaluation CLI."""
    args = parse_args()
    
    # Validate paths
    test_path = Path(args.test)
    if not test_path.exists():
        print(f"Error: Test dataset not found: {test_path}")
        return 1
    
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Error: ONNX model not found: {model_path}")
            return 1
        model_type = "ONNX"
    else:
        model_path = Path(args.checkpoint)
        if not model_path.exists():
            print(f"Error: Checkpoint not found: {model_path}")
            return 1
        model_type = "PyTorch"
    
    print("=" * 60)
    print("LNN-S4 Model Evaluation")
    print("=" * 60)
    print(f"Model: {model_path} ({model_type})")
    print(f"Test data: {test_path}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Run evaluation
    print("Running evaluation...")
    
    if args.model:
        predictions, targets, avg_time_ms = evaluate_onnx(
            str(model_path),
            str(test_path),
            args.seq_length,
            args.batch_size,
        )
    else:
        predictions, targets, avg_time_ms = evaluate_pytorch(
            str(model_path),
            str(test_path),
            args.seq_length,
            args.batch_size,
            args.device,
        )
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets, per_servo=args.per_servo)
    metrics["inference_time_ms"] = avg_time_ms
    
    # Print results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print()
    print("Overall Metrics:")
    print(f"  MAE:  {metrics['overall']['mae']:.2f}°")
    print(f"  RMSE: {metrics['overall']['rmse']:.2f}°")
    print(f"  Max Error: {metrics['overall']['max_error']:.2f}°")
    print(f"  Within 2°: {metrics['overall']['within_2deg_pct']:.1f}%")
    print(f"  Within 5°: {metrics['overall']['within_5deg_pct']:.1f}%")
    print()
    print(f"Inference Time: {avg_time_ms:.2f} ms/sample")
    print(f"Throughput: {1000/avg_time_ms:.1f} samples/sec")
    
    if args.per_servo and "per_servo" in metrics:
        print()
        print("Per-Servo Metrics:")
        print("-" * 40)
        print(f"{'Servo':<8} {'MAE':>8} {'RMSE':>8} {'Max':>8}")
        print("-" * 40)
        for name, m in metrics["per_servo"].items():
            print(f"{name:<8} {m['mae']:>7.2f}° {m['rmse']:>7.2f}° {m['max_error']:>7.2f}°")
    
    # Save results if requested
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print()
        print(f"Results saved to: {output_path}")
    
    print()
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
