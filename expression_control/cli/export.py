#!/usr/bin/env python3
"""
ONNX export CLI for LNN-S4 expression control model.

Usage:
    python -m expression_control.cli.export \
        --checkpoint checkpoints/best_model.pt \
        --output models/expression_model.onnx

Requirements: 5.5
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export LNN-S4 model to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save ONNX model",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX verification step",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device to use for export",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point for export CLI."""
    args = parse_args()
    
    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Import here to avoid slow startup for --help
    import torch
    from expression_control.models.config import LNNS4Config
    from expression_control.models.liquid_s4 import create_model
    
    print("=" * 60)
    print("LNN-S4 ONNX Export")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"Device: {args.device}")
    print()
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    # Reconstruct config
    config_dict = checkpoint.get("config", {})
    config = LNNS4Config.from_dict(config_dict) if config_dict else LNNS4Config()
    
    print(f"Model config: hidden_dim={config.hidden_dim}, "
          f"num_layers={config.num_layers}, liquid_units={config.liquid_units}")
    
    # Create and load model
    device = torch.device(args.device)
    model = create_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Model loaded successfully")
    print()
    
    # Export to ONNX
    print("Exporting to ONNX...")
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, config.input_dim).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["features"],
        output_names=["angles", "s4_states", "ltc_state"],
        dynamic_axes={
            "features": {0: "batch", 1: "seq_len"},
            "angles": {0: "batch", 1: "seq_len"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    
    print(f"✓ Model exported to {output_path}")
    
    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    
    # Verify if requested
    if not args.no_verify:
        print()
        print("Verifying ONNX model...")
        
        try:
            import onnxruntime as ort
            import numpy as np
            
            # Get PyTorch output
            with torch.no_grad():
                torch_output, _, _ = model(dummy_input)
                torch_output = torch_output.cpu().numpy()
            
            # Get ONNX output
            session = ort.InferenceSession(str(output_path))
            onnx_input = {"features": dummy_input.cpu().numpy()}
            onnx_outputs = session.run(None, onnx_input)
            onnx_output = onnx_outputs[0]
            
            # Compare
            max_diff = np.max(np.abs(torch_output - onnx_output))
            
            if max_diff < 1e-4:
                print(f"✓ Verification passed (max diff: {max_diff:.6f})")
            else:
                print(f"⚠ Verification warning: max diff = {max_diff:.6f}")
                
        except ImportError:
            print("⚠ onnxruntime not installed, skipping verification")
    
    print()
    print("=" * 60)
    print("Export Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
