"""
CLI subpackage for command-line interface tools.

Available CLI scripts:
- train: Train the LNN-S4 expression control model
- export: Export trained model to ONNX format
- evaluate: Evaluate model on test dataset

Usage:
    python -m expression_control.cli.train --help
    python -m expression_control.cli.export --help
    python -m expression_control.cli.evaluate --help
"""

__all__ = ["train", "export", "evaluate"]
