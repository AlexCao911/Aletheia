#!/usr/bin/env python3
"""
Training CLI for LNN-S4 expression control model.

Usage:
    python -m expression_control.cli.train \
        --train data/train.json \
        --val data/val.json \
        --epochs 100 \
        --batch-size 32 \
        --checkpoint-dir checkpoints/

Requirements: 5.1
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LNN-S4 expression control model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Path to training dataset JSON file",
    )
    parser.add_argument(
        "--val",
        type=str,
        required=True,
        help="Path to validation dataset JSON file",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs)",
    )
    
    # Model arguments
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for S4 blocks",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=32,
        help="S4 state dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of S4 blocks",
    )
    parser.add_argument(
        "--liquid-units",
        type=int,
        default=32,
        help="Number of LTC neurons",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=16,
        help="Sequence length for training",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    
    # Output arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to train on (auto-detect if not specified)",
    )
    
    # Misc arguments
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress training progress output",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point for training CLI."""
    args = parse_args()
    
    # Validate paths
    train_path = Path(args.train)
    val_path = Path(args.val)
    
    if not train_path.exists():
        print(f"Error: Training dataset not found: {train_path}")
        return 1
    
    if not val_path.exists():
        print(f"Error: Validation dataset not found: {val_path}")
        return 1
    
    # Import here to avoid slow startup for --help
    from expression_control.models.config import LNNS4Config
    from expression_control.trainer import Trainer
    
    # Create configuration
    config = LNNS4Config(
        hidden_dim=args.hidden_dim,
        state_dim=args.state_dim,
        num_layers=args.num_layers,
        liquid_units=args.liquid_units,
        sequence_length=args.seq_length,
        dropout=args.dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
    )
    
    print("=" * 60)
    print("LNN-S4 Expression Control Training")
    print("=" * 60)
    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Device: {args.device or 'auto'}")
    print()
    print("Model Configuration:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  State dim: {config.state_dim}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Liquid units: {config.liquid_units}")
    print(f"  Sequence length: {config.sequence_length}")
    print()
    print("Training Configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Early stopping patience: {config.early_stopping_patience}")
    print("=" * 60)
    print()
    
    # Create trainer
    trainer = Trainer(
        config=config,
        train_path=str(train_path),
        val_path=str(val_path),
        device=args.device,
    )
    
    print(f"Model parameters: {trainer.get_num_parameters():,}")
    print(f"Model size: {trainer.get_model_size_mb():.2f} MB")
    print()
    
    # Resume from checkpoint if specified
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"Error: Checkpoint not found: {resume_path}")
            return 1
        print(f"Resuming from checkpoint: {resume_path}")
        trainer.load_checkpoint(str(resume_path))
    
    # Train
    try:
        best_val_loss = trainer.train(
            checkpoint_dir=args.checkpoint_dir,
            verbose=not args.quiet,
        )
        
        print()
        print("=" * 60)
        print("Training Complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best model saved to: {args.checkpoint_dir}/best_model.pt")
        print("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
