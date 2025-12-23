"""
Training infrastructure for LNN-S4 expression control model.

This module provides the Trainer class for training the LiquidS4Model,
including checkpoint saving/loading, early stopping, validation metrics,
and ONNX export functionality.

Requirements: 5.3, 5.4, 5.5, 5.6, 5.8
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from expression_control.models.config import LNNS4Config
from expression_control.models.liquid_s4 import LiquidS4Model, create_model
from expression_control.dataset import ExpressionDataset


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    
    train_loss: float
    val_loss: float
    mae: float  # Mean Absolute Error in degrees
    rmse: float  # Root Mean Square Error in degrees
    epoch: int
    learning_rate: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "mae": self.mae,
            "rmse": self.rmse,
            "epoch": self.epoch,
            "learning_rate": self.learning_rate,
        }



class Trainer:
    """
    Trainer for LNN-S4 expression control model.
    
    Handles the complete training pipeline including:
    - Data loading and batching
    - Training loop with gradient clipping
    - Validation with MAE/RMSE metrics
    - Checkpoint saving and loading
    - Early stopping based on validation loss
    - Learning rate scheduling
    - ONNX model export
    
    Usage:
        config = LNNS4Config()
        trainer = Trainer(config, "data/train.json", "data/val.json")
        trainer.train(checkpoint_dir="checkpoints/")
        trainer.export_onnx("model.onnx")
    
    Requirements: 5.3, 5.4, 5.6, 5.8
    """
    
    def __init__(
        self,
        config: LNNS4Config,
        train_path: str,
        val_path: str,
        device: Optional[str] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Model configuration.
            train_path: Path to training dataset JSON file.
            val_path: Path to validation dataset JSON file.
            device: Device to train on ("cuda", "cpu", or None for auto-detect).
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Install with: pip install torch"
            )
        
        self.config = config
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load datasets
        self.train_dataset = ExpressionDataset(
            train_path,
            sequence_length=config.sequence_length,
            augment=True,
        )
        self.val_dataset = ExpressionDataset(
            val_path,
            sequence_length=config.sequence_length,
            augment=False,
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=self.device.type == "cuda",
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        
        # Create model
        self.model = create_model(config).to(self.device)
        
        # Optimizer with weight decay (AdamW)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler (Cosine Annealing)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
        )
        
        # Loss function: Smooth L1 Loss for robust regression
        self.criterion = nn.SmoothL1Loss()
        
        # Early stopping state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        
        # Training history
        self.history: List[TrainingMetrics] = []
        self.current_epoch = 0
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for features, targets in self.train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            pred, _, _ = self.model(features)
            
            # Compute loss (targets are normalized to [0, 1], scale pred back)
            # Model outputs [0, 180], targets are [0, 1] if normalized
            pred_normalized = pred / 180.0
            loss = self.criterion(pred_normalized, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def validate(self) -> Tuple[float, float, float]:
        """
        Validate the model on the validation set.
        
        Returns:
            Tuple of (val_loss, mae, rmse) where:
            - val_loss: Average validation loss
            - mae: Mean Absolute Error in degrees
            - rmse: Root Mean Square Error in degrees
        """
        self.model.eval()
        total_loss = 0.0
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        num_batches = 0
        
        with torch.no_grad():
            for features, targets in self.val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                pred, _, _ = self.model(features)
                
                # Compute loss
                pred_normalized = pred / 180.0
                loss = self.criterion(pred_normalized, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions and targets for metrics (in degrees)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(targets.cpu().numpy() * 180.0)  # Scale back to degrees
        
        # Compute metrics
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        mae = compute_mae(preds, targets)
        rmse = compute_rmse(preds, targets)
        
        return total_loss / max(num_batches, 1), mae, rmse

    
    def train(
        self,
        checkpoint_dir: str = "checkpoints",
        verbose: bool = True,
    ) -> float:
        """
        Train the model for the configured number of epochs.
        
        Implements early stopping based on validation loss.
        
        Args:
            checkpoint_dir: Directory to save checkpoints.
            verbose: Whether to print training progress.
            
        Returns:
            Best validation loss achieved during training.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, mae, rmse = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Record metrics
            metrics = TrainingMetrics(
                train_loss=train_loss,
                val_loss=val_loss,
                mae=mae,
                rmse=rmse,
                epoch=epoch + 1,
                learning_rate=current_lr,
            )
            self.history.append(metrics)
            
            epoch_time = time.time() - start_time
            
            if verbose:
                print(f"Epoch {epoch + 1}/{self.config.num_epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, MAE: {mae:.2f}°, RMSE: {rmse:.2f}°")
                print(f"  LR: {current_lr:.6f}")
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                self.save_checkpoint(best_path)
                
                if verbose:
                    print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                if verbose:
                    print(f"  No improvement ({self.patience_counter}/{self.config.early_stopping_patience})")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                periodic_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                self.save_checkpoint(periodic_path)
            
            # Early stopping check
            if self.patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        return self.best_val_loss
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save a training checkpoint.
        
        Checkpoint includes:
        - Model state dict
        - Optimizer state dict
        - Scheduler state dict
        - Configuration
        - Training state (epoch, best_val_loss, etc.)
        
        Args:
            path: File path to save checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
            "best_val_loss": self.best_val_loss,
            "current_epoch": self.current_epoch,
            "patience_counter": self.patience_counter,
            "history": [m.to_dict() for m in self.history],
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load a training checkpoint.
        
        Restores model, optimizer, scheduler, and training state.
        
        Args:
            path: File path to load checkpoint from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.patience_counter = checkpoint.get("patience_counter", 0)
        
        # Restore history
        history_dicts = checkpoint.get("history", [])
        self.history = [
            TrainingMetrics(**h) for h in history_dicts
        ]
    
    def export_onnx(self, output_path: str, verify: bool = True) -> None:
        """
        Export the model to ONNX format for edge deployment.
        
        Args:
            output_path: Path to save the ONNX model.
            verify: Whether to verify the exported model produces same outputs.
            
        Requirements: 5.5
        """
        self.model.eval()
        
        # Create dummy input (batch=1, seq_len=1, input_dim=14)
        dummy_input = torch.randn(1, 1, self.config.input_dim).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=["features"],
            output_names=["angles", "s4_states", "ltc_state"],
            dynamic_axes={
                "features": {0: "batch", 1: "seq_len"},
                "angles": {0: "batch", 1: "seq_len"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
        
        print(f"Model exported to {output_path}")
        
        # Verify exported model
        if verify:
            self._verify_onnx_export(output_path, dummy_input)
    
    def _verify_onnx_export(self, onnx_path: str, dummy_input: torch.Tensor) -> None:
        """
        Verify that ONNX model produces same outputs as PyTorch model.
        
        Args:
            onnx_path: Path to the ONNX model.
            dummy_input: Input tensor used for export.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            print("Warning: onnxruntime not installed, skipping verification")
            return
        
        # Get PyTorch output
        with torch.no_grad():
            torch_output, _, _ = self.model(dummy_input)
            torch_output = torch_output.cpu().numpy()
        
        # Get ONNX output
        session = ort.InferenceSession(onnx_path)
        onnx_input = {"features": dummy_input.cpu().numpy()}
        onnx_outputs = session.run(None, onnx_input)
        onnx_output = onnx_outputs[0]
        
        # Compare outputs
        max_diff = np.max(np.abs(torch_output - onnx_output))
        
        if max_diff < 1e-4:
            print(f"✓ ONNX verification passed (max diff: {max_diff:.6f})")
        else:
            print(f"⚠ ONNX verification warning: max diff = {max_diff:.6f}")
    
    def get_model_size_mb(self) -> float:
        """Get model size in megabytes."""
        return self.model.get_model_size_bytes() / (1024 * 1024)
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return self.model.get_num_parameters()


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        
    Returns:
        Mean Absolute Error.
        
    Requirements: 5.6
    """
    return float(np.mean(np.abs(predictions - targets)))


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Root Mean Square Error.
    
    Args:
        predictions: Predicted values.
        targets: Ground truth values.
        
    Returns:
        Root Mean Square Error.
        
    Requirements: 5.6
    """
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))
