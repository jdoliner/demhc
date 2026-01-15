"""Checkpoint saving and loading utilities."""

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    step: int,
    loss: float,
    config: Any,
    checkpoint_dir: str,
    filename: str | None = None,
) -> str:
    """Save a training checkpoint.

    Args:
        model: The model to save
        optimizer: The optimizer state to save
        scheduler: The learning rate scheduler (optional)
        step: Current training step
        loss: Current loss value
        config: The experiment configuration
        checkpoint_dir: Directory to save checkpoints
        filename: Optional filename (defaults to checkpoint_{step}.pt)

    Returns:
        Path to the saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_{step}.pt"

    checkpoint_path = checkpoint_dir / filename

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "loss": loss,
        "config": config,
    }

    torch.save(checkpoint, checkpoint_path)

    # Also save as "latest" for easy resumption
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)

    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load weights into
        optimizer: The optimizer to load state into (optional)
        scheduler: The scheduler to load state into (optional)
        device: Device to load tensors to

    Returns:
        Dictionary with checkpoint metadata (step, loss, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "step": checkpoint["step"],
        "loss": checkpoint["loss"],
        "config": checkpoint.get("config"),
    }


def get_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Get the path to the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint, or None if no checkpoints exist
    """
    checkpoint_dir = Path(checkpoint_dir)
    latest_path = checkpoint_dir / "checkpoint_latest.pt"

    if latest_path.exists():
        return str(latest_path)

    # Fall back to finding the highest numbered checkpoint
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None

    # Sort by step number
    def get_step(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    checkpoints.sort(key=get_step)
    return str(checkpoints[-1])
