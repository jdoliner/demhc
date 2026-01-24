"""Main training script for DEMHC experiments."""

import argparse
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from demhc.config import (
    DataConfig,
    DEQConfig,
    ExperimentConfig,
    mHCConfig,
    ModelConfig,
    TrainConfig,
)
from demhc.data.datasets import create_dataloaders
from demhc.models.baseline import BaselineTransformer
from demhc.models.demhc import DEQmHCModel
from demhc.utils.checkpointing import get_latest_checkpoint, load_checkpoint, save_checkpoint
from demhc.utils.logging import MetricsLogger, setup_tensorboard


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)


def compute_annealing_values(
    step: int,
    config: ExperimentConfig,
) -> tuple[float | None, float | None]:
    """Compute annealed tolerance multiplier and alpha override for current step.

    Args:
        step: Current training step
        config: Experiment configuration

    Returns:
        Tuple of (tolerance_multiplier, alpha_override)
        - tolerance_multiplier: Multiplier for base tolerances (1.0 = no change), or None if no annealing
        - alpha_override: Override alpha value, or None to use learned contraction factors
    """
    deq_config = config.model.deq

    # Tolerance annealing: multiplier decreases from tol_anneal_start to 1.0
    tol_multiplier = None
    if deq_config.tol_anneal_start is not None:
        anneal_steps = deq_config.tol_anneal_steps or config.train.warmup_steps
        if step < anneal_steps:
            # Linear interpolation from tol_anneal_start to 1.0
            progress = step / anneal_steps
            tol_multiplier = deq_config.tol_anneal_start * (1 - progress) + 1.0 * progress
        else:
            tol_multiplier = 1.0

    # Alpha annealing: alpha increases from alpha_start to alpha_end
    alpha_override = None
    if deq_config.alpha_start is not None and deq_config.alpha_end is not None:
        anneal_steps = deq_config.alpha_anneal_steps or config.train.warmup_steps
        if step < anneal_steps:
            # Linear interpolation from alpha_start to alpha_end
            progress = step / anneal_steps
            alpha_override = (
                deq_config.alpha_start * (1 - progress) + deq_config.alpha_end * progress
            )
        else:
            alpha_override = deq_config.alpha_end

    return tol_multiplier, alpha_override


def apply_annealing(
    model: nn.Module,
    step: int,
    config: ExperimentConfig,
) -> tuple[float | None, float | None]:
    """Apply annealing schedules to the model.

    Args:
        model: The model (may be compiled)
        step: Current training step
        config: Experiment configuration

    Returns:
        Tuple of (tolerance_multiplier, alpha_override) that were applied
    """
    # Get the actual model (handle torch.compile wrapper)
    actual_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    if not isinstance(actual_model, DEQmHCModel):
        return None, None

    tol_multiplier, alpha_override = compute_annealing_values(step, config)

    if tol_multiplier is not None:
        actual_model.set_tolerance_multiplier(tol_multiplier)

    # Apply alpha override (None means use learned contraction factors)
    actual_model.set_alpha_override(alpha_override)

    return tol_multiplier, alpha_override
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_str]


def create_model(config: ExperimentConfig, device: torch.device) -> nn.Module:
    """Create the model based on configuration."""
    if config.model_type == "baseline":
        model = BaselineTransformer(config.model)
    else:
        model = DEQmHCModel(config.model)

    model = model.to(device)

    # Print model info
    n_params = model.get_num_params(non_embedding=True)
    print(f"Model type: {config.model_type}")
    print(f"Number of parameters (non-embedding): {n_params:,}")
    print(f"Number of parameters (total): {model.get_num_params(non_embedding=False):,}")

    return model


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: TrainConfig,
    mhc_lr_mult: float = 1.0,
) -> tuple[AdamW, torch.optim.lr_scheduler.LRScheduler]:
    """Create optimizer and learning rate scheduler."""
    # Separate parameter groups:
    # 1. mHC parameters (mixing matrices, lane projections) - may need higher LR
    # 2. Regular params with weight decay
    # 3. Regular params without weight decay (biases, LayerNorm)

    mhc_params = []
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # mHC-specific parameters
        if "mix_logits" in name or "agg_logits" in name or "lane_projection" in name:
            mhc_params.append(param)
        # Don't apply weight decay to biases and LayerNorm
        elif "bias" in name or "ln" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Add mHC params with potentially higher learning rate
    if mhc_params:
        optimizer_groups.append(
            {
                "params": mhc_params,
                "weight_decay": 0.0,  # No weight decay for mHC params
                "lr": config.learning_rate * mhc_lr_mult,
            }
        )

    optimizer = AdamW(
        optimizer_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )

    # Learning rate schedule: warmup then decay
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=config.warmup_steps,
    )

    if config.lr_decay == "cosine":
        decay_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.max_steps - config.warmup_steps,
            eta_min=config.learning_rate * 0.1,
        )
    elif config.lr_decay == "linear":
        decay_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=config.max_steps - config.warmup_steps,
        )
    else:  # constant
        decay_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=config.max_steps - config.warmup_steps,
        )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[config.warmup_steps],
    )

    return optimizer, scheduler


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int = 50,
) -> float:
    """Evaluate the model on a dataset.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run on
        dtype: Data type for mixed precision
        max_batches: Maximum number of batches to evaluate

    Returns:
        Average loss over the evaluation set
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        if num_batches >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=dtype):
            if isinstance(model, DEQmHCModel):
                logits = model(input_ids)
            else:
                logits = model(input_ids)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


def train(config: ExperimentConfig, resume: str | None = None) -> None:
    """Main training loop.

    Args:
        config: Experiment configuration
        resume: Path to checkpoint to resume from (optional)
    """
    # Setup
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_dtype(config.train.dtype)

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Create output directories
    output_dir = Path(config.output_dir) / config.name
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    writer = setup_tensorboard(str(log_dir), config.name)
    logger = MetricsLogger(writer)

    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config.data,
        batch_size=config.train.batch_size,
        seed=config.seed,
    )
    train_iter = iter(train_loader)

    # Create model
    print("Creating model...")
    model = create_model(config, device)

    # Optionally compile the model
    if config.train.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Create optimizer and scheduler
    mhc_lr_mult = config.train.mhc_lr_mult if config.model_type == "demhc" else 1.0
    optimizer, scheduler = create_optimizer_and_scheduler(model, config.train, mhc_lr_mult)

    # Resume from checkpoint if specified
    start_step = 0
    if resume:
        print(f"Resuming from checkpoint: {resume}")
        checkpoint_info = load_checkpoint(resume, model, optimizer, scheduler, device)
        start_step = checkpoint_info["step"]
        print(f"Resumed from step {start_step}")

    # Training loop
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(range(start_step, config.train.max_steps), desc="Training")

    for step in progress_bar:
        # Get next batch (with automatic reset)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Apply annealing schedules (tolerance and alpha)
        tol_multiplier, alpha_override = apply_annealing(model, step, config)

        # Forward pass with mixed precision
        with torch.autocast(device_type="cuda", dtype=dtype):
            if isinstance(model, DEQmHCModel) or (
                hasattr(model, "_orig_mod") and isinstance(model._orig_mod, DEQmHCModel)
            ):
                # For DEQ model, get stats for logging
                logits, stats = (
                    model._orig_mod(input_ids, return_stats=True)
                    if hasattr(model, "_orig_mod")
                    else model(input_ids, return_stats=True)
                )
            else:
                logits = model(input_ids)
                stats = None

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip)

        # Optimizer step
        optimizer.step()
        scheduler.step()

        # Update running loss
        running_loss += loss.item()
        logger.step = step

        # Logging
        if step % config.train.log_interval == 0:
            avg_loss = running_loss / config.train.log_interval if step > 0 else loss.item()
            running_loss = 0.0

            lr = scheduler.get_last_lr()[0]
            logger.log_training_step(
                loss=avg_loss,
                learning_rate=lr,
                grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                step=step,
            )

            # Log DEQ stats if available
            if stats is not None:
                actual_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                layer_iters = [s.forward_iters for s in stats.layer_stats]
                layer_residuals = [s.forward_residual for s in stats.layer_stats]
                layer_tolerances = [layer.layer_tol for layer in actual_model.layers]
                logger.log_deq_stats(
                    layer_iters,
                    layer_residuals,
                    step=step,
                    layer_tolerances=layer_tolerances,
                    tol_multiplier=tol_multiplier,
                    alpha_override=alpha_override,
                )

            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{lr:.2e}",
                ppl=f"{2**avg_loss:.2f}",
            )

        # Evaluation
        if step > 0 and step % config.train.eval_interval == 0:
            print(f"\nEvaluating at step {step}...")
            val_loss = evaluate(
                model, val_loader, device, dtype, max_batches=config.train.num_eval_batches
            )
            logger.log_validation(val_loss, step=step)
            print(f"Validation loss: {val_loss:.4f}, perplexity: {2**val_loss:.2f}")

            # Log mHC stats for DEQ model
            if isinstance(model, DEQmHCModel) or (
                hasattr(model, "_orig_mod") and isinstance(model._orig_mod, DEQmHCModel)
            ):
                actual_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                mixing_matrices = actual_model.get_mixing_matrices()
                agg_weights = actual_model.get_aggregation_weights()
                logger.log_mhc_stats(mixing_matrices, agg_weights, step=step)

        # Checkpointing
        if step > 0 and step % config.train.checkpoint_interval == 0:
            print(f"\nSaving checkpoint at step {step}...")
            save_checkpoint(
                model=model._orig_mod if hasattr(model, "_orig_mod") else model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                loss=loss.item(),
                config=asdict(config),
                checkpoint_dir=str(checkpoint_dir),
            )

    # Final evaluation
    print("\nFinal evaluation...")
    val_loss = evaluate(model, val_loader, device, dtype, max_batches=100)
    test_loss = evaluate(model, test_loader, device, dtype, max_batches=100)
    print(f"Final validation loss: {val_loss:.4f}, perplexity: {2**val_loss:.2f}")
    print(f"Final test loss: {test_loss:.4f}, perplexity: {2**test_loss:.2f}")

    # Save final checkpoint
    save_checkpoint(
        model=model._orig_mod if hasattr(model, "_orig_mod") else model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=config.train.max_steps,
        loss=val_loss,
        config=asdict(config),
        checkpoint_dir=str(checkpoint_dir),
        filename="checkpoint_final.pt",
    )

    logger.close()
    print("Training complete!")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DEMHC models")

    # Experiment settings
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "demhc"],
        default="demhc",
        help="Model type to train",
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=["small", "medium"],
        default="small",
        help="Model size preset",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (defaults to {model}_{size})",
    )

    # Model hyperparameters
    parser.add_argument("--hidden-dim", type=int, default=None, help="Hidden dimension")
    parser.add_argument("--num-heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")

    # DEQ settings
    parser.add_argument(
        "--anderson-m",
        type=int,
        default=None,
        help="Anderson acceleration history size (default: 3)",
    )
    parser.add_argument(
        "--deq-max-iters", type=int, default=None, help="Max DEQ forward iterations (default: 8)"
    )
    parser.add_argument(
        "--deq-tol",
        type=float,
        default=None,
        help="DEQ convergence tolerance for final layer (default: 0.15)",
    )
    parser.add_argument(
        "--deq-tol-start",
        type=float,
        default=None,
        help="DEQ tolerance for layer 0; linearly interpolates to --deq-tol (default: None = uniform)",
    )
    parser.add_argument(
        "--deq-beta", type=float, default=None, help="Anderson mixing parameter (default: 1.0)"
    )
    parser.add_argument(
        "--deq-backward-iters",
        type=int,
        default=None,
        help="Max DEQ backward iterations (default: 8)",
    )
    parser.add_argument(
        "--deq-backward-tol",
        type=float,
        default=None,
        help="DEQ backward tolerance (default: 0.15)",
    )

    # DEQ annealing settings
    parser.add_argument(
        "--tol-anneal-start",
        type=float,
        default=None,
        help="Starting tolerance multiplier for annealing (e.g., 2.0 = start 2x looser)",
    )
    parser.add_argument(
        "--tol-anneal-steps",
        type=int,
        default=None,
        help="Steps to anneal tolerance over (default: warmup_steps)",
    )
    parser.add_argument(
        "--alpha-start",
        type=float,
        default=None,
        help="Starting alpha (contraction) value for annealing (lower = stronger contraction)",
    )
    parser.add_argument(
        "--alpha-end",
        type=float,
        default=None,
        help="Ending alpha (contraction) value for annealing (higher = weaker contraction)",
    )
    parser.add_argument(
        "--alpha-anneal-steps",
        type=int,
        default=None,
        help="Steps to anneal alpha over (default: warmup_steps)",
    )

    # DEQ regularization settings
    parser.add_argument(
        "--iter-dropout",
        type=float,
        default=None,
        help="Probability of skipping each DEQ iteration (regularization, default: 0.0)",
    )
    parser.add_argument(
        "--iter-noise",
        type=float,
        default=None,
        help="Std dev of noise added during DEQ iterations (regularization, default: 0.0)",
    )

    # mHC settings
    parser.add_argument("--num-lanes", type=int, default=None, help="Number of lanes for mHC")
    parser.add_argument(
        "--sinkhorn-iters", type=int, default=None, help="Sinkhorn-Knopp iterations"
    )
    parser.add_argument(
        "--lane-dropout",
        type=float,
        default=None,
        help="Probability of dropping each lane (regularization, default: 0.0)",
    )

    # Training settings
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Warmup steps")
    parser.add_argument(
        "--mhc-lr-mult",
        type=float,
        default=None,
        help="Learning rate multiplier for mHC params (default: 30.0)",
    )

    # Other settings
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Create base config from presets
    if args.model == "baseline":
        if args.size == "small":
            config = ExperimentConfig.small_baseline()
        else:
            config = ExperimentConfig.medium_baseline()
    else:
        if args.size == "small":
            config = ExperimentConfig.small_demhc()
        else:
            config = ExperimentConfig.medium_demhc()

    # Override with command line arguments
    if args.name:
        config.name = args.name
    else:
        config.name = f"{args.model}_{args.size}"

    if args.hidden_dim:
        config.model.hidden_dim = args.hidden_dim
    if args.num_heads:
        config.model.num_heads = args.num_heads
    if args.num_layers:
        config.model.num_layers = args.num_layers
    if args.dropout:
        config.model.dropout = args.dropout

    if args.anderson_m:
        config.model.deq.anderson_m = args.anderson_m
    if args.deq_max_iters:
        config.model.deq.max_iters = args.deq_max_iters
    if args.deq_tol:
        config.model.deq.tol = args.deq_tol
    if args.deq_tol_start is not None:
        config.model.deq.tol_start = args.deq_tol_start
    if args.deq_beta:
        config.model.deq.beta = args.deq_beta
    if args.deq_backward_iters:
        config.model.deq.implicit_diff_max_iters = args.deq_backward_iters
    if args.deq_backward_tol:
        config.model.deq.implicit_diff_tol = args.deq_backward_tol

    # Annealing settings
    if args.tol_anneal_start is not None:
        config.model.deq.tol_anneal_start = args.tol_anneal_start
    if args.tol_anneal_steps is not None:
        config.model.deq.tol_anneal_steps = args.tol_anneal_steps
    if args.alpha_start is not None:
        config.model.deq.alpha_start = args.alpha_start
    if args.alpha_end is not None:
        config.model.deq.alpha_end = args.alpha_end
    if args.alpha_anneal_steps is not None:
        config.model.deq.alpha_anneal_steps = args.alpha_anneal_steps

    # Regularization settings
    if args.iter_dropout is not None:
        config.model.deq.iter_dropout = args.iter_dropout
    if args.iter_noise is not None:
        config.model.deq.iter_noise = args.iter_noise

    if args.num_lanes:
        config.model.mhc.num_lanes = args.num_lanes
    if args.sinkhorn_iters:
        config.model.mhc.sinkhorn_iters = args.sinkhorn_iters
    if args.lane_dropout is not None:
        config.model.mhc.lane_dropout = args.lane_dropout

    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.learning_rate:
        config.train.learning_rate = args.learning_rate
    if args.max_steps:
        config.train.max_steps = args.max_steps
    if args.warmup_steps:
        config.train.warmup_steps = args.warmup_steps
    if args.mhc_lr_mult:
        config.train.mhc_lr_mult = args.mhc_lr_mult

    config.output_dir = args.output_dir
    config.seed = args.seed
    config.train.compile = not args.no_compile

    # Print configuration
    print("=" * 60)
    print("DEMHC Training")
    print("=" * 60)
    print(f"Experiment: {config.name}")
    print(f"Model: {config.model_type}")
    print(f"Size: {config.size}")
    print(f"Output: {config.output_dir}/{config.name}")
    if config.model_type == "demhc":
        print("-" * 60)
        print("DEQ Settings:")
        print(f"  max_iters: {config.model.deq.max_iters}")
        print(f"  tol: {config.model.deq.tol}")
        print(f"  tol_start: {config.model.deq.tol_start}")
        print(f"  anderson_m: {config.model.deq.anderson_m}")
        print(f"  beta: {config.model.deq.beta}")
        print(f"  backward_iters: {config.model.deq.implicit_diff_max_iters}")
        print(f"  backward_tol: {config.model.deq.implicit_diff_tol}")
        print("Annealing Settings:")
        print(f"  tol_anneal_start: {config.model.deq.tol_anneal_start}")
        print(f"  tol_anneal_steps: {config.model.deq.tol_anneal_steps or 'warmup_steps'}")
        print(f"  alpha_start: {config.model.deq.alpha_start}")
        print(f"  alpha_end: {config.model.deq.alpha_end}")
        print(f"  alpha_anneal_steps: {config.model.deq.alpha_anneal_steps or 'warmup_steps'}")
        print("Regularization Settings:")
        print(f"  iter_dropout: {config.model.deq.iter_dropout}")
        print(f"  iter_noise: {config.model.deq.iter_noise}")
        print(f"  lane_dropout: {config.model.mhc.lane_dropout}")
        print("mHC Settings:")
        print(f"  num_lanes: {config.model.mhc.num_lanes}")
        print(f"  sinkhorn_iters: {config.model.mhc.sinkhorn_iters}")
        print(f"  lr_mult: {config.train.mhc_lr_mult}")
    print("=" * 60)

    # Run training
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
