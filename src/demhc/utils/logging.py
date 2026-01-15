"""TensorBoard logging utilities."""

import os
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


def setup_tensorboard(log_dir: str, experiment_name: str) -> SummaryWriter:
    """Set up TensorBoard writer.

    Args:
        log_dir: Base directory for logs
        experiment_name: Name of the experiment (used as subdirectory)

    Returns:
        TensorBoard SummaryWriter instance
    """
    full_path = Path(log_dir) / experiment_name
    full_path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(full_path))


class MetricsLogger:
    """Helper class for logging metrics to TensorBoard."""

    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, value: int) -> None:
        self._step = value

    def log_scalar(self, tag: str, value: float, step: int | None = None) -> None:
        """Log a scalar value."""
        step = step if step is not None else self._step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict[str, float], step: int | None = None) -> None:
        """Log multiple scalars under a main tag."""
        step = step if step is not None else self._step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values: Any, step: int | None = None) -> None:
        """Log a histogram of values."""
        step = step if step is not None else self._step
        self.writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: int | None = None) -> None:
        """Log text."""
        step = step if step is not None else self._step
        self.writer.add_text(tag, text, step)

    def log_training_step(
        self,
        loss: float,
        learning_rate: float,
        grad_norm: float | None = None,
        step: int | None = None,
    ) -> None:
        """Log standard training metrics."""
        step = step if step is not None else self._step
        self.log_scalar("train/loss", loss, step)
        self.log_scalar("train/learning_rate", learning_rate, step)
        self.log_scalar("train/perplexity", 2**loss, step)  # Perplexity = 2^loss for cross-entropy
        if grad_norm is not None:
            self.log_scalar("train/grad_norm", grad_norm, step)

    def log_validation(
        self,
        loss: float,
        step: int | None = None,
    ) -> None:
        """Log validation metrics."""
        step = step if step is not None else self._step
        self.log_scalar("val/loss", loss, step)
        self.log_scalar("val/perplexity", 2**loss, step)

    def log_deq_stats(
        self,
        layer_iters: list[int],
        layer_residuals: list[float],
        step: int | None = None,
    ) -> None:
        """Log DEQ convergence statistics."""
        step = step if step is not None else self._step

        for i, (iters, residual) in enumerate(zip(layer_iters, layer_residuals)):
            self.log_scalar(f"deq/layer_{i}_iters", iters, step)
            self.log_scalar(f"deq/layer_{i}_residual", residual, step)

        self.log_scalar("deq/total_iters", sum(layer_iters), step)
        self.log_scalar("deq/mean_iters", sum(layer_iters) / len(layer_iters), step)
        self.log_scalar("deq/max_residual", max(layer_residuals), step)

    def log_mhc_stats(
        self,
        mixing_matrices: list[Any],
        aggregation_weights: Any,
        step: int | None = None,
    ) -> None:
        """Log mHC statistics (mixing matrix properties)."""
        step = step if step is not None else self._step

        for i, M in enumerate(mixing_matrices):
            # Log matrix statistics
            self.log_scalar(f"mhc/layer_{i}_max_weight", M.max().item(), step)
            self.log_scalar(f"mhc/layer_{i}_min_weight", M.min().item(), step)
            self.log_scalar(f"mhc/layer_{i}_entropy", -(M * M.log().clamp(min=-100)).sum().item(), step)

        # Aggregation weights
        self.log_scalar("mhc/agg_max_weight", aggregation_weights.max().item(), step)
        self.log_scalar("mhc/agg_min_weight", aggregation_weights.min().item(), step)

    def flush(self) -> None:
        """Flush the writer."""
        self.writer.flush()

    def close(self) -> None:
        """Close the writer."""
        self.writer.close()
