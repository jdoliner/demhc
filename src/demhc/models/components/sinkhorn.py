"""Sinkhorn-Knopp algorithm for projecting onto the Birkhoff polytope."""

import torch
from torch import Tensor


@torch.compile
def sinkhorn_knopp(
    A: Tensor,
    num_iters: int = 10,
    eps: float = 1e-8,
) -> Tensor:
    """Project a positive matrix onto the Birkhoff polytope (doubly stochastic matrices).

    Uses the Sinkhorn-Knopp algorithm which alternates between row and column
    normalization until convergence to a doubly stochastic matrix.

    Args:
        A: Input positive matrix of shape (..., n, n). Should already have
           non-negativity applied (e.g., via sigmoid).
        num_iters: Number of alternating normalization iterations.
        eps: Small constant for numerical stability.

    Returns:
        Doubly stochastic matrix of the same shape as input.
    """
    # Ensure positivity with eps floor
    A = A.clamp(min=eps)

    for _ in range(num_iters):
        # Row normalization: each row sums to 1
        A = A / (A.sum(dim=-1, keepdim=True) + eps)
        # Column normalization: each column sums to 1
        A = A / (A.sum(dim=-2, keepdim=True) + eps)

    return A


def project_to_birkhoff(
    logits: Tensor,
    num_iters: int = 10,
    eps: float = 1e-8,
) -> Tensor:
    """Project unconstrained logits to a doubly stochastic matrix.

    This is the full pipeline: sigmoid (non-negativity) -> Sinkhorn (doubly stochastic).
    Uses straight-through estimator for better gradient flow.

    Args:
        logits: Unconstrained learnable parameters of shape (..., n, n).
        num_iters: Number of Sinkhorn-Knopp iterations.
        eps: Numerical stability constant.

    Returns:
        Doubly stochastic matrix of the same shape.
    """
    # Apply sigmoid for non-negativity (values in (0, 1))
    A = torch.sigmoid(logits)
    
    # Project to Birkhoff polytope
    A_projected = sinkhorn_knopp(A, num_iters=num_iters, eps=eps)
    
    # Straight-through estimator: forward uses projected, backward uses pre-projection
    # This helps gradients flow through even when Sinkhorn "flattens" them
    return A + (A_projected - A).detach()


def project_to_simplex(logits: Tensor, dim: int = -1) -> Tensor:
    """Project unconstrained logits to a probability simplex.

    Simply applies softmax to ensure non-negativity and sum-to-one constraint.

    Args:
        logits: Unconstrained parameters of shape (..., n).
        dim: Dimension along which to normalize.

    Returns:
        Probability vector summing to 1 along the specified dimension.
    """
    return torch.softmax(logits, dim=dim)


def check_doubly_stochastic(A: Tensor, tol: float = 1e-3) -> tuple[bool, dict]:
    """Check if a matrix is approximately doubly stochastic.

    Args:
        A: Matrix to check of shape (..., n, n).
        tol: Tolerance for sum-to-one checks.

    Returns:
        Tuple of (is_valid, diagnostics_dict).
    """
    row_sums = A.sum(dim=-1)
    col_sums = A.sum(dim=-2)

    row_err = (row_sums - 1.0).abs().max().item()
    col_err = (col_sums - 1.0).abs().max().item()
    min_val = A.min().item()

    is_valid = row_err < tol and col_err < tol and min_val >= 0

    diagnostics = {
        "row_sum_error": row_err,
        "col_sum_error": col_err,
        "min_value": min_val,
        "is_valid": is_valid,
    }

    return is_valid, diagnostics
