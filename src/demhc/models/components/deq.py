"""Deep Equilibrium Model solver with Anderson Acceleration and implicit differentiation."""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class DEQStats:
    """Statistics from DEQ forward/backward passes."""

    forward_iters: int
    forward_residual: float
    backward_iters: int | None = None
    backward_residual: float | None = None


def anderson_acceleration(
    f: Callable[[Tensor], Tensor],
    x0: Tensor,
    m: int = 5,
    max_iters: int = 30,
    tol: float = 1e-5,
    beta: float = 1.0,
) -> tuple[Tensor, DEQStats]:
    """Find fixed point x* such that f(x*) = x* using Anderson Acceleration.

    Anderson Acceleration maintains a history of previous iterates and uses
    least-squares to find an optimal linear combination that accelerates
    convergence.

    Args:
        f: The function to find a fixed point of.
        x0: Initial guess, shape (batch, seq, hidden) or similar.
        m: Number of previous iterates to use (history size).
        max_iters: Maximum number of iterations.
        tol: Convergence tolerance (relative residual norm).
        beta: Mixing parameter (1.0 = pure Anderson, 0.0 = pure fixed-point).

    Returns:
        Tuple of (fixed_point, stats).
    """
    # Flatten x for easier manipulation
    batch_shape = x0.shape
    x = x0.reshape(x0.shape[0], -1)  # (batch, features)
    device = x.device
    dtype = x.dtype

    # History buffers
    X_history: list[Tensor] = []  # Previous iterates
    F_history: list[Tensor] = []  # f(x) - x residuals

    for k in range(max_iters):
        # Compute f(x) and residual
        x_full = x.reshape(batch_shape)
        fx_full = f(x_full)
        fx = fx_full.reshape(x.shape)

        residual = fx - x
        residual_norm = residual.norm(dim=-1).mean()
        x_norm = x.norm(dim=-1).mean() + 1e-8
        rel_residual = (residual_norm / x_norm).item()

        # Check convergence
        if rel_residual < tol:
            stats = DEQStats(forward_iters=k + 1, forward_residual=rel_residual)
            return fx_full, stats

        # Update history
        X_history.append(x.clone())
        F_history.append(residual.clone())

        # Keep only last m entries
        if len(X_history) > m:
            X_history.pop(0)
            F_history.pop(0)

        # Anderson acceleration step
        if len(X_history) >= 2:
            # Build matrices for least-squares
            # We want to find alpha such that sum(alpha_i * F_i) is minimized
            mk = len(F_history)

            # Stack residuals: (batch, mk, features)
            F_stack = torch.stack(F_history, dim=1)

            # Compute differences: F_i - F_{mk-1} for i < mk-1
            F_diff = F_stack[:, :-1, :] - F_stack[:, -1:, :]  # (batch, mk-1, features)

            # Solve least squares: minimize ||F_diff @ gamma - (-F_last)||
            # Using normal equations: (F_diff^T F_diff) gamma = F_diff^T (-F_last)
            F_last = F_stack[:, -1, :]  # (batch, features)

            # Cast to float32 for linear algebra (bfloat16 not supported by linalg.solve)
            F_diff_f32 = F_diff.float()
            F_last_f32 = F_last.float()

            # (batch, mk-1, mk-1)
            FtF = torch.bmm(F_diff_f32, F_diff_f32.transpose(1, 2))
            # Add regularization for stability
            FtF = FtF + 1e-6 * torch.eye(mk - 1, device=device, dtype=torch.float32).unsqueeze(0)

            # (batch, mk-1) - ensure float32
            Ftf = torch.bmm(F_diff_f32, (-F_last_f32).unsqueeze(-1)).squeeze(-1).float()

            # Solve for gamma using pseudo-inverse (more robust and torch.compile friendly)
            # gamma = (F_diff^T F_diff)^-1 F_diff^T (-F_last)
            gamma = torch.linalg.lstsq(FtF, Ftf.unsqueeze(-1)).solution.squeeze(-1)

            # Cast alpha back to original dtype
            alpha = torch.cat([gamma, 1.0 - gamma.sum(dim=-1, keepdim=True)], dim=-1)
            alpha = alpha.to(dtype)

            # Compute new x as weighted combination
            X_stack = torch.stack(X_history, dim=1)  # (batch, mk, features)

            # x_new = sum_i alpha_i * (x_i + beta * f_i)
            x_updated = X_stack + beta * F_stack
            x_new = (alpha.unsqueeze(-1) * x_updated).sum(dim=1)

            x = x_new
        else:
            # Simple fixed-point iteration for first step
            x = x + beta * residual

    # Compute final residual after the last iteration's update
    # The loop checks convergence at the START of each iteration, so after the last
    # x update we need to check once more to see if we actually converged
    x_full = x.reshape(batch_shape)
    fx_full = f(x_full)
    fx = fx_full.reshape(x.shape)

    residual = fx - x
    residual_norm = residual.norm(dim=-1).mean()
    x_norm = x.norm(dim=-1).mean() + 1e-8
    rel_residual = (residual_norm / x_norm).item()

    # Check if we actually converged on the final iteration
    if rel_residual < tol:
        stats = DEQStats(forward_iters=max_iters, forward_residual=rel_residual)
        return fx_full, stats

    # Return last iterate if we didn't converge
    stats = DEQStats(forward_iters=max_iters, forward_residual=rel_residual)
    return fx_full, stats


def simple_fixed_point(
    f: Callable[[Tensor], Tensor],
    x0: Tensor,
    max_iters: int = 30,
    tol: float = 1e-5,
    beta: float = 1.0,
) -> tuple[Tensor, DEQStats]:
    """Simple fixed-point iteration: x_{k+1} = (1-beta)*x_k + beta*f(x_k).

    This is a fallback solver that is simpler but slower to converge than
    Anderson acceleration.
    """
    x = x0

    for k in range(max_iters):
        fx = f(x)
        residual = fx - x
        residual_norm = residual.norm()
        x_norm = x.norm() + 1e-8
        rel_residual = (residual_norm / x_norm).item()

        if rel_residual < tol:
            stats = DEQStats(forward_iters=k + 1, forward_residual=rel_residual)
            return fx, stats

        x = x + beta * residual

    # Compute final residual after the last iteration's update
    # The loop checks convergence at the START of each iteration, so after the last
    # x update we need to check once more to see if we actually converged
    fx = f(x)
    residual = fx - x
    residual_norm = residual.norm()
    x_norm = x.norm() + 1e-8
    rel_residual = (residual_norm / x_norm).item()

    # Return with the correct residual (and the iteration count indicates max was reached)
    stats = DEQStats(forward_iters=max_iters, forward_residual=rel_residual)
    return fx, stats


class DEQFixedPoint(torch.autograd.Function):
    """Custom autograd function for DEQ with implicit differentiation.

    Uses the implicit function theorem to compute gradients without
    backpropagating through all the forward iterations.

    At the fixed point x* where f(x*) = x*, we have:
        dx*/dtheta = (I - df/dx)^{-1} @ df/dtheta

    For the backward pass, we solve:
        (I - J^T) v = grad_output
    where J = df/dx, using fixed-point iteration.
    """

    @staticmethod
    def forward(
        ctx,
        f_module: nn.Module,
        x0: Tensor,
        solver: str,
        anderson_m: int,
        max_iters: int,
        tol: float,
        beta: float,
        implicit_diff_max_iters: int,
        implicit_diff_tol: float,
    ) -> tuple[Tensor, DEQStats]:
        """Forward pass: find fixed point."""
        # Choose solver
        if solver == "anderson":
            x_star, stats = anderson_acceleration(
                f_module, x0, m=anderson_m, max_iters=max_iters, tol=tol, beta=beta
            )
        else:
            x_star, stats = simple_fixed_point(
                f_module, x0, max_iters=max_iters, tol=tol, beta=beta
            )

        # Save for backward - we need the fixed point and the function
        ctx.f_module = f_module
        ctx.x_star = x_star.detach().clone()
        ctx.implicit_diff_max_iters = implicit_diff_max_iters
        ctx.implicit_diff_tol = implicit_diff_tol

        return x_star, stats

    @staticmethod
    def backward(ctx, grad_output: Tensor, grad_stats: None) -> tuple:
        """Backward pass: implicit differentiation."""
        f_module = ctx.f_module
        x_star = ctx.x_star.requires_grad_(True)
        max_iters = ctx.implicit_diff_max_iters
        tol = ctx.implicit_diff_tol

        # We need to solve: (I - J^T) v = grad_output
        # where J = df/dx at x_star
        # We do this via fixed-point iteration: v_{k+1} = grad_output + J^T v_k

        v = grad_output.clone()
        backward_iters = 0
        backward_residual = float("inf")

        for k in range(max_iters):
            # Compute J^T v via vector-Jacobian product
            with torch.enable_grad():
                fx = f_module(x_star)
                # vjp: compute v^T @ J = (J^T @ v)^T
                (Jt_v,) = torch.autograd.grad(
                    fx, x_star, grad_outputs=v, retain_graph=True, create_graph=False
                )

            v_new = grad_output + Jt_v

            # Check convergence
            diff = (v_new - v).norm()
            v_norm = v.norm() + 1e-8
            rel_diff = (diff / v_norm).item()

            v = v_new
            backward_iters = k + 1
            backward_residual = rel_diff

            if rel_diff < tol:
                break

        # Now compute gradients w.r.t. parameters
        # We need: v^T @ (df/dtheta)
        with torch.enable_grad():
            x_star_grad = x_star.detach().requires_grad_(True)
            fx = f_module(x_star_grad)

        # Compute parameter gradients
        params = list(f_module.parameters())
        if params:
            param_grads = torch.autograd.grad(
                fx,
                params,
                grad_outputs=v,
                allow_unused=True,
            )
        else:
            param_grads = []

        # Gradient w.r.t. x0 is v (the implicit gradient propagates to input)
        grad_x0 = v

        # Return gradients for all forward inputs
        # (f_module, x0, solver, anderson_m, max_iters, tol, beta, implicit_max, implicit_tol)
        return (None, grad_x0, None, None, None, None, None, None, None)


class DEQSolver(nn.Module):
    """Wrapper module for DEQ solving with configurable settings.

    This module wraps the fixed-point solver and handles the autograd
    function setup cleanly.
    """

    def __init__(
        self,
        solver: str = "anderson",
        anderson_m: int = 5,
        max_iters: int = 30,
        tol: float = 1e-5,
        beta: float = 1.0,
        implicit_diff_max_iters: int = 30,
        implicit_diff_tol: float = 1e-5,
    ):
        super().__init__()
        self.solver = solver
        self.anderson_m = anderson_m
        self.max_iters = max_iters
        self.tol = tol
        self.beta = beta
        self.implicit_diff_max_iters = implicit_diff_max_iters
        self.implicit_diff_tol = implicit_diff_tol

    def forward(
        self,
        f: Callable[[Tensor], Tensor],
        x0: Tensor,
    ) -> tuple[Tensor, DEQStats]:
        """Find fixed point of f starting from x0.

        Args:
            f: The layer function (must be a callable, typically a nn.Module).
            x0: Initial guess.

        Returns:
            Tuple of (fixed_point, stats).
        """
        if self.training:
            # Use unrolled differentiation - simpler and works well for small iteration counts
            return deq_forward_unrolled(
                f,
                x0,
                solver=self.solver,
                anderson_m=self.anderson_m,
                max_iters=self.max_iters,
                tol=self.tol,
                beta=self.beta,
            )
        else:
            # During eval, just run the solver without custom backward
            if self.solver == "anderson":
                return anderson_acceleration(
                    f, x0, m=self.anderson_m, max_iters=self.max_iters, tol=self.tol, beta=self.beta
                )
            else:
                return simple_fixed_point(
                    f, x0, max_iters=self.max_iters, tol=self.tol, beta=self.beta
                )


def deq_forward_unrolled(
    f: Callable[[Tensor], Tensor],
    x0: Tensor,
    solver: str = "anderson",
    anderson_m: int = 5,
    max_iters: int = 30,
    tol: float = 1e-5,
    beta: float = 1.0,
    **kwargs,  # Ignore implicit diff params
) -> tuple[Tensor, DEQStats]:
    """Forward pass with unrolled differentiation.

    Instead of implicit differentiation, we simply unroll the fixed-point
    iterations and let autograd backprop through them. This is simpler and
    works well when max_iters is small (< 20).

    For larger iteration counts, consider using gradient checkpointing.
    """
    if solver == "anderson":
        return anderson_acceleration(f, x0, m=anderson_m, max_iters=max_iters, tol=tol, beta=beta)
    else:
        return simple_fixed_point(f, x0, max_iters=max_iters, tol=tol, beta=beta)
