"""Manifold-Constrained Hyper Connections (mHC) implementation."""

import torch
import torch.nn as nn
from torch import Tensor

from demhc.models.components.sinkhorn import project_to_birkhoff, project_to_simplex


class HyperConnection(nn.Module):
    """Hyper Connection layer with Birkhoff manifold constraint.

    This module implements the lane mixing mechanism where k lanes are combined
    using a doubly stochastic mixing matrix. The matrix is constrained to the
    Birkhoff polytope via sigmoid (non-negativity) + Sinkhorn-Knopp (doubly stochastic).

    The mixing can be thought of as a generalization of residual connections:
    - Traditional residual: 2 lanes (input, layer output) mixed with static [[0.5, 0.5], [0.5, 0.5]]
    - mHC: k lanes mixed with learnable doubly stochastic matrix
    """

    def __init__(
        self,
        num_lanes: int,
        sinkhorn_iters: int = 10,
        sinkhorn_eps: float = 1e-8,
    ):
        super().__init__()
        self.num_lanes = num_lanes
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_eps = sinkhorn_eps

        # Learnable mixing parameters (unconstrained logits)
        # Initialize near-identity: each lane primarily keeps its own value
        # but can learn to mix with others. The diagonal bias encourages
        # each lane to maintain its identity initially.
        init_logits = torch.zeros(num_lanes, num_lanes)
        init_logits += 2.0 * torch.eye(num_lanes)  # Bias toward identity
        init_logits += 0.1 * torch.randn(num_lanes, num_lanes)  # Small noise
        self.mix_logits = nn.Parameter(init_logits)

    def get_mixing_matrix(self) -> Tensor:
        """Compute the doubly stochastic mixing matrix.

        Returns:
            Tensor of shape (num_lanes, num_lanes) that is doubly stochastic
            (rows and columns sum to 1, all entries non-negative).
        """
        return project_to_birkhoff(
            self.mix_logits,
            num_iters=self.sinkhorn_iters,
            eps=self.sinkhorn_eps,
        )

    def forward(self, lanes: Tensor) -> Tensor:
        """Mix lanes using the doubly stochastic matrix.

        Args:
            lanes: Tensor of shape (batch, seq, num_lanes, hidden_dim)

        Returns:
            Mixed lanes of shape (batch, seq, num_lanes, hidden_dim)
        """
        # Get doubly stochastic mixing matrix
        M = self.get_mixing_matrix()  # (k, k)

        # Apply mixing: each output lane is a weighted combination of input lanes
        # lanes: (batch, seq, k, hidden) -> (batch, seq, hidden, k) for matmul
        # M: (k, k) where M[i, j] = weight of input lane j for output lane i
        mixed = torch.einsum("ij, bsjd -> bsid", M, lanes)

        return mixed


class LaneAggregator(nn.Module):
    """Aggregates k lanes into a single output stream.

    Uses a learnable probability vector (simplex) to combine lanes.
    This is the 1D version of the Birkhoff constraint.
    """

    def __init__(self, num_lanes: int):
        super().__init__()
        self.num_lanes = num_lanes

        # Learnable aggregation weights (unconstrained logits)
        # Initialize with small random values to break symmetry
        self.agg_logits = nn.Parameter(0.1 * torch.randn(num_lanes))

    def get_weights(self) -> Tensor:
        """Get the aggregation weights (probability simplex).

        Returns:
            Tensor of shape (num_lanes,) summing to 1.
        """
        return project_to_simplex(self.agg_logits)

    def forward(self, lanes: Tensor) -> Tensor:
        """Aggregate lanes into a single stream.

        Args:
            lanes: Tensor of shape (batch, seq, num_lanes, hidden_dim)

        Returns:
            Aggregated output of shape (batch, seq, hidden_dim)
        """
        weights = self.get_weights()  # (k,)

        # Weighted sum over lanes
        # lanes: (batch, seq, k, hidden)
        # weights: (k,)
        output = torch.einsum("k, bskd -> bsd", weights, lanes)

        return output


class LaneExpander(nn.Module):
    """Expands a single input stream into k lanes.

    Each lane gets a learned linear projection of the input, giving
    each lane a different "view" of the data from the start.
    """

    def __init__(self, num_lanes: int, hidden_dim: int):
        super().__init__()
        self.num_lanes = num_lanes
        self.hidden_dim = hidden_dim

        # Each lane gets its own projection (initialized near-identity)
        self.lane_projections = nn.Parameter(
            torch.eye(hidden_dim).unsqueeze(0).expand(num_lanes, -1, -1).clone()
            + 0.01 * torch.randn(num_lanes, hidden_dim, hidden_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Expand input to k lanes with learned projections.

        Args:
            x: Input tensor of shape (batch, seq, hidden_dim)

        Returns:
            Expanded output of shape (batch, seq, num_lanes, hidden_dim)
        """
        # x: (batch, seq, hidden)
        # lane_projections: (num_lanes, hidden, hidden)
        # output: (batch, seq, num_lanes, hidden)
        return torch.einsum("bsh, khd -> bskd", x, self.lane_projections)
