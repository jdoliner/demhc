"""DEQ-mHC Model: Deep Equilibrium Model with Manifold-Constrained Hyper Connections."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from demhc.config import ModelConfig
from demhc.models.components.attention import MultiHeadAttention
from demhc.models.components.deq import DEQSolver, DEQStats
from demhc.models.components.feedforward import FeedForward
from demhc.models.components.hyper_conn import HyperConnection, LaneAggregator, LaneExpander


@dataclass
class DEQmHCStats:
    """Statistics from DEQ-mHC forward pass."""

    layer_stats: list[DEQStats]

    @property
    def total_forward_iters(self) -> int:
        return sum(s.forward_iters for s in self.layer_stats)

    @property
    def mean_forward_iters(self) -> float:
        return self.total_forward_iters / len(self.layer_stats)

    @property
    def max_forward_residual(self) -> float:
        return max(s.forward_residual for s in self.layer_stats)


class DEQmHCLayerFunction(nn.Module):
    """The layer function that we find a fixed point of.

    This is the inner transformation: attention + FFN applied to the mixed input.
    The fixed point satisfies: x* = f(x*) where f is this function.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attn = MultiHeadAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
        )
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.ffn = FeedForward(
            hidden_dim=config.hidden_dim,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
        )

        # Contraction factor - ensures the layer is a contraction for DEQ convergence
        # This scales the layer output to ensure ||f(x) - f(y)|| < ||x - y||
        # Initialize to -2.0 so sigmoid gives ~0.12, meaning strong contraction
        # (output is 12% new + 88% old, like a small learning rate)
        self.contraction_factor = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq, hidden_dim)

        Returns:
            Output tensor of shape (batch, seq, hidden_dim)
        """
        # Pre-norm transformer block with contraction
        h = x + self.attn(self.ln1(x))
        h = h + self.ffn(self.ln2(h))

        # Apply contraction factor (clamped to ensure < 1)
        alpha = torch.sigmoid(self.contraction_factor)  # in (0, 1)
        return alpha * h + (1 - alpha) * x


class DEQmHCLayer(nn.Module):
    """A single DEQ-mHC layer.

    This layer:
    1. Mixes the k input lanes using a doubly stochastic matrix
    2. Finds a fixed point of the layer function for each lane
    3. Returns the k output lanes
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Hyper connection for lane mixing
        self.hyper_conn = HyperConnection(
            num_lanes=config.mhc.num_lanes,
            sinkhorn_iters=config.mhc.sinkhorn_iters,
            sinkhorn_eps=config.mhc.sinkhorn_eps,
        )

        # The layer function that we find fixed point of
        self.layer_fn = DEQmHCLayerFunction(config)

        # DEQ solver
        self.deq_solver = DEQSolver(
            solver=config.deq.solver,
            anderson_m=config.deq.anderson_m,
            max_iters=config.deq.max_iters,
            tol=config.deq.tol,
            beta=config.deq.beta,
            implicit_diff_max_iters=config.deq.implicit_diff_max_iters,
            implicit_diff_tol=config.deq.implicit_diff_tol,
        )

    def forward(self, lanes: Tensor) -> tuple[Tensor, DEQStats]:
        """
        Args:
            lanes: Input lanes of shape (batch, seq, num_lanes, hidden_dim)

        Returns:
            Tuple of (output_lanes, stats) where output_lanes has same shape as input
        """
        batch, seq, num_lanes, hidden = lanes.shape

        # Step 1: Mix lanes using doubly stochastic matrix
        mixed = self.hyper_conn(lanes)  # (batch, seq, num_lanes, hidden)

        # Step 2: Find fixed point for each lane independently
        # We process all lanes in parallel by folding them into batch dimension
        mixed_flat = mixed.reshape(batch * seq * num_lanes, hidden)

        # Wrapper to handle the flattened input
        def layer_fn_wrapper(x: Tensor) -> Tensor:
            # x: (batch*seq*num_lanes, hidden)
            x_reshaped = x.reshape(batch, seq, num_lanes, hidden)
            # Apply layer function to each lane position
            # We need to process each (batch, seq) position for attention
            out = torch.zeros_like(x_reshaped)
            for k in range(num_lanes):
                out[:, :, k, :] = self.layer_fn(x_reshaped[:, :, k, :])
            return out.reshape(batch * seq * num_lanes, hidden)

        # Actually, let's do this more efficiently - process all lanes together
        # by treating lanes as batch dimension for attention
        def layer_fn_efficient(x: Tensor) -> Tensor:
            # x: (batch, seq, num_lanes, hidden)
            b, s, k, h = x.shape
            # Reshape to (batch * num_lanes, seq, hidden) for attention
            x_for_attn = x.permute(0, 2, 1, 3).reshape(b * k, s, h)
            out = self.layer_fn(x_for_attn)
            # Reshape back
            return out.reshape(b, k, s, h).permute(0, 2, 1, 3)

        # Find fixed point
        output, stats = self.deq_solver(layer_fn_efficient, mixed)

        return output, stats


class DEQmHCModel(nn.Module):
    """Deep Equilibrium Model with Manifold-Constrained Hyper Connections.

    Architecture:
    - Token embeddings + learned positional embeddings
    - Expand to k lanes
    - N DEQ-mHC layers (each finds a fixed point)
    - Lane aggregation (collapse k lanes to 1)
    - Output projection to vocabulary
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim)

        # Lane expansion (1 -> k lanes)
        self.lane_expander = LaneExpander(config.mhc.num_lanes)

        # DEQ-mHC layers
        self.layers = nn.ModuleList([
            DEQmHCLayer(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Lane aggregation (k -> 1)
        self.lane_aggregator = LaneAggregator(config.mhc.num_lanes)

        # Output
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using GPT-2 style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, DEQmHCStats]:
        """
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            return_stats: If True, also return DEQ statistics

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
            stats (optional): DEQ convergence statistics
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get embeddings
        tok_emb = self.token_emb(input_ids)  # (batch, seq, hidden)
        pos = torch.arange(seq_len, device=device)
        pos_emb = self.pos_emb(pos)  # (seq, hidden)
        x = tok_emb + pos_emb

        # Expand to k lanes
        lanes = self.lane_expander(x)  # (batch, seq, k, hidden)

        # Apply DEQ-mHC layers
        all_stats = []
        for layer in self.layers:
            lanes, stats = layer(lanes)
            all_stats.append(stats)

        # Aggregate lanes
        x = self.lane_aggregator(lanes)  # (batch, seq, hidden)

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if return_stats:
            return logits, DEQmHCStats(layer_stats=all_stats)
        return logits

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model.

        Args:
            non_embedding: If True, exclude embedding parameters from count.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_emb.weight.numel()
            n_params -= self.pos_emb.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Starting token IDs of shape (batch, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens

        Returns:
            Generated token IDs of shape (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids[:, -self.config.max_seq_len:]

            # Get logits for next token
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def get_mixing_matrices(self) -> list[Tensor]:
        """Get the current doubly stochastic mixing matrices for all layers.

        Useful for visualization and debugging.
        """
        return [layer.hyper_conn.get_mixing_matrix() for layer in self.layers]

    def get_aggregation_weights(self) -> Tensor:
        """Get the current lane aggregation weights.

        Useful for visualization and debugging.
        """
        return self.lane_aggregator.get_weights()
