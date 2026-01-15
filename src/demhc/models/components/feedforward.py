"""Feed-forward network implementation."""

import torch.nn as nn
from torch import Tensor


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation.

    Uses the standard transformer FFN architecture:
    FFN(x) = Linear(GELU(Linear(x)))

    Can optionally use the "SwiGLU" variant which has shown better performance.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_glu: bool = False,
    ):
        super().__init__()
        self.use_glu = use_glu

        if use_glu:
            # SwiGLU variant: gate and value projections
            self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
            self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
            self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)
            self.act = nn.SiLU()
        else:
            # Standard GELU FFN
            self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
            self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)
            self.act = nn.GELU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        if self.use_glu:
            # SwiGLU: gate * activation(up)
            gate = self.act(self.gate_proj(x))
            up = self.up_proj(x)
            x = gate * up
        else:
            # Standard: activation(up)
            x = self.act(self.up_proj(x))

        x = self.down_proj(x)
        x = self.dropout(x)
        return x
