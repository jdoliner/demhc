"""Multi-head self-attention implementation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking.

    Uses pre-norm architecture (LayerNorm applied before attention in the caller).
    Supports both learned positional embeddings (applied externally) and RoPE.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_rope: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_rope = use_rope

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # RoPE embeddings (optional)
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, base=rope_base)
        else:
            self.rope = None

        # Register causal mask as a buffer (kept for compatibility, but not used with SDPA)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE if enabled
        if self.rope is not None:
            cos, sin = self.rope(x, seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention with causal mask
        # Using PyTorch's optimized scaled_dot_product_attention
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,  # Use is_causal instead for efficiency
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_out)
        output = self.dropout(output)

        return output


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    This is an optional enhancement that can be used instead of absolute
    positional embeddings. It encodes position through rotation of the
    query and key vectors.
    """

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute rotation matrices
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build the cos/sin cache for the given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
        """Return cos and sin for the given sequence length."""
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    # cos, sin: (seq_len, dim)
    # q, k: (batch, heads, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
