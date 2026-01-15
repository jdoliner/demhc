"""Baseline transformer model (NanoGPT-style)."""

import math

import torch
import torch.nn as nn
from torch import Tensor

from demhc.config import ModelConfig
from demhc.models.components.attention import MultiHeadAttention
from demhc.models.components.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """A single transformer block with pre-norm architecture."""

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

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq, hidden_dim)

        Returns:
            Output tensor of shape (batch, seq, hidden_dim)
        """
        # Pre-norm residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class BaselineTransformer(nn.Module):
    """Standard GPT-style transformer for baseline comparison.

    Architecture:
    - Token embeddings + learned positional embeddings
    - N transformer blocks with pre-norm
    - Final layer norm
    - Output projection to vocabulary
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Output
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying (optional but common)
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
        return_hidden: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            return_hidden: If True, also return hidden states

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
            hidden (optional): Final hidden states of shape (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get embeddings
        tok_emb = self.token_emb(input_ids)  # (batch, seq, hidden)
        pos = torch.arange(seq_len, device=device)
        pos_emb = self.pos_emb(pos)  # (seq, hidden)
        x = tok_emb + pos_emb

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if return_hidden:
            return logits, x
        return logits

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model.

        Args:
            non_embedding: If True, exclude embedding parameters from count.
                          This is useful for comparing model sizes fairly since
                          embeddings scale with vocab size, not model capacity.
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
