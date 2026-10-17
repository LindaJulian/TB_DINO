from __future__ import annotations

import torch
from torch import nn


class AdaptiveTemporalInteraction(nn.Module):
    """Two-head temporal attention independently applied at each spatial token."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 2,
        dropout: float = 0.1,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"ATIM 维度 {dim} 必须能被注意力头数 {num_heads} 整除")
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        features: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError("ATIM 输入应为 [B,T,C,H,W]")
        batch, time, channels, height, width = features.shape
        if temporal_mask.shape != (batch, time):
            raise ValueError("temporal_mask 形状必须为 [B,T]")
        tokens = features.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, time, channels
        )
        padding_mask = (
            ~temporal_mask[:, None, None, :]
            .expand(batch, height, width, time)
            .reshape(batch * height * width, time)
        )
        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        tokens = self.norm1(tokens + attended)
        tokens = self.norm2(tokens + self.feed_forward(tokens))
        valid_queries = (~padding_mask).unsqueeze(-1).to(tokens.dtype)
        tokens = tokens * valid_queries
        return tokens.reshape(batch, height, width, time, channels).permute(
            0, 3, 4, 1, 2
        )

