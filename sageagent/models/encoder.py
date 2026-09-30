"""Multimodal Transformer encoder with learnable mask tokens for missing modalities."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MultimodalTransformerEncoder(nn.Module):
    """Encodes a set of modality feature vectors into one fused embedding.

    Each modality has its own 2-layer MLP projection. Missing modalities are
    replaced by learnable mask tokens, a CLS token is prepended, and a pre-LN
    Transformer mixes the tokens. The output fuses the CLS token with a masked
    mean over the present modality tokens.

    Args:
        input_dims: feature dimension of each modality (in pathway order).
        hidden_dim: token dimension d.
        n_layers:   Transformer layers.
        n_heads:    attention heads.
        ffn_mult:   feed-forward width as a multiple of d.
        dropout:    dropout rate.
        positional: add learnable positional embeddings.
    """

    def __init__(self, input_dims: Sequence[int], hidden_dim: int = 128, n_layers: int = 2,
                 n_heads: int = 4, ffn_mult: int = 4, dropout: float = 0.1, positional: bool = True):
        super().__init__()
        self.input_dims = list(input_dims)
        n_mod = len(self.input_dims)

        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            )
            for d in self.input_dims
        ])
        self.modality_embed = nn.Parameter(torch.randn(1, n_mod, hidden_dim) * 0.02)
        self.mask_tokens = nn.Parameter(torch.randn(n_mod, hidden_dim) * 0.02)
        self.positional = positional
        if positional:
            self.pos_embed = nn.Parameter(torch.randn(1, 1 + n_mod, hidden_dim) * 0.02)
            self.pos_drop = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=ffn_mult * hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.fusion = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, M, D) modality features, zero-padded to the largest dim.
            mask:     (B, M) 1 = modality present, 0 = missing.

        Returns:
            fused embedding (B, d) and per-modality tokens (B, M, d).
        """
        tokens = torch.stack(
            [proj(features[:, i, :d]) for i, (proj, d) in enumerate(zip(self.input_projs, self.input_dims))],
            dim=1,
        ) + self.modality_embed
        present = mask.unsqueeze(-1)
        tokens = tokens * present + self.mask_tokens.unsqueeze(0) * (1.0 - present)

        x = torch.cat([self.cls_token.expand(len(tokens), -1, -1), tokens], dim=1)
        if self.positional:
            x = self.pos_drop(x + self.pos_embed)
        x = self.output_norm(self.transformer(x))

        cls, modality_tokens = x[:, 0], x[:, 1:]
        pooled = (modality_tokens * present).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return self.fusion(torch.cat([cls, pooled], dim=-1)), modality_tokens
