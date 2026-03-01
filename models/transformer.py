"""
Multimodal Transformer Encoder.

Encodes multimodal features with missing-modality handling via learnable mask tokens.
All architecture hyper-parameters are passed in — nothing hardcoded.

Enhanced with:
- Per-modality input projections
- Modality type embeddings
- Positional encoding covering CLS token
- Masked mean pooling
- Dual fusion: CLS + masked mean pool
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MultimodalTransformerEncoder(nn.Module):
    """
    Transformer encoder for multimodal data.

    - Per-modality 2-layer MLP projections
    - Learnable mask tokens replace missing modalities
    - CLS token aggregates cross-modal information
    - Pre-LayerNorm transformer layers for stable training
    - Dual fusion: CLS (cross-attention) + masked mean pool (robust averaging)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_modalities: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.1,
        use_positional: bool = True,
        debug: bool = False,
    ):
        """
        Args:
            input_dim: per-modality feature dimension (e.g. 32)
            hidden_dim: transformer hidden dimension (e.g. 128)
            n_modalities: number of modalities (e.g. 4)
            n_layers: number of transformer layers
            n_heads: number of attention heads
            dropout: dropout rate
            use_positional: add learnable positional encoding
            debug: enable debug prints
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_modalities = n_modalities
        self.debug = debug

        # per-modality 2-layer MLP projections
        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(n_modalities)
        ])

        # learnable modality type embeddings
        self.modality_embed = nn.Parameter(torch.randn(1, n_modalities, hidden_dim) * 0.02)

        # learnable mask token per modality (used when modality is missing)
        self.mask_tokens = nn.Parameter(torch.randn(n_modalities, hidden_dim) * 0.02)

        # learnable positional encoding (includes CLS position)
        self.use_positional = use_positional
        if use_positional:
            self.pos_embed = nn.Parameter(torch.randn(1, 1 + n_modalities, hidden_dim) * 0.02)
            self.pos_drop = nn.Dropout(dropout)

        # CLS token for aggregated output
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # transformer layers (pre-LN for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)

        # dual fusion: combine CLS + masked mean pool
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self._init_weights()

        if self.debug:
            n_params = sum(p.numel() for p in self.parameters())
            print(f"[TRANSFORMER DEBUG] init: input_dim={input_dim}, hidden_dim={hidden_dim}, "
                  f"n_mod={n_modalities}, n_layers={n_layers}, n_heads={n_heads}, "
                  f"params={n_params:,}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        return_per_modality: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            features: (B, n_modalities, input_dim)
            mask:     (B, n_modalities)  1=present, 0=missing
            return_per_modality: also return per-modality representations

        Returns:
            fused: (B, hidden_dim) — dual fusion of CLS + masked mean pool
            per_modality: (B, n_modalities, hidden_dim) or None
        """
        B = features.size(0)

        if self.debug:
            present = mask.sum(dim=1).mean().item()
            print(f"[TRANSFORMER DEBUG] forward: B={B}, avg_present_mod={present:.2f}")

        # per-modality projection
        projected = []
        for i in range(self.n_modalities):
            projected.append(self.input_projs[i](features[:, i, :]))  # (B, hidden_dim)
        x = torch.stack(projected, dim=1)  # (B, n_mod, hidden_dim)

        # add modality type embeddings
        x = x + self.modality_embed

        # replace missing modalities with mask tokens
        mask_exp = mask.unsqueeze(-1)  # (B, n_mod, 1)
        mask_tok = self.mask_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, n_mod, hidden_dim)
        x = x * mask_exp + mask_tok * (1.0 - mask_exp)

        # prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        x = torch.cat([cls, x], dim=1)  # (B, 1+n_mod, hidden_dim)

        # positional encoding (applied AFTER CLS prepend, covers all positions)
        if self.use_positional:
            x = self.pos_drop(x + self.pos_embed[:, : 1 + self.n_modalities, :])

        # transformer
        x = self.transformer(x)
        x = self.output_norm(x)

        cls_out = x[:, 0, :]  # (B, hidden_dim) — CLS output
        mod_tokens = x[:, 1:, :]  # (B, n_mod, hidden_dim) — modality tokens

        # masked mean pooling: average only present modality tokens
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        mask_exp_pool = mask.unsqueeze(-1)  # (B, n_mod, 1)
        pooled = (mod_tokens * mask_exp_pool).sum(dim=1) / mask_sum  # (B, hidden_dim)

        # dual fusion: CLS + masked mean pool
        fused = self.fusion(torch.cat([cls_out, pooled], dim=-1))  # (B, hidden_dim)

        per_modality = mod_tokens if return_per_modality else None

        if self.debug:
            print(f"[TRANSFORMER DEBUG] output: fused={fused.shape}, "
                  f"norm={fused.norm(dim=-1).mean().item():.4f}")

        return fused, per_modality
