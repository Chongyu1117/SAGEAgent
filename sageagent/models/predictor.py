"""Multimodal survival predictor: Transformer encoder + Cox proportional hazards head."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from .encoder import MultimodalTransformerEncoder


@dataclass
class PredictorOutput:
    risk: torch.Tensor                   # (B,) Cox risk score, higher = worse prognosis
    embedding: torch.Tensor              # (B, d) fused patient embedding e_t
    modality_tokens: torch.Tensor        # (B, M, d)
    reconstruction: torch.Tensor | None  # (B, M, D) or None


def _mlp_head(in_dim: int, hidden_dims: Sequence[int], out_dim: int, dropout: float) -> nn.Sequential:
    layers, prev = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class SurvivalPredictor(nn.Module):
    """Maps the available modalities to a risk score r_t and an embedding e_t.

    Missing modalities are handled by the encoder's mask tokens. Optional
    reconstruction heads predict every modality's features from the fused
    embedding (an auxiliary training loss for missing-modality robustness).
    """

    def __init__(self, input_dims: Sequence[int], hidden_dim: int = 128, n_layers: int = 2,
                 n_heads: int = 4, ffn_mult: int = 4, dropout: float = 0.1,
                 head_hidden_dims: Sequence[int] = (64, 32), head_dropout: float = 0.3,
                 reconstruction: bool = True):
        super().__init__()
        self.hparams = dict(input_dims=list(input_dims), hidden_dim=hidden_dim, n_layers=n_layers,
                            n_heads=n_heads, ffn_mult=ffn_mult, dropout=dropout,
                            head_hidden_dims=list(head_hidden_dims), head_dropout=head_dropout,
                            reconstruction=reconstruction)
        self.encoder = MultimodalTransformerEncoder(input_dims, hidden_dim, n_layers, n_heads, ffn_mult, dropout)
        self.cox_head = _mlp_head(hidden_dim, head_hidden_dims, 1, head_dropout)
        nn.init.xavier_uniform_(self.cox_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.cox_head[-1].bias)
        self.recon_heads = None
        if reconstruction:
            self.recon_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(head_dropout),
                              nn.Linear(hidden_dim // 2, max(input_dims)))
                for _ in input_dims
            ])

    @property
    def embedding_dim(self) -> int:
        return self.hparams["hidden_dim"]

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> PredictorOutput:
        embedding, tokens = self.encoder(features, mask)
        risk = self.cox_head(embedding).squeeze(-1)
        recon = None
        if self.recon_heads is not None:
            recon = torch.stack([head(embedding) for head in self.recon_heads], dim=1)
        return PredictorOutput(risk, embedding, tokens, recon)

    @torch.no_grad()
    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper: (embedding, risk)."""
        out = self.forward(features, mask)
        return out.embedding, out.risk


# =============================================================================
# Losses
# =============================================================================
def cox_partial_likelihood(risk: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood (Breslow), averaged over events."""
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    log_risk_set = torch.logcumsumexp(risk, dim=0)
    return -((risk - log_risk_set) * event).sum() / event.sum().clamp(min=1e-8)


def reconstruction_loss(recon: torch.Tensor, features: torch.Tensor, mask: torch.Tensor,
                        dims: Sequence[int]) -> torch.Tensor:
    """MSE between reconstructed and original features, over available modalities only."""
    per_modality = torch.stack(
        [((recon[:, i, :d] - features[:, i, :d]) ** 2).mean(dim=-1) for i, d in enumerate(dims)], dim=1)
    return (per_modality * mask).sum() / mask.sum().clamp(min=1e-8)


def alignment_loss(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared distance between the tokens of every pair of present modalities."""
    n_mod = tokens.shape[1]
    losses = []
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            both = mask[:, i] * mask[:, j]
            if both.sum() > 0:
                dist = ((tokens[:, i] - tokens[:, j]) ** 2).mean(dim=-1)
                losses.append((dist * both).sum() / both.sum())
    return torch.stack(losses).mean() if losses else tokens.new_zeros(())


# =============================================================================
# Construction and checkpoints
# =============================================================================
def build_predictor(cfg, input_dims: Sequence[int]) -> SurvivalPredictor:
    p = cfg.predictor
    return SurvivalPredictor(input_dims, p.hidden_dim, p.n_layers, p.n_heads, p.ffn_mult, p.dropout,
                             p.head_hidden_dims, p.head_dropout, p.reconstruction)


def save_predictor(model: SurvivalPredictor, path: str, **metadata) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"hparams": model.hparams, "state_dict": model.state_dict(), "metadata": metadata}, path)


def load_predictor(path: str, device: str = "cpu") -> SurvivalPredictor:
    """Load a frozen predictor (eval mode, no gradients)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SurvivalPredictor(**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval().requires_grad_(False)
    return model
