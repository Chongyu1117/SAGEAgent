"""
Survival Predictor — Transformer encoder + discrete-time hazard head
                     + Cox PH + NLL survival loss
                     + Reconstruction heads + Alignment loss.

Hazard head outputs N interval logits (not a single scalar).
Combined NLL + Cox loss with reconstruction and alignment.

Note: Uncertainty is handled by a separate post-hoc calibrated head
(see models/calibrated_uncertainty.py), NOT inside this predictor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

from .transformer import MultimodalTransformerEncoder


class SurvivalPredictor(nn.Module):
    """
    Multimodal survival predictor with missing-modality support.

    Outputs:
        - logits (B, n_intervals) discrete hazard logits for NLL loss
        - cox_risk (B,) scalar risk for Cox loss
        - embedding (B, hidden_dim) fused representation
        - reconstruction (optional) for missing-modality auxiliary loss
        - per_modality representations for alignment loss
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_modalities: int,
        n_encoder_layers: int,
        n_heads: int,
        predictor_hidden_dims: List[int],
        n_intervals: int = 20,
        encoder_dropout: float = 0.1,
        predictor_dropout: float = 0.3,
        use_positional: bool = True,
        use_reconstruction: bool = False,
        debug: bool = False,
        
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_modalities = n_modalities
        self.n_intervals = n_intervals
        self.use_reconstruction = use_reconstruction
        self.debug = debug

        # multimodal encoder
        self.encoder = MultimodalTransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_modalities=n_modalities,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            dropout=encoder_dropout,
            use_positional=use_positional,
            debug=debug,
        )

        # NLL head: outputs n_intervals logits for discrete survival curve
        # sigmoid(logits) = per-interval hazard probabilities
        nll_layers = []
        prev = hidden_dim
        for h in predictor_hidden_dims:
            nll_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(predictor_dropout)]
            prev = h
        nll_layers.append(nn.Linear(prev, n_intervals))
        self.hazard_head = nn.Sequential(*nll_layers)

        # Cox head: outputs scalar risk score (direct gradient path for Cox loss)
        cox_layers = []
        prev = hidden_dim
        for h in predictor_hidden_dims:
            cox_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(predictor_dropout)]
            prev = h
        cox_layers.append(nn.Linear(prev, 1))
        self.cox_head = nn.Sequential(*cox_layers)

        # reconstruction heads: from fused embedding → each modality's original features
        if use_reconstruction:
            self.recon_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(predictor_dropout),
                    nn.Linear(hidden_dim // 2, input_dim),
                )
                for _ in range(n_modalities)
            ])

        self._init_output_weights()

        if self.debug:
            n_params = sum(p.numel() for p in self.parameters())
            print(f"[SURVIVAL DEBUG] init: hidden={hidden_dim}, n_intervals={n_intervals}, "
                  f"reconstruction={use_reconstruction}, params={n_params:,}")

    def _init_output_weights(self):
        """Small init for output layers to start near zero."""
        for head in [self.hazard_head, self.cox_head]:
            last = [m for m in head if isinstance(m, nn.Linear)][-1]
            nn.init.xavier_uniform_(last.weight, gain=0.01)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, None, torch.Tensor,
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            features: (B, n_mod, input_dim)
            mask:     (B, n_mod)

        Returns:
            logits:       (B, n_intervals)  discrete hazard logits for NLL loss
            cox_risk:     (B,)  scalar risk for Cox loss (direct gradient path)
            _unused:      None (kept for backward-compatible unpacking)
            embedding:    (B, hidden_dim)  encoder output (for FAISS)
            recon:        (B, n_mod, input_dim) or None
            per_modality: (B, n_mod, hidden_dim) or None
        """
        embedding, per_modality = self.encoder(features, mask, return_per_modality=True)

        logits = self.hazard_head(embedding)  # (B, n_intervals)
        cox_risk = self.cox_head(embedding).squeeze(-1)  # (B,)

        recon = None
        if self.use_reconstruction:
            recon = torch.stack(
                [head(embedding) for head in self.recon_heads], dim=1
            )  # (B, n_mod, input_dim)

        if self.debug:
            print(f"[SURVIVAL DEBUG] forward: "
                  f"logits=[{logits.min().item():.3f}, {logits.max().item():.3f}], "
                  f"cox_risk=[{cox_risk.min().item():.3f}, {cox_risk.max().item():.3f}]")

        return logits, cox_risk, None, embedding, recon, per_modality

    def get_risk_score(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Get risk scores (B,). Higher = worse prognosis. Uses Cox head directly."""
        _, cox_risk, _, _, _, _ = self.forward(features, mask)
        return cox_risk

    def get_embedding(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Get encoder embedding (B, hidden_dim). Used for FAISS index."""
        _, _, _, emb, _, _ = self.forward(features, mask)
        return emb

    def get_survival_curve(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Get survival probabilities at each interval (B, n_intervals).

        S(t_k) = prod_{j=0}^{k-1} (1 - h_j)
        Requires external `cuts` array to map interval indices to actual times.
        """
        logits, _, _, _, _, _ = self.forward(features, mask)
        hazards = torch.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)
        return surv


# ═══════════════════════════════════════════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════════════════════════════════════════

class NLLSurvivalLoss(nn.Module):
    """
    Negative Log-Likelihood for discrete-time logistic hazard model.

    
    Numerically stable via softplus:
        loss_i = sum_{j=0}^{bin_i} softplus(phi_j) - event_i * phi_{bin_i}
    """

    def __init__(self, debug: bool = False):
        super().__init__()
        self.debug = debug

    def forward(
        self,
        logits: torch.Tensor,
        time_bins: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:    (B, n_intervals) raw hazard logits (before sigmoid)
            time_bins: (B,) long tensor — bin index for each patient
            events:    (B,) float — 1=event, 0=censored

        Returns:
            loss: scalar
        """
        B, N = logits.shape

        # softplus(phi) = -log(1 - sigmoid(phi)) — "survived this interval" cost
        sp = F.softplus(logits)  # (B, N)

        # mask: intervals 0..bin_i are active for patient i
        idx = torch.arange(N, device=logits.device).unsqueeze(0)  # (1, N)
        active = (idx <= time_bins.unsqueeze(1)).float()  # (B, N)

        # cumulative survival cost: sum softplus over active intervals
        cumulative = (sp * active).sum(dim=1)  # (B,)

        # event cost: -event_i * phi_{bin_i}
        event_phi = logits[torch.arange(B, device=logits.device), time_bins]  # (B,)

        # NLL = cumulative - event * phi
        nll = cumulative - events * event_phi  # (B,)
        loss = nll.mean()

        if self.debug:
            print(f"[NLL_LOSS DEBUG] loss={loss.item():.4f}, n_events={int(events.sum().item())}")

        return loss


class CoxPHLoss(nn.Module):
    """
    Negative Cox Partial Log-Likelihood (Breslow approximation).

    Standard survival prediction loss. Numerically stabilised via logsumexp.
    """

    def __init__(self, debug: bool = False):
        super().__init__()
        self.debug = debug

    def forward(
        self,
        risk_scores: torch.Tensor,
        events: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            risk_scores: (B,)  scalar risk per patient (higher = worse)
            events: (B,)  1=event, 0=censored
            times: (B,)   survival times

        Returns:
            loss: scalar
        """
        order = torch.argsort(times, descending=True)
        risk_sorted = risk_scores[order]
        events_sorted = events[order]

        max_risk = risk_sorted.max().detach()
        exp_risk = torch.exp(risk_sorted - max_risk)
        cumsum_exp = torch.cumsum(exp_risk, dim=0)
        log_cumsum = torch.log(cumsum_exp + 1e-8) + max_risk

        partial_ll = risk_sorted - log_cumsum
        n_events = events_sorted.sum() + 1e-8
        cox_loss = -torch.sum(partial_ll * events_sorted) / n_events

        if self.debug:
            print(f"[COX_LOSS DEBUG] n_events={int(n_events.item())}, cox={cox_loss.item():.4f}")

        return cox_loss


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for missing-modality robustness.

    From fused embedding, reconstruct each modality's original features.
    Loss computed ONLY on available (present) modalities.
    
    """

    def __init__(self, debug: bool = False):
        super().__init__()
        self.debug = debug

    def forward(
        self,
        recon: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        per_mod_mse = ((recon - features) ** 2).mean(dim=-1)  # (B, n_mod)
        masked_mse = per_mod_mse * mask
        n_present = mask.sum() + 1e-8
        loss = masked_mse.sum() / n_present

        if self.debug:
            print(f"[RECON_LOSS DEBUG] loss={loss.item():.4f}, n_present={int(n_present.item())}")
        return loss


class AlignmentLoss(nn.Module):
    """
    Cross-modal alignment loss.

    Encourages per-modality representations to be close in latent space.
    Only computed between pairs of PRESENT modalities.
    """

    def __init__(self, debug: bool = False):
        super().__init__()
        self.debug = debug

    def forward(
        self,
        per_modality: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, n_mod, D = per_modality.shape
        total_loss = torch.tensor(0.0, device=per_modality.device)
        n_valid_pairs = 0

        for i in range(n_mod):
            for j in range(i + 1, n_mod):
                both_present = mask[:, i] * mask[:, j]
                n_pairs = both_present.sum()
                if n_pairs > 0:
                    diff = per_modality[:, i, :] - per_modality[:, j, :]
                    pair_dist = (diff ** 2).mean(dim=-1)
                    total_loss = total_loss + (pair_dist * both_present).sum() / n_pairs
                    n_valid_pairs += 1

        if n_valid_pairs > 0:
            total_loss = total_loss / n_valid_pairs

        if self.debug:
            print(f"[ALIGN_LOSS DEBUG] loss={total_loss.item():.4f}, n_valid_pairs={n_valid_pairs}")
        return total_loss
