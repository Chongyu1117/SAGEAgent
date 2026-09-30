"""Calibrated uncertainty head.

On top of the frozen predictor, a small MLP estimates whether the current
prediction would change if the remaining modalities were acquired:

    u_t = sigmoid(f([e_t, a_t]) / T)

The training label is 1 when |r_t - r_full| > tau, where r_t is the risk on a
clinical-order prefix of the modalities and r_full the risk with all of them.
T is fitted on validation data after training (temperature scaling).
"""

from __future__ import annotations

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..clinical import ClinicalPathway
from ..data import Cohort
from ..metrics import auroc, expected_calibration_error
from .predictor import SurvivalPredictor


class UncertaintyHead(nn.Module):
    def __init__(self, embedding_dim: int, n_modalities: int, hidden_dim: int = 64):
        super().__init__()
        self.hparams = dict(embedding_dim=embedding_dim, n_modalities=n_modalities, hidden_dim=hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + n_modalities, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        self.register_buffer("temperature", torch.ones(1))

    def logits(self, embedding: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([embedding, mask], dim=-1)).squeeze(-1)

    def forward(self, embedding: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Calibrated probability that more modalities would change the prediction."""
        return torch.sigmoid(self.logits(embedding, mask) / self.temperature)


# =============================================================================
# Training data: clinical-order prefixes
# =============================================================================
@torch.no_grad()
def prefix_samples(predictor: SurvivalPredictor, cohort: Cohort, pathway: ClinicalPathway,
                   tau: float, min_modalities: int | None, device: str, batch_size: int = 512):
    """Embeddings, masks and labels for every clinical-order prefix of eligible patients.

    A patient is eligible if its available prefix covers at least
    ``min_modalities`` modalities (``None`` = all modalities, i.e. complete
    patients). The reference risk r_full uses every modality the patient has,
    so for complete patients the full prefix is included with label 0.
    """
    n_mod = len(pathway)
    need = n_mod if min_modalities is None else int(min_modalities)
    depths = np.array([pathway.available_depth(m) for m in cohort.mask])
    eligible = np.where(depths >= max(need, pathway.initial))[0]

    rows, masks = [], []
    for i in eligible:
        for depth in range(pathway.initial, depths[i] + 1):
            rows.append(i)
            masks.append(pathway.prefix_mask(depth))
    rows = np.array(rows, dtype=np.int64)
    masks = np.stack(masks) if masks else np.zeros((0, n_mod), np.float32)

    def run(features, mask):
        emb, risk = [], []
        for s in range(0, len(features), batch_size):
            e, r = predictor.encode(torch.from_numpy(features[s:s + batch_size]).to(device),
                                    torch.from_numpy(mask[s:s + batch_size]).to(device))
            emb.append(e.cpu().numpy())
            risk.append(r.cpu().numpy())
        return (np.concatenate(emb), np.concatenate(risk)) if emb else (np.zeros((0, 0)), np.zeros(0))

    _, full_risk = run(cohort.features[eligible], cohort.mask[eligible])
    reference = dict(zip(eligible.tolist(), full_risk))

    feats = cohort.features[rows] * masks[:, :, None]
    emb, risk = run(feats.astype(np.float32), masks)
    labels = np.array([abs(r - reference[i]) > tau for r, i in zip(risk, rows)], dtype=np.float32)
    return emb.astype(np.float32), masks, labels, masks.sum(axis=1).astype(int)


# =============================================================================
# Training
# =============================================================================
def fit_temperature(head: UncertaintyHead, emb: np.ndarray, masks: np.ndarray, labels: np.ndarray,
                    device: str) -> float:
    """Temperature scaling (Guo et al., 2017): minimize validation NLL over T."""
    with torch.no_grad():
        logits = head.logits(torch.from_numpy(emb).to(device), torch.from_numpy(masks).to(device))
    target = torch.from_numpy(labels).to(device)
    log_t = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / log_t.exp(), target)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_t.detach().exp().clamp(0.1, 10.0))
    head.temperature.fill_(temperature)
    return temperature


def train_uncertainty_head(predictor: SurvivalPredictor, train: Cohort, val: Cohort,
                           pathway: ClinicalPathway, cfg, device: str) -> tuple[UncertaintyHead, dict]:
    ucfg, tcfg = cfg.uncertainty, cfg.uncertainty.train
    tr = prefix_samples(predictor, train, pathway, ucfg.tau, ucfg.min_modalities, device)
    va = prefix_samples(predictor, val, pathway, ucfg.tau, ucfg.min_modalities, device)
    if len(tr[0]) == 0 or len(va[0]) == 0:
        raise ValueError("no eligible patients for the uncertainty head; check uncertainty.min_modalities")

    head = UncertaintyHead(predictor.embedding_dim, len(pathway), ucfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
    tensors = [torch.from_numpy(a).to(device) for a in tr[:3]]
    val_tensors = [torch.from_numpy(a).to(device) for a in va[:3]]

    best, best_loss, stale = copy.deepcopy(head.state_dict()), float("inf"), 0
    for _ in range(tcfg.epochs):
        head.train()
        for batch in torch.randperm(len(tensors[0]), device=device).split(tcfg.batch_size):
            loss = F.binary_cross_entropy_with_logits(head.logits(tensors[0][batch], tensors[1][batch]),
                                                      tensors[2][batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            val_loss = F.binary_cross_entropy_with_logits(head.logits(*val_tensors[:2]), val_tensors[2]).item()
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best, best_loss, stale = copy.deepcopy(head.state_dict()), val_loss, 0
        else:
            stale += 1
            if stale >= tcfg.patience:
                break

    head.load_state_dict(best)
    head.eval()
    temperature = fit_temperature(head, *va[:3], device)
    metrics = evaluate_uncertainty_head(head, *va, device)
    metrics.update(temperature=temperature, n_train=len(tr[0]), n_val=len(va[0]),
                   positive_rate=float(tr[2].mean()))
    return head, metrics


@torch.no_grad()
def evaluate_uncertainty_head(head: UncertaintyHead, emb, masks, labels, depths, device: str) -> dict:
    probs = head(torch.from_numpy(emb).to(device), torch.from_numpy(masks).to(device)).cpu().numpy()
    per_depth = {int(d): {"mean_u": float(probs[depths == d].mean()),
                          "positive_rate": float(labels[depths == d].mean())}
                 for d in np.unique(depths)}
    return {"auroc": auroc(probs, labels), "ece": expected_calibration_error(probs, labels),
            "per_depth": per_depth}


# =============================================================================
# Checkpoints
# =============================================================================
def save_uncertainty_head(head: UncertaintyHead, path: str, **metadata) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"hparams": head.hparams, "state_dict": head.state_dict(), "metadata": metadata}, path)


def load_uncertainty_head(path: str, device: str = "cpu") -> UncertaintyHead:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    head = UncertaintyHead(**ckpt["hparams"])
    head.load_state_dict(ckpt["state_dict"])
    return head.to(device).eval().requires_grad_(False)
