"""Step 1: train the multimodal survival predictor of every (outer, inner) pipeline.

    python train_predictor.py --config configs/glioma.yaml [--outer 1 2] [--inner 1] [--device cuda:0]
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from sageagent.config import base_parser, load_config, save_config, select_folds
from sageagent.data import Cohort
from sageagent.metrics import concordance_index
from sageagent.models import (alignment_loss, build_predictor, cox_partial_likelihood, reconstruction_loss,
                              save_predictor)
from sageagent.pipeline import Workspace, load_data
from sageagent.utils import fold_seed, get_logger, resolve_device, save_json, set_seed, update_json

log = get_logger()


def modality_dropout(mask: torch.Tensor, p: float) -> torch.Tensor:
    """Drop each present modality with probability p, keeping at least one per patient."""
    keep = mask * (torch.rand_like(mask) >= p).float()
    empty = keep.sum(dim=1) == 0
    if empty.any():
        rescue = torch.multinomial(mask[empty], 1)          # one of the originally present modalities
        keep[empty] = keep[empty].scatter(1, rescue, 1.0)
    return keep


def lr_lambda(step: int, warmup: int, total: int, floor: float) -> float:
    """Linear warmup, then cosine decay to `floor` (as a fraction of the base LR)."""
    if step < warmup:
        return 0.01 + 0.99 * step / max(warmup, 1)
    progress = min(1.0, (step - warmup) / max(total - warmup, 1))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def c_index_of(model, cohort: Cohort, device: str) -> float:
    model.eval()
    risk = model(torch.from_numpy(cohort.features).to(device), torch.from_numpy(cohort.mask).to(device)).risk
    return concordance_index(risk.cpu().numpy(), cohort.event, cohort.time)


def train_fold(cfg, train: Cohort, val: Cohort, device: str, seed: int):
    t = cfg.predictor.train
    set_seed(seed)
    model = build_predictor(cfg, train.modality_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    steps_per_epoch = math.ceil(len(train) / t.batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, t.warmup_epochs * steps_per_epoch, t.epochs * steps_per_epoch,
                                       t.min_lr / t.lr))
    features, mask = torch.from_numpy(train.features).to(device), torch.from_numpy(train.mask).to(device)
    event, time = torch.from_numpy(train.event).to(device), torch.from_numpy(train.time).to(device)

    best, best_c, stale, history = None, -1.0, 0, []
    for epoch in range(1, t.epochs + 1):
        model.train()
        losses = []
        for idx in torch.randperm(len(train), device=device).split(t.batch_size):
            kept = modality_dropout(mask[idx], t.modality_dropout)
            out = model(features[idx] * kept.unsqueeze(-1), kept)
            loss = cox_partial_likelihood(out.risk, event[idx], time[idx])
            if out.reconstruction is not None:
                loss = loss + t.recon_weight * reconstruction_loss(out.reconstruction, features[idx], mask[idx],
                                                                   train.modality_dims)
            loss = loss + t.align_weight * alignment_loss(out.modality_tokens, kept)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        val_c = c_index_of(model, val, device)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_c_index": val_c})
        if epoch >= t.min_epochs:
            if val_c > best_c:
                best, best_c, stale = (copy.deepcopy(model.state_dict()), epoch), val_c, 0
            else:
                stale += 1
                if stale >= t.patience:
                    break

    if best is None:                       # fewer epochs than min_epochs: keep the last model
        best, best_c = (copy.deepcopy(model.state_dict()), epoch), val_c
    model.load_state_dict(best[0])
    return model, {"best_epoch": best[1], "val_c_index": best_c, "history": history}


def main():
    parser = base_parser("Train the multimodal survival predictor (nested cross-validation).")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    device = resolve_device(args.device)
    ws = Workspace(cfg)
    pathway, cohort, splits = load_data(cfg)
    save_config(cfg, f"{ws.root}/predictors/config.yaml")

    for outer in select_folds(args.outer, splits.n_outer):
        test = cohort.subset(splits.test(outer))
        for inner in select_folds(args.inner, splits.n_inner):
            train, val = cohort.subset(splits.train(outer, inner)), cohort.subset(splits.val(outer, inner))
            log.info(f"[outer {outer} / inner {inner}] predictor: {len(train)} train, {len(val)} val, "
                     f"{len(test)} test patients")
            model, info = train_fold(cfg, train, val, device, fold_seed(cfg.experiment.seed, outer, inner))
            test_c = c_index_of(model, test, device)
            save_predictor(model, ws.predictor(outer, inner), best_epoch=info["best_epoch"],
                           val_c_index=info["val_c_index"], test_c_index=test_c)
            save_json(info["history"], f"{ws.fold_dir(outer, inner)}/predictor_history.json")
            update_json(f"{ws.root}/predictors/summary.json", {f"outer_{outer}/inner_{inner}": {
                "best_epoch": info["best_epoch"], "val_c_index": info["val_c_index"], "test_c_index": test_c}})
            log.info(f"[outer {outer} / inner {inner}] best epoch {info['best_epoch']}, "
                     f"val C-index {info['val_c_index']:.4f}, test C-index (all modalities) {test_c:.4f}")


if __name__ == "__main__":
    main()
