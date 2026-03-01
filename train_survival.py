#!/usr/bin/env python3
"""
Phase 0 — Pre-train Survival Predictor.

Trains the multimodal Transformer + discrete-time hazard survival model.
Combined loss: alpha*NLL + (1-alpha)*Cox + recon + align.
The trained model is frozen and reused by the RL agent in later phases.

Usage:
    python train_survival.py --debug
    python train_survival.py --config configs/default_config.yaml --gpu 0
    python train_survival.py --test_fold 1 --val_fold 2 --epochs 60
    python train_survival.py --all_folds --epochs 60

    # Nested 5x5 CV (recommended):
    python train_survival.py --nested --gpu 0
    python train_survival.py --nested --outer_fold 1 --gpu 0
    python train_survival.py --nested --exp_name nested5x5_run1 --gpu 0
"""

import os
import sys
import argparse
import yaml
import time
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Tuple

from models import (
    SurvivalPredictor, NLLSurvivalLoss, CoxPHLoss,
    ReconstructionLoss, AlignmentLoss,
)
from utils.data_loader import load_pkl_data, create_dataloaders
from utils.metrics import compute_c_index


# ── helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0: Pre-train Survival Predictor")
    p.add_argument("--config", type=str, default="configs/default_config.yaml")

    # overrides (all optional — fall back to config file)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--test_fold", type=int, default=None)
    p.add_argument("--val_fold", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--mask_augment_prob", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--nll_alpha", type=float, default=None)
    p.add_argument("--recon_weight", type=float, default=None)
    p.add_argument("--align_weight", type=float, default=None)
    p.add_argument("--min_epochs", type=int, default=None,
                   help="Min epochs before early stopping starts (default: from config or 20)")

    p.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--exp_name", type=str, default=None)
    p.add_argument("--all_folds", action="store_true",
                   help="Run all 15 folds and report mean ± std C-Index")

    # nested 5x5 CV mode
    p.add_argument("--nested", action="store_true",
                   help="Use nested 5x5 CV splits from splits/outer_N/")
    p.add_argument("--outer_fold", type=int, default=None,
                   help="Outer fold (1-5). None = run all 5 outer folds")
    p.add_argument("--inner_fold", type=int, default=None,
                   help="Inner fold (1-5). None = run all 5 inner folds (true 5x5 CV)")
    p.add_argument("--n_inner", type=int, default=5,
                   help="Number of inner folds")
    p.add_argument("--splits_dir", type=str, default=None,
                   help="Path to nested splits directory (default: <data_root>/splits)")
    p.add_argument("--save_every", type=int, default=0,
                   help="Save checkpoint every N epochs (0 = off)")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args) -> dict:
    """Override config values with CLI args (if provided)."""
    if args.data_root:
        cfg["data"]["data_root"] = args.data_root
    if args.test_fold is not None:
        cfg["data"]["test_fold"] = args.test_fold
    if args.val_fold is not None:
        cfg["data"]["val_fold"] = args.val_fold
    if args.epochs is not None:
        cfg["training"]["survival"]["n_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["survival"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["survival"]["lr"] = args.lr
    if args.patience is not None:
        cfg["training"]["survival"]["patience"] = args.patience
    if args.mask_augment_prob is not None:
        cfg["training"]["survival"]["mask_augment_prob"] = args.mask_augment_prob
    if args.weight_decay is not None:
        cfg["training"]["survival"]["weight_decay"] = args.weight_decay
    if args.nll_alpha is not None:
        cfg["training"]["survival"]["nll_alpha"] = args.nll_alpha
    if args.recon_weight is not None:
        cfg["training"]["survival"]["recon_weight"] = args.recon_weight
    if args.align_weight is not None:
        cfg["training"]["survival"]["align_weight"] = args.align_weight
    if args.min_epochs is not None:
        cfg["training"]["survival"]["min_epochs"] = args.min_epochs
    if args.hidden_dim is not None:
        cfg["model"]["encoder"]["hidden_dim"] = args.hidden_dim
    if args.n_layers is not None:
        cfg["model"]["encoder"]["n_layers"] = args.n_layers
    if args.n_heads is not None:
        cfg["model"]["encoder"]["n_heads"] = args.n_heads
    if args.dropout is not None:
        cfg["model"]["encoder"]["dropout"] = args.dropout
        cfg["model"]["survival"]["dropout"] = args.dropout
    cfg["logging"]["debug"] = cfg["logging"]["debug"] or args.debug
    return cfg


# ── time discretization ─────────────────────────────────────────────────────

def compute_time_cuts(train_times: np.ndarray, train_events: np.ndarray,
                      n_intervals: int = 20) -> np.ndarray:
    """Compute quantile-based cut points from training event times.

    Returns:
        cuts: (n_intervals + 1,) array of strictly increasing cut points
    """
    event_times = train_times[train_events > 0.5]
    percentiles = np.linspace(0, 100, n_intervals + 1)
    cuts = np.percentile(event_times, percentiles)
    # ensure strictly increasing (handle ties in event times)
    for i in range(1, len(cuts)):
        if cuts[i] <= cuts[i - 1]:
            cuts[i] = cuts[i - 1] + 1e-5
    return cuts


def times_to_bins(times: torch.Tensor, cuts_inner: torch.Tensor,
                  n_intervals: int) -> torch.Tensor:
    """Map continuous survival times to discrete bin indices.

    Args:
        times: (B,) continuous survival times
        cuts_inner: (n_intervals - 1,) inner bin boundaries (cuts[1:-1])
        n_intervals: number of bins

    Returns:
        bins: (B,) long tensor in [0, n_intervals-1]
    """
    bins = torch.bucketize(times, cuts_inner)
    return bins.clamp(max=n_intervals - 1)


# ── training / evaluation ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, nll_criterion, cox_criterion, recon_criterion, align_criterion,
             loader, device, cuts_inner, n_intervals, nll_alpha=0.7,
             use_recon=False, recon_weight=1.0, align_weight=0.1, debug=False):
    """Run evaluation pass. Returns dict with loss, c_index, etc."""
    model.eval()
    all_risks, all_events, all_times = [], [], []
    total_loss = 0.0
    total_nll = 0.0
    total_cox = 0.0
    total_recon = 0.0
    total_align = 0.0
    n_batches = 0

    for batch in loader:
        feat = batch["features"].to(device)
        mask = batch["masks"].to(device)
        e = batch["events"].to(device)
        t = batch["times"].to(device)

        # Original features/masks for reconstruction (val/test have no dropout)
        orig_feat = batch["orig_features"].to(device)
        orig_mask = batch["orig_masks"].to(device)

        logits, cox_risk, unc, emb, recon, per_mod = model(feat, mask)

        # NLL loss (discrete survival)
        t_bins = times_to_bins(t, cuts_inner, n_intervals)
        nll_loss = nll_criterion(logits, t_bins, e)

        # Cox loss (direct scalar risk from cox_head)
        cox_loss = cox_criterion(cox_risk, e, t)

        # combined survival loss
        loss = nll_alpha * nll_loss + (1 - nll_alpha) * cox_loss

        # auxiliary losses
        recon_loss_val = 0.0
        align_loss_val = 0.0

        if use_recon and recon is not None:
            recon_loss = recon_criterion(recon, orig_feat, orig_mask)
            loss = loss + recon_weight * recon_loss
            recon_loss_val = recon_loss.item()

        if per_mod is not None:
            align_loss = align_criterion(per_mod, mask)
            loss = loss + align_weight * align_loss
            align_loss_val = align_loss.item()

        total_loss += loss.item()
        total_nll += nll_loss.item()
        total_cox += cox_loss.item()
        total_recon += recon_loss_val
        total_align += align_loss_val
        n_batches += 1
        all_risks.append(cox_risk.cpu().numpy())
        all_events.append(e.cpu().numpy())
        all_times.append(t.cpu().numpy())

    all_risks = np.concatenate(all_risks)
    all_events = np.concatenate(all_events)
    all_times = np.concatenate(all_times)
    c_idx = compute_c_index(all_risks, all_events, all_times)
    n = max(n_batches, 1)

    return {
        "loss": total_loss / n,
        "nll_loss": total_nll / n,
        "cox_loss": total_cox / n,
        "recon_loss": total_recon / n,
        "align_loss": total_align / n,
        "c_index": c_idx,
        "n_patients": len(all_risks),
    }


def train_one_epoch(model, nll_criterion, cox_criterion, recon_criterion,
                    align_criterion, optimizer, scheduler, loader, device,
                    cuts_inner, n_intervals, nll_alpha=0.7,
                    use_recon=False, recon_weight=1.0, align_weight=0.1,
                    max_grad_norm=1.0, debug=False):
    """Train for one epoch. Returns dict with loss breakdown."""
    model.train()
    total_loss = 0.0
    total_nll = 0.0
    total_cox = 0.0
    total_recon = 0.0
    total_align = 0.0
    n_batches = 0

    for batch in loader:
        feat = batch["features"].to(device)
        mask = batch["masks"].to(device)
        e = batch["events"].to(device)
        t = batch["times"].to(device)

        # Original features/masks for reconstruction (before modality dropout)
        orig_feat = batch["orig_features"].to(device)
        orig_mask = batch["orig_masks"].to(device)

        logits, cox_risk, unc, emb, recon, per_mod = model(feat, mask)

        # NLL loss (discrete survival)
        t_bins = times_to_bins(t, cuts_inner, n_intervals)
        nll_loss = nll_criterion(logits, t_bins, e)

        # Cox loss (direct scalar risk from cox_head)
        cox_loss = cox_criterion(cox_risk, e, t)

        # combined survival loss
        loss = nll_alpha * nll_loss + (1 - nll_alpha) * cox_loss

        # auxiliary losses
        recon_loss_val = 0.0
        align_loss_val = 0.0

        if use_recon and recon is not None:
            # Reconstruct ALL available modalities (including dropped ones)
            # Uses original features/masks as targets
            recon_loss = recon_criterion(recon, orig_feat, orig_mask)
            loss = loss + recon_weight * recon_loss
            recon_loss_val = recon_loss.item()

        if per_mod is not None:
            align_loss = align_criterion(per_mod, mask)
            loss = loss + align_weight * align_loss
            align_loss_val = align_loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_nll += nll_loss.item()
        total_cox += cox_loss.item()
        total_recon += recon_loss_val
        total_align += align_loss_val
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "nll_loss": total_nll / n,
        "cox_loss": total_cox / n,
        "recon_loss": total_recon / n,
        "align_loss": total_align / n,
    }


def build_scheduler(optimizer, train_cfg, n_batches_per_epoch):
    """Build cosine annealing scheduler with linear warmup."""
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

    n_epochs = train_cfg["n_epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    min_lr = train_cfg.get("min_lr", 1e-6)

    warmup_steps = warmup_epochs * n_batches_per_epoch
    total_steps = n_epochs * n_batches_per_epoch
    cosine_steps = total_steps - warmup_steps

    if warmup_steps == 0:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)

    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(cosine_steps, 1), eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    return scheduler


# ── single fold training ─────────────────────────────────────────────────────

def train_single_fold(cfg, args, test_fold, val_fold, device, pkl_data, save_dir,
                      save_every=0):
    """Train on a single fold. Returns test C-Index."""
    debug = cfg["logging"]["debug"]
    print_every = cfg["logging"]["print_every"]

    seed = cfg["training"]["seed"]
    set_seed(seed)

    # ── data ──────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    modality_keys = data_cfg["modality_keys"]
    train_cfg = cfg["training"]["survival"]

    train_loader, val_loader, test_loader = create_dataloaders(
        pkl_data=pkl_data,
        test_fold=test_fold,
        val_fold=val_fold,
        modality_keys=modality_keys,
        batch_size=train_cfg["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        mask_augment_prob=train_cfg["mask_augment_prob"],
        debug=debug,
    )
    print(f"[INFO] Fold {test_fold}: train={len(train_loader.dataset)}, "
          f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    # ── time discretization ───────────────────────────────────────────────
    n_intervals = train_cfg.get("n_intervals", 20)

    # extract training times/events for computing cuts
    train_ds = train_loader.dataset
    train_times_np = train_ds.times.numpy()
    train_events_np = train_ds.events.numpy()
    cuts = compute_time_cuts(train_times_np, train_events_np, n_intervals)
    # inner boundaries for torch.bucketize (cuts[1:-1])
    cuts_inner = torch.tensor(cuts[1:-1], dtype=torch.float32, device=device)
    print(f"[INFO] Time discretization: {n_intervals} intervals, "
          f"range=[{cuts[0]:.0f}, {cuts[-1]:.0f}]")

    # ── model ─────────────────────────────────────────────────────────────
    enc_cfg = cfg["model"]["encoder"]
    surv_cfg = cfg["model"]["survival"]
    use_recon = surv_cfg.get("use_reconstruction", False)

    model = SurvivalPredictor(
        input_dim=data_cfg["feature_dim"],
        hidden_dim=enc_cfg["hidden_dim"],
        n_modalities=data_cfg["n_modalities"],
        n_encoder_layers=enc_cfg["n_layers"],
        n_heads=enc_cfg["n_heads"],
        predictor_hidden_dims=surv_cfg["hidden_dims"],
        n_intervals=n_intervals,
        encoder_dropout=enc_cfg["dropout"],
        predictor_dropout=surv_cfg["dropout"],
        use_positional=enc_cfg["use_positional"],
        use_reconstruction=use_recon,
        debug=debug,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model params: {n_params:,}")

    # ── optimiser + scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    n_batches_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, train_cfg, n_batches_per_epoch)

    nll_criterion = NLLSurvivalLoss(debug=debug)
    cox_criterion = CoxPHLoss(debug=debug)
    recon_criterion = ReconstructionLoss(debug=debug)
    align_criterion = AlignmentLoss(debug=debug)

    nll_alpha = train_cfg.get("nll_alpha", 0.7)
    recon_weight = train_cfg.get("recon_weight", 1.0)
    align_weight = train_cfg.get("align_weight", 0.1)

    # ── save dir ──────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Checkpoints → {save_dir}")

    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── training loop ─────────────────────────────────────────────────────
    n_epochs = train_cfg["n_epochs"]
    patience = train_cfg["patience"]
    min_epochs = train_cfg.get("min_epochs", 20)
    best_val_c_index = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n[INFO] Training for up to {n_epochs} epochs (patience={patience}, min_epochs={min_epochs})")
    print(f"[INFO] Loss: {nll_alpha:.1f}*NLL + {1-nll_alpha:.1f}*Cox "
          f"+ {recon_weight}*Recon + {align_weight}*Align")
    print("-" * 70)

    for epoch in range(1, n_epochs + 1):
        t_start = time.time()

        train_stats = train_one_epoch(
            model, nll_criterion, cox_criterion, recon_criterion, align_criterion,
            optimizer, scheduler, train_loader, device,
            cuts_inner=cuts_inner, n_intervals=n_intervals, nll_alpha=nll_alpha,
            use_recon=use_recon, recon_weight=recon_weight, align_weight=align_weight,
            debug=debug,
        )
        val_stats = evaluate(
            model, nll_criterion, cox_criterion, recon_criterion, align_criterion,
            val_loader, device,
            cuts_inner=cuts_inner, n_intervals=n_intervals, nll_alpha=nll_alpha,
            use_recon=use_recon, recon_weight=recon_weight, align_weight=align_weight,
            debug=debug,
        )

        elapsed = time.time() - t_start
        lr_now = optimizer.param_groups[0]['lr']

        if epoch % print_every == 0 or epoch == 1:
            print(f"[Epoch {epoch:3d}/{n_epochs}]  "
                  f"train={train_stats['loss']:.4f} "
                  f"(nll={train_stats['nll_loss']:.4f} cox={train_stats['cox_loss']:.4f} "
                  f"recon={train_stats['recon_loss']:.4f} align={train_stats['align_loss']:.4f})  "
                  f"val_c={val_stats['c_index']:.4f}  val_loss={val_stats['loss']:.4f}  "
                  f"lr={lr_now:.6f}  ({elapsed:.1f}s)")

        # model selection: only start after min_epochs
        can_save = (epoch >= min_epochs)

        if can_save and val_stats["c_index"] > best_val_c_index:
            best_val_c_index = val_stats["c_index"]
            best_val_loss = val_stats["loss"]
            patience_counter = 0
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_c_index": best_val_c_index,
                "val_loss": val_stats["loss"],
                "cuts": cuts,
                "n_intervals": n_intervals,
                "config": cfg,
            }, best_path)
            print(f"  ★ New best val c={best_val_c_index:.4f} → saved")
        elif can_save:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[INFO] Early stopping at epoch {epoch} (patience={patience})")
                break

        # periodic checkpoint saving
        if save_every > 0 and epoch % save_every == 0:
            ep_path = os.path.join(save_dir, f"epoch_{epoch}_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_c_index": val_stats["c_index"],
                "val_loss": val_stats["loss"],
                "cuts": cuts,
                "n_intervals": n_intervals,
                "config": cfg,
            }, ep_path)

    # ── final test evaluation ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  Final Evaluation — Fold {test_fold} Test Set")
    print("=" * 70)

    best_path = os.path.join(save_dir, "best_model.pt")
    if not os.path.exists(best_path):
        # no best saved (training ended before min_epochs) — save current model
        print(f"[WARN] No best model saved (min_epochs={min_epochs}), using final model")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_c_index": val_stats["c_index"],
            "val_loss": val_stats["loss"],
            "cuts": cuts,
            "n_intervals": n_intervals,
            "config": cfg,
        }, best_path)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] Loaded best model from epoch {ckpt['epoch']} "
          f"(val_c={ckpt['val_c_index']:.4f})")

    test_stats = evaluate(
        model, nll_criterion, cox_criterion, recon_criterion, align_criterion,
        test_loader, device,
        cuts_inner=cuts_inner, n_intervals=n_intervals, nll_alpha=nll_alpha,
        use_recon=use_recon, recon_weight=recon_weight, align_weight=align_weight,
        debug=debug,
    )
    print(f"[RESULT] Test loss    = {test_stats['loss']:.4f} "
          f"(nll={test_stats['nll_loss']:.4f} cox={test_stats['cox_loss']:.4f})")
    print(f"[RESULT] Test c_index = {test_stats['c_index']:.4f}")
    print(f"[RESULT] Test patients = {test_stats['n_patients']}")

    # save final
    final_path = os.path.join(save_dir, "final_model.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "test_c_index": test_stats["c_index"],
        "cuts": cuts,
        "n_intervals": n_intervals,
        "config": cfg,
    }, final_path)
    print(f"[INFO] Final model → {final_path}")

    return test_stats["c_index"]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # resolve config path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
    cfg = load_config(config_path)
    cfg = apply_overrides(cfg, args)

    debug = cfg["logging"]["debug"]

    # ── banner ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  Phase 0: Survival Predictor Pre-training")
    print("=" * 70)

    # ── seed ──────────────────────────────────────────────────────────────
    seed = cfg["training"]["seed"]
    set_seed(seed)
    print(f"[INFO] seed={seed}")

    # ── device ────────────────────────────────────────────────────────────
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"[INFO] device=cuda:{args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
    else:
        device = torch.device("cpu")
        print("[INFO] device=cpu")

    # ── data ──────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    n_folds = data_cfg.get("n_folds", 15)

    if args.nested:
        # nested mode loads per-outer-fold pkls inside the loop
        pkl_data = None  # not used
    else:
        pkl_path = os.path.join(data_cfg["data_root"], data_cfg["features_file"])
        print(f"\n[INFO] Loading data from: {pkl_path}")
        t0 = time.time()
        pkl_data = load_pkl_data(pkl_path, debug=debug)
        print(f"[INFO] Pkl loaded in {time.time() - t0:.1f}s")

    if args.nested:
        # ── nested 5x5 CV mode ───────────────────────────────────────────
        data_root = data_cfg["data_root"]
        splits_dir = args.splits_dir or os.path.join(data_root, "splits")
        n_inner = args.n_inner
        inner_folds = [args.inner_fold] if args.inner_fold else list(range(1, n_inner + 1))
        outer_folds = [args.outer_fold] if args.outer_fold else list(range(1, 6))

        print(f"\n[INFO] Nested 5x{n_inner} CV mode")
        print(f"[INFO] Splits dir: {splits_dir}")
        print(f"[INFO] Outer folds: {outer_folds}, Inner folds: {inner_folds}")
        print("=" * 70)

        exp_name = args.exp_name or f"nested5x5_{datetime.now():%Y%m%d_%H%M%S}"
        base_save_dir = args.save_dir or os.path.join(
            data_root, cfg["logging"]["checkpoint_dir"], exp_name)

        # Collect results: outer_results[outer] = list of inner C-indices
        outer_results = {}
        for outer in outer_folds:
            pkl_path = os.path.join(splits_dir, f"outer_{outer}", "predictor.pkl")
            print(f"\n{'='*70}")
            print(f"  OUTER FOLD {outer}/5")
            print(f"{'='*70}")
            print(f"[INFO] Loading: {pkl_path}")
            t0 = time.time()
            pkl_data = load_pkl_data(pkl_path, debug=debug)
            print(f"[INFO] Pkl loaded in {time.time() - t0:.1f}s")

            inner_c_indices = []
            for inner in inner_folds:
                print(f"\n  --- Inner fold {inner}/{n_inner} ---")
                fold_save_dir = os.path.join(
                    base_save_dir, f"outer_{outer}", f"inner_{inner}")

                # cv_splits[inner]['train'] = inner training patients
                # cv_splits[inner]['val']   = inner validation patients
                # cv_splits[inner]['test']  = outer test patients (same across inner folds)
                fold_c = train_single_fold(
                    cfg, args, inner, inner, device, pkl_data, fold_save_dir,
                    save_every=args.save_every)
                inner_c_indices.append(fold_c)
                print(f"  [O{outer}/I{inner}] Test C-Index = {fold_c:.4f}")

            outer_results[outer] = inner_c_indices
            inner_mean = np.mean(inner_c_indices)
            print(f"\n  [OUTER {outer}] Mean across {len(inner_folds)} inner folds: "
                  f"{inner_mean:.4f}")

            del pkl_data

        # ── summary ───────────────────────────────────────────────────────
        # Per outer fold: average C-index across inner folds (training summary)
        outer_means = []
        for outer in outer_folds:
            outer_means.append(np.mean(outer_results[outer]))
        outer_means = np.array(outer_means)
        mean_c = outer_means.mean()
        std_c = outer_means.std()

        print("\n" + "=" * 70)
        print(f"  NESTED 5x{n_inner} CV SUMMARY")
        print("=" * 70)
        for outer in outer_folds:
            inner_cs = outer_results[outer]
            print(f"  Outer {outer}: inner C-indices = "
                  f"{[f'{c:.4f}' for c in inner_cs]} → mean = {np.mean(inner_cs):.4f}")
        print("-" * 40)
        print(f"  Mean C-Index = {mean_c:.4f} +/- {std_c:.4f}")
        print(f"  (MMD two-stage baseline: 0.7857)")
        print("=" * 70)

        # save summary
        summary_path = os.path.join(base_save_dir, "summary.yaml")
        summary = {
            "strategy": f"nested_5x{n_inner}_cv",
            "inner_folds": inner_folds,
            "outer_folds": outer_folds,
            "results": {
                f"outer_{o}": {
                    f"inner_{inner_folds[i]}": float(c)
                    for i, c in enumerate(outer_results[o])
                }
                for o in outer_folds
            },
            "outer_means": {f"outer_{o}": float(np.mean(outer_results[o]))
                            for o in outer_folds},
            "mean_c_index": float(mean_c),
            "std_c_index": float(std_c),
            "timestamp": datetime.now().isoformat(),
        }
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False)
        print(f"[INFO] Summary → {summary_path}")

    elif args.all_folds:
        # ── run all 15 folds ────────────────────────────────────
        print(f"\n[INFO] Running ALL {n_folds} folds")
        print("=" * 70)

        exp_name = args.exp_name or f"survival_allfolds_{datetime.now():%Y%m%d_%H%M%S}"
        base_save_dir = args.save_dir or os.path.join(
            data_cfg["data_root"], cfg["logging"]["checkpoint_dir"], exp_name)

        fold_c_indices = []
        for fold_idx in range(1, n_folds + 1):
            print(f"\n{'='*70}")
            print(f"  FOLD {fold_idx}/{n_folds}")
            print(f"{'='*70}")

            # val fold = next fold (wrapping around)
            val_fold = (fold_idx % n_folds) + 1

            fold_save_dir = os.path.join(base_save_dir, f"fold_{fold_idx}")
            fold_c = train_single_fold(cfg, args, fold_idx, val_fold, device, pkl_data, fold_save_dir,
                                         save_every=args.save_every)
            fold_c_indices.append(fold_c)

            print(f"\n[FOLD {fold_idx}] Test C-Index = {fold_c:.4f}")

        # ── summary ───────────────────────────────────────────────────────
        fold_c_indices = np.array(fold_c_indices)
        mean_c = fold_c_indices.mean()
        std_c = fold_c_indices.std()

        print("\n" + "=" * 70)
        print("  ALL FOLDS SUMMARY")
        print("=" * 70)
        for i, c in enumerate(fold_c_indices, 1):
            print(f"  Fold {i:2d}: C-Index = {c:.4f}")
        print("-" * 40)
        print(f"  Mean C-Index = {mean_c:.4f} ± {std_c:.4f}")
        print("=" * 70)

        # save summary
        summary_path = os.path.join(base_save_dir, "summary.yaml")
        summary = {
            "n_folds": n_folds,
            "fold_c_indices": [float(c) for c in fold_c_indices],
            "mean_c_index": float(mean_c),
            "std_c_index": float(std_c),
        }
        with open(summary_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False)
        print(f"[INFO] Summary → {summary_path}")

    else:
        # ── single fold ──────────────────────────────────────────────────
        test_fold = data_cfg["test_fold"]
        val_fold = data_cfg["val_fold"]

        exp_name = args.exp_name or f"survival_fold{test_fold}_{datetime.now():%Y%m%d_%H%M%S}"
        save_dir = args.save_dir or os.path.join(
            data_cfg["data_root"], cfg["logging"]["checkpoint_dir"], exp_name)

        test_c = train_single_fold(cfg, args, test_fold, val_fold, device, pkl_data, save_dir,
                                       save_every=args.save_every)

    print("\n[INFO] Phase 0 complete.")


if __name__ == "__main__":
    main()
