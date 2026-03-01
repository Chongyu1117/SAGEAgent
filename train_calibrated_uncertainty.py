"""
Train Calibrated Uncertainty Head — post-hoc, per-fold.

Predicts p(|risk_partial - risk_oracle| > tau) with BCE loss + temperature scaling.

Usage:
    python train_calibrated_uncertainty.py --nested --gpu 0
    python train_calibrated_uncertainty.py --nested --outer_fold 1 --gpu 0 --debug
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

from utils.data_loader import load_pkl_data, aggregate_patches_to_patients
from envs.clinical_env import load_predictor
from models.calibrated_uncertainty import (
    CalibratedUncertaintyHead,
    train_calibrated_head,
    save_calibrated_head,
    load_calibrated_head,
    generate_training_data,
    calibrate_temperature,
    _compute_ece,
)


def train_single_fold(
    fold: int,
    pkl_data: dict,
    config: dict,
    args,
    ckpt_path_override: str = None,
    save_dir_override: str = None,
    fold_label: str = None,
) -> dict:
    """Train calibrated uncertainty head for a single fold.

    Returns dict with fold results.
    """
    dc = config["data"]
    modality_keys = dc["modality_keys"]
    label = fold_label or f"FOLD {fold}"

    # Load predictor
    if ckpt_path_override:
        ckpt_path = ckpt_path_override
    else:
        ckpt_path = os.path.join(args.checkpoint_dir, f"fold_{fold}", "best_model.pt")

    if not os.path.exists(ckpt_path):
        print(f"[{label}] Checkpoint not found: {ckpt_path}, skipping")
        return {"fold": fold, "status": "skipped"}

    predictor = load_predictor(ckpt_path, config, device=args.device)

    # Build train/val data
    train_data = pkl_data["cv_splits"][fold]["train"]
    train_features, train_masks, _, _, train_names, _ = aggregate_patches_to_patients(
        train_data, modality_keys=modality_keys, debug=args.debug,
    )

    val_data = pkl_data["cv_splits"][fold]["val"]
    val_features, val_masks, _, _, _, _ = aggregate_patches_to_patients(
        val_data, modality_keys=modality_keys, debug=False,
    )

    # Count complete patients
    n_mod = train_features.shape[1]
    n_complete_train = sum(
        1 for i in range(len(train_features))
        if sum(1 for j in range(n_mod) if train_masks[i][j] > 0.5) >= args.min_modalities
    )
    n_complete_val = sum(
        1 for i in range(len(val_features))
        if sum(1 for j in range(n_mod) if val_masks[i][j] > 0.5) >= args.min_modalities
    )
    print(
        f"[{label}] Train: {len(train_features)} patients "
        f"({n_complete_train} with >={args.min_modalities} modalities), "
        f"Val: {len(val_features)} ({n_complete_val})"
    )

    # Determine mode and mask subset behavior
    mode = "regression" if args.regression else "binary"
    clinical_ordering_only = True
    if getattr(args, 'all_subsets', False):
        clinical_ordering_only = False

    # Train
    t0 = time.time()
    head = train_calibrated_head(
        predictor=predictor,
        train_features=train_features,
        train_masks=train_masks,
        val_features=val_features,
        val_masks=val_masks,
        embedding_dim=config["model"]["encoder"]["hidden_dim"],
        n_modalities=dc["n_modalities"],
        hidden_dim=args.hidden_dim,
        mode=mode,
        binary_threshold=args.threshold,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        min_modalities=args.min_modalities,
        clinical_ordering_only=clinical_ordering_only,
        device=args.device,
        debug=args.debug,
    )
    dt = time.time() - t0
    print(f"[{label}] Training took {dt:.1f}s")

    # Evaluate
    if mode == "binary":
        stats = evaluate_binary(
            head, predictor, val_features, val_masks,
            threshold=args.threshold, device=args.device, debug=args.debug,
        )
    else:
        stats = evaluate_discrimination(
            head, predictor, val_features, val_masks, args.device, args.debug
        )
    stats["fold"] = fold
    stats["train_time"] = dt
    stats["mode"] = mode

    # Save
    if save_dir_override:
        save_dir = save_dir_override
    else:
        save_dir = os.path.join(args.save_dir, f"fold_{fold}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "calibrated_head.pt")
    save_calibrated_head(head, save_path, metadata=stats)

    return stats


def evaluate_binary(
    head: CalibratedUncertaintyHead,
    predictor,
    features: np.ndarray,
    masks: np.ndarray,
    threshold: float = 0.3,
    device: str = "cuda",
    debug: bool = False,
) -> dict:
    """Evaluate binary classification cal head.

    Reports:
    1. Per-depth AUROC, positive rate, mean predicted probability
    2. Overall AUROC and ECE (with temperature scaling)
    3. Monotonicity of predicted probabilities
    4. Patient variance at depth 0
    """
    head.eval()
    n_patients, n_mod, feat_dim = features.shape

    # Find patients with all modalities
    complete_idx = [
        i for i in range(n_patients)
        if all(masks[i][j] > 0.5 for j in range(n_mod))
    ]

    if len(complete_idx) < 3:
        print(f"[EVAL] Only {len(complete_idx)} complete patients, limited evaluation")
        return {"n_complete": len(complete_idx)}

    mask_configs = [
        np.array([1, 0, 0, 0], dtype=np.float32),  # demo only
        np.array([1, 1, 0, 0], dtype=np.float32),  # demo+rad
        np.array([1, 1, 1, 0], dtype=np.float32),  # demo+rad+path
        np.array([1, 1, 1, 1], dtype=np.float32),  # all
    ]
    mask_labels = ["D...", "DR..", "DRP.", "DRPG"]

    all_probs = []
    all_true_labels = []
    results_by_mask = {}

    with torch.no_grad():
        # Oracle risks
        oracle_risks = []
        for i in complete_idx:
            feat_t = torch.from_numpy(features[i]).float().unsqueeze(0).to(device)
            mask_t = torch.from_numpy(masks[i]).float().unsqueeze(0).to(device)
            oracle_risks.append(predictor.get_risk_score(feat_t, mask_t).cpu().item())
        oracle_risks = np.array(oracle_risks)

        for mask_cfg, label in zip(mask_configs, mask_labels):
            probs = []
            true_labels = []

            for idx, i in enumerate(complete_idx):
                sub_feat = features[i].copy()
                for j in range(n_mod):
                    if mask_cfg[j] < 0.5:
                        sub_feat[j] = 0.0

                feat_t = torch.from_numpy(sub_feat).float().unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask_cfg).float().unsqueeze(0).to(device)

                emb = predictor.get_embedding(feat_t, mask_t)
                prob = head(emb, mask_t).cpu().item()
                risk = predictor.get_risk_score(feat_t, mask_t).cpu().item()
                true_label = float(abs(risk - oracle_risks[idx]) > threshold)

                probs.append(prob)
                true_labels.append(true_label)
                all_probs.append(prob)
                all_true_labels.append(true_label)

            probs = np.array(probs)
            true_labels = np.array(true_labels)

            # Per-depth AUROC (only if both classes present)
            auroc = _compute_auroc(probs, true_labels)

            results_by_mask[label] = {
                "mean_prob": float(probs.mean()),
                "std_prob": float(probs.std()),
                "pos_rate": float(true_labels.mean()),
                "auroc": auroc,
                "n_samples": len(probs),
            }

    all_probs = np.array(all_probs)
    all_true_labels = np.array(all_true_labels)

    # Overall AUROC
    overall_auroc = _compute_auroc(all_probs, all_true_labels)

    # ECE
    ece = _compute_ece(all_probs, all_true_labels)

    # Monotonicity: D > DR > DRP > DRPG in predicted probability
    means = [results_by_mask[l]["mean_prob"] for l in mask_labels]
    is_monotone = all(means[i] >= means[i + 1] for i in range(len(means) - 1))

    # Patient variance at depth 0
    patient_std_demo = results_by_mask["D..."]["std_prob"]

    # Print results
    print(f"[EVAL-BINARY] Results on {len(complete_idx)} complete patients (τ={threshold}):")
    for label in mask_labels:
        r = results_by_mask[label]
        auroc_str = f"{r['auroc']:.3f}" if r['auroc'] is not None else "N/A"
        print(
            f"  {label}: p̂={r['mean_prob']:.3f} ± {r['std_prob']:.3f}, "
            f"pos_rate={r['pos_rate']:.3f}, AUROC={auroc_str}"
        )
    print(f"  Overall AUROC: {overall_auroc if overall_auroc is not None else 'N/A'}")
    print(f"  ECE (15 bins): {ece:.4f}")
    print(f"  Monotone (more mods → lower p̂): {is_monotone}")
    print(f"  Patient variance (depth-0): {patient_std_demo:.4f}")
    print(f"  Temperature: {head.temperature.item():.4f}")

    return {
        "n_complete": len(complete_idx),
        "results_by_mask": results_by_mask,
        "overall_auroc": overall_auroc,
        "ece": ece,
        "is_monotone": is_monotone,
        "patient_std_demo": patient_std_demo,
        "temperature": head.temperature.item(),
        "threshold": threshold,
    }


def _compute_auroc(probs: np.ndarray, labels: np.ndarray):
    """Compute AUROC. Returns None if only one class present."""
    if len(np.unique(labels)) < 2:
        return None
    # Manual AUROC: sort by predicted prob descending, compute AUC
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    sorted_idx = np.argsort(-probs)
    sorted_labels = labels[sorted_idx]

    tp = 0
    fp = 0
    auc = 0.0
    for label in sorted_labels:
        if label > 0.5:
            tp += 1
        else:
            fp += 1
            auc += tp  # number of positive samples ranked above this negative

    return float(auc / (n_pos * n_neg))


def evaluate_discrimination(
    head: CalibratedUncertaintyHead,
    predictor,
    features: np.ndarray,
    masks: np.ndarray,
    device: str = "cuda",
    debug: bool = False,
) -> dict:
    """Regression evaluation. Check discrimination on val set.

    Tests:
    1. Variance across patients with same mask
    2. Monotonicity: more modalities → lower uncertainty
    3. Correlation with actual prediction discrepancy
    """
    head.eval()
    n_patients, n_mod, feat_dim = features.shape

    complete_idx = [
        i for i in range(n_patients)
        if all(masks[i][j] > 0.5 for j in range(n_mod))
    ]

    if len(complete_idx) < 3:
        print(f"[EVAL] Only {len(complete_idx)} complete patients, limited evaluation")
        return {"n_complete": len(complete_idx)}

    mask_configs = [
        np.array([1, 0, 0, 0], dtype=np.float32),
        np.array([1, 1, 0, 0], dtype=np.float32),
        np.array([1, 1, 1, 0], dtype=np.float32),
        np.array([1, 1, 1, 1], dtype=np.float32),
    ]
    mask_labels = ["D...", "DR..", "DRP.", "DRPG"]

    results_by_mask = {}
    all_pred_unc = []
    all_actual_disc = []

    with torch.no_grad():
        oracle_risks = []
        for i in complete_idx:
            feat_t = torch.from_numpy(features[i]).float().unsqueeze(0).to(device)
            mask_t = torch.from_numpy(masks[i]).float().unsqueeze(0).to(device)
            oracle_risks.append(predictor.get_risk_score(feat_t, mask_t).cpu().item())
        oracle_risks = np.array(oracle_risks)

        for mask_cfg, label in zip(mask_configs, mask_labels):
            uncertainties = []
            discrepancies = []

            for idx, i in enumerate(complete_idx):
                sub_feat = features[i].copy()
                for j in range(n_mod):
                    if mask_cfg[j] < 0.5:
                        sub_feat[j] = 0.0

                feat_t = torch.from_numpy(sub_feat).float().unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask_cfg).float().unsqueeze(0).to(device)

                emb = predictor.get_embedding(feat_t, mask_t)
                unc = head(emb, mask_t).cpu().item()
                risk = predictor.get_risk_score(feat_t, mask_t).cpu().item()

                uncertainties.append(unc)
                discrepancies.append(abs(risk - oracle_risks[idx]))

                all_pred_unc.append(unc)
                all_actual_disc.append(abs(risk - oracle_risks[idx]))

            uncertainties = np.array(uncertainties)
            results_by_mask[label] = {
                "mean_unc": float(uncertainties.mean()),
                "std_unc": float(uncertainties.std()),
                "min_unc": float(uncertainties.min()),
                "max_unc": float(uncertainties.max()),
                "mean_discrepancy": float(np.mean(discrepancies)),
            }

    all_pred_unc = np.array(all_pred_unc)
    all_actual_disc = np.array(all_actual_disc)
    correlation = float(np.corrcoef(all_pred_unc, all_actual_disc)[0, 1])

    means = [results_by_mask[l]["mean_unc"] for l in mask_labels]
    is_monotone = all(means[i] >= means[i + 1] for i in range(len(means) - 1))

    patient_std_demo = results_by_mask["D..."]["std_unc"]

    print(f"[EVAL-REGRESSION] Results on {len(complete_idx)} complete patients:")
    for label in mask_labels:
        r = results_by_mask[label]
        print(
            f"  {label}: mean_unc={r['mean_unc']:.4f} ± {r['std_unc']:.4f} "
            f"(range [{r['min_unc']:.4f}, {r['max_unc']:.4f}]), "
            f"actual_disc={r['mean_discrepancy']:.4f}"
        )
    print(f"  Correlation(pred_unc, actual_disc): {correlation:.4f}")
    print(f"  Monotone (more mods → lower unc): {is_monotone}")
    print(f"  Patient variance (demo-only mask): {patient_std_demo:.4f}")

    return {
        "n_complete": len(complete_idx),
        "results_by_mask": results_by_mask,
        "correlation": correlation,
        "is_monotone": is_monotone,
        "patient_std_demo": patient_std_demo,
    }


def print_summary(all_stats, label=""):
    """Print summary stats across folds."""
    mode = all_stats[0].get("mode", "regression") if all_stats else "unknown"

    print(f"\n{'='*60}")
    print(f"  SUMMARY {label}({len(all_stats)} folds, mode={mode})")
    print(f"{'='*60}")

    if mode == "binary":
        aurocs = [
            s["overall_auroc"] for s in all_stats
            if s.get("overall_auroc") is not None
        ]
        eces = [s["ece"] for s in all_stats if "ece" in s]
        monos = [s.get("is_monotone", False) for s in all_stats if "is_monotone" in s]
        patient_stds = [s.get("patient_std_demo", 0) for s in all_stats if "patient_std_demo" in s]
        temps = [s.get("temperature", 1.0) for s in all_stats if "temperature" in s]

        if aurocs:
            print(f"  AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
        if eces:
            print(f"  ECE: {np.mean(eces):.4f} ± {np.std(eces):.4f}")
        if monos:
            print(f"  Monotone folds: {sum(monos)}/{len(monos)}")
        if patient_stds:
            print(f"  Patient variance (depth-0): {np.mean(patient_stds):.4f} ± {np.std(patient_stds):.4f}")
        if temps:
            print(f"  Temperature: {np.mean(temps):.4f} ± {np.std(temps):.4f}")

        # Per-depth summary
        mask_labels = ["D...", "DR..", "DRP.", "DRPG"]
        for ml in mask_labels:
            depth_probs = [
                s["results_by_mask"][ml]["mean_prob"]
                for s in all_stats if "results_by_mask" in s and ml in s["results_by_mask"]
            ]
            depth_aurocs = [
                s["results_by_mask"][ml]["auroc"]
                for s in all_stats
                if "results_by_mask" in s and ml in s["results_by_mask"]
                and s["results_by_mask"][ml].get("auroc") is not None
            ]
            depth_pos = [
                s["results_by_mask"][ml]["pos_rate"]
                for s in all_stats if "results_by_mask" in s and ml in s["results_by_mask"]
            ]
            if depth_probs:
                auroc_str = f"{np.mean(depth_aurocs):.3f}" if depth_aurocs else "N/A"
                print(
                    f"  {ml}: p̂={np.mean(depth_probs):.3f} ± {np.std(depth_probs):.3f}, "
                    f"pos_rate={np.mean(depth_pos):.3f}, AUROC={auroc_str}"
                )
    else:
        # Regression summary
        corrs = [
            s["correlation"] for s in all_stats
            if "correlation" in s and not np.isnan(s["correlation"])
        ]
        monos = [s.get("is_monotone", False) for s in all_stats if "is_monotone" in s]
        patient_stds = [
            s.get("patient_std_demo", 0) for s in all_stats if "patient_std_demo" in s
        ]
        if corrs:
            print(f"  Correlation (pred vs actual): {np.mean(corrs):.4f} ± {np.std(corrs):.4f}")
        if monos:
            print(f"  Monotone folds: {sum(monos)}/{len(monos)}")
        if patient_stds:
            print(f"  Patient variance (demo-only): {np.mean(patient_stds):.4f} ± {np.std(patient_stds):.4f}")

    times = [s.get("train_time", 0) for s in all_stats]
    print(f"  Avg training time: {np.mean(times):.1f}s per fold")


def main():
    parser = argparse.ArgumentParser(
        description="Train calibrated uncertainty head (post-hoc, per-fold)"
    )
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="15-fold checkpoint dir (default: from config)")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--all_folds", action="store_true")

    # Nested 5x5 CV mode
    parser.add_argument("--nested", action="store_true",
                        help="Use nested 5x5 CV splits")
    parser.add_argument("--outer_fold", type=int, default=None,
                        help="Outer fold (1-5). None = run all 5")
    parser.add_argument("--inner_fold", type=int, default=None,
                        help="Inner fold (1-5). None = run all 5 inner folds")
    parser.add_argument("--n_inner", type=int, default=5,
                        help="Number of inner folds")
    parser.add_argument("--splits_dir", type=str, default=None,
                        help="Path to nested splits directory")
    parser.add_argument("--predictor_dir", type=str, default=None,
                        help="Path to predictor checkpoints (default: from config)")

    # Mode selection
    parser.add_argument("--regression", action="store_true",
                        help="Use regression mode instead of binary classification.")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Binary threshold τ (default 0.3). Only used in binary mode.")

    # Training hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_modalities", type=int, default=2,
                        help="Minimum modalities a patient must have to be used for training")
    parser.add_argument("--all_subsets", action="store_true",
                        help="Use all mask subsets instead of clinical ordering only")

    # Output
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Output dir for calibrated heads (default: from config)")
    parser.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # Resolve device
    if args.device is None:
        if args.gpu >= 0 and torch.cuda.is_available():
            args.device = f"cuda:{args.gpu}"
        else:
            args.device = "cpu"

    # Load config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dc = config["data"]
    data_root = dc["data_root"]
    paths_cfg = config.get("paths", {})

    mode = "regression" if args.regression else "binary"
    print(f"[CALIB] Mode: {mode}" + (f", τ={args.threshold}" if mode == "binary" else ""))

    # Resolve defaults from config
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(
            data_root, paths_cfg.get("checkpoint_dir_15fold", "checkpoints/15fold")
        )
    if args.save_dir is None:
        args.save_dir = os.path.join(data_root, "checkpoints", "calibrated_uncertainty")

    if args.nested:
        # ── nested 5x5 CV mode ───────────────────────────────────────
        splits_dir = args.splits_dir or os.path.join(
            data_root, paths_cfg.get("splits_dir", "splits")
        )
        predictor_dir = args.predictor_dir or os.path.join(
            data_root, paths_cfg.get("predictor_checkpoint_dir", "checkpoints/nested5x5")
        )
        n_inner = args.n_inner
        inner_folds = [args.inner_fold] if args.inner_fold else list(range(1, n_inner + 1))
        outer_folds = [args.outer_fold] if args.outer_fold else list(range(1, 6))

        print(f"[CALIB] Nested 5x{n_inner} CV mode")
        print(f"[CALIB] Splits: {splits_dir}")
        print(f"[CALIB] Predictors: {predictor_dir}")
        print(f"[CALIB] Outer folds: {outer_folds}, Inner folds: {inner_folds}")
        print(f"[CALIB] Device: {args.device}")

        all_stats = []
        for outer in outer_folds:
            print(f"\n{'='*60}")
            print(f"  OUTER FOLD {outer}/5")
            print(f"{'='*60}")

            # Load per-outer-fold pkl
            pkl_path = os.path.join(splits_dir, f"outer_{outer}", "predictor.pkl")
            print(f"[CALIB] Loading: {pkl_path}")
            pkl_data = load_pkl_data(pkl_path, debug=args.debug)

            for inner in inner_folds:
                print(f"\n  --- Inner fold {inner}/{n_inner} ---")

                # Predictor checkpoint: outer_N/inner_M/best_model.pt
                ckpt_path = os.path.join(
                    predictor_dir, f"outer_{outer}", f"inner_{inner}", "best_model.pt")

                # Save calibrated head alongside predictor
                save_dir = os.path.join(
                    predictor_dir, f"outer_{outer}", f"inner_{inner}")

                stats = train_single_fold(
                    fold=inner,
                    pkl_data=pkl_data,
                    config=config,
                    args=args,
                    ckpt_path_override=ckpt_path,
                    save_dir_override=save_dir,
                    fold_label=f"O{outer}/I{inner}",
                )
                stats["outer_fold"] = outer
                stats["inner_fold"] = inner
                all_stats.append(stats)

            del pkl_data

        if len(all_stats) > 1:
            print_summary(all_stats, label=f"nested 5x{n_inner} ")

    else:
        # ── 15-fold mode ──────────────────────────────────────
        pkl_path = os.path.join(dc["data_root"], dc["features_file"])
        print(f"[CALIB] Loading data from {pkl_path}")
        pkl_data = load_pkl_data(pkl_path, debug=args.debug)

        n_folds = dc["n_folds"]
        folds = list(range(1, n_folds + 1)) if args.all_folds else [args.fold]

        all_stats = []
        for fold in folds:
            print(f"\n{'='*60}")
            print(f"  FOLD {fold}")
            print(f"{'='*60}")
            stats = train_single_fold(fold, pkl_data, config, args)
            all_stats.append(stats)

        if len(folds) > 1:
            print_summary(all_stats)


if __name__ == "__main__":
    main()
