"""
Data Loader — Patch-level pkl → Patient-level aggregated features.

The raw pkl stores data at PATCH level (multiple patches per patient).
This module aggregates patches to patient level for survival prediction.

Key design: ALL parameters (feature keys, fold numbers, etc.) come from
function arguments — nothing hardcoded.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional


def load_pkl_data(pkl_path: str, debug: bool = False) -> dict:
    """
    Load the raw pkl file.

    Args:
        pkl_path: absolute path to the pkl file
        debug: print debug info

    Returns:
        dict with keys 'cv_splits' and optionally 'data_pd'
    """
    if debug:
        print(f"[DATA DEBUG] Loading pkl from: {pkl_path}")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pkl file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if debug:
        print(f"[DATA DEBUG] Top-level keys: {list(data.keys())}")
        if "cv_splits" in data:
            folds = sorted(data["cv_splits"].keys())
            print(f"[DATA DEBUG] Available folds: {folds}")
            first_fold = folds[0]
            splits = list(data["cv_splits"][first_fold].keys())
            print(f"[DATA DEBUG] Splits in fold {first_fold}: {splits}")

    return data


def aggregate_patches_to_patients(
    fold_split_data: dict,
    modality_keys: Dict[str, str],
    pathology_agg: str = "mean",
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Aggregate patch-level features to patient-level.

    Pathology features (x_path_fea) vary per patch → aggregated (mean/max).
    Other modalities (rad, omic, demo) are patient-level → take first occurrence.

    Args:
        fold_split_data: dict from pkl['cv_splits'][fold][split]
        modality_keys: ordered dict mapping modality_name -> pkl_key
            e.g. {"demographics": "x_demo", "radiology": "x_rad", ...}
        pathology_agg: how to aggregate pathology patches ("mean" or "max")
        debug: print debug info

    Returns:
        features: (N_patients, n_modalities, feature_dim) float32
        masks:    (N_patients, n_modalities) float32, 1=present 0=missing
        events:   (N_patients,) float32
        times:    (N_patients,) float32
        names:    list of patient name strings
        grades:   (N_patients,) float32
    """
    patnames = fold_split_data["x_patname"]
    if isinstance(patnames, np.ndarray):
        patnames = patnames.tolist()

    # ordered unique patients (preserve original order)
    unique_patients = list(OrderedDict.fromkeys(patnames))
    n_patients = len(unique_patients)

    # build name -> list of row indices
    pat2idx: Dict[str, List[int]] = {}
    for i, name in enumerate(patnames):
        pat2idx.setdefault(name, []).append(i)

    modality_names = list(modality_keys.keys())
    pkl_keys = list(modality_keys.values())
    n_modalities = len(modality_names)

    # detect feature_dim from first modality array
    sample_arr = fold_split_data[pkl_keys[0]]
    if sample_arr.ndim == 3:
        feature_dim = sample_arr.shape[2]
    elif sample_arr.ndim == 2:
        feature_dim = sample_arr.shape[1]
    else:
        raise ValueError(f"Unexpected feature array ndim={sample_arr.ndim}")

    if debug:
        print(f"[DATA DEBUG] Aggregating {len(patnames)} patches → {n_patients} patients")
        print(f"[DATA DEBUG] Modalities: {modality_names}")
        print(f"[DATA DEBUG] Pkl keys:   {pkl_keys}")
        print(f"[DATA DEBUG] feature_dim={feature_dim}, pathology_agg={pathology_agg}")

    features = np.zeros((n_patients, n_modalities, feature_dim), dtype=np.float32)
    events = np.zeros(n_patients, dtype=np.float32)
    times = np.zeros(n_patients, dtype=np.float32)
    grades = np.zeros(n_patients, dtype=np.float32)

    # identify which key is pathology (needs multi-patch aggregation)
    path_key = modality_keys.get("pathology", None)

    for p_i, pname in enumerate(unique_patients):
        idx_list = pat2idx[pname]
        first_idx = idx_list[0]

        # labels — same across all patches of a patient
        events[p_i] = float(fold_split_data["e"][first_idx])
        times[p_i] = float(fold_split_data["t"][first_idx])
        if "g" in fold_split_data:
            grades[p_i] = float(fold_split_data["g"][first_idx])

        for m_i, (mod_name, pkl_key) in enumerate(modality_keys.items()):
            arr = fold_split_data[pkl_key]

            if pkl_key == path_key and len(idx_list) > 1:
                # pathology: aggregate across patches
                patches = arr[idx_list]  # (n_patches, 1, feat_dim) or (n_patches, feat_dim)
                if patches.ndim == 3:
                    patches = patches[:, 0, :]  # -> (n_patches, feat_dim)
                if pathology_agg == "mean":
                    features[p_i, m_i, :] = patches.mean(axis=0)
                elif pathology_agg == "max":
                    features[p_i, m_i, :] = patches.max(axis=0)
                else:
                    features[p_i, m_i, :] = patches.mean(axis=0)
            else:
                # other modalities: same across patches, take first
                val = arr[first_idx]
                if val.ndim == 2:
                    val = val[0]  # (1, feat_dim) -> (feat_dim,)
                features[p_i, m_i, :] = val.astype(np.float32)

    # derive masks: modality is present if feature vector is NOT all-zero
    masks = (np.abs(features).sum(axis=2) > 1e-8).astype(np.float32)

    if debug:
        print(f"[DATA DEBUG] Aggregation complete:")
        print(f"  features shape: {features.shape}")
        print(f"  masks shape:    {masks.shape}")
        for m_i, name in enumerate(modality_names):
            n_present = int(masks[:, m_i].sum())
            print(f"  {name:15s}: {n_present}/{n_patients} present ({100*n_present/n_patients:.1f}%)")
        print(f"  events: {int(events.sum())} deaths / {n_patients} total")
        print(f"  time range: [{times.min():.0f}, {times.max():.0f}]")

    return features, masks, events, times, unique_patients, grades


class PatientDataset(Dataset):
    """
    Patient-level dataset for survival prediction.

    Optionally applies modality-dropout augmentation during training:
    randomly sets some present modalities to zero (simulates missing data).
    """

    def __init__(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        events: np.ndarray,
        times: np.ndarray,
        names: List[str],
        grades: Optional[np.ndarray] = None,
        augment: bool = False,
        mask_augment_prob: float = 0.15,
    ):
        """
        Args:
            features: (N, n_mod, feat_dim)
            masks:    (N, n_mod)
            events:   (N,)
            times:    (N,)
            names:    list of str
            grades:   (N,) or None
            augment:  whether to apply random modality dropout
            mask_augment_prob: prob of dropping each present modality
        """
        self.features = torch.from_numpy(features).float()
        self.masks = torch.from_numpy(masks).float()
        self.events = torch.from_numpy(events).float()
        self.times = torch.from_numpy(times).float()
        self.names = names
        self.grades = torch.from_numpy(grades).float() if grades is not None else None
        self.augment = augment
        self.mask_augment_prob = mask_augment_prob

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        feat = self.features[idx].clone()   # (n_mod, feat_dim)
        mask = self.masks[idx].clone()       # (n_mod,)

        # Keep original features and mask for reconstruction targets
        orig_feat = self.features[idx].clone()
        orig_mask = self.masks[idx].clone()

        if self.augment and self.mask_augment_prob > 0:
            # randomly drop some present modalities (augmentation)
            drop = (torch.rand(mask.shape) < self.mask_augment_prob) & (mask > 0.5)
            mask[drop] = 0.0
            feat[drop] = 0.0
            # ensure at least one modality remains
            if mask.sum() == 0:
                orig_present = (self.masks[idx] > 0.5).nonzero(as_tuple=True)[0]
                if len(orig_present) > 0:
                    keep = orig_present[torch.randint(len(orig_present), (1,))]
                    mask[keep] = 1.0
                    feat[keep] = self.features[idx, keep]

        sample = {
            "features": feat,
            "masks": mask,
            "orig_features": orig_feat,
            "orig_masks": orig_mask,
            "events": self.events[idx],
            "times": self.times[idx],
            "name": self.names[idx],
        }
        if self.grades is not None:
            sample["grades"] = self.grades[idx]
        return sample


def build_patient_dataset(
    pkl_data: dict,
    fold: int,
    split: str,
    modality_keys: Dict[str, str],
    pathology_agg: str = "mean",
    augment: bool = False,
    mask_augment_prob: float = 0.15,
    debug: bool = False,
) -> PatientDataset:
    """
    Build a PatientDataset from pkl data for a given fold and split.

    Args:
        pkl_data: loaded pkl dict (from load_pkl_data)
        fold: fold number (1-15)
        split: "train", "val", or "test"
        modality_keys: {modality_name: pkl_key}
        pathology_agg: aggregation for pathology patches
        augment: enable modality-dropout augmentation
        mask_augment_prob: dropout probability per modality
        debug: print debug info

    Returns:
        PatientDataset
    """
    if debug:
        print(f"\n[DATA DEBUG] Building dataset: fold={fold}, split={split}")

    fold_split_data = pkl_data["cv_splits"][fold][split]

    features, masks, events, times, names, grades = aggregate_patches_to_patients(
        fold_split_data,
        modality_keys=modality_keys,
        pathology_agg=pathology_agg,
        debug=debug,
    )

    dataset = PatientDataset(
        features=features,
        masks=masks,
        events=events,
        times=times,
        names=names,
        grades=grades,
        augment=augment,
        mask_augment_prob=mask_augment_prob,
    )

    if debug:
        print(f"[DATA DEBUG] Dataset created: {len(dataset)} patients, augment={augment}")

    return dataset


def create_dataloaders(
    pkl_data: dict,
    test_fold: int,
    val_fold: int,
    modality_keys: Dict[str, str],
    batch_size: int = 32,
    num_workers: int = 4,
    pathology_agg: str = "mean",
    mask_augment_prob: float = 0.15,
    debug: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train / val / test DataLoaders for one CV configuration.

    Train = all folds except test_fold and val_fold (merged).
    Val   = val_fold.
    Test  = test_fold.

    Args:
        pkl_data: loaded pkl dict
        test_fold: fold number for test
        val_fold: fold number for validation
        modality_keys: {modality_name: pkl_key}
        batch_size: batch size for all loaders
        num_workers: dataloader workers
        pathology_agg: aggregation method for pathology
        mask_augment_prob: modality dropout prob (train only)
        debug: print debug info

    Returns:
        (train_loader, val_loader, test_loader)
    """
    available_folds = sorted(pkl_data["cv_splits"].keys())

    if debug:
        print(f"\n[DATA DEBUG] Creating dataloaders:")
        print(f"  available folds: {available_folds}")
        print(f"  test_fold={test_fold}, val_fold={val_fold}")

    # --- test & val datasets (from their fold's 'test'/'val' split) ---
    test_ds = build_patient_dataset(
        pkl_data, fold=test_fold, split="test",
        modality_keys=modality_keys, pathology_agg=pathology_agg,
        augment=False, debug=debug,
    )
    val_ds = build_patient_dataset(
        pkl_data, fold=val_fold, split="val",
        modality_keys=modality_keys, pathology_agg=pathology_agg,
        augment=False, debug=debug,
    )
    train_ds = build_patient_dataset(
        pkl_data, fold=test_fold, split="train",
        modality_keys=modality_keys, pathology_agg=pathology_agg,
        augment=True, mask_augment_prob=mask_augment_prob, debug=debug,
    )

    if debug:
        print(f"\n[DATA DEBUG] Final sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
