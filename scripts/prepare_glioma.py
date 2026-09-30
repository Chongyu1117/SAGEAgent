"""Build the glioma cohort file from the MMD feature pickle (Cui et al., MICCAI 2022).

The result ships with the repository (embedding/glioma/cohort.npz); this script
shows how it was built. The source is the patch-level 15-fold pickle
`patches_1212_gbmlgg15cv_32_all_15fold_original.pkl`
with 32-d features per modality. The train, validation and test sets of one fold
together contain every patient once. Pathology features are averaged over a
patient's patches; the other modalities are patient-level. A modality is
available when its feature vector is non-zero.

    python scripts/prepare_glioma.py --source /path/to/patches_..._15fold_original.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sageagent.config import resolve_path  # noqa: E402
from sageagent.data import Cohort  # noqa: E402

MODALITY_KEYS = {"demographics": "x_demo", "radiology": "x_rad", "pathology": "x_path_fea", "genomics": "x_omic"}
PATCH_LEVEL = {"pathology"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True, help="MMD patch-level feature pickle")
    parser.add_argument("--out", default="embedding/glioma/cohort.npz", help="output cohort file")
    parser.add_argument("--fold", type=int, default=1, help="fold whose train/val/test are pooled")
    args = parser.parse_args()

    with open(args.source, "rb") as f:
        source = pickle.load(f)["cv_splits"][args.fold]
    parts = [source[s] for s in ("train", "val", "test")]
    names = [str(n) for part in parts for n in part["x_patname"]]
    rows = {k: np.concatenate([np.asarray(part[k], dtype=np.float64) for part in parts])
            for k in list(MODALITY_KEYS.values()) + ["e", "t", "g"]}

    patients, first, members = [], {}, {}
    for i, name in enumerate(names):
        if name not in first:
            patients.append(name)
            first[name] = i
        members.setdefault(name, []).append(i)

    dims = [rows[k].reshape(len(names), -1).shape[1] for k in MODALITY_KEYS.values()]
    features = np.zeros((len(patients), len(MODALITY_KEYS), max(dims)), dtype=np.float32)
    for j, (modality, key) in enumerate(MODALITY_KEYS.items()):
        values = rows[key].reshape(len(names), -1)
        for p, name in enumerate(patients):
            idx = members[name] if modality in PATCH_LEVEL else [first[name]]
            features[p, j, : dims[j]] = values[idx].mean(axis=0)
    mask = (np.abs(features).sum(axis=2) > 1e-8).astype(np.float32)

    lead = [first[n] for n in patients]
    event, time, grade = rows["e"][lead], rows["t"][lead], rows["g"][lead]
    for label, values in (("event", event), ("time", time)):
        if np.isnan(values).any():
            raise ValueError(f"{np.isnan(values).sum()} patients have no {label}")
    cohort = Cohort(ids=np.array(patients), features=features, mask=mask, event=event.astype(np.float32),
                    time=time.astype(np.float32), modality_dims=dims,
                    strata=np.nan_to_num(grade, nan=0).astype(np.int64))

    out = resolve_path(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cohort.save(out, list(MODALITY_KEYS))
    print(f"{len(patients)} patients, {int(cohort.complete.sum())} with all modalities -> {out}")
    for j, modality in enumerate(MODALITY_KEYS):
        print(f"  {modality:<13s} available for {int(mask[:, j].sum())} patients")


if __name__ == "__main__":
    main()
