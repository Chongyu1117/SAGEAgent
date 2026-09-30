"""Patient-level cohort data and nested cross-validation splits.

Cohort file (``.npz``), one row per patient:
    patient_id  (N,)      str
    modalities  (M,)      str, modality names in pathway order
    x_<name>    (N, d_m)  float features of each modality (zeros when missing)
    mask        (N, M)    1 if the modality is available for the patient
    event       (N,)      1 = event (death) observed, 0 = censored
    time        (N,)      survival or follow-up time
    strata      (N,)      optional integer used for stratified splitting

Splits file (``.json``): for each outer fold the test patients (complete
modalities only) and, for each inner fold, the train/validation patients.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .utils import load_json, save_json


# =============================================================================
# Cohort
# =============================================================================
@dataclass
class Cohort:
    ids: np.ndarray               # (N,)
    features: np.ndarray          # (N, M, D) zero-padded to the largest modality dim
    mask: np.ndarray              # (N, M)
    event: np.ndarray             # (N,)
    time: np.ndarray              # (N,)
    modality_dims: list[int]
    strata: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def complete(self) -> np.ndarray:
        """Boolean array: patient has every modality."""
        return self.mask.min(axis=1) > 0.5

    @classmethod
    def load(cls, path: str, modality_names: Sequence[str]) -> "Cohort":
        with np.load(path, allow_pickle=False) as npz:
            stored = list(npz["modalities"])
            missing = [m for m in modality_names if m not in stored]
            if missing:
                raise KeyError(f"{path} has no features for modalities {missing}; found {stored}")
            order = [stored.index(m) for m in modality_names]
            blocks = [npz[f"x_{m}"].astype(np.float32) for m in modality_names]
            dims = [b.shape[1] for b in blocks]
            features = np.zeros((len(npz["patient_id"]), len(blocks), max(dims)), dtype=np.float32)
            for j, block in enumerate(blocks):
                features[:, j, : block.shape[1]] = block
            mask = npz["mask"].astype(np.float32)[:, order]
            if (mask.sum(axis=1) == 0).any():
                raise ValueError(f"{int((mask.sum(axis=1) == 0).sum())} patients in {path} have no modality")
            features *= mask[:, :, None]
            return cls(
                ids=npz["patient_id"].astype(str),
                features=features,
                mask=mask,
                event=npz["event"].astype(np.float32),
                time=npz["time"].astype(np.float32),
                modality_dims=dims,
                strata=npz["strata"].astype(np.int64) if "strata" in npz.files else None,
            )

    def save(self, path: str, modality_names: Sequence[str]) -> None:
        arrays = {
            "patient_id": self.ids.astype(str),
            "modalities": np.array(modality_names),
            "mask": self.mask.astype(np.float32),
            "event": self.event.astype(np.float32),
            "time": self.time.astype(np.float32),
        }
        for j, name in enumerate(modality_names):
            arrays[f"x_{name}"] = self.features[:, j, : self.modality_dims[j]]
        if self.strata is not None:
            arrays["strata"] = self.strata
        np.savez_compressed(path, **arrays)

    def index_of(self, ids: Sequence[str]) -> np.ndarray:
        lookup = {pid: i for i, pid in enumerate(self.ids)}
        try:
            return np.array([lookup[pid] for pid in ids], dtype=np.int64)
        except KeyError as err:
            raise KeyError(f"patient {err} is in the splits but not in the cohort") from None

    def subset(self, ids: Sequence[str]) -> "Cohort":
        idx = self.index_of(ids)
        return Cohort(
            ids=self.ids[idx], features=self.features[idx], mask=self.mask[idx],
            event=self.event[idx], time=self.time[idx], modality_dims=list(self.modality_dims),
            strata=None if self.strata is None else self.strata[idx],
        )


# =============================================================================
# Nested cross-validation splits
# =============================================================================
class NestedSplits:
    def __init__(self, folds: dict, meta: dict):
        self._folds = folds          # {outer: {"test": [...], "inner": {inner: {"train", "val"}}}}
        self.meta = meta

    @property
    def n_outer(self) -> int:
        return len(self._folds)

    @property
    def n_inner(self) -> int:
        return len(next(iter(self._folds.values()))["inner"])

    def test(self, outer: int) -> list[str]:
        return self._folds[outer]["test"]

    def train(self, outer: int, inner: int) -> list[str]:
        return self._folds[outer]["inner"][inner]["train"]

    def val(self, outer: int, inner: int) -> list[str]:
        return self._folds[outer]["inner"][inner]["val"]

    def save(self, path: str) -> None:
        folds = [
            {"outer": o, "test": f["test"],
             "inner": [{"inner": i, **f["inner"][i]} for i in sorted(f["inner"])]}
            for o, f in sorted(self._folds.items())
        ]
        save_json({**self.meta, "folds": folds}, path, indent=1)

    @classmethod
    def load(cls, path: str) -> "NestedSplits":
        data = load_json(path)
        folds = {
            f["outer"]: {"test": f["test"],
                         "inner": {s["inner"]: {"train": s["train"], "val": s["val"]} for s in f["inner"]}}
            for f in data.pop("folds")
        }
        return cls(folds, data)


def _strata_labels(cohort: Cohort, idx: np.ndarray, min_per_class: int) -> np.ndarray:
    """Event x strata labels; classes too rare to stratify fall back to event only."""
    events = cohort.event[idx].astype(int)
    if cohort.strata is None:
        return events.astype(str)
    labels = np.array([f"{e}_{s}" for e, s in zip(events, cohort.strata[idx])])
    counts = Counter(labels)
    return np.array([lab if counts[lab] >= min_per_class else lab.split("_")[0] for lab in labels])


def make_nested_splits(cohort: Cohort, n_outer: int = 5, n_inner: int = 5, seed: int = 42) -> NestedSplits:
    """Outer folds over complete-modality patients; inner folds over all remaining patients.

    The outer test sets partition the complete-modality patients (the evaluation
    set). For every outer fold, the remaining complete patients and all
    partial-modality patients are split into inner train/validation folds. The
    predictor trains on all of them; the agent uses the complete ones.
    """
    from sklearn.model_selection import StratifiedKFold

    complete = np.where(cohort.complete)[0]
    partial = np.where(~cohort.complete)[0]
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    outer_splits = outer_cv.split(complete, _strata_labels(cohort, complete, n_outer))
    partial_labels = _strata_labels(cohort, partial, n_inner)

    folds = {}
    for o, (keep, test) in enumerate(outer_splits, start=1):
        remaining = complete[keep]
        inner_seed = seed + o
        complete_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=inner_seed)
        partial_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=inner_seed)
        complete_splits = complete_cv.split(remaining, _strata_labels(cohort, remaining, n_inner))
        partial_splits = partial_cv.split(partial, partial_labels)
        inner = {}
        for i, ((c_tr, c_va), (p_tr, p_va)) in enumerate(zip(complete_splits, partial_splits), start=1):
            inner[i] = {
                "train": cohort.ids[np.concatenate([remaining[c_tr], partial[p_tr]])].tolist(),
                "val": cohort.ids[np.concatenate([remaining[c_va], partial[p_va]])].tolist(),
            }
        folds[o] = {"test": cohort.ids[complete[test]].tolist(), "inner": inner}

    meta = {
        "description": "Outer folds split the complete-modality patients (test sets); inner folds "
                       "split the remaining complete and partial-modality patients (train/val).",
        "seed": seed, "n_outer": n_outer, "n_inner": n_inner,
        "n_patients": len(cohort), "n_complete": int(len(complete)), "n_partial": int(len(partial)),
    }
    splits = NestedSplits(folds, meta)
    check_splits(splits, cohort)
    return splits


def check_splits(splits: NestedSplits, cohort: Cohort) -> None:
    """Assert there is no leakage between train, validation and test sets."""
    complete_ids = set(cohort.ids[cohort.complete])
    all_ids = set(cohort.ids)
    seen_test: set[str] = set()
    for o in range(1, splits.n_outer + 1):
        test = set(splits.test(o))
        assert test <= complete_ids, f"outer {o}: test set contains patients with missing modalities"
        assert not test & seen_test, f"outer {o}: test patients repeated across outer folds"
        seen_test |= test
        for i in range(1, splits.n_inner + 1):
            train, val = set(splits.train(o, i)), set(splits.val(o, i))
            assert not train & val and not train & test and not val & test, f"fold {o}/{i}: overlap"
            assert train | val | test == all_ids, f"fold {o}/{i}: patients missing from the split"
    assert seen_test == complete_ids, "outer test sets must cover every complete-modality patient"
