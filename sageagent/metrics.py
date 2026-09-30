"""Evaluation metrics: C-index, hypervolume, calibration and bootstrap intervals."""

from __future__ import annotations

import numpy as np


def concordance_index(risk: np.ndarray, event: np.ndarray, time: np.ndarray, tol: float = 1e-8) -> float:
    """Harrell's C-index. Comparable pairs: i had the event and j survived longer.

    Higher risk should mean shorter survival; tied risks count one half.
    """
    risk = np.asarray(risk, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    comparable = (event[:, None] == 1) & (time[None, :] > time[:, None])
    if not comparable.any():
        return 0.5
    diff = risk[:, None] - risk[None, :]
    concordant = (diff > tol) & comparable
    tied = (np.abs(diff) <= tol) & comparable
    return float((concordant.sum() + 0.5 * tied.sum()) / comparable.sum())


def hypervolume(c_index: float, burden: float) -> float:
    """Accuracy-cost trade-off HV = (C - 0.5) * (1 - burden)."""
    return (c_index - 0.5) * (1.0 - burden)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Area under the ROC curve (ties count one half); None if only one class is present."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels) > 0.5
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bins = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(labels[in_bin].mean() - probs[in_bin].mean())
    return float(ece)


# =============================================================================
# Fold-aware summaries (paper protocol: mean over outer folds)
# =============================================================================
def summarize(risk, event, time, burden, fold) -> dict:
    """Mean over outer folds of the per-fold C-index; mean burden; HV of the means."""
    folds = np.unique(fold)
    c_per_fold = [concordance_index(risk[fold == f], event[fold == f], time[fold == f]) for f in folds]
    c, b = float(np.mean(c_per_fold)), float(np.mean(burden))
    return {"c_index": c, "c_index_std": float(np.std(c_per_fold)), "burden": b,
            "hv": hypervolume(c, b), "c_index_per_fold": dict(zip(map(int, folds), c_per_fold))}


def _resample_within_folds(fold: np.ndarray, rng: np.random.Generator):
    return np.concatenate([rng.choice(np.where(fold == f)[0], size=(fold == f).sum(), replace=True)
                           for f in np.unique(fold)])


def _boot_stats(risk, event, time, burden, fold, idx) -> tuple[float, float, float]:
    folds = fold[idx]
    c = np.mean([concordance_index(risk[idx][folds == f], event[idx][folds == f], time[idx][folds == f])
                 for f in np.unique(folds)])
    b = burden[idx].mean()
    return float(c), float(b), hypervolume(c, b)


def bootstrap_ci(risk, event, time, burden, fold, n_boot: int = 1000, seed: int = 42,
                 level: float = 0.95) -> dict:
    """Percentile intervals from bootstrap resamples drawn within each outer fold."""
    rng = np.random.default_rng(seed)
    draws = np.array([_boot_stats(risk, event, time, burden, fold, _resample_within_folds(fold, rng))
                      for _ in range(n_boot)])
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return {name: [float(np.percentile(draws[:, k], lo)), float(np.percentile(draws[:, k], hi))]
            for k, name in enumerate(["c_index", "burden", "hv"])}


def paired_bootstrap(a: dict, b: dict, n_boot: int = 1000, seed: int = 42) -> dict:
    """Paired comparison of two methods evaluated on the same patients.

    ``a`` and ``b`` hold per-patient arrays ``risk, event, time, burden, fold``.
    Returns the mean difference (a - b), its 95% interval and a two-sided
    bootstrap p-value for C-index, burden and HV.
    """
    for key in ("event", "time", "fold"):
        if not np.array_equal(a[key], b[key]):
            raise ValueError("paired bootstrap needs the same patients in the same order")
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = _resample_within_folds(a["fold"], rng)
        sa = _boot_stats(a["risk"], a["event"], a["time"], a["burden"], a["fold"], idx)
        sb = _boot_stats(b["risk"], b["event"], b["time"], b["burden"], b["fold"], idx)
        deltas.append(np.subtract(sa, sb))
    deltas = np.array(deltas)
    out = {}
    for k, name in enumerate(["c_index", "burden", "hv"]):
        d = deltas[:, k]
        out[name] = {"delta": float(d.mean()),
                     "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                     "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
    return out
