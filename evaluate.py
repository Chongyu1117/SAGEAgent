"""
Unified Evaluation Framework for Clinical Modality Acquisition.

All methods (heuristic, RL, LLM agent, ablations) are evaluated through the
same pipeline for fair comparison.

Evaluation protocol:
    - Nested 5x5 CV (5 outer folds, each with 5 inner folds)
    - Test on complete-modality patients in each outer fold's test set
    - Report mean +/- std C-Index across folds
    - Statistical tests: Wilcoxon signed-rank, bootstrap CI

Usage:
    python evaluate.py --results_dir results/ --generate_plots
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

from scipy import stats

from utils.metrics import compute_c_index
from envs.clinical_env import (
    ClinicalEnv,
    MODALITY_NAMES,
    N_MODALITIES,
)


# ═══════════════════════════════════════════════════════════════════════════
# Single-fold evaluation
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_policy(
    policy,
    env: ClinicalEnv,
    test_indices: List[int],
    debug: bool = False,
) -> dict:
    """Evaluate *policy* on a list of patients in *env*.

    Args:
        policy:       object with ``select_action(state, env) -> int`` and
                      ``reset()`` methods.
        env:          ClinicalEnv with patient data loaded.
        test_indices: which patient indices to evaluate on.
        debug:        per-patient logging.

    Returns:
        dict with ``c_index``, ``avg_burden``, ``risk_scores``, etc.
    """
    risk_scores: List[float] = []
    events: List[float] = []
    times: List[float] = []
    burdens: List[float] = []
    uncertainties: List[float] = []
    n_modalities_list: List[int] = []
    acquisition_counts = np.zeros(N_MODALITIES, dtype=int)
    trajectories: List[dict] = []

    for patient_idx in test_indices:
        policy.reset()
        # Demographics always pre-acquired (chart review, burden=0.03)
        initial_mask = np.zeros(N_MODALITIES, dtype=np.float32)
        initial_mask[0] = 1.0
        state = env.reset(patient_idx, initial_mask=initial_mask)

        done = False
        while not done:
            action = policy.select_action(state, env)
            state, reward, done, info = env.step(action)

        summary = env.get_episode_summary()
        risk_scores.append(state["risk_score"])
        events.append(summary["event"])
        times.append(summary["time"])
        burdens.append(summary["total_burden"])
        uncertainties.append(float(state.get("uncertainty", 0.0)))
        n_modalities_list.append(summary["n_acquired"])
        acquisition_counts += summary["acquired_mask"].astype(int)
        trajectories.append(summary)

    risk_arr = np.array(risk_scores)
    event_arr = np.array(events)
    time_arr = np.array(times)

    c_index = compute_c_index(risk_arr, event_arr, time_arr)

    unc_arr = np.array(uncertainties)

    result = {
        "c_index": c_index,
        "avg_burden": float(np.mean(burdens)),
        "std_burden": float(np.std(burdens)),
        "avg_n_modalities": float(np.mean(n_modalities_list)),
        "n_patients": len(test_indices),
        "risk_scores": risk_arr,
        "events": event_arr,
        "times": time_arr,
        "uncertainties": unc_arr,
        "burdens": burdens,
        "n_modalities_acquired": n_modalities_list,
        "acquisition_counts": acquisition_counts,
        "trajectories": trajectories,
    }

    if debug:
        acq_str = " ".join(
            f"{MODALITY_NAMES[i]}={acquisition_counts[i]}"
            for i in range(N_MODALITIES)
        )
        print(
            f"[EVAL] {policy.name}: C={c_index:.4f} "
            f"burden={np.mean(burdens):.3f}±{np.std(burdens):.3f} "
            f"mods={np.mean(n_modalities_list):.2f} | {acq_str}"
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Statistical tests
# ═══════════════════════════════════════════════════════════════════════════
def compute_statistics(
    results_a: dict,
    results_b: dict,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Compare two methods via Wilcoxon signed-rank + bootstrap CI.

    Args:
        results_a, results_b: dicts with ``fold_results`` key.
        n_bootstrap:          number of bootstrap resamples.
        seed:                 random seed.

    Returns:
        dict with p-values and confidence intervals.
    """
    c_a = np.array([r["c_index"] for r in results_a["fold_results"]])
    c_b = np.array([r["c_index"] for r in results_b["fold_results"]])

    # Align folds
    n = min(len(c_a), len(c_b))
    c_a, c_b = c_a[:n], c_b[:n]

    if n < 5:
        return {"error": f"too few folds ({n}) for statistical test"}

    # Wilcoxon signed-rank
    try:
        stat, p_val = stats.wilcoxon(c_a, c_b, alternative="two-sided")
    except ValueError:
        stat, p_val = 0.0, 1.0

    # Bootstrap CI for the difference
    diff = c_a - c_b
    rng = np.random.RandomState(seed)
    boot_means = np.array(
        [diff[rng.choice(n, n, replace=True)].mean() for _ in range(n_bootstrap)]
    )

    return {
        "method_a_mean": float(c_a.mean()),
        "method_b_mean": float(c_b.mean()),
        "mean_diff": float(diff.mean()),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p_val),
        "significant_p05": bool(p_val < 0.05),
        "bootstrap_ci_95": (
            float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)),
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════
def generate_plots(
    all_results: Dict[str, dict],
    output_dir: str = "plots",
    figsize: Tuple[int, int] = (10, 6),
):
    """Generate standard experiment figures.

    Args:
        all_results: ``{method_name: eval_results_dict}``
        output_dir:  where to save PNGs.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    # --- 1. C-Index comparison bar chart ---
    names = list(all_results.keys())
    means = [all_results[n]["mean_c_index"] for n in names]
    stds = [all_results[n]["std_c_index"] for n in names]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=sns.color_palette("muted"))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("C-Index")
    ax.set_title("Survival Prediction Performance (Nested 5x5 CV)")
    ax.set_ylim(0.5, 1.0)
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "c_index_comparison.png"), dpi=150)
    plt.close()
    print(f"[PLOT] Saved c_index_comparison.png")

    # --- 2. Burden vs C-Index scatter ---
    fig, ax = plt.subplots(figsize=figsize)
    for name in names:
        r = all_results[name]
        ax.scatter(
            r["avg_burden"],
            r["mean_c_index"],
            s=100,
            label=name,
            zorder=5,
        )
        ax.errorbar(
            r["avg_burden"],
            r["mean_c_index"],
            yerr=r["std_c_index"],
            fmt="none",
            capsize=3,
            color="gray",
            alpha=0.5,
        )
    ax.set_xlabel("Average Clinical Burden")
    ax.set_ylabel("C-Index")
    ax.set_title("Prediction Quality vs. Clinical Burden")
    ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "burden_vs_cindex.png"), dpi=150)
    plt.close()
    print(f"[PLOT] Saved burden_vs_cindex.png")

    # --- 3. Per-fold C-Index box plot ---
    fold_data = {}
    for name in names:
        r = all_results[name]
        fold_data[name] = [fr["c_index"] for fr in r["fold_results"]]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(
        [fold_data[n] for n in names],
        labels=names,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
    )
    ax.set_ylabel("C-Index")
    ax.set_title("Per-Fold C-Index Distribution")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_fold_boxplot.png"), dpi=150)
    plt.close()
    print(f"[PLOT] Saved per_fold_boxplot.png")


def generate_learning_curve(
    checkpoint_data: List[Tuple[int, dict]],
    output_dir: str = "plots",
):
    """Plot learning curve (C-Index vs. number of training patients seen).

    Args:
        checkpoint_data: list of ``(n_patients_seen, eval_results)`` tuples.
        output_dir:      where to save.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    xs = [d[0] for d in checkpoint_data]
    ys = [d[1]["mean_c_index"] for d in checkpoint_data]
    errs = [d[1]["std_c_index"] for d in checkpoint_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=4, linewidth=2)
    ax.set_xlabel("Training Patients Seen")
    ax.set_ylabel("C-Index")
    ax.set_title("Agent Learning Curve")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curve.png"), dpi=150)
    plt.close()
    print(f"[PLOT] Saved learning_curve.png")


def generate_acquisition_pattern(
    results: dict,
    method_name: str,
    output_dir: str = "plots",
):
    """Stacked bar chart of modality acquisition frequencies.

    Args:
        results: output of ``evaluate_policy`` (single fold or merged).
        method_name: for the plot title.
        output_dir:  save directory.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    counts = results["acquisition_counts"]
    n = results["n_patients"]
    freqs = counts / n

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    ax.bar(MODALITY_NAMES, freqs, color=colors)
    ax.set_ylabel("Acquisition Frequency")
    ax.set_title(f"Modality Acquisition Pattern — {method_name}")
    ax.set_ylim(0, 1.1)
    for i, (freq, name) in enumerate(zip(freqs, MODALITY_NAMES)):
        ax.text(i, freq + 0.02, f"{freq:.2f}", ha="center", fontsize=10)
    plt.tight_layout()
    safe_name = method_name.replace(" ", "_").lower()
    plt.savefig(
        os.path.join(output_dir, f"acquisition_{safe_name}.png"), dpi=150
    )
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Results I/O
# ═══════════════════════════════════════════════════════════════════════════
def save_results(results: dict, path: str):
    """Save results to JSON (numpy arrays → lists)."""

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    serialisable = json.loads(json.dumps(results, default=_convert))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"[EVAL] Results saved to {path}")


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Evaluate & compare methods")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing per-method JSON result files",
    )
    parser.add_argument(
        "--generate_plots",
        action="store_true",
        help="Generate comparison plots",
    )
    parser.add_argument(
        "--compute_stats",
        action="store_true",
        help="Compute pairwise statistical tests",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Reference method for pairwise comparison",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples for CI",
    )
    args = parser.parse_args()

    # Load all result files
    all_results = {}
    if os.path.isdir(args.results_dir):
        for fname in sorted(os.listdir(args.results_dir)):
            if fname.endswith(".json"):
                name = fname.replace(".json", "")
                all_results[name] = load_results(
                    os.path.join(args.results_dir, fname)
                )
                print(
                    f"  {name}: C-Index = "
                    f"{all_results[name].get('mean_c_index', '?'):.4f} ± "
                    f"{all_results[name].get('std_c_index', '?'):.4f}"
                )

    if not all_results:
        print("[EVAL] No results found.")
        return

    if args.generate_plots:
        generate_plots(all_results, output_dir=args.output_dir)

    if args.compute_stats and args.reference:
        ref = all_results.get(args.reference)
        if ref is None:
            print(f"[EVAL] Reference '{args.reference}' not found.")
            return
        print(f"\n--- Statistical comparison vs. {args.reference} ---")
        for name, res in all_results.items():
            if name == args.reference:
                continue
            st = compute_statistics(ref, res, n_bootstrap=args.n_bootstrap)
            sig = "*" if st.get("significant_p05") else ""
            print(
                f"  {name}: Δ={st.get('mean_diff', 0):+.4f}  "
                f"p={st.get('wilcoxon_p', 1):.4f}{sig}  "
                f"95%CI={st.get('bootstrap_ci_95', (0, 0))}"
            )


if __name__ == "__main__":
    main()
