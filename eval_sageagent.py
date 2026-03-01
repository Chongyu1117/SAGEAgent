"""
SAGEAgent Evaluation — Majority Vote + Full CoT Capture + Plots.

Runs ALL 5 inner folds per outer fold and uses majority vote (≥3/5) to
determine per-patient acquisition decisions. Per outer fold, risk scores
are averaged across 5 inner predictors per patient, then one C-index is
computed. We report mean ± std of 5 outer fold C-indices.

Ablation flags control which agentic components are active:
  --no_tools:    disable tool calling (uncertainty, predictor, retriever, VoI)
  --no_episodic: disable episodic memory (FAISS retrieval of similar cases)
  --no_semantic: disable semantic memory (learned decision rules)

Usage:
    # Full SAGEAgent with majority vote + CoT + plots
    python eval_sageagent.py --gpu 1 --generate_plots

    # Base LLM zero-shot (same prompt, no agentic components)
    python eval_sageagent.py --no_tools --no_episodic --no_semantic --gpu 2

    # Single outer fold (for testing)
    python eval_sageagent.py --outer_fold 1 --gpu 1 --debug
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from utils.data_loader import aggregate_patches_to_patients
from envs.clinical_env import (
    ClinicalEnv,
    load_predictor,
    load_calibrated_head_for_fold,
    load_cavs_head_for_fold,
    load_rdvs_head_for_fold,
    N_MODALITIES,
    MODALITY_NAMES,
    ACTION_NAMES,
    PREDICT,
)
from utils.metrics import compute_c_index

# Clinical burden per modality (same as ClinicalEnv.DEFAULT_BURDEN)
BURDEN = {0: 0.03, 1: 0.14, 2: 0.53, 3: 0.30}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers (same pattern as run_agent.py / run_baselines.py)
# ═══════════════════════════════════════════════════════════════════════════
def _load_inner_fold_models(checkpoint_dir, config, outer, inner, device):
    """Load predictor + calibrated head for a specific (outer, inner) fold."""
    ckpt_path = os.path.join(
        checkpoint_dir, f"outer_{outer}", f"inner_{inner}", "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"  WARNING: {ckpt_path} not found, skipping")
        return None, None
    predictor = load_predictor(ckpt_path, config, device=device)
    cal_head = load_calibrated_head_for_fold(
        outer, inner_fold=inner, checkpoint_dir=checkpoint_dir, device=device)
    return predictor, cal_head


def _build_env(predictor, cal_head, features, masks, events, times, names,
               budget, device, cost_weight=0.6, decision_signal="uncertainty",
               cavs_head=None, rdvs_head=None):
    """Build a test ClinicalEnv (no process/UCPR reward)."""
    env = ClinicalEnv(
        predictor=predictor,
        patient_features=features,
        patient_masks=masks,
        patient_events=events,
        patient_times=times,
        patient_names=names,
        budget=budget,
        reward_coeff=2.0,
        process_reward_alpha=0.0,
        ucpr_beta=0.0,
        cost_weight=cost_weight,
        decision_signal=decision_signal,
        device=device,
        debug=False,
    )
    if cal_head is not None:
        env.set_calibrated_head(cal_head)
    if cavs_head is not None:
        env.set_cavs_head(cavs_head)
    if rdvs_head is not None:
        env.set_rdvs_head(rdvs_head)
    return env


def _compute_unc_thresholds(env, complete_idx):
    """Compute per-fold uncertainty quintile thresholds from training data."""
    prefix_masks = [
        np.array([1, 0, 0, 0], dtype=np.float32),
        np.array([1, 1, 0, 0], dtype=np.float32),
        np.array([1, 1, 1, 0], dtype=np.float32),
        np.array([1, 1, 1, 1], dtype=np.float32),
    ]
    all_uncs = []
    for pidx in complete_idx:
        for mask in prefix_masks:
            state = env.reset(int(pidx), initial_mask=mask)
            all_uncs.append(state["uncertainty"])
    all_uncs = np.array(all_uncs)
    return {
        "p20": float(np.percentile(all_uncs, 20)),
        "p40": float(np.percentile(all_uncs, 40)),
        "p60": float(np.percentile(all_uncs, 60)),
        "p80": float(np.percentile(all_uncs, 80)),
    }


def _compute_cavs_thresholds(cavs_head, predictor, env, complete_idx,
                             device="cuda"):
    """Compute per-fold CAVS value thresholds from training data."""
    prefix_masks = [
        np.array([1, 0, 0, 0], dtype=np.float32),
        np.array([1, 1, 0, 0], dtype=np.float32),
        np.array([1, 1, 1, 0], dtype=np.float32),
    ]
    all_values = []
    with torch.no_grad():
        for pidx in complete_idx:
            for mask in prefix_masks:
                features = env.patient_features[int(pidx)].copy()
                for j in range(4):
                    if mask[j] < 0.5:
                        features[j] = 0.0
                feat_t = torch.from_numpy(features).float().unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(device)
                emb = predictor.get_embedding(feat_t, mask_t)
                value = cavs_head(emb, mask_t).cpu().item()
                all_values.append(value)
    all_values = np.array(all_values)
    return {
        "p20": float(np.percentile(all_values, 20)),
        "p40": float(np.percentile(all_values, 40)),
        "p60": float(np.percentile(all_values, 60)),
        "p80": float(np.percentile(all_values, 80)),
    }


def _compute_rdvs_thresholds(rdvs_head, predictor, env, complete_idx,
                             device="cuda"):
    """Compute per-fold RDVS value thresholds from training data."""
    prefix_masks = [
        np.array([1, 0, 0, 0], dtype=np.float32),
        np.array([1, 1, 0, 0], dtype=np.float32),
        np.array([1, 1, 1, 0], dtype=np.float32),
    ]
    all_values = []
    with torch.no_grad():
        for pidx in complete_idx:
            for mask in prefix_masks:
                features = env.patient_features[int(pidx)].copy()
                for j in range(4):
                    if mask[j] < 0.5:
                        features[j] = 0.0
                feat_t = torch.from_numpy(features).float().unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(device)
                emb = predictor.get_embedding(feat_t, mask_t)
                value = rdvs_head(emb, mask_t).cpu().item()
                all_values.append(value)
    all_values = np.array(all_values)
    return {
        "p20": float(np.percentile(all_values, 20)),
        "p40": float(np.percentile(all_values, 40)),
        "p60": float(np.percentile(all_values, 60)),
        "p80": float(np.percentile(all_values, 80)),
    }


def _compute_concordance_thresholds(conc_tool, env, complete_idx, debug=False):
    """Compute per-fold concordance influence thresholds from training data.

    Evaluates concordance sensitivity at 3 prefix depths (D, DR, DRP) on
    complete training patients using the existing ConcordanceInfluenceTool
    instance (reuses pre-computed training risks).

    Returns tercile dict {p33, p67}.
    """
    prefix_masks = [
        np.array([1, 0, 0, 0], dtype=np.float32),
        np.array([1, 1, 0, 0], dtype=np.float32),
        np.array([1, 1, 1, 0], dtype=np.float32),
    ]
    all_sensitivities = []
    for pidx in complete_idx:
        for mask in prefix_masks:
            features = env.patient_features[int(pidx)].copy()
            for j in range(4):
                if mask[j] < 0.5:
                    features[j] = 0.0
            result = conc_tool(features, mask)
            all_sensitivities.append(result["sensitivity"])

    all_sensitivities = np.array(all_sensitivities)
    thresholds = {
        "p33": float(np.percentile(all_sensitivities, 33)),
        "p67": float(np.percentile(all_sensitivities, 67)),
    }

    if debug:
        print(f"  [CONC THRESH] terciles: "
              f"LOW < {thresholds['p33']:.4f} < MOD < {thresholds['p67']:.4f} < HIGH "
              f"(N={len(all_sensitivities)})")

    return thresholds


def _load_llm(model_name, device):
    """Load LLM model and tokenizer once (bfloat16 for Qwen)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Always use bfloat16 for Qwen — float16 causes CUDA assertion in sampling
    dtype = torch.bfloat16

    print(f"[EVAL] Loading LLM: {model_name} on {device} (dtype={dtype}) ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype,
        attn_implementation="sdpa",
        device_map=device)
    model.eval()
    print(f"[EVAL] LLM loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════
# Per-fold evaluation (LLMAgent with CoT)
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_fold(
    agent,
    env: ClinicalEnv,
    test_indices: List[int],
    patient_names: Optional[List[str]] = None,
    capture_cot: bool = True,
    debug: bool = False,
) -> dict:
    """Evaluate LLMAgent on test patients, optionally capturing full CoT."""
    risk_scores, events, times, burdens = [], [], [], []
    uncertainties, n_modalities_list = [], []
    acquisition_counts = np.zeros(N_MODALITIES, dtype=int)
    per_patient_masks = []  # (n_patients, N_MODALITIES)
    patient_traces = []

    for i, patient_idx in enumerate(test_indices):
        agent.reset()
        initial_mask = np.zeros(N_MODALITIES, dtype=np.float32)
        initial_mask[0] = 1.0
        state = env.reset(patient_idx, initial_mask=initial_mask)
        pname = patient_names[patient_idx] if patient_names else f"patient_{patient_idx}"

        steps = []
        done = False
        step_num = 0

        while not done:
            if capture_cot:
                action, reasoning, context = agent.decide_with_context(
                    state, env)
                step_info = {
                    "step": step_num,
                    "mask_before": state["mask"].tolist(),
                    "uncertainty_before": float(state["uncertainty"]),
                    "risk_score": float(state["risk_score"]),
                    "total_burden_before": float(state["total_burden"]),
                    "valid_actions": [ACTION_NAMES[a] for a in state["valid_actions"]],
                    "action_taken": ACTION_NAMES[action],
                    "reasoning": reasoning,
                    "tool_results": context["tool_results"],
                    "episodic_context": context["episodic_context"],
                    "semantic_context": context["semantic_context"],
                }
            else:
                action = agent.select_action(state, env)
                step_info = None

            state, reward, done, info = env.step(action)
            if capture_cot and step_info:
                step_info["uncertainty_after"] = float(state["uncertainty"])
                step_info["total_burden_after"] = float(state["total_burden"])
                step_info["reward"] = float(reward)
                steps.append(step_info)
            step_num += 1

        summary = env.get_episode_summary()
        risk_scores.append(state["risk_score"])
        events.append(summary["event"])
        times.append(summary["time"])
        burdens.append(summary["total_burden"])
        uncertainties.append(float(state.get("uncertainty", 0.0)))
        n_modalities_list.append(summary["n_acquired"])
        acq_mask = summary["acquired_mask"].astype(int)
        acquisition_counts += acq_mask
        per_patient_masks.append(acq_mask.tolist())

        if capture_cot:
            patient_traces.append({
                "patient_name": pname,
                "patient_idx": int(patient_idx),
                "event": float(summary["event"]),
                "time": float(summary["time"]),
                "oracle_risk": float(summary["oracle_risk"]),
                "final_risk": float(state["risk_score"]),
                "final_uncertainty": float(state.get("uncertainty", 0.0)),
                "total_burden": float(summary["total_burden"]),
                "n_acquired": int(summary["n_acquired"]),
                "acquired_mask": acq_mask.tolist(),
                "steps": steps,
            })

        if debug or (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(test_indices)}] {pname}: "
                  f"mods={summary['n_acquired']} burden={summary['total_burden']:.3f}")

    c_index = compute_c_index(
        np.array(risk_scores), np.array(events), np.array(times))

    return {
        "c_index": float(c_index),
        "avg_burden": float(np.mean(burdens)),
        "avg_n_modalities": float(np.mean(n_modalities_list)),
        "n_patients": len(test_indices),
        "acquisition_counts": acquisition_counts.tolist(),
        "per_patient_masks": per_patient_masks,
        "burdens": [float(b) for b in burdens],
        "n_modalities_acquired": n_modalities_list,
        "patient_traces": patient_traces,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Majority vote aggregation
# ═══════════════════════════════════════════════════════════════════════════
def majority_vote(inner_results: List[dict], n_inner: int = 5) -> dict:
    """Majority vote across inner folds for one outer fold.

    For each patient, if ≥ ceil(n_inner/2) inner models acquired a modality,
    the majority vote says "acquired".

    Returns:
        dict with voted acquisition counts, burden, n_modalities.
    """
    n_patients = inner_results[0]["n_patients"]
    n_voted = len(inner_results)
    threshold = (n_voted + 1) // 2  # ≥3 for 5 folds, ≥2 for 3 folds

    # Stack per-patient masks: (n_inner_voted, n_patients, N_MODALITIES)
    all_masks = np.array([r["per_patient_masks"] for r in inner_results])

    # Vote: sum across inner folds, threshold
    vote_sum = all_masks.sum(axis=0)  # (n_patients, N_MODALITIES)
    voted_masks = (vote_sum >= threshold).astype(int)  # (n_patients, N_MODALITIES)

    # Compute voted stats
    voted_acq_counts = voted_masks.sum(axis=0)  # (N_MODALITIES,)
    voted_burdens = []
    voted_n_mods = []
    for i in range(n_patients):
        burden = sum(BURDEN[j] for j in range(N_MODALITIES) if voted_masks[i, j])
        voted_burdens.append(burden)
        voted_n_mods.append(int(voted_masks[i].sum()))

    return {
        "voted_masks": voted_masks.tolist(),
        "acquisition_counts": voted_acq_counts.tolist(),
        "avg_burden": float(np.mean(voted_burdens)),
        "avg_n_modalities": float(np.mean(voted_n_mods)),
        "n_patients": n_patients,
        "burdens": voted_burdens,
        "n_modalities_acquired": voted_n_mods,
        "vote_threshold": threshold,
        "n_inner_voted": n_voted,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════
def generate_all_plots(outer_votes, outer_c_indices, all_traces,
                       method_name, output_dir):
    """Generate comprehensive plots for the paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    outer_folds = sorted(outer_votes.keys())

    # --- 1. Per-outer-fold C-index bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(outer_folds))
    cs = [outer_c_indices[o] for o in outer_folds]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]
    bars = ax.bar(x, cs, color=colors[:len(outer_folds)])
    mean_c = np.mean(cs)
    ax.axhline(y=mean_c, color="red", linestyle="--", alpha=0.7,
               label=f"Mean: {mean_c:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Outer {o}" for o in outer_folds])
    ax.set_ylabel("C-Index")
    ax.set_title(f"{method_name} — Per-Fold C-Index (Nested 5x5 CV)")
    ax.set_ylim(0.65, 0.95)
    ax.legend()
    for bar, c in zip(bars, cs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{c:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_fold_cindex.png"), dpi=150)
    plt.close()
    print(f"[PLOT] Saved per_fold_cindex.png")

    # --- 2. Majority-voted acquisition pattern ---
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.15
    mod_colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    mod_labels = ["Demographics", "Radiology", "Pathology", "Genomics"]

    for j, (mod_name, color) in enumerate(zip(mod_labels, mod_colors)):
        rates = []
        for o in outer_folds:
            v = outer_votes[o]
            rates.append(v["acquisition_counts"][j] / v["n_patients"] * 100)
        ax.bar(x + j * width, rates, width, label=mod_name, color=color)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"Outer {o}" for o in outer_folds])
    ax.set_ylabel("Acquisition Rate (%, majority vote)")
    ax.set_title(f"{method_name} — Modality Acquisition (Majority Vote)")
    ax.set_ylim(0, 110)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "acquisition_majority_vote.png"), dpi=150)
    plt.close()
    print(f"[PLOT] Saved acquisition_majority_vote.png")

    # --- 3. Burden distribution (majority-voted) ---
    all_burdens = []
    for o in outer_folds:
        all_burdens.extend(outer_votes[o]["burdens"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_burdens, bins=15, color="#4c72b0", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(all_burdens), color="red", linestyle="--",
               label=f"Mean: {np.mean(all_burdens):.3f}")
    ax.set_xlabel("Clinical Burden (majority vote)")
    ax.set_ylabel("Count")
    ax.set_title(f"{method_name} — Per-Patient Burden Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "burden_distribution.png"), dpi=150)
    plt.close()
    print(f"[PLOT] Saved burden_distribution.png")

    # --- 4. Number of modalities acquired ---
    all_n_mods = []
    for o in outer_folds:
        all_n_mods.extend(outer_votes[o]["n_modalities_acquired"])

    fig, ax = plt.subplots(figsize=(6, 5))
    unique_mods = sorted(set(all_n_mods))
    mod_counts = [all_n_mods.count(m) for m in unique_mods]
    ax.bar(unique_mods, mod_counts, color="#55a868", edgecolor="white")
    ax.set_xlabel("Number of Modalities Acquired")
    ax.set_ylabel("Count (170 patients)")
    ax.set_title(f"{method_name} — Stopping Point Distribution")
    ax.set_xticks(unique_mods)
    for m, c in zip(unique_mods, mod_counts):
        ax.text(m, c + 0.5, str(c), ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stopping_distribution.png"), dpi=150)
    plt.close()
    print(f"[PLOT] Saved stopping_distribution.png")

    # --- 5. Example patient CoT ---
    if all_traces:
        sorted_by_burden = sorted(all_traces, key=lambda t: t["total_burden"])
        early_stop = sorted_by_burden[0]
        full_acq = sorted_by_burden[-1]

        for label, trace in [("early_stop", early_stop), ("full_acquire", full_acq)]:
            path = os.path.join(output_dir, f"example_cot_{label}.txt")
            with open(path, "w") as f:
                f.write(f"Patient: {trace['patient_name']}\n")
                f.write(f"Event: {trace['event']}, Time: {trace['time']:.1f}\n")
                f.write(f"Final acquired: {trace['acquired_mask']}\n")
                f.write(f"Total burden: {trace['total_burden']:.3f}\n")
                f.write(f"N modalities: {trace['n_acquired']}\n")
                f.write(f"Final uncertainty: {trace['final_uncertainty']:.4f}\n")
                f.write(f"Final risk: {trace['final_risk']:.4f}\n")
                f.write(f"Oracle risk: {trace['oracle_risk']:.4f}\n")
                f.write("=" * 60 + "\n\n")
                for step in trace["steps"]:
                    f.write(f"--- Step {step['step']} ---\n")
                    f.write(f"Mask: {step['mask_before']}\n")
                    f.write(f"Uncertainty: {step['uncertainty_before']:.4f}\n")
                    f.write(f"Valid actions: {step['valid_actions']}\n")
                    f.write(f"Action: {step['action_taken']}\n")
                    f.write(f"Reward: {step['reward']:.4f}\n")
                    f.write(f"\n--- Reasoning ---\n{step['reasoning']}\n\n")
            print(f"[PLOT] Saved example_cot_{label}.txt")


# ═══════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════
def run_evaluation(
    config,
    splits_dir,
    checkpoint_dir,
    model_name,
    memory_dir="results/agent_training",
    n_outer=5,
    n_inner=5,
    outer_fold=None,
    budget=1.0,
    device="cuda",
    llm_device=None,
    prompt_template="prompts/decision_prompt.txt",
    save_dir="results/eval_sageagent",
    do_plots=False,
    no_tools=False,
    no_episodic=False,
    no_semantic=False,
    use_concordance_influence=False,
    cost_weight=0.6,
    decision_signal="uncertainty",
    debug=False,
):
    """Run evaluation with majority vote across all 5 inner folds.

    Args:
        no_tools:    disable tools (uncertainty, predictor, retriever, VoI).
        no_episodic: disable episodic memory (FAISS retrieval).
        no_semantic: disable semantic memory (learned rules).
    """
    os.makedirs(save_dir, exist_ok=True)
    modality_keys = config["data"]["modality_keys"]
    llm_dev = llm_device or device

    # Pre-load LLM once
    llm_model, tokenizer = _load_llm(model_name, llm_dev)

    outer_range = [outer_fold] if outer_fold else list(range(1, n_outer + 1))

    # Results storage
    outer_inner_results = defaultdict(list)  # {outer: [inner_results]}
    outer_votes = {}                          # {outer: majority_vote_dict}
    outer_c_indices = {}                      # {outer: mean_c_index}
    all_cot_traces = []                       # CoT traces (from inner=1)

    for outer in outer_range:
        print(f"\n{'='*60}")
        print(f"  EVAL — OUTER FOLD {outer}/{n_outer}")
        print(f"{'='*60}")

        # Load agent.pkl
        agent_pkl_path = os.path.join(splits_dir, f"outer_{outer}", "agent.pkl")
        with open(agent_pkl_path, "rb") as f:
            agent_data = pickle.load(f)

        # Test data (same across all inner folds)
        test_data = agent_data["cv_splits"][1]["test"]
        te_features, te_masks, te_events, te_times, te_names, _ = (
            aggregate_patches_to_patients(test_data, modality_keys=modality_keys))
        complete_test_idx = np.where(
            te_masks.sum(axis=1) >= N_MODALITIES - 0.5)[0]
        if len(complete_test_idx) == 0:
            print(f"  No complete test patients, skipping")
            continue
        print(f"  {len(complete_test_idx)} complete test patients")

        for inner in range(1, n_inner + 1):
            print(f"\n  --- Inner fold {inner}/{n_inner} ---")

            # Load predictor + calibrated head
            predictor, cal_head = _load_inner_fold_models(
                checkpoint_dir, config, outer, inner, device)
            if predictor is None:
                continue

            # Load CAVS head if decision_signal == "cavs"
            cavs_head = None
            if decision_signal == "cavs":
                cavs_head = load_cavs_head_for_fold(
                    outer, inner, checkpoint_dir=checkpoint_dir, device=device)

            # Load RDVS head if decision_signal == "risk_delta"
            rdvs_head = None
            if decision_signal == "risk_delta":
                rdvs_head = load_rdvs_head_for_fold(
                    outer, inner, checkpoint_dir=checkpoint_dir, device=device)

            # Training data (for FAISS index + unc thresholds)
            train_data = agent_data["cv_splits"][inner]["train"]
            tr_features, tr_masks, tr_events, tr_times, tr_names, _ = (
                aggregate_patches_to_patients(
                    train_data, modality_keys=modality_keys))
            tr_complete_idx = np.where(
                tr_masks.sum(axis=1) >= N_MODALITIES - 0.5)[0]

            # Build training env for unc thresholds
            train_env = _build_env(
                predictor, cal_head, tr_features, tr_masks,
                tr_events, tr_times, tr_names, budget, device,
                cost_weight=cost_weight,
                decision_signal=decision_signal,
                cavs_head=cavs_head,
                rdvs_head=rdvs_head)
            unc_thresholds = _compute_unc_thresholds(train_env, tr_complete_idx)

            # Compute CAVS thresholds if needed
            cavs_thresholds = None
            if decision_signal == "cavs" and cavs_head is not None:
                cavs_thresholds = _compute_cavs_thresholds(
                    cavs_head, predictor, train_env, tr_complete_idx,
                    device=device)

            # Compute RDVS thresholds if needed
            rdvs_thresholds = None
            if decision_signal == "risk_delta" and rdvs_head is not None:
                rdvs_thresholds = _compute_rdvs_thresholds(
                    rdvs_head, predictor, train_env, tr_complete_idx,
                    device=device)

            t0 = time.time()

            from agents.llm_agent import build_llm_agent
            from agents.memory import EpisodicMemory, SemanticMemory

            use_episodic = not no_episodic
            use_semantic = not no_semantic
            use_tools = not no_tools

            # Load saved memory (skip loading if ablated)
            mem_fold_dir = os.path.join(
                memory_dir, f"outer{outer}_inner{inner}")
            episodic_memory = EpisodicMemory(debug=debug)
            semantic_memory = SemanticMemory(
                enabled=use_semantic, debug=debug)
            semantic_memory.MAX_ACTIVE_RULES = 10

            if use_episodic:
                ep_path = os.path.join(mem_fold_dir, "episodic_final.json")
                if os.path.exists(ep_path):
                    episodic_memory.load(ep_path)
                    print(f"  Loaded episodic: {len(episodic_memory)} entries")
                else:
                    print(f"  WARNING: {ep_path} not found")
            else:
                print(f"  Episodic memory DISABLED (ablation)")

            if use_semantic:
                sem_path = os.path.join(mem_fold_dir, "semantic_final.json")
                if os.path.exists(sem_path):
                    semantic_memory.load(sem_path)
                    print(f"  Loaded semantic: {len(semantic_memory.rules)} rules")
                else:
                    print(f"  WARNING: {sem_path} not found")
            else:
                print(f"  Semantic memory DISABLED (ablation)")

            if not use_tools:
                print(f"  Tools DISABLED (ablation)")

            # Build agent
            agent = build_llm_agent(
                predictor=predictor,
                calibrated_head=cal_head,
                cavs_head=cavs_head,
                rdvs_head=rdvs_head,
                decision_signal=decision_signal,
                train_features=tr_features,
                train_masks=tr_masks,
                train_events=tr_events,
                train_times=tr_times,
                train_names=tr_names,
                model_name=model_name,
                prompt_template_path=prompt_template,
                episodic_memory=episodic_memory,
                semantic_memory=semantic_memory,
                device=device,
                llm_device=llm_dev,
                n_retrieval=3,
                use_tools=use_tools,
                use_episodic=use_episodic,
                use_semantic=use_semantic,
                use_concordance_influence=use_concordance_influence,
                model=llm_model,
                tokenizer=tokenizer,
                debug=debug,
            )
            agent.set_unc_thresholds(thresholds=unc_thresholds)
            if cavs_thresholds is not None:
                agent.set_cavs_thresholds(thresholds=cavs_thresholds)
            if rdvs_thresholds is not None:
                agent.set_rdvs_thresholds(thresholds=rdvs_thresholds)

            # Compute and inject concordance influence thresholds if needed
            if use_concordance_influence:
                conc_tool = agent.tools.get("concordance_influence")
                conc_thresholds = _compute_concordance_thresholds(
                    conc_tool, train_env, tr_complete_idx, debug=debug)
                agent.set_concordance_thresholds(thresholds=conc_thresholds)

            # Build test env
            test_env = _build_env(
                predictor, cal_head, te_features, te_masks,
                te_events, te_times, te_names, budget, device,
                cost_weight=cost_weight,
                decision_signal=decision_signal,
                cavs_head=cavs_head,
                rdvs_head=rdvs_head)

            # Capture CoT only for inner=1 (representative)
            capture_cot = (inner == 1)
            result = evaluate_fold(
                agent, test_env, complete_test_idx.tolist(),
                patient_names=te_names,
                capture_cot=capture_cot, debug=debug)

            if capture_cot and result["patient_traces"]:
                all_cot_traces.extend(result["patient_traces"])
                # Save CoT traces
                fold_dir = os.path.join(save_dir, f"outer{outer}_inner{inner}")
                os.makedirs(fold_dir, exist_ok=True)
                with open(os.path.join(fold_dir, "patient_traces.json"), "w") as f:
                    json.dump(result["patient_traces"], f, indent=2)
                print(f"  Saved CoT to {fold_dir}/patient_traces.json")

            elapsed = time.time() - t0
            acq = result["acquisition_counts"]
            acq_str = " ".join(
                f"{MODALITY_NAMES[i]}={acq[i]}" for i in range(N_MODALITIES))
            print(f"  O{outer}/I{inner}: C={result['c_index']:.4f}  "
                  f"burden={result['avg_burden']:.3f}  "
                  f"mods={result['avg_n_modalities']:.2f}  "
                  f"| {acq_str}  ({elapsed:.1f}s)")

            # Save per-fold summary
            summary = {k: v for k, v in result.items()
                       if k not in ("patient_traces",)}
            summary["outer"] = outer
            summary["inner"] = inner
            summary["eval_time"] = elapsed
            outer_inner_results[outer].append(summary)

            fold_dir = os.path.join(save_dir, f"outer{outer}_inner{inner}")
            os.makedirs(fold_dir, exist_ok=True)
            with open(os.path.join(fold_dir, "eval_result.json"), "w") as f:
                json.dump(summary, f, indent=2)

            # Cleanup
            del predictor, cal_head, train_env
            torch.cuda.empty_cache()

        del agent_data

        # Majority vote for this outer fold
        if outer in outer_inner_results and len(outer_inner_results[outer]) > 0:
            vote = majority_vote(outer_inner_results[outer],
                                 n_inner=len(outer_inner_results[outer]))
            outer_votes[outer] = vote

            # ── Correct nested CV: average risk across inner predictors ──
            # Re-load all inner predictors and compute risk with voted masks,
            # then average per-patient risk scores and compute ONE C-index.
            voted_masks_np = np.array(vote["voted_masks"], dtype=np.float32)
            test_feat = te_features[complete_test_idx]
            test_events = te_events[complete_test_idx]
            test_times = te_times[complete_test_idx]

            risk_accum = np.zeros(len(complete_test_idx))
            n_models = 0
            for inner in range(1, n_inner + 1):
                predictor_i, _ = _load_inner_fold_models(
                    checkpoint_dir, config, outer, inner, device)
                if predictor_i is None:
                    continue
                n_models += 1
                feat_t = torch.tensor(test_feat, dtype=torch.float32).to(device)
                mask_t = torch.tensor(voted_masks_np, dtype=torch.float32).to(device)
                with torch.no_grad():
                    risk = predictor_i.get_risk_score(feat_t, mask_t)
                risk_accum += risk.cpu().numpy().flatten()
                del predictor_i
                torch.cuda.empty_cache()

            if n_models > 0:
                avg_risk = risk_accum / n_models
                outer_c_indices[outer] = float(
                    compute_c_index(avg_risk, test_events, test_times))
            else:
                # Fallback: average inner C-indices (should not happen)
                inner_cs = [r["c_index"] for r in outer_inner_results[outer]]
                outer_c_indices[outer] = float(np.mean(inner_cs))

            acq = vote["acquisition_counts"]
            print(f"\n  OUTER {outer} MAJORITY VOTE: "
                  f"C={outer_c_indices[outer]:.4f}  "
                  f"burden={vote['avg_burden']:.3f}  "
                  f"mods={vote['avg_n_modalities']:.2f}")
            print(f"  Voted acquisition: "
                  f"demo={acq[0]} rad={acq[1]} path={acq[2]} gen={acq[3]}"
                  f" (out of {vote['n_patients']})")

    # ── Final aggregation ────────────────────────────────────────────────
    if not outer_votes:
        print("[EVAL] No results to aggregate.")
        return {}

    all_cs = [outer_c_indices[o] for o in sorted(outer_votes.keys())]

    # Aggregate majority-voted counts across outer folds
    total_voted_acq = np.zeros(N_MODALITIES, dtype=int)
    total_patients = 0
    all_voted_burdens = []
    all_voted_n_mods = []
    for o in sorted(outer_votes.keys()):
        v = outer_votes[o]
        total_voted_acq += np.array(v["acquisition_counts"])
        total_patients += v["n_patients"]
        all_voted_burdens.extend(v["burdens"])
        all_voted_n_mods.extend(v["n_modalities_acquired"])

    # Build label reflecting ablation config
    method_label = "SAGEAgent"
    ablation_parts = []
    if no_tools:
        ablation_parts.append("no_tools")
    if no_episodic:
        ablation_parts.append("no_episodic")
    if no_semantic:
        ablation_parts.append("no_semantic")
    if ablation_parts:
        method_label += "_" + "_".join(ablation_parts)

    final = {
        "method": method_label,
        "model": model_name,
        "aggregation": "majority_vote",
        "ablation": {
            "no_tools": no_tools,
            "no_episodic": no_episodic,
            "no_semantic": no_semantic,
        },
        "mean_c_index": float(np.mean(all_cs)),
        "std_c_index": float(np.std(all_cs)),
        "per_outer_c_index": {
            str(o): outer_c_indices[o] for o in sorted(outer_votes.keys())},
        "n_total_patients": total_patients,
        # Majority-voted stats
        "voted_acquisition_counts": total_voted_acq.tolist(),
        "voted_avg_burden": float(np.mean(all_voted_burdens)),
        "voted_avg_n_modalities": float(np.mean(all_voted_n_mods)),
        "voted_per_outer": {
            str(o): {
                "acquisition_counts": outer_votes[o]["acquisition_counts"],
                "avg_burden": outer_votes[o]["avg_burden"],
                "avg_n_modalities": outer_votes[o]["avg_n_modalities"],
                "n_patients": outer_votes[o]["n_patients"],
            }
            for o in sorted(outer_votes.keys())
        },
    }

    print(f"\n{'='*60}")
    print(f"  {method_label} FINAL (majority vote)")
    print(f"  C = {final['mean_c_index']:.4f} ± {final['std_c_index']:.4f}")
    print(f"  Burden = {final['voted_avg_burden']:.3f}  "
          f"Mods = {final['voted_avg_n_modalities']:.2f}")
    print(f"  Acquisition (170 patients): "
          f"demo={total_voted_acq[0]} rad={total_voted_acq[1]} "
          f"path={total_voted_acq[2]} gen={total_voted_acq[3]}")
    print(f"{'='*60}")

    with open(os.path.join(save_dir, "aggregate.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(f"[EVAL] Saved {os.path.join(save_dir, 'aggregate.json')}")

    if do_plots:
        generate_all_plots(
            outer_votes, outer_c_indices, all_cot_traces,
            method_name=method_label,
            output_dir=os.path.join(save_dir, "plots"))

    return final


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="SAGEAgent Evaluation — Majority Vote + CoT + Plots")

    # Ablation flags
    parser.add_argument("--no_tools", action="store_true",
                        help="Disable tools (uncertainty, predictor, retriever, VoI)")
    parser.add_argument("--no_episodic", action="store_true",
                        help="Disable episodic memory (FAISS retrieval)")
    parser.add_argument("--no_semantic", action="store_true",
                        help="Disable semantic memory (learned rules)")

    # CV structure
    parser.add_argument("--n_outer", type=int, default=5)
    parser.add_argument("--n_inner", type=int, default=5)
    parser.add_argument("--outer_fold", type=int, default=None,
                        help="Run only this outer fold")

    # Paths
    parser.add_argument("--config", type=str,
                        default="configs/default_config.yaml")
    parser.add_argument("--splits_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--memory_dir", type=str,
                        default="results/agent_training",
                        help="Directory with saved memory from training")

    # LLM
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct")

    # Reward
    parser.add_argument("--cost_weight", type=float, default=0.6,
                        help="Burden penalty weight in terminal reward "
                             "(LA-CDM normalization, 0=off)")
    parser.add_argument("--decision_signal", type=str, default="uncertainty",
                        choices=["uncertainty", "cavs", "risk_delta"],
                        help="Decision signal: 'uncertainty' (default), "
                             "'cavs' (concordance-aware value scorer), or "
                             "'risk_delta' (risk-delta value scorer)")
    parser.add_argument("--use_concordance_influence", action="store_true",
                        help="Add concordance influence tool (supplementary signal, "
                             "requires tools enabled)")

    # Hardware
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--llm_gpu", type=int, default=None)
    parser.add_argument("--budget", type=float, default=1.0)

    # Output
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--generate_plots", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────
    if args.use_concordance_influence and args.no_tools:
        raise SystemExit(
            "ERROR: --use_concordance_influence and --no_tools are incompatible")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    llm_device = (
        f"cuda:{args.llm_gpu}" if args.llm_gpu is not None else device)

    data_root = config["data"]["data_root"]
    paths_cfg = config.get("paths", {})
    if args.splits_dir is None:
        args.splits_dir = os.path.join(
            data_root, paths_cfg.get("splits_dir", "splits"))
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(
            data_root,
            paths_cfg.get("predictor_checkpoint_dir",
                          "checkpoints/nested5x5"))

    if args.save_dir is None:
        parts = ["results", "eval_sageagent"]
        if args.no_tools:
            parts.append("no_tools")
        if args.no_episodic:
            parts.append("no_episodic")
        if args.no_semantic:
            parts.append("no_semantic")
        args.save_dir = "_".join(parts)

    ablations = []
    if args.no_tools:
        ablations.append("no_tools")
    if args.no_episodic:
        ablations.append("no_episodic")
    if args.no_semantic:
        ablations.append("no_semantic")

    print(f"[EVAL] Model:       {args.model_name}")
    print(f"[EVAL] Splits:      {args.splits_dir}")
    print(f"[EVAL] Checkpoints: {args.checkpoint_dir}")
    print(f"[EVAL] Memory:      {args.memory_dir}")
    print(f"[EVAL] Decision:    {args.decision_signal}")
    print(f"[EVAL] Ablations:   {', '.join(ablations) if ablations else 'none (full SAGEAgent)'}")
    print(f"[EVAL] Cost weight: {args.cost_weight}")
    print(f"[EVAL] Device:      {device}  LLM: {llm_device}")
    print(f"[EVAL] Outer folds: {args.outer_fold or 'all'} / {args.n_outer}")
    print(f"[EVAL] Aggregation: majority vote (all {args.n_inner} inner folds)")
    print(f"[EVAL] Save:        {args.save_dir}")

    results = run_evaluation(
        config=config,
        splits_dir=args.splits_dir,
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        memory_dir=args.memory_dir,
        n_outer=args.n_outer,
        n_inner=args.n_inner,
        outer_fold=args.outer_fold,
        budget=args.budget,
        device=device,
        llm_device=llm_device,
        save_dir=args.save_dir,
        do_plots=args.generate_plots,
        no_tools=args.no_tools,
        no_episodic=args.no_episodic,
        no_semantic=args.no_semantic,
        use_concordance_influence=args.use_concordance_influence,
        cost_weight=args.cost_weight,
        decision_signal=args.decision_signal,
        debug=args.debug,
    )

    if results:
        print(
            f"\n[EVAL] Done. C = {results['mean_c_index']:.4f} ± "
            f"{results['std_c_index']:.4f}  "
            f"burden = {results['voted_avg_burden']:.3f}")


if __name__ == "__main__":
    main()
