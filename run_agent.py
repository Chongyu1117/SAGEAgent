"""
LLM Agent — Nested 5×5 CV Training and Evaluation.

For each (outer_fold, inner_fold):
    1. Load predictor + calibrated head from outer_N/inner_M/
    2. Build env with inner fold's complete training patients
    3. Train agent (experience accumulation + periodic reflection)
    4. Evaluate on outer fold's 34 test patients

Aggregation (training summary): per outer fold = mean C-index across inner folds.
Final evaluation (eval_sageagent.py): average risk across inner predictors, then C-index.

Usage:
    # Full 5×5 (single GPU, sequential)
    python run_agent.py --gpu 0

    # Single outer fold (for parallel execution across GPUs)
    python run_agent.py --outer_fold 1 --gpu 0

    # Ablations
    python run_agent.py --no_tools --tag no_tools
    python run_agent.py --no_episodic --tag no_episodic
    python run_agent.py --no_semantic --tag no_semantic
"""

import argparse
import json
import os
import pickle
import time

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
)
from agents.llm_agent import build_llm_agent
from agents.memory import EpisodicMemory, SemanticMemory
from agents.reflection import ReflectionModule
from evaluate import evaluate_policy, save_results
from utils.metrics import compute_c_index


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers (same pattern as run_baselines.py)
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
               budget, device, reward_coeff=2.0, process_reward_alpha=0.5,
               ucpr_beta=0.0, cost_weight=0.6, decision_signal="uncertainty",
               cavs_head=None, rdvs_head=None):
    """Build a ClinicalEnv with predictor and optional calibrated head."""
    env = ClinicalEnv(
        predictor=predictor,
        patient_features=features,
        patient_masks=masks,
        patient_events=events,
        patient_times=times,
        patient_names=names,
        budget=budget,
        reward_coeff=reward_coeff,
        process_reward_alpha=process_reward_alpha,
        ucpr_beta=ucpr_beta,
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


def _build_semantic_memory(no_semantic, max_active_rules=10, debug=False):
    """Build semantic memory (starts empty, populated by reflection)."""
    mem = SemanticMemory(enabled=not no_semantic, debug=debug)
    mem.MAX_ACTIVE_RULES = max_active_rules
    return mem


def _compute_unc_thresholds(env, complete_idx, debug=False):
    """Compute per-fold uncertainty thresholds from training data.

    Evaluates uncertainty at all 4 clinical-chain depths for complete
    patients, then returns quintile (p20/p40/p60/p80) thresholds for
    5-level categorization: VERY_LOW / LOW / MODERATE / HIGH / VERY_HIGH.

    5-level quintile design:
    - VERY_LOW (<p20): ~20% of states, strong signal to PREDICT
    - LOW (p20-p40): ~20%, lean toward PREDICT
    - MODERATE (p40-p60): only 20% ambiguous (vs 80% with old p10/p90)
    - HIGH (p60-p80): ~20%, lean toward ACQUIRE
    - VERY_HIGH (>p80): ~20%, strong signal to ACQUIRE

    """
    import torch
    N_MODALITIES = 4
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
    thresholds = {
        "p20": float(np.percentile(all_uncs, 20)),
        "p40": float(np.percentile(all_uncs, 40)),
        "p60": float(np.percentile(all_uncs, 60)),
        "p80": float(np.percentile(all_uncs, 80)),
    }

    if debug:
        print(f"  [UNC THRESH] quintiles: "
              f"VERY_LOW < {thresholds['p20']:.3f} < LOW < {thresholds['p40']:.3f} "
              f"< MOD < {thresholds['p60']:.3f} < HIGH < {thresholds['p80']:.3f} < VERY_HIGH "
              f"(N={len(all_uncs)})")

    return thresholds


def _compute_cavs_thresholds(cavs_head, predictor, env, complete_idx,
                             device="cuda", debug=False):
    """Compute per-fold CAVS value thresholds from training data.

    Evaluates CAVS head at 3 prefix masks (D, DR, DRP) on complete
    training patients, then returns quintile (p20/p40/p60/p80) thresholds.
    DRPG is excluded (oracle depth — always label=0).
    """
    import torch
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
    thresholds = {
        "p20": float(np.percentile(all_values, 20)),
        "p40": float(np.percentile(all_values, 40)),
        "p60": float(np.percentile(all_values, 60)),
        "p80": float(np.percentile(all_values, 80)),
    }

    if debug:
        print(f"  [CAVS THRESH] quintiles: "
              f"VERY_LOW < {thresholds['p20']:.3f} < LOW < {thresholds['p40']:.3f} "
              f"< MOD < {thresholds['p60']:.3f} < HIGH < {thresholds['p80']:.3f} < VERY_HIGH "
              f"(N={len(all_values)})")

    return thresholds


def _compute_rdvs_thresholds(rdvs_head, predictor, env, complete_idx,
                             device="cuda", debug=False):
    """Compute per-fold RDVS value thresholds from training data.

    Same pattern as _compute_cavs_thresholds but using RDVS head.
    Evaluates RDVS head at 3 prefix masks (D, DR, DRP) on complete
    training patients, then returns quintile (p20/p40/p60/p80) thresholds.
    """
    import torch
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
    thresholds = {
        "p20": float(np.percentile(all_values, 20)),
        "p40": float(np.percentile(all_values, 40)),
        "p60": float(np.percentile(all_values, 60)),
        "p80": float(np.percentile(all_values, 80)),
    }

    if debug:
        print(f"  [RDVS THRESH] quintiles: "
              f"VERY_LOW < {thresholds['p20']:.3f} < LOW < {thresholds['p40']:.3f} "
              f"< MOD < {thresholds['p60']:.3f} < HIGH < {thresholds['p80']:.3f} < VERY_HIGH "
              f"(N={len(all_values)})")

    return thresholds


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


def _compute_reward_thresholds(env, complete_idx, debug=False):
    """Compute per-stage hypothetical terminal reward thresholds.

    For each complete patient at each prefix depth (D, DR, DRP), computes
    the terminal reward they WOULD get if they stopped at that stage:
        terminal = reward_coeff * ((1 - uncertainty) - cost_weight * burden)

    Per-stage median = "good outcome" threshold for outcome tracking.
    This accounts for both prediction quality AND burden cost.

    Returns:
        dict {"after_demographics": float, "after_radiology": float,
              "after_pathology": float}
    """
    prefix_masks = [
        np.array([1, 0, 0, 0], dtype=np.float32),  # after_demographics
        np.array([1, 1, 0, 0], dtype=np.float32),  # after_radiology
        np.array([1, 1, 1, 0], dtype=np.float32),  # after_pathology
    ]
    stage_names = ["after_demographics", "after_radiology", "after_pathology"]
    stage_rewards = {s: [] for s in stage_names}

    reward_coeff = env.reward_coeff      # 2.0
    cost_weight = env.cost_weight        # 0.6

    for pidx in complete_idx:
        for mask, stage in zip(prefix_masks, stage_names):
            state = env.reset(int(pidx), initial_mask=mask)
            unc = state["uncertainty"]
            burden = state["total_burden"]
            quality = 1.0 - unc
            terminal = reward_coeff * (quality - cost_weight * burden)
            stage_rewards[stage].append(terminal)

    thresholds = {}
    for stage in stage_names:
        vals = np.array(stage_rewards[stage])
        thresholds[stage] = float(np.median(vals))

    if debug:
        print(f"  [REWARD THRESH] per-stage medians: "
              f"demo={thresholds['after_demographics']:.3f}, "
              f"rad={thresholds['after_radiology']:.3f}, "
              f"path={thresholds['after_pathology']:.3f} "
              f"(N={len(complete_idx)} patients)")

    return thresholds


def _load_llm(model_name, device):
    """Load LLM model and tokenizer once (shared across all folds)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[AGENT] Loading LLM: {model_name} on {device} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device)
    model.eval()
    print(f"[AGENT] LLM loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════
# Experience accumulation (training loop for one fold)
# ═══════════════════════════════════════════════════════════════════════════
def run_experience_accumulation(
    agent,
    env,
    patient_indices,
    episodic_memory,
    reflection_module=None,
    n_starts_per_patient=1,
    reflect_every=50,
    checkpoint_at=None,
    reward_thresholds=None,
    save_dir=None,
    debug=False,
):
    """Run the agent through training patients, accumulating experience.

    Args:
        agent:                LLMAgent instance.
        env:                  ClinicalEnv (training patients loaded).
        patient_indices:      list of patient indices to train on.
        episodic_memory:      EpisodicMemory (shared with agent).
        reflection_module:    ReflectionModule (optional).
        n_starts_per_patient: random initial subsets per patient.
        reflect_every:        reflection frequency (every N patients).
        checkpoint_at:        set of patient counts to save checkpoints.
        reward_thresholds:   per-stage prediction quality thresholds (from
                              _compute_reward_thresholds).
        save_dir:             where to save checkpoints.
        debug:                verbose.

    Returns:
        dict with training stats.
    """
    checkpoint_at = set(checkpoint_at or [])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    n_total = len(patient_indices)
    stats = {
        "n_patients": 0,
        "n_episodes": 0,
        "total_reward": 0.0,
        "avg_burden": [],
        "avg_n_modalities": [],
    }

    # Per-episode training curve log (JSONL)
    curve_path = os.path.join(save_dir, "training_curve.jsonl") if save_dir else None
    curve_file = open(curve_path, "w") if curve_path else None

    rng = np.random.RandomState(42)
    t_start = time.time()

    for count, patient_idx in enumerate(patient_indices, 1):
        for start_i in range(n_starts_per_patient):
            # Demographics is always pre-acquired (chart review, burden=0.03).
            # Clinically, every patient starts with demographics — it's not a
            # test to "decide" about.  The real sequential decision starts at
            # Radiology (MRI).
            available = env.patient_masks[patient_idx]
            initial_mask = np.zeros_like(available)
            initial_mask[0] = 1.0  # Demographics always acquired

            # For diversity (start_i > 0), randomly extend the prefix
            # beyond demographics: [D,R], [D,R,P], etc.
            if start_i > 0:
                chain = [0, 1, 2, 3]
                max_prefix = 0
                for mod_idx in chain:
                    if available[mod_idx] > 0.5:
                        max_prefix += 1
                    else:
                        break
                if max_prefix > 1:  # already have D, extend to D+R, D+R+P, …
                    n_init = rng.randint(1, max_prefix + 1)  # 1..max_prefix
                    for k in range(n_init):
                        initial_mask[chain[k]] = 1.0

            # Run episode — collect step-level entries for episodic memory
            state = env.reset(patient_idx, initial_mask=initial_mask)
            step_uncertainties = [state["uncertainty"]]
            step_cavs_values = []  # CAVS values at each decision point
            episode_reward = 0.0
            done = False
            step_entries = []
            step_decisions = []  # (state_dict, action) at every step for outcome tracking

            while not done:
                # Snapshot state at decision point
                step_embedding = state["embedding"].copy()
                step_mask = state["mask"].copy()
                step_unc = state["uncertainty"]

                # Track decision state for outcome tracking (all steps)
                step_decisions.append((dict(state), None))  # action filled below
                action = agent.select_action(state, env)
                step_decisions[-1] = (step_decisions[-1][0], action)

                # Capture CAVS value from agent's tool call (available for
                # ALL steps — ACQUIRE and PREDICT — since agent always calls
                # CAVS tool in _call_tools). This is essential for reflection
                # to learn "low CAVS → predict" rules.
                agent_cavs = getattr(agent, "_last_cavs_result", None)
                step_cavs_val = (agent_cavs["acquisition_value"]
                                 if agent_cavs else None)

                state, reward, done, info = env.step(action)
                step_uncertainties.append(state["uncertainty"])
                episode_reward += reward

                if step_cavs_val is not None:
                    step_cavs_values.append(step_cavs_val)

                step_entries.append({
                    "embedding": step_embedding,
                    "mask_at_decision": step_mask.tolist(),
                    "action_taken": action,
                    "reward": reward,
                    "unc_before": step_unc,
                    "unc_after": state["uncertainty"],
                    "cavs_value": step_cavs_val,
                })

            # Episode-level context
            summary = env.get_episode_summary()
            prediction_quality = float(
                np.exp(-abs(state["risk_score"] - summary["oracle_risk"])))
            outcome = "deceased" if summary["event"] > 0.5 else "censored"

            # Store step-level entries in FAISS (for retrieval)
            for entry in step_entries:
                entry["episode_reward"] = episode_reward
                entry["acquired_mask"] = summary["acquired_mask"].tolist()
                entry["outcome"] = outcome
                entry["patient_name"] = summary["patient_name"]
                entry["oracle_risk"] = summary["oracle_risk"]
                entry["prediction_quality"] = prediction_quality
                episodic_memory.add_episode(entry)

            # Store full episode (for reflection module)
            episodic_memory.add_full_episode({
                "embedding": state["embedding"],
                "acquired_mask": summary["acquired_mask"].tolist(),
                "n_acquired": summary["n_acquired"],
                "total_burden": summary["total_burden"],
                "risk_score": state["risk_score"],
                "oracle_risk": summary["oracle_risk"],
                "event": summary["event"],
                "time": summary["time"],
                "outcome": outcome,
                "patient_name": summary["patient_name"],
                "trajectory": [
                    {"action": s["action"], "reward": s["reward"]}
                    for s in summary["trajectory"]
                ],
                "episode_reward": episode_reward,
                "step_uncertainties": step_uncertainties,
                "step_cavs_values": step_cavs_values,
                "prediction_quality": prediction_quality,
                "initial_mask": initial_mask.tolist(),
            })

            stats["n_episodes"] += 1
            stats["total_reward"] += episode_reward
            stats["avg_burden"].append(summary["total_burden"])
            stats["avg_n_modalities"].append(summary["n_acquired"])

            # Outcome tracking: update semantic memory rule counters
            # Uses per-stage reward thresholds (accounts for burden) and
            # tracks ALL decision steps (not just the last one)
            if (reward_thresholds is not None
                    and step_decisions
                    and hasattr(agent, "semantic_memory")
                    and agent.semantic_memory is not None):
                # Terminal reward: same formula as env
                terminal_unc = state["uncertainty"]
                terminal_quality = 1.0 - terminal_unc
                terminal_burden = summary["total_burden"]
                terminal_reward = env.reward_coeff * (
                    terminal_quality - env.cost_weight * terminal_burden)
                agent.semantic_memory.track_episode_outcome(
                    step_decisions=step_decisions,
                    terminal_reward=terminal_reward,
                    reward_thresholds=reward_thresholds,
                    current_episode=stats["n_episodes"],
                )

            # Write per-episode training curve entry
            if curve_file:
                actions_taken = [s["action"] for s in summary["trajectory"]]
                curve_entry = {
                    "episode": stats["n_episodes"],
                    "patient": count,
                    "start": start_i,
                    "reward": round(episode_reward, 4),
                    "burden": round(summary["total_burden"], 3),
                    "n_mods": summary["n_acquired"],
                    "quality": round(prediction_quality, 4),
                    "actions": actions_taken,
                    "cavs_values": [round(v, 4) for v in step_cavs_values],
                    "unc_start": round(step_uncertainties[0], 4),
                    "unc_end": round(step_uncertainties[-1], 4),
                }
                curve_file.write(json.dumps(curve_entry) + "\n")
                curve_file.flush()

        stats["n_patients"] = count

        # Progress logging
        if count % 20 == 0 or debug:
            elapsed = time.time() - t_start
            avg_r = stats["total_reward"] / max(stats["n_episodes"], 1)
            avg_b = np.mean(stats["avg_burden"][-50:])
            avg_m = np.mean(stats["avg_n_modalities"][-50:])
            print(
                f"    patient {count}/{n_total} "
                f"(eps={stats['n_episodes']}, avg_r={avg_r:.3f}, "
                f"burden={avg_b:.3f}, mods={avg_m:.1f}, {elapsed:.0f}s)")

        # Self-reflection
        if reflection_module and count % reflect_every == 0:
            # Decay stale rules before reflection opens slots for new patterns
            if hasattr(agent, "semantic_memory") and agent.semantic_memory is not None:
                agent.semantic_memory.decay_stale_rules(stats["n_episodes"])
            ref_result = reflection_module.reflect(n_recent=reflect_every)
            n_active = len(reflection_module.semantic_memory.get_active_rules())
            if debug:
                print(
                    f"    Reflection: "
                    f"+{ref_result.get('new_rules_added', 0)} "
                    f"-{ref_result.get('new_rules_discarded', 0)} "
                    f"upd={ref_result.get('rules_updated', 0)} "
                    f"dep={ref_result.get('rules_deprecated', 0)} "
                    f"active={n_active}")
            # Log reflection event to training curve
            if curve_file:
                active_rules = reflection_module.semantic_memory.get_active_rules()
                ref_entry = {
                    "event": "reflection",
                    "patient": count,
                    "episode": stats["n_episodes"],
                    "new_added": ref_result.get("new_rules_added", 0),
                    "discarded": ref_result.get("new_rules_discarded", 0),
                    "updated": ref_result.get("rules_updated", 0),
                    "deprecated": ref_result.get("rules_deprecated", 0),
                    "n_active_rules": n_active,
                    "rules": [
                        {"id": r["id"],
                         "stage": r.get("stage", ""),
                         "direction": r.get("direction", ""),
                         "pattern": r.get("pattern", ""),
                         "action": r.get("action_guidance", ""),
                         "confidence": r.get("confidence", 0),
                         "effectiveness": r.get("effectiveness"),
                         "n_applied": r.get("n_applied", 0)}
                        for r in active_rules
                    ],
                }
                curve_file.write(json.dumps(ref_entry) + "\n")
                curve_file.flush()

        # Checkpoint
        if save_dir and count in checkpoint_at:
            episodic_memory.save(
                os.path.join(save_dir, f"episodic_at_{count}.json"))
            if hasattr(agent, "semantic_memory") and agent.semantic_memory:
                agent.semantic_memory.save(
                    os.path.join(save_dir, f"semantic_at_{count}.json"))

    stats["total_time"] = time.time() - t_start
    if curve_file:
        curve_file.close()
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Main nested 5×5 CV loop
# ═══════════════════════════════════════════════════════════════════════════
def run_agent_nested(
    config,
    splits_dir,
    checkpoint_dir,
    n_outer=5,
    n_inner=5,
    outer_fold=None,
    budget=1.0,
    reward_coeff=2.0,
    process_reward_alpha=0.5,
    ucpr_beta=0.0,
    cost_weight=0.6,
    decision_signal="uncertainty",
    device="cuda",
    llm_device=None,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    prompt_template="prompts/decision_prompt.txt",
    # ablation flags
    no_tools=False,
    no_episodic=False,
    no_semantic=False,
    # training
    n_starts=3,
    reflect_every=10,
    checkpoint_at=None,
    n_retrieval=3,
    # configurable parameters
    max_active_rules=10,
    reward_weight=0.3,
    max_ep_per_reflection=10,
    n_train_patients=None,
    start_inner=1,
    # concordance influence
    use_concordance_influence=False,
    # output
    save_dir="results/agent_nested",
    debug=False,
):
    """Train and evaluate LLM agent with true nested 5×5 CV.

    For each (outer, inner):
        - Load predictor + calibrated head
        - Build env with inner fold's complete training patients
        - Train agent (experience accumulation + reflection)
        - Evaluate on outer test set (34 complete patients)

    Training summary: per outer fold = mean C-index across inner folds.
    Final evaluation uses eval_sageagent.py (average risk, then C-index).
    """
    os.makedirs(save_dir, exist_ok=True)
    modality_keys = config["data"]["modality_keys"]
    llm_dev = llm_device or device

    # Pre-load LLM once (shared across all folds)
    llm_model, tokenizer = _load_llm(model_name, llm_dev)

    # Which outer folds to run
    outer_range = (
        [outer_fold] if outer_fold else list(range(1, n_outer + 1)))

    # {outer: [inner_results]}
    outer_inner_results = {}

    for outer in outer_range:
        print(f"\n{'='*60}")
        print(f"  LLM AGENT — OUTER FOLD {outer}/{n_outer}")
        print(f"{'='*60}")

        # Load agent.pkl
        agent_pkl_path = os.path.join(
            splits_dir, f"outer_{outer}", "agent.pkl")
        with open(agent_pkl_path, "rb") as f:
            agent_data = pickle.load(f)

        # Test data (same across all inner folds)
        test_data = agent_data["cv_splits"][1]["test"]
        te_features, te_masks, te_events, te_times, te_names, _ = (
            aggregate_patches_to_patients(
                test_data, modality_keys=modality_keys))
        complete_test_idx = np.where(
            te_masks.sum(axis=1) >= N_MODALITIES - 0.5)[0]
        if len(complete_test_idx) == 0:
            print(f"  No complete test patients, skipping")
            continue
        print(f"  {len(complete_test_idx)} complete test patients")

        for inner in range(1, n_inner + 1):
            if inner < start_inner:
                # Try to load existing result for skipped fold
                skip_dir = os.path.join(
                    save_dir, f"outer{outer}_inner{inner}")
                skip_eval = os.path.join(skip_dir, "eval_result.json")
                if os.path.exists(skip_eval):
                    with open(skip_eval) as f:
                        skip_result = json.load(f)
                    outer_inner_results.setdefault(outer, []).append(
                        skip_result)
                    print(f"\n  --- Inner fold {inner}/{n_inner} --- "
                          f"LOADED (C={skip_result['c_index']:.4f})")
                else:
                    print(f"\n  --- Inner fold {inner}/{n_inner} --- "
                          f"SKIPPED (no saved result)")
                continue
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

            # Training data: inner fold's train patients
            train_data = agent_data["cv_splits"][inner]["train"]
            tr_features, tr_masks, tr_events, tr_times, tr_names, _ = (
                aggregate_patches_to_patients(
                    train_data, modality_keys=modality_keys))

            # Complete patients for experience accumulation
            tr_complete_idx = np.where(
                tr_masks.sum(axis=1) >= N_MODALITIES - 0.5)[0]
            print(f"  Train: {len(tr_names)} total, "
                  f"{len(tr_complete_idx)} complete")

            # Build training env (with all patients for FAISS richness)
            train_env = _build_env(
                predictor, cal_head, tr_features, tr_masks,
                tr_events, tr_times, tr_names, budget, device,
                reward_coeff=reward_coeff,
                process_reward_alpha=process_reward_alpha,
                ucpr_beta=ucpr_beta,
                cost_weight=cost_weight,
                decision_signal=decision_signal,
                cavs_head=cavs_head,
                rdvs_head=rdvs_head)

            # Compute per-fold uncertainty percentile thresholds (quintile dict)
            unc_thresholds = _compute_unc_thresholds(
                train_env, tr_complete_idx, debug=debug)

            # Compute CAVS thresholds if needed
            cavs_thresholds = None
            if decision_signal == "cavs" and cavs_head is not None:
                cavs_thresholds = _compute_cavs_thresholds(
                    cavs_head, predictor, train_env, tr_complete_idx,
                    device=device, debug=debug)

            # Compute RDVS thresholds if needed
            rdvs_thresholds = None
            if decision_signal == "risk_delta" and rdvs_head is not None:
                rdvs_thresholds = _compute_rdvs_thresholds(
                    rdvs_head, predictor, train_env, tr_complete_idx,
                    device=device, debug=debug)

            # Compute per-stage prediction quality thresholds for outcome tracking
            reward_thresholds = _compute_reward_thresholds(
                train_env, tr_complete_idx, debug=debug)

            # Fresh memory per (outer, inner) fold
            episodic_memory = EpisodicMemory(
                reward_weight=reward_weight, debug=debug)
            semantic_memory = _build_semantic_memory(
                no_semantic, max_active_rules=max_active_rules, debug=debug)

            # Build agent (shared LLM, fresh tools + memory)
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
                n_retrieval=n_retrieval,
                use_tools=not no_tools,
                use_episodic=not no_episodic,
                use_semantic=not no_semantic,
                use_concordance_influence=use_concordance_influence,
                model=llm_model,
                tokenizer=tokenizer,
                debug=debug,
            )

            # Inject per-fold uncertainty thresholds (5-level quintiles)
            agent.set_unc_thresholds(thresholds=unc_thresholds)

            # Inject CAVS thresholds if needed
            if cavs_thresholds is not None:
                agent.set_cavs_thresholds(thresholds=cavs_thresholds)

            # Inject RDVS thresholds if needed
            if rdvs_thresholds is not None:
                agent.set_rdvs_thresholds(thresholds=rdvs_thresholds)

            # Compute and inject concordance influence thresholds if needed
            if use_concordance_influence:
                conc_tool = agent.tools.get("concordance_influence")
                conc_thresholds = _compute_concordance_thresholds(
                    conc_tool, train_env, tr_complete_idx, debug=debug)
                agent.set_concordance_thresholds(thresholds=conc_thresholds)

            # Build reflection module (shared LLM) — disabled when no_semantic
            reflection_module = None
            if not no_semantic:
                reflection_module = ReflectionModule(
                    model_name=model_name,
                    semantic_memory=semantic_memory,
                    episodic_memory=episodic_memory,
                    max_episodes_per_reflection=max_ep_per_reflection,
                    decision_signal=decision_signal,
                    device=llm_dev,
                    debug=debug,
                )
                reflection_module.set_model(llm_model, tokenizer)
                reflection_module.set_unc_thresholds(thresholds=unc_thresholds)
                if cavs_thresholds is not None:
                    reflection_module.set_cavs_thresholds(thresholds=cavs_thresholds)
                if rdvs_thresholds is not None:
                    reflection_module.set_rdvs_thresholds(thresholds=rdvs_thresholds)

            # Experience accumulation on complete training patients
            train_patient_list = tr_complete_idx.tolist()
            if n_train_patients is not None and n_train_patients < len(train_patient_list):
                train_patient_list = train_patient_list[:n_train_patients]
                print(f"  [TRAIN] Limited to {n_train_patients}/{len(tr_complete_idx)} "
                      f"complete patients")
            fold_save_dir = os.path.join(
                save_dir, f"outer{outer}_inner{inner}")
            train_stats = run_experience_accumulation(
                agent=agent,
                env=train_env,
                patient_indices=train_patient_list,
                episodic_memory=episodic_memory,
                reflection_module=reflection_module,
                n_starts_per_patient=n_starts,
                reflect_every=reflect_every,
                checkpoint_at=checkpoint_at,
                reward_thresholds=reward_thresholds,
                save_dir=fold_save_dir,
                debug=debug,
            )

            # Save final memory state
            os.makedirs(fold_save_dir, exist_ok=True)
            episodic_memory.save(
                os.path.join(fold_save_dir, "episodic_final.json"))
            semantic_memory.save(
                os.path.join(fold_save_dir, "semantic_final.json"))
            if reflection_module:
                reflection_module.save_log(
                    os.path.join(fold_save_dir, "reflection_log.json"))

            # Evaluate on outer test set
            test_env = _build_env(
                predictor, cal_head, te_features, te_masks,
                te_events, te_times, te_names, budget, device,
                reward_coeff=reward_coeff,
                process_reward_alpha=0.0,
                ucpr_beta=0.0,
                cost_weight=cost_weight,
                decision_signal=decision_signal,
                cavs_head=cavs_head,
                rdvs_head=rdvs_head)  # cost_weight active at test too
            result = evaluate_policy(
                agent, test_env, complete_test_idx.tolist(), debug=debug)

            inner_result = {
                "outer": outer,
                "inner": inner,
                "c_index": result["c_index"],
                "avg_burden": result["avg_burden"],
                "avg_n_modalities": result["avg_n_modalities"],
                "n_complete": len(complete_test_idx),
                "acquisition_counts": result["acquisition_counts"].tolist(),
                "train_stats": {
                    "n_patients": train_stats["n_patients"],
                    "n_episodes": train_stats["n_episodes"],
                    "total_time": train_stats.get("total_time", 0),
                },
            }
            outer_inner_results.setdefault(outer, []).append(inner_result)

            print(
                f"  O{outer}/I{inner}: C={result['c_index']:.4f}  "
                f"burden={result['avg_burden']:.3f}  "
                f"mods={result['avg_n_modalities']:.2f}")

            # Save per-fold result
            save_results(
                inner_result,
                os.path.join(fold_save_dir, "eval_result.json"))

        del agent_data

    # ── Aggregate ─────────────────────────────────────────────────────────
    if not outer_inner_results:
        print("[AGENT] No results to aggregate.")
        return {}

    fold_results = []
    for outer in sorted(outer_inner_results.keys()):
        inner_list = outer_inner_results[outer]
        inner_cs = [r["c_index"] for r in inner_list]
        # Sum acquisition counts across inner folds
        total_acq = np.sum(
            [r["acquisition_counts"] for r in inner_list], axis=0)
        fold_results.append({
            "fold": outer,
            "c_index": float(np.mean(inner_cs)),
            "inner_c_indices": inner_cs,
            "avg_burden": float(np.mean(
                [r["avg_burden"] for r in inner_list])),
            "avg_n_modalities": float(np.mean(
                [r["avg_n_modalities"] for r in inner_list])),
            "n_complete": inner_list[0]["n_complete"],
            "acquisition_counts_total": total_acq.tolist(),
        })

    per_fold_c = [r["c_index"] for r in fold_results]
    final_results = {
        "method": "llm_agent",
        "fold_results": fold_results,
        "mean_c_index": float(np.mean(per_fold_c)),
        "std_c_index": float(np.std(per_fold_c)),
        "n_total_patients": sum(r["n_complete"] for r in fold_results),
        "avg_burden": float(np.mean(
            [r["avg_burden"] for r in fold_results])),
        "avg_n_modalities": float(np.mean(
            [r["avg_n_modalities"] for r in fold_results])),
    }

    print(f"\n{'='*60}")
    print(
        f"  LLM AGENT FINAL: C = {np.mean(per_fold_c):.4f} ± "
        f"{np.std(per_fold_c):.4f}  "
        f"burden = {final_results['avg_burden']:.3f}")
    print(f"{'='*60}")

    save_results(final_results, os.path.join(save_dir, "llm_agent.json"))
    return final_results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="LLM Agent — Nested 5×5 CV Training & Evaluation")

    # CV structure
    parser.add_argument("--n_outer", type=int, default=5)
    parser.add_argument("--n_inner", type=int, default=5)
    parser.add_argument("--outer_fold", type=int, default=None,
                        help="Run only this outer fold (for parallel)")
    parser.add_argument("--start_inner", type=int, default=1,
                        help="Start from this inner fold (skip earlier folds, for resume)")

    # Paths
    parser.add_argument("--config", type=str,
                        default="configs/default_config.yaml")
    parser.add_argument("--splits_dir", type=str, default=None,
                        help="Directory with outer_N/ splits (default: config)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Predictor checkpoint dir (default: config)")

    # LLM
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt_template", type=str,
                        default="prompts/decision_prompt.txt")

    # Agent
    parser.add_argument("--n_retrieval", type=int, default=3)
    parser.add_argument("--n_starts", type=int, default=3,
                        help="Random starts per patient (>=3 for exploration diversity)")
    parser.add_argument("--reflect_every", type=int, default=10)
    parser.add_argument("--reward_coeff", type=float, default=2.0,
                        help="Terminal reward multiplier (higher = quality > cost)")
    parser.add_argument("--process_alpha", type=float, default=0.5,
                        help="Weight for uncertainty reduction bonus (0=off)")
    parser.add_argument("--ucpr_beta", type=float, default=0.0,
                        help="UCPR weight in terminal reward (0=off, default)")
    parser.add_argument("--cost_weight", type=float, default=0.6,
                        help="Burden penalty weight in terminal reward "
                             "(LA-CDM normalization, 0=off by default)")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated patient counts for checkpoints")
    parser.add_argument("--max_active_rules", type=int, default=10,
                        help="Max active semantic rules (oldest deprecated when exceeded)")
    parser.add_argument("--reward_weight", type=float, default=0.3,
                        help="Weight for reward in episodic retrieval ranking")
    parser.add_argument("--max_ep_per_reflection", type=int, default=10,
                        help="Max episodes per category in reflection prompt")
    parser.add_argument("--n_train_patients", type=int, default=None,
                        help="Limit training to first N complete patients "
                             "(default: all). Set to ~40 for fast runs.")
    parser.add_argument("--decision_signal", type=str, default="uncertainty",
                        choices=["uncertainty", "cavs", "risk_delta"],
                        help="Decision signal: 'uncertainty' (default), "
                             "'cavs' (concordance-aware value scorer), or "
                             "'risk_delta' (risk-delta value scorer)")
    parser.add_argument("--use_concordance_influence", action="store_true",
                        help="Add concordance influence tool (supplementary signal, "
                             "requires tools enabled)")

    # Ablation flags
    parser.add_argument("--no_tools", action="store_true",
                        help="Disable tools (uncertainty, predictor, retriever, VoI)")
    parser.add_argument("--no_episodic", action="store_true",
                        help="Disable episodic memory (FAISS retrieval)")
    parser.add_argument("--no_semantic", action="store_true",
                        help="Disable semantic memory AND reflection "
                             "(no rules learned or shown)")

    # Hardware
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--llm_gpu", type=int, default=None,
                        help="GPU for LLM (default: same as --gpu)")
    parser.add_argument("--budget", type=float, default=1.0)

    # Output
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="full")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────
    if args.use_concordance_influence and args.no_tools:
        raise SystemExit(
            "ERROR: --use_concordance_influence and --no_tools are incompatible")

    # ── Config ────────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    llm_device = (
        f"cuda:{args.llm_gpu}" if args.llm_gpu is not None else device)

    # Resolve paths from config (CLI overrides take priority)
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

    checkpoint_at = None
    if args.checkpoints:
        checkpoint_at = [
            int(x) for x in args.checkpoints.split(",") if x.strip()]

    # Build save directory with ablation tag
    tag = args.tag
    if args.no_tools:
        tag += "_no_tools"
    if args.no_episodic:
        tag += "_no_episodic"
    if args.no_semantic:
        tag += "_no_semantic"
    save_dir = args.save_dir or f"results/agent_{tag}"

    # ── Print config ──────────────────────────────────────────────────────
    print(f"[AGENT] Config:      {args.config}")
    print(f"[AGENT] Splits:      {args.splits_dir}")
    print(f"[AGENT] Checkpoints: {args.checkpoint_dir}")
    print(f"[AGENT] Device:      {device}  LLM: {llm_device}")
    print(f"[AGENT] Decision:    {args.decision_signal}")
    print(f"[AGENT] Tag:         {tag}")
    print(f"[AGENT] Save:        {save_dir}")
    print(f"[AGENT] Outer folds: "
          f"{args.outer_fold or 'all'} / {args.n_outer}")
    print(f"[AGENT] Inner folds: {args.n_inner}")

    # ── Run ───────────────────────────────────────────────────────────────
    results = run_agent_nested(
        config=config,
        splits_dir=args.splits_dir,
        checkpoint_dir=args.checkpoint_dir,
        n_outer=args.n_outer,
        n_inner=args.n_inner,
        outer_fold=args.outer_fold,
        budget=args.budget,
        reward_coeff=args.reward_coeff,
        process_reward_alpha=args.process_alpha,
        ucpr_beta=args.ucpr_beta,
        decision_signal=args.decision_signal,
        device=device,
        llm_device=llm_device,
        model_name=args.model_name,
        prompt_template=args.prompt_template,
        no_tools=args.no_tools,
        no_episodic=args.no_episodic,
        no_semantic=args.no_semantic,
        n_starts=args.n_starts,
        reflect_every=args.reflect_every,
        checkpoint_at=checkpoint_at,
        n_retrieval=args.n_retrieval,
        max_active_rules=args.max_active_rules,
        cost_weight=args.cost_weight,
        reward_weight=args.reward_weight,
        max_ep_per_reflection=args.max_ep_per_reflection,
        n_train_patients=args.n_train_patients,
        start_inner=args.start_inner,
        use_concordance_influence=args.use_concordance_influence,
        save_dir=save_dir,
        debug=args.debug,
    )

    if results:
        print(
            f"\n[AGENT] Done. C = {results['mean_c_index']:.4f} ± "
            f"{results['std_c_index']:.4f}  "
            f"burden = {results['avg_burden']:.3f}")


if __name__ == "__main__":
    main()
