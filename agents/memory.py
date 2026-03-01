"""
Memory System — Episodic + Semantic memory for the LLM agent.

Episodic Memory:
    FAISS-indexed patient trajectories. Retrieves similar past cases
    to inform current decision-making (case-based reasoning).

Semantic Memory:
    Starts empty. Populated during training by the reflection module,
    which distills data-driven stopping heuristics from experience.
    No hand-coded rules — all knowledge is discovered from data.

Ablation:
    No semantic memory  →  empty, reflection disabled or rules not stored
    With reflection     →  learned rules accumulated during training
"""

import json
import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Episodic Memory
# ═══════════════════════════════════════════════════════════════════════════
class EpisodicMemory:
    """FAISS-backed episodic memory of past patient trajectories.

    Each episode stores an embedding, a trajectory summary, and outcome.
    Retrieval is by cosine similarity on normalised embeddings.

    Args:
        embedding_dim: dimension of patient embeddings (default 128).
        max_episodes:  capacity; oldest episodes evicted when full.
        debug:         verbose logging.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        max_episodes: int = 5000,
        reward_weight: float = 0.3,
        debug: bool = False,
    ):
        self.embedding_dim = embedding_dim
        self.max_episodes = max_episodes
        self.reward_weight = reward_weight
        self.debug = debug

        # Step-level entries (one per decision point) — indexed by FAISS
        self.episodes: List[dict] = []
        self._embeddings: List[np.ndarray] = []
        self._index = None  # lazy init
        self._dirty = True  # rebuild index when True

        # Full episode entries (one per episode) — used by reflection module
        self.full_episodes: List[dict] = []

    def add_episode(self, episode: dict):
        """Store a completed episode.

        Args:
            episode: dict with at least ``embedding`` (np array), plus any
                     metadata (``trajectory``, ``outcome``, ``patient_name``,
                     ``acquired_mask``, ``total_burden``, ``risk_score``, …).
        """
        emb = np.array(episode["embedding"], dtype=np.float32)
        self._embeddings.append(emb)
        self.episodes.append(episode)
        self._dirty = True

        # Evict oldest if over capacity
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)
            self._embeddings.pop(0)

        if self.debug and len(self.episodes) % 100 == 0:
            print(f"[EPISODIC] {len(self.episodes)} step entries stored")

    def add_full_episode(self, episode: dict):
        """Store a full episode (for reflection module).

        Args:
            episode: dict with trajectory, outcome, step_uncertainties, etc.
        """
        self.full_episodes.append(episode)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        reward_weight: Optional[float] = None,
        query_depth: Optional[int] = None,
    ) -> List[dict]:
        """Stage-matched retrieval: FAISS similarity → depth filter → re-rank.

        Phase 1: Fetch broad candidates by cosine similarity (FAISS).
        Phase 2: Stage-match filter — only keep entries whose
                 ``mask_at_decision`` depth matches ``query_depth``.
                 This ensures the agent sees decisions made at the SAME
                 acquisition stage (e.g. demo-rad), not a different one
                 (e.g. demo-rad-path).
        Phase 3: Reward-weighted re-ranking. Returns top-k by combined
                 score (similarity + reward_weight × normalised reward).

        Note: Outcome-diverse selection was removed because it
        amplified rare early-stop episodes and caused cascading feedback
        loops with early implementations.

        Args:
            query_embedding: (embedding_dim,) numpy array.
            k:               number of neighbours to return.
            reward_weight:   how much to weight normalised episode reward
                             relative to similarity (default 0.3).
            query_depth:     number of currently acquired modalities. If
                             provided, only return entries at this depth.

        Returns:
            list of episode dicts, ordered by descending combined score.
        """
        if reward_weight is None:
            reward_weight = self.reward_weight

        if len(self.episodes) == 0:
            return []

        # Phase 1: broad FAISS retrieval (fetch more to allow depth filtering)
        k_fetch = min(k * 10 if query_depth is not None else k * 3,
                      len(self.episodes))
        k_return = min(k, len(self.episodes))
        self._maybe_rebuild_index()

        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(query) + 1e-8
        query_normed = query / norm

        scores, indices = self._index.search(query_normed, k_fetch)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.episodes):
                continue
            ep = dict(self.episodes[idx])
            ep["retrieval_similarity"] = float(score)

            # Phase 2: stage-match filter by mask_at_decision depth
            if query_depth is not None:
                decision_mask = ep.get("mask_at_decision", [])
                ep_depth = sum(
                    1 for m in decision_mask
                    if isinstance(m, (int, float)) and m > 0.5)
                if ep_depth != query_depth:
                    continue

            candidates.append(ep)

        if not candidates:
            return []

        # Phase 3: reward-weighted re-ranking (pure top-k, no diversity tricks)
        rewards = np.array([
            c.get("episode_reward", c.get("reward", 0.0))
            for c in candidates
        ], dtype=np.float32)

        # Normalise rewards to [0, 1] within the candidate set
        r_min, r_max = rewards.min(), rewards.max()
        if r_max - r_min > 1e-8:
            norm_rewards = (rewards - r_min) / (r_max - r_min)
        else:
            norm_rewards = np.ones_like(rewards) * 0.5

        for i, c in enumerate(candidates):
            c["_combined_score"] = (
                c["retrieval_similarity"] + reward_weight * norm_rewards[i]
            )

        candidates.sort(key=lambda c: -c["_combined_score"])

        results = []
        for c in candidates[:k_return]:
            c.pop("_combined_score", None)
            results.append(c)
        return results

    def _maybe_rebuild_index(self):
        if not self._dirty and self._index is not None:
            return
        import faiss

        embs = np.stack(self._embeddings).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embs_normed = embs / norms

        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._index.add(embs_normed)
        self._dirty = False

    def __len__(self):
        return len(self.episodes)

    def format_for_prompt(self, episodes: List[dict], max_episodes: int = 3) -> str:
        """Format retrieved step-level entries for the LLM prompt.

        Each entry is a past decision point at a similar acquisition stage.
        Shows what action was taken, immediate reward, uncertainty change,
        and the episode-level outcome for context.
        """
        if not episodes:
            return "No similar past cases available."

        _action_names = {
            0: "PREDICT", 1: "ACQUIRE_DEMO", 2: "ACQUIRE_RAD",
            3: "ACQUIRE_PATH", 4: "ACQUIRE_GEN",
        }

        lines = ["=== Similar Past Decisions ==="]
        for i, ep in enumerate(episodes[:max_episodes]):
            sim = ep.get("retrieval_similarity", 0)
            mask_str = _mask_to_modalities(
                ep.get("mask_at_decision", ep.get("acquired_mask", [])))
            action = _action_names.get(ep.get("action_taken", -1), "unknown")
            reward = ep.get("episode_reward", ep.get("reward", 0))
            unc_before = ep.get("unc_before", None)
            unc_after = ep.get("unc_after", None)
            outcome = ep.get("outcome", "unknown")
            ep_reward = ep.get("episode_reward", None)
            final_mask_str = _mask_to_modalities(
                ep.get("acquired_mask", ep.get("final_mask", [])))

            if unc_before is not None and unc_after is not None:
                unc_str = f"{unc_before:.2f} → {unc_after:.2f}"
            else:
                unc_str = "n/a"

            ep_reward_str = f"{ep_reward:+.3f}" if ep_reward is not None else "n/a"

            lines.append(
                f"Case {i+1} (similarity={sim:.2f}):\n"
                f"  State: {mask_str or 'none'} acquired\n"
                f"  Decision: {action} → Reward: {reward:+.3f} | "
                f"Uncertainty: {unc_str}\n"
                f"  Episode: {final_mask_str} → PREDICT | "
                f"Outcome: {outcome} | Total reward: {ep_reward_str}"
            )
        return "\n".join(lines)

    def save(self, path: str):
        """Save step entries + full episodes to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        step_entries = []
        for ep, emb in zip(self.episodes, self._embeddings):
            entry = dict(ep)
            entry["embedding"] = emb.tolist()
            for k, v in entry.items():
                if isinstance(v, np.ndarray):
                    entry[k] = v.tolist()
            step_entries.append(entry)

        full_entries = []
        for ep in self.full_episodes:
            entry = dict(ep)
            for k, v in entry.items():
                if isinstance(v, np.ndarray):
                    entry[k] = v.tolist()
            full_entries.append(entry)

        data = {"step_entries": step_entries, "full_episodes": full_entries}
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[EPISODIC] Saved {len(step_entries)} steps, "
              f"{len(full_entries)} episodes to {path}")

    def load(self, path: str):
        """Load step entries + full episodes from JSON.

        Expects format: {step_entries: [...], full_episodes: [...]}.
        """
        with open(path) as f:
            data = json.load(f)
        self.episodes = []
        self._embeddings = []
        self.full_episodes = []

        if not isinstance(data, dict) or "step_entries" not in data:
            raise ValueError(
                f"Unexpected episodic memory format in {path}. "
                f"Expected dict with 'step_entries' key."
            )

        for entry in data["step_entries"]:
            if "embedding" not in entry:
                raise ValueError(
                    f"Step entry missing 'embedding' field in {path}."
                )
            if "mask_at_decision" not in entry:
                raise ValueError(
                    f"Step entry missing 'mask_at_decision' field in {path}."
                )
            emb = np.array(entry.pop("embedding"), dtype=np.float32)
            self._embeddings.append(emb)
            self.episodes.append(entry)
        self.full_episodes = data.get("full_episodes", [])

        self._dirty = True
        print(f"[EPISODIC] Loaded {len(self.episodes)} steps, "
              f"{len(self.full_episodes)} episodes from {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Semantic Memory — Learned Patterns from Reflection
# ═══════════════════════════════════════════════════════════════════════════
#
# Starts empty. During training, the reflection module analyses the agent's
# experience and distills multi-signal clinical patterns with outcome tracking.
#
# No hand-coded rules: clinical ordering is enforced by the environment's
# action mask; all semantic knowledge is discovered from data.
#
# Ablation:
#   --no_semantic    → disabled (rules cannot be added)
#   --no_reflection  → no reflection runs (memory stays empty)
#   default          → reflection populates rules during training
# ═══════════════════════════════════════════════════════════════════════════

RULE_TYPE_LEARNED = "learned"

# Stage names indexed by n_acquired modalities
_N_ACQUIRED_TO_STAGE = {
    1: "after_demographics",
    2: "after_radiology",
    3: "after_pathology",
}


def _infer_stage_from_condition(condition: str) -> str:
    """Infer stage from condition string."""
    cond_lower = condition.lower()
    for stage in ["after_demographics", "after_radiology", "after_pathology"]:
        if stage in cond_lower:
            return stage
    return "after_demographics"


def _infer_direction_from_guidance(guidance: str) -> str:
    """Infer direction from action_guidance string."""
    g = guidance.lower()
    if "predict" in g or "stop" in g:
        return "stop"
    return "acquire"


class SemanticMemory:
    """Data-driven semantic memory populated by reflection.

    Starts empty. Rules are added at runtime by the reflection module,
    which analyses past episodes and distills multi-signal clinical patterns
    with outcome tracking for continuous learning.

    Args:
        enabled:  if False, rules cannot be added (ablation: --no_semantic).
        debug:    verbose logging.
    """

    def __init__(
        self,
        enabled: bool = True,
        debug: bool = False,
    ):
        self.debug = debug
        self.enabled = enabled
        self.rules: List[dict] = []
        self._next_learned_id = 1

        if self.debug:
            print(f"[SEMANTIC] Init: empty ({'enabled' if enabled else 'disabled'})")

    # Maximum active learned rules. When exceeded, the worst-effectiveness
    # (or lowest-confidence) rule is deprecated before adding a new one.
    MAX_ACTIVE_RULES = 10

    def add_rule(self, rule: dict) -> str:
        """Add a new pattern discovered by reflection.

        Args:
            rule: dict with ``pattern`` (text), ``stage``, ``direction``,
                  ``supporting_evidence``, ``action_guidance``, ``confidence``.

        Returns:
            assigned rule ID, or empty string if disabled.
        """
        if not self.enabled:
            if self.debug:
                print("[SEMANTIC] Disabled, skipping add_rule")
            return ""

        # Cap: deprecate worst-effectiveness active rule if at limit
        active = [r for r in self.rules if r.get("active", False)]
        if len(active) >= self.MAX_ACTIVE_RULES:
            worst = min(active, key=lambda r: (
                r.get("effectiveness", r.get("confidence", 0)),
                -r.get("created_at", 0)))
            worst["active"] = False
            if self.debug:
                print(f"[SEMANTIC] Deprecated {worst['id']} (cap reached): "
                      f"{worst.get('pattern', worst.get('rule', ''))[:60]}")

        rule_id = f"learned_{self._next_learned_id}"
        self._next_learned_id += 1

        entry = {
            "id": rule_id,
            "rule_type": RULE_TYPE_LEARNED,
            # New multi-signal format
            "pattern": rule.get("pattern", rule.get("rule", "")),
            "stage": rule.get("stage", "after_demographics"),
            "direction": rule.get("direction", "acquire"),
            "supporting_evidence": rule.get("supporting_evidence", ""),
            "action_guidance": rule.get("action_guidance", ""),
            "confidence": min(0.8, rule.get("confidence", 0.5)),
            "source": rule.get("source", "reflection"),
            # Outcome tracking counters
            "n_applied": 0,
            "n_followed_good": 0,
            "n_followed_poor": 0,
            "n_violated_good": 0,
            "n_violated_poor": 0,
            "effectiveness": None,  # computed after sufficient data
            # Metadata
            "active": True,
            "created_at": time.time(),
            "last_applied_episode": 0,
        }
        self.rules.append(entry)

        if self.debug:
            print(f"[SEMANTIC] Added rule {rule_id} "
                  f"(stage={entry['stage']}, dir={entry['direction']}): "
                  f"{entry['pattern'][:60]}...")
        return rule_id

    def get_active_rules(self) -> List[dict]:
        """Return all currently active rules."""
        return [r for r in self.rules if r.get("active", False)]

    def retrieve_for_state(self, state: dict) -> List[dict]:
        """Stage-based retrieval from current environment state.

        Matches rules by acquisition stage only (no uncertainty/tool dependency).
        Sorts by effectiveness (if available), then confidence.
        """
        mask = state.get("mask", [])
        n_acquired = sum(1 for m in mask if m > 0.5)
        current_stage = _N_ACQUIRED_TO_STAGE.get(n_acquired)

        active = self.get_active_rules()
        if not active:
            return []

        # Filter by stage: match current stage, or rules with no stage
        matched = []
        for r in active:
            rule_stage = r.get("stage", "")
            if rule_stage == current_stage or not rule_stage:
                matched.append(r)

        # Sort by effectiveness (if computed), then confidence
        def _sort_key(r):
            eff = r.get("effectiveness")
            conf = r.get("confidence", 0)
            # Rules with effectiveness data sort first, then by value
            if eff is not None:
                return (1, eff, conf)
            return (0, conf, conf)

        matched.sort(key=_sort_key, reverse=True)
        return matched[:self.MAX_ACTIVE_RULES]

    def track_episode_outcome(
        self,
        step_decisions: list,
        terminal_reward: float,
        reward_thresholds: dict,
        current_episode: int = 0,
    ):
        """Track outcome for active rules at every decision stage in an episode.

        For each step's (state, action), finds matching rules at that stage,
        checks if the agent followed or violated, and crosses with per-stage
        outcome (terminal reward vs hypothetical stop reward threshold).

        Args:
            step_decisions:     list of (state_dict, action_int) for each step.
            terminal_reward:    terminal reward of the episode
                                (reward_coeff * (quality - cost_weight * burden)).
            reward_thresholds:  per-stage hypothetical stop reward thresholds
                                {"after_demographics": 0.96, "after_radiology": 1.04, ...}.
            current_episode:    current episode count (for staleness tracking).
        """
        if not self.enabled:
            return

        active = self.get_active_rules()
        if not active:
            return

        for state_at_decision, action_taken in step_decisions:
            mask = state_at_decision.get("mask", [])
            n_acquired = sum(1 for m in mask if m > 0.5)
            stage = _N_ACQUIRED_TO_STAGE.get(n_acquired)
            if stage is None:
                continue

            # Per-stage threshold: "good" = terminal reward above what
            # a typical patient would get by stopping at this stage
            threshold = reward_thresholds.get(stage, 0.0)
            outcome_good = terminal_reward >= threshold

            # Determine if agent stopped or acquired
            agent_direction = "stop" if action_taken == 0 else "acquire"

            for r in active:
                if r.get("stage") != stage:
                    continue

                r["n_applied"] = r.get("n_applied", 0) + 1
                r["last_applied_episode"] = current_episode

                followed = (r.get("direction") == agent_direction)

                if followed and outcome_good:
                    r["n_followed_good"] = r.get("n_followed_good", 0) + 1
                elif followed and not outcome_good:
                    r["n_followed_poor"] = r.get("n_followed_poor", 0) + 1
                elif not followed and outcome_good:
                    r["n_violated_good"] = r.get("n_violated_good", 0) + 1
                else:
                    r["n_violated_poor"] = r.get("n_violated_poor", 0) + 1

                # Compute effectiveness: proportion of followed cases that were good
                n_followed = r.get("n_followed_good", 0) + r.get("n_followed_poor", 0)
                if n_followed >= 3:
                    r["effectiveness"] = r["n_followed_good"] / n_followed

                # Auto-deprecate: effectiveness < 0.3 after 5+ applications
                if r.get("n_applied", 0) >= 5 and r.get("effectiveness") is not None:
                    if r["effectiveness"] < 0.3:
                        r["active"] = False
                        if self.debug:
                            print(f"[SEMANTIC] Auto-deprecated {r['id']} "
                                  f"(effectiveness={r['effectiveness']:.2f}, "
                                  f"n_applied={r['n_applied']})")

    def decay_stale_rules(self, current_episode: int, stale_threshold: int = 30):
        """Decay confidence of rules not validated for many episodes.

        Rules not applied for stale_threshold+ episodes get confidence *= 0.95.
        Rules below 0.2 confidence are auto-deprecated.

        Args:
            current_episode:  current episode count.
            stale_threshold:  episodes without application before decay kicks in.
        """
        if not self.enabled:
            return

        for r in self.get_active_rules():
            last_applied = r.get("last_applied_episode", 0)
            if current_episode - last_applied >= stale_threshold:
                old_conf = r.get("confidence", 0.5)
                r["confidence"] = old_conf * 0.95
                if r["confidence"] < 0.2:
                    r["active"] = False
                    if self.debug:
                        print(f"[SEMANTIC] Stale-deprecated {r['id']} "
                              f"(conf={r['confidence']:.3f}, "
                              f"stale for {current_episode - last_applied} eps)")

    def update_rule(
        self,
        rule_id: str,
        supporting: bool = True,
        new_confidence: Optional[float] = None,
    ):
        """Update a rule's confidence based on new evidence.

        Args:
            rule_id:        ID of the rule to update.
            supporting:     whether the new evidence supports the rule.
            new_confidence: override confidence (if None, auto-compute).
        """
        for rule in self.rules:
            if rule["id"] == rule_id:
                if new_confidence is not None:
                    rule["confidence"] = new_confidence
                else:
                    # Use outcome tracking data if available
                    fg = rule.get("n_followed_good", 0)
                    fp = rule.get("n_followed_poor", 0)
                    s = fg + (1 if supporting else 0)
                    c = fp + (0 if supporting else 1)
                    rule["confidence"] = max(0.1, min(0.90, (s + 1) / (s + c + 2)))
                return

    def deprecate_rule(self, rule_id: str):
        """Mark a rule as inactive (deprecated)."""
        for rule in self.rules:
            if rule["id"] == rule_id:
                rule["active"] = False
                if self.debug:
                    print(f"[SEMANTIC] Deprecated rule {rule_id}")
                return

    def get_all_active(self) -> List[dict]:
        """Return all active rules."""
        return [r for r in self.rules if r.get("active", False)]

    def format_for_prompt(
        self,
        rules: Optional[List[dict]] = None,
        max_rules: int = 10,
    ) -> str:
        """Format rules as clinical experience patterns for the LLM prompt."""
        if rules is None:
            rules = self.get_all_active()

        if not rules:
            return "No learned patterns yet."

        lines = ["=== Clinical Experience Patterns ==="]
        for i, r in enumerate(rules[:max_rules], 1):
            conf = r.get("confidence", 0)
            eff = r.get("effectiveness")
            n_applied = r.get("n_applied", 0)
            pattern = r.get("pattern", r.get("rule", ""))
            stage = r.get("stage", "unknown")
            direction = r.get("direction", "unknown")

            # Build track record string
            if eff is not None and n_applied > 0:
                track = f"effectiveness={eff:.0%}, applied {n_applied}x"
            elif n_applied > 0:
                track = f"applied {n_applied}x (tracking)"
            else:
                track = "new"

            lines.append(
                f"Pattern {i} [{stage}, {direction}] "
                f"(conf={conf:.2f}, {track}):\n"
                f"  {pattern}\n"
                f"  Guidance: {r.get('action_guidance', 'n/a')}"
            )
        return "\n".join(lines)

    def format_rule_effectiveness(self) -> str:
        """Format outcome tracking data for the reflection prompt."""
        active = self.get_active_rules()
        if not active:
            return "No active rules to evaluate."

        lines = ["=== Rule Effectiveness Report ==="]
        for r in active:
            n_applied = r.get("n_applied", 0)
            fg = r.get("n_followed_good", 0)
            fp = r.get("n_followed_poor", 0)
            vg = r.get("n_violated_good", 0)
            vp = r.get("n_violated_poor", 0)
            eff = r.get("effectiveness")
            pattern = r.get("pattern", r.get("rule", ""))

            eff_str = f"{eff:.0%}" if eff is not None else "pending"
            lines.append(
                f"{r['id']} [{r.get('stage', '?')}, {r.get('direction', '?')}] "
                f"conf={r.get('confidence', 0):.2f}:\n"
                f"  Pattern: {pattern[:80]}\n"
                f"  Applied {n_applied}x | "
                f"Followed: {fg} good + {fp} poor | "
                f"Violated: {vg} good + {vp} poor | "
                f"Effectiveness: {eff_str}"
            )
        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Return summary statistics."""
        active = self.get_all_active()
        effs = [r["effectiveness"] for r in active
                if r.get("effectiveness") is not None]
        return {
            "n_learned": len(active),
            "n_total": len(active),
            "n_deprecated": sum(1 for r in self.rules if not r.get("active", True)),
            "avg_confidence": float(np.mean([
                r["confidence"] for r in active
            ])) if active else 0.0,
            "avg_effectiveness": float(np.mean(effs)) if effs else None,
            "n_with_effectiveness": len(effs),
        }

    def __len__(self):
        return len([r for r in self.rules if r.get("active", True)])

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "rules": self.rules,
            "next_learned_id": self._next_learned_id,
            "stats": self.get_stats(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        stats = self.get_stats()
        print(f"[SEMANTIC] Saved {stats['n_learned']} rules to {path}")

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict) or "rules" not in data:
            raise ValueError(
                f"Unexpected semantic memory format in {path}. "
                f"Expected dict with 'rules' key."
            )
        self.rules = data["rules"]
        self._next_learned_id = data.get("next_learned_id", 1)

        # Handle older format if needed
        for r in self.rules:
            if "pattern" not in r:
                # Normalize field names
                r["pattern"] = r.get("rule", r.get("condition", ""))
                r["stage"] = _infer_stage_from_condition(
                    r.get("condition", ""))
                r["direction"] = _infer_direction_from_guidance(
                    r.get("action_guidance", ""))
                r["supporting_evidence"] = ""
            # Initialize outcome tracking counters if missing
            for key in ("n_applied", "n_followed_good", "n_followed_poor",
                        "n_violated_good", "n_violated_poor"):
                if key not in r:
                    r[key] = 0
            if "effectiveness" not in r:
                r["effectiveness"] = None
            if "last_applied_episode" not in r:
                r["last_applied_episode"] = 0

        learned_ids = [
            int(r["id"].split("_")[-1])
            for r in self.rules
            if r.get("rule_type") == RULE_TYPE_LEARNED and "_" in r["id"]
        ]
        if learned_ids:
            self._next_learned_id = max(max(learned_ids) + 1, self._next_learned_id)
        stats = self.get_stats()
        print(f"[SEMANTIC] Loaded {stats['n_learned']} rules from {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _mask_to_modalities(mask) -> str:
    """Convert a mask array to human-readable modality list."""
    names = ["demographics", "radiology", "pathology", "genomics"]
    if isinstance(mask, (list, np.ndarray)):
        present = [names[i] for i in range(min(len(mask), len(names))) if mask[i] > 0.5]
        return ", ".join(present) if present else "none"
    return "unknown"
