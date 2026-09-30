"""Dual memory of SAGEAgent.

Episodic memory stores every past decision (embedding, stage, action, rewards)
and recalls stage-matched decisions for similar patients, re-ranked by
similarity plus min-max-normalized episode reward.

Semantic memory stores interpretable rules indexed by (stage, direction). The
agent's adherence to each relevant rule is recorded at every decision; at each
reflection cycle the episodes are labelled good/poor by reward quartile and
each rule's effectiveness (share of rule-following episodes that were good) is
updated. Rules that consistently lead to poor outcomes are deprecated.

Rules are plain text that the LLM reads in its prompt, so any chat model can use
them. Rule files (`save_rule_file` / `load_rule_file`) share them between runs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from ..clinical import PREDICT, ClinicalPathway
from ..utils import load_json, save_json

STOP, ACQUIRE = "stop", "acquire"


def direction_of(action: str) -> str:
    return STOP if action == PREDICT else ACQUIRE


# =============================================================================
# Episodic memory
# =============================================================================
class EpisodicMemory:
    def __init__(self, k: int = 3, reward_weight: float = 1.0, candidates: int = 30):
        self.k = k
        self.reward_weight = reward_weight
        self.candidates = candidates
        self.decisions: list[dict] = []      # one entry per decision (recalled at test time)
        self.episodes: list[dict] = []       # one entry per episode (used by reflection)
        self._index: dict[int, tuple] = {}   # depth -> (faiss index, entry positions)

    def __len__(self) -> int:
        return len(self.decisions)

    def add_episode(self, episode: dict) -> None:
        for step in episode["steps"]:
            if step.get("embedding") is None:     # forced stop, not an agent decision
                continue
            self.decisions.append({
                "patient_id": episode["patient_id"], "depth": step["depth"], "action": step["action"],
                "reward": step["reward"], "episode_reward": episode["total_reward"],
                "final_depth": episode["final_depth"], "embedding": np.asarray(step["embedding"], np.float32),
            })
        lean_steps = [{k: v for k, v in s.items() if k not in ("embedding", "reasoning")} for s in episode["steps"]]
        self.episodes.append({k: v for k, v in episode.items() if k != "steps"} | {"steps": lean_steps})
        self._index.clear()

    def _depth_index(self, depth: int):
        if depth not in self._index:
            import faiss

            pos = [i for i, d in enumerate(self.decisions) if d["depth"] == depth]
            index = None
            if pos:
                emb = np.stack([self.decisions[i]["embedding"] for i in pos])
                emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
                index = faiss.IndexFlatIP(emb.shape[1])
                index.add(emb.astype(np.float32))
            self._index[depth] = (index, pos)
        return self._index[depth]

    def recall(self, embedding: np.ndarray, depth: int, exclude: str | None = None) -> list[dict]:
        """Past decisions at the same stage for similar patients, re-ranked by reward."""
        index, pos = self._depth_index(depth)
        if index is None:
            return []
        query = (embedding / (np.linalg.norm(embedding) + 1e-8)).astype(np.float32)[None]
        scores, hits = index.search(query, min(self.candidates, index.ntotal))
        found = [(float(s), self.decisions[pos[h]]) for s, h in zip(scores[0], hits[0])
                 if h >= 0 and self.decisions[pos[h]]["patient_id"] != exclude]
        if not found:
            return []
        rewards = np.array([d["episode_reward"] for _, d in found])
        spread = rewards.max() - rewards.min()
        norm = (rewards - rewards.min()) / spread if spread > 1e-8 else np.full(len(rewards), 0.5)
        ranked = sorted(zip(found, norm), key=lambda x: -(x[0][0] + self.reward_weight * x[1]))
        return [{**d, "similarity": s} for (s, d), _ in ranked[: self.k]]

    @staticmethod
    def describe(recalled: list[dict], pathway: ClinicalPathway) -> str:
        if not recalled:
            return "Past decisions at this stage: none yet."
        stage = pathway.modalities[recalled[0]["depth"] - 1].label
        lines = [f"Past decisions after {stage} for similar patients:"]
        for n, d in enumerate(recalled, 1):
            stopped = pathway.modalities[d["final_depth"] - 1].label
            lines.append(f"  Case {n} (sim={d['similarity']:.2f}): {d['action']} → episode reward "
                         f"{d['episode_reward']:+.3f} (stopped after {stopped})")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        save_json({"decisions": self.decisions, "episodes": self.episodes}, path, indent=None)

    def load(self, path: str) -> "EpisodicMemory":
        data = load_json(path)
        self.decisions = [{**d, "embedding": np.asarray(d["embedding"], np.float32)} for d in data["decisions"]]
        self.episodes = data["episodes"]
        self._index.clear()
        return self


# =============================================================================
# Semantic memory
# =============================================================================
OUTCOME_KEYS = ("followed", "followed_good", "followed_poor", "violated", "violated_good", "violated_poor")
RULE_FILE_KEYS = ("id", "stage", "direction", "pattern", "action_guidance", "supporting_evidence", "confidence",
                  "effectiveness") + OUTCOME_KEYS


def normalize_rule(rule: dict, pathway: ClinicalPathway) -> tuple[dict | None, str]:
    """Check a rule against the pathway; returns (normalized rule, "") or (None, reason).

    A rule names a decision point of the pathway (e.g. 'after_radiology'), a
    direction ('stop' or 'acquire') and a pattern. Its action guidance must agree
    with the direction, and an acquire rule must name the next modality.
    """
    stage, direction = str(rule.get("stage", "")).strip(), str(rule.get("direction", "")).strip().lower()
    pattern = str(rule.get("pattern", "")).strip()
    guidance = str(rule.get("action_guidance", "")).lower()
    if stage not in pathway.stages:
        return None, f"unknown stage '{stage}'"
    if direction not in (STOP, ACQUIRE):
        return None, f"unknown direction '{direction}'"
    if not pattern:
        return None, "empty pattern"
    next_modality = pathway.modalities[pathway.stages.index(stage) + pathway.initial].name
    says_stop = "predict" in guidance or "stop" in guidance
    says_acquire = "acquire" in guidance
    if says_stop == says_acquire:
        return None, "guidance is not clearly stop or acquire"
    if (direction == STOP) != says_stop:
        return None, "guidance contradicts the direction"
    if direction == ACQUIRE and next_modality not in guidance:
        return None, f"acquire guidance must name the next modality ({next_modality})"
    try:
        confidence = float(rule.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "stage": stage, "direction": direction, "pattern": pattern,
        "action_guidance": "predict now" if direction == STOP else f"acquire {next_modality}",
        "supporting_evidence": str(rule.get("supporting_evidence", "")), "confidence": confidence,
    }, ""


def save_rule_file(path: str, rule_sets: dict[str, list[dict]], description: str = "") -> None:
    """Write a rule file with one rule set per pipeline, keyed 'outer_k/inner_m'."""
    save_json({"description": description, "pipelines": rule_sets}, path)


def load_rule_file(path: str, pipeline: str) -> list[dict]:
    """The rules of one pipeline ('outer_k/inner_m') from a rule file.

    A rule file holds either one rule set per pipeline, {"pipelines": {"outer_1/inner_1": [...], ...}},
    or a single set that every pipeline uses, {"rules": [...]}.
    """
    data = load_json(path)
    if "rules" in data:
        return data["rules"]
    try:
        return data["pipelines"][pipeline]
    except KeyError:
        raise KeyError(f"{path} has no rules for pipeline {pipeline}") from None


class SemanticMemory:
    def __init__(self, pathway: ClinicalPathway, max_active_rules: int = 10,
                 max_rules_per_stage_direction: int = 2, min_support: int = 5,
                 deprecate_poor_rate: float = 0.5):
        self.pathway = pathway
        self.max_active = max_active_rules
        self.max_per_slot = max_rules_per_stage_direction
        self.min_support = min_support
        self.deprecate_poor_rate = deprecate_poor_rate
        self.rules: list[dict] = []
        self._next_id = 1

    @classmethod
    def from_config(cls, cfg, pathway: ClinicalPathway) -> "SemanticMemory":
        s = cfg.semantic
        return cls(pathway, s.max_active_rules, s.max_rules_per_stage_direction, s.min_support,
                   s.deprecate_poor_rate)

    # ---------------------------------------------------------------- queries
    def active(self, stage: str | None = None, direction: str | None = None) -> list[dict]:
        return [r for r in self.rules if r["active"]
                and (stage is None or r["stage"] == stage) and (direction is None or r["direction"] == direction)]

    def get(self, rule_id: str) -> dict | None:
        return next((r for r in self.rules if r["id"] == rule_id), None)

    @staticmethod
    def _strength(rule: dict) -> tuple:
        eff = rule["effectiveness"]
        return (-1.0 if eff is None else eff, rule["confidence"])

    # ---------------------------------------------------------------- updates
    @staticmethod
    def _new_rule(rule_id: str, rule: dict, cycle: int | None) -> dict:
        return {
            "id": rule_id, "stage": rule["stage"], "direction": rule["direction"],
            "pattern": rule["pattern"], "action_guidance": rule["action_guidance"],
            "supporting_evidence": rule.get("supporting_evidence", ""),
            "confidence": float(rule.get("confidence", 0.5)), "created_in_cycle": cycle, "active": True,
            **{key: 0 for key in OUTCOME_KEYS}, "effectiveness": None, "deprecation_reason": None,
        }

    def add(self, rule: dict, cycle: int) -> str:
        """Add a validated rule; the weakest rule gives way when a cap is reached."""
        slot = self.active(rule["stage"], rule["direction"])
        if len(slot) >= self.max_per_slot:
            self.deprecate(min(slot, key=self._strength)["id"], "replaced by a newer rule for the same decision")
        if len(self.active()) >= self.max_active:
            self.deprecate(min(self.active(), key=self._strength)["id"], "active-rule limit reached")
        while self.get(f"rule_{self._next_id}") is not None:
            self._next_id += 1
        rule_id = f"rule_{self._next_id}"
        self._next_id += 1
        self.rules.append(self._new_rule(rule_id, rule, cycle))
        return rule_id

    def deprecate(self, rule_id: str, reason: str) -> bool:
        rule = self.get(rule_id)
        if rule is None or not rule["active"]:
            return False
        rule["active"], rule["deprecation_reason"] = False, reason
        return True

    def adherence(self, stage: str, action: str) -> dict[str, bool]:
        """Which active rules apply at this stage and whether the chosen action follows them."""
        chosen = direction_of(action)
        return {r["id"]: r["direction"] == chosen for r in self.active(stage)}

    def update_outcomes(self, episodes: Iterable[dict], labels: Iterable[str]) -> list[str]:
        """Credit rules with the labelled outcomes of one reflection cycle.

        ``labels`` are 'good', 'poor' or 'neutral' per episode. Returns the ids
        of rules deprecated because they consistently led to poor outcomes.
        """
        for episode, label in zip(episodes, labels):
            seen = defaultdict(bool)
            for step in episode["steps"]:
                for rule_id, followed in step.get("adherence", {}).items():
                    seen[rule_id] |= followed
            for rule_id, followed in seen.items():
                rule = self.get(rule_id)
                if rule is None:
                    continue
                key = "followed" if followed else "violated"
                rule[key] += 1
                if label in ("good", "poor"):
                    rule[f"{key}_{label}"] += 1
        failing = []
        for rule in self.active():
            if rule["followed"]:
                rule["effectiveness"] = rule["followed_good"] / rule["followed"]
            if (rule["followed"] >= self.min_support
                    and rule["followed_poor"] / rule["followed"] >= self.deprecate_poor_rate):
                self.deprecate(rule["id"], "consistently followed by poor outcomes")
                failing.append(rule["id"])
        return failing

    # ------------------------------------------------------------ description
    def describe(self, stage: str) -> str:
        """Rules for the current decision stage, as shown to the agent."""
        rules = sorted(self.active(stage), key=self._strength, reverse=True)
        if not rules:
            return "No learned rules for this stage yet."
        lines = []
        for n, r in enumerate(rules, 1):
            eff = "not yet evaluated" if r["effectiveness"] is None else f"eff={r['effectiveness']:.0%}"
            lines.append(f"Rule {n} [{r['stage']}, {r['direction']}] (conf={r['confidence']:.2f}, {eff}): "
                         f"{r['pattern']} Guidance: {r['action_guidance']}.")
        return "\n".join(lines)

    def effectiveness_report(self) -> str:
        """All active rules with their outcome statistics, for the reflection prompt."""
        if not self.active():
            return "No active rules."
        lines = []
        for r in self.active():
            eff = "n/a" if r["effectiveness"] is None else f"{r['effectiveness']:.0%}"
            lines.append(f"{r['id']} [{r['stage']}, {r['direction']}] conf={r['confidence']:.2f}: {r['pattern']}\n"
                         f"  followed {r['followed']}x ({r['followed_good']} good, {r['followed_poor']} poor), "
                         f"violated {r['violated']}x ({r['violated_good']} good, {r['violated_poor']} poor), "
                         f"effectiveness {eff}")
        return "\n".join(lines)

    # ------------------------------------------------------------- rule files
    def import_rules(self, rules: Iterable[dict]) -> "SemanticMemory":
        """Add shared rules (e.g. from a rule file) as they are, keeping their outcome statistics."""
        for rule in rules:
            normalized, reason = normalize_rule(rule, self.pathway)
            if normalized is None:
                raise ValueError(f"rule {rule.get('id', '?')}: {reason}")
            rule_id = str(rule.get("id") or f"rule_{len(self.rules) + 1}")
            if self.get(rule_id) is not None:
                raise ValueError(f"duplicate rule id '{rule_id}'")
            entry = self._new_rule(rule_id, normalized, cycle=None)
            entry.update({key: int(rule.get(key, 0)) for key in OUTCOME_KEYS}, effectiveness=rule.get("effectiveness"))
            self.rules.append(entry)
        return self

    def export_rules(self) -> list[dict]:
        """The active rules in rule-file form (see `save_rule_file`)."""
        return [{key: r[key] for key in RULE_FILE_KEYS} for r in self.active()]

    def save(self, path: str) -> None:
        save_json({"rules": self.rules, "next_id": self._next_id}, path)

    def load(self, path: str) -> "SemanticMemory":
        data = load_json(path)
        self.rules, self._next_id = data["rules"], data["next_id"]
        return self
