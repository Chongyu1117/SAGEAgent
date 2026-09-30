"""Self-reflection: turns a cycle of episodes into interpretable decision rules.

Every reflection cycle:
  1. rank the cycle's episodes by total reward; the top and bottom quartiles
     are labelled good and poor,
  2. credit the semantic-memory rules the agent followed or violated in each
     episode and deprecate rules that consistently lead to poor outcomes,
  3. ask the frozen LLM to compare good and poor episodes and propose rules,
  4. keep only proposals that name a valid decision point, have a clear
     direction, and are not near-duplicates of existing rules.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

import numpy as np

from ..clinical import ClinicalPathway
from .llm import ChatLLM
from .memory import ACQUIRE, STOP, SemanticMemory, normalize_rule

SYSTEM_PROMPT = "You are an expert clinical AI analyst. You derive decision rules from an agent's experience."


def extract_json(text: str) -> dict | None:
    """First JSON object in an LLM answer (tolerates code fences and trailing commas)."""
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                candidates.append(text[start:i + 1])
                break
    for candidate in candidates:
        for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                data = json.loads(attempt)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue
    return None


class Reflection:
    def __init__(self, llm: ChatLLM, semantic: SemanticMemory, pathway: ClinicalPathway, template: str, cfg):
        self.llm = llm
        self.semantic = semantic
        self.pathway = pathway
        self.template = template
        self.cfg = cfg.semantic
        self.cycle = 0
        self.log: list[dict] = []

    # --------------------------------------------------------------- labelling
    def label(self, episodes: list[dict]) -> list[str]:
        rewards = np.array([e["total_reward"] for e in episodes])
        good_at = np.quantile(rewards, self.cfg.good_quantile)
        poor_at = np.quantile(rewards, self.cfg.poor_quantile)
        if good_at <= poor_at:              # no spread: nothing to learn from
            return ["neutral"] * len(episodes)
        return ["good" if r >= good_at else "poor" if r <= poor_at else "neutral" for r in rewards]

    # ----------------------------------------------------------------- prompt
    def _describe(self, episode: dict) -> str:
        m = self.pathway.modalities
        uncertainty = " → ".join(f"{u:.3f}" for u in episode["uncertainties"])
        return (f"  - started after {m[episode['start_depth'] - 1].label}, "
                f"stopped after {m[episode['final_depth'] - 1].label} | burden {episode['burden']:.2f} | "
                f"uncertainty {uncertainty} | risk error {episode['risk_error']:.3f} | "
                f"total reward {episode['total_reward']:+.3f}")

    def build_prompt(self, episodes: list[dict], labels: list[str]) -> str:
        n = self.cfg.reflection.max_examples
        good = sorted((e for e, lab in zip(episodes, labels) if lab == "good"), key=lambda e: -e["total_reward"])
        poor = sorted((e for e, lab in zip(episodes, labels) if lab == "poor"), key=lambda e: e["total_reward"])
        uncovered = [f"  - {stage}: {d}" for stage in self.pathway.stages for d in (STOP, ACQUIRE)
                     if not self.semantic.active(stage, d)]
        return self.template.format(
            disease=self.pathway.disease, pathway=self.pathway.describe(),
            n_episodes=len(episodes), n_good=len(good), n_poor=len(poor),
            good_episodes="\n".join(map(self._describe, good[:n])) or "  (none)",
            poor_episodes="\n".join(map(self._describe, poor[:n])) or "  (none)",
            baseline=1.0 - self.cfg.good_quantile, rule_report=self.semantic.effectiveness_report(),
            uncovered="\n".join(uncovered) or "  (every decision point has at least one rule)",
            stages=", ".join(self.pathway.stages), guidance=", ".join(f'"{g}"' for g in self.pathway.guidance_options()),
        )

    # -------------------------------------------------------------- validation
    def validate(self, rule: dict) -> tuple[dict | None, str]:
        """Normalized rule, or None and the reason it was rejected."""
        normalized, reason = normalize_rule(rule, self.pathway)
        if normalized is None:
            return None, reason
        for existing in self.semantic.active(normalized["stage"], normalized["direction"]):
            similarity = SequenceMatcher(None, normalized["pattern"].lower(), existing["pattern"].lower()).ratio()
            if similarity > self.cfg.novelty_threshold:
                return None, f"near-duplicate of {existing['id']}"
        normalized["confidence"] = float(np.clip(normalized["confidence"], 0.5, 0.8))
        return normalized, ""

    # ------------------------------------------------------------------- cycle
    def run(self, episodes: list[dict]) -> dict:
        self.cycle += 1
        labels = self.label(episodes)
        auto_deprecated = self.semantic.update_outcomes(episodes, labels)
        prompt = self.build_prompt(episodes, labels)

        r = self.cfg.reflection
        data, answer = None, ""
        for _ in range(r.max_attempts):
            answer = self.llm.generate(SYSTEM_PROMPT, prompt, temperature=r.temperature, top_p=r.top_p,
                                       max_new_tokens=r.max_new_tokens)
            data = extract_json(answer)
            if data is not None:
                break

        added, rejected, updated, deprecated = [], [], [], []
        for proposal in (data or {}).get("new_rules") or []:
            rule, reason = self.validate(proposal) if isinstance(proposal, dict) else (None, "not an object")
            if rule is None:
                rejected.append({"proposal": proposal, "reason": reason})
            else:
                added.append(self.semantic.add(rule, self.cycle))
        for change in (data or {}).get("update_rules") or []:
            rule = self.semantic.get(str(change.get("rule_id"))) if isinstance(change, dict) else None
            if rule is not None and rule["active"] and isinstance(change.get("new_confidence"), (int, float)):
                rule["confidence"] = float(np.clip(change["new_confidence"], 0.1, 0.9))
                updated.append(rule["id"])
        for rule_id in (data or {}).get("deprecate_rules") or []:
            if self.semantic.deprecate(str(rule_id), "deprecated by reflection"):
                deprecated.append(str(rule_id))

        entry = {
            "cycle": self.cycle, "n_episodes": len(episodes),
            "n_good": labels.count("good"), "n_poor": labels.count("poor"),
            "parsed": data is not None, "added": added, "rejected": rejected, "updated": updated,
            "deprecated_by_reflection": deprecated, "deprecated_for_poor_outcomes": auto_deprecated,
            "active_rules": [r["id"] for r in self.semantic.active()], "answer": answer,
        }
        self.log.append(entry)
        return entry
