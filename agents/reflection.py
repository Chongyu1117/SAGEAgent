"""
Self-Reflection Module — periodic analysis of agent experience.

Runs every N episodes during experience accumulation:
1. Collects recent episodes, groups by outcome quality
2. Formats a reflection prompt with multi-signal episode data
3. Calls LLM to analyse patterns → output JSON rules
4. Parses and updates semantic memory (add / update / deprecate)

Rules are multi-signal patterns with outcome tracking, not single-signal
lookup tables. This enables continuous learning beyond initial reflections.
"""

import json
import os
import re
import time
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


def _text_similarity(a: str, b: str) -> float:
    """Quick text similarity via SequenceMatcher (0-1)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

import numpy as np

from agents.memory import EpisodicMemory, SemanticMemory


class ReflectionModule:
    """Self-reflection for semantic memory updates.

    Args:
        model_name:          HuggingFace model ID (shared with LLMAgent if possible).
        semantic_memory:     SemanticMemory to update.
        episodic_memory:     EpisodicMemory to read recent episodes.
        prompt_template_path: path to reflection prompt template.
        max_new_tokens:      max generation length for reflection.
        temperature:         sampling temperature.
        device:              torch device for LLM.
        max_episodes_per_reflection: cap on episodes to include in prompt.
        debug:               verbose logging.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        semantic_memory: Optional[SemanticMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        prompt_template_path: str = "prompts/reflection_prompt.txt",
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        device: str = "cuda",
        max_episodes_per_reflection: int = 10,
        decision_signal: str = "uncertainty",
        debug: bool = False,
    ):
        self.model_name = model_name
        self.semantic_memory = semantic_memory if semantic_memory is not None else SemanticMemory(debug=debug)
        self.episodic_memory = episodic_memory if episodic_memory is not None else EpisodicMemory(debug=debug)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.max_episodes_per_reflection = max_episodes_per_reflection
        self.decision_signal = decision_signal
        self.debug = debug

        # Load prompt template
        self.prompt_template = ""
        if os.path.exists(prompt_template_path):
            with open(prompt_template_path) as f:
                self.prompt_template = f.read()

        # LLM (lazy loaded, can share with LLMAgent)
        self._model = None
        self._tokenizer = None

        self.reflection_count = 0
        self.reflection_log: List[dict] = []

    def set_model(self, model, tokenizer):
        """Share an already-loaded model with the LLMAgent."""
        self._model = model
        self._tokenizer = tokenizer

    
    def set_unc_thresholds(self, thresholds: dict):
        """Optional: store uncertainty thresholds (not used for rules)."""
        self.unc_thresholds = thresholds

    def set_cavs_thresholds(self, thresholds: dict):
        """Optional: store CAVS thresholds (not used for rules)."""
        self.cavs_thresholds = thresholds

    def set_rdvs_thresholds(self, thresholds: dict):
        """Optional: store RDVS thresholds (not used for rules)."""
        self.rdvs_thresholds = thresholds

    # ── Episode formatting (multi-signal) ──────────────────────────────

    def _format_episodes(self, episodes: List[dict], max_n: int) -> str:
        """Format episode list with multi-signal data for prompt."""
        if not episodes:
            return "  (none)"

        lines = []
        for ep in episodes[:max_n]:
            mask = ep.get("acquired_mask", [])
            mod_names = ["demographics", "radiology", "pathology", "genomics"]
            acquired = [
                mod_names[i]
                for i in range(min(len(mask), len(mod_names)))
                if isinstance(mask, (list, np.ndarray)) and (
                    mask[i] > 0.5 if isinstance(mask[i], (int, float)) else False
                )
            ]
            n_mods = len(acquired)
            stopped_at = acquired[-1] if acquired else "none"
            burden = ep.get("total_burden", 0)
            risk = ep.get("risk_score", ep.get("oracle_risk", 0))
            oracle_risk = ep.get("oracle_risk", 0)
            risk_error = abs(risk - oracle_risk)
            outcome = ep.get("outcome", "unknown")
            total_reward = sum(
                s.get("reward", 0) for s in ep.get("trajectory", [])
            )

            quality = ep.get("prediction_quality", None)
            quality_str = f"{quality:.2f}" if quality is not None else "n/a"

            # Uncertainty progression (raw numbers, no quintile labels)
            step_uncs = ep.get("step_uncertainties", [])
            if step_uncs:
                unc_str = " → ".join(f"{u:.3f}" for u in step_uncs)
            else:
                unc_str = "n/a"

            # CAVS/RDVS values if available
            step_cavs = ep.get("step_cavs_values", [])
            cavs_str = ""
            if step_cavs:
                cavs_str = f" | CAVS: {' → '.join(f'{v:.3f}' for v in step_cavs)}"

            lines.append(
                f"  - Stopped at: {stopped_at} ({n_mods} mods) | "
                f"Burden: {burden:.2f} | "
                f"Risk error: {risk_error:.4f} | "
                f"Outcome: {outcome} | Return: {total_reward:.3f}\n"
                f"    Quality: {quality_str} | "
                f"Uncertainty: {unc_str}{cavs_str}"
            )
        return "\n".join(lines)

    def _compute_coverage_gaps(self) -> str:
        """Identify which (stage, direction) combos lack rules.

        5 slots total:
        - (after_demographics, acquire)
        - (after_radiology, stop)
        - (after_radiology, acquire)
        - (after_pathology, stop)
        - (after_pathology, acquire)
        """
        active_rules = self.semantic_memory.get_active_rules()

        covered = set()
        for r in active_rules:
            stage = r.get("stage", "")
            direction = r.get("direction", "")
            if stage and direction:
                covered.add((stage, direction))

        needed = [
            ("after_demographics", "acquire",
             "acquire radiology",
             "MRI is cheap (0.14), almost always worth acquiring"),
            ("after_radiology", "stop",
             "predict now",
             "skip invasive pathology (0.53) when prediction is confident"),
            ("after_radiology", "acquire",
             "acquire pathology",
             "pathology warranted when prediction quality is insufficient"),
            ("after_pathology", "stop",
             "predict now",
             "skip expensive genomics (0.30) when prediction is accurate"),
            ("after_pathology", "acquire",
             "acquire genomics",
             "genomics may improve prediction for uncertain cases"),
        ]

        gaps = []
        for stage, direction, hint, reason in needed:
            if (stage, direction) not in covered:
                gaps.append(f"  {stage} + {direction}: consider {hint} ({reason})")

        if not gaps:
            return "All key decision points covered. Review existing rules — deprecate any with low effectiveness."
        return "These decision points need patterns:\n" + "\n".join(gaps)

    def _format_rule_effectiveness(self) -> str:
        """Format outcome tracking data for existing rules."""
        return self.semantic_memory.format_rule_effectiveness()

    def _load_model(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[REFLECTION] Loading {self.model_name} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        self._model.eval()
        print(f"[REFLECTION] Model loaded on {self.device}")

    def reflect(
        self,
        recent_episodes: Optional[List[dict]] = None,
        n_recent: int = 50,
    ) -> dict:
        """Run one reflection cycle.

        Args:
            recent_episodes: episodes to analyse (if None, take last *n_recent*
                             from episodic memory).
            n_recent:        how many recent episodes to consider.

        Returns:
            dict with new_rules, updated_rules, deprecated_rules counts.
        """
        self._load_model()

        # 1. Get episodes (use full episodes, not step-level entries)
        if recent_episodes is None:
            recent_episodes = self.episodic_memory.full_episodes[-n_recent:]

        if len(recent_episodes) < 5:
            if self.debug:
                print(f"[REFLECTION] Only {len(recent_episodes)} episodes, skipping")
            return {"skipped": True, "reason": "too_few_episodes"}

        # 2. Classify by quality
        good, mediocre, poor = self._classify_episodes(recent_episodes)

        if self.debug:
            print(
                f"[REFLECTION] Episodes: {len(recent_episodes)} total, "
                f"{len(good)} good, {len(mediocre)} mediocre, {len(poor)} poor"
            )

        # 3. Build reflection prompt
        prompt = self._build_prompt(good, mediocre, poor)

        # 4. Generate reflection (with retry on JSON parse failure)
        t0 = time.time()
        max_retries = 3
        response = None
        for attempt in range(max_retries):
            response = self._generate(prompt)
            parsed = self._extract_json(response)
            if parsed is not None:
                if self.debug and attempt > 0:
                    print(f"[REFLECTION] JSON parsed on attempt {attempt + 1}")
                break
            if self.debug:
                print(f"[REFLECTION] JSON parse failed attempt {attempt + 1}/{max_retries} "
                      f"(len={len(response)})")
        dt = time.time() - t0

        if self.debug:
            print(f"[REFLECTION] Generated in {dt:.1f}s, length={len(response)}")

        # 5. Parse & apply
        result = self._parse_and_apply(response)
        result["generation_time"] = dt
        result["n_episodes"] = len(recent_episodes)

        self.reflection_count += 1
        self.reflection_log.append(result)

        return result

    def _classify_episodes(
        self, episodes: List[dict]
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        """Split episodes into good / mediocre / poor by relative percentile.

        Uses pure percentile-based split (no absolute quality gate):
        - Top 25% by total reward → good
        - Bottom 25% → poor
        - Middle 50% → mediocre
        """
        scored = []
        for ep in episodes:
            total_reward = sum(
                step.get("reward", 0)
                for step in ep.get("trajectory", [])
            )
            scored.append((total_reward, ep))

        if len(scored) < 5:
            return [s[1] for s in scored], [], []

        rewards = np.array([s[0] for s in scored])
        q75 = float(np.percentile(rewards, 75))
        q25 = float(np.percentile(rewards, 25))

        good, mediocre, poor = [], [], []
        for reward, ep in scored:
            if reward >= q75:
                good.append(ep)
            elif reward <= q25:
                poor.append(ep)
            else:
                mediocre.append(ep)

        if self.debug:
            print(f"[REFLECTION] Classify: q25={q25:.3f}, q75={q75:.3f} "
                  f"→ {len(good)} good, {len(mediocre)} mid, {len(poor)} poor")

        return good, mediocre, poor

    def _build_prompt(
        self,
        good: List[dict],
        mediocre: List[dict],
        poor: List[dict],
    ) -> str:
        """Fill in the reflection prompt template."""
        max_ep = self.max_episodes_per_reflection

        current_rules = self.semantic_memory.format_for_prompt()
        coverage_gaps = self._compute_coverage_gaps()
        rule_effectiveness = self._format_rule_effectiveness()

        return self.prompt_template.format(
            n_episodes=len(good) + len(mediocre) + len(poor),
            n_good=len(good),
            n_mediocre=len(mediocre),
            n_poor=len(poor),
            good_episodes=self._format_episodes(good, max_ep),
            poor_episodes=self._format_episodes(poor, max_ep),
            current_rules=current_rules,
            coverage_gaps=coverage_gaps,
            rule_effectiveness=rule_effectiveness,
        )

    def _generate(self, prompt: str) -> str:
        """Generate reflection text."""
        import torch

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert clinical AI analyst. Analyse the agent's "
                    "decision-making episodes and distill actionable patterns."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        input_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=8192
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def _extract_json(self, response: str) -> dict:
        """Robustly extract JSON from LLM response.

        Tries multiple strategies:
        1. Extract from markdown code blocks (```json ... ```)
        2. Find outermost balanced braces
        3. Greedy regex fallback
        """
        # Strategy 1: markdown code block
        code_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 2: find balanced braces (outermost pair)
        start = response.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(response)):
                if response[i] == "{":
                    depth += 1
                elif response[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(response[start:i + 1])
                        except json.JSONDecodeError:
                            break

        # Strategy 3: greedy regex
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                # Try fixing common issues: trailing commas
                text = json_match.group()
                text = re.sub(r",\s*([}\]])", r"\1", text)
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    pass

        return None

    def _verify_rule(self, rule: dict, recent_episodes: List[dict]):
        """Verify a candidate rule before adding to semantic memory.

        Checks:
        1. Direction clarity: guidance must be unambiguously stop OR acquire
        2. (stage, direction) cap of 2: if full, replace worst-effectiveness one
        3. Text novelty: similarity < 0.75 to existing patterns

        Returns:
            (True, None):      accept the new rule
            (True, rule_id):   accept AND deprecate the old rule (replace worst)
            (False, None):     reject the new rule
        """
        guidance = rule.get("action_guidance", "").lower()
        pattern = rule.get("pattern", rule.get("rule", "")).lower()
        stage = rule.get("stage", "")
        direction = rule.get("direction", "")

        # Check 1: Direction clarity — guidance must match direction field
        has_predict = "predict" in guidance or "stop" in guidance
        has_acquire = "acquire" in guidance
        if has_predict and has_acquire:
            if self.debug:
                print(f"[REFLECTION] Discarded ambiguous rule (both predict+acquire): "
                      f"{pattern[:60]}")
            return False, None
        if not has_predict and not has_acquire:
            if self.debug:
                print(f"[REFLECTION] Discarded rule with no clear action: "
                      f"{pattern[:60]}")
            return False, None

        # Infer direction from guidance if not explicitly set
        if not direction:
            direction = "stop" if has_predict else "acquire"
            rule["direction"] = direction

        # Verify consistency: direction field must match guidance
        if direction == "stop" and not has_predict:
            if self.debug:
                print(f"[REFLECTION] Discarded inconsistent rule "
                      f"(direction=stop, guidance={guidance}): {pattern[:60]}")
            return False, None
        if direction == "acquire" and not has_acquire:
            if self.debug:
                print(f"[REFLECTION] Discarded inconsistent rule "
                      f"(direction=acquire, guidance={guidance}): {pattern[:60]}")
            return False, None

        # Check 2: (stage, direction) cap of 2
        if self.semantic_memory and stage:
            existing = self.semantic_memory.get_active_rules()
            same_slot = [r for r in existing
                         if r.get("stage") == stage
                         and r.get("direction") == direction]

            if len(same_slot) >= 2:
                # Replace the one with worst effectiveness
                worst = min(same_slot, key=lambda r: (
                    r.get("effectiveness") if r.get("effectiveness") is not None else -1,
                    r.get("confidence", 0)))
                if self.debug:
                    print(f"[REFLECTION] Slot ({stage}, {direction}) full, "
                          f"replacing {worst['id']}")
                return True, worst["id"]

        # Check 3: Text novelty — only within same (stage, direction) slot.
        # Cross-slot rules naturally share vocabulary (e.g. "radiology" vs
        # "pathology" differs by one word → high false-positive similarity).
        if self.semantic_memory and stage and direction:
            existing = self.semantic_memory.get_active_rules()
            for er in existing:
                if er.get("stage") != stage or er.get("direction") != direction:
                    continue
                er_pattern = er.get("pattern", er.get("rule", "")).lower()
                if _text_similarity(pattern, er_pattern) > 0.95:
                    if self.debug:
                        print(f"[REFLECTION] Discarded duplicate pattern "
                              f"(sim>0.95 in {stage}/{direction}): {pattern[:60]}")
                    return False, None

        return True, None

    def _parse_and_apply(self, response: str) -> dict:
        """Parse JSON rules from LLM response, verify, and apply to semantic
        memory. Rules that fail verification are discarded."""
        result = {
            "new_rules_added": 0,
            "new_rules_discarded": 0,
            "rules_updated": 0,
            "rules_deprecated": 0,
            "parse_success": False,
        }

        data = self._extract_json(response)
        if data is None:
            if self.debug:
                print(f"[REFLECTION] No valid JSON found in response "
                      f"(length={len(response)})")
            return result

        result["parse_success"] = True

        # Get recent episodes for verification context
        recent_episodes = self.episodic_memory.full_episodes[-50:]

        # Apply new rules (with verification)
        for rule in data.get("new_rules", []):
            # Accept both "pattern" and "rule" keys for flexibility
            if "pattern" not in rule and "rule" not in rule:
                continue
            rule["source"] = "reflection"

            # Normalize: ensure pattern field exists
            if "pattern" not in rule:
                rule["pattern"] = rule.get("rule", "")

            accept, replace_id = self._verify_rule(rule, recent_episodes)
            if accept:
                if replace_id:
                    self.semantic_memory.deprecate_rule(replace_id)
                    result["rules_deprecated"] += 1
                rule_id = self.semantic_memory.add_rule(rule)
                if rule_id:
                    result["new_rules_added"] += 1
            else:
                result["new_rules_discarded"] += 1

        # Update existing rules
        for update in data.get("update_rules", []):
            rule_id = update.get("rule_id")
            if not rule_id:
                continue
            self.semantic_memory.update_rule(
                rule_id,
                supporting=update.get("supporting", True),
                new_confidence=update.get("new_confidence"),
            )
            result["rules_updated"] += 1

        # Deprecate rules
        for rule_id in data.get("deprecate_rules", []):
            self.semantic_memory.deprecate_rule(rule_id)
            result["rules_deprecated"] += 1

        if self.debug:
            print(
                f"[REFLECTION] Applied: {result['new_rules_added']} new, "
                f"{result.get('new_rules_discarded', 0)} discarded, "
                f"{result['rules_updated']} updated, "
                f"{result['rules_deprecated']} deprecated"
            )

        return result

    def save_log(self, path: str):
        """Save reflection log to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.reflection_log, f, indent=2, default=str)
        print(f"[REFLECTION] Log saved to {path}")
