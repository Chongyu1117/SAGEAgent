"""SAGEAgent: a frozen LLM that decides, stage by stage, whether to acquire the next modality.

At each stage the prompt combines three signal sources:
  - clinical tools   (uncertainty u_t and risk r_t as text),
  - episodic memory  (similar training patients and similar past decisions),
  - semantic memory  (learned rules for the current stage),
and the LLM answers with a chain of thought ending in ``ACTION: <action>``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..clinical import PREDICT, ClinicalPathway
from ..env import AcquisitionEnv, State
from .llm import ChatLLM
from .memory import EpisodicMemory, SemanticMemory
from .tools import CaseRetriever, SurvivalPredictorTool, UncertaintyTool

SYSTEM_PROMPT = "You are a clinical decision support AI."
UNAVAILABLE = "Not available."


@dataclass
class Decision:
    action: str
    reasoning: str
    prompt: str
    parsed: bool          # False if no explicit action was found and the fallback was used


class SAGEAgent:
    def __init__(self, llm: ChatLLM, pathway: ClinicalPathway, template: str, generation: dict,
                 uncertainty_tool: UncertaintyTool | None = None,
                 predictor_tool: SurvivalPredictorTool | None = None,
                 retriever: CaseRetriever | None = None,
                 episodic: EpisodicMemory | None = None,
                 semantic: SemanticMemory | None = None):
        self.llm = llm
        self.pathway = pathway
        self.template = template
        self.generation = generation
        self.uncertainty_tool = uncertainty_tool
        self.predictor_tool = predictor_tool
        self.retriever = retriever
        self.episodic = episodic
        self.semantic = semantic

    # ----------------------------------------------------------------- prompt
    def prompt(self, state: State, env: AcquisitionEnv, exclude: str | None = None) -> str:
        """Decision prompt for one state; ``exclude`` hides that patient from retrieval."""
        modalities = self.pathway.modalities
        nxt = modalities[state.depth]
        tools = [t(state) for t in (self.uncertainty_tool, self.predictor_tool) if t is not None]
        cases = []
        if self.retriever is not None:
            cases.append(CaseRetriever.describe(self.retriever.search(state.embedding, exclude)))
        if self.episodic is not None:
            cases.append(EpisodicMemory.describe(self.episodic.recall(state.embedding, state.depth, exclude),
                                                 self.pathway))
        rules = self.semantic.describe(self.pathway.stage(state.depth)) if self.semantic is not None else UNAVAILABLE
        return self.template.format(
            disease=self.pathway.disease,
            acquired=", ".join(m.label for m in modalities[: state.depth]),
            next_modality=f"{nxt.label}, {nxt.description} (burden {nxt.burden:.2f})",
            remaining=", ".join(f"{m.label} ({m.burden:.2f})" for m in modalities[state.depth + 1:]) or "none",
            burden=state.burden, full_burden=float(self.pathway.burdens.sum()), next_burden=nxt.burden,
            tools="\n".join(tools) or UNAVAILABLE, episodic="\n\n".join(cases) or UNAVAILABLE, semantic=rules,
            actions=" or ".join(env.valid_actions(state)),
        )

    # ---------------------------------------------------------------- decide
    def decide(self, state: State, env: AcquisitionEnv, exclude: str | None = None) -> Decision:
        return self.decide_batch([state], env, [exclude])[0]

    def decide_batch(self, states: list[State], env: AcquisitionEnv,
                     exclude: list[str | None] | None = None) -> list[Decision]:
        exclude = exclude or [None] * len(states)
        prompts = [self.prompt(s, env, ex) for s, ex in zip(states, exclude)]
        answers = self.llm.chat(SYSTEM_PROMPT, prompts, **self.generation)
        decisions = []
        for state, prompt, answer in zip(states, prompts, answers):
            action, parsed, reasoning = self.parse_action(answer, env.valid_actions(state))
            decisions.append(Decision(action, reasoning, prompt, parsed))
        return decisions

    def parse_action(self, text: str, valid: list[str]) -> tuple[str, bool, str]:
        """Map the answer to a valid action; returns (action, parsed, reasoning).

        Uses the first ``ACTION:`` line that names a valid action and cuts the
        reasoning there (anything the model writes afterwards is not part of the
        decision). Without such a line, a final line naming exactly one action
        is used. When the answer is unusable the agent continues the workup
        (acquires), so a formatting error never silently skips a test.
        """
        acquire = next((a for a in valid if a != PREDICT), None)
        next_name = acquire.split("_", 1)[1] if acquire else None

        def classify(fragment: str) -> str | None:
            token = fragment.upper()
            says_predict = "PREDICT" in token or re.search(r"\bSTOP\b", token) is not None
            says_acquire = acquire is not None and ("ACQUIRE" in token or next_name in token)
            if says_predict != says_acquire:
                return PREDICT if says_predict else acquire
            return None

        for match in re.finditer(r"ACTION\s*:\s*\**\s*([A-Za-z_]+)", text, flags=re.I):
            action = classify(match.group(1))
            if action is not None:
                end = text.find("\n", match.end())
                return action, True, text[: len(text) if end < 0 else end].strip()
        lines = [line for line in text.strip().splitlines() if line.strip()]
        for line in reversed(lines[-3:]):
            action = classify(line)
            if action is not None:
                return action, True, text.strip()
        return (acquire or PREDICT), False, text.strip()


class UncertaintyThresholdPolicy:
    """Naive baseline without an LLM: stop as soon as the calibrated uncertainty u_t < tau."""

    def __init__(self, tau: float):
        self.tau = tau

    def decide_batch(self, states: list[State], env: AcquisitionEnv, exclude=None) -> list[Decision]:
        decisions = []
        for state in states:
            acquire = [a for a in env.valid_actions(state) if a != PREDICT]
            stop = state.uncertainty < self.tau or not acquire
            action = PREDICT if stop else acquire[0]
            reason = f"u_t = {state.uncertainty:.3f} {'<' if stop else '≥'} tau = {self.tau}"
            decisions.append(Decision(action, reason, "", True))
        return decisions
