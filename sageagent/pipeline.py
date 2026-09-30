"""Wiring shared by the pipeline scripts: output layout, data loading and agent assembly."""

from __future__ import annotations

import os

from .agent import ChatLLM, EpisodicMemory, SAGEAgent, SemanticMemory, build_tools
from .clinical import ClinicalPathway
from .config import resolve_path
from .data import Cohort, NestedSplits
from .env import AcquisitionEnv
from .models import load_predictor, load_uncertainty_head


class Workspace:
    """Where each pipeline step reads and writes, derived from ``experiment.output_dir``.

    <output_dir>/
      predictors/outer_k/inner_m/{predictor.pt, uncertainty_head.pt}
      agent/<run>/outer_k/inner_m/{episodic.json, semantic.json, reflections.json, episodes.jsonl}
      eval/<name>/{summary.json, patients.csv, traces/}
    """

    def __init__(self, cfg):
        self.root = resolve_path(cfg.experiment.output_dir)

    def fold_dir(self, outer: int, inner: int) -> str:
        return os.path.join(self.root, "predictors", f"outer_{outer}", f"inner_{inner}")

    def predictor(self, outer: int, inner: int) -> str:
        return os.path.join(self.fold_dir(outer, inner), "predictor.pt")

    def uncertainty_head(self, outer: int, inner: int) -> str:
        return os.path.join(self.fold_dir(outer, inner), "uncertainty_head.pt")

    def agent_dir(self, run: str, outer: int, inner: int) -> str:
        return os.path.join(self.root, "agent", run, f"outer_{outer}", f"inner_{inner}")

    def eval_dir(self, name: str) -> str:
        return os.path.join(self.root, "eval", name)


def load_data(cfg) -> tuple[ClinicalPathway, Cohort, NestedSplits]:
    pathway = ClinicalPathway.from_config(cfg)
    cohort = Cohort.load(resolve_path(cfg.data.cohort), pathway.names)
    splits = NestedSplits.load(resolve_path(cfg.data.splits))
    return pathway, cohort, splits


def complete_ids(cohort: Cohort, ids: list[str]) -> list[str]:
    """The patients among ``ids`` who have every modality (the agent's training patients)."""
    idx = cohort.index_of(ids)
    return [str(i) for i in cohort.ids[idx][cohort.complete[idx]]]


def load_fold_models(ws: Workspace, outer: int, inner: int, device: str):
    """Frozen predictor and uncertainty head of one pipeline."""
    paths = {"predictor": ws.predictor(outer, inner), "uncertainty head": ws.uncertainty_head(outer, inner)}
    steps = {"predictor": "train_predictor.py", "uncertainty head": "train_uncertainty.py"}
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} for outer {outer} / inner {inner} not found at {path}; "
                                    f"run {steps[name]} first")
    return load_predictor(paths["predictor"], device), load_uncertainty_head(paths["uncertainty head"], device)


def make_env(cfg, cohort: Cohort, ids: list[str], predictor, head, pathway, device) -> AcquisitionEnv:
    return AcquisitionEnv.from_config(cfg, cohort.subset(ids), predictor, head, pathway, device)


def build_agent(cfg, llm: ChatLLM, pathway: ClinicalPathway, train_env: AcquisitionEnv,
                episodic: EpisodicMemory | None, semantic: SemanticMemory | None,
                use_tools: bool = True) -> tuple[SAGEAgent, dict]:
    """Agent for one pipeline; tools and case retrieval are calibrated on its training patients."""
    uncertainty_tool, predictor_tool, retriever, calibration = build_tools(
        train_env, list(range(len(train_env.cohort))), cfg)
    a = cfg.agent
    with open(resolve_path(a.prompts.decision)) as f:
        template = f.read()
    generation = dict(temperature=a.temperature, top_p=a.top_p, max_new_tokens=a.max_new_tokens,
                      min_new_tokens=a.min_new_tokens)
    agent = SAGEAgent(
        llm, pathway, template, generation,
        uncertainty_tool=uncertainty_tool if use_tools else None,
        predictor_tool=predictor_tool if use_tools else None,
        retriever=retriever if episodic is not None else None,
        episodic=episodic, semantic=semantic,
    )
    return agent, calibration


def new_episodic_memory(cfg) -> EpisodicMemory:
    e = cfg.agent.episodic
    return EpisodicMemory(k=e.k, reward_weight=e.reward_weight, candidates=e.candidates)


# =============================================================================
# Agent components (ablations)
# =============================================================================
COMPONENTS = ("tools", "episodic", "semantic")


def add_component_args(parser) -> None:
    parser.add_argument("--no-tools", action="store_true", help="ablation: no clinical tools")
    parser.add_argument("--no-episodic", action="store_true",
                        help="ablation: no episodic memory (no case retrieval, no episode recall)")
    parser.add_argument("--no-semantic", action="store_true",
                        help="ablation: no semantic memory (no learned rules, no reflection)")


def enabled_components(cfg, args) -> dict[str, bool]:
    """Components from the config, minus any switched off on the command line."""
    return {c: bool(cfg.agent.components[c]) and not getattr(args, f"no_{c}") for c in COMPONENTS}


def component_name(enabled: dict[str, bool]) -> str:
    """'full', 'base_llm', or the enabled components joined by '+', e.g. 'tools+episodic'."""
    on = [c for c in COMPONENTS if enabled[c]]
    return "full" if len(on) == len(COMPONENTS) else "+".join(on) or "base_llm"
