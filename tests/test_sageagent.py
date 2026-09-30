"""End-to-end checks on a small synthetic cohort with a scripted LLM (CPU, no downloads).

    python -m pytest tests            # or: python tests/test_sageagent.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sageagent.agent import (EpisodicMemory, Reflection, SemanticMemory, accumulate_experience,  # noqa: E402
                             load_rule_file, run_test_episodes, save_rule_file)
from sageagent.clinical import PREDICT, ClinicalPathway  # noqa: E402
from sageagent.config import load_config, resolve_path  # noqa: E402
from sageagent.data import Cohort, NestedSplits, check_splits, make_nested_splits  # noqa: E402
from sageagent.env import AcquisitionEnv  # noqa: E402
from sageagent.metrics import bootstrap_ci, concordance_index, paired_bootstrap, summarize  # noqa: E402
from sageagent.models import train_uncertainty_head  # noqa: E402
from sageagent.pipeline import build_agent, new_episodic_memory  # noqa: E402
from sageagent.utils import save_json  # noqa: E402

OVERRIDES = [
    "predictor.hidden_dim=16", "predictor.n_layers=1", "predictor.n_heads=2", "predictor.head_hidden_dims=[8]",
    "predictor.train.epochs=3", "predictor.train.min_epochs=1", "predictor.train.batch_size=16",
    "uncertainty.hidden_dim=8", "uncertainty.train.epochs=5",
    "experience.episodes_per_patient=2", "experience.reflect_every=3", "evaluation.n_bootstrap=20",
]


class ScriptedLLM:
    """Stands in for the chat LLM: alternates PREDICT / ACQUIRE and returns one valid and one invalid rule."""

    def __init__(self):
        self.calls = 0

    def chat(self, system, users, **kwargs):
        replies = []
        for user in users:
            self.calls += 1
            if "Answer with one JSON object" in user:
                replies.append("```json\n" + json.dumps({
                    "new_rules": [
                        {"pattern": "After radiology, stop when uncertainty is low and risk error is small.",
                         "stage": "after_radiology", "direction": "stop", "action_guidance": "predict now",
                         "supporting_evidence": "good episodes", "confidence": 0.9},
                        {"pattern": "Invalid stage.", "stage": "after_genomics", "direction": "stop",
                         "action_guidance": "predict now"},
                    ],
                    "update_rules": [], "deprecate_rules": []}) + "\n```")
            else:
                replies.append("Step 1: ...\nStep 3: ...\nACTION: " + ("PREDICT" if self.calls % 2 else "ACQUIRE"))
        return replies

    def generate(self, system, user, **kwargs):
        return self.chat(system, [user], **kwargs)[0]


def synthetic_cohort(n: int = 80, seed: int = 0) -> Cohort:
    rng = np.random.default_rng(seed)
    dims = [3, 5, 4, 6]
    mask = np.ones((n, 4), np.float32)
    partial = rng.random(n) < 0.4
    mask[partial, 1 + rng.integers(0, 3, partial.sum())] = 0.0
    features = np.zeros((n, 4, max(dims)), np.float32)
    for j, d in enumerate(dims):
        features[:, j, :d] = rng.normal(size=(n, d))
    features *= mask[:, :, None]
    signal = features[:, :, 0].sum(axis=1)
    time = np.exp(-0.5 * signal + rng.normal(scale=0.3, size=n)) * 100
    return Cohort(np.array([f"P{i:03d}" for i in range(n)]), features, mask,
                  (rng.random(n) < 0.7).astype(np.float32), time.astype(np.float32), dims,
                  rng.integers(0, 3, n))


def setup():
    cfg = load_config("configs/glioma.yaml", OVERRIDES)
    pathway = ClinicalPathway.from_config(cfg)
    cohort = synthetic_cohort()
    splits = make_nested_splits(cohort, n_outer=3, n_inner=2, seed=1)
    return cfg, pathway, cohort, splits


def trained_models(cfg, pathway, cohort, splits):
    from train_predictor import train_fold

    train, val = cohort.subset(splits.train(1, 1)), cohort.subset(splits.val(1, 1))
    predictor, info = train_fold(cfg, train, val, "cpu", seed=0)
    head, metrics = train_uncertainty_head(predictor, train, val, pathway, cfg, "cpu")
    return predictor.eval(), head, info, metrics


# ----------------------------------------------------------------------------- tests
def test_concordance_index_matches_brute_force():
    rng = np.random.default_rng(1)
    risk, event, time = rng.normal(size=40), rng.random(40) < 0.6, rng.random(40)
    risk[:5] = risk[5]                      # ties in risk
    num = den = 0.0
    for i in range(40):
        for j in range(40):
            if event[i] and time[j] > time[i]:
                den += 1
                num += 1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0
    assert abs(concordance_index(risk, event, time) - num / den) < 1e-12


def test_cohort_roundtrip_and_splits():
    cfg, pathway, cohort, splits = setup()
    with tempfile.TemporaryDirectory() as tmp:
        cohort.save(f"{tmp}/c.npz", pathway.names)
        loaded = Cohort.load(f"{tmp}/c.npz", pathway.names)
        assert loaded.modality_dims == cohort.modality_dims
        assert np.allclose(loaded.features, cohort.features) and (loaded.ids == cohort.ids).all()
        splits.save(f"{tmp}/s.json")
        again = NestedSplits.load(f"{tmp}/s.json")
        assert again.test(2) == splits.test(2) and again.train(3, 2) == splits.train(3, 2)
    assert sorted(sum((splits.test(o) for o in (1, 2, 3)), [])) == sorted(cohort.ids[cohort.complete])


def test_rewards_follow_the_paper():
    cfg, pathway, cohort, splits = setup()
    predictor, head, _, metrics = trained_models(cfg, pathway, cohort, splits)
    assert 0.0 < metrics["temperature"] <= 10.0
    env = AcquisitionEnv(cohort.subset(splits.test(1)), predictor, head, pathway, alpha=1.0, lam=1.0)
    state = env.reset(0)
    assert state.depth == pathway.initial and abs(state.burden - 0.03) < 1e-6
    assert env.valid_actions(state) == [PREDICT, "ACQUIRE_RADIOLOGY"]
    nxt, reward, done = env.step(state, "ACQUIRE_RADIOLOGY")
    assert not done and abs(reward - (-0.14 + max(0.0, state.uncertainty - nxt.uncertainty))) < 1e-6
    _, reward, done = env.step(nxt, PREDICT)
    assert done and abs(reward - ((1 - nxt.uncertainty) - nxt.burden)) < 1e-6


def test_action_parsing():
    cfg, pathway, cohort, splits = setup()
    agent_cls = __import__("sageagent.agent", fromlist=["SAGEAgent"]).SAGEAgent
    agent = agent_cls(None, pathway, "", {})
    valid = [PREDICT, "ACQUIRE_PATHOLOGY"]
    parse = lambda text: agent.parse_action(text, valid)[:2]  # noqa: E731
    assert parse("reasoning...\nACTION: PREDICT (skip pathology)") == (PREDICT, True)
    assert parse("**ACTION:** ACQUIRE_PATHOLOGY") == ("ACQUIRE_PATHOLOGY", True)
    assert parse("I would acquire pathology.\nACTION: pathology") == ("ACQUIRE_PATHOLOGY", True)
    assert parse("no decision here") == ("ACQUIRE_PATHOLOGY", False)
    # text after the decision (e.g. a hallucinated next turn) is dropped and never parsed
    action, _, reasoning = agent.parse_action("Low risk.\nACTION: PREDICT\nuser\n... ACTION: ACQUIRE_PATHOLOGY", valid)
    assert action == PREDICT and reasoning == "Low risk.\nACTION: PREDICT"


def test_experience_reflection_and_evaluation():
    cfg, pathway, cohort, splits = setup()
    predictor, head, _, _ = trained_models(cfg, pathway, cohort, splits)
    llm = ScriptedLLM()
    train_ids = [str(i) for i in cohort.subset(splits.train(1, 1)).ids if i in set(cohort.ids[cohort.complete])]
    train_env = AcquisitionEnv(cohort.subset(train_ids), predictor, head, pathway)
    episodic, semantic = new_episodic_memory(cfg), SemanticMemory.from_config(cfg, pathway)
    agent, _ = build_agent(cfg, llm, pathway, train_env, episodic, semantic)
    with open(resolve_path(cfg.agent.prompts.reflection)) as f:
        reflection = Reflection(llm, semantic, pathway, f.read(), cfg)

    accumulate_experience(agent, train_env, reflection, list(range(9)), 2, 3, np.random.default_rng(0))
    assert len(episodic.episodes) == 18 and len(episodic) > 0
    assert reflection.cycle == 3 and semantic.active("after_radiology", "stop")
    assert all("after_genomics" not in r["stage"] for r in semantic.rules)
    first = reflection.log[0]
    assert first["parsed"] and first["rejected"][0]["reason"].startswith("unknown stage")
    assert semantic.rules[0]["confidence"] == 0.8          # clipped to the allowed range

    with tempfile.TemporaryDirectory() as tmp:
        episodic.save(f"{tmp}/e.json")
        semantic.save(f"{tmp}/s.json")
        episodic = EpisodicMemory(k=3).load(f"{tmp}/e.json")
        semantic = SemanticMemory.from_config(cfg, pathway).load(f"{tmp}/s.json")
    test_env = AcquisitionEnv(cohort.subset(splits.test(1)), predictor, head, pathway)
    agent, _ = build_agent(cfg, llm, pathway, train_env, episodic, semantic)
    episodes = run_test_episodes(agent, test_env, list(range(len(test_env.cohort))), batch_size=4)
    assert len(episodes) == len(test_env.cohort)
    assert all(pathway.initial <= e["final_depth"] <= len(pathway) for e in episodes)
    assert all(abs(e["burden"] - pathway.burden_of(e["final_depth"])) < 1e-6 for e in episodes)

    risk = np.array([e["final_risk"] for e in episodes])
    burden = np.array([e["burden"] for e in episodes])
    fold = np.arange(len(episodes)) % 2 + 1
    stats = summarize(risk, test_env.cohort.event, test_env.cohort.time, burden, fold)
    ci = bootstrap_ci(risk, test_env.cohort.event, test_env.cohort.time, burden, fold, n_boot=20)
    assert ci["burden"][0] <= stats["burden"] <= ci["burden"][1]
    same = {"risk": risk, "event": test_env.cohort.event, "time": test_env.cohort.time, "burden": burden, "fold": fold}
    assert paired_bootstrap(same, same, n_boot=10)["c_index"]["delta"] == 0.0


def test_rule_files():
    cfg, pathway, _, _ = setup()
    semantic = SemanticMemory.from_config(cfg, pathway)
    semantic.add({"stage": "after_radiology", "direction": "stop", "pattern": "Low uncertainty after MRI.",
                  "action_guidance": "predict now", "confidence": 0.7}, cycle=1)
    semantic.rules[0].update(followed=4, followed_good=3, effectiveness=0.75)
    with tempfile.TemporaryDirectory() as tmp:
        save_rule_file(f"{tmp}/rules.json", {"outer_1/inner_1": semantic.export_rules()})
        loaded = SemanticMemory.from_config(cfg, pathway).import_rules(
            load_rule_file(f"{tmp}/rules.json", "outer_1/inner_1"))
        assert loaded.export_rules() == semantic.export_rules()
        assert loaded.describe("after_radiology") == semantic.describe("after_radiology")
        # a file with a single rule set applies to every pipeline; rules are checked against the pathway
        save_json({"rules": [{"stage": "after_genomics", "direction": "stop", "pattern": "Too late.",
                              "action_guidance": "predict now"}]}, f"{tmp}/single.json")
        try:
            SemanticMemory.from_config(cfg, pathway).import_rules(load_rule_file(f"{tmp}/single.json", "outer_2/inner_3"))
        except ValueError as err:
            assert "unknown stage" in str(err)
        else:
            raise AssertionError("a rule for a decision point that does not exist was accepted")


def test_released_embeddings_and_rules():
    cfg = load_config("configs/glioma.yaml")
    pathway = ClinicalPathway.from_config(cfg)
    cohort = Cohort.load(resolve_path(cfg.data.cohort), pathway.names)
    splits = NestedSplits.load(resolve_path(cfg.data.splits))
    check_splits(splits, cohort)
    assert len(cohort) == 962 and int(cohort.complete.sum()) == 170
    for outer in range(1, splits.n_outer + 1):
        for inner in range(1, splits.n_inner + 1):
            rules = load_rule_file(resolve_path("rules/glioma.json"), f"outer_{outer}/inner_{inner}")
            semantic = SemanticMemory.from_config(cfg, pathway).import_rules(rules)
            assert len(semantic.active()) == len(rules) > 0


if __name__ == "__main__":
    torch.set_num_threads(4)
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
