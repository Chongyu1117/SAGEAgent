"""Running the agent: experience accumulation (training) and evaluation episodes."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from ..clinical import PREDICT
from ..env import AcquisitionEnv, State
from .agent import Decision, SAGEAgent
from .reflection import Reflection


def _step_record(state: State, action: str, reward: float, decision: Decision | None,
                 adherence: dict | None) -> dict:
    return {
        "depth": state.depth, "action": action, "reward": reward, "uncertainty": state.uncertainty,
        "risk": state.risk, "embedding": state.embedding if decision is not None else None,
        "adherence": adherence or {}, "parsed": True if decision is None else decision.parsed,
        "reasoning": None if decision is None else decision.reasoning,
    }


def _finish(env: AcquisitionEnv, patient: int, start: int, state: State, steps: list[dict],
            uncertainties: list[float], reference_risk: float | None = None) -> dict:
    if reference_risk is None:
        reference_risk = float(env.reference_risk([patient])[0])
    return {
        "patient_id": str(env.cohort.ids[patient]), "start_depth": start, "final_depth": state.depth,
        "burden": state.burden, "total_reward": float(sum(s["reward"] for s in steps)),
        "final_risk": state.risk, "final_uncertainty": state.uncertainty,
        "risk_error": float(abs(state.risk - reference_risk)),
        "uncertainties": uncertainties, "steps": steps,
    }


def run_episode(agent: SAGEAgent, env: AcquisitionEnv, patient: int, start_depth: int,
                exclude_self: bool = True) -> dict:
    """One training episode; the patient itself is hidden from retrieval."""
    patient_id = str(env.cohort.ids[patient])
    state = env.reset(patient, start_depth)
    steps, uncertainties = [], [state.uncertainty]
    while True:
        decision = None
        if len(env.valid_actions(state)) > 1:
            decision = agent.decide(state, env, exclude=patient_id if exclude_self else None)
        action = decision.action if decision else PREDICT
        adherence = (agent.semantic.adherence(env.pathway.stage(state.depth), action)
                     if decision is not None and agent.semantic is not None else None)
        next_state, reward, done = env.step(state, action)
        steps.append(_step_record(state, action, reward, decision, adherence))
        if done:
            return _finish(env, patient, start_depth, state, steps, uncertainties)
        state = next_state
        uncertainties.append(state.uncertainty)


def accumulate_experience(agent: SAGEAgent, env: AcquisitionEnv, reflection: Reflection | None,
                          patients: Sequence[int], episodes_per_patient: int, reflect_every: int,
                          rng: np.random.Generator,
                          on_episode: Callable[[dict], None] | None = None,
                          on_reflection: Callable[[dict], None] | None = None) -> None:
    """Process each training patient several times, growing memory and reflecting periodically.

    The first episode of a patient starts after the initial modalities; later
    episodes start at a random earlier decision stage, so that every stage is
    visited. Reflection runs every ``reflect_every`` patients on the episodes
    of that cycle.
    """
    pathway = env.pathway
    cycle: list[dict] = []
    for count, patient in enumerate(patients, start=1):
        last_start = min(int(env.available_depth[patient]), len(pathway)) - 1
        for e in range(episodes_per_patient):
            start = pathway.initial if e == 0 else int(rng.integers(pathway.initial, last_start + 1))
            episode = run_episode(agent, env, patient, start)
            if agent.episodic is not None:
                agent.episodic.add_episode(episode)
            cycle.append(episode)
            if on_episode:
                on_episode(episode)
        if reflection is not None and count % reflect_every == 0:
            entry = reflection.run(cycle)
            cycle = []
            if on_reflection:
                on_reflection(entry)


def run_test_episodes(agent: SAGEAgent, env: AcquisitionEnv, patients: Sequence[int],
                      batch_size: int = 8) -> list[dict]:
    """Evaluation episodes from the first decision stage.

    All patients advance stage by stage: the agent decides for every patient
    still in the workup (in batches of ``batch_size``), then the patients who
    acquire move to the next stage together.
    """
    start = env.pathway.initial
    reference = dict(zip(patients, env.reference_risk(patients)))
    active = env.observe(patients, start)
    progress = {s.patient: {"steps": [], "uncertainties": [s.uncertainty]} for s in active}
    finished: dict[int, dict] = {}
    while active:
        pending = [s for s in active if len(env.valid_actions(s)) > 1]
        decisions: dict[int, Decision] = {}
        for i in range(0, len(pending), batch_size):
            chunk = pending[i:i + batch_size]
            decisions.update(zip((s.patient for s in chunk), agent.decide_batch(chunk, env)))
        acquiring = [s for s in active if s.patient in decisions and decisions[s.patient].action != PREDICT]
        advanced = {s.patient: s for s in env.observe([s.patient for s in acquiring],
                                                      [s.depth + 1 for s in acquiring])} if acquiring else {}
        next_active = []
        for state in active:
            decision = decisions.get(state.patient)
            action = decision.action if decision else PREDICT
            reward = (env.terminal_reward(state) if action == PREDICT
                      else env.stage_reward(state, advanced[state.patient]))
            record = _step_record(state, action, reward, decision, None)
            record["prompt"] = None if decision is None else decision.prompt
            record.pop("embedding")
            p = progress[state.patient]
            p["steps"].append(record)
            if action == PREDICT:
                finished[state.patient] = _finish(env, state.patient, start, state, p["steps"],
                                                  p["uncertainties"], float(reference[state.patient]))
            else:
                p["uncertainties"].append(advanced[state.patient].uncertainty)
                next_active.append(advanced[state.patient])
        active = next_active
    return [finished[p] for p in patients]
