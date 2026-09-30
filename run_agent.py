"""Step 3: experience accumulation (agent training without gradient updates).

For every pipeline, the frozen LLM processes each complete-modality training
patient several times. Every decision is stored in episodic memory, and every
few patients a self-reflection cycle distills rules into semantic memory.

    python run_agent.py --config configs/glioma.yaml [--outer 1] [--llm-device cuda:1]
"""

from __future__ import annotations

import json
import os

import numpy as np

from sageagent.agent import ChatLLM, Reflection, SemanticMemory, accumulate_experience
from sageagent.config import base_parser, load_config, resolve_path, save_config, select_folds
from sageagent.pipeline import (Workspace, add_component_args, build_agent, complete_ids, component_name,
                                enabled_components, load_data, load_fold_models, make_env, new_episodic_memory)
from sageagent.utils import fold_seed, get_logger, resolve_device, save_json, set_seed

log = get_logger()


def main():
    parser = base_parser("Accumulate experience: episodic memory and self-reflection with a frozen LLM.")
    add_component_args(parser)
    parser.add_argument("--name", default=None, help="run name (default: from the enabled components, e.g. 'full')")
    parser.add_argument("--llm-device", default=None, help="device for the LLM (default: same as --device)")
    parser.add_argument("--overwrite", action="store_true", help="re-run pipelines that already finished")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    device = resolve_device(args.device)
    components = enabled_components(cfg, args)
    name = args.name or component_name(components)
    ws = Workspace(cfg)
    pathway, cohort, splits = load_data(cfg)
    save_config(cfg, os.path.join(ws.root, "agent", name, "config.yaml"))
    with open(resolve_path(cfg.agent.prompts.reflection)) as f:
        reflection_template = f.read()

    llm = None
    for outer in select_folds(args.outer, splits.n_outer):
        for inner in select_folds(args.inner, splits.n_inner):
            out = ws.agent_dir(name, outer, inner)
            if os.path.exists(os.path.join(out, "summary.json")) and not args.overwrite:
                log.info(f"[outer {outer} / inner {inner}] already done, skipping ({out})")
                continue
            if llm is None:
                log.info(f"loading {cfg.agent.llm}")
                llm = ChatLLM(cfg.agent.llm, resolve_device(args.llm_device or device), cfg.agent.dtype)

            seed = fold_seed(cfg.experiment.seed, outer, inner)
            set_seed(seed)
            rng = np.random.default_rng(seed)
            predictor, head = load_fold_models(ws, outer, inner, device)
            env = make_env(cfg, cohort, complete_ids(cohort, splits.train(outer, inner)), predictor, head,
                           pathway, device)
            episodic = new_episodic_memory(cfg) if components["episodic"] else None
            semantic = SemanticMemory.from_config(cfg, pathway) if components["semantic"] else None
            agent, calibration = build_agent(cfg, llm, pathway, env, episodic, semantic, components["tools"])
            reflection = Reflection(llm, semantic, pathway, reflection_template, cfg) if semantic else None

            os.makedirs(out, exist_ok=True)
            episodes = []
            with open(os.path.join(out, "episodes.jsonl"), "w") as log_file:
                def on_episode(episode):
                    episodes.append(episode)
                    record = {**episode, "steps": [{k: v for k, v in s.items() if k != "embedding"}
                                                   for s in episode["steps"]]}
                    log_file.write(json.dumps(record, default=float) + "\n")
                    log_file.flush()

                def on_reflection(entry):
                    log.info(f"[outer {outer} / inner {inner}] reflection {entry['cycle']}: "
                             f"+{len(entry['added'])} rules, {len(entry['rejected'])} rejected, "
                             f"{len(entry['deprecated_by_reflection']) + len(entry['deprecated_for_poor_outcomes'])}"
                             f" deprecated, {len(entry['active_rules'])} active")

                patients = rng.permutation(len(env.cohort))
                log.info(f"[outer {outer} / inner {inner}] {len(patients)} training patients x "
                         f"{cfg.experience.episodes_per_patient} episodes ({name})")
                accumulate_experience(agent, env, reflection, patients, cfg.experience.episodes_per_patient,
                                      cfg.experience.reflect_every, rng, on_episode, on_reflection)

            if episodic is not None:
                episodic.save(os.path.join(out, "episodic.json"))
            if semantic is not None:
                semantic.save(os.path.join(out, "semantic.json"))
                save_json(reflection.log, os.path.join(out, "reflections.json"))
            save_json(calibration, os.path.join(out, "tool_calibration.json"))
            summary = {
                "episodes": len(episodes),
                "mean_total_reward": float(np.mean([e["total_reward"] for e in episodes])),
                "mean_burden": float(np.mean([e["burden"] for e in episodes])),
                "unparsed_decisions": sum(not s["parsed"] for e in episodes for s in e["steps"]),
                "active_rules": len(semantic.active()) if semantic else None,
            }
            save_json(summary, os.path.join(out, "summary.json"))
            log.info(f"[outer {outer} / inner {inner}] done: {summary}")


if __name__ == "__main__":
    main()
