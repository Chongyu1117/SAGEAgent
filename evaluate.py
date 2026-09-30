"""Step 4: evaluate on the outer test folds.

For every outer fold, each inner pipeline's agent (frozen memory from step 3)
decides the acquisitions for the test patients. A modality counts as acquired
when a majority of the inner pipelines acquired it. The risk under the voted
modalities is averaged over the inner predictors and one C-index is computed
per outer fold. Reported: mean ± std over outer folds, mean burden,
HV = (C - 0.5)(1 - burden), and bootstrap 95% intervals.

    python evaluate.py --config configs/glioma.yaml                       # full SAGEAgent
    python evaluate.py --config configs/glioma.yaml --no-semantic         # ablation
    python evaluate.py --config configs/glioma.yaml --rules rules/glioma.json --no-episodic
                                                                          # released rules, no step 3
"""

from __future__ import annotations

import csv
import os

import numpy as np
import torch

from sageagent.agent import ChatLLM, SemanticMemory, UncertaintyThresholdPolicy, load_rule_file, run_test_episodes
from sageagent.config import base_parser, load_config, resolve_path, save_config, select_folds
from sageagent.metrics import bootstrap_ci, summarize
from sageagent.pipeline import (Workspace, add_component_args, build_agent, complete_ids, component_name,
                                enabled_components, load_data, load_fold_models, make_env, new_episodic_memory)
from sageagent.utils import fold_seed, get_logger, load_json, resolve_device, save_json, set_seed

log = get_logger()


def majority_depth(depths: list[int]) -> int:
    """Deepest prefix acquired by a majority of pipelines (modality-wise majority vote)."""
    need = (len(depths) + 1) // 2
    return max(d for d in range(1, max(depths) + 1) if sum(x >= d for x in depths) >= need)


def load_memories(cfg, ws, args, outer, inner, components, pathway):
    """Episodic memory from an agent run (--memory); rules from the same run or from a rule file (--rules)."""
    folder = ws.agent_dir(args.memory, outer, inner)
    if components["episodic"] or (components["semantic"] and not args.rules):
        if not os.path.exists(os.path.join(folder, "summary.json")):
            raise FileNotFoundError(f"no agent memory in {folder}; run run_agent.py first, pass --memory, "
                                    f"or evaluate a rule file with --rules FILE --no-episodic")
    episodic = semantic = None
    if components["episodic"]:
        episodic = new_episodic_memory(cfg).load(os.path.join(folder, "episodic.json"))
    if components["semantic"]:
        semantic = SemanticMemory.from_config(cfg, pathway)
        if args.rules:
            semantic.import_rules(load_rule_file(args.rules, f"outer_{outer}/inner_{inner}"))
        else:
            semantic.load(os.path.join(folder, "semantic.json"))
    return episodic, semantic


def main():
    parser = base_parser("Evaluate SAGEAgent with majority vote across inner pipelines.")
    add_component_args(parser)
    parser.add_argument("--memory", default="full", help="agent run whose memory is loaded (default: full)")
    parser.add_argument("--rules", default=None, metavar="FILE",
                        help="take the learned rules from a rule file (e.g. rules/glioma.json) instead of --memory")
    parser.add_argument("--name", default=None, help="evaluation name (default: from the enabled components)")
    parser.add_argument("--llm-device", default=None, help="device for the LLM (default: same as --device)")
    parser.add_argument("--overwrite", action="store_true", help="recompute pipelines that were already evaluated")
    parser.add_argument("--uncertainty-threshold", type=float, default=None, metavar="TAU",
                        help="evaluate the naive baseline that stops once u_t < TAU (no LLM) instead of the agent")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    device = resolve_device(args.device)
    components = enabled_components(cfg, args)
    threshold = args.uncertainty_threshold
    if args.rules:
        if threshold is not None or not components["semantic"]:
            parser.error("--rules needs the semantic memory; drop --no-semantic and --uncertainty-threshold")
        args.rules = resolve_path(args.rules)
    if threshold is not None:
        name = f"threshold_{threshold:g}"
    else:
        rules = f"_{os.path.splitext(os.path.basename(args.rules))[0]}" if args.rules else ""
        name = component_name(components) + rules
    name = args.name or name
    ws = Workspace(cfg)
    out = ws.eval_dir(name)
    pathway, cohort, splits = load_data(cfg)
    save_config(cfg, os.path.join(out, "config.yaml"))
    outers, inners = select_folds(args.outer, splits.n_outer), select_folds(args.inner, splits.n_inner)

    llm, rows = None, []
    for outer in outers:
        test_ids = splits.test(outer)
        depths = {}                                   # inner -> {patient_id: final depth}
        for inner in inners:
            trace_path = os.path.join(out, "traces", f"outer_{outer}", f"inner_{inner}.json")
            if os.path.exists(trace_path) and not args.overwrite:
                episodes = load_json(trace_path)
            else:
                set_seed(fold_seed(cfg.experiment.seed, outer, inner))
                predictor, head = load_fold_models(ws, outer, inner, device)
                test_env = make_env(cfg, cohort, test_ids, predictor, head, pathway, device)
                if threshold is not None:
                    agent = UncertaintyThresholdPolicy(threshold)
                else:
                    if llm is None:
                        log.info(f"loading {cfg.agent.llm}")
                        llm = ChatLLM(cfg.agent.llm, resolve_device(args.llm_device or device), cfg.agent.dtype)
                    train_env = make_env(cfg, cohort, complete_ids(cohort, splits.train(outer, inner)), predictor,
                                         head, pathway, device)
                    episodic, semantic = load_memories(cfg, ws, args, outer, inner, components, pathway)
                    agent, _ = build_agent(cfg, llm, pathway, train_env, episodic, semantic, components["tools"])
                episodes = run_test_episodes(agent, test_env, list(range(len(test_ids))), cfg.agent.batch_size)
                if not cfg.evaluation.save_traces:
                    episodes = [{k: v for k, v in e.items() if k != "steps"} for e in episodes]
                save_json(episodes, trace_path)
            depths[inner] = {e["patient_id"]: e["final_depth"] for e in episodes}
            log.info(f"[outer {outer} / inner {inner}] mean burden "
                     f"{np.mean([pathway.burden_of(d) for d in depths[inner].values()]):.3f}")

        voted = {pid: majority_depth([depths[i][pid] for i in inners]) for pid in test_ids}
        test = cohort.subset(test_ids)
        masks = np.stack([pathway.prefix_mask(voted[pid]) for pid in test_ids])
        risk = np.zeros(len(test_ids))
        for inner in inners:
            predictor, _ = load_fold_models(ws, outer, inner, device)
            _, r = predictor.encode(torch.from_numpy(test.features * masks[:, :, None]).to(device),
                                    torch.from_numpy(masks).to(device))
            risk += r.cpu().numpy() / len(inners)
        for k, pid in enumerate(test_ids):
            rows.append({"outer": outer, "patient_id": pid, "event": float(test.event[k]),
                         "time": float(test.time[k]), "depth": voted[pid], "burden": pathway.burden_of(voted[pid]),
                         "risk": float(risk[k]), **{f"depth_inner_{i}": depths[i][pid] for i in inners}})

    arr = {key: np.array([row[key] for row in rows]) for key in ("risk", "event", "time", "burden", "outer")}
    metrics = summarize(arr["risk"], arr["event"], arr["time"], arr["burden"], arr["outer"])
    ci = bootstrap_ci(arr["risk"], arr["event"], arr["time"], arr["burden"], arr["outer"],
                      n_boot=cfg.evaluation.n_bootstrap, seed=cfg.experiment.seed)
    acquired = {m.name: int(sum(row["depth"] > j for row in rows)) for j, m in enumerate(pathway.modalities)}
    uses_run = components["episodic"] or (components["semantic"] and not args.rules)
    policy = ({"uncertainty_threshold": threshold} if threshold is not None
              else {"llm": cfg.agent.llm, "components": components, "memory": args.memory if uses_run else None,
                    "rules": args.rules})
    summary = {"name": name, **policy, "n_patients": len(rows), **metrics, "ci95": ci,
               "patients_acquiring": acquired}
    save_json(summary, os.path.join(out, "summary.json"))
    with open(os.path.join(out, "patients.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"{name}: C-index {metrics['c_index']:.3f} [{ci['c_index'][0]:.2f}, {ci['c_index'][1]:.2f}]  "
             f"burden {metrics['burden']:.3f} [{ci['burden'][0]:.2f}, {ci['burden'][1]:.2f}]  "
             f"HV {metrics['hv']:.3f}  acquired {acquired} of {len(rows)} patients")
    log.info(f"results written to {out}")
    if len(outers) < splits.n_outer:
        log.info(f"this summary covers outer folds {outers} only; run evaluate.py again without --outer "
                 f"to combine all folds (finished pipelines are read from their traces)")


if __name__ == "__main__":
    main()
