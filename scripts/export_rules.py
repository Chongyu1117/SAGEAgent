"""Collect the rules learned by an agent run (step 3) into one rule file.

The file holds the active rules of every pipeline, keyed outer_k/inner_m, and can
be evaluated without step 3:

    python scripts/export_rules.py outputs/glioma/agent/full --out rules/my_rules.json
    python evaluate.py --config configs/glioma.yaml --rules rules/my_rules.json --no-episodic
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sageagent.agent import SemanticMemory, save_rule_file  # noqa: E402
from sageagent.clinical import ClinicalPathway  # noqa: E402
from sageagent.config import load_config, resolve_path  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", help="agent run folder written by run_agent.py, e.g. outputs/glioma/agent/full")
    parser.add_argument("--out", required=True, help="rule file to write")
    parser.add_argument("--config", default="configs/glioma.yaml", help="experiment config (clinical pathway)")
    parser.add_argument("--description", default="", help="free text stored in the rule file")
    args = parser.parse_args()

    pathway = ClinicalPathway.from_config(load_config(args.config))
    rule_sets = {}
    for path in glob.glob(os.path.join(args.run, "outer_*", "inner_*", "semantic.json")):
        outer, inner = map(int, re.findall(r"outer_(\d+)[/\\]inner_(\d+)", path)[-1])
        rule_sets[(outer, inner)] = SemanticMemory(pathway).load(path).export_rules()
    if not rule_sets:
        raise SystemExit(f"no semantic memory (outer_*/inner_*/semantic.json) under {args.run}")

    out = resolve_path(args.out)
    save_rule_file(out, {f"outer_{o}/inner_{i}": rule_sets[(o, i)] for o, i in sorted(rule_sets)}, args.description)
    n_rules = [len(rules) for rules in rule_sets.values()]
    print(f"{sum(n_rules)} rules from {len(rule_sets)} pipelines ({min(n_rules)}-{max(n_rules)} each) -> {out}")


if __name__ == "__main__":
    main()
