"""Paired bootstrap comparison of two evaluation runs on the same patients.

    python scripts/compare_runs.py outputs/glioma/eval/full outputs/glioma/eval/base_llm
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sageagent.metrics import paired_bootstrap  # noqa: E402


def read(folder: str) -> dict:
    with open(os.path.join(folder, "patients.csv")) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: (int(r["outer"]), r["patient_id"]))
    return {"ids": [(r["outer"], r["patient_id"]) for r in rows],
            **{k: np.array([float(r[k]) for r in rows]) for k in ("risk", "event", "time", "burden")},
            "fold": np.array([int(r["outer"]) for r in rows])}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("a", help="evaluation folder of method A")
    parser.add_argument("b", help="evaluation folder of method B")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    a, b = read(args.a), read(args.b)
    if a["ids"] != b["ids"]:
        raise SystemExit("the two runs were evaluated on different patients")
    result = paired_bootstrap(a, b, n_boot=args.n_bootstrap, seed=args.seed)
    print(f"A = {args.a}\nB = {args.b}\n")
    for metric, r in result.items():
        print(f"  Δ{metric:<8s} (A - B) {r['delta']:+.4f}  95% CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]  "
              f"p = {r['p']:.3f}")


if __name__ == "__main__":
    main()
