"""Create nested cross-validation splits for a cohort.

Outer folds partition the patients with every modality (the evaluation set).
Within each outer fold, the remaining complete patients and all patients with
missing modalities are split into inner train/validation folds. Splits are
stratified by event status (and by the cohort's `strata`, e.g. tumor grade).

    python scripts/make_splits.py --config configs/glioma.yaml
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sageagent.clinical import ClinicalPathway  # noqa: E402
from sageagent.config import base_parser, load_config, resolve_path  # noqa: E402
from sageagent.data import Cohort, make_nested_splits  # noqa: E402


def main():
    args = base_parser(__doc__.split("\n\n")[0]).parse_args()
    cfg = load_config(args.config, args.set)
    pathway = ClinicalPathway.from_config(cfg)
    cohort = Cohort.load(resolve_path(cfg.data.cohort), pathway.names)
    splits = make_nested_splits(cohort, cfg.cv.n_outer, cfg.cv.n_inner, cfg.experiment.seed)
    path = resolve_path(cfg.data.splits)
    splits.save(path)
    print(f"{splits.meta['n_patients']} patients ({splits.meta['n_complete']} complete) -> {path}")
    for outer in range(1, splits.n_outer + 1):
        sizes = [f"{len(splits.train(outer, i))}/{len(splits.val(outer, i))}" for i in range(1, splits.n_inner + 1)]
        print(f"  outer {outer}: {len(splits.test(outer))} test | inner train/val {', '.join(sizes)}")


if __name__ == "__main__":
    main()
