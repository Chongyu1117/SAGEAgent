"""Step 2: train the calibrated uncertainty head on top of each frozen predictor.

    python train_uncertainty.py --config configs/glioma.yaml [--outer 1] [--inner 1] [--device cuda:0]
"""

from __future__ import annotations

import os

from sageagent.config import base_parser, load_config, save_config, select_folds
from sageagent.models import load_predictor, save_uncertainty_head, train_uncertainty_head
from sageagent.pipeline import Workspace, load_data
from sageagent.utils import fold_seed, get_logger, resolve_device, set_seed, update_json

log = get_logger()


def main():
    parser = base_parser("Train the calibrated uncertainty head (nested cross-validation).")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    device = resolve_device(args.device)
    ws = Workspace(cfg)
    pathway, cohort, splits = load_data(cfg)
    save_config(cfg, f"{ws.root}/predictors/uncertainty_config.yaml")

    for outer in select_folds(args.outer, splits.n_outer):
        for inner in select_folds(args.inner, splits.n_inner):
            path = ws.predictor(outer, inner)
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found; run train_predictor.py first")
            set_seed(fold_seed(cfg.experiment.seed, outer, inner))
            predictor = load_predictor(path, device)
            train, val = cohort.subset(splits.train(outer, inner)), cohort.subset(splits.val(outer, inner))
            head, metrics = train_uncertainty_head(predictor, train, val, pathway, cfg, device)
            save_uncertainty_head(head, ws.uncertainty_head(outer, inner), tau=cfg.uncertainty.tau, **metrics)
            update_json(f"{ws.root}/predictors/uncertainty_summary.json", {f"outer_{outer}/inner_{inner}": metrics})
            auroc = "n/a" if metrics["auroc"] is None else f"{metrics['auroc']:.3f}"
            log.info(f"[outer {outer} / inner {inner}] uncertainty head: {metrics['n_train']} training prefixes, "
                     f"validation AUROC {auroc}, ECE {metrics['ece']:.3f}, T={metrics['temperature']:.3f}")


if __name__ == "__main__":
    main()
