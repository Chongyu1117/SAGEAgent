"""Clinical decision tools: they turn the frozen predictor's numbers into text.

- Uncertainty tool: u_t with a categorical level from quantiles of the training distribution.
- Survival predictor tool: r_t and its position relative to training patients at the same stage.
- Case retrieval: the k most similar training patients (cosine similarity of
  L2-normalized embeddings, FAISS) with their event status and survival time.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..env import AcquisitionEnv, State


class QuantileLevels:
    """Maps values to ordered categories using quantiles of a reference sample."""

    def __init__(self, reference: np.ndarray, labels: Sequence[str]):
        self.labels = list(labels)
        self.edges = np.percentile(np.asarray(reference, dtype=np.float64),
                                   np.linspace(0, 100, len(self.labels) + 1)[1:-1])

    def index(self, value: float) -> int:
        return int(np.searchsorted(self.edges, value, side="right"))

    def level(self, value: float) -> str:
        return self.labels[self.index(value)]

    def interval(self, value: float) -> str:
        """Readable comparison with the neighbouring thresholds, e.g. '0.003 ≤ 0.023 < 0.061'."""
        i = self.index(value)
        lower = f"{self.edges[i - 1]:.3f} ≤ " if i > 0 else ""
        upper = f" < {self.edges[i]:.3f}" if i < len(self.edges) else ""
        return f"{lower}{value:.3f}{upper}"

    def to_dict(self) -> dict:
        return {"labels": self.labels, "edges": self.edges.tolist()}


class UncertaintyTool:
    def __init__(self, levels: QuantileLevels):
        self.levels = levels

    def __call__(self, state: State) -> str:
        u = state.uncertainty
        return f"Uncertainty: {u:.3f} → {self.levels.level(u).upper()} ({self.levels.interval(u)})"


class SurvivalPredictorTool:
    def __init__(self, levels_by_depth: dict[int, QuantileLevels]):
        self.levels_by_depth = levels_by_depth

    def __call__(self, state: State) -> str:
        level = self.levels_by_depth[state.depth].level(state.risk)
        return f"Risk score: {state.risk:.3f} ({level} among training patients at this stage)"


class CaseRetriever:
    """Nearest training patients by cosine similarity of their full-workup embeddings."""

    def __init__(self, embeddings: np.ndarray, patient_ids: Sequence[str], event: np.ndarray,
                 time: np.ndarray, k: int = 3):
        import faiss

        vectors = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors.astype(np.float32))
        self.ids = list(patient_ids)
        self.event, self.time, self.k = event, time, k

    def search(self, embedding: np.ndarray, exclude: str | None = None) -> list[dict]:
        query = (embedding / (np.linalg.norm(embedding) + 1e-8)).astype(np.float32)[None]
        scores, idx = self.index.search(query, min(self.k + 1, self.index.ntotal))
        hits = [{"patient_id": self.ids[i], "similarity": float(s),
                 "outcome": "deceased" if self.event[i] > 0.5 else "censored", "time": float(self.time[i])}
                for s, i in zip(scores[0], idx[0]) if i >= 0 and self.ids[i] != exclude]
        return hits[: self.k]

    @staticmethod
    def describe(hits: list[dict]) -> str:
        if not hits:
            return "Similar patients: none."
        lines = ["Similar patients:"]
        lines += [f"  {n}. {h['patient_id']} (sim={h['similarity']:.2f}): {h['outcome']}, time={h['time']:.0f}"
                  for n, h in enumerate(hits, 1)]
        return "\n".join(lines)


def build_tools(env: AcquisitionEnv, train_patients: Sequence[int], cfg):
    """Calibrate the tools on the training patients of one pipeline.

    Returns (uncertainty_tool, predictor_tool, retriever, calibration_summary).
    """
    pathway = env.pathway
    depths = range(pathway.initial, len(pathway) + 1)
    patients = [p for p in train_patients if env.available_depth[p] == len(pathway)]
    states = {d: env.observe(patients, d) for d in depths}

    unc_levels = QuantileLevels([s.uncertainty for d in depths for s in states[d]],
                                cfg.agent.tools.uncertainty_levels)
    risk_levels = {d: QuantileLevels([s.risk for s in states[d]], cfg.agent.tools.risk_levels) for d in depths}

    full = env.observe(train_patients, env.available_depth[np.asarray(train_patients)])
    retriever = CaseRetriever(np.stack([s.embedding for s in full]), env.cohort.ids[train_patients],
                              env.cohort.event[train_patients], env.cohort.time[train_patients],
                              k=cfg.agent.episodic.k)
    summary = {"uncertainty": unc_levels.to_dict(), "risk": {d: lv.to_dict() for d, lv in risk_levels.items()}}
    return UncertaintyTool(unc_levels), SurvivalPredictorTool(risk_levels), retriever, summary
