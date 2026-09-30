"""The clinical pathway: ordered diagnostic modalities m_1 -> ... -> m_N with burdens b(m).

Because the order is clinically mandated, the acquired modalities after t steps
always form a prefix of the pathway. A state is therefore described by its
depth (number of acquired modalities), and each decision is binary: acquire the
next modality or stop and predict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PREDICT = "PREDICT"


@dataclass(frozen=True)
class Modality:
    name: str
    label: str
    description: str
    burden: float


class ClinicalPathway:
    def __init__(self, modalities: list[Modality], initial_modalities: int = 1, disease: str = ""):
        if not modalities:
            raise ValueError("the pathway needs at least one modality")
        if not 1 <= initial_modalities < len(modalities):
            raise ValueError("initial_modalities must be between 1 and N-1")
        names = [m.name for m in modalities]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate modality names: {names}")
        self.modalities = list(modalities)
        self.initial = initial_modalities
        self.disease = disease

    @classmethod
    def from_config(cls, cfg) -> "ClinicalPathway":
        clinical = cfg.clinical
        modalities = [Modality(m.name, m.label, m.description, float(m.burden)) for m in clinical.modalities]
        return cls(modalities, int(clinical.initial_modalities), clinical.disease)

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self.modalities)

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.modalities]

    @property
    def burdens(self) -> np.ndarray:
        return np.array([m.burden for m in self.modalities], dtype=np.float32)

    @property
    def decision_depths(self) -> list[int]:
        """Depths at which the agent decides: after the initial modalities up to N-1."""
        return list(range(self.initial, len(self)))

    def prefix_mask(self, depth: int) -> np.ndarray:
        mask = np.zeros(len(self), dtype=np.float32)
        mask[:depth] = 1.0
        return mask

    def depth_of(self, mask: np.ndarray) -> int:
        return int(np.asarray(mask).sum().round())

    def burden_of(self, depth: int) -> float:
        return float(self.burdens[:depth].sum())

    def available_depth(self, availability: np.ndarray) -> int:
        """Longest prefix of the pathway that a patient actually has."""
        depth = 0
        for present in availability:
            if present < 0.5:
                break
            depth += 1
        return depth

    # ------------------------------------------------------------ stages/actions
    def stage(self, depth: int) -> str:
        """Name of the decision point after `depth` modalities, e.g. 'after_radiology'."""
        return f"after_{self.modalities[depth - 1].name}"

    @property
    def stages(self) -> list[str]:
        return [self.stage(d) for d in self.decision_depths]

    def acquire_action(self, depth: int) -> str:
        return f"ACQUIRE_{self.modalities[depth].name.upper()}"

    def guidance_options(self) -> list[str]:
        """Allowed `action_guidance` values for learned rules."""
        return ["predict now"] + [f"acquire {m.name}" for m in self.modalities[self.initial:]]

    def describe(self) -> str:
        """One-line summary of the order and burdens, e.g. for prompts."""
        return " → ".join(f"{m.label} ({m.burden:.2f})" for m in self.modalities)
