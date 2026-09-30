"""Sequential modality acquisition environment.

At decision stage t the first t modalities of the clinical pathway have been
acquired. The frozen predictor gives the embedding e_t and risk r_t, and the
uncertainty head gives u_t. The agent either acquires m_{t+1} or stops:

    acquire m_{t+1}:  R_stage = -b(m_{t+1}) + alpha * max(0, u_t - u_{t+1})
    stop at t:        R_term  = (1 - u_t) - lambda * B_t,   B_t = sum_{i<=t} b(m_i)

The environment is stateless: ``step(state, action)`` returns the next state,
so many patients can be advanced in parallel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .clinical import PREDICT, ClinicalPathway
from .data import Cohort
from .models import SurvivalPredictor, UncertaintyHead


@dataclass
class State:
    patient: int              # row in the environment's cohort
    depth: int                # number of acquired modalities (a prefix of the pathway)
    mask: np.ndarray          # (M,) acquired modalities a_t
    embedding: np.ndarray     # (d,) e_t
    risk: float               # r_t
    uncertainty: float        # u_t
    burden: float             # B_t


class AcquisitionEnv:
    def __init__(self, cohort: Cohort, predictor: SurvivalPredictor, uncertainty_head: UncertaintyHead,
                 pathway: ClinicalPathway, alpha: float = 1.0, lam: float = 1.0, device: str = "cpu"):
        self.cohort = cohort
        self.predictor = predictor
        self.head = uncertainty_head
        self.pathway = pathway
        self.alpha = alpha
        self.lam = lam
        self.device = device
        self.available_depth = np.array([pathway.available_depth(m) for m in cohort.mask])

    @classmethod
    def from_config(cls, cfg, cohort, predictor, uncertainty_head, pathway, device) -> "AcquisitionEnv":
        return cls(cohort, predictor, uncertainty_head, pathway, cfg.reward.alpha, cfg.reward["lambda"], device)

    # ------------------------------------------------------------------ observe
    @torch.no_grad()
    def observe(self, patients: Sequence[int], depths: Sequence[int] | int) -> list[State]:
        """States of several patients at the given prefix depths (one batched forward pass)."""
        patients = np.asarray(patients, dtype=np.int64)
        depths = np.broadcast_to(np.asarray(depths, dtype=np.int64), patients.shape)
        if np.any(depths > self.available_depth[patients]) or np.any(depths < 1):
            raise ValueError("requested modalities that a patient does not have")
        masks = np.stack([self.pathway.prefix_mask(d) for d in depths])
        features = self.cohort.features[patients] * masks[:, :, None]
        mask_t = torch.from_numpy(masks).to(self.device)
        emb, risk = self.predictor.encode(torch.from_numpy(features).to(self.device), mask_t)
        unc = self.head(emb, mask_t)
        emb, risk, unc = emb.cpu().numpy(), risk.cpu().numpy(), unc.cpu().numpy()
        return [State(int(p), int(d), masks[k], emb[k], float(risk[k]), float(unc[k]), self.pathway.burden_of(d))
                for k, (p, d) in enumerate(zip(patients, depths))]

    def reset(self, patient: int, depth: int | None = None) -> State:
        return self.observe([patient], self.pathway.initial if depth is None else depth)[0]

    @torch.no_grad()
    def reference_risk(self, patients: Sequence[int]) -> np.ndarray:
        """Risk with every modality the patient has (the 'full workup' prediction)."""
        patients = np.asarray(patients, dtype=np.int64)
        mask = torch.from_numpy(self.cohort.mask[patients]).to(self.device)
        _, risk = self.predictor.encode(torch.from_numpy(self.cohort.features[patients]).to(self.device), mask)
        return risk.cpu().numpy()

    # --------------------------------------------------------------------- act
    def valid_actions(self, state: State) -> list[str]:
        actions = [PREDICT]
        if state.depth < self.available_depth[state.patient]:
            actions.append(self.pathway.acquire_action(state.depth))
        return actions

    def terminal_reward(self, state: State) -> float:
        return (1.0 - state.uncertainty) - self.lam * state.burden

    def stage_reward(self, state: State, next_state: State) -> float:
        cost = float(self.pathway.burdens[state.depth])
        return -cost + self.alpha * max(0.0, state.uncertainty - next_state.uncertainty)

    def step(self, state: State, action: str) -> tuple[State, float, bool]:
        """Apply an action; returns (next_state, reward, done)."""
        if action not in self.valid_actions(state):
            raise ValueError(f"invalid action {action} at depth {state.depth}")
        if action == PREDICT:
            return state, self.terminal_reward(state), True
        next_state = self.observe([state.patient], state.depth + 1)[0]
        return next_state, self.stage_reward(state, next_state), False
