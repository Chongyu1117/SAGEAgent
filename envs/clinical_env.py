"""
Clinical Environment for Sequential Modality Acquisition.

Simulates the clinical decision process for glioma patients:
an agent sequentially decides which diagnostic modalities to acquire
before making a survival prediction.

Clinical workflow (glioma):
    Demographics → Radiology (MRI) → Pathology (biopsy) → Genomics (sequencing)

    - MRI is required before stereotactic biopsy (navigation imaging)
    - Pathology and Genomics come from the same tissue (biopsy/resection)
    - Genomics depends on Pathology (same tissue sample)

Ordering:
    - Always enforces strict clinical ordering (must follow dependency chain)
    - Demo → Rad → Path → Gen
"""

import os

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Action & modality constants
# ---------------------------------------------------------------------------
PREDICT = 0
ACQUIRE_DEMO = 1
ACQUIRE_RAD = 2
ACQUIRE_PATH = 3
ACQUIRE_GEN = 4

ACTION_NAMES = {
    PREDICT: "PREDICT",
    ACQUIRE_DEMO: "ACQUIRE_DEMO",
    ACQUIRE_RAD: "ACQUIRE_RAD",
    ACQUIRE_PATH: "ACQUIRE_PATH",
    ACQUIRE_GEN: "ACQUIRE_GEN",
}

MODALITY_NAMES = ["demographics", "radiology", "pathology", "genomics"]
N_ACTIONS = 5
N_MODALITIES = 4


class ClinicalEnv:
    """
    Clinical modality acquisition environment.

    The agent observes the current state (acquired modalities, uncertainty, …)
    and chooses to either acquire another modality or make a final prediction.

    Args:
        predictor:         Frozen ``SurvivalPredictor`` model (eval mode).
        patient_features:  (N, n_mod, feat_dim) numpy array.
        patient_masks:     (N, n_mod) numpy array — which modalities each
                           patient *actually* has in the dataset.
        patient_events:    (N,) numpy array — 1=event, 0=censored.
        patient_times:     (N,) numpy array — survival times.
        patient_names:     list of str, optional.
        clinical_burden:   dict {mod_idx: burden_value}, overrides defaults.
        budget:            maximum total clinical burden allowed per episode.
                           NOTE: default 1.0 == sum of all modality burdens,
                           so this is effectively a safety-net and never binds.
                           The agent decides to stop via uncertainty/cost-benefit
                           reasoning in the LLM prompt, not via budget exhaustion.
        reward_coeff:      weight for prediction quality in terminal reward.
        process_reward_alpha: weight for uncertainty reduction bonus on
                           acquisition steps (default 0.5).
        ucpr_beta:         weight for uncertainty-calibrated process reward
                           in terminal reward (default 0.0).
        cost_weight:       weight for burden penalty in terminal reward
                           (LA-CDM normalization).  With cost_weight=0.6,
                           stop@rad is optimal for average patients while
                           high-uncertainty patients still benefit from more
                           modalities.  Set to 0.0 to disable.
        device:            torch device string.
        max_steps:         hard cap on steps per episode (safety).
        debug:             enable verbose logging.
    """

    # Clinical dependency chain:  Demo → Rad → Path → Gen
    CLINICAL_DEPS: Dict[int, List[int]] = {
        0: [],           # Demographics: no dependencies
        1: [0],          # Radiology: requires Demographics
        2: [0, 1],       # Pathology: requires Demo + Rad (MRI for navigation)
        3: [0, 1, 2],    # Genomics: requires Demo + Rad + Path (same tissue)
    }

    # Default clinical burden — MCDA-derived (see doc/mcda_burden_quantification.csv)
    # Dimensions: monetary cost (0.25), turnaround time (0.25),
    #             invasiveness/risk (0.35), infrastructure (0.15)
    DEFAULT_BURDEN: Dict[int, float] = {
        0: 0.03,   # Demographics — chart review / clinician consultation
        1: 0.14,   # Radiology (MRI) — non-invasive, contrast, ~$1.5K, <1 day
        2: 0.53,   # Pathology (biopsy) — craniotomy/stereotactic, ~$40K, 3-7 days
        3: 0.30,   # Genomics (NGS) — ~$3K, 14-day turnaround, specialized lab
    }

    def __init__(
        self,
        predictor,
        patient_features: np.ndarray,
        patient_masks: np.ndarray,
        patient_events: np.ndarray,
        patient_times: np.ndarray,
        patient_names: Optional[List[str]] = None,
        clinical_burden: Optional[Dict[int, float]] = None,
        budget: float = 1.0,
        reward_coeff: float = 2.0,
        process_reward_alpha: float = 0.5,
        ucpr_beta: float = 0.0,
        cost_weight: float = 0.6,
        decision_signal: str = "uncertainty",
        device: str = "cuda",
        max_steps: int = 10,
        debug: bool = False,
    ):
        self.predictor = predictor
        self.predictor.eval()
        self._calibrated_head = None  # set via set_calibrated_head()
        self._cavs_head = None        # set via set_cavs_head()
        self._rdvs_head = None        # set via set_rdvs_head()
        self.decision_signal = decision_signal  # "uncertainty", "cavs", or "risk_delta"

        self.patient_features = patient_features.astype(np.float32)
        self.patient_masks = patient_masks.astype(np.float32)
        self.patient_events = patient_events.astype(np.float32)
        self.patient_times = patient_times.astype(np.float32)
        self.patient_names = patient_names or [
            f"patient_{i}" for i in range(len(patient_features))
        ]

        self.burden = clinical_burden if clinical_burden is not None else dict(self.DEFAULT_BURDEN)
        self.budget = budget
        self.reward_coeff = reward_coeff
        self.process_reward_alpha = process_reward_alpha
        self.ucpr_beta = ucpr_beta
        self.cost_weight = cost_weight
        self.device = device
        self.max_steps = max_steps
        self.debug = debug

        self.n_patients = len(patient_features)
        self.n_modalities = patient_features.shape[1]
        self.feat_dim = patient_features.shape[2]

        # Pre-compute oracle predictions (all available modalities)
        self._precompute_oracles()

        # Episode state (set in reset)
        self._patient_idx: Optional[int] = None
        self._acquired_mask: Optional[np.ndarray] = None
        self._total_burden = 0.0
        self._last_embedding: Optional[np.ndarray] = None  # cached for process reward
        self._step_count = 0
        self._done = False
        self._trajectory: List[dict] = []

    def set_calibrated_head(self, calibrated_head):
        """Attach a post-hoc calibrated uncertainty head."""
        self._calibrated_head = calibrated_head
        if self.debug:
            print("[ENV] Calibrated uncertainty head attached")

    def set_cavs_head(self, cavs_head):
        """Attach a CAVS (Concordance-Aware Value Scorer) head."""
        self._cavs_head = cavs_head
        if self.debug:
            print("[ENV] CAVS head attached")

    def set_rdvs_head(self, rdvs_head):
        """Attach an RDVS (Risk-Delta Value Scorer) head."""
        self._rdvs_head = rdvs_head
        if self.debug:
            print("[ENV] RDVS head attached")

    # ------------------------------------------------------------------
    # Oracle pre-computation
    # ------------------------------------------------------------------
    def _precompute_oracles(self):
        """Risk scores using each patient's *full* available modalities."""
        self.oracle_risks = np.zeros(self.n_patients, dtype=np.float32)

        batch_size = 64
        with torch.no_grad():
            for start in range(0, self.n_patients, batch_size):
                end = min(start + batch_size, self.n_patients)
                feat = torch.from_numpy(
                    self.patient_features[start:end]
                ).to(self.device)
                mask = torch.from_numpy(
                    self.patient_masks[start:end]
                ).to(self.device)

                self.oracle_risks[start:end] = (
                    self.predictor.get_risk_score(feat, mask).cpu().numpy()
                )

        if self.debug:
            print(
                f"[ENV DEBUG] Oracles for {self.n_patients} patients — "
                f"risk [{self.oracle_risks.min():.4f}, {self.oracle_risks.max():.4f}]"
            )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(
        self,
        patient_idx: int,
        initial_mask: Optional[np.ndarray] = None,
    ) -> dict:
        """Start a new episode for *patient_idx*."""
        self._patient_idx = patient_idx
        self._total_burden = 0.0
        self._step_count = 0
        self._done = False
        self._trajectory = []

        if initial_mask is not None:
            self._acquired_mask = initial_mask.astype(np.float32).copy()
        else:
            self._acquired_mask = np.zeros(self.n_modalities, dtype=np.float32)

        # Can only acquire what the patient actually has
        self._acquired_mask *= self.patient_masks[patient_idx]

        # Include burden of pre-acquired modalities (e.g. demographics)
        for mod_idx in range(self.n_modalities):
            if self._acquired_mask[mod_idx] > 0.5:
                self._total_burden += self.burden[mod_idx]

        if self.debug:
            avail = _mask_str(self.patient_masks[patient_idx])
            acq = _mask_str(self._acquired_mask)
            print(
                f"[ENV DEBUG] reset patient={self.patient_names[patient_idx]} "
                f"available={avail} acquired={acq}"
            )

        state = self._get_state()
        self._initial_uncertainty = state["uncertainty"]
        return state

    def step(self, action: int) -> Tuple[dict, float, bool, dict]:
        """Execute *action*, return ``(state, reward, done, info)``.

        For valid acquisitions the process reward is:
            reward = -cost + α × max(0, unc_before − unc_after)
        where unc_before uses the *current* embedding (pre-acquisition) and
        unc_after comes from the post-acquisition state (computed once and
        reused as the returned observation — no redundant forward passes).
        """
        assert not self._done, "Episode already done — call reset()"
        assert 0 <= action < N_ACTIONS, f"Invalid action {action}"

        info: dict = {"action": action, "action_name": ACTION_NAMES[action]}
        reward = 0.0
        # Will be set for valid acquisitions; otherwise computed at end.
        _cached_state = None

        if action == PREDICT:
            reward = self._compute_terminal_reward()
            self._done = True
            info["terminal"] = True
        else:
            mod_idx = action - 1
            available = self.patient_masks[self._patient_idx]

            if self._acquired_mask[mod_idx] > 0.5:
                reward, info["invalid"], info["reason"] = -0.1, True, "already_acquired"
            elif available[mod_idx] < 0.5:
                reward, info["invalid"], info["reason"] = -0.1, True, "modality_unavailable"
            elif not self._check_deps(mod_idx):
                reward, info["invalid"], info["reason"] = -0.1, True, "dependency_not_met"
            elif self._total_burden + self.burden[mod_idx] > self.budget + 1e-6:
                reward, info["invalid"], info["reason"] = -0.1, True, "over_budget"
            else:
                # Valid acquisition — compute process reward efficiently.
                # unc_before: use current embedding (cheap — only uncertainty head)
                unc_before = self._compute_uncertainty(
                    self._acquired_mask, embedding=self._last_embedding)

                if self.decision_signal == "cavs":
                    # CAVS-aligned process reward:
                    #   R_proc = -cost + α × cavs_value
                    # where cavs_value is the CAVS head output at the
                    # pre-acquisition state. High CAVS → acquiring is
                    # well-justified → positive reward net of cost.
                    cavs_value = self._compute_cavs_value(
                        self._acquired_mask, embedding=self._last_embedding)
                elif self.decision_signal == "risk_delta":
                    # RDVS-aligned process reward:
                    #   R_proc = -cost + α × rdvs_value
                    rdvs_value = self._compute_rdvs_value(
                        self._acquired_mask, embedding=self._last_embedding)

                self._acquired_mask[mod_idx] = 1.0
                cost = self.burden[mod_idx]
                self._total_burden += cost

                # Single _get_state() for the post-acquisition observation.
                # Reused as the returned state (no redundant forward pass).
                _cached_state = self._get_state()
                unc_after = _cached_state["uncertainty"]

                if self.decision_signal == "cavs":
                    reward = -cost + self.process_reward_alpha * cavs_value
                    info["cavs_value"] = cavs_value
                elif self.decision_signal == "risk_delta":
                    reward = -cost + self.process_reward_alpha * rdvs_value
                    info["rdvs_value"] = rdvs_value
                else:
                    unc_reduction = max(0.0, unc_before - unc_after)
                    reward = -cost + self.process_reward_alpha * unc_reduction
                    info["unc_reduction"] = unc_reduction

                info["burden"] = cost
                info["total_burden"] = self._total_burden
                info["unc_before"] = unc_before
                info["unc_after"] = unc_after

        self._step_count += 1
        self._trajectory.append(
            {"step": self._step_count, "action": action, "reward": reward, "info": info}
        )

        # Safety: force termination at max_steps
        if not self._done and self._step_count >= self.max_steps:
            self._done = True
            info["forced_predict"] = True

        # Reuse cached state from valid acquisition, otherwise compute once.
        state = _cached_state if _cached_state is not None else self._get_state()

        if self.debug:
            acq = _mask_str(self._acquired_mask)
            print(
                f"[ENV DEBUG] step {self._step_count}: {ACTION_NAMES[action]:16s} "
                f"reward={reward:+.4f}  acquired={acq}  done={self._done}"
            )

        return state, reward, self._done, info

    # ------------------------------------------------------------------
    # Action masking
    # ------------------------------------------------------------------
    def get_valid_actions(self) -> List[int]:
        """Return the list of valid actions in the current state."""
        if self._done:
            return []

        valid = [PREDICT]  # always allowed
        available = self.patient_masks[self._patient_idx]

        for mod_idx in range(self.n_modalities):
            if self._acquired_mask[mod_idx] > 0.5:
                continue
            if available[mod_idx] < 0.5:
                continue
            if self._total_burden + self.burden[mod_idx] > self.budget + 1e-6:
                continue
            if not self._check_deps(mod_idx):
                continue
            valid.append(mod_idx + 1)

        return valid

    def _check_deps(self, mod_idx: int) -> bool:
        """Check clinical dependency constraints for *mod_idx*.

        Always enforces strict clinical ordering: Demo → Rad → Path → Gen.
        """
        for dep in self.CLINICAL_DEPS.get(mod_idx, []):
            if self._acquired_mask[dep] < 0.5:
                return False
        return True

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------
    def _get_state(self) -> dict:
        """Build observation dict for the agent."""
        p = self._patient_idx
        features = self.patient_features[p].copy()  # (n_mod, feat_dim)
        for i in range(self.n_modalities):
            if self._acquired_mask[i] < 0.5:
                features[i] = 0.0

        with torch.no_grad():
            feat_t = torch.from_numpy(features).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(self._acquired_mask).unsqueeze(0).to(self.device)

            embedding = self.predictor.get_embedding(feat_t, mask_t).cpu().numpy()[0]
            self._last_embedding = embedding  # cache for process reward
            risk_score = float(
                self.predictor.get_risk_score(feat_t, mask_t).cpu().item()
            )

        uncertainty = self._compute_uncertainty(
            self._acquired_mask, embedding=embedding
        )

        return {
            "features": features,
            "mask": self._acquired_mask.copy(),
            "available_mask": self.patient_masks[p].copy(),
            "embedding": embedding,
            "uncertainty": uncertainty,
            "risk_score": risk_score,
            "budget_remaining": self.budget - self._total_burden,
            "total_burden": self._total_burden,
            "step": self._step_count,
            "patient_idx": p,
            "valid_actions": self.get_valid_actions(),
        }

    def _compute_uncertainty(
        self, acquired_mask: np.ndarray,
        embedding: Optional[np.ndarray] = None,
    ) -> float:
        """Compute uncertainty using calibrated head (required).

        Raises RuntimeError if calibrated head is not attached or embedding
        is not provided — mask-based fallback was removed because it gives
        identical values for all patients with the same modality pattern.
        """
        if self._calibrated_head is None or embedding is None:
            raise RuntimeError(
                "ClinicalEnv._compute_uncertainty requires a calibrated head "
                "and embedding. Mask-based fallback (1 - n_acquired/n_mod) "
                "was removed — it is not patient-discriminative."
            )
        with torch.no_grad():
            emb_t = torch.from_numpy(embedding).float().unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(acquired_mask).float().unsqueeze(0).to(self.device)
            return float(self._calibrated_head(emb_t, mask_t).cpu().item())

    def _compute_cavs_value(
        self, acquired_mask: np.ndarray,
        embedding: Optional[np.ndarray] = None,
    ) -> float:
        """Compute CAVS value (concordance improvement probability).

        Raises RuntimeError if CAVS head is not attached.
        """
        if self._cavs_head is None or embedding is None:
            raise RuntimeError(
                "ClinicalEnv._compute_cavs_value requires a CAVS head "
                "and embedding. Set decision_signal='uncertainty' or "
                "attach a CAVS head via set_cavs_head()."
            )
        with torch.no_grad():
            emb_t = torch.from_numpy(embedding).float().unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(acquired_mask).float().unsqueeze(0).to(self.device)
            return float(self._cavs_head(emb_t, mask_t).cpu().item())

    def _compute_rdvs_value(
        self, acquired_mask: np.ndarray,
        embedding: Optional[np.ndarray] = None,
    ) -> float:
        """Compute RDVS value (risk-error reduction probability).

        Raises RuntimeError if RDVS head is not attached.
        """
        if self._rdvs_head is None or embedding is None:
            raise RuntimeError(
                "ClinicalEnv._compute_rdvs_value requires an RDVS head "
                "and embedding. Set decision_signal='uncertainty' or "
                "attach an RDVS head via set_rdvs_head()."
            )
        with torch.no_grad():
            emb_t = torch.from_numpy(embedding).float().unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(acquired_mask).float().unsqueeze(0).to(self.device)
            return float(self._rdvs_head(emb_t, mask_t).cpu().item())

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_terminal_reward(self) -> float:
        """Terminal reward when the agent chooses PREDICT.

        Reward with cost normalization (LA-CDM-style):
            reward = λ · (quality − w · total_burden)

        where quality = 1 − uncertainty,  w = cost_weight.

        With cost_weight=0.6 the net benefit is:
            stop@rad  (unc≈0.38): λ·(0.62 − 0.6·0.17) = λ·0.518  ← optimal
            stop@path (unc≈0.15): λ·(0.85 − 0.6·0.70) = λ·0.430
            all mods  (unc≈0.01): λ·(0.99 − 0.6·1.00) = λ·0.390

        This makes "stop@rad" the optimal strategy for average patients
        while high-uncertainty patients still benefit from more modalities.
        Inspired by LA-CDM (ICLR 2026) where all-test cost = diagnosis
        reward, making "acquire everything" a zero-reward strategy.

        UCPR term (beta > 0) is optional and
        defaults to 0.0.
        """
        p = self._patient_idx

        features = self.patient_features[p].copy()
        for i in range(self.n_modalities):
            if self._acquired_mask[i] < 0.5:
                features[i] = 0.0

        with torch.no_grad():
            feat_t = torch.from_numpy(features).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(self._acquired_mask).unsqueeze(0).to(self.device)
            embedding = self.predictor.get_embedding(
                feat_t, mask_t).cpu().numpy()[0]
            risk_score = float(
                self.predictor.get_risk_score(feat_t, mask_t).cpu().item())

        uncertainty = self._compute_uncertainty(
            self._acquired_mask, embedding=embedding)

        if self.decision_signal in ("cavs", "risk_delta"):
            # CAVS/RDVS-aligned terminal reward:
            #   quality = exp(-|risk - oracle_risk|)
            # Directly measures prediction quality (risk accuracy)
            # instead of 1 - uncertainty.
            oracle_risk = float(self.oracle_risks[p])
            quality = float(np.exp(-abs(risk_score - oracle_risk)))
        else:
            quality = 1.0 - uncertainty

        # LA-CDM-style burden penalty
        burden_penalty = self.cost_weight * self._total_burden

        # UCPR (optional, default off)
        unc_reduction = max(0.0, self._initial_uncertainty - uncertainty)
        ucpr = self.ucpr_beta * unc_reduction

        terminal = self.reward_coeff * (quality - burden_penalty + ucpr)

        if self.debug:
            print(
                f"[ENV DEBUG] terminal: "
                f"quality={quality:.4f} unc={uncertainty:.4f} "
                f"burden_penalty={burden_penalty:.4f} "
                f"total_burden={self._total_burden:.4f} "
                f"ucpr={ucpr:.4f} reward={terminal:.4f}"
                + (f" decision_signal={self.decision_signal}" if self.decision_signal != "uncertainty" else "")
            )

        return terminal

    # ------------------------------------------------------------------
    # Episode summary
    # ------------------------------------------------------------------
    def get_episode_summary(self) -> dict:
        """Structured summary of a completed (or in-progress) episode."""
        p = self._patient_idx
        return {
            "patient_idx": p,
            "patient_name": self.patient_names[p],
            "acquired_mask": self._acquired_mask.copy(),
            "n_acquired": int(self._acquired_mask.sum()),
            "total_burden": self._total_burden,
            "trajectory": list(self._trajectory),
            "n_steps": self._step_count,
            "event": float(self.patient_events[p]),
            "time": float(self.patient_times[p]),
            "oracle_risk": float(self.oracle_risks[p]),
            "done": self._done,
        }

    @property
    def total_burden_possible(self) -> float:
        """Sum of all modality burdens (= minimum budget to acquire all)."""
        return sum(self.burden.values())

    def __repr__(self) -> str:
        return (
            f"ClinicalEnv(n_patients={self.n_patients}, "
            f"budget={self.budget}, burden={self.burden})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mask_str(mask: np.ndarray) -> str:
    """Compact string representation of a modality mask, e.g. ``'D.P.'``."""
    labels = ["D", "R", "P", "G"]
    return "".join(l if mask[i] > 0.5 else "." for i, l in enumerate(labels))


def load_predictor(checkpoint_path: str, config: dict, device: str = "cuda"):
    """
    Load a frozen ``SurvivalPredictor`` from a Phase-0 checkpoint.

    Args:
        checkpoint_path: path to ``best_model.pt``
        config:          full config dict (needs ``model`` and ``data`` keys)
        device:          torch device

    Returns:
        predictor on *device*, in eval mode, with gradients disabled.
    """
    from models.survival_predictor import SurvivalPredictor

    mc = config["model"]
    dc = config["data"]

    predictor = SurvivalPredictor(
        input_dim=dc["feature_dim"],
        hidden_dim=mc["encoder"]["hidden_dim"],
        n_modalities=dc["n_modalities"],
        n_encoder_layers=mc["encoder"]["n_layers"],
        n_heads=mc["encoder"]["n_heads"],
        predictor_hidden_dims=mc["survival"]["hidden_dims"],
        n_intervals=config["training"]["survival"].get("n_intervals", 4),
        encoder_dropout=mc["encoder"]["dropout"],
        predictor_dropout=mc["survival"]["dropout"],
        use_positional=mc["encoder"].get("use_positional", True),
        use_reconstruction=mc["survival"].get("use_reconstruction", True),
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise ValueError(
            f"Checkpoint missing 'model_state_dict' key: {checkpoint_path}"
        )
    state_dict = ckpt["model_state_dict"]
    predictor.load_state_dict(state_dict, strict=True)
    predictor.to(device).eval()

    for p in predictor.parameters():
        p.requires_grad_(False)

    print(f"[ENV] Loaded predictor from {checkpoint_path}")
    return predictor


def load_calibrated_head_for_fold(
    outer_fold: int,
    inner_fold: int,
    checkpoint_dir: str,
    device: str = "cuda",
):
    """Load a pre-trained calibrated uncertainty head for a specific fold.

    Args:
        outer_fold:      outer fold number (1-5).
        inner_fold:      inner fold number (1-5).
        checkpoint_dir:  directory containing outer_N/inner_M/ subdirectories.
        device:          torch device.

    Returns:
        CalibratedUncertaintyHead on *device*, or None if not found.
    """
    if checkpoint_dir is None:
        raise ValueError(
            f"checkpoint_dir is required for calibrated head (outer {outer_fold})"
        )
    if inner_fold is None:
        raise ValueError(
            f"inner_fold is required for calibrated head (outer {outer_fold})"
        )

    path = os.path.join(
        checkpoint_dir, f"outer_{outer_fold}", f"inner_{inner_fold}",
        "calibrated_head.pt")

    if not os.path.exists(path):
        print(f"[ENV] Calibrated head not found: {path}")
        return None

    from models.calibrated_uncertainty import load_calibrated_head
    return load_calibrated_head(path, device=device)


def load_cavs_head_for_fold(
    outer_fold: int,
    inner_fold: int,
    checkpoint_dir: str,
    device: str = "cuda",
):
    """Load a pre-trained CAVS head for a specific fold.

    Args:
        outer_fold:      outer fold number (1-5).
        inner_fold:      inner fold number (1-5).
        checkpoint_dir:  directory containing outer_N/inner_M/ subdirectories.
        device:          torch device.

    Returns:
        CalibratedUncertaintyHead (CAVS) on *device*.

    Raises:
        RuntimeError if cavs_head.pt is not found.
    """
    if checkpoint_dir is None:
        raise ValueError(
            f"checkpoint_dir is required for CAVS head (outer {outer_fold})"
        )
    if inner_fold is None:
        raise ValueError(
            f"inner_fold is required for CAVS head (outer {outer_fold})"
        )

    path = os.path.join(
        checkpoint_dir, f"outer_{outer_fold}", f"inner_{inner_fold}",
        "cavs_head.pt")

    if not os.path.exists(path):
        raise RuntimeError(
            f"cavs_head.pt not found for outer_{outer_fold}/inner_{inner_fold}: "
            f"{path}. Run train_cavs.py first."
        )

    from models.calibrated_uncertainty import load_calibrated_head
    head = load_calibrated_head(path, device=device)
    print(f"[ENV] Loaded CAVS head from {path}")
    return head


def load_rdvs_head_for_fold(
    outer_fold: int,
    inner_fold: int,
    checkpoint_dir: str,
    device: str = "cuda",
):
    """Load a pre-trained RDVS head for a specific fold.

    Args:
        outer_fold:      outer fold number (1-5).
        inner_fold:      inner fold number (1-5).
        checkpoint_dir:  directory containing outer_N/inner_M/ subdirectories.
        device:          torch device.

    Returns:
        CalibratedUncertaintyHead (RDVS) on *device*.

    Raises:
        RuntimeError if rdvs_head.pt is not found.
    """
    if checkpoint_dir is None:
        raise ValueError(
            f"checkpoint_dir is required for RDVS head (outer {outer_fold})"
        )
    if inner_fold is None:
        raise ValueError(
            f"inner_fold is required for RDVS head (outer {outer_fold})"
        )

    path = os.path.join(
        checkpoint_dir, f"outer_{outer_fold}", f"inner_{inner_fold}",
        "rdvs_head.pt")

    if not os.path.exists(path):
        raise RuntimeError(
            f"rdvs_head.pt not found for outer_{outer_fold}/inner_{inner_fold}: "
            f"{path}. Run train_rdvs.py first."
        )

    from models.calibrated_uncertainty import load_calibrated_head
    head = load_calibrated_head(path, device=device)
    print(f"[ENV] Loaded RDVS head from {path}")
    return head
