"""
Tool Wrappers — thin interfaces around the frozen SurvivalPredictor.

Six tools are exposed to the LLM agent:
    1. UncertaintyTool       — current prediction uncertainty
    2. SurvivalPredictorTool — risk score + survival curve
    3. SimilarPatientRetriever — FAISS-based nearest-neighbour lookup
    4. AcquisitionValueTool  — CAVS: predicts p(acquiring next modality
                                improves concordance ranking)
    5. RiskDeltaValueTool    — RDVS: predicts expected risk-error reduction
                                from acquiring next modality (DIME-inspired)
    6. ConcordanceInfluenceTool — measures how sensitive the cohort's
                                concordance ranking is to this patient's
                                risk score (smooth C-index gradient)
"""

import numpy as np
import torch
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. Uncertainty Tool
# ---------------------------------------------------------------------------
class UncertaintyTool:
    """Query the calibrated uncertainty for the current observation.

    Uses the post-hoc calibrated uncertainty head (patient+mask specific).
    Raises RuntimeError if no calibrated head is provided.

    Args:
        predictor:       frozen SurvivalPredictor
        calibrated_head: CalibratedUncertaintyHead (required)
        device:          torch device string
    """

    name = "uncertainty_tool"
    description = (
        "Returns the calibrated predictive uncertainty given current "
        "observed modalities. Higher value = less confident."
    )

    def __init__(self, predictor, calibrated_head=None, device: str = "cuda"):
        self.predictor = predictor
        self.calibrated_head = calibrated_head
        self.device = device
        self.unc_low_thresh = None
        self.unc_high_thresh = None
        self.unc_thresholds = None  # 5-level quintile dict {p20,p40,p60,p80}

    def __call__(
        self, features: np.ndarray, mask: np.ndarray
    ) -> Dict[str, object]:
        """
        Args:
            features: (n_mod, feat_dim) — zeros for unobserved modalities.
            mask:     (n_mod,)          — 1=observed, 0=missing.

        Returns:
            dict with ``uncertainty`` (float) and ``level`` (str).
        """
        with torch.no_grad():
            feat_t = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)

            if self.calibrated_head is None:
                raise RuntimeError(
                    "UncertaintyTool requires a calibrated uncertainty head. "
                    "Mask-based fallback is disabled — it provides identical "
                    "values for all patients with the same modality pattern "
                    "and is not discriminative."
                )
            emb = self.predictor.get_embedding(feat_t, mask_t)
            unc = self.calibrated_head(emb, mask_t).cpu().item()

        return {
            "uncertainty": round(unc, 4),
            "level": _unc_level(unc, thresholds=self.unc_thresholds,
                               low_thresh=self.unc_low_thresh,
                               high_thresh=self.unc_high_thresh),
        }

    def format_for_prompt(self, result: dict) -> str:
        val = result['uncertainty']
        lvl = result['level'].upper()
        reason = self._threshold_reasoning(val)
        return (
            f"Uncertainty: {val:.4f} → {lvl} ({reason})"
        )

    def _threshold_reasoning(self, value: float) -> str:
        """Build explicit threshold comparison string."""
        vt = self.unc_thresholds
        if vt is None or "p20" not in vt:
            return "thresholds not set"
        if value < vt["p20"]:
            return f"{value:.4f} < {vt['p20']:.3f}"
        elif value < vt["p40"]:
            return f"{vt['p20']:.3f} ≤ {value:.4f} < {vt['p40']:.3f}"
        elif value < vt["p60"]:
            return f"{vt['p40']:.3f} ≤ {value:.4f} < {vt['p60']:.3f}"
        elif value < vt["p80"]:
            return f"{vt['p60']:.3f} ≤ {value:.4f} < {vt['p80']:.3f}"
        else:
            return f"{value:.4f} ≥ {vt['p80']:.3f}"


# ---------------------------------------------------------------------------
# 2. Survival Predictor Tool
# ---------------------------------------------------------------------------
class SurvivalPredictorTool:
    """Query risk score and discrete survival curve.

    Args:
        predictor: frozen SurvivalPredictor
        device:    torch device string
    """

    name = "survival_predictor_tool"
    description = (
        "Returns the predicted risk score (higher = worse prognosis) "
        "and a discrete survival curve S(t) at each time interval."
    )

    def __init__(self, predictor, device: str = "cuda"):
        self.predictor = predictor
        self.device = device

    def __call__(
        self, features: np.ndarray, mask: np.ndarray
    ) -> Dict[str, object]:
        with torch.no_grad():
            feat_t = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)

            risk = self.predictor.get_risk_score(feat_t, mask_t).cpu().item()
            surv = self.predictor.get_survival_curve(feat_t, mask_t).cpu().numpy()[0]

        return {
            "risk_score": round(risk, 4),
            "risk_group": "above average" if risk > 0 else "below average",
            "survival_curve": surv.tolist(),
            "median_survival_prob": round(float(np.median(surv)), 4),
        }

    def format_for_prompt(self, result: dict) -> str:
        surv_str = ", ".join(f"{s:.2f}" for s in result["survival_curve"])
        return (
            f"Risk score: {result['risk_score']:.4f} ({result['risk_group']} risk)\n"
            f"Survival curve S(t): [{surv_str}]\n"
            f"Median survival prob: {result['median_survival_prob']:.4f}"
        )


# ---------------------------------------------------------------------------
# 3. Similar Patient Retriever (FAISS)
# ---------------------------------------------------------------------------
class SimilarPatientRetriever:
    """Retrieve the K most similar training patients via FAISS.

    Uses cosine similarity on normalised predictor embeddings.

    Args:
        predictor:       frozen SurvivalPredictor
        train_features:  (N, n_mod, feat_dim) numpy array
        train_masks:     (N, n_mod) numpy array
        train_events:    (N,) numpy array
        train_times:     (N,) numpy array
        train_names:     list[str], optional
        k:               default number of neighbours
        device:          torch device string
        debug:           verbose
    """

    name = "similar_patient_retriever"
    description = (
        "Finds the K training patients most similar to the current patient "
        "(by embedding cosine similarity) and returns their outcomes."
    )

    def __init__(
        self,
        predictor,
        train_features: np.ndarray,
        train_masks: np.ndarray,
        train_events: np.ndarray,
        train_times: np.ndarray,
        train_names: Optional[List[str]] = None,
        k: int = 5,
        device: str = "cuda",
        debug: bool = False,
    ):
        self.predictor = predictor
        self.k = k
        self.device = device
        self.train_events = train_events.astype(np.float32)
        self.train_times = train_times.astype(np.float32)
        self.train_names = train_names

        # Compute training embeddings
        embeddings = []
        batch_size = 64
        with torch.no_grad():
            for start in range(0, len(train_features), batch_size):
                end = min(start + batch_size, len(train_features))
                feat = torch.from_numpy(
                    train_features[start:end].astype(np.float32)
                ).to(device)
                mask = torch.from_numpy(
                    train_masks[start:end].astype(np.float32)
                ).to(device)
                emb = self.predictor.get_embedding(feat, mask).cpu().numpy()
                embeddings.append(emb)

        self.embeddings = np.vstack(embeddings).astype(np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        self.embeddings_normed = (self.embeddings / norms).astype(np.float32)

        # Build FAISS index (inner product on L2-normalised = cosine)
        import faiss

        dim = self.embeddings_normed.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings_normed)

        if debug:
            print(
                f"[TOOLS DEBUG] FAISS index built: {self.index.ntotal} vectors, "
                f"dim={dim}"
            )

    def __call__(
        self,
        features: np.ndarray,
        mask: np.ndarray,
        k: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        """
        Args:
            features: (n_mod, feat_dim)
            mask:     (n_mod,)
            k:        override default K

        Returns:
            list of dicts, one per similar patient.
        """
        k = k or self.k

        with torch.no_grad():
            feat_t = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)
            query = self.predictor.get_embedding(feat_t, mask_t).cpu().numpy()

        query_norm = query / (np.linalg.norm(query) + 1e-8)
        scores, indices = self.index.search(query_norm.astype(np.float32), k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            name = self.train_names[idx] if self.train_names else f"patient_{idx}"
            event = float(self.train_events[idx])
            time = float(self.train_times[idx])
            results.append(
                {
                    "rank": rank + 1,
                    "similarity": round(float(score), 4),
                    "patient_name": name,
                    "event": event,
                    "time": time,
                    "outcome": "deceased" if event > 0.5 else "censored",
                }
            )

        return results

    def format_for_prompt(self, results: List[dict]) -> str:
        lines = ["Similar patients:"]
        for r in results:
            lines.append(
                f"  #{r['rank']}: {r['patient_name']} "
                f"(sim={r['similarity']:.3f}, {r['outcome']}, "
                f"time={r['time']:.0f})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Acquisition Value Tool (CAVS)
# ---------------------------------------------------------------------------
class AcquisitionValueTool:
    """Predicts p(acquiring next modality improves concordance ranking).

    Uses the CAVS (Concordance-Aware Value Scorer) head trained on
    concordance pair changes as supervision.

    Args:
        predictor:  frozen SurvivalPredictor
        cavs_head:  trained CalibratedUncertaintyHead (with concordance labels)
        device:     torch device string
    """

    name = "acquisition_value_tool"
    description = (
        "Returns the predicted probability that acquiring the next "
        "modality will improve the patient's concordance ranking."
    )

    def __init__(self, predictor, cavs_head, device: str = "cuda"):
        self.predictor = predictor
        self.cavs_head = cavs_head
        self.device = device
        self.value_thresholds = None  # 5-level quintile dict {p20,p40,p60,p80}

    def __call__(
        self, features: np.ndarray, mask: np.ndarray
    ) -> Dict[str, object]:
        """
        Args:
            features: (n_mod, feat_dim) — zeros for unobserved modalities.
            mask:     (n_mod,)          — 1=observed, 0=missing.

        Returns:
            dict with ``acquisition_value`` (float) and ``level`` (str).
        """
        with torch.no_grad():
            feat_t = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)

            emb = self.predictor.get_embedding(feat_t, mask_t)
            value = self.cavs_head(emb, mask_t).cpu().item()

        return {
            "acquisition_value": round(value, 4),
            "level": _value_level(value, thresholds=self.value_thresholds),
        }

    def format_for_prompt(self, result: dict) -> str:
        val = result['acquisition_value']
        lvl = result['level'].upper()
        reason = self._threshold_reasoning(val)
        return (
            f"Acquisition Value: {val:.4f} → {lvl} ({reason})\n"
            f"  Probability that acquiring the next modality improves "
            f"concordance ranking."
        )

    def _threshold_reasoning(self, value: float) -> str:
        """Build explicit threshold comparison string."""
        vt = self.value_thresholds
        if vt is None or "p20" not in vt:
            return "thresholds not set"
        if value < vt["p20"]:
            return f"{value:.4f} < {vt['p20']:.3f}"
        elif value < vt["p40"]:
            return f"{vt['p20']:.3f} ≤ {value:.4f} < {vt['p40']:.3f}"
        elif value < vt["p60"]:
            return f"{vt['p40']:.3f} ≤ {value:.4f} < {vt['p60']:.3f}"
        elif value < vt["p80"]:
            return f"{vt['p60']:.3f} ≤ {value:.4f} < {vt['p80']:.3f}"
        else:
            return f"{value:.4f} ≥ {vt['p80']:.3f}"


# ---------------------------------------------------------------------------
# 5. Risk-Delta Value Tool (RDVS)
# ---------------------------------------------------------------------------
class RiskDeltaValueTool:
    """Predicts the expected risk-error reduction from acquiring the next modality.

    Uses the RDVS (Risk-Delta Value Scorer) head trained with risk-delta labels
    and random mask augmentation (DIME-inspired).
    Different from AcquisitionValueTool (CAVS) which uses concordance-delta labels.

    Args:
        predictor:  frozen SurvivalPredictor
        rdvs_head:  trained CalibratedUncertaintyHead (with risk-delta labels)
        device:     torch device string
    """

    name = "risk_delta_value_tool"
    description = (
        "Returns the predicted probability that acquiring the next "
        "modality will reduce prediction risk error (improve accuracy)."
    )

    def __init__(self, predictor, rdvs_head, device: str = "cuda"):
        self.predictor = predictor
        self.rdvs_head = rdvs_head
        self.device = device
        self.value_thresholds = None  # 5-level quintile dict {p20,p40,p60,p80}

    def __call__(
        self, features: np.ndarray, mask: np.ndarray
    ) -> Dict[str, object]:
        """
        Args:
            features: (n_mod, feat_dim) — zeros for unobserved modalities.
            mask:     (n_mod,)          — 1=observed, 0=missing.

        Returns:
            dict with ``risk_delta_value`` (float) and ``level`` (str).
        """
        with torch.no_grad():
            feat_t = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)

            emb = self.predictor.get_embedding(feat_t, mask_t)
            value = self.rdvs_head(emb, mask_t).cpu().item()

        return {
            "risk_delta_value": round(value, 4),
            "level": _value_level(value, thresholds=self.value_thresholds),
        }

    def format_for_prompt(self, result: dict) -> str:
        val = result['risk_delta_value']
        lvl = result['level'].upper()
        reason = self._threshold_reasoning(val)
        return (
            f"Risk-Delta Value: {val:.4f} → {lvl} ({reason})\n"
            f"  Predicted probability that acquiring the next modality "
            f"reduces prediction risk error."
        )

    def _threshold_reasoning(self, value: float) -> str:
        """Build explicit threshold comparison string."""
        vt = self.value_thresholds
        if vt is None or "p20" not in vt:
            return "thresholds not set"
        if value < vt["p20"]:
            return f"{value:.4f} < {vt['p20']:.3f}"
        elif value < vt["p40"]:
            return f"{vt['p20']:.3f} ≤ {value:.4f} < {vt['p40']:.3f}"
        elif value < vt["p60"]:
            return f"{vt['p40']:.3f} ≤ {value:.4f} < {vt['p60']:.3f}"
        elif value < vt["p80"]:
            return f"{vt['p60']:.3f} ≤ {value:.4f} < {vt['p80']:.3f}"
        else:
            return f"{value:.4f} ≥ {vt['p80']:.3f}"


# ---------------------------------------------------------------------------
# 6. Concordance Influence Tool
# ---------------------------------------------------------------------------
class ConcordanceInfluenceTool:
    """Concordance Influence — measures how sensitive the cohort's
    concordance ranking is to this patient's risk score change.

    High sensitivity = patient participates in many tight-margin
    concordance pairs -> adding modality is more likely to change ranking.

    Uses smooth C-index gradient against TRAINING cohort (no leakage).
    """

    name = "concordance_influence_tool"
    description = (
        "Returns concordance sensitivity: how much the cohort's "
        "concordance ranking depends on this patient's risk accuracy."
    )

    STAGE_NAMES = {0: "D", 1: "DR", 2: "DRP"}

    def __init__(self, predictor, train_features, train_masks,
                 train_events, train_times, device="cuda"):
        """Pre-compute training risks at 3 depths for complete patients.

        Args:
            predictor:       frozen SurvivalPredictor
            train_features:  (N, n_mod, feat_dim)
            train_masks:     (N, n_mod)
            train_events:    (N,)
            train_times:     (N,)
            device:          torch device string
        """
        self.predictor = predictor
        self.device = device
        self.influence_thresholds = None  # set per fold via set thresholds

        # Identify complete training patients (all 4 modalities)
        complete_mask = train_masks.sum(axis=1) >= 3.5  # >=4
        complete_idx = np.where(complete_mask)[0]

        if len(complete_idx) == 0:
            raise RuntimeError(
                "ConcordanceInfluenceTool: no complete training patients found."
            )

        # Store events/times for complete patients
        self.train_events = torch.from_numpy(
            train_events[complete_idx].astype(np.float32)).to(device)
        self.train_times = torch.from_numpy(
            train_times[complete_idx].astype(np.float32)).to(device)
        self.n_train = len(complete_idx)

        # Pre-compute training risks at 3 prefix depths (D, DR, DRP)
        prefix_masks = [
            np.array([1, 0, 0, 0], dtype=np.float32),  # D
            np.array([1, 1, 0, 0], dtype=np.float32),  # DR
            np.array([1, 1, 1, 0], dtype=np.float32),  # DRP
        ]
        self.train_risks = {}  # {depth: tensor(n_complete,)}

        with torch.no_grad():
            for depth, pmask in enumerate(prefix_masks):
                risks = []
                for idx in complete_idx:
                    features = train_features[idx].copy()
                    for j in range(4):
                        if pmask[j] < 0.5:
                            features[j] = 0.0
                    feat_t = torch.from_numpy(features).float().unsqueeze(0).to(device)
                    mask_t = torch.from_numpy(pmask).float().unsqueeze(0).to(device)
                    risk = predictor.get_risk_score(feat_t, mask_t).cpu().item()
                    risks.append(risk)
                self.train_risks[depth] = torch.tensor(
                    risks, dtype=torch.float32, device=device)

    def __call__(self, features: np.ndarray, mask: np.ndarray) -> Dict[str, object]:
        """Compute concordance sensitivity for a patient at current depth.

        Uses sigmoid-weighted concordance sensitivity:
        - gamma=5.0 (sigmoid temperature)
        - Only event=1 training patients (concordance-relevant pairs only)
        - Raw sum of sigmoid derivatives (not normalized)

        Args:
            features: (n_mod, feat_dim)
            mask:     (n_mod,)

        Returns:
            dict with sensitivity (float), level (str), stage (str).
        """
        depth = int(mask.sum()) - 1  # 0=D, 1=DR, 2=DRP, 3=DRPG

        if depth >= 3 or depth < 0:
            return {"sensitivity": 0.0, "level": "low", "stage": "DRPG"}

        # Compute patient's risk at current depth
        with torch.no_grad():
            feat_t = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)
            risk_i = self.predictor.get_risk_score(feat_t, mask_t).item()

        # Smooth C-index gradient (validated: gamma=5.0, event=1 only, raw sum)
        gamma = 5.0
        train_risks = self.train_risks[depth]
        events = self.train_events

        # Only uncensored training patients — concordance pairs require event=1
        event_mask = events > 0.5
        if event_mask.sum() == 0:
            return {"sensitivity": 0.0, "level": "low",
                    "stage": self.STAGE_NAMES.get(depth, f"depth_{depth}")}

        ref_risks = train_risks[event_mask]
        diff = risk_i - ref_risks  # (n_event,)
        sig = torch.sigmoid(gamma * diff)
        sig_deriv = sig * (1.0 - sig)  # peaks when risk_i ≈ ref_risk_j
        sensitivity = float(sig_deriv.sum().item())

        stage = self.STAGE_NAMES.get(depth, f"depth_{depth}")

        # During threshold pre-computation, thresholds are not yet set.
        # Return raw sensitivity with level="unknown" — caller uses only
        # the sensitivity value.  Once thresholds are injected, level is
        # computed normally.
        if self.influence_thresholds is not None:
            level = _influence_level(sensitivity, self.influence_thresholds)
        else:
            level = "unknown"

        return {
            "sensitivity": round(sensitivity, 4),
            "level": level,
            "stage": stage,
        }

    def format_for_prompt(self, result: dict) -> str:
        val = result["sensitivity"]
        lvl = result["level"].upper()
        stage = result["stage"]
        reason = self._threshold_reasoning(val)
        return (
            f"Concordance Influence ({stage}): {val:.4f} → {lvl} ({reason})"
        )

    def _threshold_reasoning(self, value: float) -> str:
        """Build explicit threshold comparison string (3-level tercile)."""
        vt = self.influence_thresholds
        if vt is None or "p33" not in vt:
            return "thresholds not set"
        if value < vt["p33"]:
            return f"{value:.4f} < {vt['p33']:.4f}"
        elif value < vt["p67"]:
            return f"{vt['p33']:.4f} ≤ {value:.4f} < {vt['p67']:.4f}"
        else:
            return f"{value:.4f} ≥ {vt['p67']:.4f}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _value_level(value: float, thresholds: dict = None) -> str:
    """Categorise acquisition value using 5-level quintile thresholds.

    Same logic as _unc_level but for CAVS scores.
    Thresholds computed from CAVS head outputs on training data.
    """
    if thresholds is None or "p20" not in thresholds:
        raise ValueError(
            "5-level quintile thresholds dict (p20/p40/p60/p80) is required. "
            "Call set_cavs_thresholds(thresholds=...) first."
        )
    if value < thresholds["p20"]:
        return "very_low"
    elif value < thresholds["p40"]:
        return "low"
    elif value < thresholds["p60"]:
        return "moderate"
    elif value < thresholds["p80"]:
        return "high"
    return "very_high"


def _influence_level(value: float, thresholds: dict = None) -> str:
    """Categorise concordance influence using 3-level tercile thresholds.

    Uses p33/p67 tercile boundaries for robust binning:
      LOW (< p33), MODERATE (p33–p67), HIGH (≥ p67).

    Three levels match the signal's moderate discriminative power (AUROC 0.5–0.7)
    and avoid noisy fine-grained distinctions that 5 levels would require.
    """
    if thresholds is None or "p33" not in thresholds:
        raise ValueError(
            "3-level tercile thresholds dict (p33/p67) is required. "
            "Call set_concordance_thresholds(thresholds=...) first."
        )
    if value < thresholds["p33"]:
        return "low"
    elif value < thresholds["p67"]:
        return "moderate"
    return "high"


def _unc_level(unc: float, thresholds: dict = None,
               low_thresh: float = None, high_thresh: float = None) -> str:
    """Categorise uncertainty using 5-level quintile thresholds.

    Requires a dict with p20/p40/p60/p80 keys (5 levels).
    Thresholds computed from training distribution in run_agent.py.
    """
    if thresholds is None or "p20" not in thresholds:
        raise ValueError(
            "5-level quintile thresholds dict (p20/p40/p60/p80) is required. "
            "Call set_unc_thresholds(thresholds=...) first."
        )
    if unc < thresholds["p20"]:
        return "very_low"
    elif unc < thresholds["p40"]:
        return "low"
    elif unc < thresholds["p60"]:
        return "moderate"
    elif unc < thresholds["p80"]:
        return "high"
    return "very_high"
