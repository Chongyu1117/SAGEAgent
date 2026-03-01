"""
LLM Agent — LLM-based clinical decision agent for modality acquisition.

The agent uses chain-of-thought reasoning, tool calling (uncertainty,
predictor, similar-patient retrieval), and dual memory (episodic + semantic)
to decide which diagnostic modalities to acquire.

Usage:
    agent = LLMAgent(model_name="Qwen/Qwen2.5-7B-Instruct", ...)
    action = agent.decide(state, env)
"""

import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from envs.clinical_env import (
    ACTION_NAMES,
    MODALITY_NAMES,
    N_ACTIONS,
    PREDICT,
    ClinicalEnv,
)
from agents.tools import (
    UncertaintyTool,
    SurvivalPredictorTool,
    SimilarPatientRetriever,
    AcquisitionValueTool,
    RiskDeltaValueTool,
    ConcordanceInfluenceTool,
    _unc_level,
    _value_level,
)
from agents.memory import EpisodicMemory, SemanticMemory


# ═══════════════════════════════════════════════════════════════════════════
# Prompt loading
# ═══════════════════════════════════════════════════════════════════════════
def _load_prompt_template(path: str) -> str:
    """Load a prompt template file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path) as f:
        return f.read()


# ═══════════════════════════════════════════════════════════════════════════
# Action name mapping
# ═══════════════════════════════════════════════════════════════════════════
ACTION_NAME_TO_ID = {name: idx for idx, name in ACTION_NAMES.items()}
# Also support shorthand
ACTION_NAME_TO_ID.update(
    {
        "PREDICT": PREDICT,
        "ACQUIRE_DEMO": 1,
        "ACQUIRE_RAD": 2,
        "ACQUIRE_PATH": 3,
        "ACQUIRE_GEN": 4,
        "ACQUIRE_DEMOGRAPHICS": 1,
        "ACQUIRE_RADIOLOGY": 2,
        "ACQUIRE_PATHOLOGY": 3,
        "ACQUIRE_GENOMICS": 4,
        "DEMOGRAPHICS": 1,
        "RADIOLOGY": 2,
        "PATHOLOGY": 3,
        "GENOMICS": 4,
        "DEMO": 1,
        "RAD": 2,
        "PATH": 3,
        "GEN": 4,
    }
)


class LLMAgent:
    """LLM-based clinical modality acquisition agent.

    Args:
        model_name:          HuggingFace model ID (e.g. Qwen/Qwen2.5-7B-Instruct).
        tools:               dict of tool instances {name: tool}.
        episodic_memory:     EpisodicMemory instance.
        semantic_memory:     SemanticMemory instance.
        prompt_template_path: path to the decision prompt template.
        max_new_tokens:      max generation length.
        temperature:         sampling temperature (low = more deterministic).
        device:              torch device for LLM.
        n_retrieval:         number of similar cases to retrieve.
        use_tools:           whether to call tools before deciding.
        use_episodic:        whether to use episodic memory.
        use_semantic:        whether to use semantic memory.
        debug:               verbose logging.
    """

    name = "llm_agent"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        tools: Optional[Dict[str, object]] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        prompt_template_path: str = "prompts/decision_prompt.txt",
        max_new_tokens: int = 512,
        min_new_tokens: int = 80,
        temperature: float = 0.6,
        device: str = "cuda",
        n_retrieval: int = 3,
        use_tools: bool = True,
        use_episodic: bool = True,
        use_semantic: bool = True,
        decision_signal: str = "uncertainty",
        model=None,
        tokenizer=None,
        debug: bool = False,
    ):
        self.model_name = model_name
        self.tools = tools or {}
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.temperature = temperature
        self.device = device
        self.n_retrieval = n_retrieval
        self.use_tools = use_tools
        self.use_episodic = use_episodic
        self.use_semantic = use_semantic
        self.decision_signal = decision_signal  # "uncertainty", "cavs", or "risk_delta"
        self.retriever = None  # SimilarPatientRetriever, set externally
        self.debug = debug

        # Uncertainty level thresholds — MUST be set per fold via set_unc_thresholds()
        self.unc_low_thresh = None
        self.unc_high_thresh = None
        self.unc_thresholds = None  # 5-level quintile dict {p20,p40,p60,p80}

        # CAVS thresholds — set per fold via set_cavs_thresholds()
        self.cavs_thresholds = None  # 5-level quintile dict {p20,p40,p60,p80}
        self._last_cavs_result = None  # cached per decide() call

        # RDVS thresholds — set per fold via set_rdvs_thresholds()
        self.rdvs_thresholds = None  # 5-level quintile dict {p20,p40,p60,p80}
        self._last_rdvs_result = None  # cached per decide() call

        # Concordance influence thresholds — set per fold via set_concordance_thresholds()
        self.concordance_thresholds = None  # 3-level tercile dict {p33,p67}

        # Load prompt template
        self.prompt_template = _load_prompt_template(prompt_template_path)

        # LLM model + tokenizer (pre-loaded or lazy loaded)
        self._model = model
        self._tokenizer = tokenizer

    def _load_model(self):
        """Lazy-load the LLM model and tokenizer."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[LLM] Loading {self.model_name} ...")
        t0 = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        self._model.eval()

        dt = time.time() - t0
        print(f"[LLM] Model loaded in {dt:.1f}s on {self.device}")

    def set_unc_thresholds(self, thresholds: dict):
        """Set 5-level quintile uncertainty thresholds (computed per fold).

        Args:
            thresholds: dict with p20/p40/p60/p80 keys (required).
        """
        if thresholds is None or "p20" not in thresholds:
            raise ValueError(
                "5-level quintile thresholds dict (p20/p40/p60/p80) is required."
            )
        self.unc_thresholds = thresholds
        self.unc_low_thresh = thresholds["p20"]
        self.unc_high_thresh = thresholds["p80"]
        # Propagate to uncertainty tool so tool output matches prompt
        unc_tool = self.tools.get("uncertainty")
        if unc_tool is not None:
            unc_tool.unc_thresholds = thresholds
            unc_tool.unc_low_thresh = self.unc_low_thresh
            unc_tool.unc_high_thresh = self.unc_high_thresh

    def set_cavs_thresholds(self, thresholds: dict):
        """Set 5-level quintile CAVS thresholds (computed per fold).

        Args:
            thresholds: dict with p20/p40/p60/p80 keys (required).
        """
        if thresholds is None or "p20" not in thresholds:
            raise ValueError(
                "5-level CAVS quintile thresholds dict (p20/p40/p60/p80) is required."
            )
        self.cavs_thresholds = thresholds
        # Propagate to acquisition value tool
        cavs_tool = self.tools.get("acquisition_value")
        if cavs_tool is not None:
            cavs_tool.value_thresholds = thresholds

    def set_rdvs_thresholds(self, thresholds: dict):
        """Set 5-level quintile RDVS thresholds (computed per fold).

        Args:
            thresholds: dict with p20/p40/p60/p80 keys (required).
        """
        if thresholds is None or "p20" not in thresholds:
            raise ValueError(
                "5-level RDVS quintile thresholds dict (p20/p40/p60/p80) is required."
            )
        self.rdvs_thresholds = thresholds
        # Propagate to risk delta value tool
        rdvs_tool = self.tools.get("risk_delta_value")
        if rdvs_tool is not None:
            rdvs_tool.value_thresholds = thresholds

    def set_concordance_thresholds(self, thresholds: dict):
        """Set 3-level tercile concordance influence thresholds (computed per fold).

        Args:
            thresholds: dict with p33/p67 keys (required).
        """
        if thresholds is None or "p33" not in thresholds:
            raise ValueError(
                "3-level concordance influence tercile thresholds dict "
                "(p33/p67) is required."
            )
        self.concordance_thresholds = thresholds
        # Propagate to concordance influence tool
        conc_tool = self.tools.get("concordance_influence")
        if conc_tool is not None:
            conc_tool.influence_thresholds = thresholds

    def decide(self, state: dict, env: ClinicalEnv) -> Tuple[int, str]:
        """Decide which action to take given current state.

        Args:
            state: environment state dict.
            env:   ClinicalEnv instance.

        Returns:
            (action_id, reasoning_text).
            Use decide_with_context() if you need tool/memory context.
        """
        action, reasoning, _ = self.decide_with_context(state, env)
        return action, reasoning

    def decide_with_context(self, state: dict, env: ClinicalEnv) -> Tuple[int, str, dict]:
        """Decide which action to take, returning full context for CoT traces.

        Returns:
            (action_id, reasoning_text, context_dict)
            where context_dict contains tool_results, episodic_text,
            semantic_text, and total_burden at decision time.
        """
        self._load_model()

        # 1. Call tools
        tool_results = self._call_tools(state) if self.use_tools else "Tools disabled."

        # 2. Retrieve from episodic memory (stage-matched: same acquisition depth)
        #    Also retrieve similar patients (FAISS) — conceptually case-based reasoning
        episodic_text = "Episodic memory disabled."
        if self.use_episodic:
            episodic_parts = []

            # Similar patient retrieval (FAISS-based case retrieval)
            if self.retriever is not None:
                similar_patients = self.retriever(
                    state["features"], state["mask"], k=self.n_retrieval
                )
                episodic_parts.append(self.retriever.format_for_prompt(similar_patients))

            # Episodic memory (past episode outcomes)
            if self.episodic_memory and len(self.episodic_memory) > 0:
                query_depth = int(sum(1 for m in state["mask"] if m > 0.5))
                similar = self.episodic_memory.retrieve(
                    state["embedding"], k=self.n_retrieval,
                    query_depth=query_depth,
                )
                episodic_parts.append(self.episodic_memory.format_for_prompt(similar))

            episodic_text = "\n\n".join(episodic_parts) if episodic_parts else "No past cases available yet."

        # 3. Retrieve from semantic memory
        semantic_text = "Semantic memory disabled."
        if self.use_semantic and self.semantic_memory:
            rules = self.semantic_memory.retrieve_for_state(state)
            semantic_text = self.semantic_memory.format_for_prompt(rules)

        # 4. Build prompt
        prompt = self._build_prompt(state, tool_results, episodic_text, semantic_text, env=env)

        if self.debug:
            print(f"[LLM DEBUG] Prompt ({len(prompt)} chars):\n{prompt}\n{'─'*60}")

        # 5. Generate response
        reasoning = self._generate(prompt)

        if self.debug:
            print(f"[LLM DEBUG] Response ({len(reasoning)} chars):\n{reasoning}\n{'─'*60}")

        # 6. Parse action
        action = self._parse_action(reasoning, state["valid_actions"])

        context = {
            "tool_results": tool_results,
            "episodic_context": episodic_text,
            "semantic_context": semantic_text,
            "total_burden": float(state["total_burden"]),
        }

        return action, reasoning, context

    def _call_tools(self, state: dict) -> str:
        """Call available tools and format results."""
        results = []

        # CAVS mode: use AcquisitionValueTool as primary signal
        if self.decision_signal == "cavs":
            cavs_tool = self.tools.get("acquisition_value")
            if cavs_tool is None:
                raise RuntimeError(
                    "decision_signal='cavs' but AcquisitionValueTool not found. "
                    "Ensure cavs_head is provided to build_llm_agent()."
                )
            cavs_result = cavs_tool(state["features"], state["mask"])
            self._last_cavs_result = cavs_result  # cache for _build_prompt
            results.append(cavs_tool.format_for_prompt(cavs_result))
        elif self.decision_signal == "risk_delta":
            rdvs_tool = self.tools.get("risk_delta_value")
            if rdvs_tool is None:
                raise RuntimeError(
                    "decision_signal='risk_delta' but RiskDeltaValueTool not found. "
                    "Ensure rdvs_head is provided to build_llm_agent()."
                )
            rdvs_result = rdvs_tool(state["features"], state["mask"])
            self._last_rdvs_result = rdvs_result  # cache for _build_prompt
            results.append(rdvs_tool.format_for_prompt(rdvs_result))
        else:
            # Uncertainty mode: use UncertaintyTool
            unc_tool = self.tools.get("uncertainty")
            if unc_tool:
                unc_result = unc_tool(state["features"], state["mask"])
                results.append(unc_tool.format_for_prompt(unc_result))

        pred_tool = self.tools.get("predictor")
        if pred_tool:
            pred_result = pred_tool(state["features"], state["mask"])
            results.append(pred_tool.format_for_prompt(pred_result))

        # Concordance influence: supplementary signal (works with any decision_signal)
        conc_tool = self.tools.get("concordance_influence")
        if conc_tool:
            conc_result = conc_tool(state["features"], state["mask"])
            results.append(conc_tool.format_for_prompt(conc_result))

        return "\n\n".join(results) if results else "No tool results available."

    def _build_prompt(
        self,
        state: dict,
        tool_results: str,
        episodic_text: str,
        semantic_text: str,
        env: ClinicalEnv = None,
    ) -> str:
        """Fill in the prompt template with current state info."""
        mask = state["mask"]
        available = state["available_mask"]

        acquired = ", ".join(
            MODALITY_NAMES[i]
            for i in range(len(mask))
            if mask[i] > 0.5
        ) or "none"

        avail = ", ".join(
            MODALITY_NAMES[i]
            for i in range(len(available))
            if available[i] > 0.5 and mask[i] < 0.5
        ) or "none"

        valid_str = ", ".join(
            ACTION_NAMES[a] for a in state["valid_actions"]
        )
        action_names_str = ", ".join(
            ACTION_NAMES[a] for a in state["valid_actions"]
        )

        # NOTE: budget is NOT shown to the LLM — it equals total burden of
        # all modalities (1.0) and is never a binding constraint.  The agent
        # decides to stop based on uncertainty/cost-benefit reasoning, not a
        # hard budget cap.  Budget is kept in ClinicalEnv as a safety-net only.

        # Decision logic: unified across all modes.
        # Let the LLM reason freely from whatever information is available
        # (tool results, similar cases, learned rules) without prescribing
        # a specific reasoning framework.
        decision_logic = (
            "3. MAKE YOUR DECISION: Based on all the information above,\n"
            "   decide whether to acquire the next modality or stop and predict now.\n"
            "   Weigh the potential benefit of more information against\n"
            "   the clinical cost/burden of each additional test."
        )

        prompt = self.prompt_template.format(
            acquired_modalities=acquired,
            available_modalities=avail,
            total_burden=state["total_burden"],
            step=state["step"],
            tool_results=tool_results,
            episodic_memory=episodic_text,
            semantic_memory=semantic_text,
            valid_actions=valid_str,
            action_names=action_names_str,
            decision_logic=decision_logic,
        )
        return prompt

    def _generate(self, prompt: str) -> str:
        """Generate text from the LLM."""
        import torch

        # Format as chat
        messages = [
            {"role": "system", "content": "You are a clinical decision support AI."},
            {"role": "user", "content": prompt},
        ]

        input_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=8192
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=self.min_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def _parse_action(self, text: str, valid_actions: List[int]) -> int:
        """Parse action from LLM response.

        Strategy:
        1. Look for explicit ``ACTION: <name>`` anywhere in text.
        2. Check the last 3 non-empty lines for a standalone action name
           (avoids matching modality names mentioned in reasoning).
        3. Default to the next ACQUIRE action (not PREDICT) to avoid
           silent early-stopping on parse failures.
        """
        # Strategy 1: explicit ACTION: prefix (most reliable)
        match = re.search(
            r"ACTION:\s*(PREDICT|ACQUIRE_\w+|DEMO(?:GRAPHICS)?|"
            r"RAD(?:IOLOGY)?|PATH(?:OLOGY)?|GEN(?:OMICS)?)",
            text,
            re.IGNORECASE,
        )
        if match:
            action_name = match.group(1).upper()
            action_id = ACTION_NAME_TO_ID.get(action_name)
            if action_id is not None and action_id in valid_actions:
                return action_id

        # Strategy 2: check last 3 non-empty lines for a standalone action name
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        for line in reversed(lines[-3:]):
            for action_id in valid_actions:
                name = ACTION_NAMES[action_id]
                # Match the action name as a whole word (not embedded in reasoning)
                if re.search(r"\b" + re.escape(name) + r"\b", line, re.IGNORECASE):
                    return action_id

        # Final fallback: prefer next ACQUIRE action over PREDICT to avoid
        # silent early-stopping from parse failures
        acquire_actions = [a for a in valid_actions if a != PREDICT]
        fallback = acquire_actions[0] if acquire_actions else PREDICT
        if self.debug:
            print(f"[LLM DEBUG] Could not parse action, defaulting to "
                  f"{ACTION_NAMES[fallback]}")
        return fallback

    def reset(self):
        """Reset per-episode state (if any)."""
        pass

    def select_action(self, state: dict, env: ClinicalEnv) -> int:
        """Interface compatible with BasePolicy for evaluation."""
        action, _ = self.decide(state, env)
        return action

    def test_inference(self) -> str:
        """Quick test that the model loads and generates."""
        self._load_model()
        prompt = "What is the capital of France? Answer briefly."
        messages = [{"role": "user", "content": prompt}]
        input_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(input_text, return_tensors="pt").to(self.device)

        import torch

        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=32)
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════
def build_llm_agent(
    predictor,
    train_features: np.ndarray,
    train_masks: np.ndarray,
    train_events: np.ndarray,
    train_times: np.ndarray,
    train_names: Optional[list] = None,
    calibrated_head=None,
    cavs_head=None,
    rdvs_head=None,
    decision_signal: str = "uncertainty",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    prompt_template_path: str = "prompts/decision_prompt.txt",
    episodic_memory: Optional[EpisodicMemory] = None,
    semantic_memory: Optional[SemanticMemory] = None,
    device: str = "cuda",
    llm_device: Optional[str] = None,
    n_retrieval: int = 3,
    use_tools: bool = True,
    use_episodic: bool = True,
    use_semantic: bool = True,
    use_concordance_influence: bool = False,
    model=None,
    tokenizer=None,
    debug: bool = False,
) -> LLMAgent:
    """Build a fully-configured LLM agent.

    Args:
        predictor:           frozen SurvivalPredictor.
        train_features:      (N, n_mod, feat_dim) for building FAISS index.
        train_masks:         (N, n_mod).
        train_events:        (N,).
        train_times:         (N,).
        train_names:         patient names.
        calibrated_head:     CalibratedUncertaintyHead (optional).
        cavs_head:           CAVS head (required when decision_signal=="cavs").
        rdvs_head:           RDVS head (required when decision_signal=="risk_delta").
        decision_signal:     "uncertainty", "cavs", or "risk_delta".
        model_name:          HuggingFace model ID.
        prompt_template_path: path to prompt template.
        episodic_memory:     pre-built EpisodicMemory or None (creates new).
        semantic_memory:     pre-built SemanticMemory or None (creates clinical default).
        device:              device for predictor/tools.
        llm_device:          device for LLM (defaults to device).
        n_retrieval:         number of similar cases to retrieve.
        use_tools, use_episodic, use_semantic: feature flags.
        use_concordance_influence: add ConcordanceInfluenceTool (requires use_tools=True).
        model:               pre-loaded LLM model (avoids reloading per fold).
        tokenizer:           pre-loaded tokenizer.
        debug:               verbose.

    Returns:
        configured LLMAgent.
    """
    if decision_signal == "cavs" and cavs_head is None:
        raise RuntimeError(
            "decision_signal='cavs' requires cavs_head. "
            "Run train_cavs.py first."
        )
    if decision_signal == "risk_delta" and rdvs_head is None:
        raise RuntimeError(
            "decision_signal='risk_delta' requires rdvs_head. "
            "Run train_rdvs.py first."
        )
    if use_concordance_influence and not use_tools:
        raise RuntimeError(
            "--use_concordance_influence requires tools "
            "(--no_tools is incompatible)"
        )

    # Build tools
    tools = {}
    if use_tools:
        tools["uncertainty"] = UncertaintyTool(
            predictor, calibrated_head=calibrated_head, device=device)
        tools["predictor"] = SurvivalPredictorTool(predictor, device=device)
        # VoI tool uses episodic memory — wired after memory init below

        # CAVS tool (when decision_signal=="cavs")
        if decision_signal == "cavs" and cavs_head is not None:
            tools["acquisition_value"] = AcquisitionValueTool(
                predictor, cavs_head=cavs_head, device=device)

        # RDVS tool (when decision_signal=="risk_delta")
        if decision_signal == "risk_delta" and rdvs_head is not None:
            tools["risk_delta_value"] = RiskDeltaValueTool(
                predictor, rdvs_head=rdvs_head, device=device)

        # Concordance influence tool (supplementary, works with any decision_signal)
        if use_concordance_influence:
            tools["concordance_influence"] = ConcordanceInfluenceTool(
                predictor=predictor,
                train_features=train_features,
                train_masks=train_masks,
                train_events=train_events,
                train_times=train_times,
                device=device,
            )

    # Build memory
    if episodic_memory is None:
        episodic_memory = EpisodicMemory(debug=debug)
    if semantic_memory is None:
        semantic_memory = SemanticMemory(debug=debug)

    # Similar patient retriever — case-based reasoning, controlled by use_episodic
    retriever = None
    if use_episodic:
        retriever = SimilarPatientRetriever(
            predictor,
            train_features=train_features,
            train_masks=train_masks,
            train_events=train_events,
            train_times=train_times,
            train_names=train_names,
            k=n_retrieval,
            device=device,
            debug=debug,
        )

    agent = LLMAgent(
        model_name=model_name,
        tools=tools,
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
        prompt_template_path=prompt_template_path,
        device=llm_device or device,
        n_retrieval=n_retrieval,
        use_tools=use_tools,
        use_episodic=use_episodic,
        use_semantic=use_semantic,
        decision_signal=decision_signal,
        model=model,
        tokenizer=tokenizer,
        debug=debug,
    )
    agent.retriever = retriever
    return agent
