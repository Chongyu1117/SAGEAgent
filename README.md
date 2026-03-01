# SAGEAgent

Official code for *"SAGEAgent: Clinical Agent for Cost-Aware Sequential Acquisition in Multimodal Survival Prediction"*.

## Overview

SAGEAgent is a **training-free** LLM-based clinical agent that sequentially decides which diagnostic modalities to acquire for multimodal survival prediction. Key components:

- **Tool-augmented reasoning**: calibrated uncertainty, survival prediction, similar patient retrieval
- **Dual memory**: episodic (FAISS case-based) + semantic (reflection-learned rules)
- **Self-reflection**: distills acquisition strategies from past experience — no LLM fine-tuning

## Structure

```
├── models/
│   ├── transformer.py               # Multimodal Transformer encoder
│   ├── survival_predictor.py        # Cox + NLL survival predictor
│   └── calibrated_uncertainty.py    # Post-hoc calibrated uncertainty head
├── agents/
│   ├── llm_agent.py                 # LLM decision agent (core)
│   ├── memory.py                    # Episodic + Semantic memory
│   ├── reflection.py                # Self-reflection module
│   └── tools.py                     # Tool wrappers
├── envs/
│   └── clinical_env.py              # Clinical modality acquisition environment
├── prompts/
│   ├── decision_prompt.txt          # Agent decision prompt
│   └── reflection_prompt.txt        # Reflection prompt
├── train_survival.py                # Train survival predictor (nested 5x5 CV)
├── train_calibrated_uncertainty.py  # Train uncertainty head
├── run_agent.py                     # Train SAGEAgent (experience accumulation)
└── eval_sageagent.py                # Evaluate SAGEAgent (majority vote)
```

## Usage

### 1. Train Survival Predictor
```bash
python train_survival.py --nested --gpu 0
```

### 2. Train Calibrated Uncertainty Head
```bash
python train_calibrated_uncertainty.py --nested --gpu 0
```

### 3. Train SAGEAgent
```bash
python run_agent.py --model_name Qwen/Qwen2.5-7B-Instruct --gpu 0
```

### 4. Evaluate
```bash
python eval_sageagent.py --gpu 0
```

Ablations:
```bash
python eval_sageagent.py --no_tools --no_episodic --no_semantic --gpu 0  # Base LLM
python eval_sageagent.py --no_episodic --no_semantic --gpu 0             # Tools only
python eval_sageagent.py --no_semantic --gpu 0                           # + Episodic
python eval_sageagent.py --no_episodic --gpu 0                           # + Semantic
```

## Evaluation Protocol

Nested 5x5 cross-validation on 170 complete patients (all 4 modalities available):
- **Outer fold**: 5-fold split on 170 patients (34 test each)
- **Inner fold**: 5-fold split on remaining patients for predictor/uncertainty training
- **C-index**: Per outer fold, risk scores are averaged across 5 inner predictors per patient, then one C-index is computed. We report mean ± std of 5 outer fold C-indices.
- **Acquisition**: Majority vote (≥3/5 inner folds agree) determines per-patient modality decisions.

## Requirements

```
pip install -r requirements.txt
```

PyTorch >= 2.0, Transformers >= 4.40, faiss-cpu >= 1.7.4
