# Getting Started

[← Back to README](../README.md) · [Code structure](CODE_STRUCTURE.md) · [Clinical burden](CLINICAL_BURDEN.md)

- [Installation](#installation)
- [Data](#data)
- [Pipeline](#pipeline)
- [Released rules](#released-rules)
- [Ablations](#ablations)
- [Configuration](#configuration)
- [Evaluation protocol](#evaluation-protocol)

## Installation

```bash
git clone https://github.com/Chongyu1117/SAGEAgent.git
cd SAGEAgent

conda create -n sageagent python=3.10 -y
conda activate sageagent
pip install -r requirements.txt
```

`python tests/test_sageagent.py` checks the installation in a few seconds on a CPU (synthetic data, scripted LLM, no downloads).

The survival predictor and the uncertainty head are small and train in minutes on a GPU (or on a CPU). The frozen LLM, [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) by default, needs one GPU with about 20 GB of memory in bf16. It is downloaded from the Hugging Face Hub on first use. Any chat model on the Hub can be used instead: set `agent.llm` in the config.

## Data

SAGEAgent works on pre-extracted feature vectors, one per modality and patient. Two files describe a dataset: the cohort file holds the features and survival outcomes, and the splits file holds the nested cross-validation folds.

The glioma cohort of the paper is included in [`embedding/glioma/`](../embedding/glioma), so no download is needed:

| File | Content |
|:--|:--|
| `cohort.npz` | 962 patients from TCGA-LGG, TCGA-GBM and BraTS (170 with all four modalities): 32-d demographics, radiology, pathology and genomics features, survival event and time |
| `splits.json` | the nested 5×5 cross-validation splits used in the paper |

The features come from the pretrained extractors of [MMD (Cui et al., MICCAI 2022)](https://doi.org/10.1007/978-3-031-16443-9_60). [`scripts/prepare_glioma.py`](../scripts/prepare_glioma.py) shows how `cohort.npz` was built from MMD's patch-level feature file, and `python scripts/make_splits.py --config configs/glioma.yaml` regenerates `splits.json`.

### Cohort file

A NumPy `.npz` archive with one row per patient:

| Array | Shape | Content |
|:--|:--|:--|
| `patient_id` | `(N,)` | patient identifiers (strings) |
| `modalities` | `(M,)` | modality names, as used in the config |
| `x_<name>` | `(N, d)` | features of modality `<name>`; zeros where the modality is missing |
| `mask` | `(N, M)` | 1 if the patient has the modality, else 0 |
| `event` | `(N,)` | 1 if the event (death) was observed, 0 if censored |
| `time` | `(N,)` | survival or follow-up time |
| `strata` | `(N,)` | *optional* integer label used to stratify the splits (e.g. tumor grade) |

Feature dimensions may differ between modalities. [`sageagent/data.py`](../sageagent/data.py) documents the format, and `Cohort.save` writes it.

### Splits

Outer folds split the patients with every modality; each outer test set is part of the evaluation set. Within an outer fold, the remaining complete patients and all patients with missing modalities are split into inner train/validation folds. The predictor trains on all of them (with modality dropout); the agent uses the complete ones. Splits are stratified by event status and `strata`.

### Your own dataset

1. Write a cohort file in the format above.
2. Copy [`configs/glioma.yaml`](../configs/glioma.yaml) and edit the `clinical` section: disease, the modalities in their clinical order, a short description and a burden for each ([deriving burdens](CLINICAL_BURDEN.md#recompute-or-adapt)). These texts are what the LLM sees.
3. Point `data.cohort` and `data.splits` to your files, run `python scripts/make_splits.py --config <your config>`, then run the pipeline below.

## Pipeline

Every script takes `--config`. Each writes to `experiment.output_dir` and reads what the previous step wrote there, so the steps chain without further arguments. Use `--outer` / `--inner` to run a subset of folds, for example one outer fold per GPU, and `--device` / `--llm-device` to place the models.

```bash
python train_predictor.py   --config configs/glioma.yaml   # 1. survival predictors
python train_uncertainty.py --config configs/glioma.yaml   # 2. calibrated uncertainty heads
python run_agent.py         --config configs/glioma.yaml   # 3. experience accumulation (LLM)
python evaluate.py          --config configs/glioma.yaml   # 4. majority-vote evaluation (LLM)
```

| Step | What happens | Output (under `outputs/glioma/`) |
|:--|:--|:--|
| 1 | Trains the multimodal Transformer predictor of each (outer, inner) pipeline. It uses Cox partial likelihood, reconstruction and alignment losses, 50% modality dropout, and selection on the validation C-index. | `predictors/outer_k/inner_m/predictor.pt` |
| 2 | Fits the uncertainty head on clinical-order prefixes, then applies temperature scaling. | `predictors/outer_k/inner_m/uncertainty_head.pt` |
| 3 | The frozen LLM processes every complete training patient 3 times. Every decision goes to episodic memory, and self-reflection every 10 patients updates semantic memory. | `agent/full/outer_k/inner_m/` |
| 4 | Each inner pipeline decides the acquisitions for the outer test patients. The decisions are combined by majority vote, and the script reports C-index, burden, HV and 95% bootstrap intervals. | `eval/full/summary.json`, `patients.csv`, `traces/` |

Steps 3 and 4 skip pipelines that are already finished, so an interrupted run can simply be restarted (`--overwrite` recomputes). `traces/` keeps the prompt and the full chain of thought of every test decision. If step 4 ran one outer fold per process, run `evaluate.py` once more without `--outer`: it reads the saved traces and writes the summary over all folds.

## Released rules

[`rules/glioma.json`](../rules/glioma.json) holds the decision rules that semantic memory learned in the paper's experiments, one rule set per pipeline (keyed `outer_k/inner_m`, 2–8 rules each). A rule is plain text that goes into the decision prompt, so any chat model set in `agent.llm` can use it.

With the released rules, step 3 can be skipped. Train the predictors and uncertainty heads (steps 1 and 2), then evaluate with the rules in place of a learned memory:

```bash
python evaluate.py --config configs/glioma.yaml --rules rules/glioma.json --no-episodic
# -> outputs/glioma/eval/tools+semantic_glioma/
```

Episodic memory is left out because it stores embeddings of the predictors it was built with. Rules learned in your own runs can be collected into the same format:

```bash
python scripts/export_rules.py outputs/glioma/agent/full --out rules/my_rules.json
```

A rule file is JSON with either one rule set per pipeline, `{"pipelines": {"outer_1/inner_1": [...], ...}}`, or one set for every pipeline, `{"rules": [...]}`. Each rule needs a `stage` (a decision point such as `after_radiology`), a `direction` (`stop` or `acquire`), a `pattern` (the text of the rule) and an `action_guidance` (`predict now` or `acquire <next modality>`). `confidence` and the outcome counts are optional.

## Ablations

Components can be switched off with `--no-tools`, `--no-episodic` and `--no-semantic`. In `evaluate.py` this removes them at test time from the agent trained in step 3 (the memory is read from `--memory`, default `full`):

| Configuration | Command |
|:--|:--|
| Base LLM | `python evaluate.py --no-tools --no-episodic --no-semantic` |
| + Tools | `python evaluate.py --no-episodic --no-semantic` |
| + Tools + Episodic | `python evaluate.py --no-semantic` |
| + Tools + Semantic | `python evaluate.py --no-episodic` |
| SAGEAgent | `python evaluate.py` |

Each run is saved under `eval/<name>/`, where the name lists the active components (e.g. `tools+episodic`). The same flags in `run_agent.py` train an agent without those components (saved as `agent/<name>/`).

The naive policy that stops as soon as the calibrated uncertainty falls below a fixed τ needs no LLM:

```bash
python evaluate.py --uncertainty-threshold 0.3      # saved as eval/threshold_0.3/
```

Two evaluation runs can be compared with a paired bootstrap:

```bash
python scripts/compare_runs.py outputs/glioma/eval/full outputs/glioma/eval/base_llm
```

## Configuration

All settings live in one YAML file, and any value can be overridden on the command line:

```bash
python run_agent.py --config configs/glioma.yaml --set agent.temperature=0.3 experience.reflect_every=20
```

| Setting | Key | Default |
|:--|:--|:--|
| Data | `data.cohort`, `data.splits` | `embedding/glioma/cohort.npz`, `embedding/glioma/splits.json` |
| LLM | `agent.llm` | `Qwen/Qwen2.5-7B-Instruct` |
| Clinical order, burdens, prompt texts | `clinical.modalities` | demographics 0.03 → radiology 0.14 → pathology 0.53 → genomics 0.30 |
| Uncertainty label threshold τ | `uncertainty.tau` | 0.3 |
| Reward weights α, λ | `reward.alpha`, `reward.lambda` | 1.0, 1.0 |
| Episodes per training patient | `experience.episodes_per_patient` | 3 |
| Reflection frequency (patients) | `experience.reflect_every` | 10 |
| Retrieved cases / past decisions | `agent.episodic.k` | 3 |
| Active rules | `semantic.max_active_rules` | 10 |
| Good / poor episodes | `semantic.good_quantile`, `semantic.poor_quantile` | top / bottom quartile of total reward |

[`configs/glioma.yaml`](../configs/glioma.yaml) documents every option.

## Evaluation protocol

| | |
|:--|:--|
| Cohort | 962 glioma patients; the 170 with all four modalities form the evaluation set |
| Cross-validation | nested 5×5: 5 outer folds (136 / 34 complete patients), 5 inner folds each, giving 25 pipelines |
| Aggregation | per patient, a modality is acquired if a majority (≥ 3 of 5) of the inner pipelines acquire it |
| C-index | risk under the voted modalities, averaged over the 5 inner predictors; one C-index per outer fold, mean ± std over folds |
| Burden | demographics 0.03, radiology 0.14, pathology 0.53, genomics 0.30 (full workup = 1.00), derived by multi-criteria decision analysis of cost, turnaround time, invasiveness and infrastructure ([details](CLINICAL_BURDEN.md)) |
| Trade-off | hypervolume HV = (C-index − 0.5) × (1 − burden) |
| Intervals | 95% percentile bootstrap, 1,000 resamples drawn within each outer fold |
