# Clinical Burden

[← Back to README](../README.md) · [Getting started](GETTING_STARTED.md) · [Code structure](CODE_STRUCTURE.md)

Each modality carries a clinical burden b(m) between 0 and 1. It enters the agent's rewards and the reported burden and hypervolume. For the glioma pathway:

| Modality | Burden | Cumulative |
|:--|:-:|:-:|
| Demographics | 0.03 | 0.03 |
| Radiology (MRI) | 0.14 | 0.17 |
| Pathology (biopsy) | 0.53 | 0.70 |
| Genomics (sequencing) | 0.30 | 1.00 |

The burdens come from a multi-criteria decision analysis (MCDA) with a linear additive model [1–3]: every modality is scored on four criteria, the scores are weighted and summed, and the sums are normalized so that a full workup costs 1.0.

- [1. Criteria and weights](#1-criteria-and-weights)
- [2. Scores](#2-scores)
- [3. Weighted composite](#3-weighted-composite)
- [4. Normalization](#4-normalization)
- [Sensitivity to the weights](#sensitivity-to-the-weights)
- [Recompute or adapt](#recompute-or-adapt)
- [References](#references)

## 1. Criteria and weights

| Criterion | Weight | What it captures |
|:--|:-:|:--|
| Monetary cost | 0.25 | direct cost to the health system |
| Turnaround time | 0.25 | delay until the result is available, which postpones treatment decisions |
| Invasiveness and risk | 0.35 | risk and discomfort for the patient; weighted highest because pathology requires neurosurgery |
| Infrastructure | 0.15 | equipment and staff the test needs, which limits where it is available |

## 2. Scores

Each modality is scored on every criterion from 0 (no burden) to 10 (highest burden). The scores are ordinal judgments informed by the reference points below.

| Criterion | Demographics | Radiology (MRI) | Pathology (biopsy) | Genomics (NGS) |
|:--|:-:|:-:|:-:|:-:|
| Monetary cost | 0 | 2 | 10 | 3 |
| Turnaround time | 0 | 1 | 5 | 10 |
| Invasiveness and risk | 0 | 2 | 10 | 0 |
| Infrastructure | 0 | 5 | 7 | 9 |

- **Demographics** are collected at intake from the patient record, with no cost, delay or risk.
- **Radiology.** An MRI costs about USD 1,300 on average in the US [4] and is usually reported within a day. It is non-invasive apart from the contrast injection and the time in the scanner, but it needs an MRI scanner and a trained technologist.
- **Pathology.** A stereotactic biopsy is a neurosurgical procedure under anesthesia. The mean cost of the first 90 days after a biopsy for low-grade glioma was USD 43,219 in a national claims database [5]. Pooled over 7,471 stereotactic biopsies, morbidity was 3.5% and mortality 0.7% [6]. It needs an operating room, a neurosurgeon, anesthesia and a neuropathologist, and the report takes several days.
- **Genomics.** A next-generation sequencing (NGS) panel costs about USD 1,300–2,100 per test (USD 1,800 in a glioma cost model) and takes about 14 days [7]. It uses the tissue already taken at biopsy, so it adds no procedure, but it needs a molecular laboratory with sequencing and bioinformatics, which not every center has.

## 3. Weighted composite

The composite of a modality is the weighted sum of its scores, each divided by 10: composite(m) = Σₖ wₖ · sₖ(m) / 10.

| Modality | Cost | Turnaround | Invasiveness | Infrastructure | Composite |
|:--|:-:|:-:|:-:|:-:|:-:|
| Demographics | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Radiology | 0.050 | 0.025 | 0.070 | 0.075 | 0.220 |
| Pathology | 0.250 | 0.125 | 0.350 | 0.105 | 0.830 |
| Genomics | 0.075 | 0.250 | 0.000 | 0.135 | 0.460 |

## 4. Normalization

Demographics score 0 on every criterion, but collecting them still takes some clinician time, so their composite is raised to a floor of 0.04. The composites are then divided by their sum, 0.04 + 0.22 + 0.83 + 0.46 = 1.55:

| Modality | Composite | Normalized | Burden |
|:--|:-:|:-:|:-:|
| Demographics | 0.04 | 0.0258 | 0.03 |
| Radiology | 0.22 | 0.1419 | 0.14 |
| Pathology | 0.83 | 0.5355 | 0.53 |
| Genomics | 0.46 | 0.2968 | 0.30 |

The values are rounded to two decimals so that they still sum to 1.00 (largest-remainder rounding). Rounding each value on its own would give pathology 0.54 and a total of 1.01.

## Sensitivity to the weights

The same scores under other weightings:

| Weights (cost, turnaround, invasiveness, infrastructure) | Demographics | Radiology | Pathology | Genomics |
|:--|:-:|:-:|:-:|:-:|
| Used in the paper (0.25, 0.25, 0.35, 0.15) | 0.03 | 0.14 | 0.53 | 0.30 |
| Equal (0.25, 0.25, 0.25, 0.25) | 0.02 | 0.15 | 0.49 | 0.34 |
| Invasiveness-dominant (0.15, 0.20, 0.50, 0.15) | 0.03 | 0.15 | 0.57 | 0.25 |
| Cost-dominant (0.40, 0.20, 0.25, 0.15) | 0.03 | 0.14 | 0.54 | 0.29 |

Pathology carries the highest burden and genomics the second highest under every weighting, and the order of the four modalities never changes.

## Recompute or adapt

The whole calculation in a few lines of Python:

```python
weights = [0.25, 0.25, 0.35, 0.15]   # cost, turnaround, invasiveness, infrastructure
scores = {"demographics": [0, 0, 0, 0], "radiology": [2, 1, 2, 5],
          "pathology": [10, 5, 10, 7], "genomics": [3, 10, 0, 9]}
floor = 0.04

composite = {m: max(floor, sum(w * s / 10 for w, s in zip(weights, v))) for m, v in scores.items()}
total = sum(composite.values())
print({m: round(c / total, 4) for m, c in composite.items()})
# {'demographics': 0.0258, 'radiology': 0.1419, 'pathology': 0.5355, 'genomics': 0.2968}
```

For another disease, choose the criteria and weights that matter in that setting, score each modality, normalize so that the full workup sums to 1, and enter the results as `burden` in the `clinical.modalities` section of the config (see [Your own dataset](GETTING_STARTED.md#your-own-dataset)). Keeping the sum at 1 makes the reported burden the fraction of a full workup, which the hypervolume HV = (C-index − 0.5) × (1 − burden) assumes.

## References

1. Guitouni A, Martel JM. Tentative guidelines to help choosing an appropriate MCDA method. *European Journal of Operational Research*. 1998;109(2):501–521. [doi:10.1016/S0377-2217(98)00073-3](https://doi.org/10.1016/S0377-2217(98)00073-3)
2. Thokala P, et al. Multiple criteria decision analysis for health care decision making—an introduction: Report 1 of the ISPOR MCDA Emerging Good Practices Task Force. *Value in Health*. 2016;19(1):1–13. [doi:10.1016/j.jval.2015.12.003](https://doi.org/10.1016/j.jval.2015.12.003)
3. Marsh K, et al. Multiple criteria decision analysis for health care decision making—emerging good practices: Report 2 of the ISPOR MCDA Emerging Good Practices Task Force. *Value in Health*. 2016;19(2):125–137. [doi:10.1016/j.jval.2015.12.016](https://doi.org/10.1016/j.jval.2015.12.016)
4. GoodRx. What is the average cost of an MRI? [goodrx.com](https://www.goodrx.com/health-topic/diagnostics/how-much-does-an-mri-cost)
5. Tuohy K, et al. Early costs and complications of first-line low-grade glioma treatment using a large national database: limitations and future perspectives. *Frontiers in Surgery*. 2023;10:1001741. [doi:10.3389/fsurg.2023.1001741](https://doi.org/10.3389/fsurg.2023.1001741)
6. Hall WA. The safety and efficacy of stereotactic biopsy for intracranial lesions. *Cancer*. 1998;82(9):1749–1755. [PubMed 9576298](https://pubmed.ncbi.nlm.nih.gov/9576298/)
7. Jagasia S, et al. Cost matrix of molecular pathology in glioma—towards AI-driven rational molecular testing and precision care for the future. *Biomedicines*. 2022;10(12):3029. [doi:10.3390/biomedicines10123029](https://doi.org/10.3390/biomedicines10123029)
