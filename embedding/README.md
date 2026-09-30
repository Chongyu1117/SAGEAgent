# Pre-extracted embeddings

`glioma/` holds the cohort used in the paper, built from TCGA-LGG, TCGA-GBM and BraTS:

| File | Content |
|:--|:--|
| `cohort.npz` | 962 patients (170 with all four modalities): 32-d demographics, radiology, pathology and genomics features, availability mask, survival event and time, tumor grade |
| `splits.json` | the nested 5×5 cross-validation splits of the paper |

The features come from the pretrained extractors of [MMD (Cui et al., MICCAI 2022)](https://doi.org/10.1007/978-3-031-16443-9_60); pathology features are averaged over each patient's patches. [`scripts/prepare_glioma.py`](../scripts/prepare_glioma.py) shows how `cohort.npz` was built, and [`scripts/make_splits.py`](../scripts/make_splits.py) regenerates `splits.json`. The file format is described in [Getting Started](../docs/GETTING_STARTED.md#cohort-file).

The features and survival data are derived from TCGA and BraTS, and their use is subject to the terms of those datasets.
