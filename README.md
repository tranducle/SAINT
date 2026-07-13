<div align="center">

# SAINT: Semantic Attention for Interpretable iNsider Threat Detection

[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)]()

*Official PyTorch implementation of the SAINT architecture.*

</div>

## Overview

This repository contains the source code for **SAINT** (**S**emantic **A**ttention for **I**nterpretable **iN**sider **T**hreat Detection): a Transformer-based detector for insider threat detection that uses **Semantic Multi-Head Attention (SMA)** and a **Temporal Threat Indicator Score (TTIS)** for modality-structured explanations.

**Important:** This repository **does not redistribute** the CERT Insider Threat Dataset. Obtain the raw CERT r4.2 / r5.2 releases from the [CMU SEI CERT data page](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247) (or the official SEI distribution you are licensed for) and place them under the local layout below before running preprocessing.

## Repository layout

```
SAINT/
├── model.py                 # SAINT architecture + SAINTLoss + create_model
├── train.py                 # Generic training utilities
├── requirements.txt
├── fig1.png                 # Architecture figure
├── SAINT_Paper.pdf          # Paper PDF (if present)
├── data/
│   ├── raw/                 # YOU provide: CERT r4.2/, r5.2/, answers/
│   └── processed/           # Generated pkl tensors (gitignored)
└── scripts/
    ├── parse_labels.py
    ├── preprocess_cert_v5_ultimate.py
    ├── combine_v5_datasets.py
    ├── train_v5.py
    ├── search_golden_seeds.py
    └── run_unified_experiments.py
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Local data layout (not committed)

```
data/raw/
  r4.2/          # logon.csv, file.csv, email.csv, device.csv, http.csv, psychometric.csv, LDAP/...
  r5.2/
  answers/       # insiders.csv from the CERT answers package
data/processed/  # created by the scripts
```

## Reproduce the paper pipeline (high level)

1. **Parse labels** (from CERT answers):

```bash
python scripts/parse_labels.py --dataset r4.2
python scripts/parse_labels.py --dataset r5.2
```

2. **Preprocess** each release into session tensors:

```bash
python scripts/preprocess_cert_v5_ultimate.py --dataset r4.2
python scripts/preprocess_cert_v5_ultimate.py --dataset r5.2
```

3. **Combine** r4.2 + r5.2 → `data/processed/combined_v5.pkl`:

```bash
python scripts/combine_v5_datasets.py
```

4. **Train SAINT** (stratified 80/20 window-level split, seed 42; Focal α=0.80):

```bash
python scripts/train_v5.py --data data/processed/combined_v5.pkl --output_dir results
```

5. **Optional:** golden-seed search / unified baseline metrics:

```bash
python scripts/search_golden_seeds.py
python scripts/run_unified_experiments.py
```

## Model usage (minimal)

```python
import torch
from model import create_model

model = create_model(
    input_dim=30,
    config={"d_model": 256, "n_heads": 4, "n_layers": 2, "d_ff": 512, "seq_len": 30, "dropout": 0.3},
)
x = torch.randn(8, 30, 30)  # (batch, seq_len, n_features)
out = model(x)
```

## Experimental notes (as reported in the paper)

| Item | Value |
|---|---|
| Combined windows | 63,647 × 30 × 30 |
| Positive windows | 396 (0.62%) |
| Split | stratified 80/20, seed 42, window-level |
| Focal loss α | 0.80 |
| Attention heads | 4 |
| SAINT-Hybrid F1 | 91.4% (Table 3) |

Per-version detection tables and supervised cross-version transfer (train r4.2 → test r5.2 and reverse) are listed as future work in the revision.

## Citation

*(Update after acceptance.)*

## License

MIT License (see repository license file if present).
