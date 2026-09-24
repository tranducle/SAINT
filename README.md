# SAINT: Semantic Attention for Interpretable iNsider Threat Detection

SAINT is a Transformer based insider threat detection framework with a structurally grounded semantic attention layer and a Temporal Threat Indicator Score (TTIS).

The first attention layer uses a fixed semantic mask. Dynamic Login/File, Email, Device, and Web features can enter only their designated head channels, while psychometric and organizational context is shared across channels. TTIS is extracted from this grounded layer. Deeper Transformer layers remain free to integrate information for prediction.

## Architecture

The paper uses the following SAINT configuration:

- Input features: 30
- Sequence length: 30
- Model dimension: 256
- Attention heads: 4
- Transformer layers: 2
- Feed forward dimension: 512

The four semantic channels are Login/File, Email, Device, and Web. Psychometric and organizational variables are shared context.

## Paper evaluation

The reported evaluation uses CERT r4.2 and r5.2 with five user-disjoint outer folds and inner-only model and threshold selection.

- Eligible users: 2,991
- Combined windows: 63,647
- Positive windows: 396
- Outer folds: 5
- SAINT seeds per outer fold: 42, 123, 2026
- SAINT-Hybrid mean F1: 0.858 +/- 0.054
- SAINT-Hybrid pooled F1: 0.861
- SAINT-Hybrid mean AUC: 0.99825
- XGBoost mean F1: 0.875 +/- 0.061

Explanation analysis shows that neutralizing the six highest ranked TTIS time steps reduces malicious probability by 0.200 on average across 148 positive users, compared with 0.010 for random time steps. Seed-specific temporal rankings have mean Spearman correlation 0.691 with the ensemble consensus ranking.

These explanation results support temporal relevance and moderate stability. They are not presented as causal attribution of the full hybrid score.

## Repository structure

```
SAINT/
  model.py
  train.py
  requirements.txt
  README.md
  data/
    raw/
    processed/
  scripts/
    parse_labels.py
    preprocess_cert.py
    combine_datasets.py
```

## Dataset

The repository does not redistribute the CERT Insider Threat Dataset.

Place the CERT releases under:

```
data/raw/
  r4.2/
  r5.2/
  answers/
```

The preprocessing pipeline produces one sequence file per CERT release and preserves user identifiers and window boundaries for user-disjoint evaluation.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data preparation

Parse labels:

```bash
python scripts/parse_labels.py r4.2
python scripts/parse_labels.py r5.2
```

Preprocess each CERT release:

```bash
python scripts/preprocess_cert.py r4.2
python scripts/preprocess_cert.py r5.2
```

Combine the processed releases:

```bash
python scripts/combine_datasets.py
```

The combined dataset is written to:

```
data/processed/combined_cert.pkl
```

It contains:

- sequences
- labels
- feature_names
- user_ids
- window_starts
- window_ends
- source_release

## Training

`train.py` trains SAINT from explicit train and validation index files. The split is supplied by the caller, so training does not impose a window-level random split.

Usage:

```text
python train.py DATA TRAIN_INDICES VALIDATION_INDICES OUTPUT_DIR [CONFIG_JSON]
```

The trainer normalizes features using training data only, selects the decision threshold on validation data, and saves:

- model.pt
- normalization.npz
- metrics.json

For the paper protocol, train and validation indices are generated from user-disjoint partitions.

## Model usage

```python
import torch
from model import create_model

model = create_model(
    input_dim=30,
    config={
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 512,
        "seq_len": 30,
        "dropout": 0.3,
    },
)

x = torch.randn(8, 30, 30)
output = model(x, return_attention=True)

print(output["probs"].shape)
print(output["ttis"].shape)
```

## Reproducibility

The repository provides the model implementation, data preparation pipeline, and split-aware training utility used by the framework. The manuscript specifies the nested user-disjoint evaluation protocol, statistical branch, fusion procedure, and reported uncertainty.

The structural isolation claim is limited to the first semantic attention layer. Later layers may integrate information across channels.

## Citation

Citation information will be added after publication.

## License

MIT License.
