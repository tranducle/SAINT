<div align="center">

# SAINT: Semantic Attention for Interpretable iNsider Threat Detection

[![Status](https://img.shields.io/badge/Status-Under_Review-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)]()

*Official PyTorch implementation of the SAINT architecture.*

</div>

## 📖 Overview

This repository contains the source code for **SAINT** (**S**emantic **A**ttention for **I**nterpretable **iN**sider **T**hreat Detection), a novel deep learning architecture designed to bridge the gap between high-performance anomaly detection and rigorous operational interpretability in cybersecurity.

While traditional deep learning models (e.g., LSTMs, Unconstrained Transformers) achieve high accuracy, they suffer from the "Black Box" problem, rendering their alerts unactionable for Security Operations Centers (SOCs). SAINT introduces **Semantic Multi-Head Attention (SMA)**, which enforces a strict 1-to-1 mapping between attention heads and specific user behavior modalities (e.g., *Login/File*, *Email*, *Web*, *Device*). 

This structural constraint allows the model to output a **Temporal Threat Indicator Score (TTIS)** that visually reconstructs the temporal narrative of an attack—without relying on unstable post-hoc approximations like SHAP or LIME.

> **Note:** This repository is currently anonymized for double-blind peer review. 

---

## 🏗️ Architecture

![SAINT Architecture](https://via.placeholder.com/800x400.png?text=SAINT+Architecture)
*(Figure from the paper demonstrating the Modality Slicing and Attention Computation)*

- **Modality Slicer:** Embeds discrete logs into dense vectors, segmented by modality.
- **Semantic Attention (SMA):** Forces $Head_k$ to strictly attend to modality $M_k$, guided by a sparse binary routing matrix.
- **Temporal Threat Indicator Score (TTIS):** Maps attention weights back to original timestamps to form a causal evidence set $E_i$ for SOC analysts.
- **Hybrid Regularization:** Uses a focal loss objective alongside sparsity ($\mathcal{L}_{sparse}$) and divergence ($\mathcal{L}_{div}$) constraints.

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- PyTorch 2.0+
- NumPy, Pandas, Scikit-learn

```bash
pip install -r requirements.txt
```

*(A detailed `requirements.txt` will be provided upon publication).*

### 2. File Structure

- `model.py`: Core PyTorch implementation of the SAINT architecture, including the `SemanticAttention` layer and the `TTIS` extraction logic.
- `train.py`: Training routines, complete with Focal Loss integration and Early Stopping mechanisms to prevent data leakage.
- `SAINT_Paper.pdf`: The anonymized manuscript currently under peer review.

### 3. Usage Definition

To instantiate the model:

```python
import torch
from model import SAINTModel

# Assuming 5 modalities: [Login, File, Device, Email, Web]
model = SAINTModel(
    input_dim=128, 
    seq_len=200, 
    num_modalities=5, 
    d_model=256, 
    n_heads=5, 
    num_layers=3
)

# Dummy input: (Batch Size, Sequence Length, Input Dim)
x = torch.randn(32, 200, 128)
predictions, attention_maps = model(x)

# attention_maps shape: (Layer, Batch, Head, Seq_Len, Seq_Len)
# Head 0 strictly corresponds to Modality 0 (e.g., Login)
```

---

## 🔬 Experimental Results

Evaluations on the **CERT r4.2+r5.2** synthetic dataset demonstrate that SAINT captures complex insider threats while maintaining a high F1-score. 

| Model | Architecture Type | Explainability | F1-Score |
|---|---|---|---|
| DeepTaskAPT | LSTM (Recurrent) | Opaque (Low) | ~92% |
| UBS-Transformer | Global Attention | Requires Post-Hoc XAI | >96% |
| **SAINT-Hybrid** | **Semantic Attention** | **Intrinsic (Direct Maps)** | **91.4%** |

SAINT explicitly trades the final quantile of optimization for architectural transparency, reducing the analyst's verification complexity from $O(T \times F)$ down to $O(k)$. 

---

## 📝 Citation

*(Citation information will be updated once the peer-review process is finalized.)*

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
