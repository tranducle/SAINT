"""
SAINT training utility
======================
Trains SAINT from explicit train and validation index files. The caller controls
the split, which makes this utility suitable for user-disjoint evaluation.
"""

import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from model import SAINTLoss, create_model


DEFAULT_CONFIG = {
    "d_model": 256,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 512,
    "seq_len": 30,
    "dropout": 0.2,
    "learning_rate": 0.0005,
    "batch_size": 1024,
    "max_epochs": 20,
    "patience": 4,
    "focal_alpha": 0.90,
    "focal_gamma": 2.0,
    "lambda_div": 0.02,
    "lambda_sparse": 0.002,
    "seed": 42,
}


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, mean, std):
        normalized = (sequences - mean) / (std + 1e-8)
        self.sequences = torch.tensor(normalized.astype(np.float32))
        self.labels = torch.tensor(labels.astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_indices(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.int64)
    frame = pd.read_csv(path)
    if "record_id" in frame.columns:
        return frame["record_id"].to_numpy(dtype=np.int64)
    return frame.iloc[:, 0].to_numpy(dtype=np.int64)


def load_config(path):
    config = dict(DEFAULT_CONFIG)
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            config.update(json.load(handle))
    return config


def best_threshold(labels, probabilities):
    best_f1 = -1.0
    best_value = 0.5
    for value in np.arange(0.01, 1.00, 0.01):
        score = f1_score(labels, probabilities >= value, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_value = float(value)
    return best_value, best_f1


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    values = []
    for batch_x, _ in loader:
        output = model(batch_x.to(device, non_blocking=True))
        values.append(output["probs"].cpu().numpy())
    return np.concatenate(values)


def main():
    if len(sys.argv) not in {5, 6}:
        message = (
            "Usage: python train.py DATA TRAIN_INDICES VALIDATION_INDICES "
            "OUTPUT_DIR [CONFIG_JSON]"
        )
        raise SystemExit(message)

    data_path = Path(sys.argv[1])
    train_index_path = Path(sys.argv[2])
    validation_index_path = Path(sys.argv[3])
    output_dir = Path(sys.argv[4])
    config_path = Path(sys.argv[5]) if len(sys.argv) == 6 else None

    config = load_config(config_path)
    seed = int(config["seed"])
    set_seed(seed)

    with open(data_path, "rb") as handle:
        data = pickle.load(handle)

    sequences = np.asarray(data["sequences"])
    labels = np.asarray(data["labels"]).astype(np.int64)

    train_indices = load_indices(train_index_path)
    validation_indices = load_indices(validation_index_path)

    train_x = sequences[train_indices]
    train_y = labels[train_indices]
    validation_x = sequences[validation_indices]
    validation_y = labels[validation_indices]

    mean = train_x.mean(axis=(0, 1))
    std = train_x.std(axis=(0, 1))

    train_dataset = SequenceDataset(train_x, train_y, mean, std)
    validation_dataset = SequenceDataset(validation_x, validation_y, mean, std)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model_config = {
        "d_model": int(config["d_model"]),
        "n_heads": int(config["n_heads"]),
        "n_layers": int(config["n_layers"]),
        "d_ff": int(config["d_ff"]),
        "seq_len": int(config["seq_len"]),
        "dropout": float(config["dropout"]),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(input_dim=sequences.shape[-1], config=model_config).to(device)

    criterion = SAINTLoss(
        lambda_div=float(config["lambda_div"]),
        lambda_sparse=float(config["lambda_sparse"]),
        use_focal=True,
        focal_alpha=float(config["focal_alpha"]),
        focal_gamma=float(config["focal_gamma"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
    )

    best_state = None
    best_epoch = 0
    best_f1 = -1.0
    best_auc = -1.0
    best_threshold_value = 0.5
    bad_epochs = 0

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x)
            losses = criterion(
                output["logits"],
                batch_y,
                model.all_attention_weights,
            )
            losses["total"].backward()
            optimizer.step()

        probabilities = predict(model, validation_loader, device)
        threshold, score = best_threshold(validation_y, probabilities)
        auc = float(roc_auc_score(validation_y, probabilities))

        improved = score > best_f1
        if not improved and abs(score - best_f1) < 1e-12:
            improved = auc > best_auc

        if improved:
            best_state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            best_f1 = score
            best_auc = auc
            best_threshold_value = threshold
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"Epoch {epoch}: validation F1={score:.4f}, "
            f"AUC={auc:.6f}, threshold={threshold:.2f}"
        )

        if epoch >= 6 and bad_epochs >= int(config["patience"]):
            break

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": best_state,
        "input_dim": int(sequences.shape[-1]),
        "model_config": model_config,
        "training_config": config,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_f1": best_f1,
        "validation_auc": best_auc,
        "validation_threshold": best_threshold_value,
    }
    torch.save(checkpoint, output_dir / "model.pt")

    np.savez(
        output_dir / "normalization.npz",
        mean=mean,
        std=std,
    )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_epoch": best_epoch,
                "validation_f1": best_f1,
                "validation_auc": best_auc,
                "validation_threshold": best_threshold_value,
                "seed": seed,
            },
            handle,
            indent=2,
        )

    print(f"Saved model to {output_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
