"""
Combine CERT datasets for SAINT
===============================
Merges preprocessed CERT r4.2 and r5.2 sequence files into one dataset while
preserving user and window metadata for user-disjoint evaluation.
"""

import pickle
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def _load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing preprocessed dataset: {path}")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _align_features(left, right):
    left_names = left["feature_names"]
    right_names = right["feature_names"]

    if left_names == right_names:
        return left_names

    common = [name for name in left_names if name in right_names]
    left_idx = [left_names.index(name) for name in common]
    right_idx = [right_names.index(name) for name in common]

    left["sequences"] = left["sequences"][:, :, left_idx]
    right["sequences"] = right["sequences"][:, :, right_idx]
    return common


def _metadata(dataset, key, count, default_value):
    if key in dataset:
        return np.asarray(dataset[key])
    return np.asarray([default_value] * count)


def main():
    r42 = _load(DATA_DIR / "cert_r42.pkl")
    r52 = _load(DATA_DIR / "cert_r52.pkl")

    feature_names = _align_features(r42, r52)

    sequences = np.concatenate([r42["sequences"], r52["sequences"]], axis=0)
    labels = np.concatenate([r42["labels"], r52["labels"]], axis=0)

    n42 = len(r42["labels"])
    n52 = len(r52["labels"])

    user_ids = np.concatenate([
        _metadata(r42, "user_ids", n42, ""),
        _metadata(r52, "user_ids", n52, ""),
    ])
    window_starts = np.concatenate([
        _metadata(r42, "window_starts", n42, ""),
        _metadata(r52, "window_starts", n52, ""),
    ])
    window_ends = np.concatenate([
        _metadata(r42, "window_ends", n42, ""),
        _metadata(r52, "window_ends", n52, ""),
    ])
    source_release = np.concatenate([
        np.asarray(["r4.2"] * n42),
        np.asarray(["r5.2"] * n52),
    ])

    output_path = DATA_DIR / "combined_cert.pkl"
    with open(output_path, "wb") as handle:
        pickle.dump(
            {
                "sequences": sequences,
                "labels": labels,
                "feature_names": feature_names,
                "user_ids": user_ids,
                "window_starts": window_starts,
                "window_ends": window_ends,
                "source_release": source_release,
            },
            handle,
        )

    print(f"Sequences: {len(sequences)}")
    print(f"Positive windows: {int(labels.sum())}")
    print(f"Features: {len(feature_names)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
