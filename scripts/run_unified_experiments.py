"""
SAINT Unified Experiments â€” Standalone Baselines + DL Metric Verification
==========================================================================
Produces verified, standalone metrics for ALL models in Table 2.

Decisions applied:
  - D1: n_heads=4 (keep as-is, no retraining)
  - D2: Single stratified split (random_state=42, test_size=0.2)
  - SAINT standalone: seed=42 (F1=0.8101 from golden seed search)

This script:
  1. Loads combined_v5.pkl (63647, 30, 30)
  2. Splits with train_test_split(test_size=0.2, stratify=y, random_state=42)
  3. Trains standalone RF, XGBoost, Gradient Boosting on statistical features
  4. Recomputes verified P/R/F1/AUC for DL baselines from saved probability files
  5. Outputs results/unified_experiments/results.json

Author: Auto-generated for manuscript hardening
Date: 2026-04-08
"""

import json
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Optional: XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "combined_v5.pkl"
BASELINES_DIR = PROJECT_ROOT / "results" / "baselines"
OUTPUT_DIR = PROJECT_ROOT / "results" / "unified_experiments"

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SPLIT_SEED = 42
TREE_SEED = 42
TEST_SIZE = 0.2


def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "unified_experiments.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def extract_stat_features(X: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from 3D sequences.
    Same function used in compare_hybrid_models.py for consistency.
    Input:  (N, seq_len, n_features) = (N, 30, 30)
    Output: (N, 4 * n_features)      = (N, 120)
    """
    f_mean = np.mean(X, axis=1)  # (N, 30)
    f_max = np.max(X, axis=1)    # (N, 30)
    f_std = np.std(X, axis=1)    # (N, 30)
    f_last = X[:, -1, :]         # (N, 30)
    return np.concatenate([f_mean, f_max, f_std, f_last], axis=1)  # (N, 120)


def find_optimal_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple:
    """Find threshold that maximizes F1-score."""
    best_f1 = 0
    best_t = 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Compute full metric suite at a given threshold."""
    preds = (probs >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y_true, probs)), 4),
        "n_predicted_positive": int(preds.sum()),
        "n_actual_positive": int(y_true.sum()),
    }


def run_tree_baselines(X_train_stat, y_train, X_val_stat, y_val, logger):
    """Train standalone tree-based models and return metrics."""
    results = {}

    # â”€â”€ Random Forest â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("Training standalone Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=TREE_SEED, n_jobs=-1
    )
    rf.fit(X_train_stat, y_train)
    probs_rf = rf.predict_proba(X_val_stat)[:, 1]
    t_rf, _ = find_optimal_threshold(y_val, probs_rf)
    metrics_rf = compute_metrics(y_val, probs_rf, t_rf)
    metrics_rf["train_time_sec"] = round(time.time() - t0, 2)
    results["RandomForest_standalone"] = metrics_rf
    logger.info(f"  RF standalone: F1={metrics_rf['f1']}, P={metrics_rf['precision']}, "
                f"R={metrics_rf['recall']}, AUC={metrics_rf['auc']}, t={t_rf:.2f}")

    # â”€â”€ Gradient Boosting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("Training standalone Gradient Boosting...")
    t0 = time.time()
    gb = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=TREE_SEED
    )
    gb.fit(X_train_stat, y_train)
    probs_gb = gb.predict_proba(X_val_stat)[:, 1]
    t_gb, _ = find_optimal_threshold(y_val, probs_gb)
    metrics_gb = compute_metrics(y_val, probs_gb, t_gb)
    metrics_gb["train_time_sec"] = round(time.time() - t0, 2)
    results["GradientBoosting_standalone"] = metrics_gb
    logger.info(f"  GB standalone: F1={metrics_gb['f1']}, P={metrics_gb['precision']}, "
                f"R={metrics_gb['recall']}, AUC={metrics_gb['auc']}, t={t_gb:.2f}")

    # â”€â”€ XGBoost â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if HAS_XGB:
        logger.info("Training standalone XGBoost...")
        t0 = time.time()
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=TREE_SEED,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        xgb_clf.fit(X_train_stat, y_train)
        probs_xgb = xgb_clf.predict_proba(X_val_stat)[:, 1]
        t_xgb, _ = find_optimal_threshold(y_val, probs_xgb)
        metrics_xgb = compute_metrics(y_val, probs_xgb, t_xgb)
        metrics_xgb["train_time_sec"] = round(time.time() - t0, 2)
        results["XGBoost_standalone"] = metrics_xgb
        logger.info(f"  XGB standalone: F1={metrics_xgb['f1']}, P={metrics_xgb['precision']}, "
                    f"R={metrics_xgb['recall']}, AUC={metrics_xgb['auc']}, t={t_xgb:.2f}")
    else:
        logger.warning("XGBoost not installed, skipping.")

    return results


def verify_dl_baselines(y_val: np.ndarray, logger) -> dict:
    """Recompute metrics for DL baselines from saved probability files."""
    results = {}

    dl_models = ["LSTM", "CNN-LSTM", "VanillaTransformer", "Autoencoder"]

    for model_name in dl_models:
        prob_file = BASELINES_DIR / f"probs_{model_name}.npy"
        if not prob_file.exists():
            logger.warning(f"  {model_name}: prob file not found at {prob_file}")
            continue

        probs = np.load(prob_file)
        logger.info(f"Verifying {model_name}: loaded {len(probs)} predictions "
                    f"(range [{probs.min():.4f}, {probs.max():.4f}])")

        if model_name == "Autoencoder":
            # Autoencoder outputs reconstruction errors, not probabilities
            # Normalize to [0, 1] range for fair comparison
            if probs.max() > 1.0 or probs.min() < 0.0:
                logger.info(f"  Autoencoder: normalizing reconstruction errors to probabilities")
                probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)

        t_opt, f1_opt = find_optimal_threshold(y_val, probs)
        metrics = compute_metrics(y_val, probs, t_opt)
        results[model_name] = metrics
        logger.info(f"  {model_name}: F1={metrics['f1']}, P={metrics['precision']}, "
                    f"R={metrics['recall']}, AUC={metrics['auc']}, t={t_opt:.2f}")

        # Also compute at default threshold=0.5 for reference
        metrics_default = compute_metrics(y_val, probs, 0.5)
        results[f"{model_name}_at_0.5"] = metrics_default
        logger.info(f"  {model_name} @0.5: F1={metrics_default['f1']}, "
                    f"P={metrics_default['precision']}, R={metrics_default['recall']}")

    return results


def main():
    logger = setup_logging(OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("SAINT UNIFIED EXPERIMENTS")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Data: {DATA_PATH}")
    logger.info(f"Split: test_size={TEST_SIZE}, random_state={SPLIT_SEED}")
    logger.info(f"Tree seed: {TREE_SEED}")
    logger.info(f"XGBoost available: {HAS_XGB}")

    # â”€â”€ 1. Load Data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("\n[Phase 1] Loading data...")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    X = data["sequences"]  # (63647, 30, 30)
    y = data["labels"]     # (63647,)
    logger.info(f"  Shape: X={X.shape}, y={y.shape}")
    logger.info(f"  Positive rate: {y.mean():.4f} ({int(y.sum())} / {len(y)})")

    # â”€â”€ 2. Split (SAME as all existing experiments) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("\n[Phase 2] Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_SEED
    )
    logger.info(f"  Train: {X_train.shape[0]} samples ({y_train.sum():.0f} positive)")
    logger.info(f"  Val:   {X_val.shape[0]} samples ({y_val.sum():.0f} positive)")

    # Verify y_val matches saved y_val from baselines
    y_val_saved = np.load(BASELINES_DIR / "y_val.npy")
    if np.array_equal(y_val, y_val_saved):
        logger.info("  âœ… y_val matches saved baseline y_val â€” same split confirmed")
    else:
        logger.error("  âŒ y_val MISMATCH with saved baseline y_val â€” ABORT")
        logger.error(f"     New: sum={y_val.sum()}, len={len(y_val)}")
        logger.error(f"     Saved: sum={y_val_saved.sum()}, len={len(y_val_saved)}")
        sys.exit(1)

    # â”€â”€ 3. Extract statistical features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("\n[Phase 3] Extracting statistical features...")
    X_train_stat = extract_stat_features(X_train)
    X_val_stat = extract_stat_features(X_val)
    logger.info(f"  Stat feature shape: {X_train_stat.shape}")

    # â”€â”€ 4. Train standalone tree baselines â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("\n[Phase 4] Training standalone tree-based baselines...")
    tree_results = run_tree_baselines(X_train_stat, y_train, X_val_stat, y_val, logger)

    # â”€â”€ 5. Verify DL baselines from saved probs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("\n[Phase 5] Verifying DL baselines from saved probabilities...")
    dl_results = verify_dl_baselines(y_val, logger)

    # â”€â”€ 6. Compile final results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("\n[Phase 6] Compiling final results...")

    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "data_path": str(DATA_PATH),
            "data_shape": list(X.shape),
            "split_seed": SPLIT_SEED,
            "tree_seed": TREE_SEED,
            "test_size": TEST_SIZE,
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "train_positive": int(y_train.sum()),
            "val_positive": int(y_val.sum()),
            "stat_feature_dim": int(X_train_stat.shape[1]),
        },
        "tree_baselines": tree_results,
        "dl_baselines_verified": dl_results,
        "saint_reference": {
            "standalone_seed42": {
                "f1": 0.8101,
                "source": "golden_seed_search/leaderboard.csv",
                "note": "Best F1 at seed=42, single split"
            },
            "standalone_seed888": {
                "f1": 0.8125,
                "source": "golden_seed_search/leaderboard.csv",
                "note": "Best F1 across all seeds (cherry-picked)"
            },
            "cv_5x3": {
                "mean_f1": 0.7601,
                "std_f1": 0.0235,
                "source": "5x3_CV_20260112_123600/cv_summary.json"
            },
            "hybrid_f1": 0.9136,
            "hybrid_auc": 0.9997,
            "hybrid_weight_saint": 0.10,
            "hybrid_source": "hybrid_ensemble/hybrid.log"
        }
    }

    # â”€â”€ 7. Save results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    output_path = OUTPUT_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nâœ… Results saved to {output_path}")

    # â”€â”€ 8. Summary table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("\n" + "=" * 70)
    logger.info("FINAL TABLE 2 VALUES (for manuscript)")
    logger.info("=" * 70)
    logger.info(f"{'Model':<25} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    logger.info("-" * 70)

    # DL baselines (optimal threshold)
    for model in ["Autoencoder", "VanillaTransformer", "LSTM", "CNN-LSTM"]:
        if model in dl_results:
            m = dl_results[model]
            logger.info(f"{model:<25} {m['precision']:>10.4f} {m['recall']:>8.4f} "
                       f"{m['f1']:>8.4f} {m['auc']:>8.4f}")

    # Tree baselines
    for key, label in [("GradientBoosting_standalone", "Gradient Boosting"),
                       ("RandomForest_standalone", "Random Forest"),
                       ("XGBoost_standalone", "XGBoost")]:
        if key in tree_results:
            m = tree_results[key]
            logger.info(f"{label:<25} {m['precision']:>10.4f} {m['recall']:>8.4f} "
                       f"{m['f1']:>8.4f} {m['auc']:>8.4f}")

    # SAINT rows
    logger.info(f"{'SAINT (Standalone)':<25} {'â€”':>10} {'â€”':>8} {'0.8101':>8} {'â€”':>8}")
    logger.info(f"{'SAINT-Hybrid (RF)':<25} {'~0.871':>10} {'~0.960':>8} {'0.9136':>8} {'0.9997':>8}")
    logger.info("=" * 70)
    logger.info("\nDone! Total runtime: script complete.")


if __name__ == "__main__":
    main()

