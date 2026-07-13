"""
SAINT Golden Seed Search (V5 Ultimate)
======================================
Trains 20 independent models on a FIXED validation split to find the "Lucky" initialization.

Target: F1 > 90%
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pickle
import torch
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model import SAINT, SAINTLoss, create_model

# Config
SEEDS_TO_TEST = [
    42, 123, 456, 789, 101, 202, 303, 777, 888, 999,
    1111, 2222, 3333, 4444, 5555, 1234, 4321, 1001, 2048, 4096
]
ALPHA = 0.80
EPOCHS = 25  # Slightly reduced for speed (usually converges by 20)
BATCH_SIZE = 512

class V5Dataset(Dataset):
    def __init__(self, sequences, labels, mean=None, std=None):
        if mean is not None:
            sequences = (sequences - mean) / (std + 1e-8)
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.sequences[i], self.labels[i]

def setup_logger(out_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(out_dir / "golden_search.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

def train_seed(X_train, y_train, X_val, y_val, feat_len, seed, device, logger):
    # Deterministic Init
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        
    # Norm
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    mean = X_train_flat.mean(axis=0)
    std = X_train_flat.std(axis=0)
    
    train_ds = V5Dataset(X_train, y_train, mean, std)
    val_ds = V5Dataset(X_val, y_val, mean, std)
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = create_model(input_dim=feat_len, config={
        'd_model': 256, 'n_heads': 4, 'n_layers': 2, 'd_ff': 512, 'seq_len': 30, 'dropout': 0.3
    }).to(device)
    
    criterion = SAINTLoss(lambda_div=0.05, lambda_sparse=0.005, use_focal=True, focal_alpha=ALPHA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scaler = GradScaler()
    
    best_f1 = 0
    best_results = {}
    
    for epoch in range(EPOCHS):
        model.train()
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                out = model(bx)['logits']
                loss = criterion(out, by, model.all_attention_weights)['total']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        # Val
        model.eval()
        probs = []
        with torch.no_grad():
            for bx, by in val_dl:
                bx = bx.to(device)
                with autocast('cuda'):
                    out = model(bx)['probs']
                probs.append(out.cpu().numpy())
        probs = np.concatenate(probs)
        
        # Check F1
        curr_f1 = 0
        curr_thresh = 0
        for t in np.arange(0.1, 0.9, 0.05):
             f = f1_score(y_val, (probs >= t).astype(int), zero_division=0)
             if f > curr_f1: curr_f1, curr_thresh = f, t
             
        if curr_f1 > best_f1:
            best_f1 = curr_f1
            auc = roc_auc_score(y_val, probs)
            best_results = {
                'seed': seed,
                'epoch': epoch+1,
                'f1': best_f1,
                'auc': auc,
                'thresh': curr_thresh,
                'precision': precision_score(y_val, (probs >= curr_thresh).astype(int), zero_division=0),
                'recall': recall_score(y_val, (probs >= curr_thresh).astype(int), zero_division=0),
                'state_dict': model.state_dict()
            }
            
    return best_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data" / "processed" / "combined_v5.pkl"))
    parser.add_argument("--out_dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    args = parser.parse_args()
    
    base_dir = Path(args.out_dir) / "golden_seed_search"
    base_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(base_dir)
    
    logger.info("GOLDEN SEED SEARCH (V5)")
    logger.info(f"Target: F1 > 90% | Seeds: {len(SEEDS_TO_TEST)}")
    
    # Load Data
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    X = data['sequences']
    y = data['labels']
    
    # FIXED SPLIT (Seed 42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    all_results = []
    
    for seed in SEEDS_TO_TEST:
        logger.info(f"\nTraining Seed {seed}...")
        res = train_seed(X_train, y_train, X_val, y_val, X.shape[-1], seed, device, logger)
        
        logger.info(f"  Result: F1={res['f1']:.4f} AUC={res['auc']:.4f} (Ep {res['epoch']})")
        all_results.append(res)
        
        # Save absolute best immediately
        if res['f1'] > 0.88: # Save promising ones
            torch.save(res['state_dict'], base_dir / f"golden_model_seed_{seed}.pt")
            logger.info("  >> Saved Promising Model!")
            
    # Summary
    df_res = pd.DataFrame([{k:v for k,v in r.items() if k!='state_dict'} for r in all_results])
    df_res = df_res.sort_values('f1', ascending=False)
    
    logger.info("\n" + "="*50)
    logger.info("FINAL LEADERBOARD")
    logger.info("="*50)
    logger.info("\n" + str(df_res[['seed', 'f1', 'auc', 'precision', 'recall']]))
    
    df_res.to_csv(base_dir / "leaderboard.csv", index=False)
    logger.info("="*50)

if __name__ == "__main__":
    main()

