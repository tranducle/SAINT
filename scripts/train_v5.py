"""
SAINT Training on V5 Ultimate Dataset
=====================================
Trains SAINT model on the V5 dataset (Behavior + Context).

Config:
- Dataset: combined_v5.pkl
- Model: SAINT (Alpha=0.80)
- Validation: 80/20 Split (Benchmark)
"""

import os
import sys
import argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
import logging
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model import SAINT, SAINTLoss, create_model

class V5Dataset(Dataset):
    def __init__(self, sequences, labels, mean=None, std=None):
        if mean is not None:
            sequences = (sequences - mean) / (std + 1e-8)
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.sequences[i], self.labels[i]

def setup_logging(output_dir):
    log_format = '%(asctime)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(output_dir / "train.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, 
                        default=str(Path(__file__).resolve().parent.parent / "data" / "processed" / "combined_v5.pkl"))
    parser.add_argument('--output_dir', default=str(Path(__file__).resolve().parent.parent / "results"))
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    
    # Setup Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"v5_ultimate_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(out_dir)
    logger.info(f"Output Directory: {out_dir}")
    logger.info(f"Script: train_v5.py")
    
    # Load
    logger.info(f"Loading {args.data}...")
    try:
        with open(args.data, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {args.data}. Did you run combine_v5_datasets.py?")
        return

    sequences = data['sequences']
    labels = data['labels']
    feats = data['feature_names']
    logger.info(f"Features ({len(feats)}): {feats}")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Norm
    mean = X_train.mean(axis=(0,1))
    std = X_train.std(axis=(0,1))
    
    train_ds = V5Dataset(X_train, y_train, mean, std)
    val_ds = V5Dataset(X_val, y_val, mean, std)
    
    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=512)
    
    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    model = create_model(input_dim=len(feats), config={
        'd_model': 256, 'n_heads': 4, 'n_layers': 2, 'd_ff': 512, 'seq_len': 30, 'dropout': 0.3
    }).to(device)
    
    criterion = SAINTLoss(lambda_div=0.05, lambda_sparse=0.005, use_focal=True, focal_alpha=0.80)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scaler = GradScaler()
    
    # Train
    best_f1 = 0
    best_stats = {}
    
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        steps = 0
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                out = model(bx)['logits']
                loss = criterion(out, by, model.all_attention_weights)['total']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            steps += 1
            
        # Val
        model.eval()
        probs = []
        all_y = []
        with torch.no_grad():
            for bx, by in val_dl:
                bx = bx.to(device)
                with autocast('cuda'):
                    out = model(bx)['probs']
                probs.append(out.cpu().numpy())
                all_y.extend(by.numpy())
        probs = np.concatenate(probs)
        all_y = np.array(all_y)
        
        # Metrics
        f1, thresh = 0, 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            curr = f1_score(all_y, (probs >= t).astype(int), zero_division=0)
            if curr > f1: f1, thresh = curr, t
        
        auc = roc_auc_score(all_y, probs)
        logger.info(f"Epoch {epoch+1}: F1={f1:.4f} (Thresh={thresh:.2f}) AUC={auc:.4f} Loss={train_loss/steps:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_stats = {
                'epoch': epoch + 1,
                'f1': f1,
                'auc': auc,
                'threshold': thresh,
                'precision': precision_score(all_y, (probs >= thresh).astype(int), zero_division=0),
                'recall': recall_score(all_y, (probs >= thresh).astype(int), zero_division=0)
            }
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            np.savez(out_dir / "preds.npz", probs=probs, labels=all_y)
            logger.info(f"  --> New Best F1! Model saved.")

    logger.info("="*30)
    logger.info(f"Training Complete. Best F1: {best_f1:.4f} at Epoch {best_stats['epoch']}")
    logger.info(f"Stats: {best_stats}")

if __name__ == "__main__":
    main()

