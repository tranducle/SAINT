"""
Combine CERT V5 Ultimate Datasets
=================================
Merges cert_r42_v5.pkl and cert_r52_v5.pkl into combined_v5.pkl.
"""

import pickle
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

def main():
    print("Combining V5 Ultimate r4.2 + r5.2...")
    
    # Load r4.2
    p42 = DATA_DIR / "cert_r42_v5.pkl"
    if not p42.exists():
        print(f"Error: {p42} not found. Run preprocessing first.")
        return
    with open(p42, 'rb') as f:
        r42 = pickle.load(f)
        
    # Load r5.2
    p52 = DATA_DIR / "cert_r52_v5.pkl"
    if not p52.exists():
        print(f"Error: {p52} not found. Run preprocessing first.")
        return
    with open(p52, 'rb') as f:
        r52 = pickle.load(f)
        
    # Check Feature Alignment
    f42 = r42['feature_names']
    f52 = r52['feature_names']
    
    final_feats = f42
    
    if f42 != f52:
        print(f"Warning: Features mismatch!")
        print(f"r4.2: {len(f42)} features")
        print(f"r5.2: {len(f52)} features")
        
        # Intersect
        common = [f for f in f42 if f in f52]
        idx42 = [f42.index(f) for f in common]
        idx52 = [f52.index(f) for f in common]
        
        r42['sequences'] = r42['sequences'][:, :, idx42]
        r52['sequences'] = r52['sequences'][:, :, idx52]
        final_feats = common
        print(f"Aligned to {len(common)} common features.")
        
    # Concatenate
    X = np.concatenate([r42['sequences'], r52['sequences']], axis=0)
    y = np.concatenate([r42['labels'], r52['labels']], axis=0)
    
    print(f"Total Sequences: {len(X)}")
    print(f"Total Positive: {y.sum()} ({100*y.sum()/len(y):.2f}%)")
    print(f"Feature Names ({len(final_feats)}): {final_feats}")
    
    out_path = DATA_DIR / "combined_v5.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump({
            'sequences': X,
            'labels': y,
            'feature_names': final_feats,
            'version': 'v5_ultimate_combined'
        }, f)
        
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()

