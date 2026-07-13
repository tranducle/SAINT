"""
CERT Ultimate Preprocessor V5 for SAINT Model
=============================================
Combines:
1. V3's Rich Behavioral Features (Afterhours, Suspicious URLs, External Emails, etc.)
2. V4's Contextual Features (Psychometric Big-5, LDAP Role/Dept)

Goal: Maximize Information vs Noise ratio for F1 > 90%
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tqdm import tqdm
import pickle
from sklearn.preprocessing import LabelEncoder

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_ROOT = PROJECT_ROOT / "data" / "raw"          # place CERT r4.2 / r5.2 folders here
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = RAW_DATA_ROOT  # alias used by CERTPreprocessorV5 for raw version folders

# Configuration
WINDOW_SIZE = 30
STRIDE = 15

# Sampling (Same as V3)
SAMPLING_RATES = {
    'r4.2': {'http_normal': 0.2, 'email_normal': 0.5},
    'r5.2': {'http_normal': 0.1, 'email_normal': 0.3},
    'r6.2': {'http_normal': 0.05, 'email_normal': 0.2},
}

class CERTPreprocessorV5:
    def __init__(self, dataset_version: str):
        self.version = dataset_version
        self.data_dir = DATA_DIR / dataset_version
        self.labels_file = OUTPUT_DIR / f"{dataset_version}_insiders.json"
        
        self.malicious_users = set()
        self.malicious_periods = {}
        self.sampling = SAMPLING_RATES.get(dataset_version, SAMPLING_RATES['r4.2'])
        
        print(f"Initialized V5 ULTIMATE Preprocessor for {dataset_version}")
        
    def load_ground_truth(self):
        """Load labels"""
        print("Loading ground truth...")
        if not self.labels_file.exists():
            from parse_labels import parse_insiders_csv, save_labels
            version_float = float(self.version[1:])
            labels = parse_insiders_csv(version_float)
            save_labels(labels, self.version, OUTPUT_DIR)
            
        with open(self.labels_file, 'r') as f:
            data = json.load(f)
            
        for user, periods in data['insiders'].items():
            self.malicious_users.add(user)
            self.malicious_periods[user] = [
                (pd.to_datetime(p['start']), pd.to_datetime(p['end']))
                for p in periods
            ]
        return self.malicious_users

    # =========================================================================
    # V4 Contextual Loaders
    # =========================================================================
    
    def load_psychometric(self):
        print("Loading Psychometric data (V4 Feature)...")
        path = self.data_dir / "psychometric.csv"
        if not path.exists():
            print("  Warning: psychometric.csv not found. Skipping.")
            return None
            
        df = pd.read_csv(path)
        # Normalize
        for col in ['O', 'C', 'E', 'A', 'N']:
            if col in df.columns:
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        return df[['user_id', 'O', 'C', 'E', 'A', 'N']].rename(columns={'user_id': 'user'})

    def load_ldap(self):
        print("Loading LDAP data (V4 Feature)...")
        ldap_dir = self.data_dir / "LDAP"
        if not ldap_dir.exists():
            return None
            
        files = sorted(list(ldap_dir.glob("*.csv")))
        if not files: return None
            
        dfs = []
        for f in files:
            d = pd.read_csv(f)
            d['file_date'] = f.stem
            dfs.append(d)
            
        full = pd.concat(dfs)
        full = full.sort_values('file_date', ascending=False).drop_duplicates('user_id')
        
        print(f"  Encoding LDAP features for {len(full)} users...")
        le = LabelEncoder()
        full['role_enc'] = le.fit_transform(full['role'].astype(str))
        full['dept_enc'] = le.fit_transform(full['department'].astype(str))
        full['unit_enc'] = le.fit_transform(full['functional_unit'].astype(str))
        
        return full[['user_id', 'role_enc', 'dept_enc', 'unit_enc']].rename(columns={'user_id': 'user'})

    # =========================================================================
    # V3 Behavioral Loaders (Rich Features)
    # =========================================================================

    def load_logon(self):
        print("Loading logon (V3 Rich)...")
        df = pd.read_csv(self.data_dir / "logon.csv")
        df['date'] = pd.to_datetime(df['date'])
        df['day'] = df['date'].dt.date
        df['hour'] = df['date'].dt.hour
        
        # RICH FEATURES
        df['is_afterhours'] = ((df['hour'] < 6) | (df['hour'] > 20)).astype(int)
        df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
        
        features = df.groupby(['user', 'day']).agg({
            'id': 'count',
            'is_afterhours': 'sum',
            'is_weekend': 'max',
            'activity': lambda x: (x == 'Logon').sum(),
        }).rename(columns={
            'id': 'logon_count',
            'is_afterhours': 'afterhours_logon',
            'is_weekend': 'weekend_flag',
            'activity': 'logon_events'
        })
        return features.reset_index()

    def load_file(self):
        print("Loading file (V3 Rich)...")
        df = pd.read_csv(self.data_dir / "file.csv")
        df['date'] = pd.to_datetime(df['date'])
        df['day'] = df['date'].dt.date
        df['hour'] = df['date'].dt.hour
        
        # RICH FEATURES
        df['is_afterhours'] = ((df['hour'] < 6) | (df['hour'] > 20)).astype(int)
        df['is_exe'] = df['filename'].str.lower().str.endswith('.exe').astype(int)
        df['is_doc'] = df['filename'].str.lower().str.contains(r'\.(doc|pdf|xls|ppt|txt|csv)', regex=True, na=False).astype(int)
        df['is_zip'] = df['filename'].str.lower().str.contains(r'\.(zip|rar|7z|tar|gz)', regex=True, na=False).astype(int)
        
        features = df.groupby(['user', 'day']).agg({
            'id': 'count',
            'is_exe': 'sum',
            'is_doc': 'sum',
            'is_zip': 'sum',
            'is_afterhours': 'sum'
        }).rename(columns={
            'id': 'file_count',
            'is_exe': 'exe_count',
            'is_doc': 'doc_count',
            'is_zip': 'archive_count',
            'is_afterhours': 'afterhours_file'
        })
        return features.reset_index()

    def load_email(self):
        print("Loading email (V3 Rich)...")
        chunks = []
        sample_rate = self.sampling['email_normal']
        
        for chunk in pd.read_csv(self.data_dir / "email.csv", chunksize=500000):
            malicious_mask = chunk['user'].isin(self.malicious_users)
            sampled = chunk[~malicious_mask].sample(frac=sample_rate, random_state=42)
            chunk = pd.concat([chunk[malicious_mask], sampled])
            
            chunk['date'] = pd.to_datetime(chunk['date'])
            chunk['day'] = chunk['date'].dt.date
            chunk['hour'] = chunk['date'].dt.hour
            
            # RICH FEATURES
            chunk['is_external'] = (~chunk['to'].str.contains('@dtaa.com', na=True)).astype(int)
            chunk['has_attachment'] = (chunk['attachments'].notna() & (chunk['attachments'] != '')).astype(int)
            chunk['is_afterhours'] = ((chunk['hour'] < 6) | (chunk['hour'] > 20)).astype(int)
            chunk['recipient_count'] = chunk['to'].str.count(';') + 1
            
            agg = chunk.groupby(['user', 'day']).agg({
                'id': 'count',
                'is_external': 'sum',
                'has_attachment': 'sum',
                'size': 'sum',
                'is_afterhours': 'sum',
                'recipient_count': 'max'
            }).rename(columns={
                'id': 'email_count',
                'is_external': 'external_emails',
                'has_attachment': 'attachment_count',
                'size': 'total_email_size',
                'is_afterhours': 'afterhours_email',
                'recipient_count': 'max_recipients'
            })
            chunks.append(agg)
            
        return pd.concat(chunks).groupby(level=[0,1]).sum().reset_index()

    def load_device(self):
        print("Loading device (V3 Rich)...")
        sample_df = pd.read_csv(self.data_dir / "device.csv", nrows=5)
        use_cols = [c for c in ['id', 'date', 'user', 'pc', 'activity'] if c in sample_df.columns]
        
        df = pd.read_csv(self.data_dir / "device.csv", usecols=use_cols)
        df['date'] = pd.to_datetime(df['date'])
        df['day'] = df['date'].dt.date
        df['hour'] = df['date'].dt.hour
        
        # RICH FEATURES
        df['is_connect'] = (df['activity'] == 'Connect').astype(int)
        df['is_afterhours'] = ((df['hour'] < 6) | (df['hour'] > 20)).astype(int)
        
        features = df.groupby(['user', 'day']).agg({
            'id': 'count',
            'is_connect': 'sum',
            'is_afterhours': 'sum'
        }).rename(columns={
            'id': 'device_events',
            'is_connect': 'usb_connects',
            'is_afterhours': 'afterhours_device'
        })
        return features.reset_index()

    def load_http(self):
        print("Loading HTTP (V3 Rich)...")
        chunks = []
        sample_rate = self.sampling['http_normal']
        
        for chunk in pd.read_csv(self.data_dir / "http.csv", chunksize=500000):
            malicious_mask = chunk['user'].isin(self.malicious_users)
            sampled = chunk[~malicious_mask].sample(frac=sample_rate, random_state=42)
            chunk = pd.concat([chunk[malicious_mask], sampled])
            
            chunk['date'] = pd.to_datetime(chunk['date'])
            chunk['day'] = chunk['date'].dt.date
            chunk['hour'] = chunk['date'].dt.hour
            
            # RICH FEATURES
            chunk['is_afterhours'] = ((chunk['hour'] < 6) | (chunk['hour'] > 20)).astype(int)
            suspicious = ['job', 'career', 'linkedin', 'indeed', 'monster', 
                         'wikileaks', 'dropbox', 'drive.google', 'mega.nz']
            chunk['is_suspicious'] = chunk['url'].str.lower().str.contains('|'.join(suspicious), na=False).astype(int)
            cloud = ['dropbox', 'box.com', 'drive.google', 'onedrive', 'icloud']
            chunk['is_cloud'] = chunk['url'].str.lower().str.contains('|'.join(cloud), na=False).astype(int)
            
            agg = chunk.groupby(['user', 'day']).agg({
                'id': 'count',
                'is_suspicious': 'sum',
                'is_cloud': 'sum',
                'is_afterhours': 'sum'
            }).rename(columns={
                'id': 'http_count',
                'is_suspicious': 'suspicious_urls',
                'is_cloud': 'cloud_urls',
                'is_afterhours': 'afterhours_http'
            })
            chunks.append(agg)
            
        return pd.concat(chunks).groupby(level=[0,1]).sum().reset_index()

    def run(self):
        self.load_ground_truth()
        
        # Load Modalities
        logon = self.load_logon()
        file = self.load_file()
        email = self.load_email()
        device = self.load_device()
        http = self.load_http()
        
        # Merge Dynamic
        merged = logon.merge(file, on=['user', 'day'], how='outer')
        merged = merged.merge(email, on=['user', 'day'], how='outer')
        merged = merged.merge(device, on=['user', 'day'], how='outer')
        merged = merged.merge(http, on=['user', 'day'], how='outer')
        merged = merged.fillna(0)
        
        # Load Static
        psy = self.load_psychometric()
        ldap = self.load_ldap()
        
        # Merge Static (Broadcast)
        if psy is not None:
            merged = merged.merge(psy, on='user', how='left').fillna(0.5)
        if ldap is not None:
            merged = merged.merge(ldap, on='user', how='left').fillna(-1)
            
        merged['day'] = pd.to_datetime(merged['day'])
        
        # Create Sequences
        print("Creating sequences...")
        sequences = []
        labels = []
        
        feature_cols = [c for c in merged.columns if c not in ['user', 'day']]
        print(f"Final V5 Features ({len(feature_cols)}): {feature_cols}")
        
        for user in tqdm(merged['user'].unique()):
            user_data = merged[merged['user'] == user].sort_values('day')
            if len(user_data) < WINDOW_SIZE: continue
            
            for i in range(0, len(user_data) - WINDOW_SIZE + 1, STRIDE):
                window = user_data.iloc[i:i+WINDOW_SIZE]
                seq = window[feature_cols].values
                
                # Label
                w_start = window['day'].min()
                w_end = window['day'].max()
                
                label = 0
                if user in self.malicious_periods:
                    for s, e in self.malicious_periods[user]:
                        if w_start <= e and w_end >= s:
                            label = 1
                            break
                            
                sequences.append(seq)
                labels.append(label)
                
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        # Save
        out_path = OUTPUT_DIR / f"cert_{self.version.replace('.', '')}_v5.pkl"
        with open(out_path, 'wb') as f:
            pickle.dump({
                'sequences': sequences,
                'labels': labels,
                'feature_names': feature_cols,
                'version': 'v5_ultimate'
            }, f)
            
        print(f"Saved {len(sequences)} sequences to {out_path}")
        print(f"Positive: {labels.sum()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    CERTPreprocessorV5(args.dataset).run()

if __name__ == "__main__":
    main()


