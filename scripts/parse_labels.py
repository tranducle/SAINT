"""
Unified Label Parser for CERT Datasets (r4.2, r5.2, r6.2)
==========================================================
Extracts insider threat labels from different formats into unified JSON.

Usage:
    python scripts/parse_labels.py --dataset r5.2
    python scripts/parse_labels.py --dataset all
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANSWERS_DIR = PROJECT_ROOT / "data" / "raw" / "answers"   # place CERT answers/insiders.csv here
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_insiders_csv(dataset_version: float) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse insiders.csv for r4.2 and r5.2 datasets.
    
    Returns:
        Dict mapping user_id -> list of (start_datetime, end_datetime) tuples
    """
    insiders_file = ANSWERS_DIR / "insiders.csv"
    df = pd.read_csv(insiders_file)
    
    # Filter for specified dataset version
    dataset_df = df[df['dataset'] == dataset_version]
    
    malicious_periods = {}
    for _, row in dataset_df.iterrows():
        user = row['user']
        start = row['start']
        end = row['end']
        scenario = int(row['scenario'])
        
        if user not in malicious_periods:
            malicious_periods[user] = []
        
        malicious_periods[user].append({
            'start': start,
            'end': end,
            'scenario': scenario
        })
    
    return malicious_periods


def parse_r62_labels() -> Dict[str, List[Dict[str, str]]]:
    """
    Parse r6.2-*.csv files for r6.2 dataset.
    
    r6.2 label files have different format (individual events) - we need to
    extract the user and timestamp from each row, then find min/max per scenario.
    
    Returns:
        Dict mapping user_id -> list of {'start': start_date, 'end': end_date, 'scenario': ...}
    """
    import csv
    malicious_periods = {}
    
    # r6.2 has 5 scenario files
    for scenario_num in range(1, 6):
        scenario_file = ANSWERS_DIR / f"r6.2-{scenario_num}.csv"
        
        if not scenario_file.exists():
            print(f"  Warning: {scenario_file.name} not found, skipping")
            continue
            
        user_events = {}
        
        try:
            with open(scenario_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 5:
                        continue
                    
                    # Usually format is: type, event_id, date, user, pc, activity...
                    # Sometime it's just date, user, etc. Let's find the date and user fields.
                    # We know from sample that col 2 is date, col 3 is user.
                    date_str = row[2]
                    user = row[3]
                    
                    # Simple validation
                    if len(user) > 10 or len(date_str) < 10:
                        continue
                        
                    try:
                        # Parse datetime
                        dt = datetime.strptime(date_str, "%m/%d/%Y %H:%M:%S")
                        
                        if user not in user_events:
                            user_events[user] = []
                        user_events[user].append(dt)
                    except ValueError:
                        continue
                        
            # Aggregate to min/max per user
            for user, dt_list in user_events.items():
                min_dt = min(dt_list)
                max_dt = max(dt_list)
                
                if user not in malicious_periods:
                    malicious_periods[user] = []
                    
                malicious_periods[user].append({
                    'start': min_dt.strftime("%m/%d/%Y %H:%M:%S"),
                    'end': max_dt.strftime("%m/%d/%Y %H:%M:%S"),
                    'scenario': scenario_num
                })
        except Exception as e:
            print(f"  Warning: Could not read {scenario_file.name}: {e}")
            continue
    
    return malicious_periods


def save_labels(labels: Dict, dataset_version: str, output_dir: Path) -> Path:
    """Save labels to JSON file."""
    output_file = output_dir / f"{dataset_version}_insiders.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'version': dataset_version,
            'total_insiders': len(labels),
            'insiders': labels
        }, f, indent=2)
    
    return output_file


def parse_all_datasets() -> Dict[str, Dict]:
    """Parse labels for all datasets."""
    results = {}
    
    # r4.2
    print("Parsing r4.2 labels...")
    r42_labels = parse_insiders_csv(4.2)
    save_labels(r42_labels, 'r4.2', OUTPUT_DIR)
    results['r4.2'] = {'insiders': len(r42_labels)}
    print(f"  Found {len(r42_labels)} insiders")
    
    # r5.2
    print("\nParsing r5.2 labels...")
    r52_labels = parse_insiders_csv(5.2)
    save_labels(r52_labels, 'r5.2', OUTPUT_DIR)
    results['r5.2'] = {'insiders': len(r52_labels)}
    print(f"  Found {len(r52_labels)} insiders")
    
    # r6.2
    print("\nParsing r6.2 labels...")
    r62_labels = parse_r62_labels()
    save_labels(r62_labels, 'r6.2', OUTPUT_DIR)
    results['r6.2'] = {'insiders': len(r62_labels)}
    print(f"  Found {len(r62_labels)} insiders")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse CERT insider threat labels')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['r4.2', 'r5.2', 'r6.2', 'all'],
                       help='Dataset version to parse')
    args = parser.parse_args()
    
    print("=" * 60)
    print("CERT Label Parser")
    print("=" * 60)
    
    if args.dataset == 'all':
        results = parse_all_datasets()
    elif args.dataset == 'r4.2':
        labels = parse_insiders_csv(4.2)
        save_labels(labels, 'r4.2', OUTPUT_DIR)
        results = {'r4.2': {'insiders': len(labels)}}
        print(f"Found {len(labels)} insiders in r4.2")
    elif args.dataset == 'r5.2':
        labels = parse_insiders_csv(5.2)
        save_labels(labels, 'r5.2', OUTPUT_DIR)
        results = {'r5.2': {'insiders': len(labels)}}
        print(f"Found {len(labels)} insiders in r5.2")
    elif args.dataset == 'r6.2':
        labels = parse_r62_labels()
        save_labels(labels, 'r6.2', OUTPUT_DIR)
        results = {'r6.2': {'insiders': len(labels)}}
        print(f"Found {len(labels)} insiders in r6.2")
    
    print("\n" + "=" * 60)
    print("Summary:")
    for dataset, info in results.items():
        print(f"  {dataset}: {info['insiders']} insiders")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()

