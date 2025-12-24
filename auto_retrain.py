#!/usr/bin/env python3
"""
Automated Retraining Script for Lead Scorer

Checks if the training dataset has grown significantly (e.g., by 50 samples)
and retrains all models if so.
"""

import os
import pandas as pd
from models import train_and_save_models

# Configuration
CSV_PATH = "Data/csvfile.csv"
LAST_SIZE_FILE = "last_dataset_size.txt"
GROWTH_THRESHOLD = 50

def get_dataset_size(csv_path):
    """Get the number of rows in the CSV."""
    if not os.path.exists(csv_path):
        return 0
    try:
        df = pd.read_csv(csv_path)
        return len(df)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 0

def get_last_size():
    """Get the last recorded dataset size."""
    if os.path.exists(LAST_SIZE_FILE):
        try:
            with open(LAST_SIZE_FILE, 'r') as f:
                return int(f.read().strip())
        except:
            return 0
    return 0

def save_last_size(size):
    """Save the current dataset size."""
    with open(LAST_SIZE_FILE, 'w') as f:
        f.write(str(size))

def main():
    current_size = get_dataset_size(CSV_PATH)
    last_size = get_last_size()

    print(f"Current dataset size: {current_size}")
    print(f"Last recorded size: {last_size}")

    if current_size >= last_size + GROWTH_THRESHOLD:
        print(f"Dataset grew by {current_size - last_size} samples (>= {GROWTH_THRESHOLD}). Retraining models...")
        saved = train_and_save_models(csv_path=CSV_PATH, output_prefix="lead_classifier_")
        if saved:
            print(f"Retrained and saved {len(saved)} models.")
            save_last_size(current_size)
        else:
            print("Retraining failed.")
    else:
        print(f"Dataset growth ({current_size - last_size}) below threshold ({GROWTH_THRESHOLD}). No retraining needed.")

if __name__ == "__main__":
    main()