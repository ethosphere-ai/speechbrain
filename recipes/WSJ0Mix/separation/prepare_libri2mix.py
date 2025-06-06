#!/usr/bin/env/python3
"""Script to prepare Libri2Mix data for use with WSJ0Mix separation training script.

This script:
1. Creates train.csv, valid.csv, and test.csv files
2. Organizes the data in the format expected by the WSJ0Mix separation script
"""

import os
import csv
import glob
from pathlib import Path

def create_csv(data_dir, output_csv, split):
    """Create CSV file for a specific split (train/valid/test)"""
    # Get all mix files
    mix_files = sorted(glob.glob(os.path.join(data_dir, "mix", "*.wav")))
    
    # Create CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'mix_wav', 's1_wav', 's2_wav'])
        
        for mix_file in mix_files:
            # Get corresponding source files
            filename = os.path.basename(mix_file)
            s1_file = os.path.join(data_dir, "s1", filename)
            s2_file = os.path.join(data_dir, "s2", filename)
            
            # Write row to CSV
            writer.writerow([
                os.path.splitext(filename)[0],  # id
                mix_file,  # mix_wav
                s1_file,  # s1_wav
                s2_file,  # s2_wav
            ])

def prepare_libri2mix(libri2mix_dir, output_dir):
    """Prepare Libri2Mix data for WSJ0Mix separation training"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create CSV files for each split
    splits = {
        'train': 'train-360',  # Using train-360 for training
        'valid': 'dev',
        'test': 'test'
    }
    
    for split_name, libri_split in splits.items():
        # Create CSV file
        create_csv(
            os.path.join(libri2mix_dir, libri_split),
            os.path.join(output_dir, f"{split_name}.csv"),
            split_name
        )

if __name__ == "__main__":
    # Paths
    libri2mix_dir = "LibriMix/data_folder/Libri2Mix/wav8k/min"
    output_dir = "LibriMix/data_folder/Libri2Mix/wav8k/min/processed"
    
    # Prepare data
    prepare_libri2mix(libri2mix_dir, output_dir)
    print("Data preparation completed!") 