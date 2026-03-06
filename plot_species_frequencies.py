#!/usr/bin/env python3
"""
Analyze species frequencies across all prediction CSVs.
Counts unique species occurrences per audio file where logit score is above threshold.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm
from collections import Counter
import sys

# Increase CSV field size limit for potentially large fields
import csv
csv.field_size_limit(sys.maxsize)

def get_output_files(base_dir):
    """Recursively find all prediction CSVs in predictions_partitioned folders."""
    # Pattern: Site/predictions_partitioned/checkpoint_*.csv
    # We look for any .csv inside a 'predictions_partitioned' folder
    return list(base_dir.rglob("predictions_partitioned/*.csv"))

def process_file(csv_path, threshold):
    """
    Process a single CSV file.
    Returns a set of (filename, species) tuples for detections > threshold.
    """
    try:
        # Read only necessary columns to save memory if possible?
        # But we don't know exact column names for top/score/file beforehand without reading header.
        # So we read the whole thing.
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return set()

    if df.empty:
        return set()

    # Determine file column
    if 'file_path' in df.columns:
        file_col = 'file_path'
    elif 'file' in df.columns:
        file_col = 'file'
    else:
        # Fallback: check if first column looks like a file path
        # Heuristic: Check column name or content
        first_col = df.columns[0]
        if 'wav' in str(df.iloc[0][first_col]).lower():
            file_col = first_col
        else:
            return set()

    detections = set()
    
    # Iterate through the 10 predictions
    for i in range(1, 11):
        top_col = f'top{i}'
        score_col = f'score{i}'
        
        if top_col not in df.columns or score_col not in df.columns:
            continue
            
        # Filter rows for this prediction rank
        # We need rows where score > threshold
        mask = df[score_col] > threshold
        
        if not mask.any():
            continue
            
        # Get valid rows: file identifier and species name
        valid_rows = df.loc[mask, [file_col, top_col]]
        
        # Add to set. 
        # We store (file_path_value, species_name)
        # We cast to string to be safe
        for _, row in valid_rows.iterrows():
            f_identifier = str(row[file_col])
            species = str(row[top_col])
            
            # Simple cleaning if needed (e.g. handle nan)
            if species.lower() == 'nan' or not species:
                continue
                
            detections.add((f_identifier, species))
            
    return detections

def main():
    parser = argparse.ArgumentParser(description="Generate species frequency histogram from predictions.")
    parser.add_argument("--dir", type=str, default="output", help="Base output directory containing site folders")
    parser.add_argument("--threshold", type=float, default=10.0, help="Logit threshold (default: 10.0)")
    parser.add_argument("--output-plot", type=str, default="species_frequency_histogram.png", help="Path to save the output plot")
    
    args = parser.parse_args()
    
    base_dir = Path(args.dir)
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist.")
        return

    print(f"Searching for CSVs in {base_dir}...")
    csv_files = get_output_files(base_dir)
    print(f"Found {len(csv_files)} CSV files.")
    
    if not csv_files:
        print("No CSV files found.")
        return

    # Global set to track unique (file, species) across all CSVs 
    all_detections = set() 

    print(f"Processing files with threshold > {args.threshold}...")
    
    # Use tqdm for progress bar
    for csv_file in tqdm(csv_files):
        file_detections = process_file(csv_file, args.threshold)
        all_detections.update(file_detections)
        
    print(f"\nProcessing complete.")
    print(f"Total unique (file, species) pairs above threshold: {len(all_detections)}")
    
    if not all_detections:
        print("No detections found above threshold.")
        return
        
    # Count frequencies of species
    # Extract just the species part from the tuples
    species_list = [species for _, species in all_detections]
    species_counts = Counter(species_list)
    
    # Create DataFrame for plotting
    df_counts = pd.DataFrame.from_dict(species_counts, orient='index', columns=['Frequency']).reset_index()
    df_counts.columns = ['Species', 'Frequency']
    df_counts = df_counts.sort_values(by='Frequency', ascending=False)
    
    num_species = len(df_counts)
    print(f"Found {num_species} unique species.")
    print("\nTop 10 Species:")
    print(df_counts.head(10))
    
    # Plotting
    # Dynamic figsize based on number of species
    # Basic height + space for bars
    plot_height = max(8, num_species * 0.25)
    
    # If too huge, cap it or split? For now just plot all or top N if user prefers (user didn't specify top N, but said 'histogram for the species', implying all)
    # But if there are 500 species, a single plot is unreadable.
    # Let's plot top 50 by default if > 50, but maybe user wants all.
    # I'll plot top 60 to keep it readable, and print a message.
    
    plt.figure(figsize=(15, 12))
    plt.grid(axis='y', alpha=0.75, linestyle='--')
    
    limit_plot = 60
    if len(df_counts) > limit_plot:
        print(f"\nPlotting top {limit_plot} species out of {len(df_counts)}.")
        plot_data = df_counts.head(limit_plot)
        title_suffix = f" (Top {limit_plot})"
    else:
        plot_data = df_counts
        title_suffix = ""
        
    # Bar plot
    plt.bar(plot_data['Species'], plot_data['Frequency'], color='steelblue')
    
    plt.xticks(rotation=90)
    plt.title(f"Species Frequency (Logit > {args.threshold}) - Unique Audio Files per Species{title_suffix}")
    plt.xlabel("Species")
    plt.ylabel("Number of Audio Files")
    plt.tight_layout()
    
    try:
        plt.savefig(args.output_plot, dpi=300)
        print(f"\nHistogram saved to {args.output_plot}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    main()
