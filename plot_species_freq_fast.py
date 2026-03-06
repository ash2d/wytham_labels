#!/usr/bin/env python3
"""
Faster analysis of species frequencies across prediction CSVs.
Uses header-prefetch + usecols, vectorized selection, and optional multiprocessing.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import sys
import csv
import os
from concurrent.futures import ProcessPoolExecutor

csv.field_size_limit(sys.maxsize)


def get_output_files(base_dir):
    return list(base_dir.rglob("predictions_partitioned/*.csv"))


def process_file(args):
    """Process one CSV. Returns list of (file, species) tuples above threshold."""
    csv_path, threshold = args
    try:
        # Read header only to know columns
        header = pd.read_csv(csv_path, nrows=0)
    except Exception:
        return []

    cols = list(header.columns)

    # determine file column
    if 'file_path' in cols:
        file_col = 'file_path'
    elif 'file' in cols:
        file_col = 'file'
    else:
        # heuristic
        first_col = cols[0]
        # try peek first value
        try:
            peek = pd.read_csv(csv_path, usecols=[first_col], nrows=5, dtype=str)
            if peek.iloc[0, 0] and 'wav' in str(peek.iloc[0, 0]).lower():
                file_col = first_col
            else:
                return []
        except Exception:
            return []

    # find which top/score pairs exist
    top_cols = []
    score_cols = []
    for i in range(1, 11):
        t = f"top{i}"
        s = f"score{i}"
        if t in cols and s in cols:
            top_cols.append(t)
            score_cols.append(s)

    if not top_cols:
        return []

    usecols = [file_col] + top_cols + score_cols

    # dtype mapping: file/top -> string, score -> float32
    dtype = {file_col: str}
    for t in top_cols:
        dtype[t] = str
    for s in score_cols:
        # read scores as float32 to save memory
        dtype[s] = 'float32'

    try:
        df = pd.read_csv(csv_path, usecols=usecols, dtype=dtype, engine='c', low_memory=False, memory_map=True)
    except Exception:
        # fallback to full read
        try:
            df = pd.read_csv(csv_path)
            if file_col not in df.columns:
                return []
        except Exception:
            return []

    detections = []

    # vectorized per-column selection, append zipped arrays
    for t_col, s_col in zip(top_cols, score_cols):
        if s_col not in df.columns or t_col not in df.columns:
            continue
        # mask for score and non-null species
        try:
            mask = df[s_col].to_numpy() > threshold
        except Exception:
            # if conversion failed, skip this column
            continue
        if not mask.any():
            continue

        species_series = df[t_col]
        file_series = df[file_col]

        # refined mask to exclude null/empty species
        mask2 = mask & species_series.notna() & (species_series.astype(str) != "")
        if not mask2.any():
            continue

        files = file_series.to_numpy()[mask2]
        species = species_series.to_numpy()[mask2]

        # extend list with tuples (python-level loop over matched rows only)
        detections.extend(zip(files, species))

    return detections


def main():
    parser = argparse.ArgumentParser(description="Generate species frequency histogram from predictions.")
    parser.add_argument("--dir", type=str, default="output", help="Base output directory containing site folders")
    parser.add_argument("--threshold", type=float, default=10.0, help="Logit threshold (default: 10.0)")
    parser.add_argument("--output-plot", type=str, default="species_frequency_histogram.png", help="Path to save the output plot")
    parser.add_argument("--workers", type=int, default=0, help="Number of parallel workers (0 = auto)")
    args = parser.parse_args()

    base_dir = Path(args.dir)
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist.")
        return

    csv_files = get_output_files(base_dir)
    if not csv_files:
        print("No CSV files found.")
        return

    # prepare arguments for parallel mapping
    tasks = [(str(p), args.threshold) for p in csv_files]

    all_detections_list = []

    workers = args.workers or (os.cpu_count() or 1)

    if workers > 1 and len(tasks) > 1:
        # parallel processing
        with ProcessPoolExecutor(max_workers=workers) as ex:
            # ex.map yields results as they complete in order; wrap with tqdm
            for detections in tqdm(ex.map(process_file, tasks), total=len(tasks), desc="Processing CSVs"):
                if detections:
                    all_detections_list.extend(detections)
    else:
        # serial fallback
        for t in tqdm(tasks, desc="Processing CSVs"):
            detections = process_file(t)
            if detections:
                all_detections_list.extend(detections)

    # deduplicate once
    all_detections = set((str(f), str(s)) for f, s in all_detections_list)

    if not all_detections:
        print("No detections found above threshold.")
        return

    species_list = [species for _, species in all_detections]
    species_counts = Counter(species_list)

    df_counts = pd.DataFrame.from_dict(species_counts, orient='index', columns=['Frequency']).reset_index()
    df_counts.columns = ['Species', 'Frequency']
    df_counts = df_counts.sort_values(by='Frequency', ascending=False)

    num_species = len(df_counts)
    print(f"Found {num_species} unique species.")
    print("\nTop 10 Species:")
    print(df_counts.head(10))

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

    plt.bar(plot_data['Species'], plot_data['Frequency'], color='steelblue')
    plt.xticks(rotation=90)
    plt.title(f"Species Frequency (Logit > {args.threshold}) - Unique Audio Files per Species{title_suffix}")
    plt.xlabel("Species")
    plt.ylabel("Number of Audio Files")
    plt.tight_layout()

    try:
        plt.savefig(args.output_plot, dpi=300)
        print(f"\nHistogram saved to {args.output_plot}")
        df_counts.to_csv("species_frequency_counts.csv", index=False)
        print("Species frequency counts saved to species_frequency_counts.csv")
    except Exception as e:
        print(f"Error saving plot: {e}")


if __name__ == "__main__":
    main()
