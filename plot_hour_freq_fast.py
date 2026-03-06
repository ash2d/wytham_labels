#!/usr/bin/env python3
"""
Faster version of plot_hour_freq_fast.py:
- Vectorized parsing with pandas.str.extract
- Drop duplicates with drop_duplicates (per-file)
- Aggregate per-file Counters
- Optional parallel file reading (--workers)
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

EXTRACT_RE = r'(?P<date>\d{8})_(?P<time>\d{6})_(?P<clip>\d+)'


def get_checkpoint_files(base_dir: Path):
    return list(base_dir.rglob("checkpoint_*.csv"))


def process_checkpoint_file(csv_path: Path) -> Counter:
    """Return Counter of hour -> count for unique 1-minute recordings in this file."""
    try:
        df = pd.read_csv(
            csv_path,
            usecols=["file"],
            dtype={"file": str},
            engine="c",
            low_memory=False,
            memory_map=True,
        )
    except Exception:
        # fallback simple read
        try:
            df = pd.read_csv(csv_path, usecols=["file"])
        except Exception:
            return Counter()

    if df.empty:
        return Counter()

    s = df["file"].dropna().astype(str)

    # Vectorized extract of date/time/clip; returns DataFrame with columns ['date','time','clip']
    extracted = s.str.extract(EXTRACT_RE, expand=True)
    if extracted.empty:
        return Counter()

    # Keep rows with valid date/time
    extracted = extracted.dropna(subset=["date", "time"])
    if extracted.empty:
        return Counter()

    # Build recording id (date_time) and drop duplicates per-file so each 1-min recording counts once
    extracted["rec_id"] = extracted["date"] + "_" + extracted["time"]
    extracted = extracted.drop_duplicates(subset="rec_id")

    # Hour is first two chars of time
    hours = extracted["time"].str[:2]
    return Counter(hours.tolist())


def main():
    parser = argparse.ArgumentParser(
        description="Generate hour-of-day frequency histogram from checkpoint CSVs (faster)."
    )
    parser.add_argument(
        "--dir", type=str, default="output", help="Base directory containing checkpoint CSV files"
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="hour_frequency_histogram.png",
        help="Path to save the output plot",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="hour_frequency_counts.csv",
        help="Path to save hour frequency counts CSV",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1))),
        help="Number of worker threads for parallel CSV reading (I/O bound). Set 1 to disable.",
    )
    args = parser.parse_args()

    base_dir = Path(args.dir)
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist.")
        return

    csv_files = get_checkpoint_files(base_dir)
    if not csv_files:
        print("No checkpoint CSV files found (checkpoint_*.csv).")
        return

    total_counter = Counter()

    if args.workers and args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as exe:
            futures = {exe.submit(process_checkpoint_file, p): p for p in csv_files}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing CSVs"):
                try:
                    cnt = fut.result()
                    total_counter.update(cnt)
                except Exception:
                    # ignore problematic file
                    pass
    else:
        for p in tqdm(csv_files, desc="Processing CSVs"):
            try:
                total_counter.update(process_checkpoint_file(p))
            except Exception:
                pass

    if not total_counter:
        print("No valid recordings found.")
        return

    hour_labels = [f"{h:02d}" for h in range(24)]
    counts = [total_counter.get(h, 0) for h in hour_labels]

    df_counts = pd.DataFrame({"Hour": hour_labels, "Frequency": counts})

    total_unique_recordings = int(df_counts["Frequency"].sum())
    print(f"Found {total_unique_recordings} unique 1-minute recordings across checkpoint files.")
    print("\nHour frequency counts:")
    print(df_counts)

    plt.figure(figsize=(12, 6))
    plt.grid(axis="y", alpha=0.75, linestyle="--")
    plt.bar(df_counts["Hour"], df_counts["Frequency"], color="steelblue")
    plt.title("Recording Frequency by Hour (Unique 1-minute recordings)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Recordings")
    plt.tight_layout()

    try:
        plt.savefig(args.output_plot, dpi=300)
        print(f"\nHistogram saved to {args.output_plot}")

        df_counts.to_csv(args.output_csv, index=False)
        print(f"Hour frequency counts saved to {args.output_csv}")
    except Exception as e:
        print(f"Error saving outputs: {e}")


if __name__ == "__main__":
    main()
