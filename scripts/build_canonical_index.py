
import argparse
import logging
import sys
from pathlib import Path
from typing import List

import polars as pl
pl.Config.set_tbl_formatting("ASCII_FULL")  # Better table formatting for debugging
pl.Config.set_tbl_cols(-1)  # Better table formatting for debugging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def find_prediction_files(predictions_root: Path) -> List[Path]:
    """Find all prediction CSV files."""
    logging.info(f"Searching for prediction files in {predictions_root}...")
    files = list(predictions_root.glob("**/predictions_partitioned/*.csv"))
    print(files[0])
    if not files:
        logging.warning("No prediction files found.")
    else:
        logging.info(f"Found {len(files)} prediction files.")
    return files


def extract_site_from_path(file_path: Path, predictions_root: Path) -> str:
    """Extracts the site name from the file path."""
    try:
        # Get the parent directory of 'predictions_partitioned'
        relative_path = file_path.relative_to(predictions_root)
        # The site is the first part of the relative path
        site = relative_path.parts[0]
        return site
    except (ValueError, IndexError):
        logging.warning(f"Could not extract site from path: {file_path}")
        return "unknown"


def load_predictions(
    prediction_files: List[Path], predictions_root: Path
) -> pl.DataFrame:
    """Load predictions from CSV files into a Polars DataFrame."""
    if not prediction_files:
        return pl.DataFrame()

    logging.info("Loading prediction data...")
    df = pl.scan_csv(
        f"{predictions_root}/**/predictions_partitioned/*.csv",
        separator=","  # Prediction CSVs are tab-separated
    ).collect()
    
    # Extract site from file_path column (e.g., /mnt/.../G2-2/file.WAV -> G2-2)
    df = df.with_columns(
        pl.col("file_path")
        .str.extract(r"/([^/]+)/[^/]+\.(?:WAV|wav)$", 1)
        .alias("site")
    )

    logging.info(f"Loaded {len(df)} rows from prediction files.")
    return df


def process_predictions(df: pl.DataFrame, predictions_root: Path) -> pl.DataFrame:
    """Process the raw prediction DataFrame."""
    if df.is_empty():
        return df

    logging.info("Processing prediction data...")

    # Extract file stem from file_path (handles both .WAV and .wav)
    df = df.with_columns(
        pl.col("file_path")
        .str.extract(r"/([^/]+)\.(?:WAV|wav)$", 1)
        .alias("file_stem")
    )

    # Create clip_id
    df = df.with_columns(
        (
            pl.col("site") + "_" + 
            pl.col("file_stem") + "_" + 
            pl.col("chunk_idx").cast(pl.Utf8)
        ).alias("clip_id")
    )

    # Create lists for top species and scores
    top_species_cols = [f"top{i}" for i in range(1, 11)]
    top_scores_cols = [f"score{i}" for i in range(1, 11)]

    df = df.with_columns(
        [
            pl.concat_list([pl.col(c) for c in top_species_cols]).alias("top_species"),
            pl.concat_list([pl.col(c) for c in top_scores_cols]).alias("top_scores"),
        ]
    )

    # Rename columns
    df = df.rename({"top1": "top1_species", "score1": "top1_score"})
    
    # Add embedding_path: predictions_root/site/embeddings_partitioned/XX/part-0.parquet
    df = df.with_columns(
        (
            pl.lit(str(predictions_root)) + "/" +
            pl.col("site") + "/embeddings_partitioned/" +
            pl.col("checkpoint_id").cast(pl.Utf8) + "/part-0.parquet"
        ).alias("embedding_path")
    )
    
    required_cols = [
        "clip_id",
        "site",
        "file",
        "file_path",
        "file_stem",
        "chunk_idx",
        "start_time",
        "end_time",
        "checkpoint_id",
        "top1_species",
        "top1_score",
        "top_species",
        "top_scores",
        "embedding_path",
    ]
    
    # Ensure all required columns exist
    for col_name in required_cols:
        if col_name not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col_name))

    return df.select(required_cols)


def find_audio_files(audio_root: Path) -> pl.DataFrame:
    """Find all audio files and create a DataFrame with file metadata."""
    logging.info(f"Scanning for audio files in {audio_root}...")
    audio_files = list(audio_root.glob("**/*.wav"))
    if not audio_files:
        logging.warning("No audio files found.")
        return pl.DataFrame(
            {
                "audio_path": [],
                "site": [],
                "file_name": [],
                "file_stem": [],
                "file_exists": [],
            }
        )

    logging.info(f"Found {len(audio_files)} audio files.")

    # Create a DataFrame from the file paths
    audio_df = pl.DataFrame({"audio_path": [str(p) for p in audio_files]})

    audio_df = audio_df.with_columns(
        [
            pl.col("audio_path")
            .map_elements(lambda p: Path(p).parent.name, return_dtype=pl.Utf8)
            .alias("site"),
            pl.col("audio_path")
            .map_elements(lambda p: Path(p).name, return_dtype=pl.Utf8)
            .alias("file_name"),
        ]
    )

    audio_df = audio_df.with_columns(
        pl.col("file_name").str.replace(".wav", "").alias("file_stem")
    )
    audio_df = audio_df.with_columns(pl.lit(True).alias("file_exists"))

    return audio_df


# def join_data(
#     predictions_df: pl.DataFrame, audio_df: pl.DataFrame
# ) -> pl.DataFrame:
#     """Join prediction data with audio file data."""
#     if predictions_df.is_empty():
#         return predictions_df

#     logging.info("Joining prediction data with audio file metadata...")
    
#     # We join on site and file_stem to ensure we match the correct files
#     if audio_df.is_empty():
#         logging.warning("Audio DataFrame is empty. All clips will be marked as missing audio files.")
#         predictions_df = predictions_df.with_columns(pl.lit(False).alias("file_exists"))
#         return predictions_df
#     else:
#         merged_df = predictions_df.join(
#             audio_df, on=["site", "file_stem"], how="left"
#         )

#     # Log missing files
#     missing_files = merged_df.filter(pl.col("file_exists").is_null())
#     if not missing_files.is_empty():
#         logging.warning(f"Found {len(missing_files)} clips with no matching audio file.")
#         for row in missing_files.select("site", "file_stem").iter_rows():
#             logging.debug(f"Missing audio file for site '{row[0]}' and file '{row[1]}.wav'")

#     merged_df = merged_df.with_columns(
#         pl.col("file_exists").fill_null(False)
#     )

#     return merged_df

def join_data(
    predictions_df: pl.DataFrame, audio_df: pl.DataFrame
) -> pl.DataFrame:
    """Join prediction data with audio file data."""
    if predictions_df.is_empty():
        return predictions_df

    logging.info("Joining prediction data with audio file metadata...")
    
    # We join on site and file_stem to ensure we match the correct files
    if audio_df.is_empty():
        logging.warning("Audio DataFrame is empty. Using file_path from predictions as audio_path.")
        # Use file_path as audio_path and extract file_name from file_path
        predictions_df = predictions_df.with_columns([
            pl.col("file_path").alias("audio_path"),
            pl.col("file_path")
            .str.extract(r"/([^/]+\.(?:WAV|wav))$", 1)
            .alias("file_name"),
            pl.lit(False).alias("file_exists")
        ])
        return predictions_df
    else:
        merged_df = predictions_df.join(
            audio_df, on=["site", "file_stem"], how="left"
        )

    # Log missing files
    missing_files = merged_df.filter(pl.col("file_exists").is_null())
    if not missing_files.is_empty():
        logging.warning(f"Found {len(missing_files)} clips with no matching audio file.")
        for row in missing_files.select("site", "file_stem").iter_rows():
            logging.debug(f"Missing audio file for site '{row[0]}' and file '{row[1]}.wav'")

    # Fill nulls: for missing files, use file_path as audio_path and extract file_name
    merged_df = merged_df.with_columns([
        pl.when(pl.col("file_exists").is_null())
        .then(pl.col("file_path"))
        .otherwise(pl.col("audio_path"))
        .alias("audio_path"),
        
        pl.when(pl.col("file_exists").is_null())
        .then(pl.col("file_path").str.extract(r"/([^/]+\.(?:WAV|wav))$", 1))
        .otherwise(pl.col("file_name"))
        .alias("file_name"),
        
        pl.col("file_exists").fill_null(False)
    ])

    return merged_df


def add_datetime_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add datetime features based on filename and chunk times."""
    if df.is_empty() or "file_stem" not in df.columns:
        return df

    logging.info("Adding datetime features...")

    # Parse datetime from filename - handle both formats:
    # YYYYMMDD_HHMMSS (e.g., 20250714_194600)
    # YYYY-MM-DD_HH-MM-SS (e.g., 2025-07-14_19-46-00)
    df = df.with_columns(
        pl.when(pl.col("file_stem").str.contains("-"))
        .then(
            pl.col("file_stem").str.to_datetime("%Y-%m-%d_%H-%M-%S", strict=False)
        )
        .otherwise(
            pl.col("file_stem").str.to_datetime("%Y%m%d_%H%M%S", strict=False)
        )
        .alias("datetime_start_base")
    )

    # Calculate clip start and end datetimes
    df = df.with_columns(
        [
            (pl.col("datetime_start_base") + pl.duration(seconds=pl.col("start_time"))).alias(
                "datetime_start"
            ),
            (pl.col("datetime_start_base") + pl.duration(seconds=pl.col("end_time"))).alias(
                "datetime_end"
            ),
        ]
    )

    # Add individual datetime components
    df = df.with_columns(
        [
            pl.col("datetime_start").dt.year().alias("year"),
            pl.col("datetime_start").dt.month().alias("month"),
            pl.col("datetime_start").dt.day().alias("day"),
            pl.col("datetime_start").dt.hour().alias("hour"),
            pl.col("datetime_start").dt.minute().alias("minute"),
            pl.col("datetime_start").dt.date().alias("date"),
        ]
    )
    
    df = df.with_columns(
        (pl.col("end_time") - pl.col("start_time")).alias("duration_sec")
    )

    return df.drop("datetime_start_base")


def verify_embedding_files(df: pl.DataFrame) -> pl.DataFrame:
    """Verify that embedding files exist for each clip."""
    if df.is_empty() or "embedding_path" not in df.columns:
        return df
    
    logging.info("Verifying embedding file existence...")
    
    # Check if each embedding file exists
    df = df.with_columns(
        pl.col("embedding_path")
        .map_elements(lambda p: Path(p).exists() if p else False, return_dtype=pl.Boolean)
        .alias("embedding_exists")
    )
    
    # Log missing embedding files
    missing_embeddings = df.filter(~pl.col("embedding_exists"))
    if not missing_embeddings.is_empty():
        unique_missing = missing_embeddings.select("embedding_path").unique()
        logging.warning(f"Found {len(unique_missing)} unique embedding files missing.")
        for path in unique_missing["embedding_path"].head(10):
            logging.debug(f"Missing embedding file: {path}")
    else:
        logging.info("All embedding files exist.")
    
    return df


def validate_index(df: pl.DataFrame):
    """Perform validation checks on the final index."""
    if df.is_empty():
        logging.info("DataFrame is empty, skipping validation.")
        return

    logging.info("Performing validation checks...")

    total_clips = len(df)
    logging.info(f"Total clips: {total_clips}")

    clips_per_site = df.group_by("site").agg(pl.len().alias("count"))
    logging.info("Clips per site:")
    for row in clips_per_site.iter_rows(named=True):
        logging.info(f"  - {row['site']}: {row['count']}")

    clips_per_month = df.group_by("month").agg(pl.len().alias("count")).sort("month")
    logging.info("Clips per month:")
    for row in clips_per_month.iter_rows(named=True):
        logging.info(f"  - Month {row['month']}: {row['count']}")

    duplicate_clip_ids = df.group_by("clip_id").agg(pl.len().alias("count")).filter(pl.col("count") > 1)
    if not duplicate_clip_ids.is_empty():
        logging.warning(f"Found {len(duplicate_clip_ids)} duplicate clip_ids.")
    else:
        logging.info("No duplicate clip_ids found.")

    missing_audio_count = df.filter(~pl.col("file_exists")).height
    if missing_audio_count > 0:
        logging.warning(f"Total clips with missing audio files: {missing_audio_count}")
    else:
        logging.info("All clips have a corresponding audio file.")


def main():
    """Main script execution."""
    parser = argparse.ArgumentParser(
        description="Build a canonical 5-second clip index for a large bioacoustics dataset."
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help="Root directory for audio files.",
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        required=True,
        help="Root directory for prediction CSVs.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to save the output Parquet file.",
    )
    args = parser.parse_args()

    # 1. Find prediction files
    prediction_files = find_prediction_files(args.predictions_root)
    if not prediction_files:
        sys.exit(1)

    # 2. Load and process predictions
    predictions_df = load_predictions(prediction_files, args.predictions_root)
    processed_df = process_predictions(predictions_df, args.predictions_root)
    print(processed_df.head())

    # 3. Find audio files
    audio_df = find_audio_files(args.audio_root)

    # 4. Join data
    merged_df = join_data(processed_df, audio_df)

    # 5. Add datetime features
    final_df = add_datetime_features(merged_df)
    
    # 5.5. Verify embedding files exist
    final_df = verify_embedding_files(final_df)
    
    # 6. Final column selection and ordering
    final_cols = [
        "clip_id", "site", "audio_path", "file_name", "chunk_idx",
        "start_time", "end_time", "duration_sec",
        "datetime_start", "datetime_end", "year", "month", "day", "hour", "minute", "date",
        "checkpoint_id", "top1_species", "top1_score", "top_species", "top_scores",
        "embedding_path", "embedding_exists", "file_exists"
    ]
    
    # Ensure all final columns exist
    for col_name in final_cols:
        if col_name not in final_df.columns:
            final_df = final_df.with_columns(pl.lit(None).alias(col_name))

    final_df = final_df.select(final_cols)


    # 7. Validation
    validate_index(final_df)

    # 8. Save output
    logging.info(f"Saving canonical index to {args.output_path}...")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.write_parquet(args.output_path, compression="zstd")
    logging.info("Done.")


if __name__ == "__main__":
    main()
