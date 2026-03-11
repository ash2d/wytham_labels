#!/usr/bin/env python3
"""
Create spatial/temporal test splits and training selections for bioacoustics clips.

This script:
1. Loads a canonical clip index (parquet)
2. Creates spatial test split (350 clips from 5 held-out sites)
3. Creates temporal test split (350 clips from edge-of-month days)
4. Selects diversity-based training clips (1400) via k-means medoids
5. Selects proxy species-enriched training clips (900) via farthest-first
6. Creates a dev/calibration split (300) from training
7. Adds split labels and saves the annotated index

Usage:
    python create_splits.py --input canonical_index.parquet --output master_clip_index_with_splits.parquet --seed 42
"""

import argparse
import logging
import sys
from calendar import monthrange
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
from sklearn.cluster import KMeans

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

RANDOM_SEED = 42

# Spatial test split
N_SPATIAL_TEST_SITES = 5
CLIPS_PER_SPATIAL_SITE = 70
TOTAL_SPATIAL_TEST = N_SPATIAL_TEST_SITES * CLIPS_PER_SPATIAL_SITE  # 350

# Temporal test split
TEMPORAL_EDGE_DAYS_START = 3  # day <= 3
TEMPORAL_EDGE_DAYS_END = 2    # day >= (last_day - 2)
CLIPS_PER_MONTH_TEMPORAL = 50
N_MONTHS = 7
TOTAL_TEMPORAL_TEST = CLIPS_PER_MONTH_TEMPORAL * N_MONTHS  # 350

# Diversity sampling
CLIPS_PER_STRATUM_DIVERSITY = 100
DAWN_HOUR_END = 12  # Hours < 12 = dawn, >= 12 = dusk
N_STRATA = 14  # 7 months × 2 sessions
TOTAL_DIVERSITY = CLIPS_PER_STRATUM_DIVERSITY * N_STRATA  # 1400
MAX_CLIPS_PER_SITE_DIVERSITY = 3

# Species enrichment
N_TOP_SPECIES = 60
CLIPS_PER_SPECIES = 15
TOTAL_SPECIES_ENRICHMENT = N_TOP_SPECIES * CLIPS_PER_SPECIES  # 900
MAX_CLIPS_PER_SITE_SPECIES = 2
SPECIES_CANDIDATE_POOL_SIZE = 2000

# Dev split
DEV_SPLIT_SIZE = 300
FINAL_TRAIN_SIZE = 2000

# Spacing constraints
MIN_SPACING_MINUTES = 2

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def set_seed(seed: int) -> np.random.Generator:
    """Create a seeded random number generator."""
    return np.random.default_rng(seed)


def get_base_site(site: str) -> str:
    """
    Get the base site name, removing suffixes like '-2'.
    
    Examples:
        'A4-2' -> 'A4'
        'A4' -> 'A4'
        'H9-2' -> 'H9'
    """
    if '-' in site and site.split('-')[-1].isdigit():
        return site.rsplit('-', 1)[0]
    return site


def get_minute_key(row: dict) -> str:
    """Get a unique key for the minute of a clip (site + file_name)."""
    return f"{row['site']}_{row['file_name']}"


def get_spacing_key(row: dict) -> Tuple[str, str, float]:
    """Get key for spacing constraint: (site, date, start_time)."""
    return (row['site'], str(row['date']), row['start_time'])


def check_spacing_constraint(
    candidate_start: float,
    existing_starts: List[float],
    min_spacing_sec: float = MIN_SPACING_MINUTES * 60
) -> bool:
    """Check if candidate clip is at least min_spacing_sec away from all existing clips."""
    for start in existing_starts:
        if abs(candidate_start - start) < min_spacing_sec:
            return False
    return True


def filter_with_constraints(
    df: pl.DataFrame,
    n_select: int,
    rng: np.random.Generator,
    existing_minute_keys: Optional[Set[str]] = None,
    existing_site_day_times: Optional[Dict[Tuple[str, str], List[float]]] = None,
    max_per_site: Optional[int] = None,
    site_counts: Optional[Dict[str, int]] = None,
) -> pl.DataFrame:
    """
    Select clips from df respecting constraints:
    - No duplicate minute files
    - Minimum spacing within site/day
    - Optional max clips per site
    
    Returns selected clips as DataFrame.
    """
    if existing_minute_keys is None:
        existing_minute_keys = set()
    if existing_site_day_times is None:
        existing_site_day_times = {}
    if site_counts is None:
        site_counts = {}
    
    # Convert to list of dicts for processing
    rows = df.to_dicts()
    rng.shuffle(rows)
    
    selected = []
    local_minute_keys = set(existing_minute_keys)
    local_site_day_times = {k: list(v) for k, v in existing_site_day_times.items()}
    local_site_counts = dict(site_counts)
    
    for row in rows:
        if len(selected) >= n_select:
            break
        
        minute_key = get_minute_key(row)
        site = row['site']
        date_str = str(row['date'])
        start_time = row['start_time']
        site_day_key = (site, date_str)
        
        # Check minute key constraint
        if minute_key in local_minute_keys:
            continue
        
        # Check spacing constraint
        existing_starts = local_site_day_times.get(site_day_key, [])
        if not check_spacing_constraint(start_time, existing_starts):
            continue
        
        # Check site cap
        if max_per_site is not None:
            current_count = local_site_counts.get(site, 0)
            if current_count >= max_per_site:
                continue
        
        # Accept this clip
        selected.append(row)
        local_minute_keys.add(minute_key)
        if site_day_key not in local_site_day_times:
            local_site_day_times[site_day_key] = []
        local_site_day_times[site_day_key].append(start_time)
        local_site_counts[site] = local_site_counts.get(site, 0) + 1
    
    if not selected:
        return pl.DataFrame()
    return pl.DataFrame(selected)


def load_embeddings_for_clips(
    df: pl.DataFrame,
    embedding_col: str = "embedding_path",
    clip_id_col: str = "clip_id",
    chunk_idx_col: str = "chunk_idx",
    max_files: int = 50,
    sample_per_file: int = 500
) -> Dict[str, np.ndarray]:
    """
    Load embeddings for clips from their embedding files.
    
    Returns dict mapping clip_id -> embedding vector.
    
    For efficiency, limits the number of files loaded and samples clips per file.
    This is a best-effort loading for diversity sampling - if we can't load all,
    we fall back to random selection for the rest.
    """
    embeddings = {}
    
    # Group clips by embedding_path to minimize file reads
    paths_with_counts = (
        df.group_by(embedding_col)
        .agg([
            pl.col(clip_id_col).alias("clip_ids"),
            pl.col(chunk_idx_col).alias("chunk_indices"),
            pl.len().alias("count")
        ])
        .sort("count", descending=True)
        .head(max_files)  # Limit number of files to load
    )
    
    total_paths = len(paths_with_counts)
    logger.info(f"Loading embeddings from up to {total_paths} files...")
    
    for i, row in enumerate(paths_with_counts.iter_rows(named=True)):
        path = row[embedding_col]
        clip_ids = row["clip_ids"]
        chunk_indices = row["chunk_indices"]
        
        if not path:
            continue
            
        path_obj = Path(path)
        if not path_obj.exists():
            logger.debug(f"Embedding file not found: {path}")
            continue
        
        try:
            # Load embedding parquet - only load needed columns
            emb_df = pl.read_parquet(path)
            
            # Identify embedding columns (numeric columns starting with 'emb' or all float columns)
            emb_cols = [c for c in emb_df.columns if c.startswith('emb')]
            if not emb_cols:
                # Fallback: use all float columns except metadata
                metadata_cols = {'file', 'file_path', 'chunk_idx', 'start_time', 'end_time'}
                emb_cols = [c for c in emb_df.columns 
                           if c not in metadata_cols and emb_df[c].dtype in [pl.Float32, pl.Float64]]
            
            if not emb_cols:
                logger.warning(f"No embedding columns found in {path}")
                continue
            
            # Sample clips if too many
            indices_to_load = list(zip(clip_ids, chunk_indices))
            if len(indices_to_load) > sample_per_file:
                indices_to_load = indices_to_load[:sample_per_file]
            
            # Build lookup by chunk_idx for efficiency
            if 'chunk_idx' in emb_df.columns:
                # Convert to dict for O(1) lookup
                chunk_to_row = {}
                for j, chunk_idx in enumerate(emb_df['chunk_idx'].to_list()):
                    chunk_to_row[chunk_idx] = j
                
                for clip_id, chunk_idx in indices_to_load:
                    if chunk_idx in chunk_to_row:
                        row_idx = chunk_to_row[chunk_idx]
                        emb = emb_df[row_idx, emb_cols].to_numpy().flatten()
                        embeddings[clip_id] = emb
            else:
                # Assume row index = chunk_idx
                for clip_id, chunk_idx in indices_to_load:
                    if chunk_idx < len(emb_df):
                        emb = emb_df[chunk_idx, emb_cols].to_numpy().flatten()
                        embeddings[clip_id] = emb
            
            if (i + 1) % 10 == 0:
                logger.info(f"Loaded embeddings from {i + 1}/{total_paths} files ({len(embeddings)} clips so far)")
                
        except Exception as e:
            logger.warning(f"Failed to load embeddings from {path}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(embeddings)} embeddings")
    return embeddings


def compute_medoids(
    embeddings: Dict[str, np.ndarray],
    clip_ids: List[str],
    n_clusters: int
) -> List[str]:
    """
    Perform k-means clustering and return medoid clip_ids.
    
    Medoid = clip closest to each cluster centroid.
    """
    # Build embedding matrix
    valid_ids = [cid for cid in clip_ids if cid in embeddings]
    if len(valid_ids) < n_clusters:
        return valid_ids
    
    X = np.vstack([embeddings[cid] for cid in valid_ids])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    kmeans.fit(X)
    
    # Find medoid for each cluster (point closest to centroid)
    medoid_ids = []
    for i, centroid in enumerate(kmeans.cluster_centers_):
        cluster_mask = kmeans.labels_ == i
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        cluster_embeddings = X[cluster_indices]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        medoid_local_idx = np.argmin(distances)
        medoid_global_idx = cluster_indices[medoid_local_idx]
        medoid_ids.append(valid_ids[medoid_global_idx])
    
    return medoid_ids


def farthest_first_traversal(
    embeddings: Dict[str, np.ndarray],
    clip_ids: List[str],
    n_select: int,
    seed_id: Optional[str] = None,
    rng: Optional[np.random.Generator] = None
) -> List[str]:
    """
    Select n_select clips using farthest-first traversal in embedding space.
    """
    valid_ids = [cid for cid in clip_ids if cid in embeddings]
    if len(valid_ids) <= n_select:
        return valid_ids
    
    X = np.vstack([embeddings[cid] for cid in valid_ids])
    id_to_idx = {cid: i for i, cid in enumerate(valid_ids)}
    
    selected_indices = []
    
    # Start with seed or random point
    if seed_id and seed_id in id_to_idx:
        selected_indices.append(id_to_idx[seed_id])
    else:
        if rng is None:
            rng = np.random.default_rng(RANDOM_SEED)
        selected_indices.append(rng.integers(len(valid_ids)))
    
    # Min distances to selected set
    min_distances = np.full(len(valid_ids), np.inf)
    
    while len(selected_indices) < n_select:
        # Update min distances with last selected point
        last_selected = selected_indices[-1]
        distances = np.linalg.norm(X - X[last_selected], axis=1)
        min_distances = np.minimum(min_distances, distances)
        
        # Mark already selected as -inf
        min_distances[selected_indices] = -np.inf
        
        # Select farthest point
        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx)
    
    return [valid_ids[i] for i in selected_indices]


# =============================================================================
# SPLIT SELECTION FUNCTIONS
# =============================================================================


def select_spatial_test(
    df: pl.DataFrame,
    rng: np.random.Generator
) -> pl.DataFrame:
    """
    Select spatial test split: 5 random sites, 70 clips each.
    
    Ensures:
    - Spread across months
    - No duplicate minute files
    - ≥2 minute spacing within site/day
    
    Note: Sites with suffixes like '-2' are grouped with their base site.
    """
    logger.info("Selecting spatial test split...")
    
    # Add base_site column for grouping
    df = df.with_columns(
        pl.col("site").map_elements(get_base_site, return_dtype=pl.Utf8).alias("base_site")
    )
    
    # Get all unique base sites
    all_base_sites = df.select("base_site").unique().to_series().to_list()
    logger.info(f"Total base sites available: {len(all_base_sites)}")
    
    # Show site groupings
    site_groups = df.group_by("base_site").agg(
        pl.col("site").unique().alias("variants")
    ).sort("base_site")
    logger.info(f"Site groupings:\n{site_groups}")
    
    # Randomly select N_SPATIAL_TEST_SITES base sites
    rng.shuffle(all_base_sites)
    selected_base_sites = all_base_sites[:N_SPATIAL_TEST_SITES]
    logger.info(f"Selected spatial test base sites: {selected_base_sites}")
    
    # Filter to these base sites (includes all variants like A4 and A4-2)
    site_df = df.filter(pl.col("base_site").is_in(selected_base_sites))
    
    # Get all actual site names that were selected
    selected_sites = site_df.select("site").unique().to_series().to_list()
    logger.info(f"Actual site names included: {selected_sites}")
    
    all_selected = []
    
    for base_site in selected_base_sites:
        # Get all clips for this base site (includes all variants)
        base_site_clips = site_df.filter(pl.col("base_site") == base_site)
        actual_sites = base_site_clips.select("site").unique().to_series().to_list()
        logger.info(f"Base site {base_site} (variants: {actual_sites}): {len(base_site_clips)} clips available")
        
        # Try to spread across months
        months = base_site_clips.select("month").unique().to_series().to_list()
        clips_per_month = CLIPS_PER_SPATIAL_SITE // len(months) if months else CLIPS_PER_SPATIAL_SITE
        extra = CLIPS_PER_SPATIAL_SITE % len(months) if months else 0
        
        site_selected = []
        existing_minute_keys: Set[str] = set()
        existing_site_day_times: Dict[Tuple[str, str], List[float]] = {}
        
        for i, month in enumerate(sorted(months)):
            month_clips = base_site_clips.filter(pl.col("month") == month)
            n_to_select = clips_per_month + (1 if i < extra else 0)
            
            selected = filter_with_constraints(
                month_clips,
                n_to_select,
                rng,
                existing_minute_keys,
                existing_site_day_times
            )
            
            # Update tracking sets
            for row in selected.iter_rows(named=True):
                existing_minute_keys.add(get_minute_key(row))
                key = (row['site'], str(row['date']))
                if key not in existing_site_day_times:
                    existing_site_day_times[key] = []
                existing_site_day_times[key].append(row['start_time'])
            
            site_selected.append(selected)
        
        if site_selected:
            site_result = pl.concat(site_selected)
            logger.info(f"Base site {base_site}: selected {len(site_result)} clips")
            all_selected.append(site_result)
    
    if not all_selected:
        logger.error("No clips selected for spatial test!")
        return pl.DataFrame()
    
    result = pl.concat(all_selected)
    logger.info(f"Total spatial test clips: {len(result)}")
    
    # Log distribution across actual sites
    site_distribution = result.group_by("site").agg(pl.len().alias("count"))
    logger.info(f"Distribution across actual sites:\n{site_distribution}")
    
    return result


def select_temporal_test(
    df: pl.DataFrame,
    spatial_test_sites: List[str],
    rng: np.random.Generator
) -> pl.DataFrame:
    """
    Select temporal test split: 50 clips per month from edge days.
    
    Edge days: day <= 3 OR day >= (last_day_of_month - 2)
    """
    logger.info("Selecting temporal test split...")
    
    # Exclude spatial test sites
    df = df.filter(~pl.col("site").is_in(spatial_test_sites))
    logger.info(f"Clips available after excluding spatial sites: {len(df)}")
    
    # Define edge days for each month
    def is_edge_day(month: int, day: int, year: int = 2025) -> bool:
        last_day = monthrange(year, month)[1]
        return day <= TEMPORAL_EDGE_DAYS_START or day >= (last_day - TEMPORAL_EDGE_DAYS_END)
    
    # Add edge day flag
    df = df.with_columns(
        pl.struct(["year", "month", "day"])
        .map_elements(
            lambda x: is_edge_day(x['month'], x['day'], x['year']),
            return_dtype=pl.Boolean
        )
        .alias("is_edge_day")
    )
    
    # Filter to edge days
    edge_df = df.filter(pl.col("is_edge_day"))
    logger.info(f"Clips on edge days: {len(edge_df)}")
    
    # Select per month
    months = edge_df.select("month").unique().to_series().to_list()
    logger.info(f"Months with edge day clips: {sorted(months)}")
    
    all_selected = []
    existing_minute_keys: Set[str] = set()
    existing_site_day_times: Dict[Tuple[str, str], List[float]] = {}
    
    for month in sorted(months):
        month_clips = edge_df.filter(pl.col("month") == month)
        
        selected = filter_with_constraints(
            month_clips,
            CLIPS_PER_MONTH_TEMPORAL,
            rng,
            existing_minute_keys,
            existing_site_day_times
        )
        
        # Update tracking
        for row in selected.iter_rows(named=True):
            existing_minute_keys.add(get_minute_key(row))
            key = (row['site'], str(row['date']))
            if key not in existing_site_day_times:
                existing_site_day_times[key] = []
            existing_site_day_times[key].append(row['start_time'])
        
        logger.info(f"Month {month}: selected {len(selected)} clips")
        all_selected.append(selected)
    
    if not all_selected:
        logger.error("No clips selected for temporal test!")
        return pl.DataFrame()
    
    result = pl.concat(all_selected)
    logger.info(f"Total temporal test clips: {len(result)}")
    return result


def select_diversity(
    df: pl.DataFrame,
    rng: np.random.Generator,
    existing_clip_ids: Set[str],
    use_embeddings: bool = True
) -> pl.DataFrame:
    """
    Select 1400 diversity-based training clips via k-means medoids.
    
    Stratify by (month × session), where session = dawn/dusk.
    100 clips per stratum.
    
    If use_embeddings=False or embedding loading fails, falls back to random selection.
    """
    logger.info("Selecting diversity-based training clips...")
    
    # Exclude already selected clips
    df = df.filter(~pl.col("clip_id").is_in(existing_clip_ids))
    logger.info(f"Clips available for diversity selection: {len(df)}")
    
    if df.is_empty():
        logger.error("No clips available for diversity selection!")
        return pl.DataFrame()
    
    # Add session column
    df = df.with_columns(
        pl.when(pl.col("hour") < DAWN_HOUR_END)
        .then(pl.lit("dawn"))
        .otherwise(pl.lit("dusk"))
        .alias("session")
    )
    
    # Get strata
    strata = df.group_by(["month", "session"]).agg(pl.len().alias("count"))
    logger.info(f"Strata counts:\n{strata}")
    
    if strata.is_empty():
        logger.error("No strata found!")
        return pl.DataFrame()
    
    all_selected = []
    existing_minute_keys: Set[str] = set()
    existing_site_day_times: Dict[Tuple[str, str], List[float]] = {}
    
    for stratum_idx, row in enumerate(strata.iter_rows(named=True)):
        month, session = row['month'], row['session']
        stratum_df = df.filter(
            (pl.col("month") == month) & (pl.col("session") == session)
        )
        
        logger.info(f"Stratum {stratum_idx + 1}/{len(strata)} ({month}, {session}): {len(stratum_df)} clips available")
        
        # Decide whether to use embeddings or random selection
        use_kmeans = use_embeddings and len(stratum_df) >= CLIPS_PER_STRATUM_DIVERSITY
        
        if use_kmeans:
            logger.info(f"Attempting k-means medoid selection...")
            try:
                # Load embeddings for this stratum (with limits for efficiency)
                embeddings = load_embeddings_for_clips(stratum_df, max_files=30, sample_per_file=300)
                
                if len(embeddings) < CLIPS_PER_STRATUM_DIVERSITY:
                    logger.warning(f"Only {len(embeddings)} embeddings loaded, need {CLIPS_PER_STRATUM_DIVERSITY}. Falling back to random selection.")
                    use_kmeans = False
                else:
                    # K-means clustering and medoid selection
                    clip_ids = stratum_df.select("clip_id").to_series().to_list()
                    logger.info(f"Running k-means with {len(embeddings)} embeddings...")
                    medoid_ids = compute_medoids(embeddings, clip_ids, CLIPS_PER_STRATUM_DIVERSITY)
                    logger.info(f"K-means returned {len(medoid_ids)} medoids")
                    
                    # Filter medoids through constraints
                    medoid_df = stratum_df.filter(pl.col("clip_id").is_in(medoid_ids))
                    selected = filter_with_constraints(
                        medoid_df,
                        CLIPS_PER_STRATUM_DIVERSITY,
                        rng,
                        existing_minute_keys,
                        existing_site_day_times,
                        max_per_site=MAX_CLIPS_PER_SITE_DIVERSITY
                    )
                    
                    # If we didn't get enough from medoids, supplement with random
                    if len(selected) < CLIPS_PER_STRATUM_DIVERSITY:
                        logger.info(f"Got {len(selected)} from medoids, supplementing with random...")
                        remaining_df = stratum_df.filter(~pl.col("clip_id").is_in(selected["clip_id"]))
                        supplement = filter_with_constraints(
                            remaining_df,
                            CLIPS_PER_STRATUM_DIVERSITY - len(selected),
                            rng,
                            existing_minute_keys,
                            existing_site_day_times,
                            max_per_site=MAX_CLIPS_PER_SITE_DIVERSITY
                        )
                        if not supplement.is_empty():
                            selected = pl.concat([selected, supplement])
            except Exception as e:
                logger.warning(f"K-means selection failed: {e}. Falling back to random selection.")
                use_kmeans = False
        
        if not use_kmeans:
            # Random selection with constraints
            logger.info(f"Using random selection for stratum ({month}, {session})")
            selected = filter_with_constraints(
                stratum_df,
                CLIPS_PER_STRATUM_DIVERSITY,
                rng,
                existing_minute_keys,
                existing_site_day_times,
                max_per_site=MAX_CLIPS_PER_SITE_DIVERSITY
            )
        
        # Update tracking
        for row in selected.iter_rows(named=True):
            existing_minute_keys.add(get_minute_key(row))
            key = (row['site'], str(row['date']))
            if key not in existing_site_day_times:
                existing_site_day_times[key] = []
            existing_site_day_times[key].append(row['start_time'])
        
        logger.info(f"Stratum ({month}, {session}): selected {len(selected)} clips")
        all_selected.append(selected)
    
    if not all_selected:
        logger.error("No clips selected for diversity!")
        return pl.DataFrame()
    
    result = pl.concat(all_selected)
    logger.info(f"Total diversity clips: {len(result)}")
    return result


def select_species_enrichment(
    df: pl.DataFrame,
    rng: np.random.Generator,
    existing_clip_ids: Set[str],
    use_embeddings: bool = True
) -> pl.DataFrame:
    """
    Select 900 proxy species-enriched training clips.
    
    For top 60 species, select 15 clips per species using farthest-first traversal.
    If use_embeddings=False, uses random selection instead.
    """
    logger.info("Selecting species-enriched training clips...")
    
    # Exclude already selected clips
    df = df.filter(~pl.col("clip_id").is_in(existing_clip_ids))
    
    # Get top species by frequency in top1_species
    species_counts = (
        df.group_by("top1_species")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(N_TOP_SPECIES)
    )
    top_species = species_counts.select("top1_species").to_series().to_list()
    logger.info(f"Top {len(top_species)} species selected for enrichment")
    
    all_selected = []
    selected_clip_ids: Set[str] = set()
    existing_minute_keys: Set[str] = set()
    existing_site_day_times: Dict[Tuple[str, str], List[float]] = {}
    
    for species in top_species:
        # Find clips where species appears in top_species list
        # and get corresponding score
        species_df = df.filter(
            pl.col("top_species").list.contains(species)
        ).filter(
            ~pl.col("clip_id").is_in(selected_clip_ids)
        )
        
        if species_df.is_empty():
            logger.warning(f"No clips found for species: {species}")
            continue
        
        # Get score for this species in each clip
        species_df = species_df.with_columns(
            pl.struct(["top_species", "top_scores"])
            .map_elements(
                lambda x: x['top_scores'][x['top_species'].index(species)] 
                         if species in x['top_species'] else 0.0,
                return_dtype=pl.Float64
            )
            .alias("species_score")
        )
        
        # Sort by score and take top candidates
        candidates = species_df.sort("species_score", descending=True).head(SPECIES_CANDIDATE_POOL_SIZE)
        logger.info(f"Species '{species}': {len(candidates)} candidates")
        
        # Decide whether to use embeddings
        use_fft = use_embeddings and len(candidates) >= CLIPS_PER_SPECIES
        
        if use_fft:
            try:
                # Load embeddings for candidates
                embeddings = load_embeddings_for_clips(candidates, max_files=10, sample_per_file=200)
                
                if len(embeddings) < CLIPS_PER_SPECIES:
                    logger.warning(f"Only {len(embeddings)} embeddings for species '{species}', using random selection")
                    use_fft = False
                else:
                    # Farthest-first traversal
                    clip_ids = candidates.select("clip_id").to_series().to_list()
                    fft_ids = farthest_first_traversal(embeddings, clip_ids, CLIPS_PER_SPECIES * 3, rng=rng)
                    
                    # Filter through constraints
                    fft_df = candidates.filter(pl.col("clip_id").is_in(fft_ids))
                    selected = filter_with_constraints(
                        fft_df,
                        CLIPS_PER_SPECIES,
                        rng,
                        existing_minute_keys,
                        existing_site_day_times,
                        max_per_site=MAX_CLIPS_PER_SITE_SPECIES
                    )
                    
                    # Supplement if needed
                    if len(selected) < CLIPS_PER_SPECIES:
                        remaining = candidates.filter(~pl.col("clip_id").is_in(selected["clip_id"]))
                        supplement = filter_with_constraints(
                            remaining,
                            CLIPS_PER_SPECIES - len(selected),
                            rng,
                            existing_minute_keys,
                            existing_site_day_times,
                            max_per_site=MAX_CLIPS_PER_SITE_SPECIES
                        )
                        if not supplement.is_empty():
                            selected = pl.concat([selected, supplement])
            except Exception as e:
                logger.warning(f"FFT selection failed for '{species}': {e}. Using random selection.")
                use_fft = False
        
        if not use_fft:
            # Random selection with constraints
            selected = filter_with_constraints(
                candidates,
                CLIPS_PER_SPECIES,
                rng,
                existing_minute_keys,
                existing_site_day_times,
                max_per_site=MAX_CLIPS_PER_SITE_SPECIES
            )
        
        # Update tracking
        for row in selected.iter_rows(named=True):
            selected_clip_ids.add(row['clip_id'])
            existing_minute_keys.add(get_minute_key(row))
            key = (row['site'], str(row['date']))
            if key not in existing_site_day_times:
                existing_site_day_times[key] = []
            existing_site_day_times[key].append(row['start_time'])
        
        logger.info(f"Species '{species}': selected {len(selected)} clips")
        all_selected.append(selected)
    
    if not all_selected:
        logger.error("No clips selected for species enrichment!")
        return pl.DataFrame()
    
    result = pl.concat(all_selected)
    logger.info(f"Total species enrichment clips: {len(result)}")
    return result


def create_dev_split(
    df: pl.DataFrame,
    train_clip_ids: Set[str],
    rng: np.random.Generator
) -> Tuple[Set[str], Set[str]]:
    """
    Create dev/calibration split from training clips.
    
    Groups by (site, date, file_name) and randomly selects groups until reaching target.
    
    Returns (dev_clip_ids, remaining_train_clip_ids)
    """
    logger.info("Creating dev/calibration split...")
    
    # Filter to training clips
    train_df = df.filter(pl.col("clip_id").is_in(train_clip_ids))
    
    # Group by (site, date, file_name)
    groups = train_df.group_by(["site", "date", "file_name"]).agg(
        pl.col("clip_id").alias("clip_ids"),
        pl.len().alias("count")
    )
    
    # Also get month for distribution tracking
    groups = groups.join(
        train_df.select(["site", "date", "file_name", "month"]).unique(),
        on=["site", "date", "file_name"],
        how="left"
    )
    
    # Shuffle groups
    group_list = groups.to_dicts()
    rng.shuffle(group_list)
    
    dev_clip_ids: Set[str] = set()
    month_counts: Dict[int, int] = {}
    target_per_month = DEV_SPLIT_SIZE // N_MONTHS
    
    for group in group_list:
        if len(dev_clip_ids) >= DEV_SPLIT_SIZE:
            break
        
        month = group['month']
        current_month_count = month_counts.get(month, 0)
        
        # Try to balance across months (soft constraint)
        if current_month_count >= target_per_month * 2:
            continue
        
        clip_ids = group['clip_ids']
        if len(dev_clip_ids) + len(clip_ids) <= DEV_SPLIT_SIZE + 10:  # Allow slight overshoot
            for cid in clip_ids:
                dev_clip_ids.add(cid)
            month_counts[month] = current_month_count + len(clip_ids)
    
    remaining_train = train_clip_ids - dev_clip_ids
    
    logger.info(f"Dev split: {len(dev_clip_ids)} clips")
    logger.info(f"Dev month distribution: {month_counts}")
    logger.info(f"Remaining train: {len(remaining_train)} clips")
    
    return dev_clip_ids, remaining_train


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def add_split_columns(
    df: pl.DataFrame,
    spatial_test_ids: Set[str],
    temporal_test_ids: Set[str],
    diversity_ids: Set[str],
    species_enrichment_ids: Set[str],
    dev_ids: Set[str],
    train_ids: Set[str]
) -> pl.DataFrame:
    """Add all split-related columns to the dataframe."""
    logger.info("Adding split columns to dataframe...")
    
    # Boolean flags
    df = df.with_columns([
        pl.col("clip_id").is_in(spatial_test_ids).alias("is_spatial_test"),
        pl.col("clip_id").is_in(temporal_test_ids).alias("is_temporal_test"),
        pl.col("clip_id").is_in(diversity_ids).alias("selected_diversity"),
        pl.col("clip_id").is_in(species_enrichment_ids).alias("selected_species_enrichment"),
        pl.col("clip_id").is_in(dev_ids).alias("is_dev"),
        pl.col("clip_id").is_in(train_ids).alias("is_train"),
    ])
    
    # Eligible train
    df = df.with_columns(
        (
            ~pl.col("is_spatial_test") &
            ~pl.col("is_temporal_test") &
            pl.col("embedding_exists") &
            pl.col("file_exists")
        ).alias("eligible_train")
    )
    
    # Selected train initial
    df = df.with_columns(
        (
            pl.col("selected_diversity") | pl.col("selected_species_enrichment")
        ).alias("selected_train_initial")
    )
    
    # Is pool
    df = df.with_columns(
        (
            pl.col("eligible_train") &
            ~pl.col("selected_train_initial") &
            ~pl.col("is_dev") &
            ~pl.col("is_train")
        ).alias("is_pool")
    )
    
    # Split label
    df = df.with_columns(
        pl.when(pl.col("is_spatial_test")).then(pl.lit("test_spatial"))
        .when(pl.col("is_temporal_test")).then(pl.lit("test_temporal"))
        .when(pl.col("is_train")).then(pl.lit("train"))
        .when(pl.col("is_dev")).then(pl.lit("dev"))
        .when(pl.col("is_pool")).then(pl.lit("pool"))
        .otherwise(pl.lit("excluded"))
        .alias("split_label")
    )
    
    return df


def validate_splits(df: pl.DataFrame):
    """Validate that splits are non-overlapping and counts are correct."""
    logger.info("Validating splits...")
    
    # Check counts
    spatial_count = df.filter(pl.col("is_spatial_test")).height
    temporal_count = df.filter(pl.col("is_temporal_test")).height
    diversity_count = df.filter(pl.col("selected_diversity")).height
    species_count = df.filter(pl.col("selected_species_enrichment")).height
    dev_count = df.filter(pl.col("is_dev")).height
    train_count = df.filter(pl.col("is_train")).height
    pool_count = df.filter(pl.col("is_pool")).height
    
    logger.info(f"Spatial test clips: {spatial_count} (target: {TOTAL_SPATIAL_TEST})")
    logger.info(f"Temporal test clips: {temporal_count} (target: {TOTAL_TEMPORAL_TEST})")
    logger.info(f"Diversity clips: {diversity_count} (target: {TOTAL_DIVERSITY})")
    logger.info(f"Species enrichment clips: {species_count} (target: {TOTAL_SPECIES_ENRICHMENT})")
    logger.info(f"Dev clips: {dev_count} (target: {DEV_SPLIT_SIZE})")
    logger.info(f"Train clips: {train_count} (target: {FINAL_TRAIN_SIZE})")
    logger.info(f"Pool clips: {pool_count}")
    
    # Check split label distribution
    split_counts = df.group_by("split_label").agg(pl.len().alias("count"))
    logger.info(f"Split label distribution:\n{split_counts}")
    
    # Assert no overlaps between test splits
    spatial_temporal_overlap = df.filter(
        pl.col("is_spatial_test") & pl.col("is_temporal_test")
    ).height
    assert spatial_temporal_overlap == 0, f"Overlap between spatial and temporal test: {spatial_temporal_overlap}"
    
    # Assert no overlap between test and train
    test_train_overlap = df.filter(
        (pl.col("is_spatial_test") | pl.col("is_temporal_test")) &
        (pl.col("is_train") | pl.col("is_dev"))
    ).height
    assert test_train_overlap == 0, f"Overlap between test and train/dev: {test_train_overlap}"
    
    # Assert no overlap between dev and train
    dev_train_overlap = df.filter(
        pl.col("is_dev") & pl.col("is_train")
    ).height
    assert dev_train_overlap == 0, f"Overlap between dev and train: {dev_train_overlap}"
    
    logger.info("All validation checks passed!")


def main():
    """Main script execution."""
    parser = argparse.ArgumentParser(
        description="Create spatial/temporal test splits and training selections."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input canonical clip index parquet file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output parquet file with split annotations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding-based selection (k-means, farthest-first). Use random selection instead. Much faster.",
    )
    args = parser.parse_args()
    
    # Initialize RNG
    rng = set_seed(args.seed)
    
    # Load canonical index
    logger.info(f"Loading canonical index from {args.input}...")
    df = pl.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} clips")
    
    # Log initial statistics
    logger.info(f"Sites: {df.select('site').n_unique()}")
    logger.info(f"Months: {sorted(df.select('month').unique().to_series().to_list())}")
    logger.info(f"Clips with embeddings: {df.filter(pl.col('embedding_exists')).height}")
    logger.info(f"Clips with audio files: {df.filter(pl.col('file_exists')).height}")
    
    use_embeddings = not args.skip_embeddings
    if args.skip_embeddings:
        logger.info("Skipping embedding-based selection (--skip-embeddings flag set)")
    
    # 1. Spatial test split
    spatial_test_df = select_spatial_test(df, rng)
    spatial_test_ids = set(spatial_test_df["clip_id"].to_list())
    spatial_test_sites = spatial_test_df.select("site").unique().to_series().to_list()
    
    # 2. Temporal test split
    temporal_test_df = select_temporal_test(df, spatial_test_sites, rng)
    temporal_test_ids = set(temporal_test_df["clip_id"].to_list())
    
    # 3. Define eligible training clips
    eligible_df = df.filter(
        ~pl.col("clip_id").is_in(spatial_test_ids) &
        ~pl.col("clip_id").is_in(temporal_test_ids) &
        pl.col("embedding_exists")
    )
        # pl.col("file_exists")
    logger.info(f"Eligible training clips: {len(eligible_df)}")
    
    # 4. Diversity sampling
    existing_ids = spatial_test_ids | temporal_test_ids
    diversity_df = select_diversity(eligible_df, rng, existing_ids, use_embeddings=use_embeddings)
    if diversity_df.is_empty():
        logger.error("Diversity selection returned no clips!")
        diversity_ids = set()
    else:
        diversity_ids = set(diversity_df["clip_id"].to_list())
    
    # 5. Species enrichment
    existing_ids = spatial_test_ids | temporal_test_ids | diversity_ids
    species_df = select_species_enrichment(eligible_df, rng, existing_ids, use_embeddings=use_embeddings)
    if species_df.is_empty():
        logger.error("Species enrichment selection returned no clips!")
        species_enrichment_ids = set()
    else:
        species_enrichment_ids = set(species_df["clip_id"].to_list())
    
    # 5. Species enrichment
    existing_ids = spatial_test_ids | temporal_test_ids | diversity_ids
    species_df = select_species_enrichment(eligible_df, rng, existing_ids)
    species_enrichment_ids = set(species_df["clip_id"].to_list())
    
    # Combined training selection
    train_initial_ids = diversity_ids | species_enrichment_ids
    logger.info(f"Total initial training selection: {len(train_initial_ids)}")
    
    # 6. Dev split
    dev_ids, train_ids = create_dev_split(df, train_initial_ids, rng)
    
    # 7. Add columns
    final_df = add_split_columns(
        df,
        spatial_test_ids,
        temporal_test_ids,
        diversity_ids,
        species_enrichment_ids,
        dev_ids,
        train_ids
    )
    
    # 8. Validate
    validate_splits(final_df)
    
    # 9. Save output
    logger.info(f"Saving annotated index to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.write_parquet(args.output, compression="zstd")
    logger.info("Done!")


if __name__ == "__main__":
    main()
