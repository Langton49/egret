"""
=============================================================================
  Habitat Archetype Clustering & Conservation Scoring
  Mississippi Delta Avian Habitat Analysis
=============================================================================

Reads model_input.csv (cell × dekad rows), collapses to per-cell feature
profiles, discovers habitat archetypes via unsupervised clustering, and
scores each cell for habitat suitability and condition trajectory.

Pipeline:
  1. Load model_input.csv (2.7M rows)
  2. Collapse to per-cell profiles (~19,826 rows × ~70 features)
     - Temporal means, std, seasonal means per bird season
     - Multi-year trend slopes (2020-2025)
     - Flood frequency, hydroperiod variability
     - Bird diversity stats (where available)
  3. Standardize + PCA (retain 95% variance)
  4. Gaussian Mixture Model clustering → habitat archetypes
  5. Score each cell:
     - Habitat suitability (bird diversity of its cluster)
     - Condition trajectory (trend slopes → improving/stable/degrading)
     - Conservation priority (composite score)

Outputs:
  - habitat_results.csv     : per-cell scores, clusters, trajectories
  - cluster_profiles.csv    : mean feature values per cluster
  - model_diagnostics.json  : cluster stats, model parameters, validation
  - cell_profiles.csv       : intermediate per-cell feature matrix

Usage:
    python train_habitat_model.py                          # subsample 5000 cells
    python train_habitat_model.py --sample 2000            # smaller prototype
    python train_habitat_model.py --full                   # all cells
    python train_habitat_model.py --input model_input.csv  # custom input path
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
except ImportError:
    sys.exit("ERROR: scikit-learn required. Install: pip install scikit-learn")


# ===========================================================================
# CONFIGURATION
# ===========================================================================

INPUT_CSV = "./model_input.csv"
OUTPUT_DIR = "./model_output"
DEFAULT_SAMPLE = 5000
PCA_VARIANCE = 0.95
MAX_CLUSTERS = 20
MIN_CLUSTERS = 4

# Spectral indices to aggregate
INDEX_COLS = [
    "NDVI", "NDWI", "MNDWI", "NDMI", "EVI", "SAVI",
    "LSWI", "WRI", "wetland_moisture_index", "water_mask",
    "tc_wetness", "GCVI",
]

# Bird metrics
BIRD_COLS = [
    "n_observations", "n_species", "n_individuals", "shannon_diversity",
]

# Bird seasons
SEASONS = ["spring_migration", "breeding", "fall_migration", "winter"]

# Key indices for trend computation
TREND_INDICES = ["NDVI", "NDWI", "NDMI", "wetland_moisture_index", "water_mask", "EVI"]


# ===========================================================================
# STEP 1: LOAD DATA
# ===========================================================================

def load_input(path: str, sample_n: int = None) -> pd.DataFrame:
    """Load model_input.csv, optionally subsampling by cell."""
    print(f"\n  Loading {path} ...")
    
    # Read in chunks to manage memory
    chunks = []
    for chunk in pd.read_csv(path, chunksize=500_000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    
    print(f"    Rows: {len(df):,}")
    print(f"    Columns: {len(df.columns)}")
    
    # Parse dekad as datetime
    df["dekad"] = pd.to_datetime(df["dekad"])
    
    n_cells = df.groupby(["cell_x", "cell_y"]).ngroups
    print(f"    Unique cells: {n_cells:,}")
    
    if sample_n and sample_n < n_cells:
        print(f"\n  Subsampling {sample_n:,} cells from {n_cells:,} ...")
        # Get unique cells and sample
        cells = df.groupby(["cell_x", "cell_y"]).size().reset_index()[["cell_x", "cell_y"]]
        sampled = cells.sample(n=sample_n, random_state=42)
        df = df.merge(sampled, on=["cell_x", "cell_y"])
        print(f"    Rows after subsample: {len(df):,}")
    
    return df


# ===========================================================================
# STEP 2: COLLAPSE TO PER-CELL PROFILES
# ===========================================================================

def build_cell_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse cell × dekad rows into one feature vector per cell.
    Produces ~70 features per cell covering:
      - Overall stats (mean, std, min, max) for each index
      - Seasonal means for each bird season
      - Multi-year trend slopes
      - Flood frequency and hydroperiod variability
      - Bird diversity metrics (where available)
    """
    print("\n" + "=" * 60)
    print("  Building per-cell feature profiles")
    print("=" * 60)
    
    # Identify available index columns
    avail_indices = [c for c in INDEX_COLS if c in df.columns]
    avail_birds = [c for c in BIRD_COLS if c in df.columns]
    print(f"    Satellite indices: {len(avail_indices)}")
    print(f"    Bird metrics: {len(avail_birds)}")
    
    grouped = df.groupby(["cell_x", "cell_y"])
    
    # ── Overall temporal statistics ──
    print("    Computing overall temporal statistics ...")
    agg_funcs = {col: ["mean", "std", "min", "max", "median"] for col in avail_indices}
    stats = grouped.agg(agg_funcs)
    stats.columns = ["_".join(col) for col in stats.columns]
    profiles = stats.reset_index()
    
    # ── Seasonal means ──
    print("    Computing seasonal means ...")
    if "bird_season" in df.columns:
        for season in SEASONS:
            season_df = df[df["bird_season"] == season]
            if season_df.empty:
                continue
            season_means = season_df.groupby(["cell_x", "cell_y"])[avail_indices].mean()
            season_means.columns = [f"{c}_{season}" for c in season_means.columns]
            profiles = profiles.merge(season_means.reset_index(), on=["cell_x", "cell_y"], how="left")
    
    # ── Inter-annual variability (std across yearly means) ──
    print("    Computing inter-annual variability ...")
    if "year" in df.columns:
        yearly = df.groupby(["cell_x", "cell_y", "year"])[avail_indices].mean()
        yearly_var = yearly.groupby(["cell_x", "cell_y"]).std()
        yearly_var.columns = [f"{c}_interannual_std" for c in yearly_var.columns]
        profiles = profiles.merge(yearly_var.reset_index(), on=["cell_x", "cell_y"], how="left")
    
    # ── Multi-year trend slopes ──
    print("    Computing 6-year trend slopes ...")
    avail_trends = [c for c in TREND_INDICES if c in df.columns]
    
    if avail_trends and "dekad" in df.columns:
        # Convert time to numeric (days since start)
        df = df.copy()
        t_min = df["dekad"].min()
        df["t_days"] = (df["dekad"] - t_min).dt.days
        
        # Recompute grouped with t_days included
        grouped_with_time = df.groupby(["cell_x", "cell_y"])
        
        def compute_trends(group):
            t = group["t_days"].values.astype(float)
            results = {}
            if len(t) < 6 or np.std(t) == 0:
                for c in avail_trends:
                    results[f"{c}_trend"] = np.nan
                return pd.Series(results)
            
            for c in avail_trends:
                vals = group[c].values.astype(float)
                valid = ~np.isnan(vals)
                if valid.sum() < 6:
                    results[f"{c}_trend"] = np.nan
                else:
                    # Linear regression slope (units per day)
                    slope = np.polyfit(t[valid], vals[valid], 1)[0]
                    # Convert to per-year
                    results[f"{c}_trend"] = slope * 365.25
            return pd.Series(results)
        
        trends = grouped_with_time.apply(compute_trends).reset_index()
        profiles = profiles.merge(trends, on=["cell_x", "cell_y"], how="left")
    
    # ── Flood frequency ──
    print("    Computing flood frequency and hydroperiod ...")
    if "water_mask" in df.columns:
        flood_freq = grouped["water_mask"].mean().reset_index()
        flood_freq.columns = ["cell_x", "cell_y", "flood_frequency"]
        profiles = profiles.merge(flood_freq, on=["cell_x", "cell_y"], how="left")
    
    # ── Hydroperiod variability ──
    if "NDWI" in df.columns:
        hydro_var = grouped["NDWI"].std().reset_index()
        hydro_var.columns = ["cell_x", "cell_y", "hydroperiod_variability"]
        profiles = profiles.merge(hydro_var, on=["cell_x", "cell_y"], how="left")
    
    # ── Seasonal amplitude (growing season mean - dormant season mean) ──
    if "bird_season" in df.columns:
        for idx in ["NDVI", "NDWI", "EVI"]:
            if idx not in df.columns:
                continue
            grow = df[df["bird_season"].isin(["spring_migration", "breeding"])]
            dorm = df[df["bird_season"].isin(["fall_migration", "winter"])]
            if grow.empty or dorm.empty:
                continue
            g_mean = grow.groupby(["cell_x", "cell_y"])[idx].mean().reset_index()
            d_mean = dorm.groupby(["cell_x", "cell_y"])[idx].mean().reset_index()
            g_mean.columns = ["cell_x", "cell_y", f"{idx}_grow"]
            d_mean.columns = ["cell_x", "cell_y", f"{idx}_dorm"]
            merged = g_mean.merge(d_mean, on=["cell_x", "cell_y"], how="outer")
            merged[f"{idx}_seasonal_amplitude"] = merged[f"{idx}_grow"] - merged[f"{idx}_dorm"]
            profiles = profiles.merge(
                merged[["cell_x", "cell_y", f"{idx}_seasonal_amplitude"]],
                on=["cell_x", "cell_y"], how="left"
            )
    
    # ── Bird diversity metrics (aggregated per cell across all time) ──
    print("    Computing bird diversity summaries ...")
    if avail_birds:
        bird_stats = grouped[avail_birds].agg(["mean", "max", "sum"])
        bird_stats.columns = ["_".join(col) for col in bird_stats.columns]
        profiles = profiles.merge(bird_stats.reset_index(), on=["cell_x", "cell_y"], how="left")
        
        # Flag: has any bird data
        profiles["has_bird_data"] = (profiles.get("n_observations_sum", 0) > 0).astype(int)
    
    # ── Order-level counts if available ──
    order_cols = [c for c in df.columns if c.startswith("n_") and c not in BIRD_COLS]
    if order_cols:
        order_sums = grouped[order_cols].sum()
        order_sums.columns = [f"{c}_total" for c in order_sums.columns]
        profiles = profiles.merge(order_sums.reset_index(), on=["cell_x", "cell_y"], how="left")
    
    # ── Clean up ──
    # Drop any columns that are all NaN
    profiles = profiles.dropna(axis=1, how="all")
    
    # Fill remaining NaN with column median (for cells missing seasonal data etc.)
    feature_cols = [c for c in profiles.columns if c not in ("cell_x", "cell_y", "has_bird_data")]
    for c in feature_cols:
        if profiles[c].isna().any():
            profiles[c] = profiles[c].fillna(profiles[c].median())
    
    print(f"\n    Cell profiles: {len(profiles):,} cells × {len(profiles.columns)} features")
    
    return profiles


# ===========================================================================
# STEP 3: STANDARDIZE + PCA
# ===========================================================================

def prepare_features(profiles: pd.DataFrame):
    """
    Standardize features and reduce dimensionality with PCA.
    Returns scaled features, PCA-transformed features, and fitted objects.
    """
    print("\n" + "=" * 60)
    print("  Feature standardization & dimensionality reduction")
    print("=" * 60)
    
    # Separate metadata from features
    meta_cols = ["cell_x", "cell_y", "has_bird_data"]
    bird_summary_cols = [c for c in profiles.columns if any(
        c.startswith(b) for b in ["n_observations", "n_species", "n_individuals",
                                   "shannon_diversity", "has_bird_data"]
    )]
    order_total_cols = [c for c in profiles.columns if c.endswith("_total")]
    
    # Features for clustering = satellite-only (no bird data, to keep it unsupervised)
    exclude = set(meta_cols + bird_summary_cols + order_total_cols)
    feature_cols = [c for c in profiles.columns if c not in exclude]
    
    print(f"    Features for clustering: {len(feature_cols)}")
    print(f"    Bird metrics (held out for scoring): {len(bird_summary_cols)}")
    
    X = profiles[feature_cols].values.astype(np.float64)
    
    # Handle any remaining inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"    Scaled: {X_scaled.shape}")
    
    # PCA
    pca = PCA(n_components=PCA_VARIANCE, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    n_components = X_pca.shape[1]
    explained = pca.explained_variance_ratio_.sum()
    
    print(f"    PCA: {len(feature_cols)} features → {n_components} components "
          f"({explained:.1%} variance retained)")
    
    return X_scaled, X_pca, scaler, pca, feature_cols


# ===========================================================================
# STEP 4: CLUSTERING
# ===========================================================================

def find_optimal_clusters(X_pca: np.ndarray) -> dict:
    """
    Fit GMM with varying cluster counts. Select optimal k via BIC + silhouette.
    Returns dict with best model, scores, and all fitted models.
    """
    print("\n" + "=" * 60)
    print("  Finding optimal number of habitat archetypes")
    print("=" * 60)
    
    results = []
    models = {}
    
    k_range = range(MIN_CLUSTERS, MAX_CLUSTERS + 1)
    
    for k in k_range:
        print(f"    k={k:2d} ...", end="", flush=True)
        
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=3,
            max_iter=300,
            random_state=42,
        )
        gmm.fit(X_pca)
        
        labels = gmm.predict(X_pca)
        bic = gmm.bic(X_pca)
        aic = gmm.aic(X_pca)
        
        # Silhouette (subsample for speed if >10k points)
        if len(X_pca) > 10000:
            idx = np.random.RandomState(42).choice(len(X_pca), 10000, replace=False)
            sil = silhouette_score(X_pca[idx], labels[idx])
        else:
            sil = silhouette_score(X_pca, labels)
        
        ch = calinski_harabasz_score(X_pca, labels)
        
        results.append({
            "k": k, "bic": bic, "aic": aic,
            "silhouette": sil, "calinski_harabasz": ch,
        })
        models[k] = gmm
        
        print(f"  BIC={bic:,.0f}  silhouette={sil:.3f}  CH={ch:,.0f}")
    
    results_df = pd.DataFrame(results)
    
    # Select best k: lowest BIC with reasonable silhouette
    # Normalize both metrics and combine
    bic_norm = (results_df["bic"] - results_df["bic"].min()) / (results_df["bic"].max() - results_df["bic"].min() + 1e-10)
    sil_norm = (results_df["silhouette"] - results_df["silhouette"].min()) / (results_df["silhouette"].max() - results_df["silhouette"].min() + 1e-10)
    
    # Lower BIC is better, higher silhouette is better
    results_df["combined_score"] = (1 - bic_norm) * 0.5 + sil_norm * 0.5
    
    best_idx = results_df["combined_score"].idxmax()
    best_k = results_df.loc[best_idx, "k"]
    
    print(f"\n    ★ Best k = {best_k} (BIC={results_df.loc[best_idx, 'bic']:,.0f}, "
          f"silhouette={results_df.loc[best_idx, 'silhouette']:.3f})")
    
    return {
        "best_k": int(best_k),
        "best_model": models[best_k],
        "results": results_df,
        "all_models": models,
    }


# ===========================================================================
# STEP 5: SCORING
# ===========================================================================

def score_cells(profiles: pd.DataFrame, labels: np.ndarray,
                probabilities: np.ndarray, cluster_info: dict) -> pd.DataFrame:
    """
    Score each cell on:
      - habitat_suitability: how bird-friendly is this cell's cluster?
      - condition_trajectory: is this cell improving, stable, or degrading?
      - conservation_priority: composite score (0-100)
    """
    print("\n" + "=" * 60)
    print("  Scoring cells")
    print("=" * 60)
    
    results = profiles[["cell_x", "cell_y"]].copy()
    results["cluster"] = labels
    results["cluster_confidence"] = probabilities.max(axis=1)
    
    # ── Habitat suitability (from bird data per cluster) ──
    print("    Computing habitat suitability scores ...")
    
    # For each cluster, compute median bird diversity from cells that have data
    has_birds = profiles["has_bird_data"] == 1 if "has_bird_data" in profiles.columns else pd.Series(False, index=profiles.index)
    
    cluster_bird_scores = {}
    for k in range(labels.max() + 1):
        mask = (labels == k) & has_birds
        if mask.sum() == 0:
            cluster_bird_scores[k] = {
                "median_species": np.nan,
                "median_diversity": np.nan,
                "median_observations": np.nan,
                "cells_with_data": 0,
                "total_cells": (labels == k).sum(),
            }
        else:
            cluster_bird_scores[k] = {
                "median_species": profiles.loc[mask, "n_species_mean"].median() if "n_species_mean" in profiles.columns else np.nan,
                "median_diversity": profiles.loc[mask, "shannon_diversity_mean"].median() if "shannon_diversity_mean" in profiles.columns else np.nan,
                "median_observations": profiles.loc[mask, "n_observations_mean"].median() if "n_observations_mean" in profiles.columns else np.nan,
                "cells_with_data": int(mask.sum()),
                "total_cells": int((labels == k).sum()),
            }
    
    # Rank clusters by bird diversity → habitat suitability score
    diversity_vals = {k: v["median_diversity"] for k, v in cluster_bird_scores.items()}
    # Handle NaN: clusters with no bird data get the global median
    valid_divs = [v for v in diversity_vals.values() if not np.isnan(v)]
    global_median_div = np.median(valid_divs) if valid_divs else 0
    
    for k in diversity_vals:
        if np.isnan(diversity_vals[k]):
            diversity_vals[k] = global_median_div
    
    # Normalize to 0-100
    div_min = min(diversity_vals.values())
    div_max = max(diversity_vals.values())
    div_range = div_max - div_min if div_max > div_min else 1
    
    suitability_by_cluster = {
        k: ((v - div_min) / div_range) * 100 for k, v in diversity_vals.items()
    }
    
    results["habitat_suitability"] = results["cluster"].map(suitability_by_cluster)
    
    # ── Condition trajectory ──
    print("    Computing condition trajectories ...")
    
    trend_cols = [c for c in profiles.columns if c.endswith("_trend")]
    if trend_cols:
        # Composite trend: weighted average of key index trends
        # Positive NDVI/EVI trend = improving; negative = degrading
        # Positive water_mask trend could be either (depends on context)
        weights = {}
        for c in trend_cols:
            if "NDVI" in c: weights[c] = 0.30
            elif "EVI" in c: weights[c] = 0.20
            elif "NDMI" in c: weights[c] = 0.15
            elif "wetland_moisture" in c: weights[c] = 0.15
            elif "NDWI" in c: weights[c] = 0.10
            elif "water_mask" in c: weights[c] = 0.10
            else: weights[c] = 0.05
        
        # Normalize weights
        w_sum = sum(weights.values())
        weights = {k: v / w_sum for k, v in weights.items()}
        
        # Compute weighted composite trend
        composite_trend = np.zeros(len(profiles))
        for c, w in weights.items():
            if c in profiles.columns:
                vals = profiles[c].fillna(0).values
                composite_trend += vals * w
        
        results["trend_composite"] = composite_trend
        
        # Classify trajectory
        # Use percentiles to define thresholds
        p25 = np.percentile(composite_trend, 25)
        p75 = np.percentile(composite_trend, 75)
        
        conditions = [
            composite_trend > p75,
            composite_trend < p25,
        ]
        choices = ["improving", "degrading"]
        results["trajectory"] = np.select(conditions, choices, default="stable")
        
        # Normalize trend to 0-100 (50 = stable)
        t_min, t_max = composite_trend.min(), composite_trend.max()
        t_range = t_max - t_min if t_max > t_min else 1
        results["trend_score"] = ((composite_trend - t_min) / t_range) * 100
    else:
        results["trend_composite"] = 0
        results["trajectory"] = "unknown"
        results["trend_score"] = 50
    
    # ── Conservation priority (composite) ──
    print("    Computing conservation priority scores ...")
    
    # Priority favors:
    #   - High habitat suitability (good habitat worth protecting)
    #   - Degrading trajectory (urgent intervention needed)
    #   - Moderate suitability + degrading (habitat at risk of tipping)
    
    suit = results["habitat_suitability"].values / 100  # 0-1
    trend = results["trend_score"].values / 100          # 0-1, higher = improving
    urgency = 1 - trend  # invert: degrading = high urgency
    
    # Composite: weight suitability and urgency
    # High suitability + high urgency = highest priority (good habitat at risk)
    # Low suitability + high urgency = moderate (already degraded, may be too late)
    # High suitability + low urgency = moderate (already protected/stable)
    # Low suitability + low urgency = low priority
    
    priority = (
        0.40 * suit +                    # current ecological value
        0.30 * urgency +                 # degradation urgency
        0.30 * (suit * urgency)          # interaction: good habitat at risk
    )
    
    # Normalize to 0-100
    p_min, p_max = priority.min(), priority.max()
    p_range = p_max - p_min if p_max > p_min else 1
    results["conservation_priority"] = ((priority - p_min) / p_range) * 100
    
    # Add individual trend slopes for transparency
    for c in trend_cols:
        if c in profiles.columns:
            results[c] = profiles[c].values
    
    # ── Summary ──
    print(f"\n    Results summary:")
    print(f"    Clusters: {labels.max() + 1}")
    for traj in ["improving", "stable", "degrading"]:
        n = (results["trajectory"] == traj).sum()
        pct = 100 * n / len(results)
        print(f"      {traj:12s}: {n:,} cells ({pct:.1f}%)")
    
    print(f"    Conservation priority: mean={results['conservation_priority'].mean():.1f}, "
          f"max={results['conservation_priority'].max():.1f}")
    
    return results, cluster_bird_scores


# ===========================================================================
# STEP 6: CLUSTER PROFILES
# ===========================================================================

def build_cluster_profiles(profiles: pd.DataFrame, labels: np.ndarray,
                           feature_cols: list, cluster_bird_scores: dict) -> pd.DataFrame:
    """Build interpretable profile for each cluster."""
    print("\n  Building cluster archetype profiles ...")
    
    profiles = profiles.copy()
    profiles["cluster"] = labels
    
    # Mean feature values per cluster
    cluster_means = profiles.groupby("cluster")[feature_cols].mean()
    
    # Add bird data summary
    for k, bird_info in cluster_bird_scores.items():
        if k in cluster_means.index:
            for metric, val in bird_info.items():
                cluster_means.loc[k, f"bird_{metric}"] = val
    
    # Add cluster size
    sizes = profiles.groupby("cluster").size()
    cluster_means["n_cells"] = sizes
    
    # Generate descriptive labels based on dominant features
    labels_desc = []
    for k in cluster_means.index:
        row = cluster_means.loc[k]
        parts = []
        
        # Water character
        if "flood_frequency" in row and row.get("flood_frequency", 0) > 0.7:
            parts.append("Persistently flooded")
        elif "flood_frequency" in row and row.get("flood_frequency", 0) > 0.4:
            parts.append("Seasonally flooded")
        elif "flood_frequency" in row and row.get("flood_frequency", 0) > 0.15:
            parts.append("Intermittently wet")
        else:
            parts.append("Dry upland")
        
        # Vegetation character
        ndvi_mean = row.get("NDVI_mean", 0)
        if ndvi_mean > 0.5:
            parts.append("dense vegetation")
        elif ndvi_mean > 0.25:
            parts.append("moderate vegetation")
        elif ndvi_mean > 0.05:
            parts.append("sparse vegetation")
        else:
            parts.append("bare/water")
        
        # Trend
        ndvi_trend = row.get("NDVI_trend", 0) if "NDVI_trend" in row else 0
        if ndvi_trend > 0.01:
            parts.append("(greening)")
        elif ndvi_trend < -0.01:
            parts.append("(browning)")
        
        labels_desc.append(" — ".join(parts[:2]) + (" " + parts[2] if len(parts) > 2 else ""))
    
    cluster_means["archetype_label"] = labels_desc
    
    return cluster_means


# ===========================================================================
# STEP 7: DIAGNOSTICS
# ===========================================================================

def build_diagnostics(cluster_results: dict, profiles: pd.DataFrame,
                      results: pd.DataFrame, cluster_bird_scores: dict,
                      pca, scaler, feature_cols: list) -> dict:
    """Compile model diagnostics and validation metrics."""
    
    best_k = cluster_results["best_k"]
    search_df = cluster_results["results"]
    
    # Cross-validate: do high-suitability clusters actually have more birds?
    bird_validation = {}
    if "has_bird_data" in profiles.columns:
        has_birds = profiles["has_bird_data"] == 1
        if has_birds.sum() > 0:
            # Correlation between cluster suitability rank and actual bird diversity
            cluster_suit = results.groupby("cluster")["habitat_suitability"].first()
            cluster_div = profiles[has_birds].groupby(results.loc[has_birds.values, "cluster"])["shannon_diversity_mean"].median() if "shannon_diversity_mean" in profiles.columns else pd.Series()
            
            if len(cluster_div) > 2:
                common = cluster_suit.index.intersection(cluster_div.index)
                if len(common) > 2:
                    corr = np.corrcoef(
                        cluster_suit.loc[common].values,
                        cluster_div.loc[common].values
                    )[0, 1]
                    bird_validation["suitability_diversity_correlation"] = float(corr)
                    print(f"\n    Validation: suitability ↔ bird diversity correlation = {corr:.3f}")
    
    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "input_rows": int(len(profiles)),
        "n_cells": int(len(profiles)),
        "n_features_original": int(len(feature_cols)),
        "n_pca_components": int(pca.n_components_),
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "best_k": best_k,
        "cluster_search": search_df.to_dict(orient="records"),
        "cluster_bird_scores": {
            str(k): {kk: float(vv) if not (isinstance(vv, float) and np.isnan(vv)) else None
                      for kk, vv in v.items()}
            for k, v in cluster_bird_scores.items()
        },
        "trajectory_distribution": results["trajectory"].value_counts().to_dict(),
        "conservation_priority_stats": {
            "mean": float(results["conservation_priority"].mean()),
            "std": float(results["conservation_priority"].std()),
            "min": float(results["conservation_priority"].min()),
            "max": float(results["conservation_priority"].max()),
            "p25": float(results["conservation_priority"].quantile(0.25)),
            "p75": float(results["conservation_priority"].quantile(0.75)),
        },
        "bird_validation": bird_validation,
    }
    
    return diagnostics


# ===========================================================================
# MAIN
# ===========================================================================

def score_new_data(model_path: str, input_path: str, output_path: str):
    """
    Load a saved model and score new data without retraining.
    Input can be a model_input.csv (cell × dekad) or a cell_profiles.csv.
    """
    import pickle
    
    print("=" * 72)
    print("  Scoring new data with saved model")
    print("=" * 72)
    
    # Load model
    print(f"\n  Loading model: {model_path}")
    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)
    
    gmm = artifacts["gmm"]
    pca = artifacts["pca"]
    scaler = artifacts["scaler"]
    feature_cols = artifacts["feature_cols"]
    suitability_map = artifacts["suitability_by_cluster"]
    trend_weights = artifacts["trend_weights"]
    trend_pcts = artifacts["trend_percentiles"]
    
    print(f"    Clusters: {artifacts['n_clusters']}")
    print(f"    Features: {len(feature_cols)}")
    print(f"    Trained on: {artifacts['trained_on']}")
    
    # Load new data
    print(f"\n  Loading data: {input_path}")
    df = pd.read_csv(input_path)
    
    # Check if this is raw cell × dekad data or pre-built profiles
    if "dekad" in df.columns:
        print("    Detected cell × dekad format — building profiles ...")
        df["dekad"] = pd.to_datetime(df["dekad"])
        profiles = build_cell_profiles(df)
    else:
        print("    Detected cell profile format.")
        profiles = df
    
    # Prepare features
    missing = [c for c in feature_cols if c not in profiles.columns]
    if missing:
        print(f"    WARNING: {len(missing)} features missing, filling with 0")
        for c in missing:
            profiles[c] = 0.0
    
    X = profiles[feature_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    
    # Predict
    labels = gmm.predict(X_pca)
    probabilities = gmm.predict_proba(X_pca)
    
    # Build results
    results = profiles[["cell_x", "cell_y"]].copy()
    results["cluster"] = labels
    results["cluster_confidence"] = probabilities.max(axis=1)
    results["habitat_suitability"] = results["cluster"].map(suitability_map)
    
    # Trend scoring
    trend_cols = [c for c in profiles.columns if c.endswith("_trend")]
    if trend_cols:
        composite = np.zeros(len(profiles))
        w_sum = sum(trend_weights.get(c, 0.05) for c in trend_cols)
        for c in trend_cols:
            w = trend_weights.get(c, 0.05) / w_sum
            composite += profiles[c].fillna(0).values * w
        
        results["trend_composite"] = composite
        results["trajectory"] = np.select(
            [composite > trend_pcts["p75"], composite < trend_pcts["p25"]],
            ["improving", "degrading"], default="stable"
        )
        t_min, t_max = composite.min(), composite.max()
        t_range = t_max - t_min if t_max > t_min else 1
        results["trend_score"] = ((composite - t_min) / t_range) * 100
    else:
        results["trend_composite"] = 0
        results["trajectory"] = "unknown"
        results["trend_score"] = 50
    
    # Conservation priority
    suit = results["habitat_suitability"].values / 100
    trend = results["trend_score"].values / 100
    urgency = 1 - trend
    priority = 0.40 * suit + 0.30 * urgency + 0.30 * (suit * urgency)
    p_min, p_max = priority.min(), priority.max()
    p_range = p_max - p_min if p_max > p_min else 1
    results["conservation_priority"] = ((priority - p_min) / p_range) * 100
    
    # Save
    results.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path} ({len(results):,} cells)")
    print(f"  Clusters: {pd.Series(labels).value_counts().to_dict()}")
    print("=" * 72)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train habitat archetype model and score conservation priority"
    )
    parser.add_argument("--input", default=INPUT_CSV,
                        help=f"Input CSV path (default: {INPUT_CSV})")
    parser.add_argument("--output", default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help=f"Number of cells to subsample (default: {DEFAULT_SAMPLE})")
    parser.add_argument("--full", action="store_true",
                        help="Use all cells (no subsampling)")
    parser.add_argument("--score", type=str, default=None,
                        help="Score new data using saved model (path to model .pkl)")
    parser.add_argument("--score-output", type=str, default="./scored_results.csv",
                        help="Output path when using --score mode")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Score-only mode
    if args.score:
        score_new_data(args.score, args.input, args.score_output)
        return
    
    sample_n = None if args.full else args.sample
    
    print("=" * 72)
    print("  Habitat Archetype Clustering & Conservation Scoring")
    print("=" * 72)
    print(f"  Input       : {args.input}")
    print(f"  Output dir  : {args.output}")
    print(f"  Subsampling : {'OFF (full dataset)' if args.full else f'{sample_n:,} cells'}")
    print(f"  PCA variance: {PCA_VARIANCE:.0%}")
    print(f"  Cluster range: {MIN_CLUSTERS}–{MAX_CLUSTERS}")
    print("=" * 72)
    
    # Step 1: Load
    df = load_input(args.input, sample_n=sample_n)
    
    # Step 2: Cell profiles
    profiles = build_cell_profiles(df)
    
    # Save intermediate profiles
    profiles_path = os.path.join(args.output, "cell_profiles.csv")
    profiles.to_csv(profiles_path, index=False)
    print(f"\n  Saved cell profiles: {profiles_path}")
    
    # Free memory
    del df
    
    # Step 3: Standardize + PCA
    X_scaled, X_pca, scaler, pca, feature_cols = prepare_features(profiles)
    
    # Step 4: Clustering
    cluster_results = find_optimal_clusters(X_pca)
    best_model = cluster_results["best_model"]
    labels = best_model.predict(X_pca)
    probabilities = best_model.predict_proba(X_pca)
    
    # Step 5: Scoring
    results, cluster_bird_scores = score_cells(profiles, labels, probabilities, cluster_results)
    
    # Step 6: Cluster profiles
    cluster_profiles = build_cluster_profiles(
        profiles, labels, feature_cols, cluster_bird_scores
    )
    
    # Step 7: Diagnostics
    diagnostics = build_diagnostics(
        cluster_results, profiles, results, cluster_bird_scores,
        pca, scaler, feature_cols
    )
    
    # ── Save outputs ──
    print("\n" + "=" * 60)
    print("  Saving outputs")
    print("=" * 60)
    
    # Save trained model artifacts (so you can score new data without retraining)
    import pickle
    
    model_artifacts = {
        "gmm": best_model,
        "pca": pca,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "cluster_bird_scores": cluster_bird_scores,
        "suitability_by_cluster": {
            k: float(results.loc[results["cluster"] == k, "habitat_suitability"].iloc[0])
            for k in range(labels.max() + 1)
        },
        "trend_weights": {
            "NDVI_trend": 0.30, "EVI_trend": 0.20, "NDMI_trend": 0.15,
            "wetland_moisture_index_trend": 0.15, "NDWI_trend": 0.10,
            "water_mask_trend": 0.10,
        },
        "trend_percentiles": {
            "p25": float(np.percentile(results["trend_composite"], 25)),
            "p75": float(np.percentile(results["trend_composite"], 75)),
        },
        "n_clusters": int(cluster_results["best_k"]),
        "trained_on": datetime.now().isoformat(),
        "n_cells_trained": int(len(profiles)),
    }
    
    model_path = os.path.join(args.output, "habitat_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_artifacts, f)
    print(f"    {model_path} (trained model — reusable for scoring new data)")
    
    results_path = os.path.join(args.output, "habitat_results.csv")
    results.to_csv(results_path, index=False)
    print(f"    {results_path} ({len(results):,} cells)")
    
    cluster_path = os.path.join(args.output, "cluster_profiles.csv")
    cluster_profiles.to_csv(cluster_path)
    print(f"    {cluster_path} ({len(cluster_profiles)} clusters)")
    
    diag_path = os.path.join(args.output, "model_diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"    {diag_path}")
    
    # ── Final summary ──
    print("\n" + "=" * 72)
    print("  MODEL TRAINING COMPLETE")
    print("=" * 72)
    print(f"  Habitat archetypes discovered: {cluster_results['best_k']}")
    print(f"  Cells scored: {len(results):,}")
    
    print(f"\n  Cluster summary:")
    for k in sorted(cluster_profiles.index):
        row = cluster_profiles.loc[k]
        n = int(row["n_cells"])
        label = row.get("archetype_label", f"Cluster {k}")
        bird_cells = cluster_bird_scores[k]["cells_with_data"]
        med_sp = cluster_bird_scores[k]["median_species"]
        sp_str = f"{med_sp:.1f}" if not np.isnan(med_sp) else "no data"
        print(f"    [{k}] {label}")
        print(f"        {n:,} cells | {bird_cells} with bird data | median species: {sp_str}")
    
    print(f"\n  Trajectory breakdown:")
    for traj in ["improving", "stable", "degrading"]:
        n = (results["trajectory"] == traj).sum()
        pct = 100 * n / len(results)
        mean_pri = results.loc[results["trajectory"] == traj, "conservation_priority"].mean()
        print(f"    {traj:12s}: {n:,} cells ({pct:.1f}%) — avg priority: {mean_pri:.1f}")
    
    print(f"\n  Top 10 conservation priority cells:")
    top10 = results.nlargest(10, "conservation_priority")
    for _, row in top10.iterrows():
        print(f"    ({row['cell_x']:.4f}, {row['cell_y']:.4f}) "
              f"priority={row['conservation_priority']:.1f} "
              f"suitability={row['habitat_suitability']:.1f} "
              f"trajectory={row['trajectory']}")
    
    print(f"\n  Output directory: {args.output}/")
    print("=" * 72)


if __name__ == "__main__":
    main()