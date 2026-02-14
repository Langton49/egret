"""
=============================================================================
  Extract Bi-weekly Time Series from NetCDF → CSV
  For avian habitat suitability modeling (Mississippi Delta)
=============================================================================

Takes the bi-weekly composite NetCDFs (from fetch_biweekly_timeseries.py)
and produces a structured CSV with one row per (grid_cell, time_step).

This gives you the full temporal signal for each location — seasonal flooding,
vegetation phenology, moisture cycles — everything a model needs to assess
habitat quality for migratory and resident birds.

Usage:
    python extract_timeseries_to_csv.py
    python extract_timeseries_to_csv.py --input ./satellite_timeseries --cell-size 1.0
    python extract_timeseries_to_csv.py --pivot   # one row per cell, columns per date
"""

import os
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import xarray as xr
except ImportError:
    sys.exit("ERROR: xarray + netcdf4 required. Install: pip install xarray netcdf4")

# ===========================================================================
# CONFIGURATION
# ===========================================================================

DEFAULT_INPUT_DIR = "./satellite_timeseries"
DEFAULT_OUTPUT_DIR = "./habitat_features"
DEFAULT_CELL_SIZE_KM = 1.0

# Band names as they may appear in the NetCDF
# (handles both UDF-computed and raw band naming)
BAND_ALIASES = {
    "B02": "blue",   "blue": "blue",
    "B03": "green",  "green": "green",
    "B04": "red",    "red": "red",
    "B08": "nir",    "nir": "nir",
    "B8A": "nir_narrow", "nir_narrow": "nir_narrow",
    "B11": "swir1",  "swir1": "swir1",
    "B12": "swir2",  "swir2": "swir2",
}

INDEX_VARS = [
    "NDVI", "NDWI", "MNDWI", "NDMI", "EVI", "SAVI",
    "LSWI", "WRI", "wetland_moisture_index", "water_mask",
    "tc_wetness", "GCVI",
]

REFLECTANCE_VARS = ["blue", "green", "red", "nir", "nir_narrow", "swir1", "swir2"]

os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# CORE EXTRACTION
# ===========================================================================

def discover_nc_files(input_dir: str) -> list:
    """Find time-series NetCDFs."""
    files = sorted(glob.glob(os.path.join(input_dir, "*.nc")))
    # Prefer _indices files if both exist
    indices_files = [f for f in files if "_indices" in f]
    base_files = [f for f in files if "_indices" not in f]

    if indices_files:
        print(f"Found {len(indices_files)} index-enriched NetCDF(s)")
        return indices_files

    print(f"Found {len(base_files)} NetCDF(s) (raw bands — indices will be computed)")
    return base_files


def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Normalize coordinate and variable names."""
    # Rename coordinates
    rename = {}
    for candidate, target in [
        ("longitude", "x"), ("lon", "x"),
        ("latitude", "y"), ("lat", "y"),
        ("t", "time"),
    ]:
        if candidate in ds.coords and target not in ds.coords:
            rename[candidate] = target
    if rename:
        ds = ds.rename(rename)

    # Rename band variables to friendly names
    var_rename = {}
    for var in list(ds.data_vars):
        if var in BAND_ALIASES and BAND_ALIASES[var] != var:
            var_rename[var] = BAND_ALIASES[var]
    if var_rename:
        ds = ds.rename(var_rename)

    return ds


def compute_indices_if_needed(ds: xr.Dataset) -> xr.Dataset:
    """Compute spectral indices if only raw bands are present."""
    has_indices = any(v in ds.data_vars for v in INDEX_VARS)
    if has_indices:
        print("  Indices already present in dataset.")
        return ds

    print("  Computing indices from raw bands ...")

    available = list(ds.data_vars)
    needed = {"blue", "green", "red", "nir", "swir1", "swir2"}
    if not needed.issubset(set(available)):
        missing = needed - set(available)
        print(f"  WARNING: Missing bands {missing}, skipping index computation")
        return ds

    scale = 10000.0
    blue = ds["blue"].astype("float32") / scale
    green = ds["green"].astype("float32") / scale
    red = ds["red"].astype("float32") / scale
    nir = ds["nir"].astype("float32") / scale
    swir1 = ds["swir1"].astype("float32") / scale
    swir2 = ds["swir2"].astype("float32") / scale
    nir_n = ds["nir_narrow"].astype("float32") / scale if "nir_narrow" in available else nir

    def safe_nd(a, b):
        d = a + b
        return xr.where(d != 0, (a - b) / d, np.nan)

    ds["NDVI"] = safe_nd(nir, red)
    ds["NDWI"] = safe_nd(green, nir)
    ds["MNDWI"] = safe_nd(green, swir1)
    ds["NDMI"] = safe_nd(nir, swir1)

    ds["EVI"] = xr.where(
        (nir + 6 * red - 7.5 * blue + 1) != 0,
        2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1), np.nan
    )
    ds["SAVI"] = xr.where(
        (nir + red + 0.5) != 0,
        1.5 * (nir - red) / (nir + red + 0.5), np.nan
    )
    ds["LSWI"] = safe_nd(nir, swir1)
    ds["WRI"] = xr.where(
        (nir + swir1) != 0, (green + red) / (nir + swir1), np.nan
    )

    ds["wetland_moisture_index"] = (ds["NDWI"] + ds["NDMI"] + ds["MNDWI"]) / 3.0
    ds["water_mask"] = xr.where((ds["MNDWI"] > 0) & (ds["NDVI"] < 0.2), 1.0, 0.0)

    ds["tc_wetness"] = (
        0.1509 * blue + 0.1973 * green + 0.3279 * red
        + 0.3406 * nir - 0.7112 * swir1 - 0.4572 * swir2
    )
    ds["GCVI"] = xr.where(green != 0, (nir / green) - 1, np.nan)

    print(f"  Computed {len(INDEX_VARS)} indices.")
    return ds


def aggregate_spatial(ds: xr.Dataset, cell_size_km: float) -> xr.Dataset:
    """Coarsen spatial resolution by averaging pixels into grid cells."""
    if cell_size_km <= 0:
        return ds

    # Detect CRS from coordinate ranges
    x_range = float(ds.x.max() - ds.x.min())
    if x_range > 1000:
        # Projected CRS (meters)
        x_res = abs(float(ds.x[1] - ds.x[0])) if len(ds.x) > 1 else 10
        factor = max(1, int(round(cell_size_km * 1000 / x_res)))
    else:
        # Geographic CRS (degrees)
        x_res = abs(float(ds.x[1] - ds.x[0])) if len(ds.x) > 1 else 0.0001
        approx_m_per_deg = 111000
        factor = max(1, int(round(cell_size_km * 1000 / (x_res * approx_m_per_deg))))

    if factor <= 1:
        print(f"  Cell size {cell_size_km} km ≈ native resolution, no coarsening.")
        return ds

    print(f"  Coarsening by factor {factor}× ({cell_size_km} km cells) ...")

    # Only coarsen spatial dims, preserve time
    coarsen_dims = {"x": factor, "y": factor}

    # Drop non-numeric vars before coarsening
    numeric_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
    ds_num = ds[numeric_vars]

    ds_coarse = ds_num.coarsen(coarsen_dims, boundary="trim").mean()
    print(f"  Grid: {len(ds.x)}×{len(ds.y)} → {len(ds_coarse.x)}×{len(ds_coarse.y)}")

    return ds_coarse


def dataset_to_long_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """
    Convert xarray Dataset with (time, x, y) dims to a long-format DataFrame.
    One row per (x, y, time) combination.
    """
    print("  Converting to DataFrame (this may take a moment for large datasets) ...")

    # Select only the variables we want
    keep_vars = [v for v in ds.data_vars
                 if np.issubdtype(ds[v].dtype, np.number)
                 and v not in ("crs", "spatial_ref")]
    ds = ds[keep_vars]

    # Stack spatial dims
    stacked = ds.stack(pixel=("x", "y"))

    # Convert to dataframe
    df = stacked.to_dataframe()

    # Extract x, y from multi-index
    if isinstance(df.index, pd.MultiIndex):
        idx_names = df.index.names
        if "x" in idx_names:
            df["x"] = df.index.get_level_values("x")
        if "y" in idx_names:
            df["y"] = df.index.get_level_values("y")
        if "time" in idx_names:
            df["time"] = df.index.get_level_values("time")
        df = df.reset_index(drop=True)
    else:
        df = df.reset_index()

    # Drop all-NaN rows
    feature_cols = [c for c in df.columns if c not in ("x", "y", "time", "pixel")]
    df = df.dropna(subset=feature_cols, how="all")

    return df


def add_temporal_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-derived features useful for habitat modeling."""
    if "time" not in df.columns:
        return df

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day_of_year"] = df["time"].dt.dayofyear

    # Seasonal labels relevant to bird ecology
    # Spring migration: Mar-May, Breeding: Jun-Jul,
    # Fall migration: Aug-Oct, Winter: Nov-Feb
    conditions = [
        df["month"].isin([3, 4, 5]),
        df["month"].isin([6, 7]),
        df["month"].isin([8, 9, 10]),
        df["month"].isin([11, 12, 1, 2]),
    ]
    labels = ["spring_migration", "breeding", "fall_migration", "winter"]
    df["bird_season"] = np.select(conditions, labels, default="unknown")

    # Cyclical encoding of day-of-year (captures annual periodicity)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    return df


def compute_per_cell_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-cell summary statistics across the entire time series.
    Produces one row per (x, y) with temporal aggregates.
    """
    if "time" not in df.columns:
        return df

    print("  Computing per-cell temporal summaries ...")

    index_cols = [c for c in INDEX_VARS + REFLECTANCE_VARS if c in df.columns]
    if not index_cols:
        index_cols = [c for c in df.columns
                      if c not in ("x", "y", "time", "year", "month",
                                   "day_of_year", "bird_season", "doy_sin", "doy_cos")]

    agg_funcs = ["mean", "std", "min", "max"]

    # Overall stats
    grouped = df.groupby(["x", "y"])[index_cols].agg(agg_funcs)
    grouped.columns = ["_".join(col) for col in grouped.columns]
    grouped = grouped.reset_index()

    # Seasonal means per bird season
    for season in ["spring_migration", "breeding", "fall_migration", "winter"]:
        season_data = df[df["bird_season"] == season]
        if season_data.empty:
            continue
        season_means = season_data.groupby(["x", "y"])[index_cols].mean()
        season_means.columns = [f"{c}_{season}" for c in season_means.columns]
        grouped = grouped.merge(season_means.reset_index(), on=["x", "y"], how="left")

    # Year-over-year trends (linear slope per cell)
    print("  Computing year-over-year trends ...")
    trend_cols = ["NDVI", "NDWI", "NDMI", "wetland_moisture_index", "water_mask"]
    trend_cols = [c for c in trend_cols if c in df.columns]

    if trend_cols and "day_of_year" in df.columns:
        # Use day_of_year as x-axis for trend (captures full 5-year trajectory)
        df["time_numeric"] = (df["time"] - df["time"].min()).dt.days

        def trend_slope(group):
            slopes = {}
            t = group["time_numeric"].values
            if len(t) < 3 or np.std(t) == 0:
                for col in trend_cols:
                    slopes[f"{col}_trend"] = np.nan
                return pd.Series(slopes)
            for col in trend_cols:
                vals = group[col].values
                valid = ~np.isnan(vals)
                if valid.sum() < 3:
                    slopes[f"{col}_trend"] = np.nan
                else:
                    slope = np.polyfit(t[valid], vals[valid], 1)[0]
                    slopes[f"{col}_trend"] = slope
            return pd.Series(slopes)

        trends = df.groupby(["x", "y"]).apply(trend_slope).reset_index()
        grouped = grouped.merge(trends, on=["x", "y"], how="left")

    # Flooding frequency — fraction of time steps where water_mask == 1
    if "water_mask" in df.columns:
        flood_freq = df.groupby(["x", "y"])["water_mask"].mean().reset_index()
        flood_freq.columns = ["x", "y", "flood_frequency"]
        grouped = grouped.merge(flood_freq, on=["x", "y"], how="left")

    # Hydroperiod variability — std of NDWI (indicates tidal/flood dynamics)
    if "NDWI" in df.columns:
        hydro_var = df.groupby(["x", "y"])["NDWI"].std().reset_index()
        hydro_var.columns = ["x", "y", "hydroperiod_variability"]
        grouped = grouped.merge(hydro_var, on=["x", "y"], how="left")

    print(f"  Generated {len(grouped.columns) - 2} features for {len(grouped)} cells.")
    return grouped


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract bi-weekly satellite time series to CSV for habitat modeling"
    )
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", "-o", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cell-size", type=float, default=DEFAULT_CELL_SIZE_KM,
                        help="Grid cell size in km (default: 1.0)")
    parser.add_argument("--pivot", action="store_true",
                        help="Also produce per-cell summary CSV (one row per cell)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of files to process")
    args = parser.parse_args()

    print("=" * 72)
    print("  Bi-weekly Time Series → Habitat Feature Extraction")
    print("=" * 72)

    # Discover files
    nc_files = discover_nc_files(args.input)
    if not nc_files:
        # Also check for openEO*.nc pattern from direct downloads
        nc_files = sorted(glob.glob(os.path.join(args.input, "openEO*.nc")))
        if not nc_files:
            sys.exit(f"No NetCDF files found in {args.input}/")
        print(f"Found {len(nc_files)} openEO NetCDF(s)")

    if args.max_files:
        nc_files = nc_files[:args.max_files]

    # Process each tile
    all_long_frames = []
    all_summary_frames = []

    for nc_path in nc_files:
        fname = os.path.basename(nc_path)
        print(f"\n{'='*60}")
        print(f"  Processing: {fname}")
        print(f"{'='*60}")

        ds = xr.open_dataset(nc_path)
        ds = normalize_dataset(ds)

        # Print structure
        print(f"  Dims: {dict(ds.sizes)}")
        print(f"  Vars: {list(ds.data_vars)}")
        has_time = "time" in ds.dims and ds.sizes.get("time", 0) > 1
        if has_time:
            print(f"  Time: {str(ds.time.values[0])[:10]} → "
                  f"{str(ds.time.values[-1])[:10]} ({ds.sizes['time']} steps)")

        # Compute indices if needed
        ds = compute_indices_if_needed(ds)

        # Aggregate to grid
        ds = aggregate_spatial(ds, args.cell_size)

        # Convert to long DataFrame
        df = dataset_to_long_dataframe(ds)
        ds.close()

        if df.empty:
            print("  WARNING: No data extracted, skipping.")
            continue

        print(f"  Extracted {len(df)} rows × {len(df.columns)} columns")

        # Add temporal features
        if has_time:
            df = add_temporal_context(df)

        df["source_tile"] = fname
        all_long_frames.append(df)

        # Per-cell summary
        if has_time and args.pivot:
            summary = compute_per_cell_temporal_features(df)
            summary["source_tile"] = fname
            all_summary_frames.append(summary)

    # Combine all tiles
    if not all_long_frames:
        sys.exit("No data extracted from any file.")

    print(f"\n{'='*60}")
    print("  Combining tiles ...")
    print(f"{'='*60}")

    df_all = pd.concat(all_long_frames, ignore_index=True)

    # Save long-format CSV (one row per cell per time step)
    long_csv = os.path.join(args.output_dir, "habitat_timeseries.csv")
    df_all.to_csv(long_csv, index=False)
    print(f"\n  Saved time series: {long_csv}")
    print(f"    {len(df_all)} rows × {len(df_all.columns)} columns")

    # Save summary CSV if requested
    if all_summary_frames:
        df_summary = pd.concat(all_summary_frames, ignore_index=True)
        summary_csv = os.path.join(args.output_dir, "habitat_cell_summary.csv")
        df_summary.to_csv(summary_csv, index=False)
        print(f"\n  Saved cell summary: {summary_csv}")
        print(f"    {len(df_summary)} rows × {len(df_summary.columns)} columns")

    # Print feature overview
    print(f"\n{'='*72}")
    print("  OUTPUT SUMMARY")
    print(f"{'='*72}")
    print(f"\n  Time series CSV ({long_csv}):")
    print(f"    Each row = one grid cell at one bi-weekly time step")
    print(f"    {len(df_all)} observations across {df_all['time'].nunique() if 'time' in df_all else '?'} dates")
    if "x" in df_all.columns:
        print(f"    {df_all.groupby(['x','y']).ngroups} unique grid cells")
    print(f"\n  Columns:")
    for col in sorted(df_all.columns):
        print(f"    - {col}")

    if all_summary_frames:
        print(f"\n  Cell summary CSV ({summary_csv}):")
        print(f"    Each row = one grid cell with temporal statistics")
        print(f"    Includes: seasonal means, trends, flood frequency,")
        print(f"    hydroperiod variability — ready for habitat modeling")

    print(f"\n{'='*72}")
    print("  Done! These features capture:")
    print("    • Bi-weekly water levels and flooding patterns")
    print("    • Vegetation health and moisture through seasons")
    print("    • Migration-season habitat quality (spring/fall)")
    print("    • 5-year trends in wetland condition")
    print("    • Flood frequency and tidal dynamics")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
