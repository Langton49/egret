"""
=============================================================================
  Build Model Input: Satellite Indices + Bird Observations
  One row per (grid_cell, dekadal_period)
=============================================================================

Joins:
  - Sentinel-2 derived spectral indices (from NetCDF tiles)
  - eBird + iNaturalist observation counts and diversity metrics

Each row = one 1-km grid cell at one dekadal (10-day) time step, containing:
  - Spectral indices: NDVI, NDWI, MNDWI, NDMI, EVI, SAVI, LSWI, WRI, etc.
  - Bird metrics: n_observations, n_species, shannon_diversity, plus
    counts by taxonomic order

Usage:
    python build_model_input.py
    python build_model_input.py --cell-size 1.0
    python build_model_input.py --input ./satellite_timeseries --output model_input.csv
"""

import os
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import xarray as xr
except ImportError:
    sys.exit("ERROR: xarray + netcdf4 required.  pip install xarray netcdf4")


# ===========================================================================
# CONFIGURATION
# ===========================================================================

AOI_BOUNDS = (-90.628342, 28.927421, -89.067224, 30.106372)  # west, south, east, north
AOI_BUFFER_KM = 5  # include observations up to 5 km outside AOI

SATELLITE_DIR = "./satellite_timeseries"
EBIRD_CSV = "./data_cache/ebird_data.csv"
INAT_CSV = "./data_cache/inat_data.csv"
OUTPUT_CSV = "./model_input.csv"

CELL_SIZE_KM = 1.0

YEAR_MIN = 2020
YEAR_MAX = 2025

# Band aliases (AWS names → friendly names)
BAND_ALIASES = {
    "B02": "blue",   "blue": "blue",
    "B03": "green",  "green": "green",
    "B04": "red",    "red": "red",
    "B08": "nir",    "nir": "nir",
    "B8A": "nir_narrow", "nir_narrow": "nir_narrow", "nir08": "nir_narrow",
    "B11": "swir1",  "swir1": "swir1", "swir16": "swir1",
    "B12": "swir2",  "swir2": "swir2", "swir22": "swir2",
}

INDEX_NAMES = [
    "NDVI", "NDWI", "MNDWI", "NDMI", "EVI", "SAVI",
    "LSWI", "WRI", "wetland_moisture_index", "water_mask",
    "tc_wetness", "GCVI",
]


# ===========================================================================
# GRID HELPERS
# ===========================================================================

def build_grid_params(cell_size_km: float):
    """
    Compute grid parameters for the AOI.
    Returns cell sizes in degrees and the buffered AOI bounds.
    """
    mid_lat = (AOI_BOUNDS[1] + AOI_BOUNDS[3]) / 2.0
    km_per_deg_lon = 111.32 * np.cos(np.radians(mid_lat))
    km_per_deg_lat = 110.57

    cell_lon = cell_size_km / km_per_deg_lon
    cell_lat = cell_size_km / km_per_deg_lat

    # Buffered AOI for including nearby observations
    buf_deg = AOI_BUFFER_KM / 111.0  # rough conversion
    buffered = (
        AOI_BOUNDS[0] - buf_deg,
        AOI_BOUNDS[1] - buf_deg,
        AOI_BOUNDS[2] + buf_deg,
        AOI_BOUNDS[3] + buf_deg,
    )

    return cell_lon, cell_lat, buffered


def assign_grid_cell(lon, lat, cell_lon, cell_lat):
    """Assign a lon/lat point to its grid cell center."""
    cx = np.floor(lon / cell_lon) * cell_lon + cell_lon / 2
    cy = np.floor(lat / cell_lat) * cell_lat + cell_lat / 2
    return cx, cy


def assign_dekad(date):
    """
    Assign a date to its dekadal period.
    Dekads: 1st-10th, 11th-20th, 21st-end of month.
    Returns the start date of the dekad.
    """
    if isinstance(date, str):
        date = pd.Timestamp(date)
    day = date.day
    if day <= 10:
        return pd.Timestamp(year=date.year, month=date.month, day=1)
    elif day <= 20:
        return pd.Timestamp(year=date.year, month=date.month, day=11)
    else:
        return pd.Timestamp(year=date.year, month=date.month, day=21)


# ===========================================================================
# BIRD DATA LOADING
# ===========================================================================

def load_bird_observations(ebird_path: str, inat_path: str,
                           cell_lon: float, cell_lat: float,
                           buffered_aoi: tuple) -> pd.DataFrame:
    """
    Load and unify eBird + iNaturalist observations.
    Filter to AOI (with buffer), date range, and assign grid cells + dekads.
    Returns DataFrame with: cell_x, cell_y, dekad, species, order, count
    """
    print("\n" + "=" * 60)
    print("  Loading bird observation data")
    print("=" * 60)

    frames = []

    # ── eBird ──
    if os.path.exists(ebird_path):
        print(f"\n  Loading eBird: {ebird_path}")

        # Auto-detect delimiter and columns
        with open(ebird_path, "r", encoding="utf-8", errors="replace") as f:
            header_line = f.readline()

        # Detect delimiter
        if "\t" in header_line:
            sep = "\t"
        else:
            sep = ","

        # Get actual column names from file
        file_cols = [c.strip().strip('"') for c in header_line.split(sep)]
        file_cols = [c for c in file_cols if c]  # drop empty trailing columns
        print(f"    Delimiter: {'tab' if sep == chr(9) else 'comma'}")
        print(f"    Columns found: {len(file_cols)}")

        # Map the columns we need (handle variations in naming)
        col_map = {}
        for fc in file_cols:
            fl = fc.lower()
            if fl in ("decimallatitude", "decimal_latitude", "latitude"):
                col_map["lat"] = fc
            elif fl in ("decimallongitude", "decimal_longitude", "longitude"):
                col_map["lon"] = fc
            elif fl in ("eventdate", "event_date", "observed_on", "date"):
                col_map["date"] = fc
            elif fl == "species":
                col_map["species"] = fc
            elif fl in ("individualcount", "individual_count"):
                col_map["count"] = fc
            elif fl == "order":
                col_map["order"] = fc
            elif fl == "family":
                col_map["family"] = fc

        needed = {"lat", "lon", "date"}
        if not needed.issubset(col_map.keys()):
            missing = needed - col_map.keys()
            print(f"    WARNING: Could not find columns for {missing}")
            print(f"    Available: {file_cols[:15]}...")
        else:
            use_cols = list(col_map.values())
            print(f"    Using columns: {col_map}")

            # GBIF exports sometimes wrap entire rows in quotes with trailing commas.
            # Pre-process: strip outer quotes and trailing commas, then parse.
            import io
            import tempfile

            print("    Cleaning GBIF quote-wrapped format ...")
            clean_path = ebird_path + ".clean.tmp"
            with open(ebird_path, "r", encoding="utf-8", errors="replace") as fin, \
                 open(clean_path, "w", encoding="utf-8") as fout:
                for line in fin:
                    # Strip trailing whitespace, commas, then outer quotes
                    line = line.rstrip()
                    line = line.rstrip(",")
                    if line.startswith('"') and line.endswith('"'):
                        line = line[1:-1]
                    fout.write(line + "\n")

            print("    Reading cleaned file ...")
            chunks = pd.read_csv(
                clean_path, sep=sep,
                dtype=str,
                chunksize=500_000,
                on_bad_lines="skip",
                encoding_errors="replace",
                quoting=3,  # QUOTE_NONE — data has internal commas in fields
            )

            ebird_records = []
            total_raw = 0
            for chunk in chunks:
                total_raw += len(chunk)

                # Strip whitespace and quotes from column names
                chunk.columns = [c.strip().strip('"').strip("'") for c in chunk.columns]

                # Debug: print columns on first chunk
                if total_raw <= 500_000:
                    print(f"    Chunk columns (first 10): {list(chunk.columns)[:10]}")

                # Build rename map by matching lowercase column names
                rename = {}
                chunk_cols_lower = {c.lower(): c for c in chunk.columns}
                for key, orig_col in col_map.items():
                    orig_lower = orig_col.lower()
                    if orig_lower in chunk_cols_lower:
                        actual_col = chunk_cols_lower[orig_lower]
                        rename[actual_col] = key
                chunk = chunk.rename(columns=rename)

                # Check we have the essentials
                if not all(k in chunk.columns for k in ["lat", "lon", "date"]):
                    if total_raw <= 500_000:
                        print(f"    WARNING: Missing columns after rename. Have: {list(chunk.columns)[:10]}")
                    continue

                # Convert types
                chunk["lat"] = pd.to_numeric(chunk["lat"], errors="coerce")
                chunk["lon"] = pd.to_numeric(chunk["lon"], errors="coerce")
                chunk = chunk.dropna(subset=["lat", "lon", "date"])

                # Spatial filter (with buffer)
                chunk = chunk[
                    (chunk["lon"] >= buffered_aoi[0]) &
                    (chunk["lon"] <= buffered_aoi[2]) &
                    (chunk["lat"] >= buffered_aoi[1]) &
                    (chunk["lat"] <= buffered_aoi[3])
                ]

                if chunk.empty:
                    continue

                # Parse dates
                chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
                chunk = chunk.dropna(subset=["date"])

                # Year filter
                chunk = chunk[
                    (chunk["date"].dt.year >= YEAR_MIN) &
                    (chunk["date"].dt.year <= YEAR_MAX)
                ]

                if chunk.empty:
                    continue

                # Parse individual count
                if "count" in chunk.columns:
                    chunk["count"] = pd.to_numeric(chunk["count"], errors="coerce").fillna(1).astype(int)
                else:
                    chunk["count"] = 1

                # Assign grid cells
                cx, cy = assign_grid_cell(
                    chunk["lon"].values, chunk["lat"].values,
                    cell_lon, cell_lat,
                )

                # Assign dekads
                chunk["dekad"] = chunk["date"].apply(assign_dekad)

                out = pd.DataFrame({
                    "cell_x": cx,
                    "cell_y": cy,
                    "dekad": chunk["dekad"].values,
                    "species": chunk["species"].values if "species" in chunk.columns else np.nan,
                    "order": chunk["order"].values if "order" in chunk.columns else np.nan,
                    "family": chunk["family"].values if "family" in chunk.columns else np.nan,
                    "count": chunk["count"].values,
                    "source": "ebird",
                })
                ebird_records.append(out)

            if ebird_records:
                ebird_df = pd.concat(ebird_records, ignore_index=True)
                print(f"    Raw rows scanned: {total_raw:,}")
                print(f"    After filtering:  {len(ebird_df):,}")
                frames.append(ebird_df)
            else:
                print("    No eBird records matched filters.")

            # Clean up temp file
            if os.path.exists(clean_path):
                os.remove(clean_path)
    else:
        print(f"  eBird file not found: {ebird_path}")

    # ── iNaturalist ──
    if os.path.exists(inat_path):
        print(f"\n  Loading iNaturalist: {inat_path}")
        chunks = pd.read_csv(
            inat_path, usecols=[
                "latitude", "longitude", "observed_on",
                "scientific_name", "iconic_taxon_name",
            ],
            chunksize=500_000,
            on_bad_lines="skip",
        )

        inat_records = []
        total_raw = 0
        for chunk in chunks:
            total_raw += len(chunk)
            chunk = chunk.dropna(subset=["latitude", "longitude", "observed_on"])

            # Only birds
            chunk = chunk[chunk["iconic_taxon_name"] == "Aves"]

            # Spatial filter
            chunk = chunk[
                (chunk["longitude"] >= buffered_aoi[0]) &
                (chunk["longitude"] <= buffered_aoi[2]) &
                (chunk["latitude"] >= buffered_aoi[1]) &
                (chunk["latitude"] <= buffered_aoi[3])
            ]

            if chunk.empty:
                continue

            chunk["date"] = pd.to_datetime(chunk["observed_on"], errors="coerce")
            chunk = chunk.dropna(subset=["date"])
            chunk = chunk[
                (chunk["date"].dt.year >= YEAR_MIN) &
                (chunk["date"].dt.year <= YEAR_MAX)
            ]

            if chunk.empty:
                continue

            chunk["dekad"] = chunk["date"].apply(assign_dekad)

            cx, cy = assign_grid_cell(
                chunk["longitude"].values,
                chunk["latitude"].values,
                cell_lon, cell_lat,
            )

            out = pd.DataFrame({
                "cell_x": cx,
                "cell_y": cy,
                "dekad": chunk["dekad"].values,
                "species": chunk["scientific_name"].values,
                "order": np.nan,  # iNat export doesn't always have order
                "family": np.nan,
                "count": 1,
                "source": "inat",
            })
            inat_records.append(out)

        if inat_records:
            inat_df = pd.concat(inat_records, ignore_index=True)
            print(f"    Raw rows scanned: {total_raw:,}")
            print(f"    After filtering:  {len(inat_df):,} (Aves only)")
            frames.append(inat_df)
        else:
            print("    No iNat records matched filters.")
    else:
        print(f"  iNat file not found: {inat_path}")

    if not frames:
        print("\n  WARNING: No bird observations loaded!")
        return pd.DataFrame()

    all_obs = pd.concat(frames, ignore_index=True)
    print(f"\n  Total combined observations: {len(all_obs):,}")
    print(f"  Unique species: {all_obs['species'].nunique():,}")
    print(f"  Date range: {all_obs['dekad'].min()} → {all_obs['dekad'].max()}")

    return all_obs


def aggregate_bird_metrics(obs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw observations into per-cell, per-dekad metrics:
      - n_observations: total observation count
      - n_species: unique species count
      - shannon_diversity: Shannon diversity index
      - n_individuals: sum of individual counts
      - top order counts (e.g., n_Anseriformes, n_Charadriiformes, etc.)
    """
    if obs_df.empty:
        return pd.DataFrame()

    print("\n  Aggregating bird metrics per cell × dekad ...")

    groups = obs_df.groupby(["cell_x", "cell_y", "dekad"])

    # Basic metrics
    n_obs = groups.size().reset_index(name="n_observations")
    n_species = groups["species"].nunique().reset_index(name="n_species")
    n_individuals = groups["count"].sum().reset_index(name="n_individuals")

    # Shannon diversity
    def shannon(species_series):
        counts = species_series.value_counts()
        proportions = counts / counts.sum()
        return -np.sum(proportions * np.log(proportions + 1e-10))

    diversity = groups["species"].apply(shannon).reset_index(name="shannon_diversity")

    # Merge basic metrics
    metrics = n_obs.merge(n_species, on=["cell_x", "cell_y", "dekad"])
    metrics = metrics.merge(n_individuals, on=["cell_x", "cell_y", "dekad"])
    metrics = metrics.merge(diversity, on=["cell_x", "cell_y", "dekad"])

    # Top orders — find the most common orders across all data
    if obs_df["order"].notna().sum() > 0:
        top_orders = obs_df["order"].dropna().value_counts().head(10).index.tolist()
        print(f"  Top 10 orders: {', '.join(top_orders)}")

        for order_name in top_orders:
            col_name = f"n_{order_name.replace(' ', '_')}"
            order_counts = (
                obs_df[obs_df["order"] == order_name]
                .groupby(["cell_x", "cell_y", "dekad"])
                .size()
                .reset_index(name=col_name)
            )
            metrics = metrics.merge(
                order_counts, on=["cell_x", "cell_y", "dekad"], how="left"
            )
            metrics[col_name] = metrics[col_name].fillna(0).astype(int)

    print(f"  Bird metrics: {len(metrics):,} cell-dekad combinations")
    print(f"  Columns: {list(metrics.columns)}")

    return metrics


# ===========================================================================
# SATELLITE DATA PROCESSING
# ===========================================================================

def normalize_bands(ds: xr.Dataset) -> xr.Dataset:
    """Rename band variables to friendly names."""
    rename = {}
    for var in list(ds.data_vars):
        if var in BAND_ALIASES and BAND_ALIASES[var] != var:
            rename[var] = BAND_ALIASES[var]
    if rename:
        ds = ds.rename(rename)
    return ds


def normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rename coordinate variants to x/y/time."""
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
    return ds


def compute_indices(ds: xr.Dataset) -> xr.Dataset:
    """Compute spectral indices from raw reflectance bands."""
    available = list(ds.data_vars)
    needed = {"blue", "green", "red", "nir", "swir1", "swir2"}
    if not needed.issubset(set(available)):
        missing = needed - set(available)
        print(f"    WARNING: Missing bands {missing}, skipping index computation")
        return ds

    # Scale to reflectance (Sentinel-2 stores as DN × 10000)
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

    return ds


def detect_crs_and_get_transformer(ds: xr.Dataset):
    """
    Detect if dataset is in projected CRS (UTM meters) and return
    a transformer to convert to EPSG:4326 (lon/lat).
    Returns None if already in geographic coords.
    """
    x_range = float(ds.x.max()) - float(ds.x.min())
    if x_range > 1000:
        # Projected CRS (meters) — need to find the EPSG code
        try:
            from pyproj import Transformer, CRS

            # Try to get CRS from dataset attributes
            epsg = None
            if "crs" in ds:
                crs_var = ds["crs"]
                if hasattr(crs_var, "attrs"):
                    for attr in ["spatial_ref", "crs_wkt", "proj4"]:
                        if attr in crs_var.attrs:
                            try:
                                c = CRS.from_user_input(crs_var.attrs[attr])
                                epsg = c.to_epsg()
                                break
                            except Exception:
                                pass
            if "spatial_ref" in ds:
                sr = ds["spatial_ref"]
                if hasattr(sr, "attrs"):
                    for attr in ["spatial_ref", "crs_wkt", "proj4"]:
                        if attr in sr.attrs:
                            try:
                                c = CRS.from_user_input(sr.attrs[attr])
                                epsg = c.to_epsg()
                                break
                            except Exception:
                                pass

            if epsg is None:
                # Guess UTM zone from x, y values
                # x ~200k-800k, y > 0 = northern hemisphere UTM
                x_mid = float(ds.x.mean())
                y_mid = float(ds.y.mean())
                # Mississippi Delta is UTM zone 15N or 16N
                if x_mid < 500000:
                    epsg = 32616  # UTM 16N
                else:
                    epsg = 32615  # UTM 15N
                print(f"    Guessed CRS: EPSG:{epsg}")

            transformer = Transformer.from_crs(
                f"EPSG:{epsg}", "EPSG:4326", always_xy=True
            )
            print(f"    CRS: EPSG:{epsg} (projected) → will convert to lat/lon")
            return transformer

        except ImportError:
            print("    WARNING: pyproj not installed, cannot convert projected coords")
            print("    Install with: pip install pyproj")
            return None
    else:
        print("    CRS: Geographic (lat/lon)")
        return None


def process_single_tile(nc_path: str, cell_lon: float, cell_lat: float) -> pd.DataFrame:
    """
    Process one NetCDF tile:
      1. Open lazily with chunking
      2. Normalize band names
      3. Compute spectral indices
      4. Coarsen to ~1 km grid cells
      5. Convert to long DataFrame (one row per cell × dekad)
    """
    basename = os.path.basename(nc_path)
    size_gb = os.path.getsize(nc_path) / (1024**3)
    print(f"\n  Processing: {basename} ({size_gb:.1f} GB)")

    # Open with chunking to avoid loading everything into memory
    ds = xr.open_dataset(nc_path, chunks={"time": 1})
    ds = normalize_coords(ds)
    ds = normalize_bands(ds)

    print(f"    Dims: {dict(ds.sizes)}")
    print(f"    Vars: {list(ds.data_vars)}")

    # Detect CRS and prepare coordinate transformer
    transformer = detect_crs_and_get_transformer(ds)

    # Skip non-numeric / metadata vars
    skip_vars = ["crs", "spatial_ref"]
    for sv in skip_vars:
        if sv in ds:
            ds = ds.drop_vars(sv)

    # Compute indices
    print("    Computing spectral indices ...")
    ds = compute_indices(ds)

    # Determine coarsening factor
    if len(ds.x) > 1:
        x_res = abs(float(ds.x[1] - ds.x[0]))
        x_range = float(ds.x.max() - ds.x.min())
        if x_range > 1000:
            # Projected (meters) — target is cell_size_km * 1000 meters
            target_m = CELL_SIZE_KM * 1000
            factor = max(1, int(round(target_m / x_res)))
        else:
            # Geographic (degrees)
            factor = max(1, int(round(cell_lat / x_res)))
    else:
        factor = 1

    if factor > 1:
        print(f"    Coarsening spatial dims by {factor}× ...")
        numeric_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
        ds = ds[numeric_vars].coarsen(x=factor, y=factor, boundary="trim").mean()

    # Select only index variables (drop raw bands to save memory)
    keep = [v for v in INDEX_NAMES if v in ds.data_vars]
    if not keep:
        print(f"    WARNING: No index variables found, keeping all numeric vars")
        keep = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
    ds = ds[keep]

    # Convert time steps one at a time to avoid memory blowup
    print(f"    Extracting {ds.sizes.get('time', 0)} time steps ...")
    time_vals = ds.time.values if "time" in ds.dims else [None]

    all_rows = []
    for i, t in enumerate(time_vals):
        if t is not None:
            slice_ds = ds.sel(time=t).load()  # load one time step
            dekad = assign_dekad(pd.Timestamp(t))
        else:
            slice_ds = ds.load()
            dekad = None

        # Flatten to 1D
        df_slice = slice_ds.to_dataframe().reset_index()

        # Get x, y from index/columns
        if "x" not in df_slice.columns:
            if isinstance(df_slice.index, pd.MultiIndex):
                for name in df_slice.index.names:
                    if name in ("x", "y"):
                        df_slice[name] = df_slice.index.get_level_values(name)
                df_slice = df_slice.reset_index(drop=True)

        # Drop all-NaN rows
        feat_cols = [c for c in df_slice.columns if c in keep]
        if feat_cols:
            df_slice = df_slice.dropna(subset=feat_cols, how="all")

        if df_slice.empty:
            continue

        # Assign grid cell centers
        if "x" in df_slice.columns and "y" in df_slice.columns:
            if transformer is not None:
                # Convert projected (UTM) → geographic (lon/lat)
                lons, lats = transformer.transform(
                    df_slice["x"].values, df_slice["y"].values
                )
                if i == 0:
                    print(f"    UTM sample: x={df_slice['x'].values[0]:.1f}, y={df_slice['y'].values[0]:.1f}")
                    print(f"    → Lon/Lat:  lon={lons[0]:.6f}, lat={lats[0]:.6f}")
            else:
                lons = df_slice["x"].values
                lats = df_slice["y"].values

            cx, cy = assign_grid_cell(lons, lats, cell_lon, cell_lat)
            df_slice["cell_x"] = cx
            df_slice["cell_y"] = cy
            if i == 0:
                print(f"    Cell center: cx={cx[0]:.6f}, cy={cy[0]:.6f}")

        df_slice["dekad"] = dekad

        # Average within each grid cell (in case coarsening didn't perfectly align)
        group_cols = ["cell_x", "cell_y", "dekad"]
        if all(c in df_slice.columns for c in group_cols):
            agg = df_slice.groupby(group_cols)[feat_cols].mean().reset_index()
        else:
            agg = df_slice

        all_rows.append(agg)

        if (i + 1) % 12 == 0:
            print(f"      {i + 1}/{len(time_vals)} time steps done")

    if not all_rows:
        print(f"    No data extracted from {basename}")
        return pd.DataFrame()

    tile_df = pd.concat(all_rows, ignore_index=True)
    print(f"    Extracted: {len(tile_df):,} rows, {len(tile_df.columns)} columns")

    # Free memory
    ds.close()

    return tile_df


# ===========================================================================
# TEMPORAL CONTEXT
# ===========================================================================

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-derived features."""
    if "dekad" not in df.columns:
        return df

    df = df.copy()
    df["dekad"] = pd.to_datetime(df["dekad"])
    df["year"] = df["dekad"].dt.year
    df["month"] = df["dekad"].dt.month
    df["day_of_year"] = df["dekad"].dt.dayofyear

    # Bird-relevant seasons
    conditions = [
        df["month"].isin([3, 4, 5]),
        df["month"].isin([6, 7]),
        df["month"].isin([8, 9, 10]),
        df["month"].isin([11, 12, 1, 2]),
    ]
    labels = ["spring_migration", "breeding", "fall_migration", "winter"]
    df["bird_season"] = np.select(conditions, labels, default="unknown")

    # Cyclical encoding
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    return df


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build model input: satellite indices + bird observations"
    )
    parser.add_argument("--input", default=SATELLITE_DIR,
                        help=f"Satellite NetCDF directory (default: {SATELLITE_DIR})")
    parser.add_argument("--ebird", default=EBIRD_CSV,
                        help=f"eBird CSV path (default: {EBIRD_CSV})")
    parser.add_argument("--inat", default=INAT_CSV,
                        help=f"iNaturalist CSV path (default: {INAT_CSV})")
    parser.add_argument("--output", default=OUTPUT_CSV,
                        help=f"Output CSV path (default: {OUTPUT_CSV})")
    parser.add_argument("--cell-size", type=float, default=CELL_SIZE_KM,
                        help=f"Grid cell size in km (default: {CELL_SIZE_KM})")
    parser.add_argument("--tiles", type=str, default=None,
                        help="Comma-separated tile IDs to process (e.g., '0' or '0,1,2')")
    parser.add_argument("--years", type=str, default=None,
                        help="Comma-separated years to process (e.g., '2022' or '2022,2023')")
    args = parser.parse_args()

    print("=" * 72)
    print("  Build Model Input: Satellite + Bird Observations")
    print("=" * 72)
    print(f"  AOI          : {AOI_BOUNDS}")
    print(f"  AOI buffer   : {AOI_BUFFER_KM} km (for edge observations)")
    print(f"  Cell size    : {args.cell_size} km")
    print(f"  Years        : {YEAR_MIN}-{YEAR_MAX}")
    print(f"  Satellite dir: {args.input}")
    print(f"  eBird        : {args.ebird}")
    print(f"  iNat         : {args.inat}")
    print(f"  Output       : {args.output}")
    print("=" * 72)

    # ── Grid setup ──
    cell_lon, cell_lat, buffered_aoi = build_grid_params(args.cell_size)
    print(f"\n  Grid cell: {cell_lon:.6f}° lon × {cell_lat:.6f}° lat")

    # ── Load bird observations ──
    obs_df = load_bird_observations(
        args.ebird, args.inat, cell_lon, cell_lat, buffered_aoi
    )

    # ── Aggregate bird metrics ──
    bird_metrics = aggregate_bird_metrics(obs_df)

    # ── Process satellite tiles ──
    nc_files = sorted(glob.glob(os.path.join(args.input, "tile*.nc")))
    nc_files = [f for f in nc_files if "_indices" not in f]

    # Filter by tile ID if specified
    if args.tiles:
        tile_ids = [t.strip() for t in args.tiles.split(",")]
        nc_files = [f for f in nc_files
                    if any(f"tile{tid}_" in os.path.basename(f) for tid in tile_ids)]
        print(f"\n  Filtering to tiles: {tile_ids}")

    # Filter by year if specified
    if args.years:
        year_list = [y.strip() for y in args.years.split(",")]
        nc_files = [f for f in nc_files
                    if any(f"_{yr}.nc" in os.path.basename(f) for yr in year_list)]
        print(f"  Filtering to years: {year_list}")

    if not nc_files:
        sys.exit(f"No matching tile*.nc files found in {args.input}/")

    print(f"\n{'='*60}")
    print(f"  Processing {len(nc_files)} satellite tiles")
    print(f"{'='*60}")

    all_satellite = []
    for nc_path in nc_files:
        try:
            tile_df = process_single_tile(nc_path, cell_lon, cell_lat)
            if not tile_df.empty:
                all_satellite.append(tile_df)
        except Exception as e:
            print(f"    ERROR processing {os.path.basename(nc_path)}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_satellite:
        sys.exit("No satellite data extracted from any tile.")

    # Combine all tiles
    print(f"\n  Combining {len(all_satellite)} tile DataFrames ...")
    sat_df = pd.concat(all_satellite, ignore_index=True)

    # Deduplicate: if tiles overlap, average the values
    group_cols = ["cell_x", "cell_y", "dekad"]
    feat_cols = [c for c in sat_df.columns if c not in group_cols]
    numeric_feats = [c for c in feat_cols if np.issubdtype(sat_df[c].dtype, np.number)]

    print(f"  Deduplicating overlapping cells ...")
    n_before = len(sat_df)
    sat_df = sat_df.groupby(group_cols)[numeric_feats].mean().reset_index()
    print(f"    {n_before:,} → {len(sat_df):,} rows after dedup")

    # ── Add temporal features ──
    print("\n  Adding temporal context features ...")
    sat_df = add_temporal_features(sat_df)

    # ── Join bird data ──
    print("\n  Joining bird observation metrics ...")
    if not bird_metrics.empty:
        bird_metrics["dekad"] = pd.to_datetime(bird_metrics["dekad"])
        sat_df["dekad"] = pd.to_datetime(sat_df["dekad"])

        result = sat_df.merge(
            bird_metrics,
            on=["cell_x", "cell_y", "dekad"],
            how="left",
        )

        # Fill missing bird data with 0 (no observations = 0 birds seen)
        bird_cols = [c for c in bird_metrics.columns if c not in group_cols]
        for col in bird_cols:
            if col in result.columns:
                result[col] = result[col].fillna(0)

        n_with_birds = (result["n_observations"] > 0).sum()
        print(f"    Cells with bird data: {n_with_birds:,} / {len(result):,} "
              f"({100 * n_with_birds / len(result):.1f}%)")
    else:
        result = sat_df
        print("    No bird data to join.")

    # ── Save ──
    print(f"\n  Saving to {args.output} ...")
    result.to_csv(args.output, index=False)

    # ── Summary ──
    print(f"\n{'='*72}")
    print(f"  MODEL INPUT READY")
    print(f"{'='*72}")
    print(f"  Output file : {args.output}")
    print(f"  Total rows  : {len(result):,}")
    print(f"  Columns     : {len(result.columns)}")
    print(f"  Date range  : {result['dekad'].min()} → {result['dekad'].max()}")
    print(f"  Grid cells  : {result.groupby(['cell_x', 'cell_y']).ngroups:,}")

    # Column summary
    sat_cols = [c for c in INDEX_NAMES if c in result.columns]
    bird_cols_out = [c for c in result.columns if c.startswith("n_") or c in
                     ("shannon_diversity",)]
    time_cols = ["year", "month", "day_of_year", "bird_season", "doy_sin", "doy_cos"]
    time_cols = [c for c in time_cols if c in result.columns]

    print(f"\n  Satellite indices ({len(sat_cols)}): {', '.join(sat_cols)}")
    print(f"  Bird metrics ({len(bird_cols_out)}): {', '.join(bird_cols_out)}")
    print(f"  Temporal features ({len(time_cols)}): {', '.join(time_cols)}")

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\n  File size: {file_size:.1f} MB")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()