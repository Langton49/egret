"""
=============================================================================
  Clear-Scene Sentinel-2 Prefetch
  Mississippi Delta Avian Habitat Analysis
=============================================================================

For each tile in the study area, finds the most recent cloud-free scene
and downloads just that one. Goes back up to 180 days if needed.

Result: 1 clear scene per tile, all 12 indices populated, minimal NaN.

Output:
  ./satellite_cache/latest_indices.csv    (scoring endpoint reads this)
  ./satellite_cache/cache_metadata.json
  ./satellite_cache/raw/tile{N}_clear.nc  (raw bands per tile)

Usage:
    python prefetch_clear.py                         # default
    python prefetch_clear.py --max-cloud 10          # stricter
    python prefetch_clear.py --max-days 90           # shorter window
    python prefetch_clear.py --tile 3                # single tile
    python prefetch_clear.py --refetch               # ignore cached tiles
"""

import os
import sys
import gc
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.filterwarnings("ignore", message="All-NaN slice")
warnings.filterwarnings("ignore", message="divide by zero")
warnings.filterwarnings("ignore", message="Converting non-nanosecond")

try:
    from pystac_client import Client
    import rioxarray
    from pyproj import Transformer
except ImportError:
    sys.exit("ERROR: pip install pystac-client rioxarray pyproj")

# ===========================================================================
# CONFIGURATION
# ===========================================================================

AOI_BOUNDS = (-90.628342, 28.927421, -89.067224, 30.106372)
N_TILES_X = 3
N_TILES_Y = 3

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
MAX_TILE_CLOUD = 40
MAX_AOI_CLOUD_PCT = 15
SEARCH_DAYS_BACK = 180
MAX_CANDIDATES = 50

BANDS_10M = ["blue", "green", "red", "nir"]
BANDS_20M = ["nir08", "swir16", "swir22", "scl"]
ALL_BANDS = BANDS_10M + BANDS_20M

CACHE_DIR = Path("./satellite_cache")
RAW_DIR = CACHE_DIR / "raw"
CELL_SIZE_KM = 1.0

INDEX_NAMES = [
    "NDVI", "NDWI", "MNDWI", "NDMI", "EVI", "SAVI",
    "LSWI", "WRI", "wetland_moisture_index", "water_mask",
    "tc_wetness", "GCVI",
]


# ===========================================================================
# TILING
# ===========================================================================

def get_tiles():
    min_lon, min_lat, max_lon, max_lat = AOI_BOUNDS
    lon_step = (max_lon - min_lon) / N_TILES_X
    lat_step = (max_lat - min_lat) / N_TILES_Y

    tiles = []
    for ix in range(N_TILES_X):
        for iy in range(N_TILES_Y):
            tiles.append({
                "id": ix * N_TILES_Y + iy,
                "bbox": (
                    min_lon + ix * lon_step,
                    min_lat + iy * lat_step,
                    min_lon + (ix + 1) * lon_step,
                    min_lat + (iy + 1) * lat_step,
                ),
            })
    return tiles


# ===========================================================================
# CLOUD ASSESSMENT
# ===========================================================================

def compute_cloud_fraction(scl_da):
    scl = scl_da.values.flatten()
    valid = scl[scl > 0]
    if len(valid) == 0:
        return 1.0
    cloudy = np.isin(valid, [8, 9, 10]).sum()
    return cloudy / len(valid)


def check_index_coverage(cell_df):
    if cell_df.empty:
        return 0.0
    index_cols = [c for c in cell_df.columns if c in INDEX_NAMES]
    complete = cell_df[index_cols].notna().all(axis=1).sum()
    return complete / len(cell_df)


# ===========================================================================
# FIND CLEAREST SCENE PER TILE
# ===========================================================================

def find_clear_scene(bbox, max_cloud_pct, max_days):
    client = Client.open(STAC_URL)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=max_days)

    search = client.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
        query={"eo:cloud_cover": {"lt": MAX_TILE_CLOUD}},
        max_items=MAX_CANDIDATES,
    )

    items = list(search.items())
    items.sort(key=lambda x: x.datetime, reverse=True)

    for item in items:
        date_str = item.datetime.strftime("%Y-%m-%d")
        tile_cloud = item.properties.get("eo:cloud_cover", 99)

        if "scl" not in item.assets:
            continue

        try:
            scl = rioxarray.open_rasterio(
                item.assets["scl"].href,
                chunks={"x": 512, "y": 512}
            )
            scl = scl.rio.clip_box(
                minx=bbox[0], miny=bbox[1],
                maxx=bbox[2], maxy=bbox[3],
                crs="EPSG:4326",
            )
            if "band" in scl.dims:
                scl = scl.squeeze("band", drop=True)
            scl = scl.load()

            cloud_frac = compute_cloud_fraction(scl)
            cloud_pct = cloud_frac * 100
            del scl

            status = "✓" if cloud_pct <= max_cloud_pct else "✗"
            print(f"      {date_str} — tile: {tile_cloud:.1f}% — AOI: {cloud_pct:.1f}% {status}")

            if cloud_pct <= max_cloud_pct:
                return item

        except Exception as e:
            print(f"      {date_str} — error: {e}")
            continue

    return None


# ===========================================================================
# CRS UTILITIES
# ===========================================================================

def extract_crs(ds):
    """Extract CRS from an xarray Dataset, checking multiple sources."""
    # Try dataset-level
    if hasattr(ds, "rio") and ds.rio.crs is not None:
        return ds.rio.crs

    # Try individual variables
    for var in ds.data_vars:
        if hasattr(ds[var], "rio") and ds[var].rio.crs is not None:
            return ds[var].rio.crs

    # Try spatial_ref variable
    if "spatial_ref" in ds:
        try:
            crs_wkt = ds["spatial_ref"].attrs.get("crs_wkt") or ds["spatial_ref"].attrs.get("spatial_ref")
            if crs_wkt:
                from pyproj import CRS
                return CRS.from_wkt(crs_wkt)
        except Exception:
            pass

    return None


# ===========================================================================
# BAND LOADING
# ===========================================================================

def load_scene(item, bbox):
    ds_bands = {}
    ref_band = None

    for band_name in ALL_BANDS:
        if band_name not in item.assets:
            continue

        href = item.assets[band_name].href
        try:
            da = rioxarray.open_rasterio(href, chunks={"x": 1024, "y": 1024})
            da = da.rio.clip_box(
                minx=bbox[0], miny=bbox[1],
                maxx=bbox[2], maxy=bbox[3],
                crs="EPSG:4326",
            )
            if "band" in da.dims:
                da = da.squeeze("band", drop=True)

            # Resample 20m to 10m
            if ref_band is None and band_name in BANDS_10M:
                ref_band = da
            elif ref_band is not None and da.shape != ref_band.shape:
                da = da.rio.reproject_match(ref_band)

            ds_bands[band_name] = da
        except Exception as e:
            print(f"      Warning: {band_name} failed: {e}")
            continue

    if not ds_bands:
        return None

    ds = xr.Dataset(ds_bands)

    # Preserve CRS from the reference band
    if ref_band is not None and hasattr(ref_band, "rio") and ref_band.rio.crs is not None:
        ds.rio.write_crs(ref_band.rio.crs, inplace=True)

    return ds


# ===========================================================================
# INDEX COMPUTATION
# ===========================================================================

def compute_indices(ds):
    scale = 10000.0

    # Extract CRS before computing (raw ds has it, result won't by default)
    src_crs = extract_crs(ds)

    band_map = {
        "blue": "blue", "green": "green", "red": "red",
        "nir": "nir", "swir16": "swir1", "swir22": "swir2",
    }

    bands = {}
    for aws_name, friendly in band_map.items():
        if aws_name in ds:
            bands[friendly] = ds[aws_name].astype("float32") / scale

    required = ["blue", "green", "red", "nir", "swir1", "swir2"]
    if not all(k in bands for k in required):
        missing = [k for k in required if k not in bands]
        print(f"      Missing bands: {missing}")
        return None

    blue, green, red = bands["blue"], bands["green"], bands["red"]
    nir, swir1, swir2 = bands["nir"], bands["swir1"], bands["swir2"]

    def safe_nd(a, b):
        d = a + b
        return xr.where(d != 0, (a - b) / d, np.nan)

    result = xr.Dataset()
    result["NDVI"] = safe_nd(nir, red)
    result["NDWI"] = safe_nd(green, nir)
    result["MNDWI"] = safe_nd(green, swir1)
    result["NDMI"] = safe_nd(nir, swir1)
    result["EVI"] = xr.where(
        (nir + 6 * red - 7.5 * blue + 1) != 0,
        2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1), np.nan
    )
    result["SAVI"] = xr.where(
        (nir + red + 0.5) != 0,
        1.5 * (nir - red) / (nir + red + 0.5), np.nan
    )
    result["LSWI"] = safe_nd(nir, swir1)
    result["WRI"] = xr.where(
        (nir + swir1) != 0, (green + red) / (nir + swir1), np.nan
    )
    result["wetland_moisture_index"] = (result["NDWI"] + result["NDMI"] + result["MNDWI"]) / 3.0
    result["water_mask"] = xr.where(
        (result["MNDWI"] > 0) & (result["NDVI"] < 0.2), 1.0, 0.0
    )
    result["tc_wetness"] = (
        0.1509 * blue + 0.1973 * green + 0.3279 * red
        + 0.3406 * nir - 0.7112 * swir1 - 0.4572 * swir2
    )
    result["GCVI"] = xr.where(green != 0, (nir / green) - 1, np.nan)

    # Cloud mask — only exclude definite clouds and no-data
    # Do NOT exclude class 3 (cloud shadow) — delta dark water
    # gets misclassified as cloud shadow
    if "scl" in ds:
        invalid = ds["scl"].isin([0, 1, 8, 9])
        for var in result.data_vars:
            result[var] = result[var].where(~invalid)

    # Carry CRS forward to the result dataset
    if src_crs is not None:
        result.rio.write_crs(src_crs, inplace=True)

    return result


# ===========================================================================
# CELL AGGREGATION
# ===========================================================================

def aggregate_to_cells(indices_ds, bbox):
    # Extract CRS before converting to dataframe (which loses it)
    src_crs = extract_crs(indices_ds)

    df = indices_ds.compute().to_dataframe().reset_index()

    # Drop spatial_ref column if present
    df = df.drop(columns=["spatial_ref"], errors="ignore")

    if "x" not in df.columns and "longitude" in df.columns:
        df = df.rename(columns={"longitude": "x", "latitude": "y"})
    if "x" not in df.columns:
        print(f"      No x/y columns found. Columns: {list(df.columns)}")
        return pd.DataFrame()

    x_range = df["x"].max() - df["x"].min()
    if x_range > 1000:
        # Data is in projected coordinates — reproject to lon/lat
        if src_crs is not None:
            crs_str = str(src_crs)
        else:
            # Fallback: guess UTM zone
            center_x = df["x"].mean()
            crs_str = "EPSG:32615" if center_x < 500000 else "EPSG:32616"

        print(f"      Reprojecting from {crs_str}")
        transformer = Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)
        lons, lats = transformer.transform(df["x"].values, df["y"].values)
        df["lon"] = lons
        df["lat"] = lats
        print(f"      Lon range: {df['lon'].min():.4f} to {df['lon'].max():.4f}")
        print(f"      Lat range: {df['lat'].min():.4f} to {df['lat'].max():.4f}")
        print(f"      Bbox filter: {bbox[0]:.4f}–{bbox[2]:.4f}, {bbox[1]:.4f}–{bbox[3]:.4f}")
    else:
        df["lon"] = df["x"]
        df["lat"] = df["y"]

    # Filter to bbox
    df = df[
        (df["lon"] >= bbox[0]) & (df["lon"] <= bbox[2]) &
        (df["lat"] >= bbox[1]) & (df["lat"] <= bbox[3])
    ]

    print(f"      Pixels in bbox: {len(df):,}")

    if df.empty:
        return pd.DataFrame()

    mid_lat = (bbox[1] + bbox[3]) / 2.0
    cell_lon = CELL_SIZE_KM / (111.32 * np.cos(np.radians(mid_lat)))
    cell_lat = CELL_SIZE_KM / 110.57

    df["cell_x"] = np.floor(df["lon"] / cell_lon) * cell_lon + cell_lon / 2
    df["cell_y"] = np.floor(df["lat"] / cell_lat) * cell_lat + cell_lat / 2

    index_cols = [c for c in df.columns if c in INDEX_NAMES]
    cell_df = df.groupby(["cell_x", "cell_y"])[index_cols].mean().reset_index()
    cell_df = cell_df.dropna(subset=index_cols, how="all")

    return cell_df


# ===========================================================================
# PROCESS A SINGLE TILE
# ===========================================================================

def process_tile(tile, max_cloud_pct, max_days, refetch=False):
    tile_id = tile["id"]
    bbox = tile["bbox"]
    nc_path = RAW_DIR / f"tile{tile_id}_clear.nc"

    print(f"\n  Tile {tile_id}/{N_TILES_X * N_TILES_Y - 1} — "
          f"({bbox[0]:.3f}, {bbox[1]:.3f}) to ({bbox[2]:.3f}, {bbox[3]:.3f})")

    # ── Try cached file ──
    if nc_path.exists() and not refetch:
        size_mb = nc_path.stat().st_size / 1e6
        print(f"    Cached ({size_mb:.1f} MB). Processing...")

        ds = xr.open_dataset(nc_path, chunks={"x": 1024, "y": 1024})
        indices = compute_indices(ds)
        ds.close()
        del ds
        gc.collect()

        if indices is not None:
            cell_df = aggregate_to_cells(indices, bbox)
            del indices
            gc.collect()

            if not cell_df.empty:
                coverage = check_index_coverage(cell_df)
                print(f"    {len(cell_df)} cells, {coverage:.0%} complete")
                return cell_df, {
                    "id": tile_id, "source": "cached",
                    "cells": len(cell_df), "coverage": round(coverage, 3),
                }

            del cell_df

        print(f"    Cached file unusable. Deleting and re-fetching...")
        gc.collect()
        try:
            nc_path.unlink()
        except PermissionError:
            gc.collect()
            try:
                nc_path.unlink()
            except Exception as e:
                print(f"    Could not delete: {e}")

    # ── Find clearest scene ──
    print(f"    Searching for clear scene...")
    item = find_clear_scene(bbox, max_cloud_pct, max_days)

    if item is None:
        print(f"    No clear scene found within {max_days} days.")
        return None, {"id": tile_id, "source": "none", "cells": 0}

    scene_date = item.datetime.strftime("%Y-%m-%d")
    print(f"    Selected: {scene_date} ({item.id})")

    # ── Load bands ──
    print(f"    Loading bands...")
    ds = load_scene(item, bbox)
    if ds is None:
        print(f"    Failed to load bands.")
        return None, {"id": tile_id, "source": "failed", "cells": 0}

    # ── Save raw ──
    print(f"    Saving → {nc_path.name}...")
    ds_computed = ds.compute()
    ds_computed.to_netcdf(nc_path)
    size_mb = nc_path.stat().st_size / 1e6
    print(f"    Saved ({size_mb:.1f} MB)")

    # ── Compute indices ──
    print(f"    Computing indices...")
    indices = compute_indices(ds_computed)
    del ds, ds_computed
    gc.collect()

    if indices is None:
        print(f"    Missing bands.")
        return None, {"id": tile_id, "source": scene_date, "cells": 0}

    # ── Aggregate ──
    print(f"    Aggregating to cells...")
    cell_df = aggregate_to_cells(indices, bbox)
    del indices
    gc.collect()

    if cell_df.empty:
        print(f"    No valid cells.")
        return None, {"id": tile_id, "source": scene_date, "cells": 0}

    coverage = check_index_coverage(cell_df)
    print(f"    {len(cell_df)} cells, {coverage:.0%} complete")

    return cell_df, {
        "id": tile_id,
        "source": scene_date,
        "scene_id": item.id,
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "cells": len(cell_df),
        "coverage": round(coverage, 3),
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Prefetch clearest Sentinel-2 scene per tile")
    parser.add_argument("--max-cloud", type=float, default=MAX_AOI_CLOUD_PCT,
                        help=f"Max cloud %% per tile (default: {MAX_AOI_CLOUD_PCT})")
    parser.add_argument("--max-days", type=int, default=SEARCH_DAYS_BACK,
                        help=f"How far back to search (default: {SEARCH_DAYS_BACK})")
    parser.add_argument("--tile", type=int, default=None,
                        help="Process a single tile only")
    parser.add_argument("--refetch", action="store_true",
                        help="Ignore cached tiles and re-fetch everything")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    tiles = get_tiles()
    if args.tile is not None:
        tiles = [t for t in tiles if t["id"] == args.tile]
        if not tiles:
            sys.exit(f"Tile {args.tile} not found (valid: 0-{N_TILES_X * N_TILES_Y - 1})")

    print("=" * 72)
    print("  Clear-Scene Sentinel-2 Prefetch")
    print("=" * 72)
    print(f"  Study area     : {AOI_BOUNDS}")
    print(f"  Tiles          : {len(tiles)}")
    print(f"  Max cloud      : {args.max_cloud}%")
    print(f"  Search window  : last {args.max_days} days")
    if args.refetch:
        print(f"  Mode           : REFETCH (ignoring cache)")
    print("=" * 72)

    all_cells = []
    tile_meta = []

    for tile in tiles:
        cell_df, meta = process_tile(tile, args.max_cloud, args.max_days, args.refetch)
        tile_meta.append(meta)
        if cell_df is not None:
            all_cells.append(cell_df)

    # ── Combine and save ──
    if not all_cells:
        print("\n  ERROR: No cells produced for any tile.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Combining tiles")
    print("=" * 60)

    full_df = pd.concat(all_cells, ignore_index=True)
    full_df = full_df.groupby(["cell_x", "cell_y"]).mean().reset_index()

    total_coverage = check_index_coverage(full_df)
    print(f"    Total cells: {len(full_df):,}")
    print(f"    Index coverage: {total_coverage:.0%}")

    latest_path = CACHE_DIR / "latest_indices.csv"
    full_df.to_csv(latest_path, index=False)
    print(f"    Saved: {latest_path}")

    # Metadata
    dates = [t["source"] for t in tile_meta
             if t["source"] not in ("cached", "none", "failed")]

    metadata = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "clearest single scene per tile",
        "max_cloud_pct": args.max_cloud,
        "search_days_back": args.max_days,
        "n_cells": len(full_df),
        "index_coverage": round(total_coverage, 3),
        "aoi_bounds": list(AOI_BOUNDS),
        "cell_size_km": CELL_SIZE_KM,
        "tiles": tile_meta,
    }

    if dates:
        metadata["oldest_scene"] = min(dates)
        metadata["newest_scene"] = max(dates)
        metadata["start_date"] = min(dates)
        metadata["end_date"] = max(dates)

    meta_path = CACHE_DIR / "cache_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print("\n" + "=" * 72)
    print("  PREFETCH COMPLETE")
    print("=" * 72)
    tiles_ok = sum(1 for t in tile_meta if t["cells"] > 0)
    tiles_fail = sum(1 for t in tile_meta if t["cells"] == 0)
    print(f"  Tiles with data  : {tiles_ok}/{len(tile_meta)}")
    if tiles_fail:
        print(f"  Tiles failed     : {tiles_fail}")
    print(f"  Total cells      : {len(full_df):,}")
    print(f"  Index coverage   : {total_coverage:.0%}")
    if dates:
        print(f"  Scene dates      : {min(dates)} to {max(dates)}")
    print(f"  Cache            : {latest_path}")
    for t in tile_meta:
        status = f"{t['cells']} cells" if t["cells"] > 0 else "NO DATA"
        src = t.get("scene_id", t["source"])
        if isinstance(src, str) and len(src) > 45:
            src = src[:45] + "..."
        print(f"    Tile {t['id']}: {status} — {src}")
    print("=" * 72)


if __name__ == "__main__":
    main()