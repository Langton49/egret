#!/usr/bin/env python3
"""
Pre-extract satellite spectral indices at colony centroids from NetCDF files.

For each NC file, selects a 300 m × 300 m window centred on the colony
lat/lon, excludes water pixels, and returns the mean of remaining land pixels.

CRS is read from each file:
  - Landsat files use WGS84 (EPSG:4326) — window defined in degrees
  - Sentinel-2 files use UTM Zone 16N (EPSG:32616) — window defined in metres

Band values are stored as float32 reflectance (0–1 range) in both sensor types.
The /10000 scale factor in the original plan does not apply to these files.

Usage:
    python training_two/extract_colony_centroids.py
    python training_two/extract_colony_centroids.py --verbose
    python training_two/extract_colony_centroids.py --force
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_COLONY_DIR = _HERE / "satellite_timeseries" / "colonies"
_REGISTRY   = _COLONY_DIR / "colony_registry.json"
_OUT        = _HERE / "colony_satellite_features.csv"

# Landsat raw band names → friendly names
BAND_MAP = {
    "B03": "green",
    "B04": "red",
    "B05": "nir",
    "B06": "swir16",
    "B07": "swir22",
}

# Window half-width: 150 m in UTM, ~0.00135° in WGS84 lat (~150 m at 29°N)
WINDOW_M   = 150.0
WINDOW_DEG = 0.00135  # degrees (~150 m at Gulf Coast latitudes)


def safe_nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Normalised difference (a - b) / (a + b), clipped to [-1, 1].
    Returns NaN when |denominator| < 1e-6 to avoid blowup from near-zero sums
    caused by negative reflectance artefacts in Landsat atmospheric correction.
    """
    d = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        nd = np.where(np.abs(d) > 1e-6, (a - b) / d, np.nan)
    return np.clip(nd, -1.0, 1.0)


def get_file_crs_epsg(ds) -> str | None:
    """Extract the EPSG authority code from the crs variable, e.g. 'EPSG:32616'."""
    crs_wkt = ds["crs"].attrs.get("crs_wkt", "")
    # Find the last AUTHORITY["EPSG","XXXXX"] in the WKT
    import re
    matches = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', crs_wkt)
    if matches:
        return f"EPSG:{matches[-1]}"
    return None


_transformers: dict[str, Transformer] = {}


def _get_transformer(target_epsg: str) -> Transformer:
    """Cache pyproj Transformers by target EPSG to avoid re-creating them."""
    if target_epsg not in _transformers:
        _transformers[target_epsg] = Transformer.from_crs(
            "EPSG:4326", target_epsg, always_xy=True
        )
    return _transformers[target_epsg]


def compute_and_extract(ds, lat: float, lon: float) -> dict:
    """
    Compute spectral indices and return the mean of non-water pixels within a
    ~300 m × 300 m window centred on the colony centroid.

    Reads the CRS from the NC file and projects lat/lon to that CRS
    automatically — works for WGS84 (Landsat), UTM Zone 16N (S2 eastern),
    and any other UTM zone the files may use.

    Water pixels are excluded before averaging.  If the window is all water
    or all NaN, the mean of all valid pixels in the window is returned.
    """
    # Rename Landsat band aliases if needed
    for old, new in BAND_MAP.items():
        if old in ds and new not in ds:
            ds = ds.rename({old: new})

    required = {"red", "nir", "green", "swir16"}
    missing = required - set(ds.data_vars)
    if missing:
        raise ValueError(f"Missing bands: {missing}")

    def get_band(name: str) -> np.ndarray:
        da = ds[name]
        if "t" in da.dims:
            da = da.squeeze("t")
        # Values are float reflectance (0–1); no scale factor needed
        return da.values

    red    = get_band("red")
    nir    = get_band("nir")
    green  = get_band("green")
    swir16 = get_band("swir16")

    NDVI       = safe_nd(nir, red)
    MNDWI      = safe_nd(green, swir16)
    NDWI       = safe_nd(green, nir)
    NDMI       = safe_nd(nir, swir16)
    # Water mask: positive MNDWI and low NDVI (only reliable when swir16 valid)
    water_mask = np.where(
        ~np.isnan(MNDWI) & (MNDWI > 0) & (NDVI < 0.2), 1.0, 0.0
    )

    x_vals = ds["x"].values
    y_vals = ds["y"].values

    # Project centroid to the file's own CRS
    epsg = get_file_crs_epsg(ds)
    if epsg and epsg != "EPSG:4326":
        transformer = _get_transformer(epsg)
        cx, cy = transformer.transform(lon, lat)
        window = WINDOW_M
    else:
        # WGS84 lat/lon
        cx, cy = lon, lat
        window = WINDOW_DEG

    x_mask = np.abs(x_vals - cx) <= window
    y_mask = np.abs(y_vals - cy) <= window

    if not x_mask.any() or not y_mask.any():
        raise ValueError(
            f"Centroid ({cx:.1f}, {cy:.1f}) outside NC extent "
            f"x=[{x_vals.min():.1f},{x_vals.max():.1f}] "
            f"y=[{y_vals.min():.1f},{y_vals.max():.1f}]"
        )

    # Land mask: not water AND NDVI pixel is valid
    ndvi_window = NDVI[np.ix_(y_mask, x_mask)]
    wm_window   = water_mask[np.ix_(y_mask, x_mask)]
    land_mask   = (wm_window == 0) & ~np.isnan(ndvi_window)

    result = {}
    for name, arr in [
        ("NDVI", NDVI), ("MNDWI", MNDWI),
        ("NDWI", NDWI), ("NDMI", NDMI), ("water_mask", water_mask),
    ]:
        window_vals = arr[np.ix_(y_mask, x_mask)]
        land_vals   = window_vals[land_mask]

        if land_vals.size > 0 and not np.all(np.isnan(land_vals)):
            val = float(np.nanmean(land_vals))
        else:
            # No land pixels — use any valid pixel in the window
            with np.errstate(all="ignore"):
                val = float(np.nanmean(window_vals))

        result[name] = None if np.isnan(val) else val

    return result


def parse_year_sensor(stem: str, slug: str) -> tuple[int | None, str | None]:
    """
    Extract year and sensor label from a filename stem like:
        breton_island_2010_landsat  →  (2010, 'landsat')
        breton_island_2017_s2       →  (2017, 's2')
    """
    suffix = stem[len(slug):].lstrip("_")
    parts = suffix.split("_", 1)
    if not parts:
        return None, None
    year_str = parts[0]
    if not year_str.isdigit() or not (2000 <= int(year_str) <= 2030):
        return None, None
    year   = int(year_str)
    sensor = parts[1] if len(parts) > 1 else "unknown"
    return year, sensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Print per-file progress")
    parser.add_argument("--force", action="store_true", help="Re-extract even if CSV exists")
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    if _OUT.exists() and not args.force:
        log.info("CSV already exists at %s", _OUT)
        log.info("Use --force to re-extract.")
        sys.exit(0)

    if not _REGISTRY.exists():
        log.error("Registry not found: %s", _REGISTRY)
        sys.exit(1)

    with open(_REGISTRY) as f:
        raw = json.load(f)
    registry: dict = {c["slug"]: c for c in raw} if isinstance(raw, list) else raw
    log.info("Registry loaded: %d entries", len(registry))

    rows: list[dict] = []
    n_files  = 0
    n_errors = 0

    colony_dirs = sorted(d for d in _COLONY_DIR.iterdir() if d.is_dir())
    log.info("Found %d colony directories to process", len(colony_dirs))

    import xarray as xr

    for colony_dir in colony_dirs:
        slug   = colony_dir.name
        entry  = registry.get(slug)
        if not entry:
            log.debug("No registry entry for '%s' — skipping", slug)
            continue

        colony_name = entry.get("name") or entry.get("ColonyName", slug)

        # Build list of (lat, lon) cells to sample — one per member colony.
        # For single-colony groups the members list may be empty, so we fall
        # back to the group centroid.  Using individual member positions avoids
        # the group centroid landing in open water between island clusters.
        members = entry.get("members") or []
        cells: list[tuple[float, float]] = [
            (m["lat"], m["lon"]) for m in members
            if m.get("lat") is not None and m.get("lon") is not None
        ]
        if not cells:
            lat = entry.get("lat")
            lon = entry.get("lon")
            if lat is None or lon is None:
                log.warning("No lat/lon for '%s' — skipping", slug)
                continue
            cells = [(lat, lon)]

        nc_files = sorted(
            p for p in colony_dir.glob(f"{slug}_*.nc")
            if not p.name.endswith("_swir_patch.nc")
        )
        if not nc_files:
            log.debug("%s: no NC files", slug)
            continue

        log.info("%s (%d files, %d cells)", slug, len(nc_files), len(cells))

        INDEX_KEYS = ["NDVI", "MNDWI", "NDWI", "NDMI", "water_mask"]

        for nc_path in nc_files:
            year, sensor = parse_year_sensor(nc_path.stem, slug)
            if year is None:
                log.warning("  Cannot parse year from %s — skipping", nc_path.name)
                n_errors += 1
                continue

            try:
                ds = xr.open_dataset(nc_path)

                # Extract at each member cell, then average across cells
                cell_results: list[dict] = []
                for clat, clon in cells:
                    try:
                        cell_results.append(compute_and_extract(ds, clat, clon))
                    except Exception as ce:
                        log.debug("  %s cell (%.4f, %.4f): %s", nc_path.name, clat, clon, ce)

                ds.close()
            except Exception as exc:
                log.warning("  %s: %s", nc_path.name, exc)
                n_errors += 1
                continue

            if not cell_results:
                log.warning("  %s: all cells failed", nc_path.name)
                n_errors += 1
                continue

            # Mean across cells (nanmean — ignore cells with no data)
            indices: dict = {}
            for key in INDEX_KEYS:
                vals = [r[key] for r in cell_results if r.get(key) is not None]
                indices[key] = float(np.mean(vals)) if vals else None

            # Representative lat/lon: first member (or group centroid for singles)
            rep_lat, rep_lon = cells[0]

            rows.append({
                "slug":        slug,
                "colony_name": colony_name,
                "year":        year,
                "sensor":      sensor,
                "lat":         rep_lat,
                "lon":         rep_lon,
                **indices,
            })
            n_files += 1

            if args.verbose:
                log.debug(
                    "  %d  NDVI=%-7s  MNDWI=%-7s  sensor=%s  cells=%d",
                    year,
                    f"{indices['NDVI']:.3f}" if indices.get("NDVI") is not None else "None",
                    f"{indices['MNDWI']:.3f}" if indices.get("MNDWI") is not None else "None",
                    sensor,
                    len(cell_results),
                )

    df = pd.DataFrame(rows)
    df.to_csv(_OUT, index=False)

    n_colonies = df["slug"].nunique() if not df.empty else 0
    log.info(
        "Summary: %d colonies, %d rows written, %d files skipped/failed",
        n_colonies, len(rows), n_errors,
    )
    log.info("Output: %s", _OUT)


if __name__ == "__main__":
    main()
