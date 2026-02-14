"""
=============================================================================
  Avian Habitat Time Series — Mississippi Delta
  Year-by-year Sentinel-2 composites via CDSE OpenEO
=============================================================================

Submits ONE YEAR at a time across 4 spatial tiles to avoid CDSE memory errors.
Start with the most recent year, then backfill older years as needed.

Each job produces a NetCDF with dims (time, x, y) containing ~36 dekadal
(~10-day) composites of cloud-masked Sentinel-2 imagery.

File naming:  tile0_2024.nc, tile1_2024.nc, tile2_2024.nc, tile3_2024.nc
              tile0_2025.nc, ...

Usage:
    # Submit 2024-2025 (default — most recent year first)
    python fetch_biweekly_timeseries.py

    # Submit a specific year
    python fetch_biweekly_timeseries.py --year 2023

    # Submit all years 2020-2025
    python fetch_biweekly_timeseries.py --year all

    # Submit just one tile for one year
    python fetch_biweekly_timeseries.py --year 2024 --tile 2

    # Check status / download
    python fetch_biweekly_timeseries.py --check
    python fetch_biweekly_timeseries.py --download

    # Compute indices locally (if you used --no-udf)
    python fetch_biweekly_timeseries.py --compute-all
"""

import os
import sys
import json
import time
import glob
import argparse
import numpy as np
from datetime import datetime
from typing import List, Tuple

# ===========================================================================
# CONFIGURATION
# ===========================================================================

# Mississippi Delta AOI
AOI_BOUNDS = (-90.628342, 28.927421, -89.067224, 30.106372)

# Available years — process one at a time
ALL_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
DEFAULT_YEAR = "2024-2025"  # start here

# Spatial tiling — 3×3 = 9 smaller tiles per year
N_TILES_X = 3
N_TILES_Y = 3

# Output
OUTPUT_DIR = "./satellite_timeseries"
JOBS_FILE = os.path.join(OUTPUT_DIR, "tile_jobs.json")

# CDSE OpenEO
OPENEO_BACKEND = "https://openeo.dataspace.copernicus.eu"

# Sentinel-2 bands
BANDS = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12", "SCL"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# TILING
# ===========================================================================

def split_aoi_into_tiles(
    bounds: Tuple[float, float, float, float],
    nx: int = 2, ny: int = 2
) -> List[dict]:
    """Split bounding box into nx × ny tiles."""
    min_lon, min_lat, max_lon, max_lat = bounds
    lon_step = (max_lon - min_lon) / nx
    lat_step = (max_lat - min_lat) / ny

    tiles = []
    idx = 0
    for i in range(nx):
        for j in range(ny):
            west = min_lon + i * lon_step
            south = min_lat + j * lat_step
            east = min_lon + (i + 1) * lon_step
            north = min_lat + (j + 1) * lat_step

            tiles.append({
                "tile_id": idx,
                "bbox": (west, south, east, north),
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [west, south],
                        [east, south],
                        [east, north],
                        [west, north],
                        [west, south]
                    ]]
                }
            })
            idx += 1

    return tiles


def parse_years(year_arg: str) -> List[int]:
    """Parse the --year argument into a list of years."""
    if year_arg == "all":
        return ALL_YEARS

    # Handle ranges like "2024-2025"
    if "-" in year_arg:
        parts = year_arg.split("-")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            start, end = int(parts[0]), int(parts[1])
            return [y for y in range(start, end + 1)]

    # Single year
    if year_arg.isdigit():
        return [int(year_arg)]

    print(f"ERROR: Invalid year '{year_arg}'. Use: 2024, 2023-2025, or all")
    sys.exit(1)


# ===========================================================================
# PROCESS GRAPHS
# ===========================================================================

def build_process_graph_udf(connection, tile_geom: dict, temporal_range: tuple):
    """
    Full process graph with UDF to compute all indices server-side.
    Returns raw bands + NDVI, NDWI, MNDWI, NDMI, EVI, SAVI, LSWI, WRI,
    wetland_moisture_index, water_mask, tc_wetness.
    """
    import openeo

    s2 = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=tile_geom,
        temporal_extent=list(temporal_range),
        bands=BANDS,
    )

    # Cloud masking via SCL
    scl = s2.band("SCL")
    cloud_mask = (
        (scl != 0) & (scl != 1) & (scl != 2) & (scl != 3) &
        (scl != 8) & (scl != 9) & (scl != 10)
    )
    s2_masked = s2.mask(cloud_mask)
    s2_clean = s2_masked.filter_bands(
        ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
    )

    # Dekadal composite (~10-day median)
    s2_composite = s2_clean.aggregate_temporal_period(
        period="dekad",
        reducer="median"
    )

    # UDF to compute indices in one pass
    udf_code = """
import xarray as xr
import numpy as np

def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    blue  = cube.sel(bands="B02").astype("float32")
    green = cube.sel(bands="B03").astype("float32")
    red   = cube.sel(bands="B04").astype("float32")
    nir   = cube.sel(bands="B08").astype("float32")
    nir_n = cube.sel(bands="B8A").astype("float32")
    swir1 = cube.sel(bands="B11").astype("float32")
    swir2 = cube.sel(bands="B12").astype("float32")

    scale = 10000.0
    blue, green, red = blue / scale, green / scale, red / scale
    nir, nir_n = nir / scale, nir_n / scale
    swir1, swir2 = swir1 / scale, swir2 / scale

    def safe_ratio(a, b):
        denom = a + b
        return xr.where(denom != 0, (a - b) / denom, np.nan)

    ndvi  = safe_ratio(nir, red)
    ndwi  = safe_ratio(green, nir)
    mndwi = safe_ratio(green, swir1)
    ndmi  = safe_ratio(nir, swir1)

    evi = xr.where(
        (nir + 6 * red - 7.5 * blue + 1) != 0,
        2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1),
        np.nan
    )
    savi = xr.where(
        (nir + red + 0.5) != 0,
        1.5 * (nir - red) / (nir + red + 0.5),
        np.nan
    )
    lswi = safe_ratio(nir, swir1)
    wri = xr.where(
        (nir + swir1) != 0,
        (green + red) / (nir + swir1),
        np.nan
    )
    wetland_idx = (ndwi + ndmi + mndwi) / 3.0
    water_mask = xr.where((mndwi > 0) & (ndvi < 0.2), 1.0, 0.0)
    tc_wetness = (
        0.1509 * blue + 0.1973 * green + 0.3279 * red
        + 0.3406 * nir - 0.7112 * swir1 - 0.4572 * swir2
    )

    all_bands = xr.concat(
        [blue, green, red, nir, nir_n, swir1, swir2,
         ndvi, ndwi, mndwi, ndmi, evi, savi, lswi, wri,
         wetland_idx, water_mask, tc_wetness],
        dim="bands"
    )
    all_bands["bands"] = [
        "blue", "green", "red", "nir", "nir_narrow", "swir1", "swir2",
        "NDVI", "NDWI", "MNDWI", "NDMI", "EVI", "SAVI", "LSWI", "WRI",
        "wetland_moisture_index", "water_mask", "tc_wetness"
    ]
    return all_bands
"""

    result = s2_composite.apply_dimension(
        dimension="bands",
        process=openeo.UDF(udf_code, runtime="Python"),
    )

    return result


def build_process_graph_raw(connection, tile_geom: dict, temporal_range: tuple):
    """Simpler graph — raw bands only, indices computed locally after download."""

    s2 = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=tile_geom,
        temporal_extent=list(temporal_range),
        bands=BANDS,
    )

    scl = s2.band("SCL")
    cloud_mask = (
        (scl != 0) & (scl != 1) & (scl != 2) & (scl != 3) &
        (scl != 8) & (scl != 9) & (scl != 10)
    )
    s2_masked = s2.mask(cloud_mask)
    s2_clean = s2_masked.filter_bands(
        ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
    )

    s2_composite = s2_clean.aggregate_temporal_period(
        period="dekad",
        reducer="median"
    )

    return s2_composite


def build_process_graph_aws(connection, tile_geom: dict, temporal_range: tuple):
    """
    Load Sentinel-2 from AWS Earth Search STAC instead of CDSE's internal
    archive. This completely avoids the corrupt JP2 files on CDSE's S3.

    Note: SCL is not available as a band in AWS Earth Search STAC metadata,
    so we pre-filter by cloud cover percentage instead of per-pixel SCL masking.
    The dekadal median composite further removes remaining cloud contamination.
    """
    AWS_STAC_URL = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a"

    # Extract bbox from GeoJSON polygon coordinates
    coords = tile_geom["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    bbox = {
        "west": min(lons),
        "south": min(lats),
        "east": max(lons),
        "north": max(lats),
    }

    s2 = connection.load_stac(
        url=AWS_STAC_URL,
        spatial_extent=bbox,
        temporal_extent=list(temporal_range),
        bands=["blue", "green", "red", "nir", "nir08", "swir16", "swir22"],
        properties={"eo:cloud_cover": lambda cc: cc <= 10},
    )

    # Dekadal median composite — the median naturally rejects cloud pixels
    # when enough clear scenes are available (which is typical at 5-day revisit)
    s2_composite = s2.aggregate_temporal_period(
        period="dekad",
        reducer="median"
    )

    return s2_composite


# ===========================================================================
# CONNECTION
# ===========================================================================

def connect_cdse():
    """Connect and authenticate to CDSE OpenEO."""
    import openeo

    print(f"Connecting to {OPENEO_BACKEND} ...")
    conn = openeo.connect(OPENEO_BACKEND)
    conn.authenticate_oidc()
    user = conn.describe_account().get("user_id", "unknown")
    print(f"Authenticated as: {user}")
    return conn


# ===========================================================================
# JOB MANAGEMENT
# ===========================================================================

def job_key(tile_id: int, year: int) -> str:
    """Unique identifier for a tile+year job."""
    return f"tile{tile_id}_{year}"


def job_filename(tile_id: int, year: int) -> str:
    """Output filename for a tile+year result."""
    return f"tile{tile_id}_{year}.nc"


def submit_jobs_for_year(connection, tiles: list, year: int,
                         use_udf: bool = True, use_aws: bool = False,
                         start_date: str = None, end_date: str = None):
    """Submit batch jobs for all tiles in a single year."""
    jobs = load_jobs_file()

    date_start = start_date or f"{year}-01-01"
    date_end = end_date or f"{year}-12-31"
    temporal_range = (date_start, date_end)

    # Use a suffix if custom dates to avoid overwriting standard jobs
    if start_date or end_date:
        date_suffix = f"{date_start}_to_{date_end}"
    else:
        date_suffix = None

    src = "AWS" if use_aws else "CDSE"
    print(f"\n{'='*60}")
    print(f"  YEAR {year}  |  {date_start} -> {date_end}  |  Source: {src}")
    print(f"  Tiles: {len(tiles)}  |  Mode: {'UDF' if use_udf else 'raw bands'}")
    print(f"{'='*60}")

    for tile in tiles:
        tid = tile["tile_id"]

        # Use date-aware key if custom range, else standard tile+year
        if date_suffix:
            key = f"tile{tid}_{date_suffix}"
            fname = f"tile{tid}_{date_suffix}.nc"
        else:
            key = job_key(tid, year)
            fname = job_filename(tid, year)

        # Skip if already submitted and not failed
        existing = [j for j in jobs if j.get("job_key") == key]
        if existing and existing[0].get("status") not in ("error", "canceled", None):
            print(f"\n  [{key}] already submitted "
                  f"(job {existing[0]['job_id']}), skipping.")
            continue

        print(f"\n  Submitting {key} ...")
        print(f"  Bbox: {tile['bbox']}")
        print(f"  Period: {temporal_range[0]} -> {temporal_range[1]}")

        try:
            if use_aws:
                cube = build_process_graph_aws(
                    connection, tile["geometry"], temporal_range
                )
            elif use_udf:
                try:
                    cube = build_process_graph_udf(
                        connection, tile["geometry"], temporal_range
                    )
                except Exception as e:
                    print(f"  UDF failed ({e}), falling back to raw bands ...")
                    cube = build_process_graph_raw(
                        connection, tile["geometry"], temporal_range
                    )
            else:
                cube = build_process_graph_raw(
                    connection, tile["geometry"], temporal_range
                )

            job = cube.create_job(
                title=f"avian_habitat_tile{tid}_{date_start}_to_{date_end}",
                out_format="NetCDF",
                job_options={
                    "executor-memory": "4g",
                    "executor-cores": "2",
                    # Skip corrupt JP2 files in CDSE archive instead of
                    # failing the whole job. Known issue with some S2
                    # tiles having broken CRS metadata in their JP2s.
                    "soft-errors": "true",
                }
            )
            job.start_job()

            entry = {
                "job_key": key,
                "tile_id": tid,
                "year": year,
                "date_range": [temporal_range[0], temporal_range[1]],
                "bbox": tile["bbox"],
                "job_id": job.job_id,
                "status": "queued",
                "filename": fname,
                "submitted_at": datetime.now().isoformat(),
            }

            # Replace existing entry or append
            jobs = [j for j in jobs if j.get("job_key") != key]
            jobs.append(entry)
            save_jobs_file(jobs)

            print(f"  Submitted: {job.job_id}")
            print(f"  Will save as: {fname}")

            time.sleep(3)  # small delay between submissions

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    return jobs


def check_jobs(connection, year_filter: int = None):
    """Check status of submitted jobs."""
    jobs = load_jobs_file()
    if not jobs:
        print("No jobs found. Submit some first.")
        return

    if year_filter:
        jobs_to_check = [j for j in jobs if j.get("year") == year_filter]
        print(f"\nChecking {len(jobs_to_check)} jobs for year {year_filter} ...\n")
    else:
        jobs_to_check = jobs
        print(f"\nChecking all {len(jobs_to_check)} jobs ...\n")

    # Group by year for display
    by_year = {}
    for j in jobs_to_check:
        yr = j.get("year", "?")
        by_year.setdefault(yr, []).append(j)

    for yr in sorted(by_year.keys()):
        print(f"  -- {yr} --")
        for j in by_year[yr]:
            try:
                job = connection.job(j["job_id"])
                info = job.describe_job()
                status = info.get("status", "unknown")
                j["status"] = status

                icon = {"finished": "[done]", "running": "[running]",
                        "queued": "[queued]", "error": "[ERROR]",
                        "canceled": "[canceled]"}.get(status, "[?]")

                print(f"    {icon} {j['job_key']:20s} | {j['job_id'][:24]}... | {status}")

            except Exception as e:
                j["status"] = "error"
                print(f"    [ERROR] {j['job_key']:20s} | {j['job_id'][:24]}... | {e}")

    save_jobs_file(jobs)

    # Summary
    all_statuses = [j.get("status") for j in jobs_to_check]
    finished = all_statuses.count("finished")
    running = sum(1 for s in all_statuses if s in ("running", "queued", "created"))
    errored = all_statuses.count("error")

    print(f"\n  Summary: {finished} finished, {running} running/queued, "
          f"{errored} errored (of {len(jobs_to_check)})")


def download_results(connection, year_filter: int = None):
    """Download finished job results with proper tile+year naming."""
    jobs = load_jobs_file()
    if not jobs:
        print("No jobs found.")
        return

    if year_filter:
        to_download = [j for j in jobs if j.get("year") == year_filter]
    else:
        to_download = jobs

    print(f"\nDownloading results ({len(to_download)} jobs) ...\n")

    for j in to_download:
        fname = j.get("filename", job_filename(j["tile_id"], j.get("year", "unknown")))
        output_path = os.path.join(OUTPUT_DIR, fname)

        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  [exists] {fname} ({size_mb:.1f} MB)")
            j["downloaded"] = True
            j["output_path"] = output_path
            continue

        try:
            job = connection.job(j["job_id"])
            info = job.describe_job()
            status = info.get("status", "unknown")

            if status != "finished":
                print(f"  [skip]   {fname} — status: {status}")
                continue

            print(f"  [downloading] {fname} ...")
            results = job.get_results()
            results.download_file(output_path)

            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  [saved]  {fname} ({size_mb:.1f} MB)")
            j["downloaded"] = True
            j["output_path"] = output_path

        except Exception as e:
            print(f"  [error]  {fname} — {e}")

    save_jobs_file(jobs)

    # Show what we have
    print(f"\nFiles in {OUTPUT_DIR}/:")
    for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "tile*.nc"))):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {os.path.basename(f):30s}  {size_mb:8.1f} MB")


# ===========================================================================
# LOCAL INDEX COMPUTATION (if --no-udf was used)
# ===========================================================================

def compute_indices_locally(nc_path: str, output_path: str = None):
    """Compute habitat indices from raw bands in a downloaded NetCDF."""
    import xarray as xr

    print(f"\nComputing indices: {os.path.basename(nc_path)}")
    ds = xr.open_dataset(nc_path)

    available = list(ds.data_vars)
    print(f"  Variables: {available}")

    # Map band names — handles both CDSE (B02, B03...) and AWS (blue, green, nir, nir08, swir16, swir22)
    band_map = {}
    for name in available:
        lower = name.lower()
        if "b02" in lower or lower == "blue":
            band_map["blue"] = name
        elif "b03" in lower or lower == "green":
            band_map["green"] = name
        elif "b04" in lower or lower == "red":
            band_map["red"] = name
        elif lower == "nir" or ("b08" in lower and "b8a" not in lower.replace("b08a", "")):
            band_map["nir"] = name
        elif lower in ("nir08", "nir_narrow") or "b8a" in lower or "b08a" in lower:
            band_map["nir_narrow"] = name
        elif lower in ("swir16", "swir1") or "b11" in lower:
            band_map["swir1"] = name
        elif lower in ("swir22", "swir2") or "b12" in lower:
            band_map["swir2"] = name

    needed = {"blue", "green", "red", "nir", "swir1", "swir2"}
    if not needed.issubset(set(band_map.keys())):
        missing = needed - set(band_map.keys())
        print(f"  ERROR: Missing bands {missing}, cannot compute indices")
        ds.close()
        return None

    scale = 10000.0

    def get(key):
        return ds[band_map[key]].astype("float32") / scale

    def safe_nd(a, b):
        d = a + b
        return xr.where(d != 0, (a - b) / d, np.nan)

    blue, green, red = get("blue"), get("green"), get("red")
    nir, swir1, swir2 = get("nir"), get("swir1"), get("swir2")
    nir_n = get("nir_narrow") if "nir_narrow" in band_map else nir

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

    out = output_path or nc_path.replace(".nc", "_indices.nc")
    ds.to_netcdf(out)
    ds.close()
    print(f"  Saved: {out}")
    return out


# ===========================================================================
# HELPERS
# ===========================================================================

def load_jobs_file() -> list:
    if os.path.exists(JOBS_FILE):
        with open(JOBS_FILE) as f:
            return json.load(f)
    return []


def save_jobs_file(jobs: list):
    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2, default=str)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fetch year-by-year Sentinel-2 time series for avian habitat analysis"
    )
    parser.add_argument(
        "--year", default=DEFAULT_YEAR,
        help="Year to process: 2024, 2023-2025, or 'all' (default: 2024-2025)"
    )
    parser.add_argument("--start-date", type=str, default=None,
                        help="Override start date (e.g. 2024-03-01 to skip bad scenes)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Override end date (e.g. 2024-12-31)")
    parser.add_argument("--tile", type=int, default=None,
                        help="Submit a single tile (0-3)")
    parser.add_argument("--check", action="store_true",
                        help="Check job status")
    parser.add_argument("--download", action="store_true",
                        help="Download finished results")
    parser.add_argument("--no-udf", action="store_true",
                        help="Fetch raw bands only (compute indices locally)")
    parser.add_argument("--compute-indices", type=str, default=None,
                        help="Compute indices on a specific .nc file")
    parser.add_argument("--compute-all", action="store_true",
                        help="Compute indices on all downloaded .nc files")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Resubmit jobs that errored")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: small ~20x20km area, 3 months, 1 tile, no UDF")
    parser.add_argument("--quarterly", action="store_true",
                        help="Split each year into quarters (avoids corrupt date ranges)")
    parser.add_argument("--use-aws", action="store_true",
                        help="Load data from AWS Earth Search instead of CDSE archive "
                             "(avoids corrupt JP2 files entirely)")
    args = parser.parse_args()

    # ── TEST MODE ──
    # Small area in the heart of the delta, 3 months, 1 tile, raw bands
    # Just to verify the pipeline works end-to-end
    if args.test:
        # ~20x20 km patch centered on the Birdfoot Delta
        # Date range deliberately includes April 14 2024 — the date with
        # corrupt JP2 files on CDSE that crashes load_collection
        TEST_BOUNDS = (-89.35, 29.15, -89.15, 29.35)
        TEST_YEAR = "2024"
        TEST_START = "2024-03-01"
        TEST_END = "2024-05-31"
        use_aws = getattr(args, 'use_aws', False)

        print("=" * 72)
        print("  TEST MODE — small area, 3 months, 1 tile")
        print("=" * 72)
        print(f"  AOI    : {TEST_BOUNDS}")
        print(f"  Period : {TEST_START} -> {TEST_END}")
        print(f"  Tiles  : 1 (no tiling)")
        print(f"  Source : {'AWS Earth Search' if use_aws else 'CDSE archive'}")
        print(f"  Mode   : raw bands (no UDF)")
        print("=" * 72)

        import openeo
        conn = connect_cdse()

        # Single tile = the whole test area
        test_tile = {
            "tile_id": "test",
            "bbox": TEST_BOUNDS,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [TEST_BOUNDS[0], TEST_BOUNDS[1]],
                    [TEST_BOUNDS[2], TEST_BOUNDS[1]],
                    [TEST_BOUNDS[2], TEST_BOUNDS[3]],
                    [TEST_BOUNDS[0], TEST_BOUNDS[3]],
                    [TEST_BOUNDS[0], TEST_BOUNDS[1]],
                ]]
            }
        }

        print("\n  Building process graph ...")
        if use_aws:
            cube = build_process_graph_aws(
                conn, test_tile["geometry"], (TEST_START, TEST_END)
            )
        else:
            cube = build_process_graph_raw(
                conn, test_tile["geometry"], (TEST_START, TEST_END)
            )

        test_label = "test_aws" if use_aws else "test_run"
        test_filename = f"test_tile_2024_mar_may{'_aws' if use_aws else ''}.nc"

        print("  Submitting test job ...")
        job = cube.create_job(
            title=f"avian_habitat_TEST{'_AWS' if use_aws else ''}",
            out_format="NetCDF",
            job_options={
                "executor-memory": "2g",
                "executor-cores": "1",
                "soft-errors": "true",
            }
        )
        job.start_job()
        print(f"  Submitted: {job.job_id}")

        # Save to jobs file for --check / --download
        test_entry = {
            "job_key": test_label,
            "tile_id": "test",
            "year": 2024,
            "date_range": [TEST_START, TEST_END],
            "bbox": TEST_BOUNDS,
            "job_id": job.job_id,
            "status": "queued",
            "filename": test_filename,
            "submitted_at": datetime.now().isoformat(),
        }
        jobs = load_jobs_file()
        jobs = [j for j in jobs if j.get("job_key") != "test_run"]
        jobs.append(test_entry)
        save_jobs_file(jobs)

        print(f"\n  Output will be: {OUTPUT_DIR}/test_tile_2024_jun_aug.nc")
        print(f"\n  Next:")
        print(f"    python fetch_biweekly_timeseries.py --check")
        print(f"    python fetch_biweekly_timeseries.py --download")
        print("=" * 72)
        return

    years = parse_years(args.year)

    print("=" * 72)
    print("  Avian Habitat Time Series — Mississippi Delta")
    print("  Year-by-year Sentinel-2 composites via CDSE OpenEO")
    print("=" * 72)
    print(f"  AOI      : {AOI_BOUNDS}")
    print(f"  Year(s)  : {years}")
    print(f"  Tiles    : {N_TILES_X}x{N_TILES_Y} = {N_TILES_X * N_TILES_Y} per year")
    print(f"  Jobs     : {len(years) * N_TILES_X * N_TILES_Y} total")
    print(f"  Output   : {OUTPUT_DIR}/")
    print(f"  Naming   : tile<N>_<YEAR>.nc")
    print("=" * 72)

    # -- Local-only operations (no connection needed) --

    if args.compute_indices:
        compute_indices_locally(args.compute_indices)
        return

    if args.compute_all:
        nc_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "tile*_*.nc")))
        nc_files = [f for f in nc_files if "_indices" not in f]
        if not nc_files:
            print("No tile files found.")
            return
        print(f"Computing indices for {len(nc_files)} files ...\n")
        for f in nc_files:
            compute_indices_locally(f)
        return

    # -- Connect to CDSE --

    import openeo
    conn = connect_cdse()

    tiles = split_aoi_into_tiles(AOI_BOUNDS, N_TILES_X, N_TILES_Y)

    # Filter to specific tile if requested
    if args.tile is not None:
        tiles = [t for t in tiles if t["tile_id"] == args.tile]
        if not tiles:
            sys.exit(f"Tile {args.tile} not found (valid: 0-{N_TILES_X * N_TILES_Y - 1})")

    # -- Check status --

    if args.check:
        if len(years) == 1:
            check_jobs(conn, year_filter=years[0])
        else:
            check_jobs(conn)
        return

    # -- Download --

    if args.download:
        if len(years) == 1:
            download_results(conn, year_filter=years[0])
        else:
            download_results(conn)
        return

    # -- Retry failed --

    if args.retry_failed:
        jobs = load_jobs_file()
        failed = [j for j in jobs if j.get("status") in ("error", "canceled")]
        if not failed:
            print("No failed jobs to retry.")
            return
        print(f"Retrying {len(failed)} failed jobs ...")
        for j in failed:
            j["status"] = None  # allow resubmission
        save_jobs_file(jobs)
        for yr in set(j["year"] for j in failed):
            yr_tiles = [t for t in tiles if t["tile_id"] in
                        [j["tile_id"] for j in failed if j["year"] == yr]]
            if yr_tiles:
                submit_jobs_for_year(conn, yr_tiles, yr, use_udf=not args.no_udf)
        return

    # -- Submit jobs year by year --

    use_udf = not args.no_udf
    use_aws = getattr(args, 'use_aws', False)
    quarterly = getattr(args, 'quarterly', False)

    if use_aws:
        print("\n  Data source: AWS Earth Search (avoids CDSE corrupt files)")
    if quarterly:
        print("  Splitting each year into quarters")

    for year in years:
        if quarterly:
            # Split year into 4 quarters to isolate corrupt date ranges
            quarters = [
                (f"{year}-01-01", f"{year}-03-31"),
                (f"{year}-04-01", f"{year}-06-30"),
                (f"{year}-07-01", f"{year}-09-30"),
                (f"{year}-10-01", f"{year}-12-31"),
            ]
            for q_start, q_end in quarters:
                submit_jobs_for_year(
                    conn, tiles, year,
                    use_udf=use_udf,
                    use_aws=use_aws,
                    start_date=q_start,
                    end_date=q_end,
                )
        else:
            submit_jobs_for_year(
                conn, tiles, year,
                use_udf=use_udf,
                use_aws=use_aws,
                start_date=args.start_date,
                end_date=args.end_date,
            )

    # Summary
    jobs = load_jobs_file()
    print(f"\n{'='*72}")
    print(f"  All submissions complete!")
    print(f"  Total jobs tracked: {len(jobs)}")
    print(f"{'='*72}")
    print(f"\n  Next steps:")
    print(f"    python fetch_biweekly_timeseries.py --check")
    print(f"    python fetch_biweekly_timeseries.py --check --year 2024")
    print(f"    python fetch_biweekly_timeseries.py --download")
    print(f"    python fetch_biweekly_timeseries.py --download --year 2024")
    if args.no_udf:
        print(f"    python fetch_biweekly_timeseries.py --compute-all")
    print(f"\n  Files will be saved as:")
    for year in years:
        for t in tiles:
            print(f"    {job_filename(t['tile_id'], year)}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()