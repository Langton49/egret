"""
=============================================================================
  Colony Habitat Time Series — TWI Avian Monitoring Sites
  Per-colony annual composites via CDSE OpenEO
=============================================================================

Fetches annual spectral index composites for each TWI colony site using:
  - Landsat Bimonthly Mosaic (2010-2016) — 30m, CDSE native collection
  - Sentinel-2 L2A           (2017+)     — 10m, AWS Earth Search STAC

LANDSAT_BIMONTHLY_MOSAIC band convention (CDSE / Landsat 8/9 native):
    B03=green  B04=red  B05=NIR  B06=SWIR1  B07=SWIR2
Collection is already cloud-free (bimonthly median composites).

Efficiency optimisations:
  - Filtered to Louisiana (State=LA) by default — 190 colonies vs 442 total
  - Nearby colonies are spatially grouped: one job covers the union bbox of
    all members within GROUP_THRESHOLD_DEG of each other.  This eliminates
    redundant downloads for island clusters (e.g. Biloxi North 17/22/25…).
  - The group's representative slug is the alphabetically-first member.
  - Each member colony's lat/lon is stored in the registry so the timeline
    endpoint can still extract values at the exact colony centroid.

Each job produces a NetCDF with dims (time, x, y) containing annual median
composites with NDVI, MNDWI, NDWI, NDMI computed server-side.

Output naming:  {group_slug}_{year}_landsat.nc   (2010-2016)
                {group_slug}_{year}_s2.nc         (2017+)

Usage:
    # List all colonies discovered from the CSV
    python fetch_colony_timeseries.py --list-colonies

    # Submit all colonies, all years 2010-2024
    python fetch_colony_timeseries.py --year all

    # Submit a single year
    python fetch_colony_timeseries.py --year 2018

    # Submit a single colony
    python fetch_colony_timeseries.py --year all --colony "Queen Bess Island"

    # Check status
    python fetch_colony_timeseries.py --check

    # Download finished results
    python fetch_colony_timeseries.py --download

    # Compute indices locally on downloaded .nc files
    python fetch_colony_timeseries.py --compute-all

    # Use a different state or grouping threshold
    python fetch_colony_timeseries.py --year all --state TX --group-threshold 0.05
"""

import os
import sys
import json
import time
import glob
import argparse
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict

# ===========================================================================
# CONFIGURATION
# ===========================================================================

# Years to fetch — Landsat for pre-2017, Sentinel-2 from 2017 onward
ALL_YEARS       = list(range(2010, 2025))
SENTINEL_FROM   = 2017          # switch to S2 at this year
DEFAULT_YEAR    = "all"

# Buffer around colony centroid in degrees (~5km each side for small islands)
# Small enough to keep job sizes tiny, large enough to capture island + buffer
COLONY_BUFFER_DEG = 0.05

# Spatial grouping: colonies within this distance (degrees, centroid-to-centroid)
# of any group member are merged into one job sharing a union bbox.
# 0.10 deg ≈ 11 km — reduces 190 LA colonies to ~112 groups (41% fewer jobs).
# Capped: a group bbox may not exceed MAX_GROUP_EXTENT_DEG on any side.
GROUP_THRESHOLD_DEG  = 0.10
MAX_GROUP_EXTENT_DEG = 0.30   # ~33 km — prevents runaway island-chain merges

# State filter — only fetch colonies in this state (None = all states)
ONLY_STATE = "LA"

# Cloud cover filter — higher threshold for small islands where partial cloud
# cover over water easily registers as "clear"
MAX_CLOUD_COVER = 20

# CDSE native Landsat collection — already cloud-free, no S3 access issues
LANDSAT_COLLECTION = "LANDSAT_BIMONTHLY_MOSAIC"

# AWS Earth Search STAC endpoint for Sentinel-2
S2_STAC = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a"

# CDSE OpenEO backend
OPENEO_BACKEND = "https://openeo.dataspace.copernicus.eu"

# Output — always relative to this script, regardless of working directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "satellite_timeseries", "colonies")
JOBS_FILE  = os.path.join(OUTPUT_DIR, "colony_jobs.json")
COLONY_REGISTRY = os.path.join(OUTPUT_DIR, "colony_registry.json")

# Colony CSV — override with --csv flag or AVIAN_CSV env var
# Default looks for the file relative to this script, but the VM path may differ
AVIAN_CSV = os.environ.get(
    "AVIAN_CSV",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "repo", "data", "databases", "processed", "avianData20102021.csv.gz"
    )
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# COLONY REGISTRY
# ===========================================================================

def _make_slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("'", "")
        .replace(",", "")
    )


def load_colonies_from_csv(csv_path: str, state_filter: str = None) -> List[Dict]:
    """
    Read unique colonies with lat/lon from avianData20102021.csv.gz.
    Optionally filters to a single state (e.g. 'LA').
    Returns list of individual colony dicts: {name, slug, lat, lon, state}
    """
    import pandas as pd

    print(f"Loading colonies from {csv_path} ...")
    df = pd.read_csv(
        csv_path,
        usecols=["ColonyName", "Latitude_x", "Longitude_x", "State",
                 "GeoRegion", "PrimaryHabitat"],
        low_memory=False,
    )
    df = df.dropna(subset=["Latitude_x", "Longitude_x"])
    df = df.drop_duplicates(subset=["ColonyName"])

    if state_filter:
        before = len(df)
        df = df[df["State"].str.strip().str.upper() == state_filter.upper()]
        print(f"  State filter '{state_filter}': {len(df)} / {before} colonies")

    colonies = []
    for _, row in df.iterrows():
        name  = str(row["ColonyName"]).strip()
        lat   = float(row["Latitude_x"])
        lon   = float(row["Longitude_x"])
        state = str(row.get("State", "")).strip()
        colonies.append({
            "name":            name,
            "slug":            _make_slug(name),
            "lat":             lat,
            "lon":             lon,
            "state":           state,
            "geo_region":      str(row.get("GeoRegion", "")).strip(),
            "primary_habitat": str(row.get("PrimaryHabitat", "")).strip(),
        })

    colonies.sort(key=lambda c: c["name"])
    print(f"  {len(colonies)} unique colonies loaded")
    return colonies


def group_colonies(
    colonies: List[Dict],
    threshold_deg: float = GROUP_THRESHOLD_DEG,
    max_extent_deg: float = MAX_GROUP_EXTENT_DEG,
) -> List[Dict]:
    """
    Greedy single-linkage grouping: colonies within threshold_deg of any
    existing group member are merged.  A group stops accepting new members
    once its bbox exceeds max_extent_deg on any side (prevents island chains
    from merging into one giant tile).

    Returns a list of group dicts, each with:
      slug, name, bbox, lat, lon (centroid), members (list of colony dicts)
    """
    assigned = [False] * len(colonies)
    groups: List[Dict] = []

    for i, anchor in enumerate(colonies):
        if assigned[i]:
            continue

        # Start a new group with this colony
        members = [anchor]
        assigned[i] = True

        # Scan remaining colonies for proximity to any current member
        changed = True
        while changed:
            changed = False
            for j, candidate in enumerate(colonies):
                if assigned[j]:
                    continue
                # Is it close enough to any existing member?
                close = False
                for m in members:
                    dist = np.sqrt(
                        (candidate["lat"] - m["lat"]) ** 2 +
                        (candidate["lon"] - m["lon"]) ** 2
                    )
                    if dist <= threshold_deg:
                        close = True
                        break
                if not close:
                    continue
                # Would adding this colony exceed the max raw coordinate span?
                lats = [m["lat"] for m in members]
                lons = [m["lon"] for m in members]
                new_lat_span = max(max(lats), candidate["lat"]) - min(min(lats), candidate["lat"])
                new_lon_span = max(max(lons), candidate["lon"]) - min(min(lons), candidate["lon"])
                if new_lat_span > max_extent_deg or new_lon_span > max_extent_deg:
                    continue
                members.append(candidate)
                assigned[j] = True
                changed = True

        # Build group bbox = union of member centroids ± COLONY_BUFFER_DEG
        lats = [m["lat"] for m in members]
        lons = [m["lon"] for m in members]
        bbox = (
            min(lons) - COLONY_BUFFER_DEG,
            min(lats) - COLONY_BUFFER_DEG,
            max(lons) + COLONY_BUFFER_DEG,
            max(lats) + COLONY_BUFFER_DEG,
        )
        centroid_lat = sum(lats) / len(lats)
        centroid_lon = sum(lons) / len(lons)

        # Representative name/slug = alphabetically first member (already sorted)
        primary = members[0]

        groups.append({
            "name":    primary["name"],
            "slug":    primary["slug"],
            "lat":     centroid_lat,
            "lon":     centroid_lon,
            "bbox":    bbox,
            "state":   primary["state"],
            "members": members,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]],
                    [bbox[0], bbox[3]],
                    [bbox[0], bbox[1]],
                ]]
            },
        })

    n_colonies = sum(len(g["members"]) for g in groups)
    print(f"  Grouped {n_colonies} colonies -> {len(groups)} job groups "
          f"(threshold={threshold_deg} deg, max_extent={max_extent_deg} deg)")
    return groups


def get_colony_registry(
    state_filter: str = None,
    threshold_deg: float = GROUP_THRESHOLD_DEG,
) -> List[Dict]:
    """Load cached colony registry or build from CSV."""
    if os.path.exists(COLONY_REGISTRY):
        with open(COLONY_REGISTRY) as f:
            return json.load(f)

    if not os.path.exists(AVIAN_CSV):
        print(f"ERROR: Colony CSV not found at {AVIAN_CSV}")
        print("Run fetch_databases.py first to download the avianData CSV.")
        sys.exit(1)

    raw      = load_colonies_from_csv(AVIAN_CSV, state_filter=state_filter)
    groups   = group_colonies(raw, threshold_deg=threshold_deg)

    with open(COLONY_REGISTRY, "w") as f:
        json.dump(groups, f, indent=2)
    print(f"Colony registry saved to {COLONY_REGISTRY}")

    return groups


# ===========================================================================
# PROCESS GRAPHS
# ===========================================================================

def build_landsat_graph(connection, colony: Dict, year: int):
    """
    Load LANDSAT_BIMONTHLY_MOSAIC from CDSE for a single colony AOI and year.

    CDSE collection is already cloud-free bimonthly composites — no masking needed.
    Band naming follows Landsat 8/9 native numbering:
        B03=green  B04=red  B05=NIR  B06=SWIR1  B07=SWIR2

    Annual median collapses the bimonthly time steps to a single annual value.
    """
    import openeo

    bbox = {
        "west":  colony["bbox"][0],
        "south": colony["bbox"][1],
        "east":  colony["bbox"][2],
        "north": colony["bbox"][3],
    }

    landsat = connection.load_collection(
        LANDSAT_COLLECTION,
        spatial_extent=bbox,
        temporal_extent=[f"{year}-01-01", f"{year}-12-31"],
        bands=["B03", "B04", "B05", "B06"],  # green, red, NIR, SWIR1
    )

    # Annual median across bimonthly time steps
    composite = landsat.aggregate_temporal_period(
        period="year",
        reducer="median"
    )

    # Compute indices server-side
    udf_code = _index_udf(sensor="landsat")
    result = composite.apply_dimension(
        dimension="bands",
        process=openeo.UDF(udf_code, runtime="Python"),
    )

    return result


def build_s2_graph(connection, colony: Dict, year: int):
    """
    Load Sentinel-2 L2A from AWS Earth Search for a single colony AOI and year.

    Band names on AWS Earth Search:
        blue, green, red, nir, nir08, swir16, swir22
    """
    import openeo

    bbox = {
        "west":  colony["bbox"][0],
        "south": colony["bbox"][1],
        "east":  colony["bbox"][2],
        "north": colony["bbox"][3],
    }

    s2 = connection.load_stac(
        url=S2_STAC,
        spatial_extent=bbox,
        temporal_extent=[f"{year}-01-01", f"{year}-12-31"],
        bands=["blue", "green", "red", "nir", "nir08", "swir16", "swir22"],
        properties={"eo:cloud_cover": lambda cc: cc <= MAX_CLOUD_COVER},
    )

    composite = s2.aggregate_temporal_period(
        period="year",
        reducer="median"
    )

    udf_code = _index_udf(sensor="s2")
    result = composite.apply_dimension(
        dimension="bands",
        process=openeo.UDF(udf_code, runtime="Python"),
    )

    return result


def _index_udf(sensor: str) -> str:
    """
    UDF to compute NDVI, NDWI, MNDWI, NDMI from either Landsat or S2 bands.

    Landsat (CDSE LANDSAT_BIMONTHLY_MOSAIC, GLAD ARD / L8 native numbering):
        B03=green, B04=red, B05=NIR, B06=SWIR1

    Sentinel-2 (AWS Earth Search common names):
        green, red, nir, swir16
    """
    if sensor == "landsat":
        band_code = """
    green = get("B03")
    red   = get("B04")
    nir   = get("B05")
    swir1 = get("B06")
"""
    else:
        band_code = """
    bands = list(cube.coords["bands"].values)
    green = get("green")
    red   = get("red")
    nir   = get("nir") if "nir" in bands else get("nir08")
    swir1 = get("swir16")
"""

    return f"""
import xarray as xr
import numpy as np

def apply_datacube(cube: xr.DataArray, context: dict) -> xr.DataArray:
    def get(name):
        return cube.sel(bands=name).astype("float32") / 10000.0

    def safe_ratio(a, b):
        denom = a + b
        return xr.where(denom != 0, (a - b) / denom, np.nan)
{band_code}
    ndvi  = safe_ratio(nir, red)
    ndwi  = safe_ratio(green, nir)
    mndwi = safe_ratio(green, swir1)
    ndmi  = safe_ratio(nir, swir1)
    water_mask = xr.where((mndwi > 0) & (ndvi < 0.2), 1.0, 0.0)

    result = xr.concat([ndvi, ndwi, mndwi, ndmi, water_mask], dim="bands")
    result["bands"] = ["NDVI", "NDWI", "MNDWI", "NDMI", "water_mask"]
    return result
"""


# ===========================================================================
# CONNECTION
# ===========================================================================

def connect_cdse():
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

def load_jobs() -> list:
    if os.path.exists(JOBS_FILE):
        with open(JOBS_FILE) as f:
            return json.load(f)
    return []


def save_jobs(jobs: list):
    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2, default=str)


def job_key(slug: str, year: int, sensor: str) -> str:
    return f"{slug}_{year}_{sensor}"


def job_filename(slug: str, year: int, sensor: str) -> str:
    return f"{slug}_{year}_{sensor}.nc"


def submit_jobs(connection, colonies: List[Dict], years: List[int]):
    jobs = load_jobs()
    submitted = skipped = errored = 0

    for colony in colonies:
        slug = colony["slug"]
        name = colony["name"]

        for year in years:
            sensor = "s2" if year >= SENTINEL_FROM else "landsat"
            key    = job_key(slug, year, sensor)
            fname  = job_filename(slug, year, sensor)

            # Skip if already submitted and not failed
            existing = [j for j in jobs if j.get("job_key") == key]
            if existing and existing[0].get("status") not in ("error", "canceled", None):
                skipped += 1
                continue

            print(f"\n  [{key}]")
            print(f"  Colony : {name}")
            print(f"  Bbox   : {colony['bbox']}")
            print(f"  Year   : {year}  |  Sensor: {sensor.upper()}")

            try:
                if sensor == "landsat":
                    cube = build_landsat_graph(connection, colony, year)
                else:
                    cube = build_s2_graph(connection, colony, year)

                job = cube.create_job(
                    title=f"colony_{slug}_{year}_{sensor}",
                    out_format="NetCDF",
                    job_options={
                        "executor-memory": "2g",
                        "executor-cores":  "1",
                        "soft-errors":     "true",
                    }
                )
                job.start_job()

                entry = {
                    "job_key":    key,
                    "colony":     name,
                    "slug":       slug,
                    "year":       year,
                    "sensor":     sensor,
                    "bbox":       colony["bbox"],
                    "lat":        colony["lat"],
                    "lon":        colony["lon"],
                    "job_id":     job.job_id,
                    "status":     "queued",
                    "filename":   fname,
                    "submitted_at": datetime.now().isoformat(),
                }

                jobs = [j for j in jobs if j.get("job_key") != key]
                jobs.append(entry)
                save_jobs(jobs)

                print(f"  Submitted: {job.job_id}  ->  {fname}")
                submitted += 1
                time.sleep(5)  # avoid hammering the API

            except Exception as e:
                print(f"  ERROR: {e}")
                errored += 1

    print(f"\n{'='*60}")
    print(f"  Submitted: {submitted} | Skipped (existing): {skipped} | Errors: {errored}")
    print(f"{'='*60}")


def server_sync(connection):
    """
    Pull ALL finished jobs from the CDSE server into colony_jobs.json
    so --download can grab them, even if they were never logged locally.
    """
    print("\nSyncing server jobs to local jobs file ...\n")
    try:
        all_server_jobs = []
        limit = 100
        page = 0
        while True:
            batch = connection.get("/jobs", params={"limit": limit, "page": page}).json()
            jobs_page = batch.get("jobs", [])
            if not jobs_page:
                break
            all_server_jobs.extend(jobs_page)
            print(f"  ... fetched {len(all_server_jobs)} so far")
            if not any(l.get("rel") == "next" for l in batch.get("links", [])):
                break
            page += 1
    except Exception as e:
        print(f"ERROR listing jobs: {e}")
        return

    registry = get_colony_registry()
    slug_to_group = {}
    for g in registry:
        slug_to_group[g["slug"]] = g
        for m in g.get("members", []):
            if m["slug"] not in slug_to_group:
                slug_to_group[m["slug"]] = g

    local_jobs = load_jobs()
    local_ids  = {j["job_id"] for j in local_jobs}

    added = skipped = unparsed = 0
    for sj in all_server_jobs:
        job_id = sj.get("id")
        status = sj.get("status")

        if job_id in local_ids:
            # Update status in case it changed
            for lj in local_jobs:
                if lj["job_id"] == job_id:
                    lj["status"] = status
            skipped += 1
            continue

        title = sj.get("title", "")
        try:
            rest   = title[len("colony_"):] if title.startswith("colony_") else title
            parts  = rest.split("_")
            sensor = parts[-1]
            year   = int(parts[-2])
            slug   = "_".join(parts[:-2])
        except Exception:
            unparsed += 1
            continue

        group = slug_to_group.get(slug)
        if not group:
            unparsed += 1
            continue

        key = job_key(group["slug"], year, sensor)
        entry = {
            "job_key":    key,
            "colony":     group["name"],
            "slug":       group["slug"],
            "year":       year,
            "sensor":     sensor,
            "bbox":       group["bbox"],
            "lat":        group["lat"],
            "lon":        group["lon"],
            "job_id":     job_id,
            "status":     status,
            "filename":   job_filename(group["slug"], year, sensor),
            "submitted_at": sj.get("created", ""),
        }
        local_jobs.append(entry)
        local_ids.add(job_id)
        added += 1

    save_jobs(local_jobs)

    statuses = [j.get("status") for j in local_jobs]
    print(f"\n  Added {added} new entries | Updated {skipped} existing | Unparsed {unparsed}")
    print(f"  Total in jobs file: {len(local_jobs)}")
    print(f"  Finished: {statuses.count('finished')} | Queued/Running: {sum(1 for s in statuses if s in ('queued','running','created'))} | Errored: {statuses.count('error')}")
    print(f"\n  Run --download to fetch all finished jobs.")


def server_check_and_restart(connection, restart_errored=False):
    """
    List ALL batch jobs on the CDSE server (not just local JSON).
    Optionally restart any that have errored.
    """
    print("\nFetching all jobs from CDSE server ...\n")
    try:
        all_jobs = []
        limit = 100
        page = 0
        while True:
            batch = connection.get("/jobs", params={"limit": limit, "page": page}).json()
            jobs_page = batch.get("jobs", [])
            if not jobs_page:
                break
            all_jobs.extend(jobs_page)
            print(f"  ... fetched {len(all_jobs)} so far")
            # stop if no next link
            links = batch.get("links", [])
            if not any(l.get("rel") == "next" for l in links):
                break
            page += 1
    except Exception as e:
        print(f"ERROR listing jobs: {e}")
        return

    by_status = {}
    for j in all_jobs:
        s = j.get("status", "unknown")
        by_status.setdefault(s, []).append(j)

    print(f"  Total jobs on server: {len(all_jobs)}")
    for status, jobs in sorted(by_status.items()):
        print(f"  {status:12s}: {len(jobs)}")

    errored = by_status.get("error", [])
    if not errored:
        print("\n  No errored jobs found.")
        return

    print(f"\n  Errored jobs ({len(errored)}):")
    for j in errored:
        title = j.get("title", j.get("id", "?"))
        print(f"    {j['id']:45s}  {title}")

    if restart_errored:
        print(f"\n  Resubmitting {len(errored)} errored jobs as fresh submissions ...")
        registry = get_colony_registry()
        # Build lookup: slug -> group, and member slug -> group
        slug_to_group = {}
        for g in registry:
            slug_to_group[g["slug"]] = g
            for m in g.get("members", []):
                if m["slug"] not in slug_to_group:
                    slug_to_group[m["slug"]] = g

        submitted = skipped = failed = 0
        for j in errored:
            title = j.get("title", "")
            # Parse title format: colony_{slug}_{year}_{sensor}
            try:
                rest = title[len("colony_"):] if title.startswith("colony_") else title
                parts = rest.split("_")
                sensor = parts[-1]           # s2 or landsat
                year   = int(parts[-2])      # 4-digit year
                slug   = "_".join(parts[:-2])
            except Exception:
                print(f"    [skip] Cannot parse title: {title}")
                skipped += 1
                continue

            group = slug_to_group.get(slug)
            if not group:
                print(f"    [skip] Slug '{slug}' not in registry — {title}")
                skipped += 1
                continue

            key = job_key(group["slug"], year, sensor)
            existing_jobs = load_jobs()
            existing = [jj for jj in existing_jobs if jj.get("job_key") == key]
            if existing and existing[0].get("status") not in ("error", "canceled", None):
                print(f"    [skip] Already submitted: {key}")
                skipped += 1
                continue

            try:
                if sensor == "landsat":
                    cube = build_landsat_graph(connection, group, year)
                else:
                    cube = build_s2_graph(connection, group, year)

                new_job = cube.create_job(
                    title=f"colony_{group['slug']}_{year}_{sensor}",
                    out_format="NetCDF",
                    job_options={"executor-memory": "2g", "executor-cores": "1", "soft-errors": "true"},
                )
                new_job.start_job()

                entry = {
                    "job_key":      key,
                    "colony":       group["name"],
                    "slug":         group["slug"],
                    "year":         year,
                    "sensor":       sensor,
                    "bbox":         group["bbox"],
                    "lat":          group["lat"],
                    "lon":          group["lon"],
                    "job_id":       new_job.job_id,
                    "status":       "queued",
                    "filename":     job_filename(group["slug"], year, sensor),
                    "submitted_at": datetime.now().isoformat(),
                }
                all_jobs = load_jobs()
                all_jobs = [jj for jj in all_jobs if jj.get("job_key") != key]
                all_jobs.append(entry)
                save_jobs(all_jobs)

                print(f"    [submitted] {key}  ({new_job.job_id})")
                submitted += 1
                time.sleep(5)

            except Exception as e:
                print(f"    [failed]    {title}  {e}")
                failed += 1

        print(f"\n  Submitted: {submitted} | Skipped: {skipped} | Failed: {failed}")
    else:
        print(f"\n  Run with --server-restart-errored to restart them.")


def check_jobs(connection, colony_filter=None, year_filter=None):
    jobs = load_jobs()
    if not jobs:
        print("No jobs found. Submit some first.")
        return

    if colony_filter:
        jobs = [j for j in jobs if colony_filter.lower() in j.get("colony", "").lower()]
    if year_filter:
        jobs = [j for j in jobs if j.get("year") == year_filter]

    print(f"\nChecking {len(jobs)} jobs ...\n")

    by_colony = {}
    for j in jobs:
        by_colony.setdefault(j.get("colony", "?"), []).append(j)

    for colony in sorted(by_colony.keys()):
        print(f"  -- {colony} --")
        for j in sorted(by_colony[colony], key=lambda x: x.get("year", 0)):
            try:
                job    = connection.job(j["job_id"])
                status = job.describe_job().get("status", "unknown")
                j["status"] = status
                icon   = {"finished": "ok", "running": ">>", "queued": "..",
                          "error": "!!", "canceled": "!!"}.get(status, "?")
                print(f"    [{icon}] {j['job_key']:45s} {status}")
            except Exception as e:
                j["status"] = "error"
                print(f"    [!] {j['job_key']:45s} {e}")

    save_jobs(jobs)

    statuses   = [j.get("status") for j in jobs]
    finished   = statuses.count("finished")
    running    = sum(1 for s in statuses if s in ("running", "queued", "created"))
    errored    = statuses.count("error")

    print(f"\n  Summary: {finished} finished | {running} running/queued | {errored} errored | {len(jobs)} total")


def download_results(connection, colony_filter=None, year_filter=None):
    jobs = load_jobs()
    if not jobs:
        print("No jobs found.")
        return

    if colony_filter:
        jobs = [j for j in jobs if colony_filter.lower() in j.get("colony", "").lower()]
    if year_filter:
        jobs = [j for j in jobs if j.get("year") == year_filter]

    print(f"\nDownloading results ({len(jobs)} jobs) ...\n")

    for j in jobs:
        fname  = j.get("filename", job_filename(j["slug"], j["year"], j["sensor"]))
        outpath = os.path.join(OUTPUT_DIR, j.get("slug", "unknown"), fname)
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        if os.path.exists(outpath):
            size_mb = os.path.getsize(outpath) / (1024 * 1024)
            print(f"  [exists] {fname} ({size_mb:.1f} MB)")
            j["downloaded"] = True
            j["output_path"] = outpath
            continue

        try:
            job    = connection.job(j["job_id"])
            status = job.describe_job().get("status", "unknown")

            if status != "finished":
                print(f"  [skip]   {fname} — {status}")
                continue

            print(f"  [downloading] {fname} ...")
            job.get_results().download_file(outpath)
            size_mb = os.path.getsize(outpath) / (1024 * 1024)
            print(f"  [saved]  {fname} ({size_mb:.1f} MB)")
            j["downloaded"] = True
            j["output_path"] = outpath

        except Exception as e:
            print(f"  [error]  {fname} — {e}")

    save_jobs(jobs)


# ===========================================================================
# LOCAL INDEX COMPUTATION (fallback if UDF failed)
# ===========================================================================

def compute_indices_locally(nc_path: str):
    """Compute NDVI, MNDWI, NDWI, NDMI from raw bands if UDF wasn't applied.

    Handles two band naming conventions:
      Landsat (GLAD ARD): B03=green, B04=red, B05=NIR, B06=SWIR1
      Sentinel-2 (common names): green, red, nir/nir08, swir16
    """
    import xarray as xr

    print(f"\nComputing indices: {os.path.basename(nc_path)}")
    ds = xr.open_dataset(nc_path)
    available = list(ds.data_vars)
    print(f"  Variables: {available}")

    if "NDVI" in available:
        print("  Indices already present, skipping.")
        ds.close()
        return

    scale = 10000.0

    def get(name):
        return ds[name].astype("float32") / scale

    def safe_nd(a, b):
        d = a + b
        return xr.where(d != 0, (a - b) / d, np.nan)

    # Detect Landsat GLAD ARD vs Sentinel-2 common names
    is_landsat_glad = "B03" in available
    if is_landsat_glad:
        green = get("B03")
        red   = get("B04")
        nir   = get("B05")
        swir1 = get("B06")
    else:
        green = get("green")
        red   = get("red")
        nir   = get("nir") if "nir" in available else get("nir08")
        swir1 = get("swir16")

    ds["NDVI"]       = safe_nd(nir, red)
    ds["NDWI"]       = safe_nd(green, nir)
    ds["MNDWI"]      = safe_nd(green, swir1)
    ds["NDMI"]       = safe_nd(nir, swir1)
    ds["water_mask"] = xr.where((ds["MNDWI"] > 0) & (ds["NDVI"] < 0.2), 1.0, 0.0)

    out = nc_path.replace(".nc", "_indices.nc")
    ds.to_netcdf(out)
    ds.close()
    print(f"  Saved: {out}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_years(arg: str) -> List[int]:
    if arg == "all":
        return ALL_YEARS
    if "-" in arg and not arg.startswith("-"):
        parts = arg.split("-")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return list(range(int(parts[0]), int(parts[1]) + 1))
    if arg.isdigit():
        return [int(arg)]
    print(f"ERROR: Invalid year '{arg}'. Use: 2018, 2015-2020, or all")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch annual habitat time series for TWI colony sites"
    )
    parser.add_argument("--year", default=DEFAULT_YEAR,
                        help="Year(s) to process: 2018, 2015-2020, or all")
    parser.add_argument("--colony", default=None,
                        help="Filter to a single colony name (partial match)")
    parser.add_argument("--list-colonies", action="store_true",
                        help="List all colonies and exit")
    parser.add_argument("--check", action="store_true",
                        help="Check job status")
    parser.add_argument("--download", action="store_true",
                        help="Download finished results")
    parser.add_argument("--compute-all", action="store_true",
                        help="Compute indices locally on all downloaded .nc files")
    parser.add_argument("--csv", default=None,
                        help="Path to avianData20102021.csv.gz (overrides default/env var)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Resubmit failed/cancelled jobs")
    parser.add_argument("--state", default=ONLY_STATE,
                        help="Filter colonies by state code (default: LA). Use '' for all states.")
    parser.add_argument("--group-threshold", type=float, default=GROUP_THRESHOLD_DEG,
                        help="Max centroid-to-centroid distance (degrees) to merge colonies into one job")
    parser.add_argument("--rebuild-registry", action="store_true",
                        help="Delete cached colony_registry.json and rebuild from CSV")
    parser.add_argument("--server-check", action="store_true",
                        help="List ALL jobs on the CDSE server (not just local JSON)")
    parser.add_argument("--server-restart-errored", action="store_true",
                        help="List all server jobs and restart any that errored")
    parser.add_argument("--server-resubmit-errored", action="store_true",
                        help="Resubmit errored server jobs as fresh new jobs")
    parser.add_argument("--server-sync", action="store_true",
                        help="Pull all server jobs into local colony_jobs.json")
    args = parser.parse_args()

    # -- Colony registry --
    global AVIAN_CSV
    if args.csv:
        AVIAN_CSV = args.csv

    if args.rebuild_registry and os.path.exists(COLONY_REGISTRY):
        os.remove(COLONY_REGISTRY)
        print("Deleted cached registry — rebuilding ...")

    state_filter = args.state if args.state else None
    colonies = get_colony_registry(
        state_filter=state_filter,
        threshold_deg=args.group_threshold,
    )

    if args.colony:
        colonies = [c for c in colonies if args.colony.lower() in c["name"].lower()]
        if not colonies:
            print(f"No colonies matching '{args.colony}'")
            sys.exit(1)

    if args.list_colonies:
        n_members = sum(len(g.get("members", [g])) for g in colonies)
        print(f"\n{len(colonies)} job groups covering {n_members} colonies:\n")
        for g in sorted(colonies, key=lambda x: x["name"]):
            members = g.get("members", [g])
            if len(members) == 1:
                print(f"  {g['name']:45s}  lat={g['lat']:.4f}  lon={g['lon']:.4f}")
            else:
                names = ", ".join(m["name"] for m in members[1:])
                print(f"  {g['name']:45s}  +{len(members)-1} nearby: {names}")
        return

    # -- Local-only operations --
    if args.compute_all:
        nc_files = glob.glob(os.path.join(OUTPUT_DIR, "**", "*.nc"), recursive=True)
        nc_files = [f for f in nc_files if "_indices" not in f]
        print(f"Computing indices for {len(nc_files)} files ...")
        for f in nc_files:
            compute_indices_locally(f)
        return

    # -- Connect --
    conn = connect_cdse()

    if args.server_sync:
        server_sync(conn)
        return

    if args.server_check or args.server_restart_errored or args.server_resubmit_errored:
        server_check_and_restart(conn, restart_errored=args.server_restart_errored or args.server_resubmit_errored)
        return

    if args.check:
        check_jobs(conn, colony_filter=args.colony, year_filter=None)
        return

    if args.download:
        download_results(conn, colony_filter=args.colony)
        return

    if args.retry_failed:
        jobs = load_jobs()
        failed = [j for j in jobs if j.get("status") in ("error", "canceled")]
        if not failed:
            print("No failed jobs.")
            return
        for j in failed:
            j["status"] = None
        save_jobs(jobs)
        years = list(set(j["year"] for j in failed))
        failed_slugs = set(j["slug"] for j in failed)
        failed_colonies = [c for c in colonies if c["slug"] in failed_slugs]
        submit_jobs(conn, failed_colonies, years)
        return

    # -- Submit --
    years = parse_years(args.year)

    n_members = sum(len(g.get("members", [g])) for g in colonies)
    print("=" * 72)
    print("  Colony Habitat Time Series — TWI Avian Monitoring Sites")
    print("=" * 72)
    print(f"  State     : {state_filter or 'all'}")
    print(f"  Colonies  : {n_members} -> {len(colonies)} job groups "
          f"(threshold={args.group_threshold} deg)")
    print(f"  Years     : {years[0]}-{years[-1]}  ({len(years)} years)")
    print(f"  Landsat   : {years[0]}-{SENTINEL_FROM - 1}")
    print(f"  Sentinel-2: {SENTINEL_FROM}-{years[-1]}")
    print(f"  Total jobs: ~{len(colonies) * len(years)}")
    print(f"  Output    : {OUTPUT_DIR}/{{group_slug}}/")
    print("=" * 72)

    submit_jobs(conn, colonies, years)

    print(f"\n  Next steps:")
    print(f"    python fetch_colony_timeseries.py --check")
    print(f"    python fetch_colony_timeseries.py --download")


if __name__ == "__main__":
    main()
