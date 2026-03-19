#!/usr/bin/env python3
"""
=============================================================================
  SWIR Band Patch — Sentinel-2 swir16 / swir22
=============================================================================

The original S2 jobs returned swir16 and swir22 as all-NaN, most likely
because those 20m bands failed to resample to the 10m output grid silently.

This script:
  1. --submit   Submit SWIR-only jobs for every S2 year/colony group,
                requesting swir16 + swir22 at native 20m resolution
                (no UDF, no resampling — raw band composites only).
  2. --check    Poll job statuses on CDSE.
  3. --download Download finished results and patch the existing NC files,
                resampling swir16/swir22 from 20m to match the 10m grid
                of the original NC file before writing.

Usage:
    python training_two/fetch_swir_patch.py --submit
    python training_two/fetch_swir_patch.py --check
    python training_two/fetch_swir_patch.py --download
    python training_two/fetch_swir_patch.py --download --dry-run
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths (all relative to this script)
# ---------------------------------------------------------------------------
_HERE       = Path(__file__).parent
_COL_DIR    = _HERE / "satellite_timeseries" / "colonies"
_REGISTRY   = _COL_DIR / "colony_registry.json"
_JOBS_FILE  = _HERE / "swir_patch_jobs.json"

OPENEO_BACKEND = "https://openeo.dataspace.copernicus.eu"
S2_STAC        = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a"
SENTINEL_FROM  = 2017
MAX_CLOUD      = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_registry():
    with open(_REGISTRY) as f:
        raw = json.load(f)
    return raw if isinstance(raw, list) else list(raw.values())


def load_jobs():
    if _JOBS_FILE.exists():
        with open(_JOBS_FILE) as f:
            return json.load(f)
    return []


def save_jobs(jobs):
    with open(_JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2, default=str)


def connect():
    import openeo
    print(f"Connecting to {OPENEO_BACKEND} ...")
    conn = openeo.connect(OPENEO_BACKEND)
    conn.authenticate_oidc()
    print(f"Authenticated as: {conn.describe_account().get('user_id', '?')}")
    return conn


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

def build_swir_graph(conn, bbox, year):
    """
    Load only swir16 + swir22 from AWS Earth Search at native 20m resolution.
    No UDF, no resampling — plain annual median composite.
    """
    import openeo

    s2 = conn.load_stac(
        url=S2_STAC,
        spatial_extent={
            "west": bbox[0], "south": bbox[1],
            "east": bbox[2], "north": bbox[3],
        },
        temporal_extent=[f"{year}-01-01", f"{year}-12-31"],
        bands=["swir16", "swir22"],
        properties={"eo:cloud_cover": lambda cc: cc <= MAX_CLOUD},
    )

    return s2.aggregate_temporal_period(period="year", reducer="median")


def submit(conn):
    registry = load_registry()
    jobs     = load_jobs()
    existing = {j["job_key"] for j in jobs}

    submitted = skipped = errored = 0

    for group in registry:
        slug = group["slug"]
        bbox = group["bbox"]

        # Collect S2 years that have an existing NC file to patch
        nc_dir = _COL_DIR / slug
        if not nc_dir.exists():
            continue

        s2_years = sorted(
            int(p.stem.split("_")[-2])
            for p in nc_dir.glob(f"{slug}_*_s2.nc")
        )

        for year in s2_years:
            key = f"{slug}_{year}_swir"

            if key in existing:
                print(f"  [skip]  {key}")
                skipped += 1
                continue

            print(f"  [submit] {key}")
            try:
                cube = build_swir_graph(conn, bbox, year)
                job  = cube.create_job(
                    title=f"swir_patch_{slug}_{year}",
                    out_format="NetCDF",
                    job_options={
                        "executor-memory": "1g",
                        "executor-cores":  "1",
                        "soft-errors":     "false",
                    },
                )
                job.start_job()

                jobs.append({
                    "job_key":      key,
                    "slug":         slug,
                    "year":         year,
                    "bbox":         bbox,
                    "job_id":       job.job_id,
                    "status":       "queued",
                    "target_nc":    str(nc_dir / f"{slug}_{year}_s2.nc"),
                    "swir_nc":      str(nc_dir / f"{slug}_{year}_swir_patch.nc"),
                    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                existing.add(key)
                save_jobs(jobs)

                submitted += 1
                time.sleep(5)

            except Exception as e:
                print(f"  [error]  {key}: {e}")
                errored += 1

    print(f"\nSubmitted: {submitted} | Skipped: {skipped} | Errors: {errored}")


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

def check(conn):
    jobs = load_jobs()
    if not jobs:
        print("No SWIR patch jobs found. Run --submit first.")
        return

    print(f"\nChecking {len(jobs)} jobs ...\n")
    for j in jobs:
        try:
            status = conn.job(j["job_id"]).describe_job().get("status", "?")
            j["status"] = status
            icon = {"finished": "ok", "running": ">>", "queued": "..",
                    "error": "!!", "canceled": "!!"}.get(status, "?")
            print(f"  [{icon}] {j['job_key']:50s} {status}")
        except Exception as e:
            print(f"  [!] {j['job_key']:50s} {e}")

    save_jobs(jobs)

    counts = {}
    for j in jobs:
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    print("\nSummary:", "  ".join(f"{s}: {n}" for s, n in sorted(counts.items())))


# ---------------------------------------------------------------------------
# Sync from server
# ---------------------------------------------------------------------------

def sync(conn):
    """
    Pull all SWIR patch jobs from the CDSE server into swir_patch_jobs.json.
    Matches jobs by title prefix 'swir_patch_' — no reliance on local logs.
    """
    registry = load_registry()
    slug_to_group = {g["slug"]: g for g in registry}

    print("\nFetching all jobs from CDSE server ...")
    all_server_jobs = []
    page = 0
    while True:
        batch = conn.get("/jobs", params={"limit": 100, "page": page}).json()
        chunk = batch.get("jobs", [])
        if not chunk:
            break
        all_server_jobs.extend(chunk)
        print(f"  ... {len(all_server_jobs)} jobs fetched")
        if not any(lnk.get("rel") == "next" for lnk in batch.get("links", [])):
            break
        page += 1

    swir_jobs = [j for j in all_server_jobs if j.get("title", "").startswith("swir_patch_")]
    print(f"  Found {len(swir_jobs)} swir_patch jobs on server\n")

    local_jobs = load_jobs()
    local_by_key = {j["job_key"]: j for j in local_jobs}

    added = updated = skipped = 0

    for sj in swir_jobs:
        job_id = sj["id"]
        status = sj.get("status", "unknown")
        title  = sj.get("title", "")

        # Parse title: swir_patch_{slug}_{year}
        try:
            rest  = title[len("swir_patch_"):]
            parts = rest.rsplit("_", 1)
            year  = int(parts[-1])
            slug  = parts[0]
        except Exception:
            print(f"  [unparsed] {title}")
            skipped += 1
            continue

        group = slug_to_group.get(slug)
        if not group:
            print(f"  [unknown slug] {slug}")
            skipped += 1
            continue

        key       = f"{slug}_{year}_swir"
        nc_dir    = _COL_DIR / slug
        target_nc = nc_dir / f"{slug}_{year}_s2.nc"
        swir_nc   = nc_dir / f"{slug}_{year}_swir_patch.nc"

        if key in local_by_key:
            local_by_key[key]["status"] = status
            local_by_key[key]["job_id"] = job_id  # update in case of resubmission
            updated += 1
        else:
            entry = {
                "job_key":      key,
                "slug":         slug,
                "year":         year,
                "bbox":         group["bbox"],
                "job_id":       job_id,
                "status":       status,
                "target_nc":    str(target_nc),
                "swir_nc":      str(swir_nc),
                "submitted_at": sj.get("created", ""),
            }
            local_by_key[key] = entry
            added += 1

        icon = {"finished": "ok", "running": ">>", "queued": "..",
                "error": "!!", "canceled": "!!"}.get(status, "?")
        print(f"  [{icon}] {key:55s} {status}")

    save_jobs(list(local_by_key.values()))

    counts = {}
    for j in local_by_key.values():
        counts[j["status"]] = counts.get(j["status"], 0) + 1

    print(f"\nAdded: {added} | Updated: {updated} | Skipped: {skipped}")
    print("Status breakdown:", "  ".join(f"{s}: {n}" for s, n in sorted(counts.items())))
    print(f"\nRun --download to fetch finished jobs.")


# ---------------------------------------------------------------------------
# Download + patch
# ---------------------------------------------------------------------------

def resample_to_grid(swir_ds, target_ds):
    """
    Bilinearly resample swir16/swir22 from the patch NC onto the x/y grid
    of the target NC.  Uses scipy.interpolate.RegularGridInterpolator.
    Both files use the same CRS so coordinate spaces match directly.
    """
    from scipy.interpolate import RegularGridInterpolator

    tx = target_ds["x"].values
    ty = target_ds["y"].values

    resampled = {}
    for band in ("swir16", "swir22"):
        if band not in swir_ds.data_vars:
            continue

        da = swir_ds[band]
        if "t" in da.dims:
            da = da.squeeze("t")

        sx = swir_ds["x"].values
        sy = swir_ds["y"].values
        data = da.values.astype("float32")

        # RegularGridInterpolator expects ascending axes
        if sy[0] > sy[-1]:
            sy   = sy[::-1]
            data = data[::-1, :]
        if sx[0] > sx[-1]:
            sx   = sx[::-1]
            data = data[:, ::-1]

        interp = RegularGridInterpolator(
            (sy, sx), data,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        # Build query grid matching target x/y
        yy, xx = np.meshgrid(ty, tx, indexing="ij")
        resampled[band] = interp(np.stack([yy.ravel(), xx.ravel()], axis=1)).reshape(yy.shape).astype("float32")

    return resampled


def patch_nc(target_path: Path, swir_path: Path, dry_run=False):
    """Write swir16/swir22 into the existing target NC file."""
    import xarray as xr

    print(f"  Patching {target_path.name} ...")

    # Load both datasets fully into memory so all file handles are released
    # before we attempt to overwrite the target (required on Windows).
    with xr.open_dataset(target_path) as target_ds:
        target_ds.load()
        with xr.open_dataset(swir_path) as swir_ds:
            swir_ds.load()
            resampled = resample_to_grid(swir_ds, target_ds)
    # Both file handles are now closed; re-open target read-only to build patched ds
    with xr.open_dataset(target_path) as target_ds:
        target_ds.load()

        if not resampled:
            print(f"    [warn] no swir bands found in patch file")
            return

        for band, arr in resampled.items():
            n_valid = int(np.sum(~np.isnan(arr)))
            print(f"    {band}: {n_valid} valid pixels after resample")

        if dry_run:
            print(f"    [dry-run] would write {list(resampled.keys())} to {target_path.name}")
            return

        new_vars = {}
        for band, arr in resampled.items():
            ref_var = next(v for v in target_ds.data_vars if v != "crs")
            ref_da  = target_ds[ref_var]

            if "t" in ref_da.dims:
                new_vars[band] = xr.DataArray(
                    arr[np.newaxis, :, :],
                    dims=("t", "y", "x"),
                    coords={"t": ref_da.coords["t"], "y": target_ds["y"], "x": target_ds["x"]},
                    attrs={"long_name": band, "units": "reflectance"},
                )
            else:
                new_vars[band] = xr.DataArray(
                    arr,
                    dims=("y", "x"),
                    coords={"y": target_ds["y"], "x": target_ds["x"]},
                    attrs={"long_name": band, "units": "reflectance"},
                )

        patched_ds = target_ds.assign(new_vars)

    # All file handles are now closed — safe to write and replace on Windows
    tmp_path = target_path.with_suffix(".patching.nc")
    patched_ds.to_netcdf(tmp_path)
    patched_ds.close()

    tmp_path.replace(target_path)
    print(f"    [saved] {target_path.name}")


def download(conn, dry_run=False):
    jobs = load_jobs()
    if not jobs:
        print("No SWIR patch jobs found. Run --submit first.")
        return

    patched = skipped = failed = 0

    for j in jobs:
        swir_path   = Path(j["swir_nc"])
        target_path = Path(j["target_nc"])

        # Check target exists
        if not target_path.exists():
            print(f"  [skip]  {j['job_key']} — target NC not found: {target_path.name}")
            skipped += 1
            continue

        # Download if not already done
        if not swir_path.exists():
            try:
                job    = conn.job(j["job_id"])
                status = job.describe_job().get("status", "?")

                if status != "finished":
                    print(f"  [skip]  {j['job_key']} — {status}")
                    skipped += 1
                    continue

                print(f"  [downloading] {swir_path.name} ...")
                job.get_results().download_file(str(swir_path))
                j["status"] = "finished"
                j["downloaded"] = True
                save_jobs(jobs)
                print(f"  [saved] {swir_path.name} ({swir_path.stat().st_size / 1e6:.1f} MB)")

            except Exception as e:
                print(f"  [error] {j['job_key']} download: {e}")
                failed += 1
                continue

        # Patch the target NC
        try:
            patch_nc(target_path, swir_path, dry_run=dry_run)
            if not dry_run:
                j["patched"] = True
                save_jobs(jobs)
            patched += 1
        except Exception as e:
            print(f"  [error] {j['job_key']} patch: {e}")
            failed += 1

    print(f"\nPatched: {patched} | Skipped: {skipped} | Failed: {failed}")
    if not dry_run and patched:
        print("\nRe-run extract_colony_centroids.py --force to rebuild the feature CSV.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--submit",   action="store_true", help="Submit SWIR-only OpenEO jobs")
    group.add_argument("--sync",     action="store_true", help="Sync job list from CDSE server (no local log needed)")
    group.add_argument("--check",    action="store_true", help="Poll statuses of locally tracked jobs")
    group.add_argument("--download", action="store_true", help="Download finished results and patch NC files")
    parser.add_argument("--dry-run", action="store_true", help="With --download: show what would be patched without writing")
    args = parser.parse_args()

    conn = connect()

    if args.submit:
        submit(conn)
    elif args.sync:
        sync(conn)
    elif args.check:
        check(conn)
    elif args.download:
        download(conn, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
