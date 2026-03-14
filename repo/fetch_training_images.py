"""
fetch_training_images.py

Discovers and downloads raw JPG frames for YOLO training by querying the
TWI AvianData Feature Service for records with a populated `uid` field.

The `uid` field follows this pattern:
    avian_monitoring/high_resolution_photos/{year}/{region}/{colony}/{filename}.jpg#{speciescode}

This script:
1. Queries the Feature Service for all records with non-null uid values
2. Parses uid values to extract photo paths and associated species codes
3. Probes candidate base URLs to find where photos are served from
4. Downloads accessible photos and saves a uid_index.json for reference
   regardless of whether photos are downloadable

Output:
    data/training_images/uid_index.json         <- full index of all uid paths
    data/training_images/{colony}/{year}/       <- downloaded JPGs (if accessible)

Usage:
    python fetch_training_images.py --probe-only   # discover paths, no download
    python fetch_training_images.py                # discover + download
    python fetch_training_images.py --colony "Queen Bess Island" --year 2018
"""

import argparse
import json
import os
import urllib.parse
import requests

FEATURE_SERVICE_URL = (
    "https://gis.thewaterinstitute.org/twimanager/rest/services/Hosted"
    "/AvianData_App_Visualization_WFL1/FeatureServer/12"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "training_images")
PAGE_SIZE = 1000

# Candidate base URLs for photo assets — probed in order until one resolves.
# twi-aviandata is the full archive bucket (confirmed via file_listing.json).
# twi-avian-2024 is the devDays sampler bucket — kept as fallback.
CANDIDATE_BASE_URLS = [
    "https://twi-aviandata.s3.amazonaws.com",
    "https://twi-avian-2024.s3.us-east-1.amazonaws.com/public",
    "https://twi-avian-2024.s3.us-east-1.amazonaws.com",
    "https://gis.thewaterinstitute.org",
]


def query_uid_records(where="1=1"):
    """Fetch all Feature Service records that have a non-null uid field."""
    uid_where = f"uid IS NOT NULL AND uid <> ''"
    full_where = f"({uid_where}) AND ({where})" if where != "1=1" else uid_where

    count_resp = requests.get(
        f"{FEATURE_SERVICE_URL}/query",
        params={"where": full_where, "returnCountOnly": "true", "f": "json"},
        timeout=30,
    )
    count_resp.raise_for_status()
    total = count_resp.json().get("count", 0)
    print(f"Records with uid: {total}")

    features = []
    offset = 0
    while offset < total:
        resp = requests.get(
            f"{FEATURE_SERVICE_URL}/query",
            params={
                "where": full_where,
                "outFields": "uid,speciescode,speciesname,colonyname,georegion,year,date",
                "resultOffset": offset,
                "resultRecordCount": PAGE_SIZE,
                "f": "geojson",
            },
            timeout=60,
        )
        resp.raise_for_status()
        batch = resp.json().get("features", [])
        features.extend(batch)
        print(f"  {offset + len(batch)}/{total}")
        offset += PAGE_SIZE

    return features


def parse_uid(uid_value):
    """
    Parse a uid string into its components.

    Input:  avian_monitoring/high_resolution_photos/2018/BaratariaBay/Queen Bess Island/20May18Camera1-Card1-3815.jpg#BRPE
    Output: {
        "path": "avian_monitoring/high_resolution_photos/2018/.../filename.jpg",
        "filename": "20May18Camera1-Card1-3815.jpg",
        "species_tag": "BRPE"
    }
    """
    if not uid_value:
        return None

    # Split species tag from path
    parts = uid_value.split("#")
    path = parts[0].strip()
    species_tag = parts[1].strip() if len(parts) > 1 else None
    filename = path.split("/")[-1]

    return {
        "path": path,
        "filename": filename,
        "species_tag": species_tag,
    }


def build_uid_index(features):
    """
    Build a deduplicated index of photo paths from Feature Service records.
    Groups by colony/year for easy lookup.
    """
    index = {}  # path -> {species_codes, colony, year, georegion}

    for feat in features:
        props = feat.get("properties", {})
        uid_raw = props.get("uid")
        if not uid_raw:
            continue

        parsed = parse_uid(uid_raw)
        if not parsed:
            continue

        path = parsed["path"]
        if path not in index:
            index[path] = {
                "path": path,
                "filename": parsed["filename"],
                "species_codes": [],
                "colony": props.get("colonyname"),
                "georegion": props.get("georegion"),
                "year": props.get("year"),
            }

        tag = parsed["species_tag"] or props.get("speciescode")
        if tag and tag not in index[path]["species_codes"]:
            index[path]["species_codes"].append(tag)

    return list(index.values())


def probe_base_url(sample_path):
    """Try each candidate base URL and return the first that serves the file."""
    for base in CANDIDATE_BASE_URLS:
        url = f"{base}/{sample_path}"
        try:
            resp = requests.head(url, timeout=10)
            if resp.status_code == 200:
                print(f"  Resolved base URL: {base}")
                return base
            print(f"  {resp.status_code}: {base}")
        except Exception as e:
            print(f"  ERROR: {base} — {e}")
    return None


def download_image(base_url, path, out_path, skip_existing=True):
    if skip_existing and os.path.exists(out_path):
        return "skipped"

    url = f"{base_url}/{path}"
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return "fetched"
    except requests.HTTPError as e:
        return f"error_{e.response.status_code}"
    except Exception as e:
        return f"error_{e}"


def build_where(colony=None, year=None):
    clauses = []
    if colony:
        safe = colony.replace("'", "''")
        clauses.append(f"colonyname='{safe}'")
    if year:
        clauses.append(f"year={year}")
    return " AND ".join(clauses) if clauses else "1=1"


def run(probe_only=False, colony=None, year=None, skip_existing=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    where = build_where(colony=colony, year=year)
    print(f"Querying Feature Service for uid records...")
    print(f"Filter: {where}\n")

    features = query_uid_records(where=where)
    if not features:
        print("No records with uid found.")
        return

    uid_index = build_uid_index(features)
    print(f"\nUnique photo paths: {len(uid_index)}")

    # Save uid index regardless of download outcome
    index_path = os.path.join(OUTPUT_DIR, "uid_index.json")
    with open(index_path, "w") as f:
        json.dump(uid_index, f, indent=2)
    print(f"uid index saved: {index_path}")

    # Coverage summary
    by_colony_year = {}
    for entry in uid_index:
        key = f"{entry['colony']} / {entry['year']}"
        by_colony_year[key] = by_colony_year.get(key, 0) + 1
    print("\nPhoto coverage:")
    for key, count in sorted(by_colony_year.items()):
        print(f"  {key}: {count} frames")

    if probe_only:
        print("\nProbe-only mode — skipping download.")
        return

    # Probe base URL using first entry
    print("\nProbing base URLs to locate photo server...")
    base_url = probe_base_url(uid_index[0]["path"])

    if not base_url:
        print(
            "\nCould not resolve photos from any candidate base URL.\n"
            "The raw photos are likely in a non-public S3 path or internal TWI server.\n"
            "uid_index.json has been saved — share with TWI to confirm access paths."
        )
        return

    # Download
    print(f"\nDownloading {len(uid_index)} images from {base_url}...\n")
    fetched = skipped = errors = 0

    for entry in uid_index:
        colony_label = (entry["colony"] or "unknown").replace(" ", "_")
        year_label = str(entry["year"] or "unknown")
        out_path = os.path.join(OUTPUT_DIR, colony_label, year_label, entry["filename"])

        result = download_image(base_url, entry["path"], out_path, skip_existing)
        if result == "fetched":
            fetched += 1
            print(f"  fetched: {entry['filename']}")
        elif result == "skipped":
            skipped += 1
        else:
            errors += 1
            print(f"  FAILED: {entry['filename']} ({result})")

    print(f"\nDone. Fetched: {fetched} | Skipped: {skipped} | Errors: {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch YOLO training images via TWI Feature Service uid field")
    parser.add_argument("--probe-only", action="store_true", help="Build uid index only, no download")
    parser.add_argument("--colony", help="Filter by colony name")
    parser.add_argument("--year", type=int, help="Filter by survey year")
    parser.add_argument("--no-skip", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    run(
        probe_only=args.probe_only,
        colony=args.colony,
        year=args.year,
        skip_existing=not args.no_skip,
    )
