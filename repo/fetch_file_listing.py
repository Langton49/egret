"""
fetch_file_listing.py

Downloads and caches the TWI S3 directory index (file_listing.json).
This file maps every path in the twi-aviandata bucket to its contents —
5,889 directory entries covering all survey years, regions, and colonies.

Cached locally so the rest of the repo scripts can navigate the full dataset
without re-fetching. Also extracts useful sub-indexes:

  - photo_index.json      <- all raw JPG paths by year/region/colony
  - dotted_index.json     <- all dotted image paths by year
  - database_index.json   <- all .mdb/.accdb/.csv paths

Output:
    data/file_listing.json        <- full cached directory tree
    data/indexes/photo_index.json
    data/indexes/dotted_index.json
    data/indexes/database_index.json

Usage:
    python fetch_file_listing.py
    python fetch_file_listing.py --no-skip   # re-fetch even if cached
"""

import argparse
import json
import os
import requests

FILE_LISTING_URL = "https://twi-aviandata.s3.amazonaws.com/file_listing.json"
BASE_URL = "https://twi-aviandata.s3.amazonaws.com"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_DIR = os.path.join(OUTPUT_DIR, "indexes")
LISTING_PATH = os.path.join(OUTPUT_DIR, "file_listing.json")


def fetch_listing(skip_existing=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if skip_existing and os.path.exists(LISTING_PATH):
        print(f"Using cached file_listing.json ({os.path.getsize(LISTING_PATH) / 1000:.0f} KB)")
        with open(LISTING_PATH) as f:
            return json.load(f)

    print(f"Fetching: {FILE_LISTING_URL}")
    resp = requests.get(FILE_LISTING_URL, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    with open(LISTING_PATH, "w") as f:
        json.dump(data, f)

    print(f"Saved: {LISTING_PATH} ({len(data)} directory entries)")
    return data


def build_photo_index(listing):
    """
    Extract all raw JPG paths under /HighResolutionImages/ and
    /avian_monitoring/high_resolution_photos/, grouped by year/region/colony.
    """
    index = {"HighResolutionImages": {}, "avian_monitoring": {}}

    for path, entry in listing.items():
        files = entry.get("files", [])
        jpg_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg"))]
        if not jpg_files:
            continue

        if path.startswith("/HighResolutionImages/"):
            parts = path.strip("/").split("/")
            # /HighResolutionImages/{year}/{month?}/{date?}/...
            year = parts[1] if len(parts) > 1 else "unknown"
            index["HighResolutionImages"].setdefault(year, {})
            index["HighResolutionImages"][year][path] = {
                "url_base": f"{BASE_URL}{path}",
                "file_count": len(jpg_files),
                "sample": jpg_files[:3],
            }

        elif path.startswith("/avian_monitoring/high_resolution_photos/"):
            parts = path.strip("/").split("/")
            # /avian_monitoring/high_resolution_photos/{year}/{region}/{colony}/
            year = parts[3] if len(parts) > 3 else "unknown"
            index["avian_monitoring"].setdefault(year, {})
            index["avian_monitoring"][year][path] = {
                "url_base": f"{BASE_URL}{path}",
                "file_count": len(jpg_files),
                "sample": jpg_files[:3],
            }

    return index


def build_dotted_index(listing):
    """Extract all dotted image paths under /DottedImages/."""
    index = {}

    for path, entry in listing.items():
        if not path.startswith("/DottedImages/"):
            continue

        files = entry.get("files", [])
        jpg_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg"))]
        if not jpg_files:
            continue

        parts = path.strip("/").split("/")
        year_hint = next((p for p in parts if p[:4].isdigit()), "unknown")
        index.setdefault(year_hint, {})
        index[year_hint][path] = {
            "url_base": f"{BASE_URL}{path}",
            "file_count": len(jpg_files),
            "sample": jpg_files[:3],
        }

    return index


def build_database_index(listing):
    """Extract all .mdb, .accdb, .csv, .gz, .xlsx paths."""
    DB_EXTS = {".mdb", ".accdb", ".csv", ".gz", ".xlsx"}
    index = []

    for path, entry in listing.items():
        files = entry.get("files", [])
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in DB_EXTS:
                index.append({
                    "filename": f,
                    "directory": path,
                    "url": f"{BASE_URL}{path}/{f}",
                    "extension": ext,
                })

    index.sort(key=lambda x: (x["extension"], x["filename"]))
    return index


def build_indexes(listing):
    os.makedirs(INDEX_DIR, exist_ok=True)

    photo_index = build_photo_index(listing)
    dotted_index = build_dotted_index(listing)
    database_index = build_database_index(listing)

    photo_path = os.path.join(INDEX_DIR, "photo_index.json")
    with open(photo_path, "w") as f:
        json.dump(photo_index, f, indent=2)

    dotted_path = os.path.join(INDEX_DIR, "dotted_index.json")
    with open(dotted_path, "w") as f:
        json.dump(dotted_index, f, indent=2)

    db_path = os.path.join(INDEX_DIR, "database_index.json")
    with open(db_path, "w") as f:
        json.dump(database_index, f, indent=2)

    # Summary
    total_photos = sum(
        v["file_count"]
        for year in photo_index["HighResolutionImages"].values()
        for v in year.values()
    )
    print(f"\nIndexes written to {INDEX_DIR}/")
    print(f"  photo_index.json    — {total_photos:,} raw JPG files across HighResolutionImages/")
    print(f"  dotted_index.json   — {sum(v['file_count'] for y in dotted_index.values() for v in y.values()):,} dotted image files")
    print(f"  database_index.json — {len(database_index)} database/data files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and index TWI S3 file listing")
    parser.add_argument("--no-skip", action="store_true", help="Re-fetch even if cached")
    parser.add_argument("--listing-only", action="store_true", help="Only cache file_listing.json, skip index building")
    args = parser.parse_args()

    listing = fetch_listing(skip_existing=not args.no_skip)

    if not args.listing_only:
        build_indexes(listing)
