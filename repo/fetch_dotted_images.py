"""
fetch_dotted_images.py

Downloads all dotted screen capture images from the TWI S3 bucket.
These are the annotated training images for the YOLO pipeline — each
screen capture shows an aerial colony photo with colored dot markers
placed on every bird and nest by expert ornithologists.

22,575 images across 5 survey periods:
    2010-2013 Dotted Images
    Task 1 2015 Waterbird Colony Photo Analysis
    2018 LA Waterbird Colony Photo Analysis
    Task 2 2021 Waterbird Colony Photo Analysis
    2023

Output:
    data/dotted_images/{year}/{colony_path}/{filename}.JPG

Usage:
    python fetch_dotted_images.py                  # fetch all
    python fetch_dotted_images.py --year 2021      # single year
    python fetch_dotted_images.py --dry-run        # show what would be downloaded
    python fetch_dotted_images.py --no-skip        # re-download existing files
"""

import argparse
import json
import os
import requests

BASE_URL = "https://twi-aviandata.s3.amazonaws.com"
LISTING_URL = f"{BASE_URL}/file_listing.json"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "dotted_images")
LISTING_CACHE = os.path.join(os.path.dirname(__file__), "data", "file_listing.json")

# Maps top-level DottedImages subdirectory to survey year(s)
YEAR_MAP = {
    "2010-2013 Dotted Images": ["2010", "2011", "2012", "2013"],
    "Task 1 2015 Waterbird Colony Photo Analysis": ["2015"],
    "2018 LA Waterbird Colony Photo Analysis": ["2018"],
    "Task 2 2021 Waterbird Colony Photo Analysis": ["2021"],
    "2023": ["2023"],
}


def load_listing():
    if os.path.exists(LISTING_CACHE):
        print(f"Using cached file_listing.json")
        with open(LISTING_CACHE) as f:
            return json.load(f)

    print(f"Fetching file listing from S3...")
    resp = requests.get(LISTING_URL, timeout=60)
    resp.raise_for_status()
    listing = resp.json()

    os.makedirs(os.path.dirname(LISTING_CACHE), exist_ok=True)
    with open(LISTING_CACHE, "w") as f:
        json.dump(listing, f)

    return listing


def get_year_for_path(path):
    """Determine survey year from the DottedImages subdirectory name."""
    parts = path.strip("/").split("/")
    if len(parts) < 2:
        return None
    subdir = parts[1]
    for folder, years in YEAR_MAP.items():
        if folder in subdir or subdir in folder:
            # Try to be more specific for 2010-2013
            if len(years) > 1:
                for year in years:
                    if f"/{year}/" in path or path.endswith(f"/{year}"):
                        return year
                return years[0]
            return years[0]
    return None


def collect_images(listing, year_filter=None):
    """Walk file listing and collect all dotted JPG paths."""
    images = []

    for path, entry in listing.items():
        if not path.startswith("/DottedImages"):
            continue

        files = entry.get("files", [])
        jpgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg"))]
        if not jpgs:
            continue

        year = get_year_for_path(path)

        if year_filter and year not in year_filter:
            continue

        for filename in jpgs:
            images.append({
                "s3_path": f"{path}/{filename}",
                "url": f"{BASE_URL}{path}/{filename}",
                "filename": filename,
                "year": year or "unknown",
                "local_subpath": path.replace("/DottedImages/", "").strip("/"),
            })

    return images


def download_image(entry, skip_existing=True):
    out_dir = os.path.join(OUTPUT_DIR, entry["year"], entry["local_subpath"])
    out_path = os.path.join(out_dir, entry["filename"])

    if skip_existing and os.path.exists(out_path):
        return "skipped"

    os.makedirs(out_dir, exist_ok=True)

    try:
        resp = requests.get(entry["url"], timeout=120, stream=True)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        return "fetched"
    except requests.HTTPError as e:
        return f"error_{e.response.status_code}"
    except Exception as e:
        return f"error_{e}"


def run(year_filter=None, dry_run=False, skip_existing=True):
    listing = load_listing()
    images = collect_images(listing, year_filter=year_filter)

    # Summary by year
    by_year = {}
    for img in images:
        by_year[img["year"]] = by_year.get(img["year"], 0) + 1

    print(f"\nDotted images to fetch: {len(images)}")
    for year, count in sorted(by_year.items()):
        print(f"  {year}: {count}")

    if dry_run:
        print("\nDry run — no files downloaded.")
        return

    print()
    fetched = skipped = errors = 0

    for i, entry in enumerate(images, 1):
        result = download_image(entry, skip_existing=skip_existing)

        if result == "fetched":
            fetched += 1
            if fetched % 100 == 0:
                print(f"  [{i}/{len(images)}] fetched: {fetched} | skipped: {skipped} | errors: {errors}")
        elif result == "skipped":
            skipped += 1
        else:
            errors += 1
            print(f"  FAILED [{i}]: {entry['filename']} ({result})")

    print(f"\nDone. Fetched: {fetched} | Skipped: {skipped} | Errors: {errors}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch TWI dotted screen capture images")
    parser.add_argument(
        "--year",
        nargs="+",
        help="Filter by year(s) e.g. --year 2021 or --year 2018 2021",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-download files that already exist",
    )
    args = parser.parse_args()

    run(
        year_filter=args.year,
        dry_run=args.dry_run,
        skip_existing=not args.no_skip,
    )
