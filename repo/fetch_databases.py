"""
fetch_databases.py

Downloads the TWI colonial waterbird annotation databases and processed
data exports from the twi-aviandata S3 bucket.

Two .accdb files cover the full 2010-2023 monitoring program:
  - Colibri2010-2021CWBColonies_2Jan2023.accdb  (2010-2021, Gulf Coast wide)
  - LACWB_2022-2023.accdb                        (Louisiana 2022-2023)

Processed CSV exports from the same databases (faster for analysis):
  - avianData20102021.csv.gz
  - avianmonitoring_2010-2021_Nulls.csv.gz
  - avianmonitoring_2022-2023_Nulls.csv.gz
  - avianmonitoring_2022-2023_Nulls.xlsx
  - Colibri2010-21ColonyTotalsMayJuneCombined_8Nov22.xlsx
  - UsedPhotos2010-2021.xlsx                     <- photo-to-annotation linkage

The .accdb files require Microsoft Access or pyodbc with the Access ODBC driver
to open. The CSVs can be read directly with pandas.

Output:
    data/databases/              <- .accdb files
    data/databases/processed/    <- CSV and Excel exports

Usage:
    python fetch_databases.py
    python fetch_databases.py --csvs-only    # skip large .accdb files
    python fetch_databases.py --no-skip      # re-download existing files
"""

import argparse
import os
import requests

BASE_URL = "https://twi-aviandata.s3.amazonaws.com"
DOTTING_INFO_PATH = "avian_monitoring/dotting_information"
PROCESSED_PATH = f"{DOTTING_INFO_PATH}/processed_data"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "databases")
PROCESSED_DIR = os.path.join(OUTPUT_DIR, "processed")

# Primary annotation databases — full relational structure with photo linkage
DATABASES = [
    {
        "filename": "Colibri2010-2021CWBColonies_2Jan2023.accdb",
        "path": DOTTING_INFO_PATH,
        "description": "Complete Gulf Coast colonial waterbird annotations 2010-2021",
        "years": "2010-2021",
    },
    {
        "filename": "LACWB_2022-2023.accdb",
        "path": DOTTING_INFO_PATH,
        "description": "Louisiana colonial waterbird annotations 2022-2023",
        "years": "2022-2023",
    },
]

# Processed exports — flattened from the .accdb databases, faster for analysis
PROCESSED_FILES = [
    {
        "filename": "avianData20102021.csv.gz",
        "description": "Full observation records 2010-2021, compressed CSV",
    },
    {
        "filename": "avianmonitoring_2010-2021_Nulls.csv.gz",
        "description": "2010-2021 records including null-count surveys, compressed CSV",
    },
    {
        "filename": "avianmonitoring_2010-2021_Nulls.xlsx",
        "description": "2010-2021 records including null-count surveys, Excel",
    },
    {
        "filename": "avianmonitoring_2022-2023_Nulls.csv.gz",
        "description": "2022-2023 records including null-count surveys, compressed CSV",
    },
    {
        "filename": "avianmonitoring_2022-2023_Nulls.xlsx",
        "description": "2022-2023 records including null-count surveys, Excel",
    },
    {
        "filename": "SummaryFileGenerated.xlsx",
        "description": "Colony totals summary (2010-2021)",
    },
    {
        "filename": "SummaryFileGenerated2023.xlsx",
        "description": "Colony totals summary (2022-2023)",
    },
    {
        "filename": "Colibri2010-21ColonyTotalsMayJuneCombined_8Nov22.xlsx",
        "path_override": DOTTING_INFO_PATH,
        "description": "Combined May/June colony totals per species 2010-2021",
    },
    {
        "filename": "UsedPhotos2010-2021.xlsx",
        "description": "Index of every photo used in dotting 2010-2021 — links photo to colony/date/camera",
    },
]


def download_file(url, out_path, skip_existing=True):
    if skip_existing and os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1_000_000
        print(f"  skipped (exists, {size_mb:.1f} MB): {os.path.basename(out_path)}")
        return "skipped"

    try:
        resp = requests.get(url, timeout=300, stream=True)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)

        size_mb = downloaded / 1_000_000
        print(f"  fetched ({size_mb:.1f} MB): {os.path.basename(out_path)}")
        return "fetched"

    except requests.HTTPError as e:
        print(f"  ERROR {e.response.status_code}: {url}")
        return f"error_{e.response.status_code}"
    except Exception as e:
        print(f"  ERROR: {url} — {e}")
        return f"error_{e}"


def fetch_databases(skip_existing=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Fetching annotation databases (.accdb)...\n")

    for db in DATABASES:
        url = f"{BASE_URL}/{db['path']}/{db['filename']}"
        out_path = os.path.join(OUTPUT_DIR, db["filename"])
        print(f"{db['filename']}")
        print(f"  {db['description']} ({db['years']})")
        download_file(url, out_path, skip_existing)


def fetch_processed(skip_existing=True):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("\nFetching processed data exports...\n")

    for entry in PROCESSED_FILES:
        path = entry.get("path_override", PROCESSED_PATH)
        url = f"{BASE_URL}/{path}/{entry['filename']}"
        out_path = os.path.join(
            OUTPUT_DIR if "path_override" in entry else PROCESSED_DIR,
            entry["filename"],
        )
        print(f"{entry['filename']}")
        print(f"  {entry['description']}")
        download_file(url, out_path, skip_existing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch TWI annotation databases and processed exports")
    parser.add_argument("--csvs-only", action="store_true", help="Skip .accdb files, fetch processed exports only")
    parser.add_argument("--no-skip", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    skip = not args.no_skip

    if not args.csvs_only:
        fetch_databases(skip_existing=skip)

    fetch_processed(skip_existing=skip)
