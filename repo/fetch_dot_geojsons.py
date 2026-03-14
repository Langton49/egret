"""
fetch_dot_geojsons.py

Fetches individual bird dot annotations from two sources:

1. TWI S3 bucket (public/mosaics/) — per-dot GeoJSON files for known colonies
   with full dot-level detail including screenshot filename, nest status, etc.
   Available for: PepperfishKey 2021, QueenBessIsland 2018/2021,
   NewHarborIsland2/3 2021.

2. TWI Feature Service — dotting point layers for 2022 (Louisiana-wide) and
   2023 (Queen Bess), which are not in the S3 bucket at all.

Output:
    data/dots/{georegion}/{colony}/{year}/        <- S3 dot GeoJSONs
    data/dots/feature_service/2022_LA_all.geojson
    data/dots/feature_service/2023_queen_bess.geojson

Usage:
    python fetch_dot_geojsons.py           # fetch all sources
    python fetch_dot_geojsons.py --s3-only
    python fetch_dot_geojsons.py --fs-only
"""

import argparse
import json
import os
import requests

S3_BASE_URL = "https://twi-avian-2024.s3.us-east-1.amazonaws.com/public/mosaics"
S3_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "dots")
FS_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "dots", "feature_service")

PAGE_SIZE = 1000

# --- S3 dot file inventory ---
# Full inventory of dot files per colony/year confirmed from S3 bucket listing.
S3_COLONIES = {
    "ApalacheeBay/PepperfishKey/2021": [
        "AWPE-Bird_dots.json", "BRPE-Bird_dots.json", "CAEG-Bird_dots.json",
        "DCCO-Bird_dots.json", "GBHE-CNWA_dots.json", "GBHE-CN_dots.json",
        "GBHE-Nest_dots.json", "GREG-Bird_dots.json", "GREG-CNWA_dots.json",
        "GREG-CN_dots.json", "GREG-Nest_dots.json", "LAGU-Bird_dots.json",
        "SNEG-Bird_dots.json", "SNEG-Nest_dots.json", "TRHE-Bird_dots.json",
        "TRHE-Nest_dots.json", "dots_summary.json",
    ],
    "BaratariaBay/QueenBessIsland/2018": [
        "AWPE-Abandoned_dots.json", "AWPE-Bird_dots.json", "BCNH-Nest_dots.json",
        "BRPE-Bird_dots.json", "BRPE-CNWA_dots.json", "BRPE-CN_dots.json",
        "BRPE-Empty_dots.json", "BRPE-WBN_dots.json", "GREG-CNWA_dots.json",
        "GREG-CN_dots.json", "GREG-Nest_dots.json", "LAGU-Bird_dots.json",
        "LAGU-Nest_dots.json", "ROSP-Bird_dots.json", "ROSP-Nest_dots.json",
        "SNEG-Bird_dots.json", "SNEG-Nest_dots.json", "TRHE-Bird_dots.json",
        "TRHE-Nest_dots.json", "WHIB-Bird_dots.json", "WHIB-CN_dots.json",
        "WHIB-Nest_dots.json", "dots_summary.json",
    ],
    "BaratariaBay/QueenBessIsland/2021": [
        "AMAV-Bird_dots.json", "AMOY-Bird_dots.json", "AWPE-Bird_dots.json",
        "BCNH-Bird_dots.json", "BLSK-Bird_dots.json", "BRPE-Abandoned_dots.json",
        "BRPE-Bird_dots.json", "BRPE-CNWA_dots.json", "BRPE-CN_dots.json",
        "BRPE-Empty_dots.json", "BRPE-PBN_dots.json", "BRPE-Territory_dots.json",
        "BRPE-WBN_dots.json", "GBTE-Bird_dots.json", "GREG-Bird_dots.json",
        "GREG-CNWA_dots.json", "GREG-CN_dots.json", "GREG-Nest_dots.json",
        "HERG-Bird_dots.json", "LAGU-Bird_dots.json", "LAGU-Nest_dots.json",
        "ROSP-Bird_dots.json", "ROYT-Bird_dots.json", "ROYT-Nest_dots.json",
        "SATE-Bird_dots.json", "SATE-Nest_dots.json", "WHIB-Bird_dots.json",
        "dots_summary.json",
    ],
    "BretonChandeleurIslands/NewHarborIsland2/2021": [
        "BRPE-Bird_dots.json", "dots_summary.json",
    ],
    "BretonChandeleurIslands/NewHarborIsland3/2021": [
        "BRPE-Abandoned_dots.json", "BRPE-Bird_dots.json", "BRPE-CNWA_dots.json",
        "BRPE-CN_dots.json", "BRPE-PBN_dots.json", "BRPE-WBN_dots.json",
        "GREG-Bird_dots.json", "GREG-CNWA_dots.json", "GREG-Nest_dots.json",
        "LAGU-Bird_dots.json", "LAGU-Nest_dots.json", "TRHE-Bird_dots.json",
        "dots_summary.json",
    ],
}

# --- Feature Service dotting layers not available in S3 ---
FEATURE_SERVICE_LAYERS = [
    {
        "name": "2022_LA_all",
        "url": (
            "https://gis.thewaterinstitute.org/twimanager/rest/services/Hosted"
            "/Y2022_AllSites_LA_Dotting_Point_Master/FeatureServer/0"
        ),
        "description": "Louisiana-wide dotting observations 2022",
    },
    {
        "name": "2023_queen_bess",
        "url": (
            "https://gis.thewaterinstitute.org/twimanager/rest/services/Hosted"
            "/Queen_Bess_Dotting_Observations_2023/FeatureServer/0"
        ),
        "description": "Queen Bess Island dotting observations 2023",
    },
]


# --- S3 fetch ---

def fetch_s3_dots(skip_existing=True):
    total_files = sum(len(files) for files in S3_COLONIES.values())
    fetched = skipped = 0
    failed = []

    print(f"Fetching S3 dot files ({total_files} total)...\n")

    for colony_path, files in S3_COLONIES.items():
        out_dir = os.path.join(S3_OUTPUT_DIR, colony_path)
        os.makedirs(out_dir, exist_ok=True)

        for filename in files:
            output_path = os.path.join(out_dir, filename)

            if skip_existing and os.path.exists(output_path):
                skipped += 1
                continue

            url = f"{S3_BASE_URL}/{colony_path}/{filename}"
            try:
                print(f"Fetching: {colony_path}/{filename}")
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                data = resp.json()

                with open(output_path, "w") as f:
                    json.dump(data, f)

                count = len(data.get("features", [])) if isinstance(data, dict) else "?"
                print(f"  -> {count} features")
                fetched += 1

            except requests.HTTPError as e:
                print(f"  ERROR {e.response.status_code}: {colony_path}/{filename}")
                failed.append(f"{colony_path}/{filename}")
            except Exception as e:
                print(f"  ERROR: {colony_path}/{filename} — {e}")
                failed.append(f"{colony_path}/{filename}")

    print(f"\nS3 done. Fetched: {fetched} | Skipped: {skipped} | Failed: {len(failed)}")
    if failed:
        for f in failed:
            print(f"  FAILED: {f}")


# --- Feature Service fetch ---

def paginate_feature_service(url, where="1=1"):
    """Paginate a Feature Service layer and return all features as GeoJSON list."""
    count_resp = requests.get(
        f"{url}/query",
        params={"where": where, "returnCountOnly": "true", "f": "json"},
        timeout=30,
    )
    count_resp.raise_for_status()
    total = count_resp.json().get("count", 0)
    print(f"  Total records: {total}")

    all_features = []
    offset = 0

    while offset < total:
        resp = requests.get(
            f"{url}/query",
            params={
                "where": where,
                "outFields": "*",
                "resultOffset": offset,
                "resultRecordCount": PAGE_SIZE,
                "f": "geojson",
            },
            timeout=60,
        )
        resp.raise_for_status()
        features = resp.json().get("features", [])
        all_features.extend(features)
        print(f"  {offset + len(features)}/{total}")
        offset += PAGE_SIZE

    return all_features


def fetch_feature_service_dots(skip_existing=True):
    os.makedirs(FS_OUTPUT_DIR, exist_ok=True)
    print(f"Fetching Feature Service dotting layers...\n")

    for layer in FEATURE_SERVICE_LAYERS:
        output_path = os.path.join(FS_OUTPUT_DIR, f"{layer['name']}.geojson")

        if skip_existing and os.path.exists(output_path):
            print(f"Skipping (exists): {layer['name']}")
            continue

        print(f"{layer['name']} — {layer['description']}")
        try:
            features = paginate_feature_service(layer["url"])
            geojson = {"type": "FeatureCollection", "features": features}

            with open(output_path, "w") as f:
                json.dump(geojson, f)

            print(f"  Saved: {output_path} ({len(features)} records)\n")

        except Exception as e:
            print(f"  ERROR: {layer['name']} — {e}\n")


# --- Entry point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch TWI bird dot annotations")
    parser.add_argument("--s3-only", action="store_true", help="Only fetch S3 dot GeoJSONs")
    parser.add_argument("--fs-only", action="store_true", help="Only fetch Feature Service layers")
    parser.add_argument("--no-skip", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    skip = not args.no_skip

    if not args.fs_only:
        fetch_s3_dots(skip_existing=skip)

    if not args.s3_only:
        fetch_feature_service_dots(skip_existing=skip)
