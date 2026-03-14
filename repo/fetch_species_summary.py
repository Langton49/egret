"""
fetch_species_summary.py

Queries the TWI Avian Monitoring Feature Service for aggregated species counts
across all Gulf Coast colonies from 2010 onwards. Replaces the single S3 JSON
fetch with a paginated query against the live Summary_Avian_Data layer.

The Feature Service covers every surveyed colony from Texas to Florida,
far broader than the S3 devDays sampler (4 Louisiana islands only).

Output:
    data/summary/avian_summary_raw.geojson   <- all records as GeoJSON
    data/summary/avian_summary.json          <- structured by georegion/colony/year/species

Usage:
    python fetch_species_summary.py
    python fetch_species_summary.py --state LA          # filter by state
    python fetch_species_summary.py --year 2021         # filter by year
    python fetch_species_summary.py --colony "Queen Bess Island"
"""

import argparse
import json
import os
import requests

FEATURE_SERVICE_URL = (
    "https://gis.thewaterinstitute.org/twimanager/rest/services/Hosted"
    "/Summary_Avian_Data/FeatureServer/0"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "summary")
PAGE_SIZE = 1000


def query_feature_service(where="1=1", out_fields="*"):
    """Paginate through the Feature Service and return all features."""
    # Get total count first
    count_resp = requests.get(
        f"{FEATURE_SERVICE_URL}/query",
        params={"where": where, "returnCountOnly": "true", "f": "json"},
        timeout=30,
    )
    count_resp.raise_for_status()
    total = count_resp.json().get("count", 0)
    print(f"Total records: {total}")

    all_features = []
    offset = 0

    while offset < total:
        resp = requests.get(
            f"{FEATURE_SERVICE_URL}/query",
            params={
                "where": where,
                "outFields": out_fields,
                "resultOffset": offset,
                "resultRecordCount": PAGE_SIZE,
                "f": "geojson",
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        features = data.get("features", [])
        all_features.extend(features)
        print(f"  Fetched {offset + len(features)}/{total}")
        offset += PAGE_SIZE

    return all_features


def build_where_clause(state=None, year=None, colony=None):
    clauses = []
    if state:
        clauses.append(f"state='{state}'")
    if year:
        clauses.append(f"year={year}")
    if colony:
        safe = colony.replace("'", "''")
        clauses.append(f"colonyname='{safe}'")
    return " AND ".join(clauses) if clauses else "1=1"


def structure_summary(features):
    """Nest features by georegion -> colony -> year -> species list."""
    structured = {}
    for feat in features:
        props = feat.get("properties", {})
        region = props.get("georegion") or "Unknown"
        colony = props.get("colonyname") or "Unknown"
        year = str(props.get("year") or "Unknown")
        species = props.get("speciescode") or "Unknown"

        structured.setdefault(region, {})
        structured[region].setdefault(colony, {})
        structured[region][colony].setdefault(year, [])
        structured[region][colony][year].append({
            "species_code": species,
            "species_name": props.get("specie") or props.get("name"),
            "birds": props.get("birds"),
            "nests": props.get("nests"),
            "combined_may_june_total": props.get("combinedmayjunetotal_"),
            "state": props.get("state"),
            "huc8": props.get("huc8"),
            "longitude": props.get("longitude_y"),
            "latitude": props.get("latitude_y"),
        })

    return structured


def fetch_summary(state=None, year=None, colony=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    where = build_where_clause(state=state, year=year, colony=colony)
    print(f"Querying: {FEATURE_SERVICE_URL}")
    print(f"Filter: {where}\n")

    features = query_feature_service(where=where)

    # Save raw GeoJSON
    geojson = {"type": "FeatureCollection", "features": features}
    raw_path = os.path.join(OUTPUT_DIR, "avian_summary_raw.geojson")
    with open(raw_path, "w") as f:
        json.dump(geojson, f)
    print(f"\nRaw GeoJSON saved: {raw_path} ({len(features)} records)")

    # Save structured summary
    structured = structure_summary(features)
    structured_path = os.path.join(OUTPUT_DIR, "avian_summary.json")
    with open(structured_path, "w") as f:
        json.dump(structured, f, indent=2)
    print(f"Structured summary saved: {structured_path}")

    # Print coverage overview
    print(f"\nCoverage:")
    for region, colonies in structured.items():
        for col, years in colonies.items():
            print(f"  {region} / {col}: {sorted(years.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch TWI avian species summary")
    parser.add_argument("--state", help="Filter by state code (e.g. LA, TX, FL)")
    parser.add_argument("--year", type=int, help="Filter by survey year")
    parser.add_argument("--colony", help="Filter by colony name")
    args = parser.parse_args()

    fetch_summary(state=args.state, year=args.year, colony=args.colony)
