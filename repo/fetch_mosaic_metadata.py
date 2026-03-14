"""
fetch_mosaic_metadata.py

Fetches mosaic metadata JSON files from the TWI S3 mosaics directory and
builds a complete mosaic index covering all available TIF files across both
public/mosaics/ (COGs) and public/devDays/ (oblique + Cam3 mosaics).

The index is the primary reference for researchers who want to locate and
pull specific imagery for local analysis — it records every mosaic's S3 URL,
colony, year, camera type, and geometry notes without downloading the files.

Output:
    data/mosaics/{georegion}/{colony}/{year}/{month}_mosaic.json
    data/mosaics/mosaic_index.json   <- complete index of all TIF URLs
"""

import json
import os
import requests

BASE_URL = "https://twi-avian-2024.s3.us-east-1.amazonaws.com/public/mosaics"
DEVDAYS_URL = "https://twi-avian-2024.s3.us-east-1.amazonaws.com/public/devDays"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "mosaics")

# Mosaic metadata files and their corresponding COG URLs.
# COGs are not downloaded — URLs recorded for on-demand streaming.
MOSAICS = {
    "ApalacheeBay/PepperfishKey/2021": {
        "metadata_files": ["May_mosaic.json"],
        "cog_urls": {
            "May": f"{BASE_URL}/ApalacheeBay/PepperfishKey/2021/May_cog.tif",
        },
    },
    "BaratariaBay/QueenBessIsland/2015": {
        "metadata_files": ["May_mosaic.json"],
        "cog_urls": {
            "May": f"{BASE_URL}/BaratariaBay/QueenBessIsland/2015/May_cog.tif",
        },
    },
    "BaratariaBay/QueenBessIsland/2018": {
        "metadata_files": ["May_mosaic.json"],
        "cog_urls": {
            "May": f"{BASE_URL}/BaratariaBay/QueenBessIsland/2018/May_cog.tif",
        },
    },
    "BaratariaBay/QueenBessIsland/2021": {
        "metadata_files": ["May_mosaic.json", "June_mosaic.json"],
        "cog_urls": {
            "May": f"{BASE_URL}/BaratariaBay/QueenBessIsland/2021/May_cog.tif",
            "June": f"{BASE_URL}/BaratariaBay/QueenBessIsland/2021/June_cog.tif",
        },
    },
    "BretonChandeleurIslands/NewHarborIsland2/2021": {
        "metadata_files": ["June_mosaic.json"],
        "cog_urls": {
            "June": f"{BASE_URL}/BretonChandeleurIslands/NewHarborIsland2/2021/June_cog.tif",
        },
    },
    "BretonChandeleurIslands/NewHarborIsland3/2021": {
        "metadata_files": ["June_mosaic.json"],
        "cog_urls": {
            "June": f"{BASE_URL}/BretonChandeleurIslands/NewHarborIsland3/2021/June_cog.tif",
        },
    },
}


# devDays TIFs — oblique mosaics and Cam3 nadir mosaics.
# Not COGs, not downloaded — URLs recorded for researcher reference.
# geometry_note explains the geometric limitations for each source.
DEVDAYS_TIFS = [
    {
        "colony": "ApalachicolaBirdIsland",
        "georegion": "ApalacheeBay",
        "year": "2010",
        "month": "June",
        "camera": "Cam1+Cam2",
        "geometry": "oblique",
        "geometry_note": "Oblique mosaic, perspective distortion, not georeferenced",
        "dots_available": False,
        "urls": [
            f"{DEVDAYS_URL}/2010_ApalachicolaEast_ApalachicolaBirdIsland_June_mosaic/"
            f"2010_Apalachicola East_Apalachicola Bird Island_June_0.tif"
        ],
    },
    {
        "colony": "QueenBessIsland",
        "georegion": "BaratariaBay",
        "year": "2013",
        "month": "May",
        "camera": "Cam1+Cam2",
        "geometry": "oblique",
        "geometry_note": "Oblique mosaic, perspective distortion, not georeferenced",
        "dots_available": False,
        "urls": [
            f"{DEVDAYS_URL}/2013_QueenBess_May_mosaic/2013_Barataria Bay_Queen Bess Island_May_{tile}.tif"
            for tile in ["0_0", "0_1", "1", "2", "3", "4", "5", "6", "7"]
        ],
    },
    {
        "colony": "FelicityIsland",
        "georegion": "TerrebonneBay",
        "year": "2015",
        "month": "May",
        "camera": "Cam1+Cam2",
        "geometry": "oblique",
        "geometry_note": "Oblique mosaic, perspective distortion, not georeferenced",
        "dots_available": False,
        "urls": [
            f"{DEVDAYS_URL}/2015_FelicityIsland_mosaic/2015_Terrebonne Bay_Felicity Island_May_{tile}.tif"
            for tile in [
                "0", "1", "2_0", "2_1", "3_0_0", "3_0_1", "3_1",
                "4", "5", "6", "7", "8", "9", "10", "11",
                "12", "14", "16", "17", "18", "19",
            ]
        ],
    },
    {
        "colony": "RaccoonIsland",
        "georegion": "TerrebonneBay",
        "year": "2018",
        "month": "May",
        "camera": "Cam1+Cam2",
        "geometry": "oblique",
        "geometry_note": "Oblique mosaic, perspective distortion, not georeferenced",
        "dots_available": False,
        "urls": [
            f"{DEVDAYS_URL}/2018_RaccoonIsland_mosaic/2018_Terrebonne Bay_Raccoon Island_May_{tile}.tif"
            for tile in [
                "0_0_0_0_0_0_0_0", "0_0_0_0_0_0_0_1", "0_0_0_0_0_0_1",
                "0_0_0_0_0_1_0", "0_0_0_0_0_1_1", "0_0_0_0_1_0_1",
                "0_0_0_0_1_1_1", "0_0_0_1_1_0", "0_0_0_1_1_1",
                "0_0_1_0_0_0", "0_0_1_0_0_1_0", "0_0_1_0_0_1_1",
                "0_0_1_0_1_0_0", "0_0_1_0_1_0_1", "0_0_1_0_1_1_0",
                "0_0_1_0_1_1_1", "0_0_1_1_0_0_0", "0_0_1_1_0_0_1",
                "0_0_1_1_0_1_0", "0_0_1_1_0_1_1", "0_0_1_1_1_0_1",
                "0_0_1_1_1_1_0", "0_0_1_1_1_1_1", "0_1_0_0_0_0_0",
                "0_1_0_0_0_0_1", "0_1_0_0_0_1_0", "0_1_0_0_0_1_1",
                "0_1_0_0_1_0_0", "0_1_0_0_1_0_1", "0_1_0_0_1_1_0_0",
                "0_1_0_0_1_1_0_1", "0_1_0_0_1_1_1", "0_1_0_1_0_0_0",
                "0_1_0_1_0_0_1", "0_1_0_1_0_1_0", "0_1_0_1_0_1_1",
                "0_1_0_1_1_0_0", "0_1_0_1_1_0_1", "0_1_0_1_1_1",
                "0_1_1_0_0_0_0", "0_1_1_0_0_0_1", "0_1_1_0_0_1_0",
                "0_1_1_0_0_1_1", "0_1_1_0_1_0_0", "0_1_1_0_1_0_1",
                "0_1_1_0_1_1", "0_1_1_1_0_0_0", "0_1_1_1_0_0_1_0",
                "0_1_1_1_0_0_1_1", "0_1_1_1_0_1_0", "0_1_1_1_1_0",
                "0_1_1_1_1_1", "1", "2", "3", "4", "6", "7", "8", "9", "10", "11",
            ]
        ],
    },
    {
        "colony": "EastTimbalierIsland",
        "georegion": "TerrebonneBay",
        "year": "2024",
        "month": "June",
        "camera": "Cam3",
        "geometry": "nadir",
        "geometry_note": "Nadir orthomosaic, georeferenced via .tfwx world file, 3.5cm GSD",
        "dots_available": False,
        "urls": [
            f"{DEVDAYS_URL}/2024_EastTimbalierIsland/ETI_Cam3_2024/2024_EastTimbalier_Cam3_mosaic{n}.tif"
            for n in [1, 2, 3]
        ],
    },
    {
        "colony": "QueenBessIsland",
        "georegion": "BaratariaBay",
        "year": "2024",
        "month": "May",
        "camera": "Cam3",
        "geometry": "nadir",
        "geometry_note": "Nadir orthomosaic, georeferenced, 3.5cm GSD",
        "dots_available": False,
        "urls": [
            f"{DEVDAYS_URL}/2024_QueenBessIsland/2024_QueenBess_May_21_Cam3_georeferenced2.tif"
        ],
    },
]


def fetch_mosaic_metadata(skip_existing=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mosaic_index = {
        "cogs": {},       # public/mosaics/ COGs — streamable, dots available
        "devdays": [],    # public/devDays/ TIFs — reference only
    }

    fetched = 0
    skipped = 0
    failed = []

    # --- COGs from public/mosaics/ ---
    for colony_path, config in MOSAICS.items():
        out_dir = os.path.join(OUTPUT_DIR, colony_path)
        os.makedirs(out_dir, exist_ok=True)

        georegion, colony, year = colony_path.split("/")
        if georegion not in mosaic_index["cogs"]:
            mosaic_index["cogs"][georegion] = {}
        if colony not in mosaic_index["cogs"][georegion]:
            mosaic_index["cogs"][georegion][colony] = {}
        mosaic_index["cogs"][georegion][colony][year] = {
            "urls": config["cog_urls"],
            "dots_available": True,
            "geometry": "nadir_cog",
            "streamable": True,
        }

        # Fetch metadata JSON files
        for filename in config["metadata_files"]:
            output_path = os.path.join(out_dir, filename)

            if skip_existing and os.path.exists(output_path):
                skipped += 1
                continue

            url = f"{BASE_URL}/{colony_path}/{filename}"
            try:
                print(f"Fetching metadata: {colony_path}/{filename}")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                data = response.json()

                with open(output_path, "w") as f:
                    json.dump(data, f, indent=2)

                print(f"  -> saved ({len(response.content):,} bytes)")
                fetched += 1

            except requests.HTTPError as e:
                print(f"  ERROR {e.response.status_code}: {url}")
                failed.append(url)
            except Exception as e:
                print(f"  ERROR: {url} — {e}")
                failed.append(url)

    # --- devDays TIFs — URLs only, no download ---
    for entry in DEVDAYS_TIFS:
        mosaic_index["devdays"].append(entry)
        print(
            f"Indexed devDays: {entry['colony']} {entry['year']} "
            f"{entry['camera']} ({len(entry['urls'])} tile(s))"
        )

    # Include Feature Service endpoints for researcher reference
    mosaic_index["feature_services"] = {
        "avian_data": {
            "description": "Primary avian data layer — all Gulf Coast colonies 2010+, with uid photo references",
            "url": "https://gis.thewaterinstitute.org/twimanager/rest/services/Hosted/AvianData_App_Visualization_WFL1/FeatureServer/12",
            "query_example": "?where=colonyname='Queen Bess Island'&outFields=*&f=geojson",
        },
        "summary_avian_data": {
            "description": "Aggregated counts by colony/year/species — Gulf Coast wide",
            "url": "https://gis.thewaterinstitute.org/twimanager/rest/services/Hosted/Summary_Avian_Data/FeatureServer/0",
            "query_example": "?where=georegion='BaratariaBay'&outFields=*&f=geojson",
        },
        "dotting_2022_la": {
            "description": "Louisiana-wide dotting point observations 2022",
            "url": "https://gis.thewaterinstitute.org/twimanager/rest/services/Hosted/Y2022_AllSites_LA_Dotting_Point_Master/FeatureServer/0",
            "query_example": "?where=1=1&outFields=*&f=geojson",
        },
        "dotting_2023_queen_bess": {
            "description": "Queen Bess Island dotting observations 2023",
            "url": "https://gis.thewaterinstitute.org/twimanager/rest/services/Hosted/Queen_Bess_Dotting_Observations_2023/FeatureServer/0",
            "query_example": "?where=1=1&outFields=*&f=geojson",
        },
    }

    # Write complete mosaic index
    index_path = os.path.join(OUTPUT_DIR, "mosaic_index.json")
    with open(index_path, "w") as f:
        json.dump(mosaic_index, f, indent=2)
    print(f"\nMosaic index written to {index_path}")
    print(f"  COG entries: {sum(len(c) for r in mosaic_index['cogs'].values() for c in r.values())} colony/years")
    print(f"  devDays entries: {len(mosaic_index['devdays'])}")
    print(f"  Feature Service endpoints: {len(mosaic_index['feature_services'])}")

    print(f"\nMetadata fetch — Fetched: {fetched} | Skipped (existing): {skipped} | Failed: {len(failed)}")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    fetch_mosaic_metadata(skip_existing=True)
