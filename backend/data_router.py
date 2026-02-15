"""
AOI Quick Analysis Endpoint
Returns a high-level summary for any user-drawn area of interest.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
from shapely.geometry import shape, Point
from pathlib import Path

router = APIRouter(prefix="/aoidata", tags=["data"])

# ---------------------------------------------------------------------------
# Data loading (cached in memory on first request)
# ---------------------------------------------------------------------------

_results_df: Optional[pd.DataFrame] = None
_profiles_df: Optional[pd.DataFrame] = None

DATA_DIR = Path("../training_two/model_output")
RESULTS_PATH = DATA_DIR / "habitat_results.csv"
PROFILES_PATH = DATA_DIR / "cell_profiles.csv"


def get_results():
    global _results_df
    if _results_df is None:
        if not RESULTS_PATH.exists():
            raise HTTPException(status_code=500, detail="habitat_results.csv not found")
        _results_df = pd.read_csv(RESULTS_PATH)
    return _results_df


def get_profiles():
    global _profiles_df
    if _profiles_df is None:
        if not PROFILES_PATH.exists():
            raise HTTPException(status_code=500, detail="cell_profiles.csv not found")
        _profiles_df = pd.read_csv(PROFILES_PATH)
    return _profiles_df


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AOIRequest(BaseModel):
    aoi: dict  # GeoJSON FeatureCollection from Mapbox Draw


class AOISummary(BaseModel):
    total_sightings: int
    species_count: int
    top_species: list
    condition: str
    condition_detail: str
    vegetation_trend: str
    water_trend: str
    notable_change: str
    diversity_level: str
    data_coverage: str
    area_km2: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cells_in_polygon(polygon_geom, results_df):
    """Find all cell centers that fall within the AOI polygon."""
    poly = shape(polygon_geom)
    mask = results_df.apply(
        lambda row: poly.contains(Point(row["cell_x"], row["cell_y"])),
        axis=1
    )
    return mask


def estimate_area_km2(polygon_geom):
    """Rough area estimate from a geographic polygon."""
    from shapely.ops import transform
    import pyproj

    poly = shape(polygon_geom)
    project = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:32615", always_xy=True
    ).transform
    projected = transform(project, poly)
    return projected.area / 1e6


def describe_trend(slope, index_name):
    """Convert a per-year slope to a human-readable string."""
    if abs(slope) < 0.005:
        return "Stable"

    direction = "increased" if slope > 0 else "declined"
    # Convert to approximate percentage over 6 years
    pct = abs(slope * 6) * 100
    return f"{index_name} has {direction} ~{pct:.0f}% over 6 years"


def get_diversity_level(shannon_mean):
    """Classify Shannon diversity into plain language."""
    if shannon_mean >= 2.0:
        return "High"
    elif shannon_mean >= 1.0:
        return "Moderate"
    elif shannon_mean >= 0.3:
        return "Low"
    else:
        return "Very low"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/analyze", response_model=AOISummary)
async def analyze_aoi(request: AOIRequest):
    results = get_results()
    profiles = get_profiles()

    # Extract polygon geometry from the FeatureCollection
    features = request.aoi.get("features", [])
    if not features:
        raise HTTPException(status_code=400, detail="No polygon drawn")

    polygon_geom = features[0]["geometry"]

    # Find cells in the AOI
    mask = cells_in_polygon(polygon_geom, results)
    matched = results[mask]

    if matched.empty:
        return HTTPException(
            status_code=404,
            detail="No data found in this area. Try drawing a larger polygon."
        )

    # Also get matching cell profiles for bird stats
    profiles_matched = profiles.merge(
        matched[["cell_x", "cell_y"]], on=["cell_x", "cell_y"]
    )

    # ── Total sightings ──
    total_sightings = 0
    if "n_observations_sum" in profiles_matched.columns:
        total_sightings = int(profiles_matched["n_observations_sum"].sum())
    elif "n_observations_mean" in profiles_matched.columns:
        total_sightings = int(profiles_matched["n_observations_mean"].sum())

    # ── Species count ──
    species_count = 0
    if "n_species_max" in profiles_matched.columns:
        species_count = int(profiles_matched["n_species_max"].max())
    elif "n_species_mean" in profiles_matched.columns:
        species_count = int(profiles_matched["n_species_mean"].max())

    # ── Top species (from order-level data) ──
    top_species = []
    order_cols = [c for c in profiles_matched.columns
                  if c.startswith("n_") and c.endswith("_total") and "observation" not in c
                  and "species" not in c and "individual" not in c]
    if order_cols:
        order_totals = profiles_matched[order_cols].sum().sort_values(ascending=False)
        for col in order_totals.head(5).index:
            name = col.replace("n_", "").replace("_total", "")
            count = int(order_totals[col])
            if count > 0:
                top_species.append({"order": name, "count": count})

    # ── Condition (majority vote) ──
    if "trajectory" in matched.columns:
        trajectory_counts = matched["trajectory"].value_counts()
        condition = trajectory_counts.index[0]
        total = len(matched)
        majority_pct = trajectory_counts.iloc[0] / total * 100

        if condition == "degrading":
            condition_detail = f"This area is predominantly degrading — {majority_pct:.0f}% of the landscape shows declining trends."
        elif condition == "improving":
            condition_detail = f"This area is mostly improving — {majority_pct:.0f}% of the landscape shows positive trends."
        else:
            condition_detail = f"This area is largely stable — {majority_pct:.0f}% of the landscape shows steady conditions."
    else:
        condition = "unknown"
        condition_detail = "Trend data not available for this area."

    # ── Vegetation trend ──
    veg_trend = "No data"
    if "NDVI_trend" in matched.columns:
        ndvi_slope = matched["NDVI_trend"].mean()
        veg_trend = describe_trend(ndvi_slope, "Vegetation")

    # ── Water trend ──
    water_trend = "No data"
    if "NDWI_trend" in matched.columns:
        ndwi_slope = matched["NDWI_trend"].mean()
        water_trend = describe_trend(ndwi_slope, "Water presence")

    # ── Notable change (biggest absolute trend) ──
    notable_change = "No significant changes detected."
    trend_cols = [c for c in matched.columns if c.endswith("_trend")]
    if trend_cols:
        trend_means = matched[trend_cols].mean()
        biggest_idx = trend_means.abs().idxmax()
        biggest_val = trend_means[biggest_idx]
        index_name = biggest_idx.replace("_trend", "").replace("_", " ").upper()

        if abs(biggest_val) > 0.005:
            direction = "increasing" if biggest_val > 0 else "decreasing"
            pct = abs(biggest_val * 6) * 100
            notable_change = f"{index_name} is the most significant change — {direction} by ~{pct:.0f}% over 6 years."

    # ── Diversity level ──
    diversity_level = "No survey data"
    if "shannon_diversity_mean" in profiles_matched.columns:
        surveyed = profiles_matched[profiles_matched["shannon_diversity_mean"] > 0]
        if not surveyed.empty:
            shannon_avg = surveyed["shannon_diversity_mean"].mean()
            diversity_level = get_diversity_level(shannon_avg)

    # ── Data coverage ──
    has_data_col = "has_bird_data" if "has_bird_data" in profiles_matched.columns else None
    if has_data_col:
        surveyed_count = profiles_matched[has_data_col].sum()
        total_cells = len(profiles_matched)
        pct = surveyed_count / total_cells * 100 if total_cells > 0 else 0
        if pct >= 20:
            data_coverage = f"Well surveyed — {pct:.0f}% of this area has bird observation data"
        elif pct >= 5:
            data_coverage = f"Moderately surveyed — {pct:.0f}% coverage. Deeper analysis recommended."
        elif pct > 0:
            data_coverage = f"Sparsely surveyed — only {pct:.0f}% coverage. Results rely heavily on satellite indicators."
        else:
            data_coverage = "No bird surveys on record. Assessment based entirely on satellite data."
    else:
        data_coverage = "Survey coverage data not available."

    # ── Area estimate ──
    area_km2 = None
    try:
        area_km2 = round(estimate_area_km2(polygon_geom), 1)
    except Exception:
        pass

    return AOISummary(
        total_sightings=total_sightings,
        species_count=species_count,
        top_species=top_species,
        condition=condition,
        condition_detail=condition_detail,
        vegetation_trend=veg_trend,
        water_trend=water_trend,
        notable_change=notable_change,
        diversity_level=diversity_level,
        data_coverage=data_coverage,
        area_km2=area_km2,
    )

@router.post("/habitat-score")
async def get_habitat_score(request: AOIRequest):
    features = request.aoi.get("features", [])
    if not features:
        raise HTTPException(status_code=400, detail="No polygon drawn")
        

if __name__ == "__main__":
    import asyncio

    test_aoi = {
        "type": "FeatureCollection",
        "features": [
            {
                "id": "URYrAGC19qF4VvjMv0vid9XXw0KrAeea",
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "coordinates": [
                        [
                            [-89.63785593561656, 29.944684242028544],
                            [-89.27923052749387, 29.944684242028544],
                            [-89.28148603320498, 29.711837587801924],
                            [-89.52056963862059, 29.705960498557687],
                            [-89.63785593561656, 29.944684242028544],
                        ]
                    ],
                    "type": "Polygon",
                },
            }
        ],
    }

    async def run_test():
        print("=" * 60)
        print("  Testing AOI Analysis")
        print("=" * 60)
        print(f"  AOI bounds: ~(-89.64, 29.71) to (-89.28, 29.94)")
        print()

        try:
            request = AOIRequest(aoi=test_aoi)
            result = await analyze_aoi(request)

            print(f"  Area:              {result.area_km2} km²")
            print(f"  Total sightings:   {result.total_sightings:,}")
            print(f"  Species count:     {result.species_count}")
            print(f"  Diversity level:   {result.diversity_level}")
            print(f"  Condition:         {result.condition}")
            print(f"  Detail:            {result.condition_detail}")
            print(f"  Vegetation trend:  {result.vegetation_trend}")
            print(f"  Water trend:       {result.water_trend}")
            print(f"  Notable change:    {result.notable_change}")
            print(f"  Data coverage:     {result.data_coverage}")
            print(f"  Top species:")
            for s in result.top_species:
                print(f"    {s['order']:25s} {s['count']:,}")
            print()
            print("  Full JSON:")
            print(result.model_dump_json(indent=2))

        except HTTPException as e:
            print(f"  ERROR {e.status_code}: {e.detail}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(run_test())