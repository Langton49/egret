"""
Habitat Scoring Endpoint
Fetches last 30 days of Sentinel-2 imagery from AWS Earth Search STAC,
computes spectral indices, and scores through the supervised suitability model.

Async: returns a job ID immediately, frontend polls for results.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict
import numpy as np
import pandas as pd
import pickle
import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import math

router = APIRouter(prefix="/habitat", tags=["habitat"])

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = Path("./suitability_model/suitability_model.pkl")
STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
CLOUD_COVER_MAX = 30
BANDS = ["blue", "green", "red", "nir", "nir08", "swir16", "swir22", "scl"]
CELL_SIZE_KM = 1.0

# ---------------------------------------------------------------------------
# Job storage (in-memory — swap for Redis/DB in production)
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    pending = "pending"
    fetching = "fetching"
    processing = "processing"
    scoring = "scoring"
    complete = "complete"
    failed = "failed"

jobs: Dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AOIRequest(BaseModel):
    aoi: dict

class JobResponse(BaseModel):
    job_id: str
    status: str

class JobResult(BaseModel):
    job_id: str
    status: str
    progress: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None

# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=500, detail="suitability_model.pkl not found")
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model

# ---------------------------------------------------------------------------
# STAC search + data fetch
# ---------------------------------------------------------------------------

def search_stac(bbox, start_date, end_date):
    from pystac_client import Client
    client = Client.open(STAC_URL)
    search = client.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": CLOUD_COVER_MAX}},
        max_items=50,
    )
    return list(search.items())


def load_bands_from_items(items, bbox):
    import rioxarray
    import xarray as xr

    if not items:
        return None, None

    all_datasets = []
    scene_dates = []

    for item in items:
        scene_date = pd.Timestamp(item.datetime)
        ds_bands = {}

        for band_name in BANDS:
            if band_name not in item.assets:
                continue
            href = item.assets[band_name].href
            try:
                da = rioxarray.open_rasterio(href, chunks={"x": 1024, "y": 1024})
                da = da.rio.clip_box(
                    minx=bbox[0], miny=bbox[1],
                    maxx=bbox[2], maxy=bbox[3],
                    crs="EPSG:4326",
                )
                if "band" in da.dims:
                    da = da.squeeze("band", drop=True)
                ds_bands[band_name] = da
            except Exception as e:
                print(f"    Warning: Could not load {band_name} from {item.id}: {e}")
                continue

        if ds_bands:
            ds = xr.Dataset(ds_bands)
            ds = ds.assign_coords(time=scene_date)
            all_datasets.append(ds)
            scene_dates.append(scene_date)

    if not all_datasets:
        return None, None

    combined = xr.concat(all_datasets, dim="time")
    return combined, scene_dates


# ---------------------------------------------------------------------------
# Spectral index computation
# ---------------------------------------------------------------------------

def compute_indices(ds):
    import xarray as xr
    scale = 10000.0

    band_map = {
        "blue": "blue", "green": "green", "red": "red",
        "nir": "nir", "nir08": "nir_narrow",
        "swir16": "swir1", "swir22": "swir2",
    }

    bands = {}
    for aws_name, friendly in band_map.items():
        if aws_name in ds:
            bands[friendly] = ds[aws_name].astype("float32") / scale

    required = ["blue", "green", "red", "nir", "swir1", "swir2"]
    if not all(k in bands for k in required):
        return None

    blue, green, red = bands["blue"], bands["green"], bands["red"]
    nir, swir1, swir2 = bands["nir"], bands["swir1"], bands["swir2"]

    def safe_nd(a, b):
        d = a + b
        return xr.where(d != 0, (a - b) / d, np.nan)

    result = xr.Dataset()
    result["NDVI"] = safe_nd(nir, red)
    result["NDWI"] = safe_nd(green, nir)
    result["MNDWI"] = safe_nd(green, swir1)
    result["NDMI"] = safe_nd(nir, swir1)
    result["EVI"] = xr.where(
        (nir + 6 * red - 7.5 * blue + 1) != 0,
        2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1), np.nan
    )
    result["SAVI"] = xr.where(
        (nir + red + 0.5) != 0,
        1.5 * (nir - red) / (nir + red + 0.5), np.nan
    )
    result["LSWI"] = safe_nd(nir, swir1)
    result["WRI"] = xr.where(
        (nir + swir1) != 0, (green + red) / (nir + swir1), np.nan
    )
    result["wetland_moisture_index"] = (result["NDWI"] + result["NDMI"] + result["MNDWI"]) / 3.0
    result["water_mask"] = xr.where(
        (result["MNDWI"] > 0) & (result["NDVI"] < 0.2), 1.0, 0.0
    )
    result["tc_wetness"] = (
        0.1509 * blue + 0.1973 * green + 0.3279 * red
        + 0.3406 * nir - 0.7112 * swir1 - 0.4572 * swir2
    )
    result["GCVI"] = xr.where(green != 0, (nir / green) - 1, np.nan)

    if "scl" in ds:
        valid = ds["scl"].isin([4, 5, 6, 7, 11])
        for var in result.data_vars:
            result[var] = result[var].where(valid)

    return result


# ---------------------------------------------------------------------------
# Cell aggregation
# ---------------------------------------------------------------------------

def aggregate_to_cells(indices_ds, bbox):
    from pyproj import Transformer

    median_ds = indices_ds.median(dim="time").load()
    df = median_ds.to_dataframe().reset_index()

    if "x" not in df.columns and "longitude" in df.columns:
        df = df.rename(columns={"longitude": "x", "latitude": "y"})
    if "x" not in df.columns:
        return pd.DataFrame()

    x_range = df["x"].max() - df["x"].min()
    if x_range > 1000:
        center_x = df["x"].mean()
        epsg = "EPSG:32615" if center_x < 500000 else "EPSG:32616"
        transformer = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
        lons, lats = transformer.transform(df["x"].values, df["y"].values)
        df["lon"] = lons
        df["lat"] = lats
    else:
        df["lon"] = df["x"]
        df["lat"] = df["y"]

    # Strict bbox filter
    df = df[
        (df["lon"] >= bbox[0]) & (df["lon"] <= bbox[2]) &
        (df["lat"] >= bbox[1]) & (df["lat"] <= bbox[3])
    ]
    if df.empty:
        return pd.DataFrame()

    mid_lat = (bbox[1] + bbox[3]) / 2.0
    cell_lon = CELL_SIZE_KM / (111.32 * np.cos(np.radians(mid_lat)))
    cell_lat = CELL_SIZE_KM / 110.57

    df["cell_x"] = np.floor(df["lon"] / cell_lon) * cell_lon + cell_lon / 2
    df["cell_y"] = np.floor(df["lat"] / cell_lat) * cell_lat + cell_lat / 2

    index_cols = [c for c in df.columns if c in [
        "NDVI", "NDWI", "MNDWI", "NDMI", "EVI", "SAVI",
        "LSWI", "WRI", "wetland_moisture_index", "water_mask",
        "tc_wetness", "GCVI",
    ]]

    cell_df = df.groupby(["cell_x", "cell_y"])[index_cols].mean().reset_index()
    cell_df = cell_df.dropna(subset=index_cols, how="all")
    return cell_df


# ---------------------------------------------------------------------------
# Feature engineering (must match training exactly)
# ---------------------------------------------------------------------------

def engineer_features_for_scoring(cell_df, acquisition_date):
    model = get_model()
    feature_names = model["feature_names"]
    X = cell_df.copy()

    # Interactions
    if "NDVI" in X.columns and "NDWI" in X.columns:
        X["NDVI_x_NDWI"] = X["NDVI"] * X["NDWI"]
    if "NDVI" in X.columns and "MNDWI" in X.columns:
        X["NDVI_x_MNDWI"] = X["NDVI"] * X["MNDWI"]
    if "EVI" in X.columns and "NDMI" in X.columns:
        X["EVI_x_NDMI"] = X["EVI"] * X["NDMI"]
    if "GCVI" in X.columns and "wetland_moisture_index" in X.columns:
        X["GCVI_x_wetmoist"] = X["GCVI"] * X["wetland_moisture_index"]

    # Ratios
    if "NDVI" in X.columns and "MNDWI" in X.columns:
        X["veg_water_ratio"] = X["NDVI"] / (X["MNDWI"].abs() + 0.01)
    if "NDMI" in X.columns and "NDVI" in X.columns:
        X["moisture_per_veg"] = X["NDMI"] / (X["NDVI"].abs() + 0.01)

    # Squared
    for idx in ["NDVI", "NDWI", "EVI"]:
        if idx in X.columns:
            X[f"{idx}_sq"] = X[idx] ** 2

    # Time features
    month = acquisition_date.month
    doy = acquisition_date.timetuple().tm_yday
    X["month_sin"] = math.sin(2 * math.pi * month / 12)
    X["month_cos"] = math.cos(2 * math.pi * month / 12)
    X["doy_sin"] = math.sin(2 * math.pi * doy / 365)
    X["doy_cos"] = math.cos(2 * math.pi * doy / 365)

    # Match feature order
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0

    X = X[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
    return X.values, feature_names


# ---------------------------------------------------------------------------
# Model scoring
# ---------------------------------------------------------------------------

def score_cells(cell_df, acquisition_date):
    model = get_model()
    classifier = model["classifier"]
    regressor = model["regressor"]

    X, feature_names = engineer_features_for_scoring(cell_df, acquisition_date)

    proba = classifier.predict_proba(X)[:, 1]

    diversity_pred = np.zeros(len(X))
    if regressor is not None:
        diversity_pred = regressor.predict(X)
        diversity_pred = np.maximum(diversity_pred, 0)

    # Classify into archetypes — separate open water from unsuitable land
    archetype_labels = []
    water_mask_vals = cell_df["water_mask"].values if "water_mask" in cell_df.columns else np.zeros(len(cell_df))

    for i, p in enumerate(proba):
        if water_mask_vals[i] > 0.5:
            archetype_labels.append("Open water")
        elif p >= 0.7:
            archetype_labels.append("Highly suitable")
        elif p >= 0.4:
            archetype_labels.append("Moderately suitable")
        elif p >= 0.15:
            archetype_labels.append("Marginal")
        else:
            archetype_labels.append("Unsuitable")

    results = cell_df[["cell_x", "cell_y"]].copy()
    results["suitability_probability"] = proba
    results["suitability_pct"] = (proba * 100).round(1)
    results["archetype"] = archetype_labels
    results["predicted_diversity"] = diversity_pred.round(2)

    return results


# ---------------------------------------------------------------------------
# Background job
# ---------------------------------------------------------------------------

def run_scoring_job(job_id: str, polygon_geom: dict):
    print("HELLLLLO")
    try:
        from shapely.geometry import shape, Point
        poly = shape(polygon_geom)
        bbox = list(poly.bounds)

        # ── Cache paths (relative to this file) ──
        cache_path = Path("./satellite_cache/latest_indices.csv")
        meta_path = Path("./satellite_cache/cache_metadata.json")

        print(f"CWD: {Path.cwd()}")
        print(f"Cache path: {cache_path.resolve()}")
        print(f"Cache exists: {cache_path.exists()}")
        used_cache = False
        if cache_path.exists():
            print(f"Job {job_id}: Loading cached data from {cache_path}")
            jobs[job_id]["status"] = JobStatus.processing
            jobs[job_id]["progress"] = "Loading cached satellite data..."

            cached_df = pd.read_csv(cache_path)

            # Fast bbox pre-filter, then point-in-polygon
            cell_df = cached_df[
                (cached_df["cell_x"] >= bbox[0]) & (cached_df["cell_x"] <= bbox[2]) &
                (cached_df["cell_y"] >= bbox[1]) & (cached_df["cell_y"] <= bbox[3])
            ].copy()

            if not cell_df.empty:
                # Refine with actual polygon containment
                mask = cell_df.apply(
                    lambda row: poly.contains(Point(row["cell_x"], row["cell_y"])),
                    axis=1
                )
                cell_df = cell_df[mask].copy()

            print(f"Cells in polygon: {len(cell_df)}")

            if not cell_df.empty:
                used_cache = True

                # Get cache metadata for dates
                cache_meta = {}
                if meta_path.exists():
                    with open(meta_path) as f:
                        cache_meta = json.load(f)

                cache_date_str = cache_meta.get("end_date", datetime.utcnow().strftime("%Y-%m-%d"))
                cache_date = datetime.strptime(cache_date_str, "%Y-%m-%d")
                start_date_str = cache_meta.get("start_date", "")
                n_scenes = cache_meta.get("n_scenes", 0)

                jobs[job_id]["progress"] = f"Scoring {len(cell_df)} cells from cache..."
            else:
                jobs[job_id]["progress"] = "No cached cells in this area. Fetching live..."

        # ── Fall back to live STAC fetch ──
        if not used_cache:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)

            jobs[job_id]["status"] = JobStatus.fetching
            jobs[job_id]["progress"] = f"Searching Sentinel-2 scenes ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"

            items = search_stac(bbox, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            if not items:
                jobs[job_id]["status"] = JobStatus.failed
                jobs[job_id]["error"] = "No cloud-free Sentinel-2 scenes found in the last 30 days."
                return

            jobs[job_id]["progress"] = f"Found {len(items)} scenes. Loading imagery..."

            ds, scene_dates = load_bands_from_items(items, bbox)
            if ds is None:
                jobs[job_id]["status"] = JobStatus.failed
                jobs[job_id]["error"] = "Failed to load imagery from available scenes."
                return

            cache_date = sorted(scene_dates)[len(scene_dates) // 2]
            start_date_str = start_date.strftime("%Y-%m-%d")
            cache_date_str = end_date.strftime("%Y-%m-%d")
            n_scenes = len(items)

            jobs[job_id]["status"] = JobStatus.processing
            jobs[job_id]["progress"] = "Computing spectral indices..."

            indices = compute_indices(ds)
            if indices is None:
                jobs[job_id]["status"] = JobStatus.failed
                jobs[job_id]["error"] = "Missing required spectral bands."
                return

            jobs[job_id]["progress"] = "Aggregating to grid cells..."

            cell_df = aggregate_to_cells(indices, bbox)
            if cell_df.empty:
                jobs[job_id]["status"] = JobStatus.failed
                jobs[job_id]["error"] = "No valid pixels found after cloud masking."
                return

        jobs[job_id]["status"] = JobStatus.scoring
        jobs[job_id]["progress"] = f"Scoring {len(cell_df)} cells..."

        scored = score_cells(cell_df, cache_date if used_cache else cache_date)

        # Build response
        n_cells = len(scored)
        mean_prob = float(scored["suitability_probability"].mean())
        max_prob = float(scored["suitability_probability"].max())
        archetype_counts = scored["archetype"].value_counts().to_dict()

        top_cells = scored.nlargest(5, "suitability_probability")
        top_list = [
            {
                "lon": float(row["cell_x"]),
                "lat": float(row["cell_y"]),
                "probability": round(float(row["suitability_probability"]), 3),
                "archetype": row["archetype"],
                "predicted_diversity": round(float(row["predicted_diversity"]), 1),
            }
            for _, row in top_cells.iterrows()
        ]

        high_pct = archetype_counts.get("Highly suitable", 0) / n_cells * 100
        mod_pct = archetype_counts.get("Moderately suitable", 0) / n_cells * 100
        unsuit_pct = archetype_counts.get("Unsuitable", 0) / n_cells * 100

        if high_pct > 30:
            summary = f"This area shows strong habitat potential — {high_pct:.0f}% of cells are highly suitable for avian species."
        elif high_pct + mod_pct > 50:
            summary = f"Mixed habitat quality — {high_pct:.0f}% highly suitable and {mod_pct:.0f}% moderately suitable."
        elif unsuit_pct > 70:
            summary = f"Limited habitat potential — {unsuit_pct:.0f}% of the area is unsuitable, likely open water or bare ground."
        else:
            summary = f"Moderate habitat potential — mean suitability probability of {mean_prob:.0%}."

        # Build cell polygons for fill layer
        mid_lat = (bbox[1] + bbox[3]) / 2.0
        half_lon = (CELL_SIZE_KM / (111.32 * math.cos(math.radians(mid_lat)))) / 2
        half_lat = (CELL_SIZE_KM / 110.57) / 2

        cell_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [float(row["cell_x"]) - half_lon, float(row["cell_y"]) - half_lat],
                            [float(row["cell_x"]) + half_lon, float(row["cell_y"]) - half_lat],
                            [float(row["cell_x"]) + half_lon, float(row["cell_y"]) + half_lat],
                            [float(row["cell_x"]) - half_lon, float(row["cell_y"]) + half_lat],
                            [float(row["cell_x"]) - half_lon, float(row["cell_y"]) - half_lat],
                        ]],
                    },
                    "properties": {
                        "suitability": round(float(row["suitability_pct"]), 1),
                        "probability": round(float(row["suitability_probability"]), 3),
                        "archetype": row["archetype"],
                        "predicted_diversity": round(float(row["predicted_diversity"]), 1),
                    },
                }
                for _, row in scored.iterrows()
            ],
        }

        jobs[job_id]["status"] = JobStatus.complete
        jobs[job_id]["progress"] = "Complete"
        jobs[job_id]["result"] = {
            "n_cells": n_cells,
            "n_scenes": n_scenes,
            "date_range": f"{start_date_str} to {cache_date_str}",
            "acquisition_date": cache_date_str,
            "source": "cache" if used_cache else "live",
            "mean_suitability": round(mean_prob * 100, 1),
            "max_suitability": round(max_prob * 100, 1),
            "summary": summary,
            "archetype_breakdown": {
                k: {"count": v, "pct": round(v / n_cells * 100, 1)}
                for k, v in archetype_counts.items()
            },
            "top_cells": top_list,
            "cell_geojson": cell_geojson,
        }

    except Exception as e:
        import traceback
        jobs[job_id]["status"] = JobStatus.failed
        jobs[job_id]["error"] = str(e)
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/score", response_model=JobResponse)
async def start_scoring(request: AOIRequest, background_tasks: BackgroundTasks):
    features = request.aoi.get("features", [])
    if not features:
        raise HTTPException(status_code=400, detail="No polygon drawn")

    polygon_geom = features[0]["geometry"]
    job_id = str(uuid.uuid4())[:8]

    jobs[job_id] = {
        "status": JobStatus.pending,
        "progress": "Job queued",
        "result": None,
        "error": None,
        "created": datetime.utcnow().isoformat(),
    }

    background_tasks.add_task(run_scoring_job, job_id, polygon_geom)
    return JobResponse(job_id=job_id, status="pending")


@router.get("/score/{job_id}", response_model=JobResult)
async def get_scoring_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return JobResult(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        result=job.get("result"),
        error=job.get("error"),
    )


@router.delete("/score/{job_id}")
async def cancel_job(job_id: str):
    if job_id in jobs:
        del jobs[job_id]
        return {"detail": "Job removed"}
    raise HTTPException(status_code=404, detail="Job not found")