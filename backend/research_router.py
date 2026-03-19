"""
Research Repository API router.

Mount in server.py:
    from research_router import router as research_router
    app.include_router(research_router, prefix="/api/research")

Endpoints:
    GET  /api/research/colonies
    GET  /api/research/colonies/search?q=
    GET  /api/research/colony/{slug}/profile
    GET  /api/research/colony/{slug}/timeline
    GET  /api/research/colony/{slug}/map_frames
    POST /api/research/search
    POST /api/research/chat
"""

import json
import logging
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
_TWI_CSV = _ROOT / "training_two" / "twi_features.csv"
_REGISTRY = _ROOT / "training_two" / "satellite_timeseries" / "colonies" / "colony_registry.json"
_NC_BASE = _ROOT / "training_two" / "satellite_timeseries" / "colonies"
_MOSAIC_INDEX = _ROOT / "repo" / "data" / "mosaics" / "mosaic_index.json"

# ---------------------------------------------------------------------------
# Data loading (cached on first call)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_twi() -> pd.DataFrame:
    df = pd.read_csv(_TWI_CSV)
    # Normalise colony names for look-up
    df["_slug"] = df["ColonyName"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
    return df


@lru_cache(maxsize=1)
def _load_registry() -> dict:
    """Returns dict keyed by slug."""
    with open(_REGISTRY) as f:
        raw = json.load(f)
    # registry may be a list or a dict
    if isinstance(raw, list):
        return {c["slug"]: c for c in raw}
    return raw  # already keyed by slug


@lru_cache(maxsize=1)
def _load_satellite_features() -> pd.DataFrame:
    """Load pre-extracted satellite indices CSV. Returns empty DataFrame if absent."""
    csv_path = _ROOT / "training_two" / "colony_satellite_features.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


@lru_cache(maxsize=1)
def _load_mosaic_index() -> dict:
    if not _MOSAIC_INDEX.exists():
        return {}
    with open(_MOSAIC_INDEX) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _colony_survey_years() -> dict[str, list[int]]:
    """Return {colony_name: sorted list of years with TWI data}."""
    df = _load_twi()
    result = {}
    for name, grp in df.groupby("ColonyName"):
        result[name] = sorted(grp["Year"].dropna().astype(int).tolist())
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slug_to_colony_name(slug: str) -> str | None:
    """Resolve a URL slug → canonical ColonyName from the registry."""
    registry = _load_registry()
    entry = registry.get(slug)
    if entry:
        return entry.get("name") or entry.get("ColonyName")
    return None


def _safe(val):
    """Return None for NaN/None, else the value."""
    try:
        if math.isnan(float(val)):
            return None
    except (TypeError, ValueError):
        pass
    return val


def _nc_path(slug: str, year: int) -> Path | None:
    """Return path to the first NetCDF file for a given colony/year, or None."""
    colony_dir = _NC_BASE / slug
    if not colony_dir.exists():
        return None
    # Match any sensor suffix
    for p in colony_dir.glob(f"{slug}_{year}_*.nc"):
        return p
    return None


def _sensor_from_year(year: int) -> str:
    return "landsat" if year <= 2016 else "s2"


def _read_nc_centroid(nc_path: Path, lat: float, lon: float, index_name: str) -> float | None:
    """
    Open a NetCDF, select the pixel nearest to (lat, lon),
    and return the median value of the requested index band.
    Returns None if the file cannot be read or index is missing.
    """
    try:
        import xarray as xr

        ds = xr.open_dataset(nc_path)
        if index_name not in ds:
            return None

        da = ds[index_name]
        # Squeeze time dimension if present
        if "time" in da.dims:
            da = da.median(dim="time", skipna=True)

        # Try to select nearest point via lat/lon or x/y coordinates
        coord_dims = list(da.dims)
        if "lat" in coord_dims and "lon" in coord_dims:
            val = float(da.sel(lat=lat, lon=lon, method="nearest").values)
        elif "y" in coord_dims and "x" in coord_dims:
            val = float(da.sel(y=lat, x=lon, method="nearest").values)
        else:
            # Fall back to spatial mean
            val = float(da.mean(skipna=True).values)

        ds.close()
        return None if math.isnan(val) else val
    except Exception as e:
        logger.warning("Could not read %s from %s: %s", index_name, nc_path, e)
        return None


def _gap_fill(delta: float | None, nearest_year: int) -> tuple[str | None, str | None]:
    """Return (signal, text) from an NDVI delta relative to nearest survey year."""
    if delta is None:
        return None, None
    if delta > 0.10:
        return "improving", (
            f"Habitat conditions improved vs {nearest_year} survey — "
            "populations may have increased"
        )
    if delta < -0.10:
        return "declining", (
            f"Habitat conditions degraded vs {nearest_year} survey — "
            "populations may have declined"
        )
    return "stable", (
        f"Conditions similar to {nearest_year} survey — populations likely stable"
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(tags=["research"])


# ---------------------------------------------------------------------------
# GET /colonies
# ---------------------------------------------------------------------------


@router.get("/colonies")
def get_colonies():
    """Return all colonies with their survey years for the frontend dropdown."""
    registry = _load_registry()
    survey_years_map = _colony_survey_years()
    df = _load_twi()

    # Build a name→(lat, lon, geo_region, state) lookup from twi CSV
    # (registry has lat/lon too, prefer registry)
    out = []
    for slug, entry in registry.items():
        colony_name = entry.get("name") or entry.get("ColonyName", "")
        survey_years = survey_years_map.get(colony_name, [])

        # geo_region / state: pull from twi CSV if available
        colony_rows = df[df["ColonyName"] == colony_name]
        geo_region = None
        state = None
        if not colony_rows.empty:
            geo_region = _safe(colony_rows.iloc[-1].get("geo_region"))
            state = _safe(colony_rows.iloc[-1].get("state"))

        out.append(
            {
                "name": colony_name,
                "slug": slug,
                "lat": entry.get("lat"),
                "lon": entry.get("lon"),
                "geo_region": geo_region,
                "state": state,
                "survey_years": survey_years,
            }
        )

    out.sort(key=lambda c: c["name"])
    return out


# ---------------------------------------------------------------------------
# GET /colonies/search?q=   — fuzzy colony name lookup, no AI
# ---------------------------------------------------------------------------


@router.get("/colonies/search")
def search_colonies_by_name(q: str = Query(..., min_length=2)):
    """
    Case-insensitive substring match against colony names.
    Returns exact-prefix matches first, then remaining partial matches.
    No AI involved — pure string filter against colony_registry.json.
    """
    q_lower = q.lower().strip()
    registry = _load_registry()
    survey_years_map = _colony_survey_years()

    exact_prefix, partial = [], []
    for slug, entry in registry.items():
        name = entry.get("name") or entry.get("ColonyName", "")
        name_lower = name.lower()
        if q_lower not in name_lower:
            continue
        item = {
            "name": name,
            "slug": slug,
            "lat": entry.get("lat"),
            "lon": entry.get("lon"),
            "survey_years": survey_years_map.get(name, []),
        }
        if name_lower.startswith(q_lower):
            exact_prefix.append(item)
        else:
            partial.append(item)

    exact_prefix.sort(key=lambda c: c["name"])
    partial.sort(key=lambda c: c["name"])
    return exact_prefix + partial


# ---------------------------------------------------------------------------
# GET /colony/{slug}/profile
# ---------------------------------------------------------------------------


@router.get("/colony/{slug}/profile")
def get_colony_profile(slug: str):
    """Full colony profile: all survey years of biological data + satellite years."""
    registry = _load_registry()
    entry = registry.get(slug)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Colony '{slug}' not found in registry")

    colony_name = entry.get("name") or entry.get("ColonyName", "")
    lat = entry.get("lat")
    lon = entry.get("lon")

    df = _load_twi()
    colony_rows = df[df["ColonyName"] == colony_name].sort_values("Year")

    # Build survey_data dict keyed by year string
    survey_data = {}
    primary_habitat = None
    geo_region = None
    state = None

    survey_cols = [
        "total_birds", "active_nests", "n_species", "shannon_diversity",
        "dominant_species", "nest_success_ratio", "colony_trajectory",
        "active_nests_pct_change", "survey_months",
    ]

    for _, row in colony_rows.iterrows():
        year_key = str(int(row["Year"]))
        record = {}
        for col in survey_cols:
            record[col] = _safe(row.get(col))
        survey_data[year_key] = record
        if primary_habitat is None:
            primary_habitat = _safe(row.get("primary_habitat"))
        if geo_region is None:
            geo_region = _safe(row.get("geo_region"))
        if state is None:
            state = _safe(row.get("state"))

    # Satellite years: check which NC files exist for this slug
    colony_dir = _NC_BASE / slug
    satellite_years = []
    if colony_dir.exists():
        for nc_file in sorted(colony_dir.glob(f"{slug}_*.nc")):
            parts = nc_file.stem.split("_")
            # stem: {slug}_{year}_{sensor}  — year is second-to-last part before sensor
            for part in parts:
                if part.isdigit() and 2000 <= int(part) <= 2030:
                    satellite_years.append(int(part))
                    break
    satellite_years = sorted(set(satellite_years))

    # Imagery URLs from mosaic index
    mosaic_index = _load_mosaic_index()
    imagery = {}
    if colony_name in mosaic_index:
        for year_key, info in mosaic_index[colony_name].items():
            imagery[year_key] = info

    return {
        "colony": colony_name,
        "slug": slug,
        "lat": lat,
        "lon": lon,
        "geo_region": geo_region,
        "state": state,
        "primary_habitat": primary_habitat,
        "survey_data": survey_data,
        "satellite_years": satellite_years,
        "imagery": imagery,
    }


# ---------------------------------------------------------------------------
# GET /colony/{slug}/timeline
# ---------------------------------------------------------------------------

VALID_INDICES = {"NDVI", "MNDWI", "NDWI", "NDMI", "water_mask"}


@router.get("/colony/{slug}/timeline")
def get_colony_timeline(
    slug: str,
    index: str = Query(default="NDVI"),
    year_start: int = Query(default=2010),
    year_end: int = Query(default=2024),
):
    if index not in VALID_INDICES:
        raise HTTPException(status_code=400, detail=f"index must be one of {VALID_INDICES}")

    registry = _load_registry()
    entry = registry.get(slug)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Colony '{slug}' not found")

    colony_name = entry.get("name") or entry.get("ColonyName", "")

    df = _load_twi()
    colony_rows = df[df["ColonyName"] == colony_name]
    survey_years = set(colony_rows["Year"].dropna().astype(int).tolist())

    # Load pre-extracted satellite index values from CSV cache (fast path).
    # Never open NC files per request — they are large and slow (~300 ms each).
    sat_df = _load_satellite_features()
    if not sat_df.empty:
        slug_rows = sat_df[sat_df["slug"] == slug]
        sat_lookup: dict[int, pd.Series] = {int(r["year"]): r for _, r in slug_rows.iterrows()}
    else:
        sat_lookup = {}

    # Build NDVI baseline from survey years for gap-fill delta computation.
    # Gap-fill is always NDVI-based regardless of the requested display index.
    survey_ndvi: dict[int, float] = {}
    for sy in sorted(survey_years):
        row = sat_lookup.get(sy)
        if row is not None:
            val = _safe(row.get("NDVI"))
            if val is not None:
                survey_ndvi[sy] = val

    years_out = []
    for year in range(year_start, year_end + 1):
        has_survey = year in survey_years
        row = sat_lookup.get(year)

        if row is None:
            # No satellite data for this year — return null values gracefully
            years_out.append(
                {
                    "year": year,
                    "value": None,
                    "sensor": None,
                    "has_survey": has_survey,
                    "gap_fill_signal": None,
                    "gap_fill_text": None,
                }
            )
            continue

        value = _safe(row.get(index))
        sensor = str(row.get("sensor", _sensor_from_year(year)))

        # Gap-fill: compute from NDVI delta for non-survey years that have satellite data
        gap_signal = None
        gap_text = None
        if not has_survey and survey_ndvi:
            ndvi_val = _safe(row.get("NDVI"))
            if ndvi_val is not None:
                nearest_year = min(survey_ndvi.keys(), key=lambda sy: abs(sy - year))
                delta = ndvi_val - survey_ndvi[nearest_year]
                gap_signal, gap_text = _gap_fill(delta, nearest_year)

        years_out.append(
            {
                "year": year,
                "value": value,
                "sensor": sensor,
                "has_survey": has_survey,
                "gap_fill_signal": gap_signal,
                "gap_fill_text": gap_text,
            }
        )

    return {"colony": colony_name, "index": index, "years": years_out}


# ---------------------------------------------------------------------------
# GET /colony/{slug}/map_frames
# ---------------------------------------------------------------------------


@router.get("/colony/{slug}/map_frames")
def get_map_frames(
    slug: str,
    index: str = Query(default="NDVI"),
    year: int = Query(...),
):
    if index not in VALID_INDICES:
        raise HTTPException(status_code=400, detail=f"index must be one of {VALID_INDICES}")

    registry = _load_registry()
    entry = registry.get(slug)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Colony '{slug}' not found")

    colony_name = entry.get("name") or entry.get("ColonyName", "")
    bbox = entry.get("bbox")  # [min_lon, min_lat, max_lon, max_lat]
    lat = entry.get("lat")
    lon = entry.get("lon")

    nc = _nc_path(slug, year)
    if nc is None:
        return {
            "type": "FeatureCollection",
            "features": [],
            "colony": colony_name,
            "year": year,
            "index": index,
            "no_data": True,
        }

    # Build a coarse 10×10 grid within the bbox
    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
    elif lat and lon:
        pad = 0.01
        min_lon, min_lat, max_lon, max_lat = lon - pad, lat - pad, lon + pad, lat + pad
    else:
        return {"type": "FeatureCollection", "features": [], "no_data": True}

    n_grid = 10
    lons = np.linspace(min_lon, max_lon, n_grid)
    lats = np.linspace(min_lat, max_lat, n_grid)

    # Try to read actual values from the NetCDF
    features = []
    try:
        import xarray as xr

        ds = xr.open_dataset(nc)
        da = ds.get(index)

        for flat in lats:
            for flon in lons:
                value = None
                if da is not None:
                    da2 = da
                    if "time" in da2.dims:
                        da2 = da2.median(dim="time", skipna=True)
                    try:
                        coord_dims = list(da2.dims)
                        if "lat" in coord_dims and "lon" in coord_dims:
                            v = float(da2.sel(lat=flat, lon=flon, method="nearest").values)
                        elif "y" in coord_dims and "x" in coord_dims:
                            v = float(da2.sel(y=flat, x=flon, method="nearest").values)
                        else:
                            v = float(da2.mean(skipna=True).values)
                        value = None if math.isnan(v) else v
                    except Exception:
                        value = None

                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [flon, flat]},
                        "properties": {"value": value},
                    }
                )
        ds.close()
    except Exception as e:
        logger.warning("map_frames: could not read %s: %s", nc, e)
        for flat in lats:
            for flon in lons:
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [flon, flat]},
                        "properties": {"value": None},
                    }
                )

    return {
        "type": "FeatureCollection",
        "features": features,
        "colony": colony_name,
        "year": year,
        "index": index,
    }


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str
    tags: list[str] = []
    top_k: int = 10
    colony: Optional[str] = None  # when set, scopes search to this colony only


@router.post("/search")
async def search_colonies(body: SearchRequest):
    """RAG-powered search. Scoped to a colony when `colony` is provided."""
    from rag_service import rag

    # Build enriched query incorporating tags
    enriched_query = body.query
    if body.tags:
        tag_context = {
            "#trend": "habitat change over time NDVI trend",
            "#survey": "survey data active nests species count",
            "#species": "species diversity dominant species Shannon",
        }
        extras = " ".join(tag_context.get(t, "") for t in body.tags if t in tag_context)
        if extras:
            enriched_query = f"{body.query} {extras}"

    # If a colony is specified, prepend it to the query for better embedding match
    # and apply a ChromaDB where-filter to restrict results to that colony
    where = None
    if body.colony:
        enriched_query = f"{body.colony}: {enriched_query}"
        where = {"colony": {"$eq": body.colony}}

    # ── Colony-scoped search: pass full chronological record to Claude ─────────
    if body.colony:
        survey_context = rag.get_colony_context(body.colony)
        sat_context = rag.get_colony_satellite_context(body.colony)
        colony_context = "\n\n".join(filter(None, [survey_context, sat_context]))

        ai_summary = None
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic

                prompt = (
                    f"You are analyzing colonial waterbird survey and satellite spectral data for {body.colony}.\n"
                    f"Below is the complete year-by-year survey record and satellite spectral index timeseries.\n"
                    f"Answer this question using specific numbers and years from the data: '{body.query}'\n"
                    f"Be concise (2-4 sentences). Do not guess — only use what is in the records.\n\n"
                    f"{colony_context}"
                )
                client = anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                ai_summary = msg.content[0].text
            except Exception as e:
                logger.warning("Claude colony summary failed: %s", e)

        # Still run RAG for the result cards (individual year matches)
        results = rag.query(enriched_query, top_k=body.top_k, where=where)
        output_results = []
        for r in results:
            slug = r["colony"].lower().replace(" ", "_").replace("/", "-")
            has_satellite = _nc_path(slug, r["year"]) is not None
            meta = r["metadata"]
            parts = []
            if meta.get("active_nests") is not None:
                parts.append(f"Active nests: {meta['active_nests']:.0f}")
            if meta.get("dominant_species"):
                parts.append(f"Dominant: {meta['dominant_species']}")
            if meta.get("colony_trajectory"):
                parts.append(f"Trajectory: {meta['colony_trajectory']}")
            if meta.get("active_nests_pct_change") is not None:
                parts.append(f"Nest Δ: {meta['active_nests_pct_change']:.1%}")
            output_results.append({
                "colony": r["colony"],
                "year": r["year"],
                "relevance_score": r["relevance_score"],
                "summary": ". ".join(parts) if parts else r["document"][:120],
                "has_satellite": has_satellite,
                "has_survey": True,
            })

        return {"results": output_results, "ai_summary": ai_summary}

    # ── Global search ─────────────────────────────────────────────────────────
    results = rag.query(enriched_query, top_k=body.top_k)

    output_results = []
    for r in results:
        slug = r["colony"].lower().replace(" ", "_").replace("/", "-")
        has_satellite = _nc_path(slug, r["year"]) is not None
        meta = r["metadata"]
        summary_parts = []
        if meta.get("active_nests") is not None:
            summary_parts.append(f"Active nests: {meta['active_nests']:.0f}")
        if meta.get("dominant_species"):
            summary_parts.append(f"Dominant species: {meta['dominant_species']}")
        if meta.get("colony_trajectory"):
            summary_parts.append(f"Trajectory: {meta['colony_trajectory']}")
        if meta.get("active_nests_pct_change") is not None:
            pct = meta["active_nests_pct_change"]
            summary_parts.append(f"Nest change: {pct:.1%}")

        output_results.append(
            {
                "colony": r["colony"],
                "year": r["year"],
                "relevance_score": r["relevance_score"],
                "summary": ". ".join(summary_parts) if summary_parts else r["document"][:120],
                "has_satellite": has_satellite,
                "has_survey": True,
            }
        )

    ai_summary = None
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key and output_results:
        try:
            import anthropic

            docs_text = "\n\n".join(r["document"] for r in results[:5])
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Based on the following colonial waterbird survey records across multiple sites, "
                        f"provide a 2-3 sentence summary answering: '{body.query}'\n\n{docs_text}"
                    ),
                }],
            )
            ai_summary = msg.content[0].text
        except Exception as e:
            logger.warning("Claude summary failed: %s", e)

    return {"results": output_results, "ai_summary": ai_summary}


# ---------------------------------------------------------------------------
# POST /chat  — streaming SSE chatbot
# ---------------------------------------------------------------------------

_GRANT_KEYWORDS = {
    "grant", "abstract", "significance", "objectives", "budget",
    "noaa", "nsf", "fws", "restore", "ldwf", "funding", "proposal",
    "narrative", "expected outcomes",
}

_AGENCY_FRAMING = {
    "noaa": "Frame the response around fisheries habitat linkage, blue carbon, and coastal resilience.",
    "nsf": "Emphasise novel methods, reproducibility, broader impacts, and interdisciplinary value.",
    "fws": "Focus on species recovery, ESA compliance, and migratory bird treaty obligations.",
    "restore": "Connect to Gulf Coast ecosystem restoration and post-Deepwater Horizon recovery goals.",
    "ldwf": "Align with Louisiana coastal master plan, state priorities, and local economic benefits.",
}

_BASE_SYSTEM = """\
You are a research assistant for the Egret ecological monitoring platform.
You have access to TWI (The Water Institute) avian survey data and Egret \
satellite spectral index data for Gulf Coast colonial waterbird sites.

Answer questions ONLY about:
- TWI biological survey data (species counts, nest success, Shannon diversity, \
colony trajectories, dominant species, total birds, active nests)
- Egret spectral indices (NDVI, MNDWI, NDWI, NDMI, water_mask) and what \
they indicate about habitat condition at the selected colony

Do NOT answer general ecology questions, species biology questions, or questions \
unrelated to the TWI data and Egret indices. If asked, respond:
"I can only answer questions about the TWI survey data and spectral indices shown in Egret."

Never fabricate statistics. Only cite numbers present in the context below.
For grant writing requests, use only data from the context. \
Supported agencies: NOAA, NSF, FWS, RESTORE Act, LDWF.

CRITICAL — context rules:
- If no colony context is provided and the user asks for specific numbers, \
trends, or time-series data, do NOT attempt to answer from the small RAG sample. \
Instead, tell the user to select a colony on the map first. The RAG sample is \
only a few records and is not representative of all 442 colonies.
- Only answer aggregate/trend questions when the full colony context block is present.
- If the context shows "Active nests: 0.0" for all records, note this explicitly \
and do not speculate about causes beyond what the data shows.
- Keep responses concise. Do not use tables unless the data has 3+ meaningful rows.
"""


def _build_system_prompt(colony_context: str, rag_context: str, last_message: str) -> str:
    prompt = _BASE_SYSTEM

    if not colony_context:
        prompt += (
            "\n\nNO COLONY SELECTED: The user has not selected a colony on the map. "
            "You only have a small RAG sample (a few records) — not the full dataset. "
            "For any question about specific numbers, trends over time, or per-colony analysis, "
            "ask the user to select a colony from the map first. "
            "You may answer general questions about the dataset structure or suggest what to look for."
        )

    # Detect grant intent and inject agency framing
    msg_lower = last_message.lower()
    if any(kw in msg_lower for kw in _GRANT_KEYWORDS):
        for agency, framing in _AGENCY_FRAMING.items():
            if agency in msg_lower:
                prompt += f"\n\nGRANT WRITING GUIDANCE:\n{framing}"
                break

    if colony_context:
        prompt += f"\n\n--- SELECTED COLONY DATA ---\n{colony_context}"
    if rag_context:
        prompt += f"\n\n--- RELATED COLONY DATA (for cross-colony context) ---\n{rag_context}"

    return prompt


class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    colony_slug: Optional[str] = None


@router.post("/chat")
async def chat(body: ChatRequest):
    """
    Streaming SSE chatbot. Injects colony context when colony_slug is provided.
    Returns text/event-stream with chunks: data: <text>\\n\\n  and data: [DONE]\\n\\n
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not set")

    if not body.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    import anthropic
    from rag_service import rag

    # ── Build colony context ──────────────────────────────────────────────────
    colony_context = ""
    if body.colony_slug:
        colony_name = _slug_to_colony_name(body.colony_slug)
        if colony_name:
            try:
                survey_ctx = rag.get_colony_context(colony_name)
                sat_ctx = rag.get_colony_satellite_context(colony_name)
                colony_context = "\n\n".join(filter(None, [survey_ctx, sat_ctx]))
            except Exception as e:
                logger.warning("get_colony_context failed: %s", e)

    # ── RAG: top 5 docs relevant to last user message ─────────────────────────
    rag_context = ""
    last_user = next(
        (m.content for m in reversed(body.messages) if m.role == "user"), ""
    )
    if last_user:
        try:
            rag_results = rag.query(last_user, top_k=5)
            # Exclude docs from the selected colony (already in colony_context)
            selected_name = _slug_to_colony_name(body.colony_slug or "") or ""
            cross_colony = [r for r in rag_results if r["colony"] != selected_name][:3]
            if cross_colony:
                rag_context = "\n\n".join(r["document"] for r in cross_colony)
        except Exception as e:
            logger.warning("RAG query failed: %s", e)

    system_prompt = _build_system_prompt(colony_context, rag_context, last_user)

    # ── Stream via SSE ────────────────────────────────────────────────────────
    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            client = anthropic.Anthropic(api_key=api_key)
            messages = [{"role": m.role, "content": m.content} for m in body.messages]

            with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps(text)}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error("Chat stream error: %s", e)
            yield f"data: {json.dumps('[ERROR] ' + str(e))}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
