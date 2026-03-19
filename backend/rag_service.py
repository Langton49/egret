"""
RAG service for the Research Repository.

Loads twi_features.csv and colony_satellite_features.csv into a local ChromaDB
collection using all-MiniLM-L6-v2 embeddings. Re-embeds automatically if either
CSV has been modified since the last embed run (checked via mtime sentinel files).

Usage:
    python rag_service.py          # runs quick self-test
    from rag_service import rag    # import singleton in other modules
"""

import os
import json
import math
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent
_TWI_CSV = _REPO_ROOT / "training_two" / "twi_features.csv"
_SAT_CSV = _REPO_ROOT / "training_two" / "colony_satellite_features.csv"
_CHROMA_DIR = _HERE / "data_cache" / "rag_chroma"
_SENTINEL = _CHROMA_DIR / ".last_embed_mtime"
_SAT_SENTINEL = _CHROMA_DIR / ".last_sat_embed_mtime"
_COLLECTION_NAME = "twi_colony_years"


def _safe_float(val, fmt=None):
    """Return formatted float or 'N/A' for NaN/None."""
    try:
        f = float(val)
        if math.isnan(f):
            return "N/A"
        return f"{f:{fmt}}" if fmt else str(f)
    except (TypeError, ValueError):
        return "N/A"


def _row_to_document(row: pd.Series) -> str:
    """Convert a twi_features row to a plain-text embedding document."""
    pct = _safe_float(row.get("active_nests_pct_change"), ".1%")
    return (
        f"Colony: {row['ColonyName']} | Year: {row['Year']} "
        f"| Region: {row.get('geo_region', 'N/A')} | State: {row.get('state', 'N/A')}\n"
        f"Habitat: {row.get('primary_habitat', 'N/A')} "
        f"| Survey months: {row.get('survey_months', 'N/A')}\n"
        f"Total birds: {_safe_float(row.get('total_birds'))} "
        f"| Active nests: {_safe_float(row.get('active_nests'))} "
        f"| Species: {row.get('n_species', 'N/A')}\n"
        f"Dominant species: {row.get('dominant_species', 'N/A')} "
        f"| Shannon diversity: {_safe_float(row.get('shannon_diversity'))}\n"
        f"Nest success: {_safe_float(row.get('nest_success_ratio'))} "
        f"| Trajectory: {row.get('colony_trajectory', 'N/A')}\n"
        f"Active nest change: {pct} vs prior year"
    )


def _colony_summary_document(grp: pd.DataFrame) -> str:
    """
    Generate a natural-language summary document for one colony across all its
    survey years. Written in prose so embedding models capture semantic meaning:
    "long-term declining colony", "peaked in 2005", etc.
    """
    name = grp["ColonyName"].iloc[0]
    state = grp["state"].dropna().iloc[0] if grp["state"].notna().any() else "unknown state"
    region = grp["geo_region"].dropna().iloc[0] if grp["geo_region"].notna().any() else "unknown region"
    habitat = grp["primary_habitat"].dropna().iloc[0] if grp["primary_habitat"].notna().any() else "unknown habitat"

    years = sorted(grp["Year"].dropna().astype(int).tolist())
    n_surveys = len(years)

    nest_data = grp[["Year", "active_nests"]].dropna().sort_values("Year")
    bird_data = grp[["Year", "total_birds"]].dropna().sort_values("Year")
    trajectories = [t for t in grp["colony_trajectory"].tolist() if isinstance(t, str) and t != "unknown"]

    # ── Overall trajectory ───────────────────────────────────────────────────
    if len(nest_data) >= 2:
        first_nests = float(nest_data.iloc[0]["active_nests"])
        last_nests = float(nest_data.iloc[-1]["active_nests"])
        pct_change = ((last_nests - first_nests) / first_nests * 100) if first_nests > 0 else 0.0
        if last_nests < first_nests * 0.85:
            overall_traj = "declining"
        elif last_nests > first_nests * 1.15:
            overall_traj = "increasing"
        else:
            overall_traj = "stable"
    elif trajectories:
        overall_traj = max(set(trajectories), key=trajectories.count)
        pct_change = None
    else:
        overall_traj = "unknown"
        pct_change = None

    n_declining = trajectories.count("declining")
    n_increasing = trajectories.count("increasing")
    n_stable = trajectories.count("stable")

    # ── Nest peak / trough ───────────────────────────────────────────────────
    peak_str = low_str = recent_str = ""
    if not nest_data.empty:
        peak_idx = nest_data["active_nests"].idxmax()
        peak_nests = int(nest_data.loc[peak_idx, "active_nests"])
        peak_year = int(nest_data.loc[peak_idx, "Year"])

        low_idx = nest_data["active_nests"].idxmin()
        low_nests = int(nest_data.loc[low_idx, "active_nests"])
        low_year = int(nest_data.loc[low_idx, "Year"])

        last_row = nest_data.iloc[-1]
        recent_str = (
            f"Most recent survey ({int(last_row['Year'])}): {int(last_row['active_nests'])} active nests."
        )
        peak_str = f"Active nests peaked at {peak_nests} in {peak_year}."
        low_str = f"Lowest recorded: {low_nests} nests in {low_year}."

    # ── Species ──────────────────────────────────────────────────────────────
    sp_counts = grp["dominant_species"].dropna().value_counts()
    primary_sp = sp_counts.index[0] if not sp_counts.empty else None
    all_sp = sp_counts.index.tolist()
    sp_str = ""
    if primary_sp:
        if len(all_sp) > 1:
            sp_str = (
                f"Most frequently dominant species: {primary_sp}. "
                f"All dominant species recorded: {', '.join(all_sp)}."
            )
        else:
            sp_str = f"Dominant species: {primary_sp}."

    # ── Diversity and success ─────────────────────────────────────────────────
    avg_div = grp["shannon_diversity"].mean()
    avg_suc = grp["nest_success_ratio"].mean()
    avg_nsp = grp["n_species"].mean()
    div_str = ""
    if not math.isnan(avg_div):
        div_str = f"Average Shannon diversity: {avg_div:.2f}. Average species count: {avg_nsp:.1f}."
    suc_str = ""
    if not math.isnan(avg_suc):
        suc_str = f"Average nest success ratio: {avg_suc:.2f}."

    # ── Assemble ─────────────────────────────────────────────────────────────
    parts = []

    if n_surveys == 1:
        yr = years[0]
        nests_val = int(nest_data.iloc[0]["active_nests"]) if not nest_data.empty else None
        birds_val = int(bird_data.iloc[0]["total_birds"]) if not bird_data.empty else None
        parts.append(
            f"{name} is a {habitat} colony in {state}, {region}. "
            f"It has been surveyed once, in {yr}."
        )
        if nests_val is not None:
            parts.append(f"That survey recorded {nests_val} active nests"
                         + (f" and {birds_val} total birds." if birds_val is not None else "."))
        if sp_str:
            parts.append(sp_str)
        if div_str:
            parts.append(div_str)
    else:
        parts.append(
            f"{name} is a {habitat} colony in {state}, {region}. "
            f"It has been surveyed {n_surveys} times between {years[0]} and {years[-1]} "
            f"(survey years: {', '.join(str(y) for y in years)})."
        )
        traj_detail = (
            f"In {n_declining} of {n_surveys} surveys the colony was declining, "
            f"{n_increasing} increasing, {n_stable} stable."
        )
        pct_str = (
            f" Net nest change first to last survey: {pct_change:+.1f}%." if pct_change is not None else ""
        )
        parts.append(
            f"The colony is overall {overall_traj} across its full survey history. "
            + traj_detail + pct_str
        )
        if peak_str:
            parts.append(f"{peak_str} {low_str} {recent_str}")
        if sp_str:
            parts.append(sp_str)
        if div_str:
            parts.append(div_str)
        if suc_str:
            parts.append(suc_str)

    return " ".join(p.strip() for p in parts if p.strip())


def _colony_summary_metadata(grp: pd.DataFrame) -> dict:
    """Metadata dict for a colony summary document."""
    def _s(v):
        if v is None:
            return ""
        try:
            if math.isnan(float(v)):
                return ""
        except (TypeError, ValueError):
            pass
        return str(v)

    def _f(v):
        try:
            f = float(v)
            return None if math.isnan(f) else round(f, 4)
        except (TypeError, ValueError):
            return None

    years = sorted(grp["Year"].dropna().astype(int).tolist())
    nest_data = grp[["Year", "active_nests"]].dropna().sort_values("Year")

    peak_nests = peak_year = low_nests = low_year = last_nests = last_year = None
    overall_pct = None
    if not nest_data.empty:
        pi = nest_data["active_nests"].idxmax()
        peak_nests = _f(nest_data.loc[pi, "active_nests"])
        peak_year = int(nest_data.loc[pi, "Year"])
        li = nest_data["active_nests"].idxmin()
        low_nests = _f(nest_data.loc[li, "active_nests"])
        low_year = int(nest_data.loc[li, "Year"])
        last_nests = _f(nest_data.iloc[-1]["active_nests"])
        last_year = int(nest_data.iloc[-1]["Year"])
        if len(nest_data) >= 2:
            fn = float(nest_data.iloc[0]["active_nests"])
            ln = float(nest_data.iloc[-1]["active_nests"])
            overall_pct = _f(((ln - fn) / fn * 100) if fn > 0 else None)

    sp_counts = grp["dominant_species"].dropna().value_counts()

    return {
        "doc_type": "colony_summary",
        "colony": _s(grp["ColonyName"].iloc[0]),
        "state": _s(grp["state"].dropna().iloc[0] if grp["state"].notna().any() else None),
        "geo_region": _s(grp["geo_region"].dropna().iloc[0] if grp["geo_region"].notna().any() else None),
        "primary_habitat": _s(grp["primary_habitat"].dropna().iloc[0] if grp["primary_habitat"].notna().any() else None),
        "lat": _f(grp["lat"].dropna().iloc[0] if grp["lat"].notna().any() else None),
        "lon": _f(grp["lon"].dropna().iloc[0] if grp["lon"].notna().any() else None),
        "n_surveys": len(years),
        "first_survey_year": years[0] if years else 0,
        "last_survey_year": years[-1] if years else 0,
        "survey_years_str": ",".join(str(y) for y in years),
        "peak_nests": peak_nests,
        "peak_year": peak_year,
        "low_nests": low_nests,
        "low_year": low_year,
        "last_nests": last_nests,
        "last_survey_year_nests": last_year,
        "overall_pct_change": overall_pct,
        "avg_shannon_diversity": _f(grp["shannon_diversity"].mean()),
        "avg_nest_success": _f(grp["nest_success_ratio"].mean()),
        "avg_n_species": _f(grp["n_species"].mean()),
        "primary_species": _s(sp_counts.index[0] if not sp_counts.empty else None),
        "all_dominant_species": ",".join(sp_counts.index.tolist()),
    }


def _row_to_metadata(row: pd.Series) -> dict:
    """Build ChromaDB metadata dict from a row (must be JSON-serialisable scalars)."""

    def _s(v):
        if v is None:
            return ""
        try:
            if math.isnan(float(v)):
                return ""
        except (TypeError, ValueError):
            pass
        return str(v)

    def _f(v):
        try:
            f = float(v)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    def _i(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    return {
        "colony": _s(row.get("ColonyName")),
        "year": _i(row.get("Year")) or 0,
        "geo_region": _s(row.get("geo_region")),
        "state": _s(row.get("state")),
        "primary_habitat": _s(row.get("primary_habitat")),
        "total_birds": _f(row.get("total_birds")),
        "active_nests": _f(row.get("active_nests")),
        "n_species": _i(row.get("n_species")),
        "dominant_species": _s(row.get("dominant_species")),
        "shannon_diversity": _f(row.get("shannon_diversity")),
        "nest_success_ratio": _f(row.get("nest_success_ratio")),
        "colony_trajectory": _s(row.get("colony_trajectory")),
        "active_nests_pct_change": _f(row.get("active_nests_pct_change")),
    }


# ---------------------------------------------------------------------------
# Satellite document helpers
# ---------------------------------------------------------------------------

def _ndvi_label(val: float | None) -> str:
    if val is None:
        return "N/A"
    if val >= 0.4:
        return "dense vegetation"
    if val >= 0.2:
        return "moderate vegetation"
    if val >= 0.05:
        return "sparse vegetation"
    if val >= -0.05:
        return "bare soil / low cover"
    return "water / non-vegetated"


def _mndwi_label(val: float | None) -> str:
    if val is None:
        return "N/A"
    if val >= 0.2:
        return "open water dominant"
    if val >= 0.0:
        return "wet / water-influenced"
    return "non-water surface"


def _sat_row_to_document(row: pd.Series) -> str:
    """Convert a satellite features row to a plain-text embedding document."""
    ndvi = _safe_float(row.get("NDVI"), ".3f")
    mndwi = _safe_float(row.get("MNDWI"), ".3f")
    ndwi = _safe_float(row.get("NDWI"), ".3f")
    ndmi = _safe_float(row.get("NDMI"), ".3f")
    water = _safe_float(row.get("water_mask"), ".2f")

    ndvi_val = None if ndvi == "N/A" else float(row.get("NDVI", float("nan")))
    mndwi_val = None if mndwi == "N/A" else float(row.get("MNDWI", float("nan")))

    return (
        f"Colony: {row['colony_name']} | Year: {int(row['year'])} | Sensor: {row.get('sensor', 'N/A')}\n"
        f"Spectral indices (300m centroid): NDVI={ndvi}, MNDWI={mndwi}, NDWI={ndwi}, NDMI={ndmi}, water_mask={water}\n"
        f"NDVI interpretation: {_ndvi_label(ndvi_val)} | MNDWI interpretation: {_mndwi_label(mndwi_val)}"
    )


def _sat_colony_summary_document(grp: pd.DataFrame) -> str:
    """
    Generate a natural-language spectral timeseries summary for one colony.
    Describes NDVI trend, peak/low years, and habitat trajectory from satellite data.
    """
    name = grp["colony_name"].iloc[0]
    years = sorted(grp["year"].dropna().astype(int).tolist())
    sensors = grp["sensor"].dropna().unique().tolist()

    ndvi_data = grp[["year", "NDVI"]].dropna(subset=["NDVI"]).sort_values("year")
    mndwi_data = grp[["year", "MNDWI"]].dropna(subset=["MNDWI"]).sort_values("year")

    parts = [
        f"{name} satellite spectral timeseries covers {len(years)} years "
        f"({years[0]}–{years[-1]}) using {', '.join(sensors)} imagery."
    ]

    if len(ndvi_data) >= 2:
        first_ndvi = float(ndvi_data.iloc[0]["NDVI"])
        last_ndvi = float(ndvi_data.iloc[-1]["NDVI"])
        delta = last_ndvi - first_ndvi

        peak_idx = ndvi_data["NDVI"].idxmax()
        low_idx = ndvi_data["NDVI"].idxmin()
        peak_val = float(ndvi_data.loc[peak_idx, "NDVI"])
        peak_yr = int(ndvi_data.loc[peak_idx, "year"])
        low_val = float(ndvi_data.loc[low_idx, "NDVI"])
        low_yr = int(ndvi_data.loc[low_idx, "year"])

        if delta > 0.05:
            traj = "increasing vegetation cover (improving habitat)"
        elif delta < -0.05:
            traj = "declining vegetation cover (degrading habitat)"
        else:
            traj = "relatively stable vegetation cover"

        parts.append(
            f"NDVI trend: {traj}. "
            f"NDVI ranged from {low_val:.3f} ({_ndvi_label(low_val)}) in {low_yr} "
            f"to {peak_val:.3f} ({_ndvi_label(peak_val)}) in {peak_yr}. "
            f"Most recent NDVI ({int(ndvi_data.iloc[-1]['year'])}): {last_ndvi:.3f} ({_ndvi_label(last_ndvi)})."
        )

    if len(mndwi_data) >= 2:
        avg_mndwi = float(mndwi_data["MNDWI"].mean())
        parts.append(
            f"Mean MNDWI={avg_mndwi:.3f}: {_mndwi_label(avg_mndwi)} conditions on average."
        )

    return " ".join(parts)


def _sat_row_to_metadata(row: pd.Series) -> dict:
    def _f(v):
        try:
            f = float(v)
            return None if math.isnan(f) else round(f, 4)
        except (TypeError, ValueError):
            return None

    return {
        "doc_type": "satellite_year",
        "colony": str(row.get("colony_name", "")),
        "slug": str(row.get("slug", "")),
        "year": int(row["year"]) if not math.isnan(float(row["year"])) else 0,
        "sensor": str(row.get("sensor", "")),
        "NDVI": _f(row.get("NDVI")),
        "MNDWI": _f(row.get("MNDWI")),
        "NDWI": _f(row.get("NDWI")),
        "NDMI": _f(row.get("NDMI")),
        "water_mask": _f(row.get("water_mask")),
    }


def _sat_colony_summary_metadata(grp: pd.DataFrame) -> dict:
    def _f(v):
        try:
            f = float(v)
            return None if math.isnan(f) else round(f, 4)
        except (TypeError, ValueError):
            return None

    years = sorted(grp["year"].dropna().astype(int).tolist())
    ndvi_data = grp["NDVI"].dropna()

    return {
        "doc_type": "satellite_summary",
        "colony": str(grp["colony_name"].iloc[0]),
        "slug": str(grp["slug"].iloc[0]) if "slug" in grp.columns else "",
        "first_sat_year": years[0] if years else 0,
        "last_sat_year": years[-1] if years else 0,
        "n_sat_years": len(years),
        "mean_NDVI": _f(ndvi_data.mean()) if not ndvi_data.empty else None,
        "max_NDVI": _f(ndvi_data.max()) if not ndvi_data.empty else None,
        "min_NDVI": _f(ndvi_data.min()) if not ndvi_data.empty else None,
        "mean_MNDWI": _f(grp["MNDWI"].dropna().mean()) if "MNDWI" in grp.columns else None,
    }


class RagService:
    """Singleton wrapper around a local ChromaDB collection."""

    def __init__(self):
        self._client = None
        self._collection = None
        self._df: pd.DataFrame | None = None
        self._sat_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init(self):
        """Load CSVs, build/reload ChromaDB if stale, and expose the collection."""
        import chromadb
        from chromadb.utils import embedding_functions

        _CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        self._df = pd.read_csv(_TWI_CSV)
        self._sat_df = pd.read_csv(_SAT_CSV) if _SAT_CSV.exists() else pd.DataFrame()

        self._client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        if self._needs_rebuild():
            logger.info("RAG: rebuilding ChromaDB collection from %s + %s", _TWI_CSV, _SAT_CSV)
            self._build_collection()
            self._write_sentinels()
        else:
            logger.info(
                "RAG: collection up-to-date (%d docs)", self._collection.count()
            )

    def _needs_rebuild(self) -> bool:
        if not _SENTINEL.exists():
            return True
        if self._collection.count() == 0:
            return True
        try:
            stored_twi = float(_SENTINEL.read_text().strip())
            if _TWI_CSV.stat().st_mtime > stored_twi:
                return True
        except Exception:
            return True
        # Check satellite CSV sentinel
        if _SAT_CSV.exists():
            try:
                stored_sat = float(_SAT_SENTINEL.read_text().strip()) if _SAT_SENTINEL.exists() else 0.0
                if _SAT_CSV.stat().st_mtime > stored_sat:
                    return True
            except Exception:
                return True
        return False

    def _build_collection(self):
        # Delete existing docs if any
        try:
            existing = self._collection.get()
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])
        except Exception:
            pass

        documents, metadatas, ids = [], [], []

        # ── Per-year survey documents ────────────────────────────────────────
        for _, row in self._df.iterrows():
            doc_id = (
                f"{row['ColonyName']}_{row['Year']}"
                .replace(" ", "_").replace("/", "-")
            )
            meta = _row_to_metadata(row)
            meta["doc_type"] = "survey_year"
            documents.append(_row_to_document(row))
            metadatas.append(meta)
            ids.append(doc_id)

        # ── Colony summary documents (one per colony) ────────────────────────
        for colony_name, grp in self._df.groupby("ColonyName"):
            doc_id = f"_summary_{colony_name}".replace(" ", "_").replace("/", "-")
            documents.append(_colony_summary_document(grp))
            metadatas.append(_colony_summary_metadata(grp))
            ids.append(doc_id)

        n_survey_year = len(self._df)
        n_survey_summary = self._df["ColonyName"].nunique()

        # ── Per-year satellite documents ─────────────────────────────────────
        n_sat_year = 0
        n_sat_summary = 0
        if not self._sat_df.empty:
            for _, row in self._sat_df.iterrows():
                doc_id = (
                    f"_sat_{row.get('slug', row['colony_name'])}_{int(row['year'])}"
                    .replace(" ", "_").replace("/", "-")
                )
                documents.append(_sat_row_to_document(row))
                metadatas.append(_sat_row_to_metadata(row))
                ids.append(doc_id)
                n_sat_year += 1

            # ── Colony satellite summary documents (one per colony) ───────────
            group_col = "colony_name" if "colony_name" in self._sat_df.columns else "slug"
            for colony_name, grp in self._sat_df.groupby(group_col):
                doc_id = (
                    f"_sat_summary_{grp['slug'].iloc[0] if 'slug' in grp.columns else colony_name}"
                    .replace(" ", "_").replace("/", "-")
                )
                documents.append(_sat_colony_summary_document(grp))
                metadatas.append(_sat_colony_summary_metadata(grp))
                ids.append(doc_id)
                n_sat_summary += 1

        # Upsert in batches of 100
        batch = 100
        for start in range(0, len(documents), batch):
            self._collection.upsert(
                documents=documents[start : start + batch],
                metadatas=metadatas[start : start + batch],
                ids=ids[start : start + batch],
            )

        logger.info(
            "RAG: embedded %d survey-year + %d survey-summary + %d sat-year + %d sat-summary = %d total docs",
            n_survey_year, n_survey_summary, n_sat_year, n_sat_summary,
            n_survey_year + n_survey_summary + n_sat_year + n_sat_summary,
        )

    def _write_sentinels(self):
        _SENTINEL.write_text(str(_TWI_CSV.stat().st_mtime))
        if _SAT_CSV.exists():
            _SAT_SENTINEL.write_text(str(_SAT_CSV.stat().st_mtime))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, text: str, top_k: int = 10, where: dict | None = None) -> list[dict]:
        """
        Embed `text` and return top_k matching colony/year dicts.
        Each dict has keys: colony, year, relevance_score, document, metadata.
        Pass `where` to scope results to a specific colony, e.g. {"colony": {"$eq": "Bird Island"}}.
        """
        if self._collection is None:
            raise RuntimeError("RagService.init() not called")

        count = self._collection.count()
        kwargs = dict(
            query_texts=[text],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        out = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance → similarity score (1 - dist)
            out.append(
                {
                    "colony": meta.get("colony", ""),
                    "year": meta.get("year", 0),
                    "relevance_score": round(1.0 - dist, 4),
                    "document": doc,
                    "metadata": meta,
                }
            )

        return out

    def get_colony_context(self, colony_name: str) -> str:
        """
        Return a formatted string of all survey records for `colony_name`
        suitable for injection into a Claude prompt.
        """
        if self._df is None:
            raise RuntimeError("RagService.init() not called")

        rows = self._df[self._df["ColonyName"] == colony_name].sort_values("Year")
        if rows.empty:
            return f"No survey data found for colony: {colony_name}"

        parts = [f"Colony survey records for {colony_name}:\n"]
        for _, row in rows.iterrows():
            parts.append(_row_to_document(row))
            parts.append("")  # blank line between records

        return "\n".join(parts)

    def get_colony_satellite_context(self, colony_name: str) -> str:
        """
        Return a formatted string of all satellite spectral index records for
        `colony_name`, suitable for injection into a Claude prompt.
        """
        if self._sat_df is None or self._sat_df.empty:
            return ""

        # Match by colony_name or slug
        slug = colony_name.lower().replace(" ", "_").replace("/", "-")
        if "colony_name" not in self._sat_df.columns:
            return ""
        name_mask = self._sat_df["colony_name"] == colony_name
        slug_mask = (self._sat_df["slug"] == slug) if "slug" in self._sat_df.columns else pd.Series(False, index=self._sat_df.index)
        rows = self._sat_df[name_mask | slug_mask].sort_values("year")

        if rows.empty:
            return f"No satellite spectral data found for colony: {colony_name}"

        parts = [f"Satellite spectral index timeseries for {colony_name}:\n"]
        for _, row in rows.iterrows():
            parts.append(_sat_row_to_document(row))
            parts.append("")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
rag = RagService()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rag.init()
    print("\n--- Top 5 results for: 'declining waterbird colonies Louisiana' ---\n")
    results = rag.query("declining waterbird colonies Louisiana", top_k=5)
    for r in results:
        print(
            f"  [{r['relevance_score']:.3f}] {r['colony']} {r['year']} "
            f"– type: {r['metadata'].get('doc_type')} trajectory: {r['metadata'].get('colony_trajectory')}"
        )
    print("\n--- Top 5 results for: 'NDVI vegetation decline habitat degradation' ---\n")
    results = rag.query("NDVI vegetation decline habitat degradation", top_k=5)
    for r in results:
        print(
            f"  [{r['relevance_score']:.3f}] {r['colony']} {r['year']} "
            f"– type: {r['metadata'].get('doc_type')} NDVI: {r['metadata'].get('NDVI')}"
        )
    print("\nDone.")
