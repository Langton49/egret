"""
=============================================================================
  Build TWI Feature Table
  Aggregates avianData20102021.csv.gz to (colony, year) level
=============================================================================

Produces one row per colony per year with:
  - Colony identity and location
  - Biological features: nest counts, bird counts, species diversity
  - Survey metadata: number of photos dotted, survey months

Output:
    training_two/twi_features.csv
"""

import os
import numpy as np
import pandas as pd

CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "repo", "data", "databases", "processed", "avianData20102021.csv.gz"
)
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "twi_features.csv")


def shannon_diversity(counts):
    """Shannon diversity index from a series of species counts."""
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    total = counts.sum()
    p = counts / total
    return float(-np.sum(p * np.log(p)))


def build_features(csv_path: str) -> pd.DataFrame:
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  {len(df):,} rows, {df['ColonyName'].nunique()} colonies, "
          f"years {df['Year'].min()}-{df['Year'].max()}")

    # --- Per colony/year/species aggregation ---
    # Each row is one species on one photo — sum across all photos per colony/year
    species_agg = (
        df.groupby(["ColonyName", "Year", "SpeciesCode"], as_index=False)
        .agg(
            species_total_birds=("total_birds", "sum"),
            species_total_nests=("total_nests", "sum"),
            species_WBN=("WBN", "sum"),
            species_PBN=("PBN", "sum"),
            species_site=("Site", "sum"),
            species_brood=("Brood", "sum"),
            species_chicks=("ChicksNestlings", "sum"),
            species_aband_nests=("AbandNest", "sum"),
            species_empty_nests=("EmptyNest", "sum"),
        )
    )

    # --- Colony/year level aggregation ---
    colony_year = (
        df.groupby(["ColonyName", "Year"], as_index=False)
        .agg(
            lat=("Latitude_x", "first"),
            lon=("Longitude_x", "first"),
            geo_region=("GeoRegion", "first"),
            state=("State", "first"),
            primary_habitat=("PrimaryHabitat", "first"),
            total_birds=("total_birds", "sum"),
            total_nests=("total_nests", "sum"),
            total_WBN=("WBN", "sum"),
            total_PBN=("PBN", "sum"),
            total_site=("Site", "sum"),
            total_brood=("Brood", "sum"),
            total_chicks=("ChicksNestlings", "sum"),
            total_aband_nests=("AbandNest", "sum"),
            total_empty_nests=("EmptyNest", "sum"),
            n_photos=("PhotoNumber", "count"),
            n_dotting_areas=("DottingAreaNumber", "nunique"),
            survey_months=("month", lambda x: ",".join(sorted(x.dropna().unique()))),
        )
    )

    # --- Species diversity per colony/year ---
    print("Computing species diversity ...")
    diversity_rows = []
    for (colony, year), grp in species_agg.groupby(["ColonyName", "Year"]):
        n_species = grp["SpeciesCode"].nunique()
        species_birds = grp.groupby("SpeciesCode")["species_total_birds"].sum()
        shannon = shannon_diversity(species_birds)

        # Dominant species by bird count
        dominant = (
            species_birds.idxmax() if len(species_birds) > 0 else None
        )

        diversity_rows.append({
            "ColonyName": colony,
            "Year": year,
            "n_species": n_species,
            "shannon_diversity": round(shannon, 4),
            "dominant_species": dominant,
        })

    diversity_df = pd.DataFrame(diversity_rows)

    # --- Nest success ratio ---
    # WBN = with-bird nest (active), AbandNest = abandoned
    # Ratio of active to total attempted nests — proxy for breeding success
    colony_year["nest_success_ratio"] = np.where(
        (colony_year["total_WBN"] + colony_year["total_aband_nests"]) > 0,
        colony_year["total_WBN"] /
        (colony_year["total_WBN"] + colony_year["total_aband_nests"]),
        np.nan
    )

    # --- Nesting intensity ---
    # Total active nests (WBN + PBN) — strongest signal of colony health
    colony_year["active_nests"] = colony_year["total_WBN"] + colony_year["total_PBN"]

    # --- Merge diversity ---
    features = colony_year.merge(diversity_df, on=["ColonyName", "Year"], how="left")

    # --- Year-over-year change columns ---
    # For each colony, compute change in active_nests and total_birds vs prior year
    features = features.sort_values(["ColonyName", "Year"])

    features["active_nests_prev"] = (
        features.groupby("ColonyName")["active_nests"].shift(1)
    )
    features["active_nests_delta"] = (
        features["active_nests"] - features["active_nests_prev"]
    )
    features["active_nests_pct_change"] = np.where(
        features["active_nests_prev"] > 0,
        features["active_nests_delta"] / features["active_nests_prev"],
        np.nan
    )

    features["total_birds_prev"] = (
        features.groupby("ColonyName")["total_birds"].shift(1)
    )
    features["total_birds_delta"] = (
        features["total_birds"] - features["total_birds_prev"]
    )

    # --- Trajectory label ---
    # Based on year-over-year active nest change
    # Used later as a soft signal for Stage 2 target validation
    def trajectory(pct):
        if pd.isna(pct):
            return "unknown"
        if pct > 0.10:
            return "increasing"
        if pct < -0.10:
            return "declining"
        return "stable"

    features["colony_trajectory"] = features["active_nests_pct_change"].apply(trajectory)

    # Clean up helper columns
    features = features.drop(columns=["active_nests_prev", "total_birds_prev"])

    # --- Reorder columns ---
    id_cols = ["ColonyName", "Year", "lat", "lon", "geo_region", "state",
               "primary_habitat", "survey_months"]
    bio_cols = ["total_birds", "total_nests", "active_nests", "total_WBN",
                "total_PBN", "total_site", "total_brood", "total_chicks",
                "total_aband_nests", "total_empty_nests", "nest_success_ratio",
                "n_photos", "n_dotting_areas"]
    diversity_cols = ["n_species", "shannon_diversity", "dominant_species"]
    trend_cols = ["active_nests_delta", "active_nests_pct_change",
                  "total_birds_delta", "colony_trajectory"]

    features = features[id_cols + bio_cols + diversity_cols + trend_cols]

    return features


def main():
    features = build_features(CSV_PATH)

    print(f"\nFeature table shape: {features.shape}")
    print(f"Colony/year combinations: {len(features):,}")
    print(f"Unique colonies: {features['ColonyName'].nunique()}")
    print(f"Years: {features['Year'].min()} - {features['Year'].max()}")
    print(f"\nTrajectory distribution:")
    print(features["colony_trajectory"].value_counts().to_string())
    print(f"\nActive nests stats:")
    print(features["active_nests"].describe().round(1).to_string())
    print(f"\nShannon diversity stats:")
    print(features["shannon_diversity"].describe().round(3).to_string())

    features.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    # Preview
    print(f"\nSample rows:")
    sample = features[features["ColonyName"] == "Queen Bess Island"].sort_values("Year")
    if len(sample):
        print(sample[["ColonyName", "Year", "total_birds", "active_nests",
                       "n_species", "shannon_diversity", "colony_trajectory"]].to_string(index=False))


if __name__ == "__main__":
    main()
