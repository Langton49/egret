"""
=============================================================================
  Supervised Habitat Suitability Model
  Mississippi Delta Avian Habitat Analysis
=============================================================================

Learns which spectral signatures predict bird presence and diversity.
Trained on cell × dekad rows where bird observations exist as positive
examples, with high-coverage zero-observation cells as negatives.

Pipeline:
  1. Load model_input.csv (2.7M rows)
  2. Build training set:
     - Positive: cell-dekads with bird observations (16,341 rows)
     - Negative: cell-dekads with zero observations but high satellite
       coverage (sampled to balance classes)
  3. Engineer features from the 12 spectral indices
  4. Train gradient boosted classifier (presence/absence) and
     regressor (diversity score where present)
  5. Calibrate probability outputs
  6. Save models for real-time scoring

Outputs:
  - habitat_classifier.pkl    : predicts P(birds present | spectral values)
  - habitat_regressor.pkl     : predicts diversity score given presence
  - suitability_model.pkl     : combined model bundle for serving
  - training_report.json      : metrics, feature importance, validation
  - feature_thresholds.json   : learned ecological thresholds

Usage:
    python train_suitability_model.py                        # default settings
    python train_suitability_model.py --input model_input.csv
    python train_suitability_model.py --neg-ratio 3          # 3:1 neg:pos ratio
    python train_suitability_model.py --test-size 0.2        # 20% holdout
"""

import os
import sys
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        classification_report, roc_auc_score, precision_recall_curve,
        average_precision_score, mean_absolute_error, r2_score,
    )
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.inspection import permutation_importance
except ImportError:
    sys.exit("ERROR: scikit-learn required. Install: pip install scikit-learn")


# ===========================================================================
# CONFIGURATION
# ===========================================================================

INPUT_CSV = "./model_input.csv"
OUTPUT_DIR = "./suitability_model"

# The 12 spectral indices — these are the model features
SPECTRAL_FEATURES = [
    "NDVI", "NDWI", "MNDWI", "NDMI", "EVI", "SAVI",
    "LSWI", "WRI", "wetland_moisture_index", "water_mask",
    "tc_wetness", "GCVI",
]

# Bird columns used for labeling
BIRD_LABEL_COL = "n_species"
DIVERSITY_COL = "shannon_diversity"
OBSERVATION_COL = "n_observations"

# Negative sampling
DEFAULT_NEG_RATIO = 2  # negatives per positive
MIN_DEKADS_FOR_NEGATIVE = 10  # cell must have this many dekads to be a reliable negative

# Model hyperparameters
CLASSIFIER_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "random_state": 42,
}

REGRESSOR_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "min_samples_leaf": 15,
    "max_features": "sqrt",
    "random_state": 42,
}


# ===========================================================================
# STEP 1: LOAD AND PREPARE DATA
# ===========================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load model_input.csv."""
    print(f"\n  Loading {path} ...")
    chunks = []
    for chunk in pd.read_csv(path, chunksize=500_000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"    Rows: {len(df):,}")
    print(f"    Columns: {len(df.columns)}")
    return df


# ===========================================================================
# STEP 2: BUILD TRAINING SET
# ===========================================================================

def build_training_set(df: pd.DataFrame, neg_ratio: float = DEFAULT_NEG_RATIO) -> pd.DataFrame:
    """
    Build balanced training set from positives (bird observations)
    and reliable negatives (well-observed cells with no birds).
    """
    print("\n" + "=" * 60)
    print("  Building training set")
    print("=" * 60)

    avail_features = [c for c in SPECTRAL_FEATURES if c in df.columns]
    print(f"    Available features: {len(avail_features)}")

    # Drop rows where all spectral features are NaN
    df_valid = df.dropna(subset=avail_features, how="all").copy()
    print(f"    Rows with valid spectra: {len(df_valid):,}")

    # ── Positives: cell-dekads with bird observations ──
    has_birds = df_valid[OBSERVATION_COL] > 0
    positives = df_valid[has_birds].copy()
    positives["label"] = 1
    print(f"    Positives (bird observations): {len(positives):,}")

    # ── Negatives: cell-dekads with no observations ──
    # But only from cells that have been observed at least once
    # (so we know they're accessible/surveyable areas)
    # AND have enough dekads to be reliable

    no_birds = df_valid[~has_birds].copy()

    # Find cells that have at least some bird data in other dekads
    # These are cells where birders go, so zero-observation dekads
    # are more likely true absences (vs never-visited cells)
    cells_with_any_birds = positives.groupby(["cell_x", "cell_y"]).size().reset_index()[["cell_x", "cell_y"]]

    # Also include cells with many dekads of satellite data but never any birds
    # These are likely open water, deep marsh, or otherwise unsuitable
    cell_dekad_counts = df_valid.groupby(["cell_x", "cell_y"]).size().reset_index(name="n_dekads")
    cell_bird_sums = df_valid.groupby(["cell_x", "cell_y"])[OBSERVATION_COL].sum().reset_index(name="total_obs")
    cell_info = cell_dekad_counts.merge(cell_bird_sums, on=["cell_x", "cell_y"])

    # Reliable negatives: cells with enough dekads and zero total observations
    never_observed_cells = cell_info[
        (cell_info["total_obs"] == 0) & (cell_info["n_dekads"] >= MIN_DEKADS_FOR_NEGATIVE)
    ][["cell_x", "cell_y"]]

    # Combine: negatives from (1) known-birded cells on quiet dekads
    #          and (2) well-observed but never-birded cells
    neg_from_birded = no_birds.merge(cells_with_any_birds, on=["cell_x", "cell_y"])
    neg_from_empty = no_birds.merge(never_observed_cells, on=["cell_x", "cell_y"])

    all_negatives = pd.concat([neg_from_birded, neg_from_empty]).drop_duplicates(
        subset=["cell_x", "cell_y", "dekad"] if "dekad" in no_birds.columns
        else ["cell_x", "cell_y"]
    )
    all_negatives["label"] = 0

    print(f"    Candidate negatives: {len(all_negatives):,}")
    print(f"      From birded cells (quiet dekads): {len(neg_from_birded):,}")
    print(f"      From never-birded cells: {len(neg_from_empty):,}")

    # Sample negatives to desired ratio
    n_neg = int(len(positives) * neg_ratio)
    if len(all_negatives) > n_neg:
        negatives = all_negatives.sample(n=n_neg, random_state=42)
    else:
        negatives = all_negatives
        print(f"    WARNING: Only {len(negatives):,} negatives available "
              f"(requested {n_neg:,})")

    print(f"    Sampled negatives: {len(negatives):,}")

    # Combine
    train_df = pd.concat([positives, negatives], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n    Training set: {len(train_df):,} rows")
    print(f"    Positive rate: {train_df['label'].mean():.1%}")

    return train_df


# ===========================================================================
# STEP 3: FEATURE ENGINEERING
# ===========================================================================

def engineer_features(df: pd.DataFrame) -> tuple:
    """
    Build feature matrix from spectral indices.
    Adds interaction features and ratios that capture ecological meaning.
    """
    print("\n" + "=" * 60)
    print("  Engineering features")
    print("=" * 60)

    avail = [c for c in SPECTRAL_FEATURES if c in df.columns]

    # Start with raw spectral indices
    feature_names = list(avail)
    X = df[avail].copy()

    # ── Interaction features ──
    # Vegetation-water interactions (key for wetland birds)
    if "NDVI" in X.columns and "NDWI" in X.columns:
        X["NDVI_x_NDWI"] = X["NDVI"] * X["NDWI"]
        feature_names.append("NDVI_x_NDWI")

    if "NDVI" in X.columns and "MNDWI" in X.columns:
        X["NDVI_x_MNDWI"] = X["NDVI"] * X["MNDWI"]
        feature_names.append("NDVI_x_MNDWI")

    # Vegetation vigor vs moisture
    if "EVI" in X.columns and "NDMI" in X.columns:
        X["EVI_x_NDMI"] = X["EVI"] * X["NDMI"]
        feature_names.append("EVI_x_NDMI")

    # Wetland productivity (green vegetation near water)
    if "GCVI" in X.columns and "wetland_moisture_index" in X.columns:
        X["GCVI_x_wetmoist"] = X["GCVI"] * X["wetland_moisture_index"]
        feature_names.append("GCVI_x_wetmoist")

    # ── Ratio features ──
    # Vegetation to water balance
    if "NDVI" in X.columns and "MNDWI" in X.columns:
        denom = X["MNDWI"].abs() + 0.01
        X["veg_water_ratio"] = X["NDVI"] / denom
        feature_names.append("veg_water_ratio")

    # Canopy moisture
    if "NDMI" in X.columns and "NDVI" in X.columns:
        denom = X["NDVI"].abs() + 0.01
        X["moisture_per_veg"] = X["NDMI"] / denom
        feature_names.append("moisture_per_veg")

    # ── Squared terms (capture nonlinear optima) ──
    # Birds often prefer intermediate values, not extremes
    for idx in ["NDVI", "NDWI", "EVI"]:
        if idx in X.columns:
            X[f"{idx}_sq"] = X[idx] ** 2
            feature_names.append(f"{idx}_sq")

    # ── Temporal features ──
    if "month" in df.columns:
        X["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        X["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        feature_names.extend(["month_sin", "month_cos"])

    if "doy_sin" in df.columns:
        X["doy_sin"] = df["doy_sin"]
        X["doy_cos"] = df["doy_cos"]
        feature_names.extend(["doy_sin", "doy_cos"])

    # Fill NaN with 0
    X = X.fillna(0)

    # Handle inf
    X = X.replace([np.inf, -np.inf], 0)

    print(f"    Raw spectral features: {len(avail)}")
    print(f"    Interaction features: {len(feature_names) - len(avail)}")
    print(f"    Total features: {len(feature_names)}")

    return X[feature_names].values, feature_names


# ===========================================================================
# STEP 4: TRAIN MODELS
# ===========================================================================

def train_classifier(X_train, y_train, X_test, y_test, feature_names):
    """Train presence/absence classifier with calibrated probabilities."""
    print("\n" + "=" * 60)
    print("  Training presence/absence classifier")
    print("=" * 60)

    # Train base model
    clf = GradientBoostingClassifier(**CLASSIFIER_PARAMS)
    print(f"    Fitting GBM classifier ({CLASSIFIER_PARAMS['n_estimators']} trees) ...")
    clf.fit(X_train, y_train)

    # Calibrate probabilities
    print("    Calibrating probabilities ...")
    cal_clf = CalibratedClassifierCV(clf, cv=5, method="isotonic")
    cal_clf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = cal_clf.predict(X_test)
    y_proba = cal_clf.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)

    print(f"\n    Test set results:")
    print(f"    ROC AUC:            {roc_auc:.3f}")
    print(f"    Average Precision:  {avg_prec:.3f}")
    print(f"\n    Classification Report:")
    report = classification_report(y_test, y_pred, target_names=["absent", "present"])
    print(report)

    # Feature importance
    importances = clf.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("    Top 10 features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"      {row['feature']:30s} {row['importance']:.4f}")

    metrics = {
        "roc_auc": float(roc_auc),
        "average_precision": float(avg_prec),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["absent", "present"], output_dict=True
        ),
        "feature_importance": importance_df.to_dict(orient="records"),
    }

    return cal_clf, clf, metrics


def train_regressor(X_pos_train, y_div_train, X_pos_test, y_div_test, feature_names):
    """Train diversity regressor on positive-only samples."""
    print("\n" + "=" * 60)
    print("  Training diversity regressor")
    print("=" * 60)

    if len(X_pos_train) < 50:
        print("    WARNING: Too few positive samples for regression. Skipping.")
        return None, {}

    reg = GradientBoostingRegressor(**REGRESSOR_PARAMS)
    print(f"    Fitting GBM regressor ({REGRESSOR_PARAMS['n_estimators']} trees) ...")
    reg.fit(X_pos_train, y_div_train)

    # Evaluate
    y_pred = reg.predict(X_pos_test)
    mae = mean_absolute_error(y_div_test, y_pred)
    r2 = r2_score(y_div_test, y_pred)

    print(f"\n    Test set results:")
    print(f"    MAE:  {mae:.3f}")
    print(f"    R²:   {r2:.3f}")

    # Feature importance
    importances = reg.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("    Top 10 features for diversity prediction:")
    for _, row in importance_df.head(10).iterrows():
        print(f"      {row['feature']:30s} {row['importance']:.4f}")

    metrics = {
        "mae": float(mae),
        "r2": float(r2),
        "feature_importance": importance_df.to_dict(orient="records"),
    }

    return reg, metrics


# ===========================================================================
# STEP 5: LEARN ECOLOGICAL THRESHOLDS
# ===========================================================================

def learn_thresholds(df: pd.DataFrame, feature_names: list):
    """
    Extract interpretable ecological thresholds from the data.
    What spectral ranges correspond to high bird activity?
    """
    print("\n" + "=" * 60)
    print("  Learning ecological thresholds")
    print("=" * 60)

    thresholds = {}

    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]

    for feat in SPECTRAL_FEATURES:
        if feat not in df.columns:
            continue

        pos_vals = positives[feat].dropna()
        neg_vals = negatives[feat].dropna()

        if pos_vals.empty or neg_vals.empty:
            continue

        thresholds[feat] = {
            "bird_present_mean": float(pos_vals.mean()),
            "bird_present_median": float(pos_vals.median()),
            "bird_present_p25": float(pos_vals.quantile(0.25)),
            "bird_present_p75": float(pos_vals.quantile(0.75)),
            "bird_absent_mean": float(neg_vals.mean()),
            "bird_absent_median": float(neg_vals.median()),
            "difference": float(pos_vals.mean() - neg_vals.mean()),
        }

        diff = thresholds[feat]["difference"]
        direction = "higher" if diff > 0 else "lower"
        print(f"    {feat:30s}: birds prefer {direction} values "
              f"(present={pos_vals.mean():.3f} vs absent={neg_vals.mean():.3f})")

    return thresholds


# ===========================================================================
# STEP 6: BUILD ARCHETYPE DESCRIPTIONS
# ===========================================================================

def build_archetypes(clf, reg, X_all, feature_names, df):
    """
    Use the classifier to define habitat archetypes based on
    probability bands and their typical spectral signatures.
    """
    print("\n" + "=" * 60)
    print("  Building habitat archetypes")
    print("=" * 60)

    proba = clf.predict_proba(X_all)[:, 1]
    df = df.copy()
    df["suitability_prob"] = proba

    # Define archetype bands
    bands = [
        ("Highly suitable", 0.7, 1.0),
        ("Moderately suitable", 0.4, 0.7),
        ("Marginal", 0.15, 0.4),
        ("Unsuitable", 0.0, 0.15),
    ]

    archetypes = {}
    for name, low, high in bands:
        mask = (proba >= low) & (proba < high)
        subset = df[mask]
        if subset.empty:
            continue

        profile = {}
        for feat in SPECTRAL_FEATURES:
            if feat in subset.columns:
                profile[feat] = {
                    "mean": float(subset[feat].mean()),
                    "std": float(subset[feat].std()),
                }

        n_with_birds = int((subset[OBSERVATION_COL] > 0).sum()) if OBSERVATION_COL in subset.columns else 0
        avg_species = float(subset.loc[subset[BIRD_LABEL_COL] > 0, BIRD_LABEL_COL].mean()) if BIRD_LABEL_COL in subset.columns and (subset[BIRD_LABEL_COL] > 0).any() else 0

        archetypes[name] = {
            "probability_range": [float(low), float(high)],
            "n_samples": int(mask.sum()),
            "n_with_birds": n_with_birds,
            "avg_species_when_present": round(avg_species, 1),
            "spectral_profile": profile,
        }

        print(f"    {name:25s}: {mask.sum():,} samples, "
              f"{n_with_birds:,} with birds, "
              f"avg {avg_species:.1f} species")

    return archetypes


# ===========================================================================
# STEP 7: CROSS VALIDATION
# ===========================================================================

def cross_validate(X, y, feature_names):
    """Run 5-fold cross-validation for robust performance estimate."""
    print("\n" + "=" * 60)
    print("  Cross-validation (5-fold)")
    print("=" * 60)

    clf = GradientBoostingClassifier(**CLASSIFIER_PARAMS)

    scoring_metrics = ["roc_auc", "average_precision", "f1"]
    cv_results = {}

    for metric in scoring_metrics:
        scores = cross_val_score(clf, X, y, cv=5, scoring=metric, n_jobs=-1)
        cv_results[metric] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "folds": [float(s) for s in scores],
        }
        print(f"    {metric:25s}: {scores.mean():.3f} ± {scores.std():.3f}")

    return cv_results


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train supervised habitat suitability model"
    )
    parser.add_argument("--input", default=INPUT_CSV,
                        help=f"Input CSV path (default: {INPUT_CSV})")
    parser.add_argument("--output", default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--neg-ratio", type=float, default=DEFAULT_NEG_RATIO,
                        help=f"Negative:positive ratio (default: {DEFAULT_NEG_RATIO})")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set fraction (default: 0.2)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 72)
    print("  Supervised Habitat Suitability Model")
    print("=" * 72)
    print(f"  Input       : {args.input}")
    print(f"  Output dir  : {args.output}")
    print(f"  Neg ratio   : {args.neg_ratio}")
    print(f"  Test size   : {args.test_size}")
    print("=" * 72)

    # Step 1: Load
    df = load_data(args.input)

    # Step 2: Build training set
    train_df = build_training_set(df, neg_ratio=args.neg_ratio)

    # Step 3: Engineer features
    X, feature_names = engineer_features(train_df)
    y = train_df["label"].values

    # Diversity labels for regression (positives only)
    pos_mask = train_df["label"] == 1
    y_diversity = train_df.loc[pos_mask, DIVERSITY_COL].fillna(0).values

    # Step 4: Split
    print(f"\n  Splitting: {1-args.test_size:.0%} train / {args.test_size:.0%} test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # Also split positives for regressor
    X_pos = X[pos_mask]
    pos_indices = np.where(pos_mask)[0]
    train_mask = np.isin(pos_indices, np.where(np.isin(range(len(X)), 
        train_test_split(range(len(X)), test_size=args.test_size, random_state=42, stratify=y)[0]
    ))[0])

    X_pos_train = X_pos[train_mask]
    X_pos_test = X_pos[~train_mask]
    y_div_train = y_diversity[train_mask]
    y_div_test = y_diversity[~train_mask]

    # Step 5: Train classifier
    cal_clf, base_clf, clf_metrics = train_classifier(
        X_train, y_train, X_test, y_test, feature_names
    )

    # Step 6: Train regressor
    reg, reg_metrics = train_regressor(
        X_pos_train, y_div_train, X_pos_test, y_div_test, feature_names
    )

    # Step 7: Cross-validation
    cv_results = cross_validate(X, y, feature_names)

    # Step 8: Learn thresholds
    thresholds = learn_thresholds(train_df, feature_names)

    # Step 9: Build archetypes
    archetypes = build_archetypes(cal_clf, reg, X, feature_names, train_df)

    # ── Save everything ──
    print("\n" + "=" * 60)
    print("  Saving models and reports")
    print("=" * 60)

    # Combined model bundle
    model_bundle = {
        "classifier": cal_clf,
        "base_classifier": base_clf,
        "regressor": reg,
        "feature_names": feature_names,
        "spectral_features": SPECTRAL_FEATURES,
        "archetypes": archetypes,
        "thresholds": thresholds,
        "trained_on": datetime.now().isoformat(),
        "n_training_samples": int(len(train_df)),
        "n_positives": int(pos_mask.sum()),
        "n_features": int(len(feature_names)),
    }

    bundle_path = os.path.join(args.output, "suitability_model.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"    {bundle_path}")

    # Training report
    report = {
        "timestamp": datetime.now().isoformat(),
        "input_file": args.input,
        "training_set": {
            "total_rows": int(len(train_df)),
            "positives": int(pos_mask.sum()),
            "negatives": int((~pos_mask).sum()),
            "positive_rate": float(train_df["label"].mean()),
            "neg_ratio": args.neg_ratio,
        },
        "features": {
            "spectral": SPECTRAL_FEATURES,
            "engineered": [f for f in feature_names if f not in SPECTRAL_FEATURES],
            "total": len(feature_names),
        },
        "classifier": clf_metrics,
        "regressor": reg_metrics,
        "cross_validation": cv_results,
        "archetypes": archetypes,
    }

    report_path = os.path.join(args.output, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"    {report_path}")

    # Thresholds
    thresh_path = os.path.join(args.output, "feature_thresholds.json")
    with open(thresh_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"    {thresh_path}")

    # ── Final summary ──
    print("\n" + "=" * 72)
    print("  MODEL TRAINING COMPLETE")
    print("=" * 72)
    print(f"  Classifier ROC AUC:     {clf_metrics['roc_auc']:.3f}")
    print(f"  Classifier Avg Prec:    {clf_metrics['average_precision']:.3f}")
    if reg_metrics:
        print(f"  Regressor MAE:          {reg_metrics.get('mae', 'N/A')}")
        print(f"  Regressor R²:           {reg_metrics.get('r2', 'N/A')}")
    print(f"  Cross-val ROC AUC:      {cv_results['roc_auc']['mean']:.3f} ± {cv_results['roc_auc']['std']:.3f}")
    print(f"\n  Archetypes:")
    for name, info in archetypes.items():
        print(f"    {name:25s}: {info['n_samples']:,} samples, "
              f"avg {info['avg_species_when_present']:.1f} species")
    print(f"\n  Top 5 predictive features:")
    for item in clf_metrics["feature_importance"][:5]:
        print(f"    {item['feature']:30s} {item['importance']:.4f}")
    print(f"\n  Output directory: {args.output}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
