# Habitat Suitability Model: Technical Analysis

## Overview

The Mississippi Delta Avian Habitat Suitability Model is a supervised machine learning system that predicts bird habitat quality from satellite-derived spectral indices. It combines a gradient boosted classifier for presence/absence prediction with a regressor for species diversity estimation.

## Training Data

### Data Sources

**Satellite Data (Sentinel-2)**
- Source: Biweekly Sentinel-2 imagery via OpenEO
- Coverage: Mississippi Delta region (AOI: -90.628°W to -89.067°W, 28.927°N to 30.106°N)
- Temporal range: 2020-2025
- Resolution: 1km grid cells
- Temporal granularity: Dekadal (10-day) periods

**Bird Observation Data**
- eBird citizen science observations
- iNaturalist observations
- Aggregated to 1km grid cells × dekadal periods
- Metrics: observation counts, species counts, Shannon diversity index

### Training Set Construction

**Total Dataset**: 49,023 samples (cell × dekad combinations)

**Positive Samples (16,341 rows)**
- Cell-dekad combinations with recorded bird observations
- Represents confirmed habitat usage
- Label: 1

**Negative Samples (32,682 rows)**
- Sampled at 2:1 ratio (negatives:positives)
- Two sources:
  1. Quiet dekads in known-birded cells (temporal absences)
  2. Well-observed cells with zero bird records (≥10 dekads, 0 observations)
- Ensures negatives represent true absences, not survey gaps
- Label: 0

**Class Balance**: 33.3% positive rate after sampling


## Feature Engineering

### Spectral Indices (12 raw features)

The model uses 12 satellite-derived spectral indices that capture vegetation, water, and wetland characteristics:

1. **NDVI** (Normalized Difference Vegetation Index)
   - Formula: (NIR - Red) / (NIR + Red)
   - Measures: Green vegetation density
   - Bird preference: Higher values (mean 0.30 vs -0.11)

2. **NDWI** (Normalized Difference Water Index)
   - Formula: (Green - NIR) / (Green + NIR)
   - Measures: Water content
   - Bird preference: Lower values (mean -0.29 vs 0.18)

3. **MNDWI** (Modified NDWI)
   - Formula: (Green - SWIR1) / (Green + SWIR1)
   - Measures: Open water detection
   - Bird preference: Lower values (mean -0.22 vs 0.31)

4. **NDMI** (Normalized Difference Moisture Index)
   - Formula: (NIR - SWIR1) / (NIR + SWIR1)
   - Measures: Vegetation moisture content
   - Bird preference: Lower values (mean 0.10 vs 0.22)

5. **EVI** (Enhanced Vegetation Index)
   - Formula: 2.5 × (NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1)
   - Measures: Vegetation vigor (corrects for atmospheric effects)
   - Bird preference: Higher values (mean 0.20 vs 0.06)

6. **SAVI** (Soil Adjusted Vegetation Index)
   - Formula: 1.5 × (NIR - Red) / (NIR + Red + 0.5)
   - Measures: Vegetation in areas with exposed soil
   - Bird preference: Higher values (mean 0.20 vs 0.05)

7. **LSWI** (Land Surface Water Index)
   - Same as NDMI
   - Measures: Surface water and moisture

8. **WRI** (Water Ratio Index)
   - Formula: (Green + Red) / (NIR + SWIR1)
   - Measures: Water presence
   - Bird preference: Lower values (mean 1.53 vs 8.75)

9. **Wetland Moisture Index**
   - Custom index for wetland detection
   - Bird preference: Lower values (mean -0.14 vs 0.23)

10. **Water Mask**
    - Binary/continuous water detection
    - Bird preference: Lower values (mean 0.15 vs 0.53)

11. **Tasseled Cap Wetness**
    - Linear combination of bands emphasizing moisture
    - Bird preference: Lower values (mean -0.06 vs -0.002)

12. **GCVI** (Green Chlorophyll Vegetation Index)
    - Measures: Chlorophyll content
    - Bird preference: Higher values (mean 1.85 vs 0.48)

### Engineered Features (13 additional features)

**Interaction Features** (capture ecological relationships)
- `NDVI_x_NDWI`: Vegetation-water interaction (wetland interface)
- `NDVI_x_MNDWI`: Vegetation near open water
- `EVI_x_NDMI`: Vegetation vigor × moisture
- `GCVI_x_wetmoist`: Wetland productivity (green vegetation near water)

**Ratio Features** (ecological balances)
- `veg_water_ratio`: NDVI / (|MNDWI| + 0.01) - vegetation to water balance
- `moisture_per_veg`: NDMI / (|NDVI| + 0.01) - canopy moisture

**Nonlinear Features** (capture optimal ranges)
- `NDVI_sq`, `NDWI_sq`, `EVI_sq`: Squared terms for nonlinear optima
- Birds often prefer intermediate values, not extremes

**Temporal Features** (seasonal patterns)
- `month_sin`, `month_cos`: Cyclical month encoding
- `doy_sin`, `doy_cos`: Day-of-year cyclical encoding

**Total Features**: 25 (12 spectral + 13 engineered)


## Model Architecture

### Two-Stage Prediction System

**Stage 1: Presence/Absence Classifier**
- Algorithm: Gradient Boosting Classifier with isotonic calibration
- Purpose: Predicts probability of bird presence given spectral signature
- Output: Calibrated probability [0, 1]

**Stage 2: Diversity Regressor**
- Algorithm: Gradient Boosting Regressor
- Purpose: Predicts species diversity (Shannon index) when birds are present
- Trained only on positive samples
- Output: Diversity score

### Classifier Hyperparameters

```python
n_estimators: 500        # Number of boosting stages
max_depth: 5             # Maximum tree depth
learning_rate: 0.05      # Shrinkage parameter
subsample: 0.8           # Fraction of samples per tree
min_samples_leaf: 20     # Minimum samples in leaf nodes
max_features: 'sqrt'     # Features per split
random_state: 42         # Reproducibility
```

**Calibration**: Isotonic regression with 5-fold CV to ensure probabilities are well-calibrated

### Regressor Hyperparameters

```python
n_estimators: 300        # Fewer trees than c-lassifier
max_depth: 4             # Shallower trees
learning_rate: 0.05
subsample: 0.8
min_samples_leaf: 15
max_features: 'sqrt'
random_state: 42
```

## Model Performance

### Classifier Metrics (Test Set)

- **ROC AUC**: 0.880 (excellent discrimination)
- **Average Precision**: 0.778 (strong precision-recall performance)
- **Overall Accuracy**: 81.3%

**Per-Class Performance**:
- Absent (negative class):
  - Precision: 82.9%
  - Recall: 90.6%
  - F1-score: 86.6%
  
- Present (positive class):
  - Precision: 76.9%
  - Recall: 62.6%
  - F1-score: 69.1%

**Interpretation**: Model is conservative, preferring false negatives over false positives. High recall for absences means it reliably identifies unsuitable habitat.

### Cross-Validation Results (5-fold)

- **ROC AUC**: 0.880 ± 0.003 (very stable)
- **Average Precision**: 0.781 ± 0.007
- **F1 Score**: 0.690 ± 0.004

Low standard deviations indicate robust generalization across different data splits.

### Regressor Metrics (Diversity Prediction)

- **MAE**: 1.08 species (mean absolute error)
- **R²**: 0.076 (weak but positive correlation)

**Interpretation**: Diversity prediction is challenging. The model captures some signal but diversity is influenced by many factors beyond spectral indices (e.g., habitat structure, food availability, migration timing).


## Feature Importance

### Top 10 Most Predictive Features (Classifier)

1. **tc_wetness** (16.4%) - Tasseled cap wetness is the strongest predictor
2. **water_mask** (14.1%) - Water presence/absence critical
3. **EVI** (13.9%) - Enhanced vegetation index
4. **wetland_moisture_index** (8.3%) - Custom wetland indicator
5. **MNDWI** (6.1%) - Modified water index
6. **GCVI** (5.9%) - Green chlorophyll content
7. **EVI_sq** (3.9%) - Nonlinear vegetation response
8. **SAVI** (3.0%) - Soil-adjusted vegetation
9. **doy_cos** (2.9%) - Seasonal timing (day of year)
10. **doy_sin** (2.6%) - Seasonal timing

**Key Insights**:
- Water-related features dominate (tc_wetness, water_mask, MNDWI, wetland_moisture_index = 44.9% combined)
- Vegetation vigor matters (EVI, GCVI, SAVI = 22.8%)
- Temporal features important (doy_cos, doy_sin = 5.5%)
- Engineered features add value (EVI_sq, interaction terms)

### Top 10 Features for Diversity Prediction (Regressor)

1. **water_mask** (11.2%)
2. **tc_wetness** (7.4%)
3. **EVI** (5.4%)
4. **veg_water_ratio** (5.4%) - Engineered feature
5. **WRI** (5.0%)
6. **GCVI** (4.8%)
7. **doy_cos** (4.2%)
8. **NDWI** (4.2%)
9. **SAVI** (4.2%)
10. **EVI_x_NDMI** (3.9%) - Engineered interaction

**Diversity Drivers**: Similar to presence, but engineered features (veg_water_ratio, EVI_x_NDMI) play larger role, suggesting diversity responds to habitat complexity.

## Habitat Archetypes

The model defines four habitat suitability classes based on predicted probability:

### 1. Highly Suitable (P ≥ 0.7)

**Samples**: 8,986 (18.3% of dataset)
**Bird Presence**: 8,028 samples (89.3%)
**Average Species**: 15.7 when birds present

**Spectral Signature**:
- NDVI: 0.34 ± 0.18 (moderate vegetation)
- NDWI: -0.33 ± 0.18 (low water content)
- MNDWI: -0.29 ± 0.18 (not open water)
- EVI: 0.25 ± 2.40 (moderate-high vegetation vigor)
- water_mask: 0.10 ± 0.16 (minimal water)
- tc_wetness: -0.07 ± 0.02 (dry to moderate moisture)
- GCVI: 1.81 ± 4.95 (moderate chlorophyll)

**Ecological Interpretation**: Vegetated wetland edges, marsh-upland transitions, areas with vegetation but not flooded. Ideal foraging and nesting habitat.

### 2. Moderately Suitable (0.4 ≤ P < 0.7)

**Samples**: 8,885 (18.1%)
**Bird Presence**: 4,923 samples (55.4%)
**Average Species**: 16.0 when birds present

**Spectral Signature**:
- NDVI: 0.34 ± 0.28 (moderate vegetation, higher variance)
- NDWI: -0.32 ± 0.28
- MNDWI: -0.24 ± 0.29 (slightly more water than highly suitable)
- water_mask: 0.13 ± 0.21
- tc_wetness: -0.05 ± 0.03 (wetter than highly suitable)

**Ecological Interpretation**: Mixed habitat quality. More variable conditions. Still productive when birds present (actually slightly higher diversity).

### 3. Marginal (0.15 ≤ P < 0.4)

**Samples**: 13,201 (26.9%)
**Bird Presence**: 2,874 samples (21.8%)
**Average Species**: 14.8 when birds present

**Spectral Signature**:
- NDVI: 0.23 ± 0.32 (lower vegetation)
- NDWI: -0.21 ± 0.32
- MNDWI: -0.10 ± 0.35 (approaching neutral)
- water_mask: 0.18 ± 0.27 (more water)
- tc_wetness: -0.03 ± 0.04 (wetter)

**Ecological Interpretation**: Transitional zones. Less reliable habitat. May be suitable during specific seasons or conditions.

### 4. Unsuitable (P < 0.15)

**Samples**: 17,941 (36.6%)
**Bird Presence**: 507 samples (2.8%)
**Average Species**: 13.0 when birds present

**Spectral Signature**:
- NDVI: -0.31 ± 0.34 (negative = water or bare soil)
- NDWI: 0.40 ± 0.37 (high water content)
- MNDWI: 0.55 ± 0.40 (open water)
- water_mask: 0.86 ± 0.29 (mostly water)
- tc_wetness: 0.02 ± 0.04 (wet)
- GCVI: -0.25 ± 2.87 (low/negative chlorophyll)

**Ecological Interpretation**: Open water, deep channels, flooded areas with no emergent vegetation. Unsuitable for most wetland birds (though waterfowl may use).


## Model Input Requirements

### Runtime Prediction Input

When the model is deployed (in `habitat_router.py`), it expects the following inputs:

**Required Spectral Indices** (12 values per location):
```python
[
    "NDVI",                      # -1 to 1
    "NDWI",                      # -1 to 1
    "MNDWI",                     # -1 to 1
    "NDMI",                      # -1 to 1
    "EVI",                       # typically -1 to 1, can exceed
    "SAVI",                      # -1 to 1
    "LSWI",                      # -1 to 1
    "WRI",                       # 0 to ~20+
    "wetland_moisture_index",    # -1 to 1
    "water_mask",                # 0 to 1
    "tc_wetness",                # typically -0.2 to 0.2
    "GCVI"                       # typically -5 to 10
]
```

**Optional Temporal Features** (if available):
```python
{
    "month": 1-12,               # Calendar month
    "doy_sin": sin(2π × doy/365), # Day of year sine
    "doy_cos": cos(2π × doy/365)  # Day of year cosine
}
```

### Feature Engineering Pipeline

The model automatically engineers the 13 additional features from the 12 spectral inputs:

1. Computes interaction terms (NDVI×NDWI, etc.)
2. Computes ratio features (veg_water_ratio, etc.)
3. Computes squared terms (NDVI², etc.)
4. Encodes temporal features if month provided
5. Fills NaN with 0
6. Replaces inf/-inf with 0

**Input Shape**: (n_locations, 12 to 15) → **Feature Matrix**: (n_locations, 25)

### Data Quality Requirements

**Satellite Data Quality**:
- Cloud-free or cloud-masked pixels
- Atmospherically corrected surface reflectance
- Valid band values (typically 0-10000 for Sentinel-2 L2A)

**Spatial Requirements**:
- 1km grid cell aggregation (matches training resolution)
- Mississippi Delta region (model trained on this geography)

**Temporal Requirements**:
- Dekadal (10-day) composites preferred
- Model trained on 2020-2025 data
- Seasonal patterns encoded via temporal features

### Missing Data Handling

- Missing spectral indices filled with 0
- Infinite values replaced with 0
- Model is robust to some missing features due to redundancy
- Critical features: tc_wetness, water_mask, EVI (top 3 predictors)

## Model Outputs

### Prediction Format

For each input location, the model returns:

```python
{
    "suitability_probability": float,  # 0.0 to 1.0
    "suitability_class": str,          # "Highly suitable", "Moderately suitable", 
                                       # "Marginal", or "Unsuitable"
    "predicted_diversity": float,      # Shannon diversity index (if birds present)
    "confidence": str                  # Based on probability distance from thresholds
}
```

### Interpretation Guidelines

**Probability Ranges**:
- **0.70 - 1.00**: High confidence suitable habitat
- **0.40 - 0.70**: Moderate suitability, context-dependent
- **0.15 - 0.40**: Low suitability, occasional use
- **0.00 - 0.15**: Unsuitable for most species

**Confidence Levels**:
- High: Probability >0.8 or <0.1 (clear signal)
- Medium: Probability 0.6-0.8 or 0.1-0.2
- Low: Probability 0.4-0.6 (uncertain, near decision boundary)

## Model Limitations

### Known Constraints

1. **Geographic Specificity**
   - Trained exclusively on Mississippi Delta
   - May not generalize to other wetland systems
   - Different bird communities elsewhere

2. **Temporal Coverage**
   - Training data: 2020-2025
   - Climate change may shift relationships
   - Seasonal patterns may vary year-to-year

3. **Species Aggregation**
   - Predicts overall bird presence, not species-specific
   - Different species have different habitat needs
   - Waterfowl vs shorebirds vs wading birds

4. **Diversity Prediction Weakness**
   - R² = 0.076 for diversity regressor
   - Many factors beyond spectral indices affect diversity
   - Use diversity predictions with caution

5. **Observation Bias**
   - Training data from citizen science (eBird, iNaturalist)
   - Biased toward accessible areas
   - May underrepresent remote habitats

6. **Spectral Limitations**
   - Cannot detect habitat structure (vegetation height, complexity)
   - Cannot detect food resources directly
   - Cannot detect disturbance or human activity

7. **Resolution Constraints**
   - 1km grid cells aggregate fine-scale heterogeneity
   - Small habitat patches may be missed
   - Edge effects not well captured

### Recommended Use Cases

**Appropriate Uses**:
- Regional habitat assessment and mapping
- Identifying priority conservation areas
- Monitoring habitat change over time
- Guiding field survey efforts
- Preliminary site screening

**Inappropriate Uses**:
- Species-specific habitat modeling (use species distribution models)
- Fine-scale (<1km) habitat assessment
- Regulatory decisions without ground-truthing
- Extrapolation to other geographic regions
- Predicting rare species occurrence

## Model Maintenance

### Retraining Recommendations

**Retrain when**:
- New observation data accumulated (annually)
- Significant habitat change events (hurricanes, restoration)
- Model performance degrades (monitor predictions vs observations)
- Expanding to new geographic areas

### Performance Monitoring

**Track these metrics**:
- Prediction accuracy on new observations
- Calibration curve (predicted vs observed probabilities)
- Feature drift (are spectral distributions changing?)
- Class balance in new data

### Version Control

Current model version: Trained 2026-02-15
- Training samples: 49,023
- Features: 25
- Classifier ROC AUC: 0.880
- Cross-val ROC AUC: 0.880 ± 0.003

## Technical Implementation

### Model Serialization

**Saved Artifacts**:
1. `suitability_model.pkl` - Complete model bundle
   - Calibrated classifier
   - Base classifier (for feature importance)
   - Diversity regressor
   - Feature names and metadata
   - Archetypes and thresholds

2. `training_report.json` - Performance metrics and validation results

3. `feature_thresholds.json` - Ecological thresholds for interpretation

### Inference Pipeline

```python
# Load model
with open("suitability_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare input (12 spectral indices)
spectral_data = {
    "NDVI": 0.35, "NDWI": -0.30, "MNDWI": -0.25,
    "NDMI": 0.08, "EVI": 0.22, "SAVI": 0.21,
    "LSWI": 0.08, "WRI": 1.2, 
    "wetland_moisture_index": -0.15,
    "water_mask": 0.12, "tc_wetness": -0.06,
    "GCVI": 1.8
}

# Engineer features (done automatically by model)
X = engineer_features(spectral_data)

# Predict
probability = model["classifier"].predict_proba(X)[0, 1]
diversity = model["regressor"].predict(X)[0] if probability > 0.5 else 0

# Classify
if probability >= 0.7:
    suitability_class = "Highly suitable"
elif probability >= 0.4:
    suitability_class = "Moderately suitable"
elif probability >= 0.15:
    suitability_class = "Marginal"
else:
    suitability_class = "Unsuitable"
```

## References

### Data Sources
- **Sentinel-2**: ESA Copernicus Programme, processed via OpenEO
- **eBird**: Cornell Lab of Ornithology citizen science database
- **iNaturalist**: Community science biodiversity observations

### Methods
- **Gradient Boosting**: Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine.
- **Calibration**: Zadrozny & Elkan (2002). Transforming classifier scores into accurate multiclass probability estimates.
- **Spectral Indices**: Various sources, see individual index definitions

### Software
- **scikit-learn**: Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
- **xarray**: Hoyer & Hamman (2017). xarray: N-D labeled arrays and datasets in Python.
- **pandas**: McKinney (2010). Data Structures for Statistical Computing in Python.

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-19  
**Model Version**: 2026-02-15  
**Contact**: See project documentation
