# Landscape Ecology Metrics Update

## Overview

Updated the habitat scoring system to provide professional-grade landscape ecology metrics for ecologists at The Water Institute, replacing simplified summary statistics with rigorous spatial analysis.

## Backend Changes

### New Dependencies
- `scipy>=1.11` - Statistical functions (spatial autocorrelation, distribution analysis)
- `sklearn.neighbors.NearestNeighbors` - Patch connectivity analysis

### New Functions in `habitat_router.py`

**1. `compute_landscape_metrics(scored_df, cell_size_km)`**
- Main orchestrator for all landscape metrics
- Returns comprehensive metrics dictionary

**2. `identify_patches(habitat_df, cell_size_km)`**
- Spatial clustering to identify contiguous habitat patches
- Uses breadth-first search for connected components
- Returns patch properties: area, center, perimeter, mean suitability

**3. `compute_patch_metrics(patches, cell_size_km)`**
- FRAGSTATS-style patch analysis
- Metrics:
  - Number of patches
  - Total core area (km²)
  - Largest patch size and index
  - Mean/median patch size
  - Patch size coefficient of variation
  - Mean shape index (1.0 = circle, higher = complex)
  - Edge density (m/ha)

**4. `compute_connectivity_metrics(patches, habitat_df, cell_size_km)`**
- Connectivity and fragmentation analysis
- Metrics:
  - Number of components
  - Mean nearest neighbor distance (km)
  - Proximity index (area-weighted connectivity)
  - Aggregation index (0-1, higher = more clustered)
  - Connectivity status (high/moderate/low)

**5. `compute_diversity_metrics(habitat_df)`**
- Habitat type diversity
- Metrics:
  - Shannon diversity index
  - Simpson evenness index
  - Number of habitat types
  - Dominant type and proportion

**6. `compute_statistical_summary(habitat_df)`**
- Distribution statistics for suitability scores
- Metrics:
  - Mean, median, standard deviation
  - Coefficient of variation
  - Skewness and kurtosis
  - Quartiles (Q25, Q50, Q75) and IQR

**7. `compute_spatial_autocorrelation(habitat_df)`**
- Moran's I spatial autocorrelation
- Interpretation of clustering patterns
- Values: >0.5 = strong clustering, <-0.2 = dispersed

### API Response Changes

**Removed:**
- `mean_suitability` (misleading for large areas)
- `max_suitability` (not actionable)
- `top_cells` (coordinates meaningless to users)

**Added:**
- `landscape_metrics` - Complete metrics object with:
  - `patch_analysis`
  - `connectivity`
  - `diversity`
  - `statistical_summary`
  - `spatial_autocorrelation`
- `top_patches` - Ranked habitat patches with:
  - Patch ID
  - Archetype (quality class)
  - Area (km²) and cell count
  - Center coordinates (for map navigation)
  - Mean suitability

## Frontend Changes

### New UI Components

**1. Tabbed Interface**
- Three tabs: Overview, Landscape, Statistics
- State management with `activeTab` hook
- Clean separation of information density

**2. Overview Tab**
- Executive summary with key findings
- Habitat distribution with color-coded labels
- Core habitat metrics (area, patches, largest patch)
- Connectivity status
- Priority patches list (clickable to zoom)

**3. Landscape Tab**
- Detailed patch metrics table
- Connectivity metrics
- Habitat diversity indices
- Professional formatting for technical users

**4. Statistics Tab**
- Distribution statistics (mean, median, SD, CV, skewness, kurtosis)
- Quartile breakdown
- Spatial autocorrelation (Moran's I)
- Interpretation text

### New CSS Styling

**Added to `Map.css`:**
- `.score-tabs` - Tab navigation bar
- `.tab-btn` - Individual tab buttons with active state
- `.habitat-row` - Habitat type display rows
- `.habitat-label` - Color-coded habitat type badges
- `.patch-item` - Clickable patch cards with hover effects
- `.metrics-table` - Professional data table styling
- `.stat-note` - Supplementary statistical information
- `.interpretation` - Italicized interpretation text

### Interaction Improvements

**Clickable Patches:**
- Clicking a patch in the Priority Patches list flies the map to that location
- Smooth animation (1.5s duration, zoom level 12)
- Helps users quickly locate high-priority areas

**Visual Hierarchy:**
- Color-coded habitat types match map layer colors
- Ranked patches (#1, #2, etc.) with gold accent
- Hover effects on interactive elements

## Use Cases for Ecologists

### 1. Conservation Prioritization
- Identify largest contiguous patches for protection
- Assess connectivity between patches
- Evaluate fragmentation risk

### 2. Restoration Planning
- Find marginal habitat adjacent to core areas
- Assess potential connectivity gains
- Prioritize restoration sites

### 3. Monitoring and Assessment
- Track patch size changes over time
- Monitor fragmentation trends
- Assess habitat quality distribution

### 4. Research and Publication
- Export FRAGSTATS-compatible metrics
- Statistical rigor for peer review
- Spatial autocorrelation for pattern analysis

### 5. Grant Writing and Reporting
- Quantitative metrics for proposals
- Landscape-scale context
- Comparative analysis capability

## Technical Notes

### Performance Considerations
- Patch identification uses BFS (O(n²) worst case)
- Optimized for typical use cases (<500 cells)
- For very large areas (>1000 cells), consider:
  - Spatial indexing (R-tree)
  - Parallel processing
  - Progressive rendering

### Accuracy and Limitations
- Shape index simplified (perimeter estimation)
- Moran's I uses distance threshold (5km)
- Connectivity assumes 1.5× cell size adjacency
- Edge effects not corrected at AOI boundaries

### Future Enhancements
- Export to shapefile (patch boundaries)
- Temporal comparison (change detection)
- Species-specific habitat models
- Integration with Marxan/Zonation
- PDF report generation

## Testing Recommendations

1. **Small area (10-50 cells):**
   - Verify patch identification
   - Check metric calculations
   - Test tab switching

2. **Medium area (50-200 cells):**
   - Performance testing
   - Connectivity analysis
   - Multiple patch types

3. **Large area (200+ cells):**
   - Performance limits
   - Memory usage
   - UI responsiveness

4. **Edge cases:**
   - Single patch
   - All open water
   - Highly fragmented
   - Uniform quality

## Migration Notes

**For existing users:**
- Old `mean_suitability` → `landscape_metrics.statistical_summary.mean`
- Old `top_cells` → `top_patches` (different format)
- New metrics require no user action (automatic)

**For API consumers:**
- Update response parsing to handle new structure
- `landscape_metrics` may be `null` if no habitat cells
- `top_patches` is array (may be empty)

## References

- McGarigal, K. (2015). FRAGSTATS Help. University of Massachusetts.
- Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. Biometrika, 37(1/2), 17-23.
- Turner, M. G., Gardner, R. H., & O'Neill, R. V. (2001). Landscape Ecology in Theory and Practice. Springer.

---

**Version**: 1.0  
**Date**: 2026-02-19  
**Author**: Egret Development Team
