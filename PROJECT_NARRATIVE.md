# Egret: AI-Powered Wetland Habitat Assessment

## The Problem: Wetland Conservation in Crisis

### Background

[PLACEHOLDER: Statistics on wetland loss globally and in the Mississippi Delta]
- Global wetland loss rates: [X%] since 1900
- Mississippi Delta wetland loss: [X acres/year]
- Economic impact: $[X] billion in ecosystem services lost annually
- Biodiversity impact: [X%] of wetland-dependent bird species in decline

### Current Challenges in Habitat Assessment

**1. Scale and Accessibility**

[PLACEHOLDER: Research on traditional habitat survey limitations]
- Traditional field surveys cover only [X%] of wetland areas
- Average cost per hectare surveyed: $[X]
- Time required for comprehensive assessment: [X months/years]
- Remote/inaccessible areas remain unmonitored
- Citation needed: [Study on survey coverage gaps]

**2. Temporal Limitations**

[PLACEHOLDER: Data on seasonal monitoring gaps]
- Wetlands are highly dynamic systems
- Bird populations fluctuate seasonally
- Habitat quality changes with water levels, vegetation cycles
- Traditional surveys provide only snapshot data
- Citation needed: [Research on temporal variability in wetland ecosystems]

**3. Data Integration Challenges**

[PLACEHOLDER: Research on multi-source data integration]
- Citizen science data (eBird, iNaturalist) underutilized
- Satellite imagery available but requires expertise to interpret
- No standardized framework for combining observation + remote sensing data
- Citation needed: [Studies on data integration barriers]

**4. Decision-Making Delays**

[PLACEHOLDER: Statistics on conservation planning timelines]
- Habitat assessments take [X months] to complete
- By the time data is analyzed, conditions may have changed
- Conservation funding decisions delayed by lack of timely information
- Citation needed: [Research on conservation planning bottlenecks]

### Specific Problem: Mississippi Delta Avian Habitat

[PLACEHOLDER: Regional context and urgency]
- The Mississippi Delta is a critical stopover for [X] migratory bird species
- [X%] of North American waterfowl use the region
- Habitat loss threatens [list key species]
- Climate change accelerating wetland degradation
- Need for rapid, scalable habitat monitoring solution
- Citation needed: [Regional bird population studies, habitat loss assessments]


## The Solution: Egret Platform

### Overview

Egret is an AI-powered web platform that combines satellite remote sensing with citizen science data to provide real-time habitat suitability assessments for wetland bird species in the Mississippi Delta. The system democratizes access to sophisticated habitat analysis, enabling conservationists, land managers, and researchers to make data-driven decisions quickly and cost-effectively.

### Core Innovation

**Machine Learning Meets Earth Observation**

Egret bridges the gap between satellite imagery and ecological understanding by training a supervised machine learning model on the relationship between:
- Spectral signatures from Sentinel-2 satellite imagery (vegetation indices, water indices, wetland characteristics)
- Ground-truth bird observations from eBird and iNaturalist citizen science platforms

The result: A model that can predict bird habitat quality from satellite data alone, enabling assessment of any location without requiring field surveys.

### Technical Architecture

**1. Data Pipeline**

**Satellite Data Acquisition**
- Source: Sentinel-2 Level-2A (atmospherically corrected surface reflectance)
- Provider: AWS Earth Search STAC catalog via OpenEO
- Temporal resolution: Biweekly composites
- Spatial resolution: 10-20m pixels aggregated to 1km grid cells
- Coverage: Mississippi Delta AOI (-90.628°W to -89.067°W, 28.927°N to 30.106°N)

**Citizen Science Integration**
- eBird: [PLACEHOLDER: X million] observations (2020-2025)
- iNaturalist: [PLACEHOLDER: X thousand] observations (2020-2025)
- Aggregated to 1km × 10-day (dekadal) resolution
- Metrics computed: observation counts, species richness, Shannon diversity index

**2. Machine Learning Model**

**Training Dataset**
- 49,023 samples (grid cell × time period combinations)
- 16,341 positive samples (bird observations present)
- 32,682 negative samples (reliable absences)
- 2:1 negative-to-positive ratio for class balance

**Feature Engineering**
- 12 spectral indices: NDVI, NDWI, MNDWI, NDMI, EVI, SAVI, LSWI, WRI, wetland moisture index, water mask, tasseled cap wetness, GCVI
- 13 engineered features: vegetation-water interactions, ratio features, nonlinear terms, temporal encodings
- Total: 25 features per location

**Model Architecture**
- Stage 1: Gradient Boosting Classifier (presence/absence prediction)
  - 500 trees, max depth 5, learning rate 0.05
  - Isotonic calibration for probability estimates
  - Performance: 88.0% ROC AUC, 81.3% accuracy
  
- Stage 2: Gradient Boosting Regressor (diversity prediction)
  - 300 trees, max depth 4
  - Trained only on positive samples
  - Predicts Shannon diversity index

**Habitat Classification**
- Highly suitable: P ≥ 0.7 (89% bird presence rate, 15.7 avg species)
- Moderately suitable: 0.4 ≤ P < 0.7 (55% presence, 16.0 avg species)
- Marginal: 0.15 ≤ P < 0.4 (22% presence, 14.8 avg species)
- Unsuitable: P < 0.15 (3% presence, 13.0 avg species)
- Open water: Detected separately via water mask

**3. Web Application**

**Backend (FastAPI + Python)**
- RESTful API with async job processing
- STAC catalog integration for live satellite data
- Cached satellite indices for instant results
- Model inference pipeline with feature engineering
- GeoJSON output for map visualization

**Frontend (React + Vite + Mapbox GL)**
- Interactive map interface
- Draw polygon tool for area-of-interest selection
- Real-time job status polling
- Color-coded habitat suitability visualization
- Detailed results panel with statistics and insights
- Species information cards with audio samples

**Deployment (Docker + Docker Compose)**
- Containerized microservices architecture
- Backend: Python 3.12 with GDAL/rasterio for geospatial processing
- Frontend: Node.js build with nginx serving
- Prefetch service: Biweekly satellite data caching
- Shared volume for satellite cache persistence

### Key Features

**1. Real-Time Assessment**
- Draw any polygon on the map
- Results in seconds (cached data) or minutes (live fetch)
- No GIS expertise required

**2. Dual Data Sources**
- Cached satellite indices: Instant results from pre-processed regional data
- Live STAC fetch: On-demand analysis for any location, any time

**3. Interpretable Results**
- Probability scores (0-100%) for each grid cell
- Habitat archetype classification
- Predicted species diversity
- Summary statistics and insights
- Visual heatmap overlay

**4. Scientific Rigor**
- Model trained on 5 years of data (2020-2025)
- Cross-validated performance metrics
- Feature importance transparency
- Ecological threshold documentation

**5. Scalability**
- 1km resolution balances detail with computational efficiency
- Biweekly satellite updates keep data current
- Async job processing handles multiple concurrent requests
- Cloud-ready architecture (AWS Earth Search STAC)

### Use Cases

**Conservation Planning**
- Identify priority areas for protection or restoration
- Assess habitat quality before/after restoration projects
- Monitor habitat change over time
- Guide land acquisition decisions

**Research Applications**
- Rapid habitat characterization for field study site selection
- Landscape-scale habitat modeling
- Climate change impact assessment
- Species distribution modeling inputs

**Land Management**
- Evaluate management interventions (water level control, vegetation management)
- Optimize habitat for target species
- Compliance monitoring for conservation easements
- Adaptive management decision support

**Education and Outreach**
- Visualize habitat quality for public engagement
- Demonstrate conservation impact
- Citizen science integration and validation
- Interactive learning tool for wetland ecology


## Impact and Advantages

### Compared to Traditional Methods

**Speed**
- Traditional: [PLACEHOLDER: X weeks/months] for field survey and analysis
- Egret: Seconds to minutes for any location
- Improvement: [X]× faster

**Cost**
- Traditional: $[PLACEHOLDER: X] per hectare surveyed
- Egret: Marginal cost near zero after model training
- Improvement: [X]% cost reduction

**Coverage**
- Traditional: Limited to accessible areas, [X]% of region
- Egret: Complete regional coverage, any location on-demand
- Improvement: [X]× greater spatial coverage

**Temporal Resolution**
- Traditional: Snapshot surveys, [X] times per year
- Egret: Biweekly satellite updates, historical archive available
- Improvement: Continuous monitoring capability

**Accessibility**
- Traditional: Requires ecological expertise, field equipment, permits
- Egret: Web browser, no specialized training required
- Improvement: Democratized access to habitat assessment

### Quantifiable Benefits

[PLACEHOLDER: Projected impact metrics]
- Potential to assess [X] hectares of wetland habitat annually
- Enable [X]% faster conservation decision-making
- Reduce habitat assessment costs by [X]%
- Support monitoring of [X] priority conservation areas
- Inform management of [X] acres of protected wetlands

### Innovation Highlights

**1. Novel Data Fusion**
- First system to combine Sentinel-2 spectral indices with citizen science observations for habitat modeling in the Mississippi Delta
- Demonstrates value of integrating remote sensing with crowdsourced biodiversity data

**2. Operational Machine Learning**
- Production-ready ML pipeline from training to deployment
- Reproducible, version-controlled model development
- Transparent feature engineering and performance metrics

**3. User-Centered Design**
- Designed for non-technical users (land managers, conservationists)
- Instant visual feedback with interactive maps
- Interpretable results with ecological context

**4. Open Science Approach**
- Built on open data sources (Sentinel-2, eBird, iNaturalist)
- Reproducible methods documented
- Extensible architecture for future enhancements

## Limitations and Future Work

### Current Limitations

**Geographic Scope**
- Model trained specifically for Mississippi Delta
- May not generalize to other wetland systems without retraining
- Different bird communities in other regions

**Species Aggregation**
- Predicts overall bird presence/diversity, not species-specific habitat
- Different species have different habitat requirements
- Waterfowl vs shorebirds vs wading birds not distinguished

**Temporal Constraints**
- Training data: 2020-2025
- Climate change may shift ecological relationships over time
- Seasonal patterns may vary year-to-year

**Resolution Trade-offs**
- 1km grid cells aggregate fine-scale habitat heterogeneity
- Small habitat patches (<1km²) may be missed
- Edge effects not well captured

**Spectral Limitations**
- Cannot detect habitat structure (vegetation height, complexity)
- Cannot detect food resources directly
- Cannot detect disturbance or human activity

**Diversity Prediction**
- Weak performance (R² = 0.076) for diversity regressor
- Many factors beyond spectral indices affect diversity
- Use diversity predictions with caution

### Future Enhancements

**Model Improvements**
- Species-specific habitat models (e.g., separate models for waterfowl, shorebirds, wading birds)
- Incorporate additional data sources (LiDAR for structure, radar for water levels)
- Ensemble methods combining multiple model types
- Temporal modeling (time series analysis, phenology)

**Geographic Expansion**
- Retrain for other wetland regions (Gulf Coast, Great Lakes, etc.)
- Transfer learning approaches for new geographies
- Multi-region model with geographic covariates

**Data Integration**
- Real-time weather data integration
- Hydrological modeling (water depth, flow)
- Land use/land cover change detection
- Human disturbance indicators (roads, development)

**User Features**
- Historical trend analysis (habitat change over time)
- Scenario modeling (predict impact of restoration actions)
- Species-specific predictions
- Mobile app for field use
- API for integration with other conservation tools

**Validation and Monitoring**
- Ongoing validation with new field observations
- Model performance monitoring and retraining pipeline
- Uncertainty quantification and confidence intervals
- Comparison with ground-truth habitat assessments

**Scalability**
- Cloud deployment (AWS, Google Cloud, Azure)
- Distributed processing for larger areas
- Real-time satellite data streaming
- Multi-user collaboration features

## Technical Requirements

### System Requirements

**Backend**
- Python 3.12+
- GDAL/rasterio for geospatial processing
- scikit-learn for ML inference
- FastAPI for API server
- xarray/pandas for data manipulation
- pystac-client for STAC catalog access

**Frontend**
- Node.js 20+
- React 18+
- Mapbox GL JS for mapping
- Vite for build tooling

**Infrastructure**
- Docker + Docker Compose
- 8GB+ RAM recommended
- Storage for satellite cache (~[PLACEHOLDER: X GB] for regional coverage)

### Data Requirements

**Training Data**
- Sentinel-2 L2A imagery (2020-2025)
- eBird observations (CSV export)
- iNaturalist observations (CSV export)
- ~[PLACEHOLDER: X GB] total training data

**Runtime Data**
- Cached satellite indices: ~[PLACEHOLDER: X MB] per biweekly update
- Live STAC access: Internet connection required
- Model files: ~[PLACEHOLDER: X MB]

## Conclusion

Egret represents a paradigm shift in wetland habitat assessment, leveraging the power of machine learning and satellite remote sensing to provide rapid, scalable, and cost-effective habitat quality predictions. By integrating citizen science observations with Earth observation data, the platform bridges the gap between ecological understanding and operational conservation decision-making.

The system demonstrates that sophisticated habitat analysis can be made accessible to non-technical users through thoughtful design and robust engineering. While limitations exist—particularly in geographic scope and species-level predictions—the platform provides a strong foundation for future enhancements and serves as a model for applying AI to conservation challenges.

As wetland ecosystems face increasing pressures from climate change, development, and resource extraction, tools like Egret become essential for efficient, data-driven conservation. By enabling rapid assessment of habitat quality across large landscapes, the platform empowers conservationists to identify priorities, monitor change, and evaluate interventions with unprecedented speed and scale.

### Key Takeaways

1. **AI + Earth Observation = Scalable Conservation**: Machine learning can extract ecological insights from satellite data at scales impossible with traditional methods

2. **Citizen Science Adds Value**: Crowdsourced observations provide essential ground-truth for training and validating remote sensing models

3. **Accessibility Matters**: Sophisticated analysis tools must be usable by practitioners without technical expertise

4. **Transparency Builds Trust**: Documented methods, performance metrics, and limitations are essential for scientific credibility

5. **Operational Readiness**: Moving from research prototype to production system requires robust engineering, testing, and deployment infrastructure

### Call to Action

[PLACEHOLDER: Next steps for stakeholders]
- Conservation organizations: [Pilot testing opportunities, partnership proposals]
- Researchers: [Collaboration on validation studies, model improvements]
- Funders: [Scaling opportunities, geographic expansion]
- Developers: [Open source contributions, feature requests]

---

**Project Status**: Operational prototype  
**Version**: 1.0  
**Last Updated**: 2026-02-19  
**Contact**: [PLACEHOLDER: Contact information]  
**Repository**: [PLACEHOLDER: GitHub/GitLab URL]  
**License**: [PLACEHOLDER: License type]

## References

[PLACEHOLDER: Complete bibliography]

### Wetland Conservation
- [Citation on wetland loss rates]
- [Citation on ecosystem services valuation]
- [Citation on Mississippi Delta ecology]

### Remote Sensing
- [Sentinel-2 mission documentation]
- [Spectral indices literature]
- [STAC specification]

### Machine Learning
- [Gradient boosting methods]
- [Habitat suitability modeling approaches]
- [Model calibration techniques]

### Citizen Science
- [eBird data quality studies]
- [iNaturalist validation research]
- [Citizen science in conservation]

### Conservation Technology
- [AI for conservation review papers]
- [Remote sensing for biodiversity monitoring]
- [Decision support systems for land management]
