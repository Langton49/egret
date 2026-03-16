# Egret — Research Repository Context

## Project Overview

Egret is a web application that helps ecologists and conservation practitioners assess habitat suitability for a drawn area of interest (AOI) on a map. The user draws a polygon, the backend computes spectral indices from Sentinel-2 satellite imagery, runs a habitat suitability model, and returns a scored, classified result.

The research repository (`repo/`) is the data infrastructure layer that supports three planned workstreams:

1. **Retrained suitability model** — tighter thresholds, better resolution, calibrated against systematic survey data
2. **Timeline feature** — historical spectral indices visualised as trends per AOI, paired with bird colony count data
3. **YOLO bird detection pipeline** — automated dotting of new aerial survey imagery

---

## Data Sources

### TWI Avian Monitoring Data (`twi-aviandata.s3.amazonaws.com`)

The Water Institute of the Gulf (TWI) runs an annual colonial waterbird monitoring program across the Gulf Coast. Their data is the primary external dataset for this project.

**Key assets:**

| Asset | Location | Description |
|---|---|---|
| File listing | `file_listing.json` | Full S3 directory tree — entry point for all discovery |
| Dotted screen captures | `/DottedImages/` | 22,575 annotated aerial images, 2010–2023 |
| High-res source photos | `/HighResolutionImages/` | 18,594 raw aerial JPGs, 2010–2023 |
| Colibri database | `/avian_monitoring/dotting_information/Colibri2010-2021CWBColonies_2Jan2023.accdb` | Master annotation database, all Gulf Coast colonies 2010–2021 |
| LACWB database | `/avian_monitoring/dotting_information/LACWB_2022-2023.accdb` | Louisiana-only annotations 2022–2023 |
| Processed CSV | `/avian_monitoring/dotting_information/processed_data/avianData20102021.csv.gz` | Flattened observation records with `uid` photo path field |
| UsedPhotos manifest | `/avian_monitoring/dotting_information/processed_data/UsedPhotos2010-2021.xlsx` | Every photo used in dotting, with camera/card/photo number keys |
| Colibri readme | `/avian_monitoring/dotting_information/ColibriReadMe_2010-2021_Final.pdf` | Database schema, field definitions, dotting protocols |

**Per-photo metadata (2018 only):**
`/HighResolutionImages/2018/metadata/{filename}.json` — EXIF + GPS coordinates derived from KML flight tracks. One JSON per photo.

### TWI STAC Catalog (`twi-avian-2024.s3.us-east-1.amazonaws.com`)

A smaller, structured dataset of colony mosaics with GeoJSON dot annotations for 4 islands across 2018 and 2021.

| Asset | Description |
|---|---|
| `public/test_stac/catalog.json` | STAC catalog root — indexes all mosaics and dot collections |
| `public/mosaics/{region}/{colony}/{year}/` | Cloud-Optimised GeoTIFFs (COGs), dot GeoJSONs, mosaic metadata |
| `public/devDays/` | Sample oblique mosaics (2010–2018) and Cam3 nadir mosaics (2024) |

**Colonies with dot GeoJSONs:**
- Pepperfish Key / Apalachee Bay / 2021
- Queen Bess Island / Barataria Bay / 2018 and 2021
- New Harbor Island 2 & 3 / Breton-Chandeleur / 2021

---

## Fetch Scripts

Run in this order on a fresh environment:

```bash
python fetch_file_listing.py        # Cache the full S3 directory tree locally
python fetch_species_summary.py     # Colony-level species counts from Feature Service
python fetch_dot_geojsons.py        # GeoJSON dot annotations from STAC bucket
python fetch_mosaic_metadata.py     # Mosaic metadata JSONs + build mosaic_index.json
python fetch_databases.py           # .accdb databases + processed CSVs
python fetch_dotted_images.py       # 22,575 annotated screen captures (several GB)
python fetch_training_images.py --probe-only  # Discover high-res photo paths first
```

**Note:** `fetch_training_images.py --probe-only` builds `uid_index.json` mapping photo paths without downloading. Full download is only needed for YOLO inference, not training (training uses dotted screen captures, not raw images).

---

## YOLO Pipeline

### Why dotted screen captures, not raw images

The dotted screen captures from `/DottedImages/` are annotated aerial images — the ecologist has clicked a coloured dot on every visible bird. These are the training data. The raw high-res JPGs are the inference target (new surveys with no annotations).

Training YOLO on screen captures + extracted dot positions → model learns to detect actual birds in raw images.

### Current status: `extract_dots.py`

Extracts bird dot positions from screen captures using OpenCV HSV colour thresholding. Generates YOLO-format `.txt` label files paired with the screen capture images.

**Known issues being worked through:**
- Panel crop: the count window UI sidebar must be excluded from the aerial image crop. Current approach uses saturation channel to detect the low-saturation gray panel region.
- Circularity filter: distinguishes compact dot markers from elongated boundary polygon line fragments
- NMS: removes duplicate detections from overlapping colour ranges

**Run for debug on a single image:**
```bash
python extract_dots.py --debug "data/dotted_images/2010-2013 Dotted Images/2010/KKN 2010 Screen Captures/14June10KKN028-AREA01.JPG"
```
Saves annotated debug image to `data/yolo/debug/` showing panel boundary and detected dot positions.

**Run dry-run across all images:**
```bash
python extract_dots.py --dry-run
```

**Run full extraction:**
```bash
python extract_dots.py
```

### Screen capture anatomy

Two panel layouts exist across years:

**2010–2021:** Large fixed sidebar on the right (~30% of image width). The `Manual Point Count` window is a separate panel showing species names, marker colour/shape legend, and running totals. Must be cropped before dot detection.

**2023+:** Small floating window overlaid on the aerial image top-right. Much smaller footprint, less interference with detection.

### Dot colour scheme (consistent across years)

| Colour | Species examples | Marker shapes |
|---|---|---|
| Cyan | MAFR, SATE | Filled circle |
| Red | BRPE Ad, LAGU Stand/Roost | Filled circle, asterisk, square |
| Yellow | BRPE Imm | Filled circle |
| Green | GREG WBN, ROYT Site/Bird | Asterisk, cross |
| Magenta | LAGU Site/Bird, ROSP, TRHE | Triangle, filled circle |

Red boundary polygon lines (survey area boundaries) are also red — filtered by aspect ratio and circularity.

### Coordinate mapping

Screen captures are at 25% zoom (title bar always shows `filename (1/1)` confirming full image is shown). Dot pixel position in screen capture × 4 = position in corresponding high-res image.

The `uid` field in `avianData20102021.csv.gz` and the `HighResImage_new` field in `UsedPhotos2010-2021.xlsx` provide the path to the high-res source image for each annotated photo.

---

## Suitability Model Retraining

### Current model

Located in `training_two/`. Trained on eBird/iNaturalist citizen science occurrence data for the Louisiana coastal region. Positive label: any cell with `n_observations > 0`. Known issues:
- 1km grid cells — too coarse for small barrier islands
- Single-sighting positive threshold — too permissive
- Citizen science geographic bias — positives cluster where birders go, not where birds are

### Planned retraining

1. **Tighter positive label** — require minimum count thresholds (`n_observations > N`, `n_individuals > X`, Shannon diversity above threshold)
2. **Higher resolution grid** — 100m or 250m cells
3. **Calibrate with TWI data** — after retraining, run model over TWI colony sites and compare suitability scores against systematic dot counts. Adjust archetype probability thresholds until model classifications align with expert survey data.

TWI dot data does not retrain the model directly — it validates and calibrates. eBird/iNaturalist provides geographic coverage; TWI provides accuracy.

---

## Research Repository Goal

A queryable data layer that gives researchers programmatic access to:
- Historical spectral indices per AOI and time range (Sentinel-2, 2017–present)
- Bird colony count data per colony, species, and year
- Dot annotation GeoJSONs per colony and year
- Mosaic asset URLs for pulling specific imagery

This is the infrastructure that makes the timeline feature and the YOLO pipeline trustworthy and reproducible for scientific use.

---

## Architecture Notes

- **Backend:** Python/FastAPI, Docker (`egret-backend-1`)
- **Frontend:** React/Vite, Docker (`egret-frontend-1`)
- **Prefetch:** Separate container (`egret-prefetch-1`) for satellite data preprocessing
- **VM:** Cloud GPU (Nvidia L40S) for YOLO training and eventual model inference
- **Data:** Never committed to git — all fetched via scripts at runtime. `repo/data/` is in `.gitignore`

---

## What Needs Doing Next

1. **Fix `extract_dots.py`** — validate debug output matches expected dot count (119 for test image `7May2010 Camera 1 413.jpg`). Tune HSV ranges if needed.
2. **Run full extraction** — generate YOLO label files for all 22,575 screen captures
3. **Audit label quality** — sample check labels against source images, confirm dot counts match `avianData` CSV totals
4. **Write `train_yolo.py`** — dataset split (stratified by year), YOLOv8s training config, GPU-enabled training on L40S
5. **Retrain suitability model** — define new target variable, tighten thresholds, retrain at higher resolution
6. **Build timeline feature** — per-AOI spectral index trend chart + colony count overlay in the frontend
