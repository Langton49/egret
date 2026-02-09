import os
import openeo
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional

class BirdHabitatFeatureEngineering:
    """
    REAL OpenEO downloads - NO SIMULATION.
    Compatible with train_habitat_model.py pipeline.
    """
    
    def __init__(self, aoi_geometry: Dict = None, temporal_range: Tuple[str, str] = None):
        """
        Initialize with aoi_geometry (for pipeline compatibility).
        
        Args:
            aoi_geometry: GeoJSON polygon dict
            temporal_range: (start_date, end_date)
        """
        if aoi_geometry is None:
            # Default Mississippi Delta
            aoi_geometry = {
                "type": "Polygon",
                "coordinates": [[
                    [-90.628342, 28.927421],
                    [-89.067224, 28.927421],
                    [-89.067224, 30.106372],
                    [-90.628342, 30.106372],
                    [-90.628342, 28.927421]
                ]]
            }
        
        self.aoi = aoi_geometry
        
        # Extract bounds from geometry
        coords = aoi_geometry['coordinates'][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        
        self.spatial_extent = {
            'west': min(lons),
            'south': min(lats),
            'east': max(lons),
            'north': max(lats),
            'crs': 'EPSG:4326'
        }
        
        if temporal_range is None:
            self.temporal_range = ('2020-01-01', '2024-12-31')
        else:
            self.temporal_range = temporal_range
        
        self.conn = None
        self.output_dir = './satellite_data'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def connect_openeo(self):
        """Connect and authenticate to OpenEO."""
        if self.conn is not None:
            return self.conn
            
        print("\n" + "="*80)
        print("CONNECTING TO OPENEO (REAL CONNECTION)")
        print("="*80)
        
        try:
            self.conn = openeo.connect("openeo.dataspace.copernicus.eu")
            self.conn.authenticate_oidc()
            print("✓ Connected and authenticated to OpenEO")
            return self.conn
        except Exception as e:
            print(f"✗ Failed to connect to OpenEO: {e}")
            raise
    
    def extract_sentinel2_features(self) -> Dict:
        """
        Extract Sentinel-2 features (returns datacube dict for compatibility).
        Called by train_habitat_model.py
        """
        print("  Connecting to OpenEO and loading Sentinel-2...")
        conn = self.connect_openeo()
        return self._download_sentinel2()
    
    def extract_landsat_features(self) -> Dict:
        """
        Extract Landsat features (returns datacube dict for compatibility).
        Called by train_habitat_model.py
        """
        print("  Connecting to OpenEO and loading Landsat...")
        conn = self.connect_openeo()
        return self._download_landsat()
    
    def extract_habitat_structure_features(self) -> Dict:
        """
        Extract habitat structure features (returns datacube dict for compatibility).
        Called by train_habitat_model.py
        """
        print("  Computing habitat structure from Sentinel-2...")
        # This is already included in _download_sentinel2
        return {}
    
    def create_composite_features(self) -> Dict:
        """
        Create composite feature set (returns datacube dict for compatibility).
        Called by train_habitat_model.py
        """
        print("  Creating composite feature set...")
        s2_features = self._download_sentinel2()
        landsat_features = self._download_landsat()
        return {**s2_features, **landsat_features}
    
    def extract_features_to_grid(self, grid_gdf) -> pd.DataFrame:
        """
        DOWNLOAD REAL satellite data from OpenEO and extract to grid.
        NO SIMULATION - This actually downloads and processes the data.
        """
        print("\n" + "="*80)
        print("DOWNLOADING REAL SATELLITE DATA FROM OPENEO")
        print("="*80)
        print("⚠️  THIS WILL TAKE A LONG TIME FOR 22,490 CELLS")
        print("⚠️  Consider testing with smaller grid first (100-1000 cells)")
        print("="*80)
        
        # Connect to OpenEO
        conn = self.connect_openeo()
        
        print("\n1. Loading Sentinel-2 collection...")
        s2_features = self._download_sentinel2()
        
        print("\n2. Loading Landsat collection...")
        landsat_features = self._download_landsat()
        
        print("\n3. Combining all feature datacubes...")
        all_features = {**s2_features, **landsat_features}
        
        print(f"\n4. DOWNLOADING data for {len(grid_gdf)} grid cells...")
        print("   This creates one batch job per feature - REAL DOWNLOADS")
        
        features_list = []
        
        # Process in batches to avoid overwhelming OpenEO
        batch_size = 100
        num_batches = (len(grid_gdf) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(grid_gdf))
            batch_cells = grid_gdf.iloc[start_idx:end_idx]
            
            print(f"\n   Batch {batch_idx + 1}/{num_batches} (cells {start_idx}-{end_idx})...")
            
            # Create GeoJSON for this batch of points
            points_geojson = {
                "type": "FeatureCollection",
                "features": []
            }
            
            for idx, row in batch_cells.iterrows():
                centroid = row.geometry.centroid
                points_geojson["features"].append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [centroid.x, centroid.y]
                    },
                    "properties": {
                        "cell_id": row['cell_id'],
                        "centroid_lon": centroid.x,
                        "centroid_lat": centroid.y
                    }
                })
            
            # Sample each feature at these points
            batch_features = []
            for feature_name, datacube in all_features.items():
                print(f"      Downloading {feature_name}...", end="")
                
                try:
                    # Aggregate spatial - samples at points
                    sampled = datacube.aggregate_spatial(
                        geometries=points_geojson,
                        reducer="mean"
                    )
                    
                    # EXECUTE - THIS DOWNLOADS THE DATA
                    result = sampled.execute()
                    
                    # Result is a list of values, one per point
                    if isinstance(result, list) and len(result) == len(batch_cells):
                        batch_features.append((feature_name, result))
                        print(f" ✓")
                    else:
                        print(f" ✗ (unexpected format)")
                        batch_features.append((feature_name, [np.nan] * len(batch_cells)))
                        
                except Exception as e:
                    print(f" ✗ ({str(e)[:50]})")
                    batch_features.append((feature_name, [np.nan] * len(batch_cells)))
            
            # Combine results for this batch
            for i, (idx, row) in enumerate(batch_cells.iterrows()):
                centroid = row.geometry.centroid
                feature_dict = {
                    'cell_id': row['cell_id'],
                    'centroid_lon': centroid.x,
                    'centroid_lat': centroid.y
                }
                
                for feature_name, values in batch_features:
                    feature_dict[feature_name] = values[i]
                
                features_list.append(feature_dict)
        
        features_df = pd.DataFrame(features_list)
        
        print(f"\n✅ DOWNLOAD COMPLETE!")
        print(f"   Extracted {len(features_df.columns) - 3} features for {len(features_df)} cells")
        
        # Save to CSV
        output_csv = os.path.join(self.output_dir, 'satellite_features_grid.csv')
        features_df.to_csv(output_csv, index=False)
        print(f"   Saved to: {output_csv}")
        
        return features_df
    
    def _download_sentinel2(self) -> Dict:
        """Download Sentinel-2 features from OpenEO."""
        
        print(f"  Spatial extent: {self.spatial_extent}")
        print(f"  Temporal extent: {self.temporal_range}")
        
        # Load collection
        s2 = self.conn.load_collection(
            "SENTINEL2_L2A",
            spatial_extent=self.spatial_extent,
            temporal_extent=list(self.temporal_range),
            bands=["B02", "B03", "B04", "B05", "B08", "B11", "B12", "SCL"]
        )
        
        # Cloud masking
        s2 = s2.process("mask_scl_dilation", data=s2, scl_band_name="SCL")
        
        # Calculate indices
        ndvi = s2.ndvi(nir="B08", red="B04")
        ndwi = (s2.band("B03") - s2.band("B08")) / (s2.band("B03") + s2.band("B08"))
        ndmi = (s2.band("B08") - s2.band("B11")) / (s2.band("B08") + s2.band("B11"))
        evi = 2.5 * ((s2.band("B08") - s2.band("B04")) / 
                     (s2.band("B08") + 6 * s2.band("B04") - 7.5 * s2.band("B02") + 1))
        savi = ((s2.band("B08") - s2.band("B04")) / 
                (s2.band("B08") + s2.band("B04") + 0.5)) * 1.5
        
        # Water and vegetation fractions
        water_mask = ndwi > 0.3
        bare = ndvi < 0.2
        sparse = (ndvi >= 0.2) & (ndvi < 0.4)
        moderate = (ndvi >= 0.4) & (ndvi < 0.6)
        dense = ndvi >= 0.6
        
        # Temporal aggregations
        features = {
            'ndvi_mean': ndvi.mean_time(),
            'ndvi_std': ndvi.reduce_dimension(dimension="t", reducer="sd"),
            'ndvi_max': ndvi.max_time(),
            'ndwi_mean': ndwi.mean_time(),
            'ndwi_std': ndwi.reduce_dimension(dimension="t", reducer="sd"),
            'ndmi_mean': ndmi.mean_time(),
            'evi_mean': evi.mean_time(),
            'savi_mean': savi.mean_time(),
            'nir_mean': s2.band("B08").mean_time(),
            'red_mean': s2.band("B04").mean_time(),
            'swir1_mean': s2.band("B11").mean_time(),
            'swir2_mean': s2.band("B12").mean_time(),
            'water_fraction': water_mask.mean_time(),
            'bare_soil_fraction': bare.mean_time(),
            'sparse_veg_fraction': sparse.mean_time(),
            'moderate_veg_fraction': moderate.mean_time(),
            'dense_veg_fraction': dense.mean_time(),
            'ndvi_texture': ndvi.apply_neighborhood(
                process=lambda x: x.sd(),
                size=[{"dimension": "x", "value": 64, "unit": "px"},
                      {"dimension": "y", "value": 64, "unit": "px"}]
            )
        }
        
        print(f"  ✓ Prepared {len(features)} Sentinel-2 features")
        
        return features
    
    def _download_landsat(self) -> Dict:
        """Download Landsat features from OpenEO."""
        
        # Load collection
        landsat = self.conn.load_collection(
            "LANDSAT8_L2",
            spatial_extent=self.spatial_extent,
            temporal_extent=list(self.temporal_range),
            bands=["B02", "B03", "B04", "B05", "B06", "B07"]
        )
        
        # Calculate indices
        ndvi = landsat.ndvi(nir="B05", red="B04")
        ndwi = (landsat.band("B03") - landsat.band("B05")) / \
               (landsat.band("B03") + landsat.band("B05"))
        
        # Temporal trends
        first_obs = ndvi.reduce_dimension(dimension="t", reducer="first")
        last_obs = ndvi.reduce_dimension(dimension="t", reducer="last")
        n_obs = ndvi.reduce_dimension(dimension="t", reducer="count")
        ndvi_trend = (last_obs - first_obs) / (n_obs - 1)
        
        first_obs_ndwi = ndwi.reduce_dimension(dimension="t", reducer="first")
        last_obs_ndwi = ndwi.reduce_dimension(dimension="t", reducer="last")
        n_obs_ndwi = ndwi.reduce_dimension(dimension="t", reducer="count")
        ndwi_trend = (last_obs_ndwi - first_obs_ndwi) / (n_obs_ndwi - 1)
        
        # Seasonal amplitude
        max_val = ndvi.reduce_dimension(dimension="t", reducer="max")
        min_val = ndvi.reduce_dimension(dimension="t", reducer="min")
        seasonal_amp = max_val - min_val
        
        # Change detection
        ndvi_change = last_obs - first_obs
        
        # For absolute value in OpenEO, use the apply_dimension with absolute process
        ndvi_change_abs = ndvi_change.apply(lambda x: x.absolute())
        
        features = {
            'landsat_ndvi_trend': ndvi_trend,
            'landsat_ndwi_trend': ndwi_trend,
            'landsat_ndvi_seasonal_amplitude': seasonal_amp,
            'ndvi_change': ndvi_change,
            'ndvi_change_abs': ndvi_change_abs,
            'vegetation_loss': ndvi_change < -0.1,
            'vegetation_gain': ndvi_change > 0.1
        }
        
        print(f"  ✓ Prepared {len(features)} Landsat features")
        
        return features