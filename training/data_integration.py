import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os

class MultiSourceDataIntegration:
    """
    Integration pipeline - COMPATIBLE with existing train_habitat_model.py
    Handles both aoi_geometry (old) and aoi_bounds (new) parameters
    """
    
    def __init__(self, aoi_bounds: Tuple[float, float, float, float] = None,
                 aoi_geometry: Dict = None):
        """
        Initialize - accepts both parameter formats for compatibility.
        
        Args:
            aoi_bounds: (min_lon, min_lat, max_lon, max_lat)
            aoi_geometry: GeoJSON geometry dict (for backward compatibility)
        """
        # Handle both parameter formats
        if aoi_bounds is None and aoi_geometry is None:
            self.aoi_bounds = (-90.628342, 28.927421, -89.067224, 30.106372)
        elif aoi_geometry is not None:
            # Extract bounds from geometry
            coords = aoi_geometry['coordinates'][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            self.aoi_bounds = (min(lons), min(lats), max(lons), max(lats))
        else:
            self.aoi_bounds = aoi_bounds
            
        self.time_range = ('2010-01-01', '2026-01-31')
        
        self.inaturalist_data = None
        self.ebird_data = None
        self.satellite_features = None
        self.integrated_dataset = None
        
        self.data_dir = './data_cache'
        os.makedirs(self.data_dir, exist_ok=True)
        
    def load_ebird_data(self, 
                       filepath: str = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       filter_aoi: bool = True) -> pd.DataFrame:
        """
        Load eBird observations from GBIF CSV file.
        
        Args:
            filepath: Path to eBird CSV
            start_date: Optional date filter
            end_date: Optional date filter
            filter_aoi: Apply spatial filtering to AOI
        """
        print("\n" + "="*80)
        print("LOADING EBIRD DATA (GBIF FORMAT)")
        print("="*80)
        
        if filepath is None:
            filepath = os.path.join(self.data_dir, 'ebird_data.csv')
            
        if not os.path.exists(filepath):
            print(f"✗ File not found: {filepath}")
            return pd.DataFrame()
        
        try:
            print(f"Reading: {filepath}")
            
            # Try tab-separated first (GBIF format)
            try:
                df = pd.read_csv(filepath, sep='\t', low_memory=False, nrows=1)
                df = pd.read_csv(filepath, sep='\t', low_memory=False)
                print(f"✓ Loaded as tab-separated (GBIF format)")
            except:
                # Fall back to comma-separated
                df = pd.read_csv(filepath, low_memory=False)
                print(f"✓ Loaded as comma-separated")
            
            initial_count = len(df)
            print(f"✓ Loaded {initial_count:,} observations")
            
            # Standardize GBIF column names
            column_mapping = {
                'gbifID': 'observation_id',
                'scientificName': 'scientific_name',
                'species': 'species_gbif',
                'decimalLatitude': 'latitude',    # Map to standard name
                'decimalLongitude': 'longitude',  # Map to standard name
                'eventDate': 'observed_on',
                'individualCount': 'observation_count',
                'locality': 'location_name',
                'stateProvince': 'state'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Create species name
            if 'species' not in df.columns and 'scientific_name' in df.columns:
                df['species'] = df['scientific_name']
            elif 'species_gbif' in df.columns:
                df['species'] = df['species_gbif']
            
            # Date filtering
            if start_date is None:
                start_date = self.time_range[0]
            if end_date is None:
                end_date = self.time_range[1]
                
            if 'observed_on' in df.columns:
                df['observed_on'] = pd.to_datetime(df['observed_on'], errors='coerce')
                df = df.dropna(subset=['observed_on'])
                
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                
                df = df[(df['observed_on'] >= start_dt) & (df['observed_on'] <= end_dt)]
                df['observed_on'] = df['observed_on'].dt.strftime('%Y-%m-%d')
                
                print(f"✓ Date filtered: {len(df):,} observations")
            
            # Spatial filtering
            if filter_aoi and 'latitude' in df.columns and 'longitude' in df.columns:
                df = df.dropna(subset=['latitude', 'longitude'])
                
                df = df[
                    (df['latitude'] >= self.aoi_bounds[1]) &
                    (df['latitude'] <= self.aoi_bounds[3]) &
                    (df['longitude'] >= self.aoi_bounds[0]) &
                    (df['longitude'] <= self.aoi_bounds[2])
                ]
                print(f"✓ Spatial filtered: {len(df):,} observations")
            
            df['data_source'] = 'eBird'
            
            print(f"\n{'='*80}")
            print(f"✓ Final eBird dataset: {len(df):,} observations")
            if len(df) > 0 and 'species' in df.columns:
                print(f"✓ Unique species: {df['species'].nunique():,}")
            
            self.ebird_data = df
            return df
            
        except Exception as e:
            print(f"✗ Error loading eBird data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def load_inaturalist_data(self, 
                             filepath: str = None,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             filter_aoi: bool = True) -> pd.DataFrame:
        """
        Load iNaturalist observations from CSV file.
        
        Args:
            filepath: Path to iNaturalist CSV
            start_date: Optional date filter
            end_date: Optional date filter
            filter_aoi: Apply spatial filtering to AOI
        """
        print("\n" + "="*80)
        print("LOADING INATURALIST DATA")
        print("="*80)
        
        if filepath is None:
            filepath = os.path.join(self.data_dir, 'inat_data.csv')
            
        if not os.path.exists(filepath):
            print(f"✗ File not found: {filepath}")
            return pd.DataFrame()
        
        try:
            print(f"Reading: {filepath}")
            df = pd.read_csv(filepath)
            
            initial_count = len(df)
            print(f"✓ Loaded {initial_count:,} observations")
            
            # Standardize column names
            column_mapping = {
                'id': 'observation_id',
                'common_name': 'species',
                'scientific_name': 'scientific_name',
                'observed_on': 'observed_on',
                'latitude': 'latitude',
                'longitude': 'longitude',
                'quality_grade': 'quality_grade',
                'taxon_id': 'taxon_id'
            }
            
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            
            if 'observation_id' not in df.columns and 'id' in df.columns:
                df['observation_id'] = 'iNat_' + df['id'].astype(str)
            
            # Date filtering
            if start_date is None:
                start_date = self.time_range[0]
            if end_date is None:
                end_date = self.time_range[1]
                
            if 'observed_on' in df.columns:
                df['observed_on'] = pd.to_datetime(df['observed_on'], errors='coerce')
                df = df.dropna(subset=['observed_on'])
                
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                
                df = df[(df['observed_on'] >= start_dt) & (df['observed_on'] <= end_dt)]
                df['observed_on'] = df['observed_on'].dt.strftime('%Y-%m-%d')
                
                print(f"✓ Date filtered: {len(df):,} observations")
            
            # Spatial filtering
            if filter_aoi and 'latitude' in df.columns and 'longitude' in df.columns:
                df = df.dropna(subset=['latitude', 'longitude'])
                
                df = df[
                    (df['latitude'] >= self.aoi_bounds[1]) &
                    (df['latitude'] <= self.aoi_bounds[3]) &
                    (df['longitude'] >= self.aoi_bounds[0]) &
                    (df['longitude'] <= self.aoi_bounds[2])
                ]
                print(f"✓ Spatial filtered: {len(df):,} observations")
            
            df['data_source'] = 'iNaturalist'
            
            print(f"\n{'='*80}")
            print(f"✓ Final iNaturalist dataset: {len(df):,} observations")
            if len(df) > 0 and 'species' in df.columns:
                print(f"✓ Unique species: {df['species'].nunique():,}")
            
            self.inaturalist_data = df
            return df
            
        except Exception as e:
            print(f"✗ Error loading iNaturalist data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def create_spatial_grid(self, cell_size_km: float = 1.0) -> gpd.GeoDataFrame:
        """Create spatial grid covering AOI."""
        min_lon, min_lat, max_lon, max_lat = self.aoi_bounds
        cell_size_deg = cell_size_km / 111.0
        
        lon_cells = np.arange(min_lon, max_lon, cell_size_deg)
        lat_cells = np.arange(min_lat, max_lat, cell_size_deg)
        
        grid_cells = []
        
        for i, lon in enumerate(lon_cells[:-1]):
            for j, lat in enumerate(lat_cells[:-1]):
                cell_id = f"CELL_{i}_{j}"
                
                cell_poly = Polygon([
                    (lon, lat),
                    (lon + cell_size_deg, lat),
                    (lon + cell_size_deg, lat + cell_size_deg),
                    (lon, lat + cell_size_deg),
                    (lon, lat)
                ])
                
                grid_cells.append({
                    'cell_id': cell_id,
                    'geometry': cell_poly,
                    'centroid_lon': lon + cell_size_deg / 2,
                    'centroid_lat': lat + cell_size_deg / 2
                })
        
        gdf = gpd.GeoDataFrame(grid_cells, crs='EPSG:4326')
        return gdf
    
    def aggregate_observations_to_grid(self, 
                                      grid: gpd.GeoDataFrame,
                                      observations: pd.DataFrame,
                                      source: str = 'observations') -> gpd.GeoDataFrame:
        """Aggregate observations to grid cells."""
        
        if len(observations) == 0:
            print(f"  Warning: No {source} observations to aggregate")
            return grid
        
        # Handle different column name formats
        lat_col = None
        lon_col = None
        
        # Check for standard names first
        if 'latitude' in observations.columns and 'longitude' in observations.columns:
            lat_col = 'latitude'
            lon_col = 'longitude'
        # Check for GBIF format (eBird)
        elif 'decimalLatitude' in observations.columns and 'decimalLongitude' in observations.columns:
            lat_col = 'decimalLatitude'
            lon_col = 'decimalLongitude'
        
        if lat_col is None or lon_col is None:
            print(f"  Warning: {source} data missing coordinate columns")
            print(f"  Available columns: {list(observations.columns)[:10]}...")
            return grid
        
        obs_gdf = gpd.GeoDataFrame(
            observations,
            geometry=[Point(lon, lat) for lon, lat in 
                     zip(observations[lon_col], observations[lat_col])],
            crs='EPSG:4326'
        )
        
        joined = gpd.sjoin(grid, obs_gdf, how='left', predicate='contains')
        
        agg_dict = {
            'observation_id': 'count',
            'species': lambda x: len(set(x)) if len(x) > 0 else 0,
        }
        
        if 'year' in observations.columns:
            agg_dict['year'] = lambda x: len(set(x)) if len(x) > 0 else 0
        elif 'observed_on' in observations.columns:
            joined['year'] = pd.to_datetime(joined['observed_on'], errors='coerce').dt.year
            agg_dict['year'] = lambda x: len(set(x.dropna())) if len(x) > 0 else 0
        
        aggregated = joined.groupby('cell_id').agg(agg_dict).reset_index()
        
        rename_map = {
            'observation_id': f'{source}_obs_count',
            'species': f'{source}_species_richness',
            'year': f'{source}_years_observed'
        }
        aggregated = aggregated.rename(columns=rename_map)
        
        grid_with_obs = grid.merge(aggregated, on='cell_id', how='left')
        
        for col in aggregated.columns:
            if col != 'cell_id' and col in grid_with_obs.columns:
                grid_with_obs[col] = grid_with_obs[col].fillna(0)
        
        return grid_with_obs
    
    def integrate_all_sources(self, 
                             satellite_features: pd.DataFrame = None,
                             cell_size_km: float = 1.0) -> gpd.GeoDataFrame:
        """Integrate all data sources."""
        
        print("\n" + "="*80)
        print("INTEGRATING DATA SOURCES")
        print("="*80)
        
        grid = self.create_spatial_grid(cell_size_km=cell_size_km)
        print(f"✓ Created {len(grid):,} grid cells")
        
        if self.ebird_data is not None and len(self.ebird_data) > 0:
            print(f"Aggregating eBird observations...")
            grid = self.aggregate_observations_to_grid(grid, self.ebird_data, source='ebird')
        
        if self.inaturalist_data is not None and len(self.inaturalist_data) > 0:
            print(f"Aggregating iNaturalist observations...")
            grid = self.aggregate_observations_to_grid(grid, self.inaturalist_data, source='inaturalist')
        
        if satellite_features is not None and len(satellite_features) > 0:
            print(f"Joining satellite features...")
            grid = grid.merge(satellite_features, 
                            on=['centroid_lon', 'centroid_lat'], 
                            how='left',
                            suffixes=('', '_sat'))
        
        self.integrated_dataset = grid
        
        print(f"\n✓ Integration complete: {len(grid):,} cells × {len(grid.columns)} features")
        
        return grid
    
    def create_training_labels(self, 
                              integrated_data: gpd.GeoDataFrame,
                              threshold_obs: int = 10,
                              threshold_species: int = 3) -> gpd.GeoDataFrame:
        """Create training labels."""
        
        data = integrated_data.copy()
        
        total_obs_cols = [col for col in data.columns if '_obs_count' in col]
        if total_obs_cols:
            data['total_observations'] = data[total_obs_cols].sum(axis=1)
        
        total_species_cols = [col for col in data.columns if '_species_richness' in col]
        if total_species_cols:
            data['total_species'] = data[total_species_cols].max(axis=1)
        
        if 'total_observations' in data.columns:
            data['high_use'] = (
                (data['total_observations'] >= threshold_obs) &
                (data['total_species'] >= threshold_species)
            ).astype(int)
            
            data['moderate_use'] = (
                (data['total_observations'] >= threshold_obs / 2) &
                (data['total_observations'] < threshold_obs)
            ).astype(int)
            
            data['low_use'] = (
                data['total_observations'] < threshold_obs / 2
            ).astype(int)
        
        return data
    
    def export_for_dashboard(self, output_dir: str = './dashboard_data'):
        """Export integrated data."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.integrated_dataset is None:
            print("⚠ No integrated dataset available")
            return
        
        print("\n" + "="*80)
        print("EXPORTING DATA")
        print("="*80)
        
        geojson_path = os.path.join(output_dir, 'integrated_data.geojson')
        self.integrated_dataset.to_file(geojson_path, driver='GeoJSON')
        print(f"✓ GeoJSON: {geojson_path}")
        
        csv_path = os.path.join(output_dir, 'integrated_data.csv')
        df = pd.DataFrame(self.integrated_dataset.drop(columns='geometry'))
        df.to_csv(csv_path, index=False)
        print(f"✓ CSV: {csv_path}")
        
        summary = {
            'metadata': {
                'total_cells': len(self.integrated_dataset),
                'date_generated': datetime.now().isoformat(),
                'aoi_bounds': self.aoi_bounds,
                'time_range': self.time_range
            },
            'data_sources': []
        }
        
        if self.ebird_data is not None and len(self.ebird_data) > 0:
            summary['data_sources'].append({
                'name': 'eBird',
                'total_observations': len(self.ebird_data),
                'species_count': int(self.ebird_data['species'].nunique()) if 'species' in self.ebird_data.columns else 0
            })
        
        if self.inaturalist_data is not None and len(self.inaturalist_data) > 0:
            summary['data_sources'].append({
                'name': 'iNaturalist',
                'total_observations': len(self.inaturalist_data),
                'species_count': int(self.inaturalist_data['species'].nunique()) if 'species' in self.inaturalist_data.columns else 0
            })
        
        summary_path = os.path.join(output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary: {summary_path}")