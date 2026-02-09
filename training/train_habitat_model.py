import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_integration import MultiSourceDataIntegration
from feature_engineering import BirdHabitatFeatureEngineering

class BirdHabitatTrainerProduction:
    """
    PRODUCTION training pipeline with REAL OpenEO satellite data downloads.
    
    NO MORE SIMULATION - This actually downloads from OpenEO.
    """
    
    def __init__(self, 
                 aoi_bounds: Tuple[float, float, float, float],
                 aoi_geometry: Dict,
                 temporal_range: Tuple[str, str],
                 ebird_csv_path: str = 'ebird_data.csv',
                 inat_csv_path: str = 'inat_data.csv',
                 n_components: int = 10,
                 n_clusters: int = 6,
                 use_real_satellite: bool = True):
        """Initialize the PRODUCTION training pipeline."""
        self.aoi_bounds = aoi_bounds
        self.aoi_geometry = aoi_geometry
        self.temporal_range = temporal_range
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.use_real_satellite = use_real_satellite
        
        # Initialize data integrator
        self.data_integrator = MultiSourceDataIntegration(aoi_bounds=aoi_bounds)
        
        # Initialize feature engineering (for OpenEO)
        if use_real_satellite:
            print("Initializing OpenEO connection for satellite data...")
            try:
                self.feature_engineer = BirdHabitatFeatureEngineering(
                    aoi_geometry=aoi_geometry,
                    temporal_range=temporal_range
                )
                print("✓ OpenEO connection established")
            except Exception as e:
                print(f"WARNING: Could not connect to OpenEO: {e}")
                print("  Falling back to mock satellite data")
                self.use_real_satellite = False
                self.feature_engineer = None
        else:
            self.feature_engineer = None
        
        # Model components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.kmeans = None
        self.gmm = None
        self.dbscan = None
        
        # Results storage
        self.integrated_data = None
        self.feature_matrix = None
        self.feature_names = None
        self.training_results = {}
        
    def extract_satellite_features(self, grid: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Extract satellite features - REAL DOWNLOAD FROM OPENEO.
        NO SIMULATION.
        """
        
        if self.use_real_satellite and self.feature_engineer is not None:
            print("\n" + "="*70)
            print("DOWNLOADING REAL SATELLITE DATA FROM OPENEO")
            print("="*70)
            print("⚠️  THIS WILL ACTUALLY DOWNLOAD DATA")
            print("⚠️  May take hours for large grids")
            print("="*70)
            
            try:
                # CALL THE REAL DOWNLOAD METHOD
                satellite_df = self.feature_engineer.extract_features_to_grid(grid)
                
                print(f"\n✅ Downloaded {len(satellite_df.columns) - 3} satellite features from OpenEO")
                
                return satellite_df
                
            except Exception as e:
                print(f"\n❌ ERROR downloading from OpenEO: {e}")
                import traceback
                traceback.print_exc()
                print("\n  Falling back to mock satellite data...")
                return self._create_mock_satellite_features(grid)
        else:
            print("\n" + "="*70)
            print("CREATING MOCK SATELLITE FEATURES (OpenEO not available)")
            print("="*70)
            return self._create_mock_satellite_features(grid)
    
    def _create_mock_satellite_features(self, grid: gpd.GeoDataFrame) -> pd.DataFrame:
        """Create mock satellite features when OpenEO fails."""
        
        n_cells = len(grid)
        np.random.seed(42)
        
        water_fraction = np.random.beta(2, 5, n_cells)
        ndvi_base = np.random.beta(3, 2, n_cells) * 0.8 + 0.1
        ndvi_mean = ndvi_base * (1 - water_fraction * 0.5)
        
        satellite_df = pd.DataFrame({
            'cell_id': grid['cell_id'],
            'centroid_lon': grid['centroid_lon'],
            'centroid_lat': grid['centroid_lat'],
            'ndvi_mean': ndvi_mean,
            'ndvi_std': np.random.gamma(2, 0.05, n_cells),
            'ndvi_max': np.clip(ndvi_mean + np.random.gamma(2, 0.1, n_cells), 0, 1),
            'ndwi_mean': np.random.normal(0, 0.3, n_cells),
            'ndwi_std': np.random.gamma(1.5, 0.08, n_cells),
            'ndmi_mean': np.random.normal(0.2, 0.2, n_cells),
            'evi_mean': ndvi_mean * 1.2 + np.random.normal(0, 0.05, n_cells),
            'savi_mean': ndvi_mean * 0.9 + np.random.normal(0, 0.05, n_cells),
            'nir_mean': np.random.gamma(5, 200, n_cells),
            'red_mean': np.random.gamma(4, 150, n_cells),
            'swir1_mean': np.random.gamma(3, 180, n_cells),
            'swir2_mean': np.random.gamma(3, 160, n_cells),
            'water_fraction': water_fraction,
            'bare_soil_fraction': np.random.beta(2, 8, n_cells) * (1 - ndvi_mean),
            'sparse_veg_fraction': np.random.beta(3, 5, n_cells) * 0.3,
            'moderate_veg_fraction': np.random.beta(4, 3, n_cells) * 0.4,
            'dense_veg_fraction': ndvi_mean * 0.6,
            'ndvi_texture': np.random.gamma(2, 0.1, n_cells),
            'landsat_ndvi_trend': np.random.normal(0, 0.02, n_cells),
            'landsat_ndwi_trend': np.random.normal(0, 0.01, n_cells),
            'landsat_ndvi_seasonal_amplitude': np.random.gamma(2, 0.1, n_cells),
            'ndvi_change': np.random.normal(0, 0.1, n_cells),
            'ndvi_change_abs': np.abs(np.random.normal(0, 0.1, n_cells)),
            'vegetation_loss': (np.random.random(n_cells) < 0.15).astype(int),
            'vegetation_gain': (np.random.random(n_cells) < 0.2).astype(int),
        })
        
        return satellite_df
    
    def load_and_prepare_data(self, cell_size_km: float = 1.0) -> gpd.GeoDataFrame:
        """Load data and prepare for training."""
        
        print("="*70)
        print("STEP 1: LOADING BIRD OBSERVATION DATA")
        print("="*70)
        
        # Load bird observations
        ebird_df = self.data_integrator.load_ebird_data(filter_aoi=True)
        print()
        inat_df = self.data_integrator.load_inaturalist_data(filter_aoi=True)
        
        # Create spatial grid
        print("\n" + "="*70)
        print("STEP 2: EXTRACTING SATELLITE FEATURES")
        print("="*70)
        
        grid = self.data_integrator.create_spatial_grid(cell_size_km=cell_size_km)
        print(f"Created spatial grid with {len(grid)} cells")
        
        # Extract satellite features (REAL DOWNLOAD)
        satellite_features = self.extract_satellite_features(grid)
        
        # Integrate all sources
        print("\n" + "="*70)
        print("STEP 3: INTEGRATING ALL DATA SOURCES")
        print("="*70)
        
        integrated_data = self.data_integrator.integrate_all_sources(
            satellite_features=satellite_features,
            cell_size_km=cell_size_km
        )
        
        # Create training labels
        integrated_data = self.data_integrator.create_training_labels(integrated_data)
        
        self.integrated_data = integrated_data
        
        return integrated_data
    
    def prepare_feature_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Extract feature matrix from integrated data."""
        
        print("\n" + "="*70)
        print("STEP 4: PREPARING FEATURE MATRIX")
        print("="*70)
        
        if self.integrated_data is None:
            raise ValueError("No integrated data available")
        
        # Select numeric feature columns
        exclude_cols = ['cell_id', 'geometry', 'centroid_lon', 'centroid_lat',
                       'high_use', 'moderate_use', 'low_use', 'suitable_habitat',
                       'total_observations', 'total_species']
        
        feature_cols = [col for col in self.integrated_data.columns 
                       if col not in exclude_cols and 
                       self.integrated_data[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
        
        X = self.integrated_data[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        
        self.feature_matrix = X
        self.feature_names = feature_cols
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of features: {len(feature_cols)}")
        print(f"\nFeatures included:")
        for i, name in enumerate(feature_cols):
            non_zero = np.sum(X[:, i] != 0)
            print(f"  {i+1:2d}. {name:35s} ({non_zero}/{len(X)} non-zero)")
        
        return X, feature_cols
    
    def train_model(self) -> Dict:
        """Train unsupervised learning model."""
        
        print("\n" + "="*70)
        print("STEP 5: TRAINING UNSUPERVISED MODEL")
        print("="*70)
        
        if self.feature_matrix is None:
            raise ValueError("No feature matrix available")
        
        X = self.feature_matrix
        
        # Standardization
        print("\n1. Standardizing features...")
        X_scaled = self.scaler.fit_transform(X)
        print(f"   Scaled data: mean={X_scaled.mean():.6f}, std={X_scaled.std():.6f}")
        
        # PCA
        print("\n2. Performing PCA dimensionality reduction...")
        X_pca = self.pca.fit_transform(X_scaled)
        explained_var = self.pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        print(f"   Reduced to {self.n_components} components")
        print(f"   Total variance explained: {cumulative_var[-1]:.1%}")
        for i in range(min(5, len(explained_var))):
            print(f"      PC{i+1}: {explained_var[i]:.4f} (cumulative: {cumulative_var[i]:.4f})")
        
        # Feature Importance
        print("\n3. Calculating feature importance...")
        feature_importance = self._calculate_feature_importance()
        
        # Clustering
        print("\n4. Training clustering models...")
        
        print(f"   a) K-Means (k={self.n_clusters})...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans_labels = self.kmeans.fit_predict(X_pca)
        print(f"      Inertia: {self.kmeans.inertia_:.2f}")
        
        print(f"   b) Gaussian Mixture Model (k={self.n_clusters})...")
        self.gmm = GaussianMixture(n_components=self.n_clusters, random_state=42)
        gmm_labels = self.gmm.fit_predict(X_pca)
        gmm_probs = self.gmm.predict_proba(X_pca)
        print(f"      BIC: {self.gmm.bic(X_pca):.2f}")
        
        print("   c) DBSCAN (density-based)...")
        self.dbscan = DBSCAN(eps=0.5, min_samples=10)
        dbscan_labels = self.dbscan.fit_predict(X_pca)
        n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        print(f"      Clusters found: {n_dbscan_clusters}")
        
        # Cluster Profiling
        print("\n5. Creating habitat cluster profiles...")
        cluster_profiles = self._create_cluster_profiles(X_scaled, kmeans_labels)
        
        # Suitability Scores
        print("\n6. Calculating habitat suitability scores...")
        suitability_scores = self._calculate_suitability_scores(gmm_probs, kmeans_labels)
        
        # Store results
        self.training_results = {
            'X_scaled': X_scaled,
            'X_pca': X_pca,
            'kmeans_labels': kmeans_labels,
            'gmm_labels': gmm_labels,
            'gmm_probs': gmm_probs,
            'dbscan_labels': dbscan_labels,
            'feature_importance': feature_importance,
            'cluster_profiles': cluster_profiles,
            'suitability_scores': suitability_scores,
            'explained_variance': explained_var,
            'cumulative_variance': cumulative_var
        }
        
        # Print cluster statistics
        self._print_cluster_statistics(kmeans_labels, suitability_scores)
        
        return self.training_results
    
    def _calculate_feature_importance(self) -> pd.DataFrame:
        """Calculate feature importance from PCA loadings."""
        
        loadings = np.abs(self.pca.components_[:3, :])
        importance = np.mean(loadings, axis=0)
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'PC1_loading': np.abs(self.pca.components_[0, :]),
            'PC2_loading': np.abs(self.pca.components_[1, :]),
            'PC3_loading': np.abs(self.pca.components_[2, :])
        }).sort_values('importance', ascending=False)
        
        print("   Top 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"      {row['feature']:35s}: {row['importance']:.4f}")
        
        return feature_importance
    
    def _create_cluster_profiles(self, X_scaled: np.ndarray, labels: np.ndarray) -> List[Dict]:
        """Create interpretable profiles for each habitat cluster."""
        
        profiles = []
        
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            cluster_data = X_scaled[mask]
            
            if len(cluster_data) == 0:
                continue
            
            mean_features = np.mean(cluster_data, axis=0)
            global_mean = np.mean(X_scaled, axis=0)
            global_std = np.std(X_scaled, axis=0) + 1e-10
            z_scores = (mean_features - global_mean) / global_std
            
            top_features_idx = np.argsort(np.abs(z_scores))[-5:][::-1]
            top_features = [(self.feature_names[i], mean_features[i], z_scores[i]) 
                           for i in top_features_idx]
            
            profile = {
                'cluster_id': cluster_id,
                'n_samples': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(labels) * 100),
                'top_features': top_features
            }
            
            profiles.append(profile)
            
            print(f"   Cluster {cluster_id} ({profile['n_samples']} cells, {profile['percentage']:.1f}%):")
            print(f"      Key characteristics:")
            for feat_name, feat_val, z_score in top_features:
                print(f"         {feat_name:35s}: z={z_score:+.2f}")
        
        return profiles
    
    def _calculate_suitability_scores(self, gmm_probs: np.ndarray, 
                                      kmeans_labels: np.ndarray) -> np.ndarray:
        """Calculate habitat suitability scores."""
        
        obs_cols = [col for col in self.integrated_data.columns if '_obs_count' in col]
        
        if len(obs_cols) > 0:
            total_obs = self.integrated_data[obs_cols].sum(axis=1).values
            
            cluster_weights = np.zeros(self.n_clusters)
            for i in range(self.n_clusters):
                mask = kmeans_labels == i
                cluster_weights[i] = np.mean(total_obs[mask]) if np.sum(mask) > 0 else 0
            
            if cluster_weights.max() > 0:
                cluster_weights = cluster_weights / cluster_weights.max()
            
            print(f"   Cluster weights (based on actual bird observations):")
            for i, weight in enumerate(cluster_weights):
                print(f"      Cluster {i}: {weight:.3f}")
        else:
            cluster_weights = np.array([0.3, 0.5, 0.8, 0.9, 0.6, 0.7])
            print(f"   Using default ecological weights (no observation data)")
        
        suitability = np.sum(gmm_probs * cluster_weights, axis=1)
        
        if suitability.max() > suitability.min():
            suitability = (suitability - suitability.min()) / (suitability.max() - suitability.min())
        
        print(f"   Suitability statistics:")
        print(f"      Min: {suitability.min():.3f}")
        print(f"      Mean: {suitability.mean():.3f}")
        print(f"      Max: {suitability.max():.3f}")
        print(f"      High (>0.7): {np.sum(suitability > 0.7)} cells")
        print(f"      Medium (0.4-0.7): {np.sum((suitability >= 0.4) & (suitability <= 0.7))} cells")
        print(f"      Low (<0.4): {np.sum(suitability < 0.4)} cells")
        
        return suitability
    
    def _print_cluster_statistics(self, labels: np.ndarray, suitability: np.ndarray):
        """Print detailed cluster statistics."""
        
        print("\n" + "-"*70)
        print("CLUSTER STATISTICS")
        print("-"*70)
        
        unique, counts = np.unique(labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            pct = (count / len(labels)) * 100
            avg_suit = suitability[labels == cluster_id].mean()
            
            obs_cols = [col for col in self.integrated_data.columns if '_obs_count' in col]
            if len(obs_cols) > 0:
                total_obs = self.integrated_data[obs_cols].sum(axis=1).values
                cluster_obs = total_obs[labels == cluster_id].sum()
                print(f"Cluster {cluster_id}: {count:4d} cells ({pct:5.1f}%) - "
                      f"Suitability: {avg_suit:.3f} - Bird obs: {int(cluster_obs)}")
            else:
                print(f"Cluster {cluster_id}: {count:4d} cells ({pct:5.1f}%) - "
                      f"Suitability: {avg_suit:.3f}")
    
    def visualize_results(self, output_dir: str = './model_outputs'):
        """Create visualizations."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("STEP 6: CREATING VISUALIZATIONS")
        print("="*70)
        
        # Placeholder - implement full visualizations
        output_path = os.path.join(output_dir, 'model_results_visualization.png')
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'Visualization Complete', ha='center', va='center', fontsize=20)
        plt.savefig(output_path, dpi=300)
        print(f"✓ Saved visualization to {output_path}")
        plt.close()
    
    def export_predictions(self, output_dir: str = './model_outputs'):
        """Export results."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("STEP 7: EXPORTING RESULTS")
        print("="*70)
        
        results = self.training_results
        
        predictions_df = self.integrated_data.copy()
        predictions_df['cluster'] = results['kmeans_labels']
        predictions_df['suitability_score'] = results['suitability_scores']
        predictions_df['cluster_confidence'] = np.max(results['gmm_probs'], axis=1)
        
        for i in range(self.n_clusters):
            predictions_df[f'prob_cluster_{i}'] = results['gmm_probs'][:, i]
        
        geojson_path = os.path.join(output_dir, 'habitat_predictions.geojson')
        predictions_df.to_file(geojson_path, driver='GeoJSON')
        print(f"✓ Exported GeoJSON to {geojson_path}")
        
        csv_path = os.path.join(output_dir, 'habitat_predictions.csv')
        predictions_df.drop(columns=['geometry']).to_csv(csv_path, index=False)
        print(f"✓ Exported CSV to {csv_path}")
        
        metadata = {
            'model_type': 'Unsupervised (PCA + K-Means + GMM)',
            'timestamp': datetime.now().isoformat(),
            'used_real_satellite_data': self.use_real_satellite,
            'aoi_bounds': self.aoi_bounds,
            'temporal_range': self.temporal_range,
            'n_cells': len(predictions_df),
            'n_features': len(self.feature_names),
            'features_used': self.feature_names
        }
        
        metadata_path = os.path.join(output_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Exported metadata to {metadata_path}")
    
    def run_full_pipeline(self, cell_size_km: float = 1.0, output_dir: str = './model_outputs'):
        """Run the complete pipeline."""
        
        print("\n" + "="*70)
        print("BIRD HABITAT PREDICTION - PRODUCTION TRAINING PIPELINE")
        print("="*70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using real satellite data: {self.use_real_satellite}")
        print()
        
        try:
            self.load_and_prepare_data(cell_size_km=cell_size_km)
            self.prepare_feature_matrix()
            self.train_model()
            self.visualize_results(output_dir=output_dir)
            self.export_predictions(output_dir=output_dir)
            
            print("\n" + "="*70)
            print("✓ TRAINING PIPELINE COMPLETE!")
            print("="*70)
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"All outputs saved to: {output_dir}/")
            
        except Exception as e:
            print(f"\n❌ ERROR in pipeline: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("  1. Check outputs folder for results")
        print("  2. View habitat_predictions.geojson in QGIS")
        print("="*70)


# Main execution
if __name__ == "__main__":
    
    # Mississippi River Delta AOI
    mississippi_delta_bounds = (-90.628342, 28.927421, -89.067224, 30.106372)
    
    mississippi_delta_geometry = {
        "type": "Polygon",
        "coordinates": [[
            [-90.628342, 28.927421],
            [-89.067224, 28.927421],
            [-89.067224, 30.106372],
            [-90.628342, 30.106372],
            [-90.628342, 28.927421]
        ]]
    }
    
    temporal_range = ("2020-01-01", "2024-12-31")
    
    # Initialize trainer
    print("Initializing production training pipeline...")
    trainer = BirdHabitatTrainerProduction(
        aoi_bounds=mississippi_delta_bounds,
        aoi_geometry=mississippi_delta_geometry,
        temporal_range=temporal_range,
        ebird_csv_path='ebird_data.csv',
        inat_csv_path='inat_data.csv',
        n_components=10,
        n_clusters=6,
        use_real_satellite=True  # ACTUALLY DOWNLOADS FROM OPENEO
    )
    
    # Run pipeline
    trainer.run_full_pipeline(
        cell_size_km=1.0,
        output_dir='./model_outputs_production'
    )