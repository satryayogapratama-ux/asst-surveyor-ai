#!/usr/bin/env python3
"""
Anomaly Detector - Detect outliers and anomalies in mining data.

Implements:
  - Isolation Forest for spatial coordinate anomalies
  - Local Outlier Factor (LOF) for grade anomalies
  - Statistical methods for assay data validation
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """Multi-method anomaly detection for mining survey data."""
    
    def __init__(self, contamination=0.05):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: expected fraction of outliers (0-1)
        """
        self.contamination = contamination
        self.coordinate_detector = None
        self.grade_detector = None
        self.scaler = StandardScaler()
        self.anomalies = {}
        self.anomaly_scores = {}
    
    def detect_coordinate_anomalies(self, coordinates):
        """
        Detect spatial anomalies using Isolation Forest.
        
        Args:
            coordinates: array of shape (n, 3) with [x, y, z] or [x, y, elevation]
        
        Returns:
            {
                'anomaly_mask': boolean array,
                'anomaly_indices': indices of anomalies,
                'scores': anomaly scores for all points,
                'probable_causes': list of likely issues
            }
        """
        print("  Detecting coordinate anomalies...", end='')
        
        coords_scaled = self.scaler.fit_transform(coordinates)
        
        # Isolation Forest for coordinate anomalies
        self.coordinate_detector = IsolationForest(
            contamination=self.contamination,
            random_state=42
        )
        anomaly_preds = self.coordinate_detector.fit_predict(coords_scaled)
        anomaly_mask = anomaly_preds == -1
        anomaly_indices = np.where(anomaly_mask)[0]
        
        # Get anomaly scores (lower = more anomalous)
        scores = self.coordinate_detector.score_samples(coords_scaled)
        
        # Identify probable causes
        probable_causes = []
        if np.any(anomaly_mask):
            # Check for vertical outliers
            z_values = coordinates[anomaly_indices, 2]
            z_mean = np.mean(coordinates[:, 2])
            z_std = np.std(coordinates[:, 2])
            if np.any(np.abs(z_values - z_mean) > 3 * z_std):
                probable_causes.append("Elevation spike - GPS error or survey rebound")
            
            # Check for spatial clusters
            distances_to_mean = np.linalg.norm(
                coordinates[anomaly_indices, :2] - coordinates[:, :2].mean(axis=0),
                axis=1
            )
            if np.any(distances_to_mean > 3 * coordinates[:, :2].std()):
                probable_causes.append("Location drift - possible datum shift")
        
        self.anomalies['coordinate'] = {
            'mask': anomaly_mask,
            'indices': anomaly_indices,
            'scores': scores,
            'probable_causes': probable_causes if probable_causes else ["No clear cause"]
        }
        
        print(f" Found {len(anomaly_indices)} anomalies")
        return self.anomalies['coordinate']
    
    def detect_grade_anomalies(self, grades, coordinates=None, lithology=None):
        """
        Detect grade anomalies using Local Outlier Factor.
        
        Args:
            grades: array of grade values
            coordinates: optional spatial coordinates (x, y, z)
            lithology: optional lithology codes
        
        Returns:
            {
                'anomaly_mask': boolean array,
                'anomaly_indices': indices of anomalies,
                'outlier_factors': LOF values,
                'grade_values': grade values at anomalies,
                'probable_causes': list of likely issues
            }
        """
        print("  Detecting grade anomalies...", end='')
        
        # Prepare feature matrix
        features = [grades.reshape(-1, 1)]
        if coordinates is not None:
            features.append(self.scaler.fit_transform(coordinates))
        
        X = np.hstack(features) if len(features) > 1 else features[0]
        
        # Local Outlier Factor
        self.grade_detector = LocalOutlierFactor(
            n_neighbors=max(20, len(grades)//10),
            contamination=self.contamination
        )
        anomaly_preds = self.grade_detector.fit_predict(X)
        anomaly_mask = anomaly_preds == -1
        anomaly_indices = np.where(anomaly_mask)[0]
        
        # Get LOF values
        lof_scores = -self.grade_detector.negative_outlier_factor_
        
        # Identify probable causes
        probable_causes = []
        if np.any(anomaly_mask):
            anomaly_grades = grades[anomaly_indices]
            
            # High-grade anomalies
            if np.any(anomaly_grades > np.percentile(grades, 95)):
                probable_causes.append("High-grade enrichment zone - potential economic target")
            
            # Low-grade anomalies
            if np.any(anomaly_grades < np.percentile(grades, 5)):
                probable_causes.append("Low-grade zone - possible waste or lean material")
            
            # Statistical outliers
            grade_mean = np.mean(grades)
            grade_std = np.std(grades)
            extreme_count = np.sum(np.abs(anomaly_grades - grade_mean) > 3*grade_std)
            if extreme_count > 0:
                probable_causes.append(f"{extreme_count} extreme values - check assay lab results")
        
        self.anomalies['grade'] = {
            'mask': anomaly_mask,
            'indices': anomaly_indices,
            'scores': lof_scores,
            'grade_values': grades[anomaly_indices],
            'probable_causes': probable_causes if probable_causes else ["Unusual but not definitively anomalous"]
        }
        
        print(f" Found {len(anomaly_indices)} anomalies")
        return self.anomalies['grade']
    
    def print_report(self):
        """Print detailed anomaly report."""
        print("\n" + "="*70)
        print("ANOMALY DETECTION REPORT")
        print("="*70)
        
        if 'coordinate' in self.anomalies:
            coord_anom = self.anomalies['coordinate']
            print(f"\nSPATIAL ANOMALIES: {len(coord_anom['indices'])} detected")
            print("-" * 70)
            for idx in coord_anom['indices'][:5]:  # Show first 5
                print(f"  Index {idx}: anomaly score {coord_anom['scores'][idx]:.4f}")
            if len(coord_anom['indices']) > 5:
                print(f"  ... and {len(coord_anom['indices'])-5} more")
            print(f"Probable causes:")
            for cause in coord_anom['probable_causes']:
                print(f"  - {cause}")
        
        if 'grade' in self.anomalies:
            grade_anom = self.anomalies['grade']
            print(f"\nGRADE ANOMALIES: {len(grade_anom['indices'])} detected")
            print("-" * 70)
            for idx in grade_anom['indices'][:5]:
                print(f"  Index {idx}: grade {grade_anom['grade_values'][grade_anom['indices'].tolist().index(idx)]:.2f} ppm")
            if len(grade_anom['indices']) > 5:
                print(f"  ... and {len(grade_anom['indices'])-5} more")
            print(f"Probable causes:")
            for cause in grade_anom['probable_causes']:
                print(f"  - {cause}")
        
        print("\n" + "="*70)
    
    def filter_dataset(self, data, remove_coordinate_anomalies=True, remove_grade_anomalies=True):
        """
        Return cleaned dataset with anomalies removed.
        
        Returns:
            cleaned_data, removed_indices
        """
        mask = np.ones(len(data), dtype=bool)
        removed = []
        
        if remove_coordinate_anomalies and 'coordinate' in self.anomalies:
            mask &= ~self.anomalies['coordinate']['mask']
            removed.extend(self.anomalies['coordinate']['indices'])
        
        if remove_grade_anomalies and 'grade' in self.anomalies:
            mask &= ~self.anomalies['grade']['mask']
            removed.extend(self.anomalies['grade']['indices'])
        
        return data[mask], np.unique(removed)
