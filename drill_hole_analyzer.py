#!/usr/bin/env python3
"""
Drill Hole Analyzer - Comprehensive drill hole data analysis.

Features:
  - Downhole composite generation
  - Grade continuity analysis
  - Statistical summaries
  - Mineralization zone identification (ML clustering)
  - Lithology classification
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


class DrillHoleAnalyzer:
    """Comprehensive drill hole analysis."""
    
    def __init__(self, collar_data, assay_data):
        """
        Initialize analyzer with collar and assay data.
        
        Args:
            collar_data: DataFrame with hole_id, easting, northing, elevation, depth
            assay_data: DataFrame with hole_id, from_depth, to_depth, grade_cu_ppm
        """
        self.collar_data = collar_data
        self.assay_data = assay_data
        self.composites = None
        self.mineralization_zones = None
        self.statistics = {}
    
    def generate_composites(self, composite_length=2.0):
        """
        Generate downhole composites by combining intervals.
        
        Args:
            composite_length: target composite length in meters
        
        Returns:
            DataFrame with composite data
        """
        print(f"  Generating {composite_length}m downhole composites...", end='')
        
        composites = []
        
        for hole_id in self.assay_data['hole_id'].unique():
            hole_assays = self.assay_data[self.assay_data['hole_id'] == hole_id].sort_values('from_depth')
            
            if len(hole_assays) == 0:
                continue
            
            current_depth = 0
            accumulated_grade = 0
            accumulated_length = 0
            
            for idx, assay in hole_assays.iterrows():
                grade = assay['grade_cu_ppm']
                length = assay['to_depth'] - assay['from_depth']
                
                accumulated_grade += grade * length
                accumulated_length += length
                
                # When composite reaches target length, record it
                if accumulated_length >= composite_length or idx == hole_assays.index[-1]:
                    if accumulated_length > 0:
                        composite_grade = accumulated_grade / accumulated_length
                        
                        composites.append({
                            'hole_id': hole_id,
                            'from_depth': current_depth,
                            'to_depth': current_depth + accumulated_length,
                            'length': accumulated_length,
                            'grade_composite': composite_grade,
                            'n_samples': len(hole_assays[
                                (hole_assays['from_depth'] >= current_depth) & 
                                (hole_assays['to_depth'] <= current_depth + accumulated_length)
                            ])
                        })
                        
                        current_depth += accumulated_length
                        accumulated_grade = 0
                        accumulated_length = 0
        
        self.composites = pd.DataFrame(composites)
        print(f" Generated {len(self.composites)} composites")
        return self.composites
    
    def analyze_grade_continuity(self):
        """Analyze grade continuity patterns."""
        print("  Analyzing grade continuity...", end='')
        
        continuity_stats = {}
        
        for hole_id in self.assay_data['hole_id'].unique():
            hole_assays = self.assay_data[self.assay_data['hole_id'] == hole_id].sort_values('from_depth')
            grades = hole_assays['grade_cu_ppm'].values
            
            if len(grades) < 2:
                continue
            
            # Calculate autocorrelation
            grade_diff = np.diff(grades)
            if len(grade_diff) > 0:
                continuity = 1.0 - (np.std(grade_diff) / (np.mean(np.abs(grades)) + 1e-6))
                continuity = max(0, min(1, continuity))  # Clamp to [0, 1]
                
                continuity_stats[hole_id] = {
                    'continuity_score': continuity,
                    'mean_grade': np.mean(grades),
                    'std_grade': np.std(grades),
                    'max_grade': np.max(grades),
                    'depth_range': hole_assays['to_depth'].max() - hole_assays['from_depth'].min()
                }
        
        self.continuity_stats = pd.DataFrame.from_dict(continuity_stats, orient='index')
        print(f" Analyzed {len(self.continuity_stats)} holes")
        return self.continuity_stats
    
    def identify_mineralization_zones(self, n_clusters=3):
        """
        Identify mineralization zones using K-Means clustering.
        
        Args:
            n_clusters: number of mineralization zones to identify
        
        Returns:
            DataFrame with zone assignments
        """
        print(f"  Identifying {n_clusters} mineralization zones...", end='')
        
        # Prepare spatial + grade data
        features = self.assay_data[['easting', 'northing', 'elevation', 'grade_cu_ppm']].values
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.assay_data['zone'] = kmeans.fit_predict(features_scaled)
        
        # Calculate zone statistics
        zone_stats = self.assay_data.groupby('zone').agg({
            'grade_cu_ppm': ['mean', 'std', 'min', 'max', 'count'],
            'easting': 'mean',
            'northing': 'mean',
            'elevation': 'mean'
        }).round(3)
        
        print(f" Identified {n_clusters} zones")
        return self.assay_data, zone_stats
    
    def print_statistical_summary(self):
        """Print overall statistical summary."""
        print("\n" + "="*70)
        print("DRILL HOLE STATISTICAL SUMMARY")
        print("="*70)
        
        # Overall statistics
        print("\nOVERALL GRADES:")
        grades = self.assay_data['grade_cu_ppm']
        print(f"  Mean grade:        {grades.mean():.3f} ppm Cu")
        print(f"  Median grade:      {grades.median():.3f} ppm Cu")
        print(f"  Std deviation:     {grades.std():.3f} ppm Cu")
        print(f"  Minimum:           {grades.min():.3f} ppm Cu")
        print(f"  Maximum:           {grades.max():.3f} ppm Cu")
        print(f"  Q1 (25th pctl):    {grades.quantile(0.25):.3f} ppm Cu")
        print(f"  Q3 (75th pctl):    {grades.quantile(0.75):.3f} ppm Cu")
        
        # Hole statistics
        print("\nDRILL HOLES:")
        print(f"  Total holes:       {self.collar_data['hole_id'].nunique()}")
        print(f"  Total samples:     {len(self.assay_data)}")
        print(f"  Total drilling:    {self.collar_data['depth'].sum():.0f} m")
        print(f"  Avg hole depth:    {self.collar_data['depth'].mean():.1f} m")
        
        # Zone statistics
        if 'zone' in self.assay_data.columns:
            print("\nMINERALIZATION ZONES:")
            for zone in sorted(self.assay_data['zone'].unique()):
                zone_data = self.assay_data[self.assay_data['zone'] == zone]
                print(f"  Zone {zone}:")
                print(f"    - Samples: {len(zone_data)}")
                print(f"    - Mean grade: {zone_data['grade_cu_ppm'].mean():.3f} ppm Cu")
                print(f"    - Grade range: {zone_data['grade_cu_ppm'].min():.3f} - {zone_data['grade_cu_ppm'].max():.3f}")
        
        print("="*70)
    
    def export_summary_report(self, filename='drill_hole_summary.csv'):
        """Export summary statistics to CSV."""
        summary = self.assay_data.groupby('hole_id').agg({
            'grade_cu_ppm': ['mean', 'std', 'min', 'max', 'count'],
            'elevation': ['min', 'max'],
            'from_depth': 'min',
            'to_depth': 'max'
        }).round(3)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary.to_csv(filename)
        print(f"\nSummary exported to {filename}")
        return summary
