#!/usr/bin/env python3
"""
Generate synthetic mining drill hole dataset for Asst Surveyor AI.
Creates realistic copper porphyry deposit with spatial clustering.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_drill_collar_data(n_holes=60):
    """Generate drill collar locations with realistic spatial clustering."""
    np.random.seed(42)
    
    # Main zone (cluster 1) - high-grade mineralization
    zone1_x = np.random.normal(800, 100, 25)
    zone1_y = np.random.normal(600, 100, 25)
    
    # Secondary zone (cluster 2) - moderate grade
    zone2_x = np.random.normal(1200, 80, 20)
    zone2_y = np.random.normal(1000, 80, 20)
    
    # Peripheral zone (cluster 3) - lower grade
    zone3_x = np.random.normal(500, 120, 15)
    zone3_y = np.random.normal(400, 120, 15)
    
    # Combine zones
    x = np.concatenate([zone1_x, zone2_x, zone3_x])
    y = np.concatenate([zone1_y, zone2_y, zone3_y])
    
    # Surface elevation (DEM simulation)
    elevation = 1200 + np.random.normal(50, 20, len(x))
    
    # Drill depth (200-500m range)
    depth = np.random.uniform(200, 500, len(x))
    
    # Hole IDs
    hole_ids = [f"DDH_{i+1:03d}" for i in range(len(x))]
    
    collar_df = pd.DataFrame({
        'hole_id': hole_ids,
        'easting': x,
        'northing': y,
        'elevation': elevation,
        'depth': depth,
        'azimuth': np.random.uniform(0, 360, len(x)),
        'dip': np.random.uniform(-60, -30, len(x))
    })
    
    return collar_df

def generate_assay_data(collar_df):
    """Generate downhole assay data with grade-depth relationships."""
    np.random.seed(42)
    assay_records = []
    
    for idx, hole_row in collar_df.iterrows():
        hole_id = hole_row['hole_id']
        easting = hole_row['easting']
        northing = hole_row['northing']
        surface_elev = hole_row['elevation']
        
        # Determine which zone the hole is in
        zone = 1 if ((easting-800)**2 + (northing-600)**2) < 20000 else \
               2 if ((easting-1200)**2 + (northing-1000)**2) < 20000 else 3
        
        # Zone-specific grade baseline
        grade_baseline = {'1': 1.8, '2': 1.2, '3': 0.5}[str(zone)]
        
        # Generate samples at 10m intervals
        depths = np.arange(0, hole_row['depth'], 10)
        
        for depth in depths:
            elev = surface_elev - depth
            
            # Depth-related attenuation
            depth_factor = np.exp(-depth / 300)
            
            # Grade decreases with depth
            grade_mean = grade_baseline * depth_factor
            grade_value = max(0, np.random.normal(grade_mean, grade_mean * 0.3))
            
            # Occasional anomalies (high assay errors or genuine enrichment)
            if np.random.random() < 0.05:
                grade_value = np.abs(np.random.normal(grade_baseline * 2.5, 0.8))
            
            assay_records.append({
                'hole_id': hole_id,
                'easting': easting,
                'northing': northing,
                'elevation': elev,
                'from_depth': depth,
                'to_depth': depth + 10,
                'length': 10,
                'grade_cu_ppm': grade_value,
                'zone': zone
            })
    
    assay_df = pd.DataFrame(assay_records)
    return assay_df

def generate_survey_points(n_points=150):
    """Generate surface survey points for volume calculation demo."""
    np.random.seed(42)
    
    # Grid around mineralized area
    x = np.random.uniform(300, 1500, n_points)
    y = np.random.uniform(200, 1300, n_points)
    
    # Simulated DEM - gentle topography with local variation
    z = 1200 + 0.01 * (x - 800) + 0.015 * (y - 600) + np.random.normal(0, 15, n_points)
    
    survey_df = pd.DataFrame({
        'point_id': [f'SURF_{i+1:04d}' for i in range(n_points)],
        'easting': x,
        'northing': y,
        'elevation': z,
        'measurement_error': np.abs(np.random.normal(0.05, 0.02, n_points))
    })
    
    return survey_df

def main():
    output_dir = Path(__file__).parent
    
    print("Generating synthetic mining dataset...")
    
    # Generate collar data
    collar_df = generate_drill_collar_data(n_holes=60)
    collar_df.to_csv(output_dir / 'drill_collar.csv', index=False)
    print(f"  Created drill_collar.csv: {len(collar_df)} holes")
    
    # Generate assay data
    assay_df = generate_assay_data(collar_df)
    assay_df.to_csv(output_dir / 'drill_assay.csv', index=False)
    print(f"  Created drill_assay.csv: {len(assay_df)} samples")
    
    # Generate survey points
    survey_df = generate_survey_points(n_points=150)
    survey_df.to_csv(output_dir / 'survey_points.csv', index=False)
    print(f"  Created survey_points.csv: {len(survey_df)} surface points")
    
    print("Dataset generation complete!")
    print(f"\nDataset statistics:")
    print(f"  Grade range: {assay_df['grade_cu_ppm'].min():.2f} - {assay_df['grade_cu_ppm'].max():.2f} ppm Cu")
    print(f"  Grade mean: {assay_df['grade_cu_ppm'].mean():.2f} ppm Cu")
    print(f"  Grade std: {assay_df['grade_cu_ppm'].std():.2f} ppm Cu")
    print(f"  Easting range: {assay_df['easting'].min():.0f} - {assay_df['easting'].max():.0f}")
    print(f"  Northing range: {assay_df['northing'].min():.0f} - {assay_df['northing'].max():.0f}")

if __name__ == '__main__':
    main()
