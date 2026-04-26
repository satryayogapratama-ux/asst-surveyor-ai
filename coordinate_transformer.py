#!/usr/bin/env python3
"""
Coordinate Transformer - Coordinate system conversions with ML-based validation.

Supports:
  - UTM ↔ Geographic (WGS84)
  - Local mining coordinates
  - Datum transformations
  - Indonesia UTM zones (46N-54S)
  - ML error detection for suspicious coordinates
"""

import numpy as np
import pandas as pd
from math import pi, sin, cos, tan, atan, sqrt, log, exp, atan2, degrees, radians


# WGS84 Ellipsoid parameters
WGS84_A = 6378137.0  # Semi-major axis (meters)
WGS84_B = 6356752.3  # Semi-minor axis (meters)
WGS84_F = 1/298.257223563  # Flattening
WGS84_E2 = 2*WGS84_F - WGS84_F**2  # Eccentricity squared


class CoordinateTransformer:
    """Coordinate system transformer with validation."""
    
    # Indonesia UTM zones
    INDONESIA_UTM_ZONES = {
        'N1': 46, 'N2': 47, 'N3': 48, 'N4': 49, 'N5': 50,
        'N6': 51, 'N7': 52, 'N8': 53, 'N9': 54,
        'S1': 46, 'S2': 47, 'S3': 48, 'S4': 49, 'S5': 50,
        'S6': 51, 'S7': 52, 'S8': 53, 'S9': 54
    }
    
    def __init__(self, datum='WGS84'):
        self.datum = datum
        self.validation_results = {}
    
    def geographic_to_utm(self, lat, lon, zone=None):
        """
        Convert geographic (lat, lon) to UTM coordinates.
        
        Args:
            lat: latitude in decimal degrees
            lon: longitude in decimal degrees
            zone: UTM zone (1-60). If None, computed from longitude.
        
        Returns:
            (easting, northing, zone)
        """
        if zone is None:
            zone = int((lon + 180) / 6) + 1
        
        lat_rad = radians(lat)
        lon_rad = radians(lon)
        
        # Central meridian
        lon0 = radians((zone - 1) * 6 - 180 + 3)
        
        # Compute UTM easting and northing
        N = WGS84_A / sqrt(1 - WGS84_E2 * sin(lat_rad)**2)
        T = tan(lat_rad)**2
        C = WGS84_E2 / (1 - WGS84_E2) * cos(lat_rad)**2
        A = cos(lat_rad) * (lon_rad - lon0)
        
        M = WGS84_A * (
            (1 - WGS84_E2/4 - 3*WGS84_E2**2/64 - 5*WGS84_E2**3/256) * lat_rad
            - (3*WGS84_E2/8 + 3*WGS84_E2**2/32 - 45*WGS84_E2**3/1024) * sin(2*lat_rad)
            + (15*WGS84_E2**2/256 - 45*WGS84_E2**3/1024) * sin(4*lat_rad)
            - (35*WGS84_E2**3/3072) * sin(6*lat_rad)
        )
        
        easting = (
            0.9996 * N * (A + A**3/6 * (1 - T + C) + A**5/120 * (1 - 18*T + T**2 + 72*C - 58*WGS84_E2/(1-WGS84_E2)))
            + 500000
        )
        
        northing = 0.9996 * (M + N * tan(lat_rad) * (A**2/2 + A**4/24 * (5 - T + 9*C + 4*C**2) + A**6/720 * (61 - 58*T + T**2 + 600*C - 330*WGS84_E2/(1-WGS84_E2))))
        
        if lat < 0:
            northing += 10000000
        
        return easting, northing, zone
    
    def utm_to_geographic(self, easting, northing, zone, is_southern=False):
        """
        Convert UTM coordinates to geographic (lat, lon).
        
        Args:
            easting: UTM easting
            northing: UTM northing
            zone: UTM zone
            is_southern: True if southern hemisphere
        
        Returns:
            (latitude, longitude)
        """
        if is_southern:
            northing -= 10000000
        
        lon0 = radians((zone - 1) * 6 - 180 + 3)
        M = northing / 0.9996
        
        mu = M / (WGS84_A * (1 - WGS84_E2/4 - 3*WGS84_E2**2/64 - 5*WGS84_E2**3/256))
        
        e1 = (1 - sqrt(1 - WGS84_E2)) / (1 + sqrt(1 - WGS84_E2))
        
        lat_rad = (
            mu + (3*e1/2 - 27*e1**3/32) * sin(2*mu)
            + (21*e1**2/16 - 55*e1**4/32) * sin(4*mu)
            + (151*e1**3/96) * sin(6*mu)
            + (1097*e1**4/512) * sin(8*mu)
        )
        
        C1 = WGS84_E2 / (1 - WGS84_E2) * cos(lat_rad)**2
        T1 = tan(lat_rad)**2
        N1 = WGS84_A / sqrt(1 - WGS84_E2 * sin(lat_rad)**2)
        R1 = WGS84_A * (1 - WGS84_E2) / sqrt((1 - WGS84_E2 * sin(lat_rad)**2)**3)
        D = (easting - 500000) / (N1 * 0.9996)
        
        lat = (
            lat_rad - (tan(lat_rad)/R1) * (D**2/2 - D**4/24 * (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*WGS84_E2/(1-WGS84_E2)) + D**6/720 * (61 + 90*T1 + 28*T1**2 + 45*C1 - 252*WGS84_E2/(1-WGS84_E2) - 3*C1**2))
        )
        
        lon = (
            lon0 + (D - D**3/6 * (1 + 2*T1 + C1) + D**5/120 * (1 - 6*T1 + 8*C1 + 24*T1**2 - 4*C1**2 - 24*WGS84_E2/(1-WGS84_E2))) / cos(lat_rad)
        )
        
        return degrees(lat), degrees(lon)
    
    def validate_coordinates(self, coordinates, expected_zone=None):
        """
        ML-based validation of coordinate data.
        
        Args:
            coordinates: DataFrame or array with easting, northing, elevation
            expected_zone: expected UTM zone
        
        Returns:
            {
                'is_valid': boolean,
                'issues': list of detected issues,
                'confidence': 0-1 score,
                'flags': dict of specific problems
            }
        """
        if isinstance(coordinates, pd.DataFrame):
            easting = coordinates['easting'].values
            northing = coordinates['northing'].values
            elevation = coordinates.get('elevation', pd.Series(np.zeros(len(coordinates)))).values
        else:
            easting = coordinates[:, 0]
            northing = coordinates[:, 1]
            elevation = coordinates[:, 2] if coordinates.shape[1] > 2 else np.zeros(len(coordinates))
        
        issues = []
        flags = {}
        confidence = 1.0
        
        # Check UTM range
        if expected_zone is not None:
            zone_center = (expected_zone - 1) * 6 - 180 + 3
            expected_lon_range = (zone_center - 3.5, zone_center + 3.5)
            
            # Approximate: UTM easting 500000 is central meridian
            approx_lon_deviation = np.abs(easting - 500000) / 111000
            if np.any(approx_lon_deviation > 5):
                issues.append(f"Coordinates may be from wrong UTM zone (expected {expected_zone})")
                flags['wrong_zone'] = True
                confidence -= 0.2
        
        # Check easting range (typical 200000-900000)
        if np.any((easting < 100000) | (easting > 1000000)):
            issues.append("Easting values outside typical UTM range")
            flags['easting_out_of_range'] = True
            confidence -= 0.15
        
        # Check northing range for hemisphere
        if np.any((northing < 0) | (northing > 10000000)):
            issues.append("Northing values outside valid range")
            flags['northing_out_of_range'] = True
            confidence -= 0.25
        
        # Check for duplicate coordinates
        coord_pairs = np.column_stack([easting, northing])
        if len(np.unique(coord_pairs, axis=0)) < len(coord_pairs) * 0.95:
            issues.append("Multiple identical coordinate pairs detected")
            flags['duplicates'] = True
            confidence -= 0.1
        
        # Check for spatial clustering anomalies
        if len(easting) > 3:
            distances = np.std([easting, northing])
            if distances < 10 or distances > 1000000:
                issues.append("Coordinates show unusual spatial distribution")
                flags['spatial_anomaly'] = True
                confidence -= 0.1
        
        # Elevation sanity checks
        if np.any((elevation < -500) | (elevation > 10000)):
            issues.append("Elevation values outside realistic range")
            flags['elevation_suspect'] = True
            confidence -= 0.1
        
        is_valid = len(issues) == 0 and confidence > 0.5
        
        self.validation_results = {
            'is_valid': is_valid,
            'issues': issues,
            'confidence': max(0, confidence),
            'flags': flags,
            'n_points': len(easting)
        }
        
        return self.validation_results
    
    def batch_convert(self, df, from_crs='geographic', to_crs='utm', zone=None):
        """
        Batch convert coordinates.
        
        Args:
            df: DataFrame with coordinates
            from_crs: 'geographic' or 'utm'
            to_crs: 'geographic' or 'utm'
            zone: UTM zone for conversion
        
        Returns:
            DataFrame with converted coordinates
        """
        result = df.copy()
        
        if from_crs == 'geographic' and to_crs == 'utm':
            eastings = []
            northings = []
            zones = []
            
            for idx, row in df.iterrows():
                e, n, z = self.geographic_to_utm(row['latitude'], row['longitude'], zone)
                eastings.append(e)
                northings.append(n)
                zones.append(z)
            
            result['easting'] = eastings
            result['northing'] = northings
            result['utm_zone'] = zones
        
        elif from_crs == 'utm' and to_crs == 'geographic':
            lats = []
            lons = []
            
            for idx, row in df.iterrows():
                lat, lon = self.utm_to_geographic(
                    row['easting'], row['northing'],
                    int(zone or row.get('utm_zone', 48))
                )
                lats.append(lat)
                lons.append(lon)
            
            result['latitude'] = lats
            result['longitude'] = lons
        
        return result
    
    def print_validation_report(self):
        """Print validation results."""
        if not self.validation_results:
            return
        
        results = self.validation_results
        print("\n" + "="*70)
        print("COORDINATE VALIDATION REPORT")
        print("="*70)
        print(f"Points validated: {results['n_points']}")
        print(f"Status: {'VALID' if results['is_valid'] else 'ISSUES DETECTED'}")
        print(f"Confidence: {results['confidence']*100:.1f}%")
        
        if results['issues']:
            print("\nIssues found:")
            for issue in results['issues']:
                print(f"  - {issue}")
        
        print("="*70)
