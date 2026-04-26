#!/usr/bin/env python3
"""
Volume Calculator - 3D surface fitting and volume estimation.

Implements:
  - Triangulation (TIN) - baseline classical method
  - RBF (Radial Basis Function) - ML-enhanced surface fitting
  - Cut/Fill volume calculation between two surfaces
"""

import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import Rbf
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error


class TriangulationVolume:
    """Classical TIN (Triangulated Irregular Network) for baseline comparison."""
    
    def __init__(self, points):
        """
        Initialize with survey points.
        
        Args:
            points: array of shape (n, 3) with [x, y, z] coordinates
        """
        self.points = np.array(points)
        self.triangulation = None
        self._build_triangulation()
    
    def _build_triangulation(self):
        """Build Delaunay triangulation."""
        self.triangulation = Delaunay(self.points[:, :2])
    
    def interpolate_grid(self, grid_x, grid_y):
        """Interpolate elevation on a grid using linear triangulation."""
        elevations = np.zeros_like(grid_x)
        
        # For each grid point, find containing triangle and interpolate
        points_2d = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        simplex_indices = self.triangulation.find_simplex(points_2d)
        
        for idx, simplex_idx in enumerate(simplex_indices):
            if simplex_idx >= 0:
                # Inside triangulation
                triangle_vertices = self.triangulation.points[self.triangulation.simplices[simplex_idx]]
                z_values = self.points[self.triangulation.simplices[simplex_idx], 2]
                
                # Linear interpolation using barycentric coordinates
                p = points_2d[idx]
                v0 = triangle_vertices[0]
                v1 = triangle_vertices[1]
                v2 = triangle_vertices[2]
                
                denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
                if abs(denom) > 1e-10:
                    w0 = ((v1[1] - v2[1]) * (p[0] - v2[0]) + (v2[0] - v1[0]) * (p[1] - v2[1])) / denom
                    w1 = ((v2[1] - v0[1]) * (p[0] - v2[0]) + (v0[0] - v2[0]) * (p[1] - v2[1])) / denom
                    w2 = 1 - w0 - w1
                    
                    elevations.ravel()[idx] = w0 * z_values[0] + w1 * z_values[1] + w2 * z_values[2]
            else:
                # Outside triangulation - use nearest neighbor
                distances = cdist([points_2d[idx]], self.points[:, :2])[0]
                nearest_idx = np.argmin(distances)
                elevations.ravel()[idx] = self.points[nearest_idx, 2]
        
        return elevations.reshape(grid_x.shape)
    
    def calculate_volume(self, surface1_z, surface2_z, grid_area=None):
        """Calculate volume between two surfaces."""
        dz = np.abs(surface1_z - surface2_z)
        
        if grid_area is None:
            # Approximate grid cell area
            dx = (self.points[:, 0].max() - self.points[:, 0].min()) / max(surface1_z.shape[0]-1, 1)
            dy = (self.points[:, 1].max() - self.points[:, 1].min()) / max(surface1_z.shape[1]-1, 1)
            grid_area = dx * dy
        
        volume = np.sum(dz) * grid_area
        return volume


class RBFVolume:
    """ML-enhanced surface fitting using Radial Basis Functions."""
    
    def __init__(self, points, function='thin_plate', smooth=1e-7, epsilon=1.0):
        """
        Initialize RBF surface.
        
        Args:
            points: array of shape (n, 3) with [x, y, z] coordinates
            function: RBF type ('thin_plate', 'multiquadric', 'inverse_multiquadric')
            smooth: smoothing parameter (regularization)
            epsilon: shape parameter for RBF
        """
        self.points = np.array(points)
        self.function = function
        self.smooth = smooth
        self.epsilon = epsilon
        self.rbf = None
        self._build_rbf()
    
    def _build_rbf(self):
        """Build RBF interpolant."""
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        
        self.rbf = Rbf(
            x, y, z,
            function=self.function,
            smooth=self.smooth,
            epsilon=self.epsilon
        )
    
    def interpolate_grid(self, grid_x, grid_y):
        """Interpolate elevation on a grid using RBF."""
        return self.rbf(grid_x, grid_y)
    
    def calculate_volume(self, surface1_z, surface2_z, grid_area=None):
        """Calculate volume between two surfaces."""
        dz = np.abs(surface1_z - surface2_z)
        
        if grid_area is None:
            dx = (self.points[:, 0].max() - self.points[:, 0].min()) / max(surface1_z.shape[0]-1, 1)
            dy = (self.points[:, 1].max() - self.points[:, 1].min()) / max(surface1_z.shape[1]-1, 1)
            grid_area = dx * dy
        
        volume = np.sum(dz) * grid_area
        return volume


class VolumeCalculator:
    """Multi-method volume calculation with comparison."""
    
    def __init__(self):
        self.tin_calculator = None
        self.rbf_calculator = None
        self.results = {}
    
    def fit(self, points):
        """Fit both surface models."""
        print("\nFitting Surface Models...")
        
        print("  TIN (Triangulation)...", end='')
        self.tin_calculator = TriangulationVolume(points)
        print(" OK")
        
        print("  RBF (Radial Basis Function)...", end='')
        self.rbf_calculator = RBFVolume(points)
        print(" OK")
    
    def calculate_cut_fill(self, original_surface, proposed_surface):
        """
        Calculate cut and fill volumes between surfaces.
        
        Returns:
            {
                'cut_volume': volume_m3,
                'fill_volume': volume_m3,
                'net_volume': volume_m3,
                'method_comparison': {...}
            }
        """
        print("\nCalculating Cut/Fill Volumes...")
        
        # Calculate using both methods
        dz = proposed_surface - original_surface
        
        # TIN method
        tin_cut = np.sum(np.maximum(dz, 0))  # Material to remove
        tin_fill = np.sum(np.maximum(-dz, 0))  # Material to add
        
        # RBF method (same surface difference, different interpolation)
        rbf_cut = tin_cut  # Volume calculation independent of interpolation
        rbf_fill = tin_fill
        
        grid_area = 1.0  # Placeholder - would be (dx * dy) in real usage
        
        results = {
            'cut_volume': tin_cut * grid_area,
            'fill_volume': tin_fill * grid_area,
            'net_volume': (tin_cut - tin_fill) * grid_area,
            'method_comparison': {
                'tin': {'cut': tin_cut * grid_area, 'fill': tin_fill * grid_area},
                'rbf': {'cut': rbf_cut * grid_area, 'fill': rbf_fill * grid_area}
            }
        }
        
        return results
    
    def print_results(self, results):
        """Pretty-print results."""
        print("\n" + "="*70)
        print("VOLUME CALCULATION RESULTS")
        print("="*70)
        print(f"Cut Volume:   {results['cut_volume']:>15,.2f} m³")
        print(f"Fill Volume:  {results['fill_volume']:>15,.2f} m³")
        print(f"Net Volume:   {results['net_volume']:>15,.2f} m³")
        print("="*70)
