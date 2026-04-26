#!/usr/bin/env python3
"""
Grade Estimator - ML-enhanced mineral grade prediction.

Implements:
  - Classical Kriging (baseline)
  - Random Forest (ML)
  - XGBoost (ML)
  - Neural Network - MLPRegressor (ML)

Reference papers supporting ML > Kriging:
  - Montoya-Araque & Chica-Olmo (2018): "Comparison of ML and geostatistical methods"
  - Yilmaz et al. (2020): "Machine learning in mineral resource estimation"
  - Duarte & Teixeira (2023): "Deep learning for grade estimation"
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.spatial.distance import cdist
from scipy.linalg import solve
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class KrigingEstimator:
    """Classical ordinary kriging for baseline comparison."""
    
    def __init__(self, variogram_model='linear', nlags=6):
        self.variogram_model = variogram_model
        self.nlags = nlags
        self.semivariance = None
        self.distances = None
        self.train_x = None
        self.train_y = None
    
    def _compute_semivariance(self, distances, values):
        """Compute empirical semivariance."""
        n = len(values)
        semivar = []
        dist_bins = []
        
        # Simple semivariance - sum of squared differences
        max_dist = np.max(distances[distances < np.inf])
        
        for i in range(self.nlags):
            dmin = i * (max_dist / self.nlags)
            dmax = (i + 1) * (max_dist / self.nlags)
            
            # Find pairs within distance range
            mask = (distances >= dmin) & (distances < dmax) & (distances < np.inf)
            
            if np.any(mask):
                # Get the values for paired samples
                pairs = np.where(mask)
                if len(pairs[0]) > 0:
                    sv = np.mean((values[pairs[0]] - values[pairs[1]]) ** 2) / 2.0
                    dist = (dmin + dmax) / 2.0
                    semivar.append(sv)
                    dist_bins.append(dist)
        
        if len(semivar) == 0:
            # Fallback - use equal-length bins
            semivar = [0.1 * i for i in range(1, self.nlags + 1)]
            dist_bins = [i * (max_dist / self.nlags) for i in range(1, self.nlags + 1)]
        
        return np.array(dist_bins), np.array(semivar)
    
    def _fit_variogram(self, dist_bins, semivar):
        """Fit variogram model."""
        if self.variogram_model == 'linear':
            # Fit: gamma(h) = slope * h
            valid_idx = dist_bins > 0
            slope = np.polyfit(dist_bins[valid_idx], semivar[valid_idx], 1)[0]
            return {'type': 'linear', 'slope': slope}
        else:
            return {'type': 'linear', 'slope': 1.0}
    
    def _variogram_value(self, distance, params):
        """Get variogram value at given distance."""
        if params['type'] == 'linear':
            return params['slope'] * distance
        return distance
    
    def fit(self, X, y):
        """Fit kriging model."""
        self.train_x = X
        self.train_y = y
        
        # Compute pairwise distances
        self.distances = cdist(X, X)
        np.fill_diagonal(self.distances, np.inf)
        
        # Compute semivariance
        dist_bins, semivar = self._compute_semivariance(self.distances, y)
        self.semivariance = self._fit_variogram(dist_bins, semivar)
    
    def predict(self, X):
        """Predict using kriging."""
        predictions = []
        
        for point in X:
            # Calculate distances to training points
            dists = np.linalg.norm(self.train_x - point, axis=1)
            
            # Get variogram values
            gamma_matrix = np.zeros((len(self.train_x) + 1, len(self.train_x) + 1))
            for i in range(len(self.train_x)):
                for j in range(len(self.train_x)):
                    if i != j:
                        d = np.linalg.norm(self.train_x[i] - self.train_x[j])
                        gamma_matrix[i, j] = self._variogram_value(d, self.semivariance)
            
            # Add ones for drift term
            gamma_matrix[:, -1] = 1
            gamma_matrix[-1, :] = 1
            gamma_matrix[-1, -1] = 0
            
            # Right-hand side
            rhs = np.zeros(len(self.train_x) + 1)
            for i in range(len(self.train_x)):
                rhs[i] = self._variogram_value(dists[i], self.semivariance)
            rhs[-1] = 1
            
            try:
                # Solve for weights
                weights = solve(gamma_matrix, rhs)
                pred = np.dot(weights[:-1], self.train_y)
                predictions.append(max(0, pred))  # Grade cannot be negative
            except:
                # Fallback to inverse distance weighting
                inv_dists = 1.0 / (dists + 1e-6)
                weights = inv_dists / inv_dists.sum()
                predictions.append(np.dot(weights, self.train_y))
        
        return np.array(predictions)


class GradeEstimator:
    """Multi-model grade estimation with automatic model selection."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.models = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
    
    def fit(self, X, y):
        """Train all models and compare performance."""
        print("\nTraining Grade Estimators...")
        
        # Sample data for faster training (use ~500 samples max)
        if len(X) > 500:
            sample_indices = np.random.RandomState(42).choice(len(X), 500, replace=False)
            X = X[sample_indices]
            y = y[sample_indices]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # 1. Kriging (Baseline)
        print("  Kriging (baseline)...", end='')
        kriging = KrigingEstimator()
        kriging.fit(self.X_train, self.y_train)
        y_pred_kriging = kriging.predict(self.X_test)
        kriging_r2 = r2_score(self.y_test, y_pred_kriging)
        kriging_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_kriging))
        self.models['kriging'] = kriging
        self.results['kriging'] = {
            'r2': kriging_r2, 'rmse': kriging_rmse,
            'mae': mean_absolute_error(self.y_test, y_pred_kriging),
            'predictions': y_pred_kriging
        }
        print(f" R²={kriging_r2:.4f}, RMSE={kriging_rmse:.4f}")
        
        # 2. Random Forest
        print("  Random Forest...", end='')
        rf = RandomForestRegressor(
            n_estimators=self.config.get('rf_estimators', 50),
            max_depth=self.config.get('rf_max_depth', 12),
            min_samples_split=self.config.get('rf_min_samples', 5),
            random_state=42, n_jobs=-1
        )
        rf.fit(self.X_train_scaled, self.y_train)
        y_pred_rf = rf.predict(self.X_test_scaled)
        rf_r2 = r2_score(self.y_test, y_pred_rf)
        rf_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_rf))
        self.models['random_forest'] = rf
        self.results['random_forest'] = {
            'r2': rf_r2, 'rmse': rf_rmse,
            'mae': mean_absolute_error(self.y_test, y_pred_rf),
            'predictions': y_pred_rf
        }
        print(f" R²={rf_r2:.4f}, RMSE={rf_rmse:.4f}")
        
        # 3. XGBoost
        if HAS_XGBOOST:
            print("  XGBoost...", end='')
            xgb_model = xgb.XGBRegressor(
                n_estimators=self.config.get('xgb_estimators', 50),
                max_depth=self.config.get('xgb_max_depth', 5),
                learning_rate=self.config.get('xgb_lr', 0.1),
                random_state=42, verbosity=0
            )
            xgb_model.fit(self.X_train_scaled, self.y_train)
            y_pred_xgb = xgb_model.predict(self.X_test_scaled)
            xgb_r2 = r2_score(self.y_test, y_pred_xgb)
            xgb_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_xgb))
            self.models['xgboost'] = xgb_model
            self.results['xgboost'] = {
                'r2': xgb_r2, 'rmse': xgb_rmse,
                'mae': mean_absolute_error(self.y_test, y_pred_xgb),
                'predictions': y_pred_xgb
            }
            print(f" R²={xgb_r2:.4f}, RMSE={xgb_rmse:.4f}")
        
        # 4. Neural Network
        print("  Neural Network...", end='')
        nn = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            max_iter=200,
            random_state=42, early_stopping=True, learning_rate_init=0.01
        )
        nn.fit(self.X_train_scaled, self.y_train)
        y_pred_nn = nn.predict(self.X_test_scaled)
        nn_r2 = r2_score(self.y_test, y_pred_nn)
        nn_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_nn))
        self.models['neural_network'] = nn
        self.results['neural_network'] = {
            'r2': nn_r2, 'rmse': nn_rmse,
            'mae': mean_absolute_error(self.y_test, y_pred_nn),
            'predictions': y_pred_nn
        }
        print(f" R²={nn_r2:.4f}, RMSE={nn_rmse:.4f}")
        
        # Select best model
        best_model = max(self.results.items(), key=lambda x: x[1]['r2'])
        self.best_model_name = best_model[0]
        print(f"\n  BEST MODEL: {self.best_model_name.upper()} (R²={best_model[1]['r2']:.4f})")
    
    def predict(self, X):
        """Predict using best model."""
        if self.best_model_name == 'kriging':
            return self.models['kriging'].predict(X)
        else:
            X_scaled = self.scaler.transform(X)
            return self.models[self.best_model_name].predict(X_scaled)
    
    def confidence_interval(self, X, confidence=0.95):
        """Estimate confidence intervals using residual std."""
        predictions = self.predict(X)
        
        # Use residuals from test set to estimate uncertainty
        best_results = self.results[self.best_model_name]
        residuals = self.y_test - best_results['predictions']
        std_error = np.std(residuals)
        
        # 95% CI: ±1.96 * std_error
        z_score = 1.96 if confidence == 0.95 else 1.645
        margin = z_score * std_error
        
        return predictions, predictions - margin, predictions + margin
    
    def print_accuracy_table(self):
        """Print model comparison table."""
        print("\n" + "="*70)
        print(f"{'Model':<20} {'R² Score':<15} {'RMSE':<15} {'MAE':<15}")
        print("="*70)
        
        for model_name, metrics in self.results.items():
            print(f"{model_name.replace('_', ' ').title():<20} "
                  f"{metrics['r2']:<15.4f} "
                  f"{metrics['rmse']:<15.4f} "
                  f"{metrics['mae']:<15.4f}")
        
        print("="*70)
        print(f"ML models advantage: {(self.results['random_forest']['r2'] - self.results['kriging']['r2'])*100:+.2f}% over Kriging")
        print("="*70)
