"""
Anomaly detection module for identifying outliers and anomalies in time series segments.
"""

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from typing import Dict, List, Tuple, Optional, Union
import warnings


class AnomalyDetector:
    """
    Comprehensive anomaly detection for time series data using multiple methods.
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize the AnomalyDetector.
        
        Args:
            contamination: Expected proportion of anomalies (default: 0.1)
        """
        self.contamination = contamination
    
    def detect(self, original_data: np.ndarray, 
              lookback: np.ndarray, 
              forecast: np.ndarray) -> Dict:
        """
        Detect anomalies using multiple methods and provide comprehensive results.
        
        Args:
            original_data: Original unprocessed data segment
            lookback: Processed lookback window data
            forecast: Processed forecast period data
            
        Returns:
            Dictionary containing anomaly detection results
        """
        results = {}
        
        # Statistical anomaly detection
        results.update(self._detect_statistical_anomalies(original_data))
        
        # Machine learning based anomaly detection
        results.update(self._detect_ml_anomalies(original_data))
        
        # Time series specific anomaly detection
        results.update(self._detect_timeseries_anomalies(original_data))
        
        # Segment-specific analysis (lookback vs forecast)
        results.update(self._analyze_segment_anomalies(lookback, forecast))
        
        # Comprehensive anomaly scoring
        results.update(self._calculate_anomaly_scores(results, original_data))
        
        return results
    
    def _detect_statistical_anomalies(self, data: np.ndarray) -> Dict:
        """
        Detect anomalies using statistical methods (Z-score, IQR, etc.).
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary with statistical anomaly detection results
        """
        results = {}
        n_rows, n_cols = data.shape
        
        # Z-score based detection
        zscore_anomalies = []
        iqr_anomalies = []
        modified_zscore_anomalies = []
        
        for col in range(n_cols):
            col_data = data[:, col]
            
            # Standard Z-score (threshold = 3)
            z_scores = np.abs(stats.zscore(col_data))
            col_zscore_anomalies = np.where(z_scores > 3)[0]
            zscore_anomalies.extend([(row, col) for row in col_zscore_anomalies])
            
            # IQR method
            Q1 = np.percentile(col_data, 25)
            Q3 = np.percentile(col_data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            col_iqr_anomalies = np.where((col_data < lower_bound) | (col_data > upper_bound))[0]
            iqr_anomalies.extend([(row, col) for row in col_iqr_anomalies])
            
            # Modified Z-score (using median)
            median = np.median(col_data)
            mad = np.median(np.abs(col_data - median))
            modified_z_scores = 0.6745 * (col_data - median) / (mad + 1e-8)
            col_mod_anomalies = np.where(np.abs(modified_z_scores) > 3.5)[0]
            modified_zscore_anomalies.extend([(row, col) for row in col_mod_anomalies])
        
        results['zscore_anomalies'] = zscore_anomalies
        results['iqr_anomalies'] = iqr_anomalies
        results['modified_zscore_anomalies'] = modified_zscore_anomalies
        
        # Summary statistics
        total_points = n_rows * n_cols
        results['zscore_anomaly_ratio'] = len(zscore_anomalies) / total_points
        results['iqr_anomaly_ratio'] = len(iqr_anomalies) / total_points
        results['modified_zscore_anomaly_ratio'] = len(modified_zscore_anomalies) / total_points
        
        return results
    
    def _detect_ml_anomalies(self, data: np.ndarray) -> Dict:
        """
        Detect anomalies using machine learning methods.
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary with ML-based anomaly detection results
        """
        results = {}
        
        # Reshape data for ML algorithms
        n_rows, n_cols = data.shape
        data_reshaped = data.copy()
        
        # Handle any remaining NaN or inf values
        data_clean = np.nan_to_num(data_reshaped, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Isolation Forest
        try:
            iso_forest = IsolationForest(
                contamination=self.contamination, 
                random_state=42, 
                n_estimators=100
            )
            iso_predictions = iso_forest.fit_predict(data_clean)
            iso_anomalies = np.where(iso_predictions == -1)[0]
            results['isolation_forest_anomalies'] = iso_anomalies.tolist()
            results['isolation_forest_scores'] = iso_forest.decision_function(data_clean)
        except Exception as e:
            results['isolation_forest_anomalies'] = []
            results['isolation_forest_scores'] = np.zeros(n_rows)
        
        # Local Outlier Factor
        try:
            # Adjust n_neighbors based on data size
            n_neighbors = min(20, max(2, n_rows // 5))
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors, 
                contamination=self.contamination
            )
            lof_predictions = lof.fit_predict(data_clean)
            lof_anomalies = np.where(lof_predictions == -1)[0]
            results['lof_anomalies'] = lof_anomalies.tolist()
            results['lof_scores'] = lof.negative_outlier_factor_
        except Exception as e:
            results['lof_anomalies'] = []
            results['lof_scores'] = np.ones(n_rows) * -1
        
        # Summary
        total_points = n_rows
        results['isolation_forest_anomaly_ratio'] = len(results['isolation_forest_anomalies']) / total_points
        results['lof_anomaly_ratio'] = len(results['lof_anomalies']) / total_points
        
        return results
    
    def _detect_timeseries_anomalies(self, data: np.ndarray) -> Dict:
        """
        Detect anomalies specific to time series characteristics.
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary with time series specific anomaly results
        """
        results = {}
        n_rows, n_cols = data.shape
        
        # Change point detection (simple version)
        change_points = []
        sudden_changes = []
        
        for col in range(n_cols):
            col_data = data[:, col]
            
            # Detect sudden changes using rolling statistics
            if n_rows > 4:
                # Calculate rolling mean and std
                window_size = min(5, n_rows // 3)
                
                for i in range(window_size, n_rows - window_size):
                    before_window = col_data[i-window_size:i]
                    after_window = col_data[i:i+window_size]
                    
                    before_mean = np.mean(before_window)
                    after_mean = np.mean(after_window)
                    combined_std = np.std(col_data[i-window_size:i+window_size])
                    
                    # Check for significant mean shift
                    if combined_std > 0:
                        change_magnitude = abs(after_mean - before_mean) / combined_std
                        if change_magnitude > 2.0:  # Threshold for change point
                            change_points.append((i, col, change_magnitude))
            
            # Detect sudden spikes/drops
            if n_rows > 2:
                diff = np.diff(col_data)
                diff_threshold = np.std(diff) * 3  # 3-sigma threshold
                
                sudden_indices = np.where(np.abs(diff) > diff_threshold)[0]
                for idx in sudden_indices:
                    sudden_changes.append((idx + 1, col, diff[idx]))  # +1 because diff reduces length by 1
        
        results['change_points'] = change_points
        results['sudden_changes'] = sudden_changes
        
        # Seasonal decomposition anomalies (simplified)
        seasonal_anomalies = self._detect_seasonal_anomalies(data)
        results.update(seasonal_anomalies)
        
        # Summary
        results['change_point_count'] = len(change_points)
        results['sudden_change_count'] = len(sudden_changes)
        
        return results
    
    def _detect_seasonal_anomalies(self, data: np.ndarray) -> Dict:
        """
        Detect anomalies in seasonal patterns (simplified approach).
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary with seasonal anomaly results
        """
        results = {}
        n_rows, n_cols = data.shape
        
        if n_rows < 8:  # Not enough data for meaningful seasonal analysis
            results['seasonal_anomalies'] = []
            results['seasonal_anomaly_ratio'] = 0.0
            return results
        
        seasonal_anomalies = []
        
        # Simple seasonal analysis: look for patterns that repeat
        for col in range(n_cols):
            col_data = data[:, col]
            
            # Try different potential periods
            potential_periods = [2, 3, 4, 6, 8, 12]  # Common periods for short series
            potential_periods = [p for p in potential_periods if p < n_rows // 2]
            
            if not potential_periods:
                continue
            
            best_period = None
            best_correlation = 0
            
            # Find the period with highest autocorrelation
            for period in potential_periods:
                if n_rows >= 2 * period:
                    # Calculate autocorrelation at this lag
                    try:
                        correlation = np.corrcoef(
                            col_data[:-period], 
                            col_data[period:]
                        )[0, 1]
                        
                        if not np.isnan(correlation) and correlation > best_correlation:
                            best_correlation = correlation
                            best_period = period
                    except:
                        continue
            
            # If we found a good periodic pattern, detect anomalies
            if best_period and best_correlation > 0.5:
                # Compare each cycle with the expected pattern
                n_cycles = n_rows // best_period
                
                if n_cycles >= 2:
                    # Calculate mean cycle
                    cycles = []
                    for cycle in range(n_cycles):
                        start_idx = cycle * best_period
                        end_idx = start_idx + best_period
                        if end_idx <= n_rows:
                            cycles.append(col_data[start_idx:end_idx])
                    
                    if len(cycles) >= 2:
                        mean_cycle = np.mean(cycles, axis=0)
                        cycle_std = np.std(cycles, axis=0)
                        
                        # Check each cycle for anomalies
                        for cycle_idx, cycle in enumerate(cycles):
                            for pos in range(len(cycle)):
                                expected = mean_cycle[pos]
                                actual = cycle[pos]
                                threshold = cycle_std[pos] * 2  # 2-sigma threshold
                                
                                if threshold > 0 and abs(actual - expected) > threshold:
                                    original_idx = cycle_idx * best_period + pos
                                    seasonal_anomalies.append((original_idx, col))
        
        results['seasonal_anomalies'] = seasonal_anomalies
        results['seasonal_anomaly_ratio'] = len(seasonal_anomalies) / (n_rows * n_cols)
        
        return results
    
    def _analyze_segment_anomalies(self, lookback: np.ndarray, forecast: np.ndarray) -> Dict:
        """
        Analyze anomalies within and between lookback and forecast segments.
        
        Args:
            lookback: Lookback window data
            forecast: Forecast period data
            
        Returns:
            Dictionary with segment-specific anomaly analysis
        """
        results = {}
        
        # Compare anomaly rates between segments
        lookback_anomalies = self._count_segment_anomalies(lookback)
        forecast_anomalies = self._count_segment_anomalies(forecast)
        
        results['lookback_anomaly_ratio'] = lookback_anomalies / lookback.size
        results['forecast_anomaly_ratio'] = forecast_anomalies / forecast.size
        
        # Cross-segment anomaly analysis
        # Check if forecast values are anomalous relative to lookback distribution
        cross_segment_anomalies = 0
        total_forecast_points = forecast.size
        
        for col in range(forecast.shape[1]):
            lookback_col = lookback[:, col]
            forecast_col = forecast[:, col]
            
            # Calculate lookback statistics
            lb_mean = np.mean(lookback_col)
            lb_std = np.std(lookback_col)
            
            if lb_std > 0:
                # Check how many forecast points are anomalous relative to lookback
                z_scores = np.abs((forecast_col - lb_mean) / lb_std)
                anomalous_forecasts = np.sum(z_scores > 2.5)  # 2.5-sigma threshold
                cross_segment_anomalies += anomalous_forecasts
        
        results['cross_segment_anomaly_ratio'] = cross_segment_anomalies / total_forecast_points
        
        return results
    
    def _count_segment_anomalies(self, segment: np.ndarray) -> int:
        """
        Count anomalies in a data segment using multiple methods.
        
        Args:
            segment: Data segment to analyze
            
        Returns:
            Number of anomalies detected
        """
        if segment.size == 0:
            return 0
        
        anomaly_count = 0
        
        for col in range(segment.shape[1]):
            col_data = segment[:, col]
            
            # Z-score anomalies
            if len(col_data) > 1:
                z_scores = np.abs(stats.zscore(col_data))
                anomaly_count += np.sum(z_scores > 3)
            
            # IQR anomalies
            if len(col_data) > 3:
                Q1 = np.percentile(col_data, 25)
                Q3 = np.percentile(col_data, 75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    anomaly_count += np.sum((col_data < lower_bound) | (col_data > upper_bound))
        
        return anomaly_count
    
    def _calculate_anomaly_scores(self, detection_results: Dict, original_data: np.ndarray) -> Dict:
        """
        Calculate comprehensive anomaly scores from all detection results.
        
        Args:
            detection_results: Dictionary containing all detection results
            original_data: Original data for normalization
            
        Returns:
            Dictionary with calculated anomaly scores and indices
        """
        scores = {}
        
        # Collect all anomaly indices
        all_anomaly_indices = set()
        
        # Statistical anomalies
        for anomaly_type in ['zscore_anomalies', 'iqr_anomalies', 'modified_zscore_anomalies']:
            if anomaly_type in detection_results:
                for row, col in detection_results[anomaly_type]:
                    all_anomaly_indices.add(row)
        
        # ML-based anomalies
        for anomaly_type in ['isolation_forest_anomalies', 'lof_anomalies']:
            if anomaly_type in detection_results:
                for row in detection_results[anomaly_type]:
                    all_anomaly_indices.add(row)
        
        # Time series anomalies
        if 'change_points' in detection_results:
            for row, col, magnitude in detection_results['change_points']:
                all_anomaly_indices.add(row)
        
        if 'sudden_changes' in detection_results:
            for row, col, change in detection_results['sudden_changes']:
                all_anomaly_indices.add(row)
        
        if 'seasonal_anomalies' in detection_results:
            for row, col in detection_results['seasonal_anomalies']:
                all_anomaly_indices.add(row)
        
        # Calculate overall anomaly statistics
        total_data_points = original_data.shape[0]
        unique_anomaly_rows = len(all_anomaly_indices)
        
        scores['anomaly_indices'] = sorted(list(all_anomaly_indices))
        scores['unique_anomaly_count'] = unique_anomaly_rows
        scores['overall_anomaly_ratio'] = unique_anomaly_rows / total_data_points
        
        # Method agreement score (how many methods agree)
        method_counts = {}
        for idx in all_anomaly_indices:
            method_counts[idx] = 0
            
            # Count statistical methods
            for anomaly_type in ['zscore_anomalies', 'iqr_anomalies', 'modified_zscore_anomalies']:
                if any(row == idx for row, col in detection_results.get(anomaly_type, [])):
                    method_counts[idx] += 1
            
            # Count ML methods
            if idx in detection_results.get('isolation_forest_anomalies', []):
                method_counts[idx] += 1
            if idx in detection_results.get('lof_anomalies', []):
                method_counts[idx] += 1
            
            # Count time series methods
            if any(row == idx for row, col, _ in detection_results.get('change_points', [])):
                method_counts[idx] += 1
            if any(row == idx for row, col, _ in detection_results.get('sudden_changes', [])):
                method_counts[idx] += 1
            if any(row == idx for row, col in detection_results.get('seasonal_anomalies', [])):
                method_counts[idx] += 1
        
        # High-confidence anomalies (detected by multiple methods)
        high_confidence_anomalies = [idx for idx, count in method_counts.items() if count >= 2]
        scores['high_confidence_anomalies'] = high_confidence_anomalies
        scores['high_confidence_ratio'] = len(high_confidence_anomalies) / total_data_points
        
        # Weighted anomaly score
        avg_statistical_ratio = np.mean([
            detection_results.get('zscore_anomaly_ratio', 0),
            detection_results.get('iqr_anomaly_ratio', 0),
            detection_results.get('modified_zscore_anomaly_ratio', 0)
        ])
        
        avg_ml_ratio = np.mean([
            detection_results.get('isolation_forest_anomaly_ratio', 0),
            detection_results.get('lof_anomaly_ratio', 0)
        ])
        
        ts_ratio = detection_results.get('seasonal_anomaly_ratio', 0)
        
        weighted_anomaly_score = (
            0.4 * avg_statistical_ratio +
            0.4 * avg_ml_ratio +
            0.2 * ts_ratio
        )
        
        scores['weighted_anomaly_score'] = weighted_anomaly_score
        
        return scores