"""
Distribution similarity analysis module for comparing lookback and forecast segments.
"""

import numpy as np
import pandas as pd
from scipy import stats
try:
    from scipy.spatial.distance import wasserstein_distance
except ImportError:
    # Fallback implementation for older scipy versions
    def wasserstein_distance(u_values, v_values):
        """Fallback implementation of wasserstein distance."""
        u_values = np.asarray(u_values, dtype=float)
        v_values = np.asarray(v_values, dtype=float)
        
        # Simple approximation using sorted values
        u_sorted = np.sort(u_values)
        v_sorted = np.sort(v_values)
        
        # Pad to same length
        max_len = max(len(u_sorted), len(v_sorted))
        if len(u_sorted) < max_len:
            u_sorted = np.pad(u_sorted, (0, max_len - len(u_sorted)), mode='edge')
        if len(v_sorted) < max_len:
            v_sorted = np.pad(v_sorted, (0, max_len - len(v_sorted)), mode='edge')
        
        return np.mean(np.abs(u_sorted - v_sorted))

from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings


class DistributionSimilarityAnalyzer:
    """
    Analyzes the distributional similarity between lookback and forecast segments
    using various statistical tests and distance metrics.
    """
    
    def __init__(self):
        """Initialize the DistributionSimilarityAnalyzer."""
        pass
    
    def analyze(self, lookback: np.ndarray, forecast: np.ndarray) -> Dict:
        """
        Comprehensive analysis of distribution similarity between lookback and forecast.
        
        Args:
            lookback: Lookback window data (n_lookback, n_features)
            forecast: Forecast period data (n_forecast, n_features)
            
        Returns:
            Dictionary containing similarity analysis results
        """
        results = {}
        
        # Statistical feature comparison
        results.update(self._compare_statistical_features(lookback, forecast))
        
        # Statistical tests
        results.update(self._perform_statistical_tests(lookback, forecast))
        
        # Distance metrics
        results.update(self._calculate_distance_metrics(lookback, forecast))
        
        # Comprehensive similarity scores
        results.update(self._calculate_similarity_scores(results))
        
        return results
    
    def _compare_statistical_features(self, lookback: np.ndarray, forecast: np.ndarray) -> Dict:
        """
        Compare basic statistical features between lookback and forecast.
        
        Returns:
            Dictionary with statistical feature comparisons
        """
        results = {}
        
        # Calculate statistics for each feature/column
        n_features = lookback.shape[1]
        
        for feature_idx in range(n_features):
            lookback_col = lookback[:, feature_idx]
            forecast_col = forecast[:, feature_idx]
            
            # Basic statistics
            lookback_stats = {
                'mean': np.mean(lookback_col),
                'std': np.std(lookback_col),
                'skewness': stats.skew(lookback_col),
                'kurtosis': stats.kurtosis(lookback_col),
                'median': np.median(lookback_col),
                'iqr': np.percentile(lookback_col, 75) - np.percentile(lookback_col, 25)
            }
            
            forecast_stats = {
                'mean': np.mean(forecast_col),
                'std': np.std(forecast_col),
                'skewness': stats.skew(forecast_col),
                'kurtosis': stats.kurtosis(forecast_col),
                'median': np.median(forecast_col),
                'iqr': np.percentile(forecast_col, 75) - np.percentile(forecast_col, 25)
            }
            
            # Calculate relative differences
            feature_similarity = {}
            for stat_name in lookback_stats:
                lb_val = lookback_stats[stat_name]
                fc_val = forecast_stats[stat_name]
                
                # Handle division by zero
                if abs(lb_val) < 1e-8 and abs(fc_val) < 1e-8:
                    rel_diff = 0.0
                elif abs(lb_val) < 1e-8:
                    rel_diff = 1.0  # Maximum difference
                else:
                    rel_diff = abs(fc_val - lb_val) / (abs(lb_val) + 1e-8)
                
                feature_similarity[f'{stat_name}_rel_diff'] = min(rel_diff, 2.0)  # Cap at 200%
            
            results[f'feature_{feature_idx}_similarity'] = feature_similarity
            results[f'feature_{feature_idx}_lookback_stats'] = lookback_stats
            results[f'feature_{feature_idx}_forecast_stats'] = forecast_stats
        
        # Overall statistical similarity
        all_rel_diffs = []
        for feature_idx in range(n_features):
            feature_sim = results[f'feature_{feature_idx}_similarity']
            all_rel_diffs.extend([
                feature_sim['mean_rel_diff'],
                feature_sim['std_rel_diff'],
                feature_sim['median_rel_diff']
            ])
        
        results['overall_statistical_similarity'] = 1.0 / (1.0 + np.mean(all_rel_diffs))
        
        return results
    
    def _perform_statistical_tests(self, lookback: np.ndarray, forecast: np.ndarray) -> Dict:
        """
        Perform various statistical tests to compare distributions.
        
        Returns:
            Dictionary with statistical test results
        """
        results = {}
        n_features = lookback.shape[1]
        
        ks_stats = []
        ks_pvalues = []
        anderson_stats = []
        
        for feature_idx in range(n_features):
            lookback_col = lookback[:, feature_idx]
            forecast_col = forecast[:, feature_idx]
            
            # Kolmogorov-Smirnov test
            try:
                ks_stat, ks_pval = stats.ks_2samp(lookback_col, forecast_col)
                ks_stats.append(ks_stat)
                ks_pvalues.append(ks_pval)
            except Exception:
                ks_stats.append(1.0)  # Worst case
                ks_pvalues.append(0.0)
            
            # Anderson-Darling test (approximate using combined data)
            try:
                combined_data = np.concatenate([lookback_col, forecast_col])
                # Create groups
                groups = np.array([0] * len(lookback_col) + [1] * len(forecast_col))
                
                # Approximate Anderson-Darling using rank statistics
                from scipy.stats import ranksums
                _, ad_pval = ranksums(lookback_col, forecast_col)
                
                anderson_stats.append(1.0 - ad_pval)  # Convert to similarity measure
            except Exception:
                anderson_stats.append(0.0)
        
        # Store results
        results['ks_statistics'] = ks_stats
        results['ks_pvalues'] = ks_pvalues
        results['anderson_statistics'] = anderson_stats
        
        # Summary scores
        results['ks_test_score'] = 1.0 - np.mean(ks_stats)  # Convert to similarity score
        results['anderson_score'] = np.mean(anderson_stats)
        
        return results
    
    def _calculate_distance_metrics(self, lookback: np.ndarray, forecast: np.ndarray) -> Dict:
        """
        Calculate various distance metrics between distributions.
        
        Returns:
            Dictionary with distance metric results
        """
        results = {}
        n_features = lookback.shape[1]
        
        wasserstein_distances = []
        kl_divergences = []
        js_divergences = []
        
        for feature_idx in range(n_features):
            lookback_col = lookback[:, feature_idx]
            forecast_col = forecast[:, feature_idx]
            
            # Wasserstein distance (Earth Mover's Distance)
            try:
                wd = wasserstein_distance(lookback_col, forecast_col)
                # Normalize by the range of the data
                data_range = max(np.ptp(lookback_col), np.ptp(forecast_col), 1e-8)
                normalized_wd = wd / data_range
                wasserstein_distances.append(min(normalized_wd, 2.0))  # Cap at 2
            except Exception:
                wasserstein_distances.append(2.0)
            
            # KL divergence (using histogram approximation)
            try:
                kl_div = self._calculate_kl_divergence(lookback_col, forecast_col)
                kl_divergences.append(min(kl_div, 10.0))  # Cap at 10
            except Exception:
                kl_divergences.append(10.0)
            
            # Jensen-Shannon divergence
            try:
                js_div = self._calculate_js_divergence(lookback_col, forecast_col)
                js_divergences.append(js_div)
            except Exception:
                js_divergences.append(1.0)  # Maximum JS divergence
        
        results['wasserstein_distances'] = wasserstein_distances
        results['kl_divergences'] = kl_divergences
        results['js_divergences'] = js_divergences
        
        # Summary scores (convert distances to similarity scores)
        results['wasserstein_score'] = 1.0 / (1.0 + np.mean(wasserstein_distances))
        results['kl_divergence_score'] = 1.0 / (1.0 + np.mean(kl_divergences))
        results['js_divergence_score'] = 1.0 - np.mean(js_divergences)
        
        return results
    
    def _calculate_kl_divergence(self, data1: np.ndarray, data2: np.ndarray, bins: int = 20) -> float:
        """
        Calculate KL divergence between two distributions using histogram approximation.
        
        Args:
            data1: First distribution data
            data2: Second distribution data
            bins: Number of bins for histogram
            
        Returns:
            KL divergence value
        """
        # Create common bins
        combined_data = np.concatenate([data1, data2])
        min_val, max_val = np.min(combined_data), np.max(combined_data)
        
        if max_val == min_val:
            return 0.0  # Identical distributions
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Calculate histograms
        hist1, _ = np.histogram(data1, bins=bin_edges, density=True)
        hist2, _ = np.histogram(data2, bins=bin_edges, density=True)
        
        # Normalize to probabilities
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        hist1 = hist1 + eps
        hist2 = hist2 + eps
        
        # Calculate KL divergence
        kl_div = np.sum(hist1 * np.log(hist1 / hist2))
        
        return kl_div
    
    def _calculate_js_divergence(self, data1: np.ndarray, data2: np.ndarray, bins: int = 20) -> float:
        """
        Calculate Jensen-Shannon divergence between two distributions.
        
        Args:
            data1: First distribution data
            data2: Second distribution data
            bins: Number of bins for histogram
            
        Returns:
            JS divergence value (0 to 1)
        """
        # Create histograms
        combined_data = np.concatenate([data1, data2])
        min_val, max_val = np.min(combined_data), np.max(combined_data)
        
        if max_val == min_val:
            return 0.0
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        hist1, _ = np.histogram(data1, bins=bin_edges, density=True)
        hist2, _ = np.histogram(data2, bins=bin_edges, density=True)
        
        # Normalize
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        # Add epsilon
        eps = 1e-10
        hist1 = hist1 + eps
        hist2 = hist2 + eps
        
        # Calculate M (midpoint distribution)
        m = (hist1 + hist2) / 2
        
        # Calculate KL divergences
        kl1 = np.sum(hist1 * np.log(hist1 / m))
        kl2 = np.sum(hist2 * np.log(hist2 / m))
        
        # JS divergence
        js_div = (kl1 + kl2) / 2
        
        # Convert to similarity measure (JS divergence ranges from 0 to log(2))
        max_js = np.log(2)
        normalized_js = js_div / max_js
        
        return min(normalized_js, 1.0)
    
    def _calculate_similarity_scores(self, analysis_results: Dict) -> Dict:
        """
        Calculate comprehensive similarity scores from all analysis results.
        
        Args:
            analysis_results: Dictionary containing all previous analysis results
            
        Returns:
            Dictionary with calculated similarity scores
        """
        scores = {}
        
        # Statistical similarity score
        statistical_score = analysis_results.get('overall_statistical_similarity', 0.5)
        scores['statistical_similarity_score'] = statistical_score
        
        # Distribution test score
        ks_score = analysis_results.get('ks_test_score', 0.5)
        anderson_score = analysis_results.get('anderson_score', 0.5)
        test_score = (ks_score + anderson_score) / 2
        scores['distribution_test_score'] = test_score
        
        # Distance metric score
        wasserstein_score = analysis_results.get('wasserstein_score', 0.5)
        kl_score = analysis_results.get('kl_divergence_score', 0.5)
        js_score = analysis_results.get('js_divergence_score', 0.5)
        distance_score = (wasserstein_score + kl_score + js_score) / 3
        scores['distance_metric_score'] = distance_score
        
        # Overall similarity score (weighted combination)
        overall_score = (
            0.4 * statistical_score + 
            0.3 * test_score + 
            0.3 * distance_score
        )
        scores['overall_similarity_score'] = overall_score
        
        return scores