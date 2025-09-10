"""
Pattern analysis module for analyzing trends, volatility, and periodicity in time series segments.
"""

import numpy as np
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple, Optional, Union
import warnings


class PatternAnalyzer:
    """
    Analyzes patterns in time series data including trends, volatility, and periodicity.
    """
    
    def __init__(self):
        """Initialize the PatternAnalyzer."""
        pass
    
    def analyze(self, lookback: np.ndarray, forecast: np.ndarray) -> Dict:
        """
        Comprehensive pattern analysis of lookback and forecast segments.
        
        Args:
            lookback: Lookback window data (n_lookback, n_features)
            forecast: Forecast period data (n_forecast, n_features)
            
        Returns:
            Dictionary containing pattern analysis results
        """
        results = {}
        
        # Trend analysis
        results.update(self._analyze_trends(lookback, forecast))
        
        # Volatility analysis
        results.update(self._analyze_volatility(lookback, forecast))
        
        # Periodicity analysis
        results.update(self._analyze_periodicity(lookback, forecast))
        
        # Autocorrelation analysis
        results.update(self._analyze_autocorrelation(lookback, forecast))
        
        # Pattern consistency scores
        results.update(self._calculate_pattern_scores(results))
        
        return results
    
    def _analyze_trends(self, lookback: np.ndarray, forecast: np.ndarray) -> Dict:
        """
        Analyze trend consistency between lookback and forecast segments.
        
        Returns:
            Dictionary with trend analysis results
        """
        results = {}
        n_features = lookback.shape[1]
        
        lookback_trends = []
        forecast_trends = []
        trend_similarities = []
        
        for feature_idx in range(n_features):
            lookback_col = lookback[:, feature_idx]
            forecast_col = forecast[:, feature_idx]
            
            # Calculate linear trends
            lb_trend = self._calculate_linear_trend(lookback_col)
            fc_trend = self._calculate_linear_trend(forecast_col)
            
            lookback_trends.append(lb_trend)
            forecast_trends.append(fc_trend)
            
            # Calculate trend similarity
            similarity = self._calculate_trend_similarity(lb_trend, fc_trend)
            trend_similarities.append(similarity)
        
        results['lookback_trends'] = lookback_trends
        results['forecast_trends'] = forecast_trends
        results['trend_similarities'] = trend_similarities
        
        # Mann-Kendall trend test
        mk_results = self._mann_kendall_test(lookback, forecast)
        results.update(mk_results)
        
        # Overall trend consistency
        results['trend_consistency_score'] = np.mean(trend_similarities)
        
        return results
    
    def _calculate_linear_trend(self, data: np.ndarray) -> Dict:
        """
        Calculate linear trend parameters for a data series.
        
        Args:
            data: 1D array of time series data
            
        Returns:
            Dictionary with trend parameters
        """
        if len(data) < 2:
            return {'slope': 0, 'intercept': 0, 'r_squared': 0, 'p_value': 1.0}
        
        x = np.arange(len(data)).reshape(-1, 1)
        y = data.reshape(-1, 1)
        
        try:
            # Fit linear regression
            model = LinearRegression()
            model.fit(x, y)
            
            slope = model.coef_[0][0]
            intercept = model.intercept_[0]
            
            # Calculate R-squared
            y_pred = model.predict(x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-8))
            
            # Calculate p-value using correlation
            correlation, p_value = stats.pearsonr(x.flatten(), y.flatten())
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared[0] if isinstance(r_squared, np.ndarray) else r_squared,
                'p_value': p_value,
                'correlation': correlation
            }
            
        except Exception:
            return {'slope': 0, 'intercept': 0, 'r_squared': 0, 'p_value': 1.0, 'correlation': 0}
    
    def _calculate_trend_similarity(self, trend1: Dict, trend2: Dict) -> float:
        """
        Calculate similarity between two trend dictionaries.
        
        Returns:
            Similarity score between 0 and 1
        """
        # Compare slope directions
        slope1, slope2 = trend1['slope'], trend2['slope']
        
        # If both slopes are near zero, they are similar
        if abs(slope1) < 1e-6 and abs(slope2) < 1e-6:
            direction_similarity = 1.0
        else:
            # Check if slopes have same direction
            if (slope1 > 0 and slope2 > 0) or (slope1 < 0 and slope2 < 0):
                # Same direction, compare magnitudes
                magnitude_ratio = min(abs(slope1), abs(slope2)) / (max(abs(slope1), abs(slope2)) + 1e-8)
                direction_similarity = magnitude_ratio
            else:
                # Different directions
                direction_similarity = 0.0
        
        # Compare R-squared values (trend strength)
        r_sq1, r_sq2 = trend1['r_squared'], trend2['r_squared']
        r_sq_similarity = 1.0 - abs(r_sq1 - r_sq2)
        
        # Combined similarity
        overall_similarity = 0.7 * direction_similarity + 0.3 * r_sq_similarity
        
        return max(0.0, min(1.0, overall_similarity))
    
    def _mann_kendall_test(self, lookback: np.ndarray, forecast: np.ndarray) -> Dict:
        """
        Perform Mann-Kendall trend test on both segments.
        
        Returns:
            Dictionary with Mann-Kendall test results
        """
        results = {}
        n_features = lookback.shape[1]
        
        lookback_mk_results = []
        forecast_mk_results = []
        
        for feature_idx in range(n_features):
            lookback_col = lookback[:, feature_idx]
            forecast_col = forecast[:, feature_idx]
            
            # Mann-Kendall test for lookback
            lb_mk = self._single_mann_kendall(lookback_col)
            lookback_mk_results.append(lb_mk)
            
            # Mann-Kendall test for forecast
            fc_mk = self._single_mann_kendall(forecast_col)
            forecast_mk_results.append(fc_mk)
        
        results['lookback_mann_kendall'] = lookback_mk_results
        results['forecast_mann_kendall'] = forecast_mk_results
        
        # Trend direction consistency
        trend_direction_consistency = []
        for lb_mk, fc_mk in zip(lookback_mk_results, forecast_mk_results):
            if lb_mk['trend'] == fc_mk['trend']:
                trend_direction_consistency.append(1.0)
            else:
                trend_direction_consistency.append(0.0)
        
        results['mann_kendall_consistency'] = np.mean(trend_direction_consistency)
        
        return results
    
    def _single_mann_kendall(self, data: np.ndarray) -> Dict:
        """
        Perform Mann-Kendall test on a single time series.
        
        Returns:
            Dictionary with test results
        """
        n = len(data)
        if n < 3:
            return {'statistic': 0, 'p_value': 1.0, 'trend': 'no trend'}
        
        # Calculate S statistic
        S = 0
        for i in range(n-1):
            for j in range(i+1, n):
                if data[j] > data[i]:
                    S += 1
                elif data[j] < data[i]:
                    S -= 1
        
        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate standardized test statistic
        if S > 0:
            z = (S - 1) / np.sqrt(var_s)
        elif S < 0:
            z = (S + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Calculate p-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Determine trend direction
        alpha = 0.05
        if p_value < alpha:
            if S > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
        else:
            trend = 'no trend'
        
        return {
            'statistic': S,
            'z_score': z,
            'p_value': p_value,
            'trend': trend
        }
    
    def _analyze_volatility(self, lookback: np.ndarray, forecast: np.ndarray) -> Dict:
        """
        Analyze volatility patterns and consistency between segments.
        
        Returns:
            Dictionary with volatility analysis results
        """
        results = {}
        n_features = lookback.shape[1]
        
        lookback_volatilities = []
        forecast_volatilities = []
        volatility_similarities = []
        
        for feature_idx in range(n_features):
            lookback_col = lookback[:, feature_idx]
            forecast_col = forecast[:, feature_idx]
            
            # Calculate various volatility measures
            lb_vol = self._calculate_volatility_measures(lookback_col)
            fc_vol = self._calculate_volatility_measures(forecast_col)
            
            lookback_volatilities.append(lb_vol)
            forecast_volatilities.append(fc_vol)
            
            # Calculate volatility similarity
            vol_similarity = self._calculate_volatility_similarity(lb_vol, fc_vol)
            volatility_similarities.append(vol_similarity)
        
        results['lookback_volatilities'] = lookback_volatilities
        results['forecast_volatilities'] = forecast_volatilities
        results['volatility_similarities'] = volatility_similarities
        
        # Overall volatility consistency
        results['volatility_consistency_score'] = np.mean(volatility_similarities)
        
        return results
    
    def _calculate_volatility_measures(self, data: np.ndarray) -> Dict:
        """
        Calculate various volatility measures for a time series.
        
        Returns:
            Dictionary with volatility measures
        """
        if len(data) < 2:
            return {'std': 0, 'rolling_std_mean': 0, 'rolling_std_std': 0, 'returns_std': 0}
        
        measures = {}
        
        # Standard deviation
        measures['std'] = np.std(data)
        
        # Rolling standard deviation
        window_size = min(5, len(data) // 2)
        if window_size >= 2:
            rolling_stds = []
            for i in range(len(data) - window_size + 1):
                window_data = data[i:i+window_size]
                rolling_stds.append(np.std(window_data))
            
            measures['rolling_std_mean'] = np.mean(rolling_stds)
            measures['rolling_std_std'] = np.std(rolling_stds)
        else:
            measures['rolling_std_mean'] = np.std(data)
            measures['rolling_std_std'] = 0
        
        # Returns-based volatility
        if len(data) > 1:
            returns = np.diff(data)
            measures['returns_std'] = np.std(returns)
        else:
            measures['returns_std'] = 0
        
        # Range-based volatility
        measures['range'] = np.ptp(data)  # peak-to-peak
        
        return measures
    
    def _calculate_volatility_similarity(self, vol1: Dict, vol2: Dict) -> float:
        """
        Calculate similarity between two volatility measure dictionaries.
        
        Returns:
            Similarity score between 0 and 1
        """
        similarities = []
        
        for measure in ['std', 'rolling_std_mean', 'returns_std', 'range']:
            val1 = vol1.get(measure, 0)
            val2 = vol2.get(measure, 0)
            
            if val1 == 0 and val2 == 0:
                similarity = 1.0
            elif val1 == 0 or val2 == 0:
                similarity = 0.0
            else:
                # Calculate relative difference
                rel_diff = abs(val2 - val1) / (abs(val1) + 1e-8)
                similarity = 1.0 / (1.0 + rel_diff)
            
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def _analyze_periodicity(self, lookback: np.ndarray, forecast: np.ndarray) -> Dict:
        """
        Analyze periodic patterns using FFT and autocorrelation.
        
        Returns:
            Dictionary with periodicity analysis results
        """
        results = {}
        n_features = lookback.shape[1]
        
        lookback_periods = []
        forecast_periods = []
        periodicity_similarities = []
        
        for feature_idx in range(n_features):
            lookback_col = lookback[:, feature_idx]
            forecast_col = forecast[:, feature_idx]
            
            # FFT-based periodicity analysis
            lb_fft_analysis = self._fft_analysis(lookback_col)
            fc_fft_analysis = self._fft_analysis(forecast_col)
            
            lookback_periods.append(lb_fft_analysis)
            forecast_periods.append(fc_fft_analysis)
            
            # Calculate periodicity similarity
            period_similarity = self._calculate_periodicity_similarity(lb_fft_analysis, fc_fft_analysis)
            periodicity_similarities.append(period_similarity)
        
        results['lookback_periodicity'] = lookback_periods
        results['forecast_periodicity'] = forecast_periods
        results['periodicity_similarities'] = periodicity_similarities
        
        # Overall periodicity consistency
        results['periodicity_consistency_score'] = np.mean(periodicity_similarities)
        
        return results
    
    def _fft_analysis(self, data: np.ndarray) -> Dict:
        """
        Perform FFT analysis to identify dominant frequencies.
        
        Returns:
            Dictionary with FFT analysis results
        """
        if len(data) < 4:
            return {'dominant_freq': 0, 'dominant_power': 0, 'spectral_entropy': 0}
        
        # Remove mean to focus on oscillations
        data_centered = data - np.mean(data)
        
        # Perform FFT
        fft_values = fft(data_centered)
        frequencies = fftfreq(len(data), d=1.0)
        
        # Calculate power spectrum (positive frequencies only)
        positive_freq_idx = frequencies > 0
        positive_freqs = frequencies[positive_freq_idx]
        power_spectrum = np.abs(fft_values[positive_freq_idx]) ** 2
        
        if len(power_spectrum) == 0:
            return {'dominant_freq': 0, 'dominant_power': 0, 'spectral_entropy': 0}
        
        # Find dominant frequency
        dominant_idx = np.argmax(power_spectrum)
        dominant_freq = positive_freqs[dominant_idx]
        dominant_power = power_spectrum[dominant_idx]
        
        # Calculate spectral entropy
        normalized_power = power_spectrum / (np.sum(power_spectrum) + 1e-8)
        spectral_entropy = -np.sum(normalized_power * np.log(normalized_power + 1e-8))
        
        # Convert frequency to period
        dominant_period = 1.0 / (dominant_freq + 1e-8) if dominant_freq > 0 else float('inf')
        
        return {
            'dominant_freq': dominant_freq,
            'dominant_period': dominant_period,
            'dominant_power': dominant_power,
            'spectral_entropy': spectral_entropy,
            'power_spectrum': power_spectrum,
            'frequencies': positive_freqs
        }
    
    def _calculate_periodicity_similarity(self, fft1: Dict, fft2: Dict) -> float:
        """
        Calculate similarity between two FFT analysis results.
        
        Returns:
            Similarity score between 0 and 1
        """
        # Compare dominant frequencies
        freq1 = fft1.get('dominant_freq', 0)
        freq2 = fft2.get('dominant_freq', 0)
        
        if freq1 == 0 and freq2 == 0:
            freq_similarity = 1.0
        elif freq1 == 0 or freq2 == 0:
            freq_similarity = 0.0
        else:
            freq_diff = abs(freq2 - freq1) / (max(freq1, freq2) + 1e-8)
            freq_similarity = 1.0 / (1.0 + freq_diff)
        
        # Compare spectral entropies
        entropy1 = fft1.get('spectral_entropy', 0)
        entropy2 = fft2.get('spectral_entropy', 0)
        
        entropy_diff = abs(entropy2 - entropy1)
        entropy_similarity = 1.0 / (1.0 + entropy_diff)
        
        # Combine similarities
        overall_similarity = 0.6 * freq_similarity + 0.4 * entropy_similarity
        
        return overall_similarity
    
    def _analyze_autocorrelation(self, lookback: np.ndarray, forecast: np.ndarray) -> Dict:
        """
        Analyze autocorrelation patterns in both segments.
        
        Returns:
            Dictionary with autocorrelation analysis results
        """
        results = {}
        n_features = lookback.shape[1]
        
        lookback_autocorrs = []
        forecast_autocorrs = []
        autocorr_similarities = []
        
        for feature_idx in range(n_features):
            lookback_col = lookback[:, feature_idx]
            forecast_col = forecast[:, feature_idx]
            
            # Calculate autocorrelation functions
            lb_autocorr = self._calculate_autocorrelation(lookback_col)
            fc_autocorr = self._calculate_autocorrelation(forecast_col)
            
            lookback_autocorrs.append(lb_autocorr)
            forecast_autocorrs.append(fc_autocorr)
            
            # Calculate autocorrelation similarity
            autocorr_similarity = self._calculate_autocorr_similarity(lb_autocorr, fc_autocorr)
            autocorr_similarities.append(autocorr_similarity)
        
        results['lookback_autocorrelations'] = lookback_autocorrs
        results['forecast_autocorrelations'] = forecast_autocorrs
        results['autocorrelation_similarities'] = autocorr_similarities
        
        # Overall autocorrelation consistency
        results['autocorrelation_consistency_score'] = np.mean(autocorr_similarities)
        
        return results
    
    def _calculate_autocorrelation(self, data: np.ndarray, max_lags: int = 10) -> Dict:
        """
        Calculate autocorrelation function for a time series.
        
        Returns:
            Dictionary with autocorrelation results
        """
        if len(data) < 3:
            return {'autocorrelations': [1.0], 'significant_lags': []}
        
        # Limit max_lags to reasonable value
        max_lags = min(max_lags, len(data) // 2)
        
        autocorrelations = []
        significant_lags = []
        
        for lag in range(1, max_lags + 1):
            if lag >= len(data):
                break
                
            # Calculate autocorrelation at this lag
            x1 = data[:-lag]
            x2 = data[lag:]
            
            if len(x1) > 0:
                correlation = np.corrcoef(x1, x2)[0, 1]
                if not np.isnan(correlation):
                    autocorrelations.append(correlation)
                    
                    # Check if correlation is significant (simple threshold)
                    if abs(correlation) > 0.3:  # Arbitrary significance threshold
                        significant_lags.append(lag)
                else:
                    autocorrelations.append(0.0)
            else:
                autocorrelations.append(0.0)
        
        return {
            'autocorrelations': autocorrelations,
            'significant_lags': significant_lags,
            'max_autocorr': max(autocorrelations) if autocorrelations else 0.0,
            'mean_autocorr': np.mean(autocorrelations) if autocorrelations else 0.0
        }
    
    def _calculate_autocorr_similarity(self, autocorr1: Dict, autocorr2: Dict) -> float:
        """
        Calculate similarity between two autocorrelation analyses.
        
        Returns:
            Similarity score between 0 and 1
        """
        # Compare autocorrelation patterns
        ac1 = autocorr1.get('autocorrelations', [])
        ac2 = autocorr2.get('autocorrelations', [])
        
        if not ac1 and not ac2:
            return 1.0
        
        if not ac1 or not ac2:
            return 0.0
        
        # Pad shorter sequence with zeros
        max_len = max(len(ac1), len(ac2))
        ac1_padded = ac1 + [0.0] * (max_len - len(ac1))
        ac2_padded = ac2 + [0.0] * (max_len - len(ac2))
        
        # Calculate correlation between autocorrelation functions
        try:
            pattern_correlation = np.corrcoef(ac1_padded, ac2_padded)[0, 1]
            if np.isnan(pattern_correlation):
                pattern_correlation = 0.0
        except:
            pattern_correlation = 0.0
        
        # Compare significant lags
        lags1 = set(autocorr1.get('significant_lags', []))
        lags2 = set(autocorr2.get('significant_lags', []))
        
        if not lags1 and not lags2:
            lag_similarity = 1.0
        elif not lags1 or not lags2:
            lag_similarity = 0.0
        else:
            # Jaccard similarity for significant lags
            intersection = len(lags1.intersection(lags2))
            union = len(lags1.union(lags2))
            lag_similarity = intersection / union if union > 0 else 0.0
        
        # Combine similarities
        overall_similarity = 0.7 * abs(pattern_correlation) + 0.3 * lag_similarity
        
        return overall_similarity
    
    def _calculate_pattern_scores(self, analysis_results: Dict) -> Dict:
        """
        Calculate comprehensive pattern consistency scores.
        
        Returns:
            Dictionary with calculated pattern scores
        """
        scores = {}
        
        # Individual component scores
        trend_score = analysis_results.get('trend_consistency_score', 0.5)
        volatility_score = analysis_results.get('volatility_consistency_score', 0.5)
        periodicity_score = analysis_results.get('periodicity_consistency_score', 0.5)
        autocorr_score = analysis_results.get('autocorrelation_consistency_score', 0.5)
        mk_score = analysis_results.get('mann_kendall_consistency', 0.5)
        
        scores['trend_consistency_score'] = trend_score
        scores['volatility_consistency_score'] = volatility_score
        scores['periodicity_consistency_score'] = periodicity_score
        scores['autocorrelation_consistency_score'] = autocorr_score
        scores['mann_kendall_consistency_score'] = mk_score
        
        # Overall pattern consistency (weighted average)
        overall_pattern_score = (
            0.3 * trend_score +
            0.25 * volatility_score +
            0.2 * periodicity_score +
            0.15 * autocorr_score +
            0.1 * mk_score
        )
        
        scores['overall_pattern_consistency_score'] = overall_pattern_score
        
        return scores