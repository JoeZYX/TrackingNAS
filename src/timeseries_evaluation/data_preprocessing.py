"""
Data preprocessing module for time series segment evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, Tuple, Optional, Union


class DataPreprocessor:
    """
    Handles data preprocessing tasks including splitting, normalization, and missing value treatment.
    """
    
    def __init__(self, normalization_method: str = 'standard'):
        """
        Initialize the DataPreprocessor.
        
        Args:
            normalization_method: Method for normalization ('standard', 'robust', 'minmax', 'none')
        """
        self.normalization_method = normalization_method
        self.scaler = None
        
    def preprocess(self, data_segment: np.ndarray, lookback_ratio: float = 0.7) -> Dict:
        """
        Preprocess the data segment by splitting, normalizing, and handling missing values.
        
        Args:
            data_segment: Input data matrix (n rows, m columns)
            lookback_ratio: Ratio of data to use as lookback window
            
        Returns:
            Dictionary containing preprocessed data and metadata
        """
        # Validate input
        if not isinstance(data_segment, np.ndarray):
            data_segment = np.array(data_segment)
        
        if data_segment.ndim != 2:
            raise ValueError("Data segment must be a 2D array (n rows, m columns)")
        
        n_rows, n_cols = data_segment.shape
        if n_rows < 4:  # Minimum required for meaningful split
            raise ValueError("Data segment must have at least 4 rows")
        
        # Handle missing values
        missing_info = self._handle_missing_values(data_segment)
        cleaned_data = missing_info['cleaned_data']
        
        # Split data into lookback and forecast
        split_index = int(n_rows * lookback_ratio)
        if split_index < 2 or (n_rows - split_index) < 2:
            # Adjust split to ensure both parts have at least 2 data points
            split_index = max(2, n_rows - 2)
        
        lookback = cleaned_data[:split_index, :]
        forecast = cleaned_data[split_index:, :]
        
        # Normalize data
        normalization_info = self._normalize_data(lookback, forecast)
        
        # Calculate data quality metrics
        quality_metrics = self._calculate_quality_metrics(
            data_segment, cleaned_data, missing_info
        )
        
        return {
            'lookback': normalization_info['lookback_normalized'],
            'forecast': normalization_info['forecast_normalized'],
            'lookback_original': lookback,
            'forecast_original': forecast,
            'split_index': split_index,
            'missing_ratio': missing_info['missing_ratio'],
            'missing_positions': missing_info['missing_positions'],
            'normalization_quality': normalization_info['quality_score'],
            'scaler': normalization_info['scaler'],
            'data_quality_metrics': quality_metrics
        }
    
    def _handle_missing_values(self, data: np.ndarray) -> Dict:
        """
        Detect and handle missing values in the data.
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary with cleaned data and missing value information
        """
        # Detect missing values (NaN, inf, or extremely large values)
        missing_mask = np.isnan(data) | np.isinf(data) | (np.abs(data) > 1e10)
        missing_positions = np.where(missing_mask)
        missing_ratio = np.sum(missing_mask) / data.size
        
        # Create a copy for cleaning
        cleaned_data = data.copy()
        
        if missing_ratio > 0:
            if missing_ratio > 0.3:  # Too many missing values
                raise ValueError(f"Too many missing values ({missing_ratio:.2%}). "
                               "Consider using a different segment.")
            
            # Handle missing values using interpolation and forward/backward fill
            for col in range(data.shape[1]):
                col_data = cleaned_data[:, col]
                col_missing = missing_mask[:, col]
                
                if np.sum(col_missing) > 0:
                    # Try linear interpolation first
                    valid_indices = ~col_missing
                    if np.sum(valid_indices) >= 2:
                        valid_positions = np.where(valid_indices)[0]
                        valid_values = col_data[valid_indices]
                        
                        # Interpolate missing values
                        missing_indices = np.where(col_missing)[0]
                        interpolated_values = np.interp(
                            missing_indices, valid_positions, valid_values
                        )
                        cleaned_data[missing_indices, col] = interpolated_values
                    else:
                        # If too few valid values, use column mean
                        column_mean = np.nanmean(col_data)
                        if not np.isnan(column_mean):
                            cleaned_data[col_missing, col] = column_mean
                        else:
                            cleaned_data[col_missing, col] = 0.0
        
        return {
            'cleaned_data': cleaned_data,
            'missing_ratio': missing_ratio,
            'missing_positions': missing_positions,
            'original_data': data
        }
    
    def _normalize_data(self, lookback: np.ndarray, forecast: np.ndarray) -> Dict:
        """
        Normalize the lookback and forecast data.
        
        Args:
            lookback: Lookback window data
            forecast: Forecast period data
            
        Returns:
            Dictionary with normalized data and scaler information
        """
        if self.normalization_method == 'none':
            return {
                'lookback_normalized': lookback,
                'forecast_normalized': forecast,
                'scaler': None,
                'quality_score': 1.0
            }
        
        # Choose scaler based on method
        if self.normalization_method == 'standard':
            scaler = StandardScaler()
        elif self.normalization_method == 'robust':
            scaler = RobustScaler()
        else:  # default to standard
            scaler = StandardScaler()
        
        # Fit scaler on lookback data only (avoid data leakage)
        lookback_reshaped = lookback.reshape(-1, lookback.shape[-1])
        scaler.fit(lookback_reshaped)
        
        # Transform both lookback and forecast
        lookback_normalized = scaler.transform(lookback_reshaped).reshape(lookback.shape)
        forecast_reshaped = forecast.reshape(-1, forecast.shape[-1])
        forecast_normalized = scaler.transform(forecast_reshaped).reshape(forecast.shape)
        
        # Calculate normalization quality score
        quality_score = self._evaluate_normalization_quality(
            lookback, lookback_normalized, forecast, forecast_normalized
        )
        
        self.scaler = scaler
        
        return {
            'lookback_normalized': lookback_normalized,
            'forecast_normalized': forecast_normalized,
            'scaler': scaler,
            'quality_score': quality_score
        }
    
    def _evaluate_normalization_quality(self, lookback_orig: np.ndarray, 
                                      lookback_norm: np.ndarray,
                                      forecast_orig: np.ndarray, 
                                      forecast_norm: np.ndarray) -> float:
        """
        Evaluate the quality of normalization.
        
        Returns:
            Quality score between 0 and 1
        """
        try:
            # Check if normalized data has reasonable statistics
            lookback_std = np.std(lookback_norm, axis=0)
            forecast_std = np.std(forecast_norm, axis=0)
            
            # Good normalization should result in unit variance for lookback
            # and reasonable variance for forecast
            lookback_quality = np.mean(np.abs(lookback_std - 1.0) < 0.5)
            
            # Forecast variance should not be too extreme
            forecast_quality = np.mean((forecast_std > 0.1) & (forecast_std < 5.0))
            
            # Check for any remaining inf or nan values
            no_inf_nan = (not np.any(np.isinf(lookback_norm) | np.isnan(lookback_norm)) and
                         not np.any(np.isinf(forecast_norm) | np.isnan(forecast_norm)))
            
            quality_score = (lookback_quality + forecast_quality) / 2.0
            
            return quality_score if no_inf_nan else quality_score * 0.5
            
        except Exception:
            return 0.5  # Default quality score if evaluation fails
    
    def _calculate_quality_metrics(self, original: np.ndarray, 
                                 cleaned: np.ndarray, 
                                 missing_info: Dict) -> Dict:
        """
        Calculate comprehensive data quality metrics.
        
        Returns:
            Dictionary with various data quality indicators
        """
        metrics = {}
        
        # Basic statistics
        metrics['data_shape'] = original.shape
        metrics['missing_ratio'] = missing_info['missing_ratio']
        
        # Variance and stability metrics
        original_var = np.var(cleaned, axis=0)
        metrics['variance_per_column'] = original_var.tolist()
        metrics['variance_stability'] = np.std(original_var) / (np.mean(original_var) + 1e-8)
        
        # Range and outlier metrics
        for col in range(cleaned.shape[1]):
            col_data = cleaned[:, col]
            q75, q25 = np.percentile(col_data, [75, 25])
            iqr = q75 - q25
            
            # Count potential outliers (beyond 1.5 * IQR)
            outliers = np.sum((col_data < (q25 - 1.5 * iqr)) | 
                            (col_data > (q75 + 1.5 * iqr)))
            metrics[f'outliers_col_{col}'] = outliers
        
        # Stationarity indicator (simplified)
        try:
            # Split data into two halves and compare means/variances
            mid_point = cleaned.shape[0] // 2
            first_half = cleaned[:mid_point, :]
            second_half = cleaned[mid_point:, :]
            
            mean_diff = np.abs(np.mean(first_half, axis=0) - np.mean(second_half, axis=0))
            var_ratio = np.var(first_half, axis=0) / (np.var(second_half, axis=0) + 1e-8)
            
            metrics['mean_stability'] = np.mean(mean_diff)
            metrics['variance_ratio_stability'] = np.mean(np.abs(np.log(var_ratio + 1e-8)))
            
        except Exception:
            metrics['mean_stability'] = np.inf
            metrics['variance_ratio_stability'] = np.inf
        
        return metrics
    
    def inverse_transform(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Apply inverse transformation to normalized data.
        
        Args:
            normalized_data: Normalized data array
            
        Returns:
            Original scale data
        """
        if self.scaler is None:
            return normalized_data
        
        original_shape = normalized_data.shape
        reshaped = normalized_data.reshape(-1, normalized_data.shape[-1])
        inverse_transformed = self.scaler.inverse_transform(reshaped)
        
        return inverse_transformed.reshape(original_shape)