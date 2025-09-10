"""
Time Series Segment Evaluation Module

This module provides comprehensive evaluation of time series segments to determine
their prediction worthiness for model training.
"""

from .evaluator import TimeSeriesSegmentEvaluator

__version__ = "1.0.0"
__all__ = ["TimeSeriesSegmentEvaluator"]