# Time Series Segment Prediction Value Evaluation System

A comprehensive system for evaluating the prediction worthiness of time series segments using multiple analytical methods including distribution similarity, anomaly detection, pattern analysis, and data quality assessment.

## Overview

This module provides a complete solution for determining whether a time series data segment is suitable for training prediction models. It evaluates segments across multiple dimensions and provides actionable recommendations.

## Features

### Core Evaluation Components

1. **Data Preprocessing**
   - Automatic data splitting (lookback vs forecast windows)
   - Data normalization and standardization
   - Missing value detection and handling
   - Data quality assessment

2. **Distribution Similarity Analysis**
   - Statistical feature comparison (mean, std, skewness, kurtosis)
   - Kolmogorov-Smirnov test
   - Anderson-Darling test
   - Wasserstein distance
   - KL divergence
   - Jensen-Shannon divergence

3. **Anomaly Detection**
   - Statistical methods (Z-score, IQR, Modified Z-score)
   - Machine learning methods (Isolation Forest, LOF)
   - Time series specific methods (change point detection, seasonal anomalies)
   - Cross-segment anomaly analysis

4. **Pattern Analysis**
   - Trend consistency analysis
   - Mann-Kendall trend tests
   - Volatility analysis
   - Periodicity analysis (FFT-based)
   - Autocorrelation analysis

5. **Visualization**
   - Comprehensive evaluation dashboards
   - Distribution comparison plots
   - Anomaly detection visualizations
   - Pattern analysis charts
   - Summary reports

## Installation

### Dependencies

```bash
pip install numpy scipy scikit-learn matplotlib pandas statsmodels seaborn
```

### Module Structure

```
timeseries_evaluation/
├── __init__.py                 # Module initialization
├── evaluator.py                # Main TimeSeriesSegmentEvaluator class
├── data_preprocessing.py       # Data preprocessing utilities
├── distribution_similarity.py  # Distribution analysis methods
├── anomaly_detection.py        # Anomaly detection algorithms
├── pattern_analysis.py         # Pattern and trend analysis
├── visualization.py            # Visualization utilities
├── tests/                      # Test suite
│   ├── __init__.py
│   └── test_evaluator.py
├── example_usage.py            # Usage examples
└── README.md                   # This file
```

## Quick Start

### Basic Usage

```python
from timeseries_evaluation import TimeSeriesSegmentEvaluator
import numpy as np

# Create sample data (n_rows, n_features)
data = np.random.randn(50, 3)

# Initialize evaluator
evaluator = TimeSeriesSegmentEvaluator()

# Evaluate segment
result = evaluator.evaluate(data, segment_id="sample_001")

print(f"Overall Score: {result['overall_score']:.3f}")
print(f"Prediction Worthy: {result['prediction_worthy']}")
print(f"Recommendations: {result['recommendations']}")
```

### Batch Evaluation

```python
# Evaluate multiple segments
segments = [data1, data2, data3]  # List of numpy arrays
segment_ids = ["seg_001", "seg_002", "seg_003"]

results = evaluator.evaluate_batch(segments, segment_ids)

# Get summary statistics
summary = evaluator.get_evaluation_summary(results)
print(f"Prediction worthy: {summary['prediction_worthy_ratio']:.1%}")
```

### Custom Configuration

```python
evaluator = TimeSeriesSegmentEvaluator(
    lookback_ratio=0.8,           # Use 80% for lookback window
    anomaly_threshold=0.05,       # Lower anomaly tolerance
    similarity_threshold=0.7,     # Higher similarity requirement
    trend_threshold=0.8,          # Stricter trend consistency
    visualization_dir="./plots"   # Custom visualization directory
)
```

### Custom Scoring Weights

```python
# Emphasize anomaly detection
custom_weights = {
    "distribution_similarity": 0.20,
    "anomaly_level": 0.50,        # Increased weight
    "trend_consistency": 0.15,
    "data_quality": 0.15
}

evaluator.update_weights(custom_weights)
```

## Output Format

### Evaluation Result Structure

```python
{
    "segment_id": "sample_001",
    "overall_score": 0.75,
    "prediction_worthy": True,
    "detailed_scores": {
        "distribution_similarity": 0.72,
        "anomaly_level": 0.85,
        "trend_consistency": 0.68,
        "data_quality": 0.90
    },
    "recommendations": [
        "良好片段：建议用于预测模型训练",
        "注意第15-18行的轻微异常"
    ],
    "visualization_paths": [
        "./visualizations/sample_001_overview.png",
        "./visualizations/sample_001_dashboard.png"
    ],
    "metadata": {
        "lookback_ratio": 0.7,
        "lookback_shape": (35, 3),
        "forecast_shape": (15, 3),
        "original_shape": (50, 3)
    }
}
```

### Score Interpretation

- **Overall Score**: 0-1 range, higher is better
  - `>= 0.8`: Excellent for prediction
  - `>= 0.6`: Good for prediction
  - `>= 0.4`: Use with caution
  - `< 0.4`: Not recommended

- **Component Scores**: 0-1 range
  - **Distribution Similarity**: How similar lookback and forecast distributions are
  - **Anomaly Level**: 1 - (anomaly ratio), higher means fewer anomalies
  - **Trend Consistency**: Consistency of trends between segments
  - **Data Quality**: Overall data quality (missing values, normalization)

## Advanced Usage

### Running Tests

```bash
cd src/timeseries_evaluation
python tests/test_evaluator.py
```

### Running Examples

```bash
cd src/timeseries_evaluation
python example_usage.py
```

This will demonstrate:
- Single segment evaluation
- Batch processing
- Custom weight configuration
- Visualization generation

### Creating Visualizations

The system automatically creates comprehensive visualizations:

1. **Overview Plot**: Time series with anomalies and split points
2. **Distribution Plot**: Histogram comparisons between segments
3. **Anomaly Plot**: Detailed anomaly detection results
4. **Pattern Plot**: Trend, volatility, and periodicity analysis
5. **Dashboard**: Summary evaluation dashboard

Example:
```python
result = evaluator.evaluate(data, create_visualizations=True)
# Check result['visualization_paths'] for file locations
```

### Integration with Existing Projects

For integration with the main TrackingNAS project:

```python
# In your tracking/prediction pipeline
from src.timeseries_evaluation import TimeSeriesSegmentEvaluator

def filter_prediction_segments(time_series_data):
    evaluator = TimeSeriesSegmentEvaluator()
    
    # Evaluate each segment
    results = []
    for segment in time_series_data:
        result = evaluator.evaluate(segment)
        if result['prediction_worthy']:
            results.append(segment)
    
    return results
```

## Technical Details

### Scoring Algorithm

The overall score is calculated as a weighted combination:

```
Overall Score = 0.30 × Distribution Similarity +
                0.25 × Anomaly Level +
                0.25 × Trend Consistency +
                0.20 × Data Quality
```

### Evaluation Methods

#### Distribution Similarity
- **Statistical Features**: Compares means, standard deviations, skewness, kurtosis
- **Hypothesis Tests**: KS test, Anderson-Darling test
- **Distance Metrics**: Wasserstein, KL divergence, JS divergence

#### Anomaly Detection
- **Statistical**: Z-score (3σ), IQR method, Modified Z-score
- **Machine Learning**: Isolation Forest, Local Outlier Factor
- **Time Series**: Change point detection, seasonal anomaly detection

#### Pattern Analysis
- **Trend Analysis**: Linear regression, Mann-Kendall tests
- **Volatility**: Rolling standard deviation, returns-based volatility
- **Periodicity**: FFT analysis, autocorrelation functions

### Performance Considerations

- **Memory**: Efficient processing for segments up to 10,000 points
- **Speed**: Typical evaluation time ~0.1-0.5 seconds per segment
- **Scalability**: Batch processing supported for large datasets

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Visualization Errors**: Check matplotlib backend and directory permissions
3. **Memory Issues**: For very large segments, consider subsampling

### Error Handling

The system gracefully handles:
- Missing values in data
- Insufficient data points (minimum 4 rows required)
- Singular matrices in statistical calculations
- Visualization failures (returns empty paths)

## Contributing

To extend the system:

1. **Add New Anomaly Methods**: Extend `AnomalyDetector` class
2. **Add Analysis Methods**: Extend `PatternAnalyzer` class
3. **Custom Visualizations**: Extend `VisualizationHandler` class
4. **New Metrics**: Update scoring in `TimeSeriesSegmentEvaluator`

## License

This module is part of the TrackingNAS project and follows the same licensing terms.

## Examples

See `example_usage.py` for comprehensive examples including:
- Different data types (good, anomalous, trend changes, high volatility)
- Batch processing workflows
- Custom configuration scenarios
- Visualization generation

## API Reference

### Main Classes

- `TimeSeriesSegmentEvaluator`: Main evaluation interface
- `DataPreprocessor`: Data preprocessing utilities
- `DistributionSimilarityAnalyzer`: Distribution analysis methods
- `AnomalyDetector`: Anomaly detection algorithms
- `PatternAnalyzer`: Pattern and trend analysis
- `VisualizationHandler`: Visualization creation

For detailed API documentation, see the docstrings in each module.