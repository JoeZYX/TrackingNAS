"""
Example usage of the TimeSeriesSegmentEvaluator system.

This script demonstrates how to use the time series segment evaluation system
to assess prediction worthiness of time series data segments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evaluator import TimeSeriesSegmentEvaluator
import os


def generate_sample_data():
    """Generate sample time series data for demonstration."""
    print("Generating sample time series data...")
    
    np.random.seed(42)
    n_rows = 60  # 60 time points
    n_cols = 4   # 4 features
    
    # Create different types of time series
    segments = {}
    
    # 1. Good segment: consistent trend and low noise
    print("  - Creating 'good' segment with consistent patterns...")
    time_indices = np.arange(n_rows)
    good_segment = np.zeros((n_rows, n_cols))
    
    for col in range(n_cols):
        # Consistent upward trend
        trend = 0.15 * time_indices
        # Seasonal component
        seasonal = 2 * np.sin(2 * np.pi * time_indices / 12) + np.cos(2 * np.pi * time_indices / 8)
        # Low noise
        noise = np.random.normal(0, 0.3, n_rows)
        
        good_segment[:, col] = trend + seasonal + noise + col * 2  # Offset features
    
    segments['good_segment'] = good_segment
    
    # 2. Anomalous segment: contains outliers and jumps
    print("  - Creating 'anomalous' segment with outliers...")
    anomalous_segment = good_segment.copy()
    
    # Add some anomalies
    anomalous_segment[15, 0] = 20   # Large positive outlier
    anomalous_segment[25, 1] = -15  # Large negative outlier
    anomalous_segment[35, 2] = 25   # Another outlier
    anomalous_segment[45:50, 3] += 10  # Step change
    
    segments['anomalous_segment'] = anomalous_segment
    
    # 3. Trend change segment: different trends in lookback vs forecast
    print("  - Creating 'trend_change' segment...")
    trend_change_segment = np.zeros((n_rows, n_cols))
    split_point = int(n_rows * 0.7)
    
    for col in range(n_cols):
        # Lookback: positive trend
        lookback_indices = np.arange(split_point)
        trend_change_segment[:split_point, col] = (
            0.3 * lookback_indices + 
            np.sin(2 * np.pi * lookback_indices / 8) + 
            np.random.normal(0, 0.4, split_point) + 
            col * 3
        )
        
        # Forecast: negative trend or flat
        forecast_indices = np.arange(n_rows - split_point)
        last_value = trend_change_segment[split_point-1, col]
        trend_change_segment[split_point:, col] = (
            last_value - 0.2 * forecast_indices + 
            np.sin(2 * np.pi * (forecast_indices + split_point) / 8) + 
            np.random.normal(0, 0.4, len(forecast_indices))
        )
    
    segments['trend_change_segment'] = trend_change_segment
    
    # 4. High volatility segment: very noisy data
    print("  - Creating 'high_volatility' segment...")
    high_volatility_segment = np.zeros((n_rows, n_cols))
    
    for col in range(n_cols):
        # Weak trend
        trend = 0.05 * time_indices
        # High noise
        noise = np.random.normal(0, 2.0, n_rows)
        # Some seasonal component
        seasonal = 0.5 * np.sin(2 * np.pi * time_indices / 10)
        
        high_volatility_segment[:, col] = trend + seasonal + noise + col * 2
    
    segments['high_volatility_segment'] = high_volatility_segment
    
    # 5. Missing data segment
    print("  - Creating 'missing_data' segment...")
    missing_data_segment = good_segment.copy()
    
    # Introduce missing values (NaN)
    missing_data_segment[10:13, 0] = np.nan
    missing_data_segment[20, 1] = np.nan
    missing_data_segment[30:32, 2] = np.nan
    
    segments['missing_data_segment'] = missing_data_segment
    
    return segments


def demonstrate_single_evaluation():
    """Demonstrate evaluation of a single time series segment."""
    print("\n" + "="*60)
    print("SINGLE SEGMENT EVALUATION DEMONSTRATION")
    print("="*60)
    
    # Generate sample data
    segments = generate_sample_data()
    
    # Initialize evaluator
    evaluator = TimeSeriesSegmentEvaluator(
        lookback_ratio=0.7,
        anomaly_threshold=0.1,
        similarity_threshold=0.6,
        trend_threshold=0.7,
        visualization_dir="/tmp/timeseries_visualizations"
    )
    
    # Evaluate the 'good' segment
    segment_name = 'good_segment'
    segment_data = segments[segment_name]
    
    print(f"\nEvaluating '{segment_name}'...")
    print(f"Data shape: {segment_data.shape}")
    
    # Perform evaluation
    result = evaluator.evaluate(
        segment_data, 
        segment_id=segment_name,
        create_visualizations=True
    )
    
    # Display results
    print("\n" + "-"*40)
    print("EVALUATION RESULTS")
    print("-"*40)
    print(f"Segment ID: {result['segment_id']}")
    print(f"Overall Score: {result['overall_score']:.3f}")
    print(f"Prediction Worthy: {result['prediction_worthy']}")
    
    print("\nDetailed Scores:")
    for score_name, score_value in result['detailed_scores'].items():
        print(f"  {score_name}: {score_value:.3f}")
    
    print("\nRecommendations:")
    for i, recommendation in enumerate(result['recommendations'], 1):
        print(f"  {i}. {recommendation}")
    
    print("\nMetadata:")
    metadata = result['metadata']
    print(f"  Original shape: {metadata['original_shape']}")
    print(f"  Lookback shape: {metadata['lookback_shape']}")
    print(f"  Forecast shape: {metadata['forecast_shape']}")
    print(f"  Lookback ratio: {metadata['lookback_ratio']}")
    
    if result['visualization_paths']:
        print(f"\nVisualizations created: {len(result['visualization_paths'])} files")
        for path in result['visualization_paths']:
            print(f"  - {os.path.basename(path)}")
    
    return result


def demonstrate_batch_evaluation():
    """Demonstrate batch evaluation of multiple segments."""
    print("\n" + "="*60)
    print("BATCH EVALUATION DEMONSTRATION")
    print("="*60)
    
    # Generate sample data
    segments = generate_sample_data()
    
    # Initialize evaluator
    evaluator = TimeSeriesSegmentEvaluator()
    
    # Prepare data for batch evaluation
    segment_data_list = list(segments.values())
    segment_ids = list(segments.keys())
    
    print(f"\nEvaluating {len(segment_data_list)} segments in batch...")
    
    # Perform batch evaluation
    results = evaluator.evaluate_batch(
        segment_data_list, 
        segment_ids, 
        create_visualizations=False  # Disable for batch processing
    )
    
    # Display results
    print("\n" + "-"*60)
    print("BATCH EVALUATION RESULTS")
    print("-"*60)
    
    for result in results:
        if 'error' not in result:
            print(f"\nSegment: {result['segment_id']}")
            print(f"  Overall Score: {result['overall_score']:.3f}")
            print(f"  Prediction Worthy: {result['prediction_worthy']}")
            print(f"  Key Scores:")
            detailed = result['detailed_scores']
            print(f"    - Similarity: {detailed['distribution_similarity']:.3f}")
            print(f"    - Anomaly Level: {detailed['anomaly_level']:.3f}")
            print(f"    - Trend Consistency: {detailed['trend_consistency']:.3f}")
            print(f"    - Data Quality: {detailed['data_quality']:.3f}")
        else:
            print(f"\nSegment: {result['segment_id']} - ERROR: {result['error']}")
    
    # Generate summary
    summary = evaluator.get_evaluation_summary(results)
    
    print("\n" + "-"*40)
    print("BATCH SUMMARY")
    print("-"*40)
    print(f"Total segments: {summary['total_segments']}")
    print(f"Valid evaluations: {summary['valid_evaluations']}")
    print(f"Prediction worthy: {summary['prediction_worthy_count']} ({summary['prediction_worthy_ratio']:.1%})")
    
    print("\nOverall Score Statistics:")
    stats = summary['overall_score_stats']
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std: {stats['std']:.3f}")
    print(f"  Min: {stats['min']:.3f}")
    print(f"  Max: {stats['max']:.3f}")
    print(f"  Median: {stats['median']:.3f}")
    
    return results, summary


def demonstrate_custom_weights():
    """Demonstrate using custom scoring weights."""
    print("\n" + "="*60)
    print("CUSTOM WEIGHTS DEMONSTRATION")
    print("="*60)
    
    segments = generate_sample_data()
    
    # Initialize evaluator with default weights
    evaluator = TimeSeriesSegmentEvaluator()
    
    # Evaluate with default weights
    segment_data = segments['good_segment']
    
    print("\nEvaluating with DEFAULT weights...")
    result_default = evaluator.evaluate(segment_data, segment_id="default_weights", 
                                      create_visualizations=False)
    
    print("Default weights:")
    for weight_name, weight_value in evaluator.weights.items():
        print(f"  {weight_name}: {weight_value}")
    print(f"Overall score: {result_default['overall_score']:.3f}")
    
    # Update weights to emphasize anomaly detection
    print("\nUpdating weights to emphasize ANOMALY DETECTION...")
    new_weights = {
        "distribution_similarity": 0.20,
        "anomaly_level": 0.50,  # Increased emphasis
        "trend_consistency": 0.15,
        "data_quality": 0.15
    }
    
    evaluator.update_weights(new_weights)
    
    print("New weights:")
    for weight_name, weight_value in evaluator.weights.items():
        print(f"  {weight_name}: {weight_value}")
    
    # Evaluate with new weights
    result_custom = evaluator.evaluate(segment_data, segment_id="custom_weights", 
                                     create_visualizations=False)
    
    print(f"Overall score with custom weights: {result_custom['overall_score']:.3f}")
    
    # Compare results
    print("\n" + "-"*40)
    print("COMPARISON")
    print("-"*40)
    print(f"Score difference: {result_custom['overall_score'] - result_default['overall_score']:.3f}")
    
    # Test with anomalous data
    print("\nTesting with ANOMALOUS data...")
    anomalous_data = segments['anomalous_segment']
    
    # Reset to default weights
    default_weights = {
        "distribution_similarity": 0.30,
        "anomaly_level": 0.25,
        "trend_consistency": 0.25,
        "data_quality": 0.20
    }
    evaluator.update_weights(default_weights)
    
    result_anom_default = evaluator.evaluate(anomalous_data, segment_id="anom_default", 
                                           create_visualizations=False)
    
    # Apply anomaly-focused weights
    evaluator.update_weights(new_weights)
    result_anom_custom = evaluator.evaluate(anomalous_data, segment_id="anom_custom", 
                                          create_visualizations=False)
    
    print(f"Anomalous data - Default weights: {result_anom_default['overall_score']:.3f}")
    print(f"Anomalous data - Anomaly-focused weights: {result_anom_custom['overall_score']:.3f}")
    print(f"Difference: {result_anom_custom['overall_score'] - result_anom_default['overall_score']:.3f}")


def create_comparison_visualization():
    """Create a comparison visualization of different segment types."""
    print("\n" + "="*60)
    print("CREATING COMPARISON VISUALIZATION")
    print("="*60)
    
    segments = generate_sample_data()
    
    # Create a comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Time Series Segment Comparison', fontsize=16, fontweight='bold')
    
    segment_names = list(segments.keys())
    
    for i, (name, data) in enumerate(segments.items()):
        if i >= 6:  # Only plot first 6
            break
            
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # Plot first feature only for clarity
        time_indices = np.arange(len(data))
        ax.plot(time_indices, data[:, 0], 'b-', linewidth=1.5, alpha=0.8)
        
        # Mark lookback/forecast split
        split_point = int(len(data) * 0.7)
        ax.axvline(x=split_point, color='red', linestyle='--', alpha=0.7, 
                  label='Lookback/Forecast Split')
        
        # Shade regions
        ax.axvspan(0, split_point, alpha=0.2, color='blue', label='Lookback')
        ax.axvspan(split_point, len(data), alpha=0.2, color='green', label='Forecast')
        
        ax.set_title(name.replace('_', ' ').title())
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        if i == 0:  # Add legend to first plot
            ax.legend(loc='upper left', fontsize=8)
    
    # Hide unused subplots
    for i in range(len(segments), 6):
        row, col = i // 3, i % 3
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs("/tmp/timeseries_visualizations", exist_ok=True)
    comparison_path = "/tmp/timeseries_visualizations/segment_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison visualization saved to: {comparison_path}")
    
    return comparison_path


def main():
    """Main demonstration function."""
    print("TIME SERIES SEGMENT EVALUATION SYSTEM")
    print("=====================================")
    print("This demonstration shows how to evaluate time series segments")
    print("for their prediction worthiness using multiple analytical methods.")
    
    try:
        # Run demonstrations
        print("\n1. Generating sample data and running single evaluation...")
        single_result = demonstrate_single_evaluation()
        
        print("\n2. Running batch evaluation...")
        batch_results, batch_summary = demonstrate_batch_evaluation()
        
        print("\n3. Demonstrating custom scoring weights...")
        demonstrate_custom_weights()
        
        print("\n4. Creating comparison visualization...")
        comparison_path = create_comparison_visualization()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"✅ Single evaluation: Overall score = {single_result['overall_score']:.3f}")
        print(f"✅ Batch evaluation: {batch_summary['prediction_worthy_count']}/{batch_summary['total_segments']} segments worthy")
        print(f"✅ Visualizations created in: /tmp/timeseries_visualizations/")
        print(f"✅ System is ready for production use!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Please check the error details above.")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\nYou can now use the TimeSeriesSegmentEvaluator in your own projects!")
        print("Example usage:")
        print("  from timeseries_evaluation import TimeSeriesSegmentEvaluator")
        print("  evaluator = TimeSeriesSegmentEvaluator()")
        print("  result = evaluator.evaluate(your_data)")
    else:
        print("\nPlease resolve the issues above before using the system.")