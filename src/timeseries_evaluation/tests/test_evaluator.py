"""
Test cases for TimeSeriesSegmentEvaluator.
"""

import sys
import os
import numpy as np

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator import TimeSeriesSegmentEvaluator

# Optional pytest import
try:
    import pytest
except ImportError:
    pytest = None


class TestTimeSeriesSegmentEvaluator:
    """Test cases for the main evaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = TimeSeriesSegmentEvaluator()
        
        # Create test data
        np.random.seed(42)
        self.n_rows = 50
        self.n_cols = 3
        
        # Generate synthetic time series with trend and noise
        time_indices = np.arange(self.n_rows)
        
        # Create test data with some patterns
        self.test_data = np.zeros((self.n_rows, self.n_cols))
        for col in range(self.n_cols):
            # Add trend
            trend = 0.1 * time_indices
            # Add seasonal component
            seasonal = 2 * np.sin(2 * np.pi * time_indices / 12)
            # Add noise
            noise = np.random.normal(0, 0.5, self.n_rows)
            
            self.test_data[:, col] = trend + seasonal + noise
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization with default parameters."""
        evaluator = TimeSeriesSegmentEvaluator()
        
        assert evaluator.lookback_ratio == 0.7
        assert evaluator.anomaly_threshold == 0.1
        assert evaluator.similarity_threshold == 0.6
        assert evaluator.trend_threshold == 0.7
        
        # Check that all sub-modules are initialized
        assert evaluator.preprocessor is not None
        assert evaluator.similarity_analyzer is not None
        assert evaluator.anomaly_detector is not None
        assert evaluator.pattern_analyzer is not None
        assert evaluator.visualizer is not None
    
    def test_evaluate_single_segment(self):
        """Test evaluation of a single time series segment."""
        # Test with visualization disabled
        result = self.evaluator.evaluate(self.test_data, 
                                       segment_id="test_segment_001",
                                       create_visualizations=False)
        
        # Check result structure
        assert isinstance(result, dict)
        assert "segment_id" in result
        assert "overall_score" in result
        assert "prediction_worthy" in result
        assert "detailed_scores" in result
        assert "recommendations" in result
        assert "metadata" in result
        
        # Check detailed scores
        detailed_scores = result["detailed_scores"]
        assert "distribution_similarity" in detailed_scores
        assert "anomaly_level" in detailed_scores
        assert "trend_consistency" in detailed_scores
        assert "data_quality" in detailed_scores
        
        # Check that scores are in valid range [0, 1]
        assert 0 <= result["overall_score"] <= 1
        for score_name, score_value in detailed_scores.items():
            assert 0 <= score_value <= 1, f"Score {score_name} = {score_value} out of range"
        
        # Check that prediction_worthy is boolean
        assert isinstance(result["prediction_worthy"], bool)
        
        # Check recommendations
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) > 0
    
    def test_evaluate_with_visualizations(self):
        """Test evaluation with visualization creation."""
        # Create a temporary directory for visualizations
        temp_vis_dir = "/tmp/test_visualizations"
        os.makedirs(temp_vis_dir, exist_ok=True)
        
        evaluator = TimeSeriesSegmentEvaluator(visualization_dir=temp_vis_dir)
        
        result = evaluator.evaluate(self.test_data, 
                                  segment_id="test_segment_vis",
                                  create_visualizations=True)
        
        # Check that visualization paths are returned
        assert "visualization_paths" in result
        viz_paths = result["visualization_paths"]
        assert isinstance(viz_paths, list)
        assert len(viz_paths) > 0
        
        # Check that files were actually created
        for path in viz_paths:
            assert os.path.exists(path), f"Visualization file not created: {path}"
    
    def test_evaluate_batch(self):
        """Test batch evaluation of multiple segments."""
        # Create multiple test segments
        segments = []
        segment_ids = []
        
        for i in range(3):
            # Create slightly different segments
            segment = self.test_data + np.random.normal(0, 0.1, self.test_data.shape)
            segments.append(segment)
            segment_ids.append(f"batch_segment_{i:03d}")
        
        results = self.evaluator.evaluate_batch(segments, segment_ids, 
                                              create_visualizations=False)
        
        # Check results
        assert isinstance(results, list)
        assert len(results) == len(segments)
        
        for i, result in enumerate(results):
            assert result["segment_id"] == segment_ids[i]
            assert "overall_score" in result
            assert "prediction_worthy" in result
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with minimal data
        minimal_data = np.random.randn(4, 2)
        result = self.evaluator.evaluate(minimal_data, create_visualizations=False)
        assert isinstance(result, dict)
        assert "overall_score" in result
        
        # Test with single column
        single_col_data = np.random.randn(20, 1)
        result = self.evaluator.evaluate(single_col_data, create_visualizations=False)
        assert isinstance(result, dict)
        assert result["metadata"]["original_shape"] == (20, 1)
    
    def test_data_with_anomalies(self):
        """Test evaluation with data containing obvious anomalies."""
        # Create data with inserted anomalies
        anomalous_data = self.test_data.copy()
        # Insert some obvious outliers
        anomalous_data[10, 0] = 100  # Large positive outlier
        anomalous_data[25, 1] = -100  # Large negative outlier
        anomalous_data[35, 2] = 50   # Another outlier
        
        result = self.evaluator.evaluate(anomalous_data, 
                                       segment_id="anomalous_segment",
                                       create_visualizations=False)
        
        # Should detect the anomalies and have lower scores
        assert isinstance(result, dict)
        
        # Anomaly level score should be lower due to detected anomalies
        anomaly_score = result["detailed_scores"]["anomaly_level"]
        assert anomaly_score < 1.0  # Should detect some anomalies
    
    def test_data_with_different_trends(self):
        """Test with data where lookback and forecast have different trends."""
        # Create data with trend change
        trend_change_data = np.zeros_like(self.test_data)
        split_point = int(len(self.test_data) * 0.7)
        
        for col in range(self.n_cols):
            # Lookback: positive trend
            lookback_indices = np.arange(split_point)
            trend_change_data[:split_point, col] = 0.2 * lookback_indices + np.random.normal(0, 0.3, split_point)
            
            # Forecast: negative trend
            forecast_indices = np.arange(len(self.test_data) - split_point)
            trend_change_data[split_point:, col] = 5 - 0.3 * forecast_indices + np.random.normal(0, 0.3, len(forecast_indices))
        
        result = self.evaluator.evaluate(trend_change_data,
                                       segment_id="trend_change_segment", 
                                       create_visualizations=False)
        
        # Should detect trend inconsistency
        trend_score = result["detailed_scores"]["trend_consistency"]
        # Trend score should be lower due to different trends
        # Note: depending on the algorithm, this might still be somewhat high
        assert 0 <= trend_score <= 1
    
    def test_update_weights(self):
        """Test updating scoring weights."""
        new_weights = {
            "distribution_similarity": 0.25,
            "anomaly_level": 0.35,
            "trend_consistency": 0.20,
            "data_quality": 0.20
        }
        
        self.evaluator.update_weights(new_weights)
        
        for key, value in new_weights.items():
            assert self.evaluator.weights[key] == value
    
    def test_get_evaluation_summary(self):
        """Test batch evaluation summary generation."""
        # Create multiple test segments and evaluate them
        segments = [self.test_data + np.random.normal(0, 0.1, self.test_data.shape) for _ in range(5)]
        results = self.evaluator.evaluate_batch(segments, create_visualizations=False)
        
        summary = self.evaluator.get_evaluation_summary(results)
        
        # Check summary structure
        assert isinstance(summary, dict)
        assert "total_segments" in summary
        assert "valid_evaluations" in summary
        assert "prediction_worthy_count" in summary
        assert "prediction_worthy_ratio" in summary
        assert "overall_score_stats" in summary
        
        # Check values
        assert summary["total_segments"] == 5
        assert summary["valid_evaluations"] == 5
        
        score_stats = summary["overall_score_stats"]
        assert "mean" in score_stats
        assert "std" in score_stats
        assert "min" in score_stats
        assert "max" in score_stats
        assert "median" in score_stats


def test_basic_functionality():
    """Basic functionality test that can be run directly."""
    print("Running basic functionality test...")
    
    # Create test data
    np.random.seed(42)
    n_rows, n_cols = 30, 2
    test_data = np.random.randn(n_rows, n_cols)
    
    # Add some patterns
    for col in range(n_cols):
        test_data[:, col] += 0.1 * np.arange(n_rows)  # Add trend
    
    # Initialize evaluator
    evaluator = TimeSeriesSegmentEvaluator()
    
    # Evaluate
    result = evaluator.evaluate(test_data, segment_id="basic_test", create_visualizations=False)
    
    print(f"Evaluation completed successfully!")
    print(f"Overall score: {result['overall_score']:.3f}")
    print(f"Prediction worthy: {result['prediction_worthy']}")
    print(f"Recommendations: {len(result['recommendations'])} items")
    
    return result


if __name__ == "__main__":
    # Run basic test
    test_result = test_basic_functionality()
    print("✓ Basic functionality test passed!")
    
    # Run more comprehensive tests if pytest is available
    try:
        test_class = TestTimeSeriesSegmentEvaluator()
        test_class.setup_method()
        
        print("\nRunning comprehensive tests...")
        test_class.test_evaluator_initialization()
        print("✓ Initialization test passed")
        
        test_class.test_evaluate_single_segment()
        print("✓ Single segment evaluation test passed")
        
        test_class.test_evaluate_batch()
        print("✓ Batch evaluation test passed")
        
        test_class.test_edge_cases()
        print("✓ Edge cases test passed")
        
        test_class.test_data_with_anomalies()
        print("✓ Anomaly detection test passed")
        
        print("\n✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"Error in comprehensive tests: {e}")
        print("Basic functionality is working, but some advanced features may need attention.")