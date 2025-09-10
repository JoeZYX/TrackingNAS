"""
Main TimeSeriesSegmentEvaluator class for evaluating prediction worthiness of time series segments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from .data_preprocessing import DataPreprocessor
    from .distribution_similarity import DistributionSimilarityAnalyzer
    from .anomaly_detection import AnomalyDetector
    from .pattern_analysis import PatternAnalyzer
    from .visualization import VisualizationHandler
except ImportError:
    # When running as script, use absolute imports
    from data_preprocessing import DataPreprocessor
    from distribution_similarity import DistributionSimilarityAnalyzer
    from anomaly_detection import AnomalyDetector
    from pattern_analysis import PatternAnalyzer
    from visualization import VisualizationHandler


class TimeSeriesSegmentEvaluator:
    """
    Comprehensive evaluator for time series segment prediction worthiness.
    
    This class integrates multiple evaluation modules to assess whether a time series
    segment is suitable for prediction model training based on distribution similarity,
    anomaly detection, pattern analysis, and data quality.
    """
    
    def __init__(self, 
                 lookback_ratio: float = 0.7,
                 anomaly_threshold: float = 0.1,
                 similarity_threshold: float = 0.6,
                 trend_threshold: float = 0.7,
                 visualization_dir: str = "./visualizations"):
        """
        Initialize the TimeSeriesSegmentEvaluator.
        
        Args:
            lookback_ratio: Ratio of data to use as lookback window (default: 0.7)
            anomaly_threshold: Threshold for anomaly detection (default: 0.1)
            similarity_threshold: Threshold for distribution similarity (default: 0.6)
            trend_threshold: Threshold for trend consistency (default: 0.7)
            visualization_dir: Directory to save visualizations (default: "./visualizations")
        """
        self.lookback_ratio = lookback_ratio
        self.anomaly_threshold = anomaly_threshold
        self.similarity_threshold = similarity_threshold
        self.trend_threshold = trend_threshold
        self.visualization_dir = visualization_dir
        
        # Initialize sub-modules
        self.preprocessor = DataPreprocessor()
        self.similarity_analyzer = DistributionSimilarityAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.pattern_analyzer = PatternAnalyzer()
        self.visualizer = VisualizationHandler(visualization_dir)
        
        # Scoring weights
        self.weights = {
            "distribution_similarity": 0.30,
            "anomaly_level": 0.25,
            "trend_consistency": 0.25,
            "data_quality": 0.20
        }
    
    def evaluate(self, 
                 data_segment: np.ndarray, 
                 segment_id: Optional[str] = None,
                 create_visualizations: bool = True) -> Dict:
        """
        Evaluate a time series segment for prediction worthiness.
        
        Args:
            data_segment: Input data matrix (n rows, m columns)
            segment_id: Optional identifier for the segment
            create_visualizations: Whether to create visualization plots
            
        Returns:
            Dictionary containing evaluation results and scores
        """
        if segment_id is None:
            segment_id = f"segment_{np.random.randint(1000, 9999)}"
            
        # Step 1: Data preprocessing
        preprocessed_data = self.preprocessor.preprocess(data_segment, self.lookback_ratio)
        
        # Step 2: Distribution similarity evaluation
        similarity_results = self.similarity_analyzer.analyze(
            preprocessed_data['lookback'], 
            preprocessed_data['forecast']
        )
        
        # Step 3: Anomaly detection
        anomaly_results = self.anomaly_detector.detect(
            data_segment, 
            preprocessed_data['lookback'], 
            preprocessed_data['forecast']
        )
        
        # Step 4: Pattern analysis
        pattern_results = self.pattern_analyzer.analyze(
            preprocessed_data['lookback'], 
            preprocessed_data['forecast']
        )
        
        # Step 5: Calculate individual scores
        scores = self._calculate_scores(similarity_results, anomaly_results, pattern_results, preprocessed_data)
        
        # Step 6: Calculate overall score
        overall_score = self._calculate_overall_score(scores)
        
        # Step 7: Generate recommendations
        recommendations = self._generate_recommendations(scores, similarity_results, anomaly_results, pattern_results)
        
        # Step 8: Create visualizations if requested
        visualization_paths = []
        if create_visualizations:
            visualization_paths = self.visualizer.create_evaluation_plots(
                data_segment, 
                preprocessed_data, 
                similarity_results, 
                anomaly_results, 
                pattern_results, 
                segment_id
            )
        
        # Step 9: Compile results
        results = {
            "segment_id": segment_id,
            "overall_score": round(overall_score, 3),
            "prediction_worthy": overall_score >= 0.6,  # Configurable threshold
            "detailed_scores": {
                "distribution_similarity": round(scores["distribution_similarity"], 3),
                "anomaly_level": round(scores["anomaly_level"], 3),
                "trend_consistency": round(scores["trend_consistency"], 3),
                "data_quality": round(scores["data_quality"], 3)
            },
            "recommendations": recommendations,
            "visualization_paths": visualization_paths,
            "metadata": {
                "lookback_ratio": self.lookback_ratio,
                "lookback_shape": preprocessed_data['lookback'].shape,
                "forecast_shape": preprocessed_data['forecast'].shape,
                "original_shape": data_segment.shape
            }
        }
        
        return results
    
    def evaluate_batch(self, 
                      data_segments: List[np.ndarray], 
                      segment_ids: Optional[List[str]] = None,
                      create_visualizations: bool = False) -> List[Dict]:
        """
        Evaluate multiple time series segments in batch.
        
        Args:
            data_segments: List of data matrices to evaluate
            segment_ids: Optional list of identifiers for segments
            create_visualizations: Whether to create visualization plots
            
        Returns:
            List of evaluation result dictionaries
        """
        if segment_ids is None:
            segment_ids = [f"segment_{i:04d}" for i in range(len(data_segments))]
        
        results = []
        for i, data_segment in enumerate(data_segments):
            try:
                result = self.evaluate(data_segment, segment_ids[i], create_visualizations)
                results.append(result)
            except Exception as e:
                # Handle individual segment failures gracefully
                results.append({
                    "segment_id": segment_ids[i],
                    "error": str(e),
                    "prediction_worthy": False,
                    "overall_score": 0.0
                })
        
        return results
    
    def _calculate_scores(self, similarity_results: Dict, anomaly_results: Dict, 
                         pattern_results: Dict, preprocessed_data: Dict) -> Dict:
        """Calculate individual component scores."""
        scores = {}
        
        # Distribution similarity score (higher is better)
        similarity_score = np.mean([
            similarity_results['ks_test_score'],
            similarity_results['wasserstein_score'],
            similarity_results['statistical_similarity_score']
        ])
        scores["distribution_similarity"] = max(0.0, min(1.0, similarity_score))
        
        # Anomaly level score (lower anomaly ratio is better)
        anomaly_ratio = anomaly_results['overall_anomaly_ratio']
        scores["anomaly_level"] = max(0.0, min(1.0, 1.0 - anomaly_ratio))
        
        # Trend consistency score (higher is better)
        trend_score = np.mean([
            pattern_results['trend_consistency_score'],
            pattern_results['volatility_consistency_score']
        ])
        scores["trend_consistency"] = max(0.0, min(1.0, trend_score))
        
        # Data quality score
        missing_ratio = preprocessed_data.get('missing_ratio', 0.0)
        normalization_quality = preprocessed_data.get('normalization_quality', 1.0)
        scores["data_quality"] = max(0.0, min(1.0, (1.0 - missing_ratio) * normalization_quality))
        
        return scores
    
    def _calculate_overall_score(self, scores: Dict) -> float:
        """Calculate weighted overall score."""
        overall_score = sum(scores[key] * self.weights[key] for key in self.weights.keys())
        return max(0.0, min(1.0, overall_score))
    
    def _generate_recommendations(self, scores: Dict, similarity_results: Dict, 
                                anomaly_results: Dict, pattern_results: Dict) -> List[str]:
        """Generate actionable recommendations based on evaluation results."""
        recommendations = []
        
        # Overall assessment
        overall_score = sum(scores[key] * self.weights[key] for key in self.weights.keys())
        if overall_score >= 0.8:
            recommendations.append("优秀片段：强烈建议用于预测模型训练")
        elif overall_score >= 0.6:
            recommendations.append("良好片段：建议用于预测模型训练")
        elif overall_score >= 0.4:
            recommendations.append("中等片段：谨慎使用，可能需要额外处理")
        else:
            recommendations.append("不建议使用：该片段可能不适合预测模型训练")
        
        # Specific recommendations based on individual scores
        if scores["distribution_similarity"] < self.similarity_threshold:
            recommendations.append("分布相似性较低，预测段与回望窗口特征差异较大")
        
        if scores["anomaly_level"] < 0.8:
            anomaly_indices = anomaly_results.get('anomaly_indices', [])
            if len(anomaly_indices) > 0:
                recommendations.append(f"检测到异常值，位于索引: {anomaly_indices[:5]}")  # Show first 5
        
        if scores["trend_consistency"] < self.trend_threshold:
            recommendations.append("趋势一致性不足，可能存在结构性变化")
        
        if scores["data_quality"] < 0.9:
            recommendations.append("数据质量有待改善，建议检查缺失值和异常值")
        
        return recommendations
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update scoring weights."""
        if abs(sum(new_weights.values()) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        for key in new_weights:
            if key not in self.weights:
                raise ValueError(f"Invalid weight key: {key}")
        
        self.weights.update(new_weights)
    
    def get_evaluation_summary(self, results_list: List[Dict]) -> Dict:
        """Generate summary statistics for a batch of evaluation results."""
        if not results_list:
            return {}
        
        valid_results = [r for r in results_list if 'error' not in r]
        if not valid_results:
            return {"error": "No valid results found"}
        
        overall_scores = [r['overall_score'] for r in valid_results]
        prediction_worthy_count = sum(1 for r in valid_results if r['prediction_worthy'])
        
        summary = {
            "total_segments": len(results_list),
            "valid_evaluations": len(valid_results),
            "prediction_worthy_count": prediction_worthy_count,
            "prediction_worthy_ratio": prediction_worthy_count / len(valid_results),
            "overall_score_stats": {
                "mean": np.mean(overall_scores),
                "std": np.std(overall_scores),
                "min": np.min(overall_scores),
                "max": np.max(overall_scores),
                "median": np.median(overall_scores)
            }
        }
        
        return summary