"""
Visualization module for creating evaluation plots and charts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime, timedelta


class VisualizationHandler:
    """
    Creates comprehensive visualizations for time series evaluation results.
    """
    
    def __init__(self, output_dir: str = "./visualizations"):
        """
        Initialize the VisualizationHandler.
        
        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def create_evaluation_plots(self, 
                              original_data: np.ndarray,
                              preprocessed_data: Dict,
                              similarity_results: Dict,
                              anomaly_results: Dict,
                              pattern_results: Dict,
                              segment_id: str) -> List[str]:
        """
        Create comprehensive evaluation plots for a time series segment.
        
        Returns:
            List of paths to created visualization files
        """
        visualization_paths = []
        
        # 1. Overview plot
        overview_path = self._create_overview_plot(
            original_data, preprocessed_data, anomaly_results, segment_id
        )
        visualization_paths.append(overview_path)
        
        # 2. Distribution comparison plot
        distribution_path = self._create_distribution_plot(
            preprocessed_data, similarity_results, segment_id
        )
        visualization_paths.append(distribution_path)
        
        # 3. Anomaly detection plot
        anomaly_path = self._create_anomaly_plot(
            original_data, anomaly_results, segment_id
        )
        visualization_paths.append(anomaly_path)
        
        # 4. Pattern analysis plot
        pattern_path = self._create_pattern_plot(
            preprocessed_data, pattern_results, segment_id
        )
        visualization_paths.append(pattern_path)
        
        # 5. Summary dashboard
        dashboard_path = self._create_evaluation_dashboard(
            similarity_results, anomaly_results, pattern_results, segment_id
        )
        visualization_paths.append(dashboard_path)
        
        return visualization_paths
    
    def _create_overview_plot(self, 
                            original_data: np.ndarray,
                            preprocessed_data: Dict,
                            anomaly_results: Dict,
                            segment_id: str) -> str:
        """Create overview plot showing original data, split, and anomalies."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Time Series Overview - {segment_id}', fontsize=16, fontweight='bold')
        
        n_features = min(4, original_data.shape[1])  # Show up to 4 features
        split_index = preprocessed_data['split_index']
        
        for i in range(n_features):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Plot original data
            time_indices = np.arange(len(original_data))
            ax.plot(time_indices, original_data[:, i], 'b-', alpha=0.7, linewidth=1.5, 
                   label=f'Feature {i+1}')
            
            # Mark split point
            ax.axvline(x=split_index, color='red', linestyle='--', linewidth=2, 
                      label='Lookback/Forecast Split')
            
            # Highlight anomalies
            anomaly_indices = anomaly_results.get('anomaly_indices', [])
            if anomaly_indices:
                anomaly_values = original_data[anomaly_indices, i]
                ax.scatter(anomaly_indices, anomaly_values, color='red', s=50, 
                          alpha=0.8, zorder=5, label='Anomalies')
            
            # Shade regions
            ax.axvspan(0, split_index, alpha=0.2, color='blue', label='Lookback')
            ax.axvspan(split_index, len(original_data), alpha=0.2, color='green', 
                      label='Forecast')
            
            ax.set_title(f'Feature {i+1}')
            ax.set_xlabel('Time Index')
            ax.set_ylabel('Value')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, 4):
            row, col = i // 2, i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{segment_id}_overview.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_distribution_plot(self, 
                                preprocessed_data: Dict,
                                similarity_results: Dict,
                                segment_id: str) -> str:
        """Create distribution comparison plots."""
        lookback = preprocessed_data['lookback']
        forecast = preprocessed_data['forecast']
        n_features = lookback.shape[1]
        
        # Create subplot grid
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Distribution Similarity Analysis - {segment_id}', 
                    fontsize=16, fontweight='bold')
        
        for i in range(n_features):
            ax = axes[i] if n_features > 1 else axes[0]
            
            # Data for this feature
            lookback_col = lookback[:, i]
            forecast_col = forecast[:, i]
            
            # Create histograms
            bins = min(20, max(5, len(lookback_col) // 3))
            
            ax.hist(lookback_col, bins=bins, alpha=0.6, color='blue', 
                   label='Lookback', density=True)
            ax.hist(forecast_col, bins=bins, alpha=0.6, color='green', 
                   label='Forecast', density=True)
            
            # Add distribution statistics
            lb_mean, lb_std = np.mean(lookback_col), np.std(lookback_col)
            fc_mean, fc_std = np.mean(forecast_col), np.std(forecast_col)
            
            # Get similarity scores if available
            feature_sim_key = f'feature_{i}_similarity'
            if feature_sim_key in similarity_results:
                feature_sim = similarity_results[feature_sim_key]
                mean_diff = feature_sim.get('mean_rel_diff', 0)
                std_diff = feature_sim.get('std_rel_diff', 0)
            else:
                mean_diff = abs(fc_mean - lb_mean) / (abs(lb_mean) + 1e-8)
                std_diff = abs(fc_std - lb_std) / (abs(lb_std) + 1e-8)
            
            # Add text with statistics
            stats_text = f'Lookback: μ={lb_mean:.3f}, σ={lb_std:.3f}\n'
            stats_text += f'Forecast: μ={fc_mean:.3f}, σ={fc_std:.3f}\n'
            stats_text += f'Mean diff: {mean_diff:.3f}\n'
            stats_text += f'Std diff: {std_diff:.3f}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'Feature {i+1} Distribution')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{segment_id}_distributions.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_anomaly_plot(self, 
                           original_data: np.ndarray,
                           anomaly_results: Dict,
                           segment_id: str) -> str:
        """Create comprehensive anomaly detection visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Anomaly Detection Analysis - {segment_id}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Time series with all anomalies
        ax1 = axes[0, 0]
        time_indices = np.arange(len(original_data))
        
        # Plot first feature as example
        feature_data = original_data[:, 0]
        ax1.plot(time_indices, feature_data, 'b-', alpha=0.7, linewidth=1, 
                label='Original Data')
        
        # Mark different types of anomalies
        anomaly_indices = anomaly_results.get('anomaly_indices', [])
        if anomaly_indices:
            ax1.scatter(anomaly_indices, feature_data[anomaly_indices], 
                       color='red', s=50, alpha=0.8, label='All Anomalies')
        
        high_conf_indices = anomaly_results.get('high_confidence_anomalies', [])
        if high_conf_indices:
            ax1.scatter(high_conf_indices, feature_data[high_conf_indices], 
                       color='darkred', s=80, marker='X', label='High Confidence')
        
        ax1.set_title('Anomalies in Time Series (Feature 1)')
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Anomaly method comparison
        ax2 = axes[0, 1]
        
        methods = ['Z-Score', 'IQR', 'Isolation Forest', 'LOF', 'Seasonal']
        ratios = [
            anomaly_results.get('zscore_anomaly_ratio', 0),
            anomaly_results.get('iqr_anomaly_ratio', 0),
            anomaly_results.get('isolation_forest_anomaly_ratio', 0),
            anomaly_results.get('lof_anomaly_ratio', 0),
            anomaly_results.get('seasonal_anomaly_ratio', 0)
        ]
        
        bars = ax2.bar(methods, ratios, color=['blue', 'green', 'orange', 'red', 'purple'])
        ax2.set_title('Anomaly Detection by Method')
        ax2.set_ylabel('Anomaly Ratio')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{ratio:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Isolation Forest scores
        ax3 = axes[1, 0]
        
        iso_scores = anomaly_results.get('isolation_forest_scores', [])
        if len(iso_scores) > 0:
            ax3.plot(time_indices, iso_scores, 'g-', alpha=0.7, linewidth=1)
            ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, 
                       label='Decision Boundary')
            ax3.fill_between(time_indices, iso_scores, 0, where=(np.array(iso_scores) < 0),
                           alpha=0.3, color='red', label='Anomalous')
        
        ax3.set_title('Isolation Forest Anomaly Scores')
        ax3.set_xlabel('Time Index')
        ax3.set_ylabel('Anomaly Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Anomaly statistics
        ax4 = axes[1, 1]
        
        # Create a summary table
        stats_data = [
            ['Total Anomalies', len(anomaly_indices)],
            ['High Confidence', len(high_conf_indices)],
            ['Overall Ratio', f"{anomaly_results.get('overall_anomaly_ratio', 0):.3f}"],
            ['Change Points', anomaly_results.get('change_point_count', 0)],
            ['Sudden Changes', anomaly_results.get('sudden_change_count', 0)]
        ]
        
        # Create table
        table = ax4.table(cellText=stats_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax4.set_title('Anomaly Detection Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{segment_id}_anomalies.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_pattern_plot(self, 
                           preprocessed_data: Dict,
                           pattern_results: Dict,
                           segment_id: str) -> str:
        """Create pattern analysis visualization."""
        lookback = preprocessed_data['lookback']
        forecast = preprocessed_data['forecast']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Pattern Analysis - {segment_id}', fontsize=16, fontweight='bold')
        
        # Plot 1: Trend comparison
        ax1 = axes[0, 0]
        
        # Plot trends for first feature
        feature_idx = 0
        if lookback.shape[1] > feature_idx:
            lookback_col = lookback[:, feature_idx]
            forecast_col = forecast[:, feature_idx]
            
            # Create time indices
            lb_time = np.arange(len(lookback_col))
            fc_time = np.arange(len(lookback_col), len(lookback_col) + len(forecast_col))
            
            ax1.plot(lb_time, lookback_col, 'b-', linewidth=2, label='Lookback')
            ax1.plot(fc_time, forecast_col, 'g-', linewidth=2, label='Forecast')
            
            # Add trend lines
            lb_trends = pattern_results.get('lookback_trends', [{}])
            fc_trends = pattern_results.get('forecast_trends', [{}])
            
            if len(lb_trends) > feature_idx and len(fc_trends) > feature_idx:
                lb_trend = lb_trends[feature_idx]
                fc_trend = fc_trends[feature_idx]
                
                # Calculate trend lines
                lb_trend_line = lb_trend.get('slope', 0) * lb_time + lb_trend.get('intercept', 0)
                fc_trend_line = fc_trend.get('slope', 0) * (fc_time - len(lookback_col)) + fc_trend.get('intercept', 0)
                
                ax1.plot(lb_time, lb_trend_line, 'r--', alpha=0.7, label='LB Trend')
                ax1.plot(fc_time, fc_trend_line, 'm--', alpha=0.7, label='FC Trend')
        
        ax1.axvline(x=len(lookback_col), color='red', linestyle=':', alpha=0.5)
        ax1.set_title('Trend Analysis (Feature 1)')
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Volatility comparison
        ax2 = axes[0, 1]
        
        volatilities_lb = pattern_results.get('lookback_volatilities', [{}])
        volatilities_fc = pattern_results.get('forecast_volatilities', [{}])
        
        vol_metrics = ['std', 'rolling_std_mean', 'returns_std']
        vol_data = []
        
        for metric in vol_metrics:
            lb_val = volatilities_lb[0].get(metric, 0) if volatilities_lb else 0
            fc_val = volatilities_fc[0].get(metric, 0) if volatilities_fc else 0
            vol_data.append([lb_val, fc_val])
        
        vol_data = np.array(vol_data)
        x = np.arange(len(vol_metrics))
        width = 0.35
        
        ax2.bar(x - width/2, vol_data[:, 0], width, label='Lookback', alpha=0.7)
        ax2.bar(x + width/2, vol_data[:, 1], width, label='Forecast', alpha=0.7)
        
        ax2.set_title('Volatility Comparison')
        ax2.set_xlabel('Volatility Metric')
        ax2.set_ylabel('Value')
        ax2.set_xticks(x)
        ax2.set_xticklabels(vol_metrics, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Periodicity analysis (FFT)
        ax3 = axes[0, 2]
        
        lb_periodicity = pattern_results.get('lookback_periodicity', [{}])
        fc_periodicity = pattern_results.get('forecast_periodicity', [{}])
        
        if lb_periodicity and 'power_spectrum' in lb_periodicity[0]:
            freqs = lb_periodicity[0].get('frequencies', [])
            power_lb = lb_periodicity[0].get('power_spectrum', [])
            
            if len(freqs) > 0 and len(power_lb) > 0:
                ax3.plot(freqs[:min(len(freqs), 20)], power_lb[:min(len(power_lb), 20)], 
                        'b-', alpha=0.7, label='Lookback')
        
        if fc_periodicity and 'power_spectrum' in fc_periodicity[0]:
            freqs = fc_periodicity[0].get('frequencies', [])
            power_fc = fc_periodicity[0].get('power_spectrum', [])
            
            if len(freqs) > 0 and len(power_fc) > 0:
                ax3.plot(freqs[:min(len(freqs), 20)], power_fc[:min(len(power_fc), 20)], 
                        'g-', alpha=0.7, label='Forecast')
        
        ax3.set_title('Frequency Analysis')
        ax3.set_xlabel('Frequency')
        ax3.set_ylabel('Power')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Autocorrelation
        ax4 = axes[1, 0]
        
        lb_autocorr = pattern_results.get('lookback_autocorrelations', [{}])
        fc_autocorr = pattern_results.get('forecast_autocorrelations', [{}])
        
        if lb_autocorr and 'autocorrelations' in lb_autocorr[0]:
            autocorrs_lb = lb_autocorr[0]['autocorrelations']
            lags = range(1, len(autocorrs_lb) + 1)
            ax4.plot(lags, autocorrs_lb, 'bo-', alpha=0.7, label='Lookback')
        
        if fc_autocorr and 'autocorrelations' in fc_autocorr[0]:
            autocorrs_fc = fc_autocorr[0]['autocorrelations']
            lags = range(1, len(autocorrs_fc) + 1)
            ax4.plot(lags, autocorrs_fc, 'go-', alpha=0.7, label='Forecast')
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Significance')
        ax4.axhline(y=-0.3, color='red', linestyle='--', alpha=0.5)
        
        ax4.set_title('Autocorrelation Function')
        ax4.set_xlabel('Lag')
        ax4.set_ylabel('Correlation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Pattern consistency scores
        ax5 = axes[1, 1]
        
        score_names = ['Trend', 'Volatility', 'Periodicity', 'Autocorr']
        scores = [
            pattern_results.get('trend_consistency_score', 0),
            pattern_results.get('volatility_consistency_score', 0),
            pattern_results.get('periodicity_consistency_score', 0),
            pattern_results.get('autocorrelation_consistency_score', 0)
        ]
        
        colors = ['red' if s < 0.5 else 'orange' if s < 0.7 else 'green' for s in scores]
        bars = ax5.bar(score_names, scores, color=colors, alpha=0.7)
        
        # Add score labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        ax5.set_title('Pattern Consistency Scores')
        ax5.set_ylabel('Consistency Score')
        ax5.set_ylim(0, 1.1)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Overall summary
        ax6 = axes[1, 2]
        
        # Create radar chart of all scores
        categories = ['Trend\nConsistency', 'Volatility\nConsistency', 
                     'Periodicity\nConsistency', 'Autocorr\nConsistency']
        
        # Close the radar chart
        scores_radar = scores + [scores[0]]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        ax6 = plt.subplot(2, 3, 6, projection='polar')
        ax6.plot(angles, scores_radar, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax6.fill(angles, scores_radar, alpha=0.25, color='blue')
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categories)
        ax6.set_ylim(0, 1)
        ax6.set_title('Pattern Analysis Summary', y=1.08)
        ax6.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{segment_id}_patterns.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_evaluation_dashboard(self, 
                                   similarity_results: Dict,
                                   anomaly_results: Dict,
                                   pattern_results: Dict,
                                   segment_id: str) -> str:
        """Create a comprehensive evaluation dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Evaluation Dashboard - {segment_id}', fontsize=18, fontweight='bold')
        
        # Plot 1: Overall scores pie chart
        ax1 = axes[0, 0]
        
        # Calculate main component scores
        similarity_score = similarity_results.get('overall_similarity_score', 0.5)
        anomaly_score = 1 - anomaly_results.get('overall_anomaly_ratio', 0.5)  # Invert for scoring
        pattern_score = pattern_results.get('overall_pattern_consistency_score', 0.5)
        
        scores = [similarity_score, anomaly_score, pattern_score]
        labels = ['Distribution\nSimilarity', 'Anomaly\nLevel', 'Pattern\nConsistency']
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax1.pie(scores, labels=labels, colors=colors, 
                                          autopct='%1.3f', startangle=90)
        ax1.set_title('Component Scores')
        
        # Plot 2: Detailed similarity metrics
        ax2 = axes[0, 1]
        
        sim_metrics = ['Statistical', 'KS Test', 'Wasserstein', 'Overall']
        sim_scores = [
            similarity_results.get('statistical_similarity_score', 0),
            similarity_results.get('ks_test_score', 0),
            similarity_results.get('wasserstein_score', 0),
            similarity_results.get('overall_similarity_score', 0)
        ]
        
        colors = ['red' if s < 0.5 else 'orange' if s < 0.7 else 'green' for s in sim_scores]
        bars = ax2.barh(sim_metrics, sim_scores, color=colors, alpha=0.7)
        
        for bar, score in zip(bars, sim_scores):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', ha='left', va='center')
        
        ax2.set_title('Similarity Analysis')
        ax2.set_xlabel('Score')
        ax2.set_xlim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Anomaly detection summary
        ax3 = axes[0, 2]
        
        anomaly_methods = ['Z-Score', 'IQR', 'Isolation\nForest', 'LOF', 'Overall']
        anomaly_ratios = [
            anomaly_results.get('zscore_anomaly_ratio', 0),
            anomaly_results.get('iqr_anomaly_ratio', 0),
            anomaly_results.get('isolation_forest_anomaly_ratio', 0),
            anomaly_results.get('lof_anomaly_ratio', 0),
            anomaly_results.get('overall_anomaly_ratio', 0)
        ]
        
        colors = ['green' if r < 0.1 else 'orange' if r < 0.2 else 'red' for r in anomaly_ratios]
        bars = ax3.bar(anomaly_methods, anomaly_ratios, color=colors, alpha=0.7)
        
        for bar, ratio in zip(bars, anomaly_ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{ratio:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax3.set_title('Anomaly Detection')
        ax3.set_ylabel('Anomaly Ratio')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Pattern analysis summary
        ax4 = axes[1, 0]
        
        pattern_metrics = ['Trend', 'Volatility', 'Periodicity', 'Autocorr', 'Overall']
        pattern_scores = [
            pattern_results.get('trend_consistency_score', 0),
            pattern_results.get('volatility_consistency_score', 0),
            pattern_results.get('periodicity_consistency_score', 0),
            pattern_results.get('autocorrelation_consistency_score', 0),
            pattern_results.get('overall_pattern_consistency_score', 0)
        ]
        
        colors = ['red' if s < 0.5 else 'orange' if s < 0.7 else 'green' for s in pattern_scores]
        bars = ax4.barh(pattern_metrics, pattern_scores, color=colors, alpha=0.7)
        
        for bar, score in zip(bars, pattern_scores):
            width = bar.get_width()
            ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', ha='left', va='center')
        
        ax4.set_title('Pattern Consistency')
        ax4.set_xlabel('Score')
        ax4.set_xlim(0, 1.1)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Final recommendation
        ax5 = axes[1, 1]
        
        # Calculate overall weighted score (mimicking main evaluator)
        weights = {'similarity': 0.30, 'anomaly': 0.25, 'pattern': 0.25, 'quality': 0.20}
        
        overall_score = (
            weights['similarity'] * similarity_score +
            weights['anomaly'] * anomaly_score +
            weights['pattern'] * pattern_score +
            weights['quality'] * 0.8  # Assume decent quality
        )
        
        # Create gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # Color sections
        ax5.fill_between(theta[0:20], 0, r, color='red', alpha=0.3, label='Poor (0-0.4)')
        ax5.fill_between(theta[20:60], 0, r, color='orange', alpha=0.3, label='Fair (0.4-0.7)')
        ax5.fill_between(theta[60:100], 0, r, color='green', alpha=0.3, label='Good (0.7-1.0)')
        
        # Add score indicator
        score_angle = overall_score * np.pi
        ax5.arrow(0, 0, np.cos(np.pi - score_angle), np.sin(np.pi - score_angle), 
                 head_width=0.05, head_length=0.1, fc='black', ec='black', linewidth=3)
        
        # Add score text
        ax5.text(0, -0.3, f'Overall Score\n{overall_score:.3f}', 
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Recommendation
        if overall_score >= 0.7:
            recommendation = "RECOMMENDED\nfor prediction"
            rec_color = 'green'
        elif overall_score >= 0.4:
            recommendation = "CAUTION\nrequires review"
            rec_color = 'orange'
        else:
            recommendation = "NOT RECOMMENDED\nfor prediction"
            rec_color = 'red'
        
        ax5.text(0, -0.6, recommendation, ha='center', va='center', 
                fontsize=12, fontweight='bold', color=rec_color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax5.set_xlim(-1.2, 1.2)
        ax5.set_ylim(-0.8, 1.2)
        ax5.set_aspect('equal')
        ax5.axis('off')
        ax5.set_title('Final Recommendation')
        
        # Plot 6: Key statistics table
        ax6 = axes[1, 2]
        
        stats_data = [
            ['Overall Score', f'{overall_score:.3f}'],
            ['Similarity Score', f'{similarity_score:.3f}'],
            ['Anomaly Level', f'{1-anomaly_score:.3f}'],
            ['Pattern Score', f'{pattern_score:.3f}'],
            ['Anomalies Found', str(len(anomaly_results.get('anomaly_indices', [])))],
            ['High Confidence', str(len(anomaly_results.get('high_confidence_anomalies', [])))],
            ['Change Points', str(anomaly_results.get('change_point_count', 0))],
            ['Recommendation', recommendation.replace('\n', ' ')]
        ]
        
        # Create table
        table = ax6.table(cellText=stats_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Color code recommendation row
        table[(len(stats_data), 1)].set_facecolor(rec_color)
        table[(len(stats_data), 1)].set_text_props(weight='bold', color='white')
        
        ax6.set_title('Summary Statistics')
        ax6.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{segment_id}_dashboard.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath