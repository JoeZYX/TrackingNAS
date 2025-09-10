"""
Integration example showing how to use the TimeSeriesSegmentEvaluator 
with the TrackingNAS project for trajectory prediction quality assessment.

This example demonstrates how time series evaluation can be applied to:
1. Object tracking trajectory data
2. Motion prediction sequences
3. Feature quality assessment for training data
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the timeseries_evaluation module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'timeseries_evaluation'))

try:
    from timeseries_evaluation import TimeSeriesSegmentEvaluator
except ImportError:
    print("Error: TimeSeriesSegmentEvaluator not found. Please ensure the module is properly installed.")
    sys.exit(1)


def generate_trajectory_data():
    """
    Generate synthetic trajectory data that mimics object tracking scenarios.
    
    Returns:
        Dictionary containing different types of trajectory segments
    """
    print("Generating synthetic trajectory data for tracking scenarios...")
    
    np.random.seed(42)
    trajectory_segments = {}
    
    # 1. Smooth trajectory - good for prediction
    print("  - Creating smooth trajectory segment...")
    t = np.linspace(0, 10, 100)
    
    # Smooth curved path with consistent motion
    x = 10 * np.sin(0.5 * t) + 0.1 * np.random.randn(len(t))
    y = 5 * np.cos(0.3 * t) + 0.1 * np.random.randn(len(t))
    vx = np.gradient(x)  # Velocity in x
    vy = np.gradient(y)  # Velocity in y
    
    smooth_trajectory = np.column_stack([x, y, vx, vy])
    trajectory_segments['smooth_trajectory'] = smooth_trajectory
    
    # 2. Erratic trajectory - poor for prediction
    print("  - Creating erratic trajectory segment...")
    # Sudden direction changes and speed variations
    x_erratic = np.cumsum(np.random.randn(100) * 2)
    y_erratic = np.cumsum(np.random.randn(100) * 2)
    
    # Add sudden jumps (like occlusions/re-detections)
    jump_indices = [25, 50, 75]
    for idx in jump_indices:
        x_erratic[idx:] += np.random.randn() * 10
        y_erratic[idx:] += np.random.randn() * 10
    
    vx_erratic = np.gradient(x_erratic)
    vy_erratic = np.gradient(y_erratic)
    
    erratic_trajectory = np.column_stack([x_erratic, y_erratic, vx_erratic, vy_erratic])
    trajectory_segments['erratic_trajectory'] = erratic_trajectory
    
    # 3. Occluded trajectory - missing data
    print("  - Creating trajectory with occlusions...")
    x_occluded = x.copy()
    y_occluded = y.copy()
    vx_occluded = vx.copy()
    vy_occluded = vy.copy()
    
    # Simulate occlusion periods with NaN values
    occlusion_periods = [(20, 25), (45, 55), (80, 85)]
    for start, end in occlusion_periods:
        x_occluded[start:end] = np.nan
        y_occluded[start:end] = np.nan
        vx_occluded[start:end] = np.nan
        vy_occluded[start:end] = np.nan
    
    occluded_trajectory = np.column_stack([x_occluded, y_occluded, vx_occluded, vy_occluded])
    trajectory_segments['occluded_trajectory'] = occluded_trajectory
    
    # 4. Changing behavior trajectory
    print("  - Creating trajectory with behavior change...")
    # First half: straight line motion
    # Second half: circular motion
    
    x_changing = np.zeros(100)
    y_changing = np.zeros(100)
    
    # First 50 points: linear motion
    x_changing[:50] = np.linspace(0, 25, 50) + 0.1 * np.random.randn(50)
    y_changing[:50] = 0.2 * np.arange(50) + 0.1 * np.random.randn(50)
    
    # Last 50 points: circular motion
    t_circle = np.linspace(0, 4*np.pi, 50)
    x_changing[50:] = 25 + 5 * np.cos(t_circle) + 0.1 * np.random.randn(50)
    y_changing[50:] = 10 + 5 * np.sin(t_circle) + 0.1 * np.random.randn(50)
    
    vx_changing = np.gradient(x_changing)
    vy_changing = np.gradient(y_changing)
    
    changing_trajectory = np.column_stack([x_changing, y_changing, vx_changing, vy_changing])
    trajectory_segments['changing_behavior_trajectory'] = changing_trajectory
    
    return trajectory_segments


def evaluate_tracking_data_quality():
    """
    Demonstrate trajectory data quality evaluation for tracking applications.
    """
    print("\n" + "="*70)
    print("TRAJECTORY DATA QUALITY EVALUATION FOR TRACKING")
    print("="*70)
    
    # Generate trajectory data
    trajectories = generate_trajectory_data()
    
    # Initialize evaluator with settings appropriate for trajectory data
    evaluator = TimeSeriesSegmentEvaluator(
        lookback_ratio=0.75,  # Use more data for lookback in tracking
        anomaly_threshold=0.15,  # More tolerant of tracking noise
        similarity_threshold=0.5,  # Trajectories can change behavior
        trend_threshold=0.6,
        visualization_dir="/tmp/tracking_evaluation"
    )
    
    # Evaluate each trajectory type
    print(f"\nEvaluating {len(trajectories)} trajectory segments...")
    
    results = []
    for traj_name, traj_data in trajectories.items():
        print(f"\nEvaluating: {traj_name}")
        print(f"  Shape: {traj_data.shape}")
        
        try:
            result = evaluator.evaluate(
                traj_data, 
                segment_id=traj_name,
                create_visualizations=True
            )
            
            results.append(result)
            
            # Display key results
            print(f"  Overall Score: {result['overall_score']:.3f}")
            print(f"  Prediction Worthy: {result['prediction_worthy']}")
            print(f"  Key Issues:")
            
            detailed = result['detailed_scores']
            if detailed['distribution_similarity'] < 0.5:
                print("    - Motion pattern changes between segments")
            if detailed['anomaly_level'] < 0.8:
                print("    - High number of anomalous tracking points")
            if detailed['trend_consistency'] < 0.6:
                print("    - Inconsistent motion trends")
            if detailed['data_quality'] < 0.9:
                print("    - Data quality issues (missing values)")
            
        except Exception as e:
            print(f"  Error evaluating {traj_name}: {e}")
            results.append({
                'segment_id': traj_name,
                'error': str(e),
                'prediction_worthy': False,
                'overall_score': 0.0
            })
    
    # Generate summary
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        summary = evaluator.get_evaluation_summary(valid_results)
        
        print("\n" + "-"*50)
        print("TRAJECTORY EVALUATION SUMMARY")
        print("-"*50)
        print(f"Total trajectories evaluated: {len(results)}")
        print(f"Successfully processed: {len(valid_results)}")
        print(f"Suitable for prediction: {summary['prediction_worthy_count']}")
        print(f"Success rate: {summary['prediction_worthy_ratio']:.1%}")
        
        print(f"\nScore distribution:")
        stats = summary['overall_score_stats']
        print(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")
    
    return results


def tracking_integration_pipeline():
    """
    Demonstrate how to integrate trajectory evaluation into a tracking pipeline.
    """
    print("\n" + "="*70)
    print("TRACKING PIPELINE INTEGRATION EXAMPLE")
    print("="*70)
    
    class TrackingQualityFilter:
        """
        Example class showing how to integrate trajectory evaluation 
        into a tracking system.
        """
        
        def __init__(self, min_score_threshold=0.6):
            self.evaluator = TimeSeriesSegmentEvaluator(
                lookback_ratio=0.8,
                visualization_dir="/tmp/tracking_pipeline"
            )
            self.min_score_threshold = min_score_threshold
            self.processed_count = 0
            self.accepted_count = 0
        
        def filter_trajectory_segments(self, trajectory_segments, track_ids=None):
            """
            Filter trajectory segments based on prediction worthiness.
            
            Args:
                trajectory_segments: List of trajectory data arrays
                track_ids: Optional list of track identifiers
                
            Returns:
                Tuple of (accepted_trajectories, rejected_trajectories, results)
            """
            if track_ids is None:
                track_ids = [f"track_{i:03d}" for i in range(len(trajectory_segments))]
            
            accepted = []
            rejected = []
            all_results = []
            
            print(f"Processing {len(trajectory_segments)} trajectory segments...")
            
            for i, (traj_data, track_id) in enumerate(zip(trajectory_segments, track_ids)):
                self.processed_count += 1
                
                # Evaluate trajectory quality
                result = self.evaluator.evaluate(
                    traj_data, 
                    segment_id=track_id,
                    create_visualizations=False  # Disable for pipeline processing
                )
                
                all_results.append(result)
                
                # Decision logic
                if (result['prediction_worthy'] and 
                    result['overall_score'] >= self.min_score_threshold):
                    
                    accepted.append((traj_data, track_id, result))
                    self.accepted_count += 1
                    print(f"  ✓ {track_id}: Score {result['overall_score']:.3f} - ACCEPTED")
                else:
                    rejected.append((traj_data, track_id, result))
                    print(f"  ✗ {track_id}: Score {result['overall_score']:.3f} - REJECTED")
            
            return accepted, rejected, all_results
        
        def get_statistics(self):
            """Get processing statistics."""
            acceptance_rate = self.accepted_count / self.processed_count if self.processed_count > 0 else 0
            return {
                'processed': self.processed_count,
                'accepted': self.accepted_count,
                'acceptance_rate': acceptance_rate
            }
    
    # Generate test trajectories
    trajectories = generate_trajectory_data()
    trajectory_list = list(trajectories.values())
    trajectory_names = list(trajectories.keys())
    
    # Initialize quality filter
    quality_filter = TrackingQualityFilter(min_score_threshold=0.65)
    
    # Process trajectories
    print("\nRunning trajectory quality filtering pipeline...")
    
    accepted, rejected, results = quality_filter.filter_trajectory_segments(
        trajectory_list, trajectory_names
    )
    
    # Display results
    stats = quality_filter.get_statistics()
    
    print(f"\n" + "-"*50)
    print("PIPELINE PROCESSING RESULTS")
    print("-"*50)
    print(f"Trajectories processed: {stats['processed']}")
    print(f"Accepted for prediction: {stats['accepted']}")
    print(f"Rejection rate: {1 - stats['acceptance_rate']:.1%}")
    
    print(f"\nAccepted trajectories:")
    for traj_data, track_id, result in accepted:
        print(f"  - {track_id}: Score {result['overall_score']:.3f}")
    
    print(f"\nRejected trajectories:")
    for traj_data, track_id, result in rejected:
        main_issues = []
        detailed = result['detailed_scores']
        if detailed['distribution_similarity'] < 0.5:
            main_issues.append("pattern change")
        if detailed['anomaly_level'] < 0.8:
            main_issues.append("anomalies")
        if detailed['trend_consistency'] < 0.6:
            main_issues.append("inconsistent motion")
        
        issue_str = ", ".join(main_issues) if main_issues else "low overall score"
        print(f"  - {track_id}: Score {result['overall_score']:.3f} ({issue_str})")
    
    return accepted, rejected, results


def create_trajectory_comparison_plot():
    """Create a visualization comparing different trajectory types."""
    print("\n" + "="*70)
    print("CREATING TRAJECTORY COMPARISON VISUALIZATION")
    print("="*70)
    
    trajectories = generate_trajectory_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Trajectory Types for Tracking Evaluation', fontsize=16, fontweight='bold')
    
    trajectory_names = list(trajectories.keys())
    
    for i, (name, data) in enumerate(trajectories.items()):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Plot trajectory path
        x, y = data[:, 0], data[:, 1]
        
        # Handle NaN values for plotting
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        
        if np.any(valid_mask):
            # Plot valid points
            ax.plot(x[valid_mask], y[valid_mask], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
            ax.scatter(x[valid_mask][0], y[valid_mask][0], color='green', s=100, 
                      marker='o', label='Start', zorder=5)
            ax.scatter(x[valid_mask][-1], y[valid_mask][-1], color='red', s=100, 
                      marker='s', label='End', zorder=5)
            
            # Mark missing points
            missing_mask = ~valid_mask
            if np.any(missing_mask):
                # Find indices of missing points
                missing_indices = np.where(missing_mask)[0]
                for idx in missing_indices:
                    # Use last valid position for visualization
                    if idx > 0:
                        prev_valid = idx - 1
                        while prev_valid >= 0 and missing_mask[prev_valid]:
                            prev_valid -= 1
                        if prev_valid >= 0:
                            ax.scatter(x[prev_valid], y[prev_valid], color='orange', 
                                     s=50, marker='x', alpha=0.7)
                
                ax.scatter([], [], color='orange', marker='x', label='Occlusion', s=50)
        
        ax.set_title(name.replace('_', ' ').title())
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs("/tmp/tracking_evaluation", exist_ok=True)
    plot_path = "/tmp/tracking_evaluation/trajectory_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Trajectory comparison plot saved to: {plot_path}")
    return plot_path


def main():
    """Main integration demonstration."""
    print("TIME SERIES EVALUATION INTEGRATION WITH TRACKINGNAS")
    print("=" * 70)
    print("This example demonstrates how to integrate time series segment")
    print("evaluation with object tracking and trajectory prediction systems.")
    
    try:
        # 1. Evaluate trajectory data quality
        print("\n1. Evaluating trajectory data quality...")
        trajectory_results = evaluate_tracking_data_quality()
        
        # 2. Demonstrate pipeline integration
        print("\n2. Demonstrating pipeline integration...")
        pipeline_results = tracking_integration_pipeline()
        
        # 3. Create comparison visualization
        print("\n3. Creating trajectory comparison visualization...")
        plot_path = create_trajectory_comparison_plot()
        
        print("\n" + "="*70)
        print("INTEGRATION DEMONSTRATION COMPLETED!")
        print("="*70)
        print("✅ Trajectory evaluation: Successfully assessed tracking data quality")
        print("✅ Pipeline integration: Demonstrated automated quality filtering")
        print("✅ Visualization: Created trajectory comparison plots")
        print(f"✅ Results available in: /tmp/tracking_evaluation/")
        
        # Usage recommendations
        print(f"\n📋 INTEGRATION RECOMMENDATIONS:")
        print(f"1. Use trajectory evaluation for training data curation")
        print(f"2. Implement quality filtering in real-time tracking pipelines")
        print(f"3. Monitor tracking quality metrics for system health")
        print(f"4. Apply custom thresholds based on application requirements")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during integration demonstration: {e}")
        print("Please check the error details above.")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🚀 Integration ready!")
        print(f"You can now integrate trajectory evaluation into TrackingNAS:")
        print(f"")
        print(f"# Example integration code:")
        print(f"from src.timeseries_evaluation import TimeSeriesSegmentEvaluator")
        print(f"")
        print(f"# In your tracking pipeline:")
        print(f"evaluator = TimeSeriesSegmentEvaluator()")
        print(f"result = evaluator.evaluate(trajectory_data)")
        print(f"if result['prediction_worthy']:")
        print(f"    # Use for training/prediction")
        print(f"    process_trajectory(trajectory_data)")
    else:
        print(f"\n❌ Please resolve the issues above before integration.")