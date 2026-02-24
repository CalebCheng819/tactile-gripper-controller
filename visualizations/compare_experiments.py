#!/usr/bin/env python3
"""
Compare multiple trajectory experiments side-by-side.
Loads trajectory.h5 from multiple directories and creates comparison visualizations.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import argparse
import sys


def load_trajectory_data(h5_path):
    """Load trajectory data from HDF5 file."""
    data = {}
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Gripper data
            if 'action/gripper_position' in f:
                data['gripper_position'] = f['action/gripper_position'][:]
            elif 'observation/robot_state/gripper_position' in f:
                data['gripper_position'] = f['observation/robot_state/gripper_position'][:]
            
            if 'action/gripper_velocity' in f:
                data['gripper_velocity'] = f['action/gripper_velocity'][:]
            
            # Cartesian position
            if 'action/cartesian_position' in f:
                cart_pos = f['action/cartesian_position'][:]
                if len(cart_pos.shape) == 2 and cart_pos.shape[1] >= 3:
                    data['cartesian_x'] = cart_pos[:, 0]
                    data['cartesian_y'] = cart_pos[:, 1]
                    data['cartesian_z'] = cart_pos[:, 2]
                    data['cartesian_frames'] = cart_pos.shape[0]
            
            # Joint positions
            if 'action/joint_position' in f:
                data['joint_positions'] = f['action/joint_position'][:]
            elif 'observation/robot_state/joint_positions' in f:
                data['joint_positions'] = f['observation/robot_state/joint_positions'][:]
            
            # Target gripper (if available)
            if 'action/target_gripper_position' in f:
                data['target_gripper'] = f['action/target_gripper_position'][:]
            
            # Timestamps
            if 'observation/timestamp' in f:
                timestamp_group = f['observation/timestamp']
                if 'robot_state' in timestamp_group:
                    robot_state_ts = timestamp_group['robot_state']
                    if 'robot_timestamp_seconds' in robot_state_ts:
                        data['timestamps'] = robot_state_ts['robot_timestamp_seconds'][:]
                    elif 'read_start' in robot_state_ts:
                        data['timestamps'] = robot_state_ts['read_start'][:]
                elif 'skip_action' in timestamp_group:
                    data['timestamps'] = timestamp_group['skip_action'][:]
            
            # Force prediction (if available)
            if 'observation/robot_state/force_prediction' in f:
                force_pred = f['observation/robot_state/force_prediction'][:]
                if len(force_pred.shape) == 3:
                    data['force_prediction'] = force_pred[:, -1, 0]
                else:
                    data['force_prediction'] = force_pred
            
            # Tactile values (if available)
            if 'observation/robot_state/tactile_values' in f:
                tactile = f['observation/robot_state/tactile_values'][:]
                if len(tactile.shape) == 3:
                    data['tactile_values'] = tactile[:, -1, :]
                    data['tactile_magnitude'] = np.linalg.norm(data['tactile_values'], axis=1)
                else:
                    data['tactile_values'] = tactile
                    data['tactile_magnitude'] = np.linalg.norm(data['tactile_values'], axis=1)
            
            # Get trajectory length
            if 'gripper_position' in data:
                data['num_frames'] = len(data['gripper_position'])
            elif 'joint_positions' in data:
                data['num_frames'] = len(data['joint_positions'])
            else:
                data['num_frames'] = 0
            
    except Exception as e:
        print(f"Error loading {h5_path}: {e}")
        return None
    
    return data


def normalize_time(data_list):
    """Normalize time to [0, 1] for comparison."""
    normalized = []
    for data in data_list:
        # Use gripper_position length as primary reference
        if 'gripper_position' in data and len(data['gripper_position']) > 0:
            n = len(data['gripper_position'])
            data['time_normalized'] = np.linspace(0, 1, n)
        elif 'num_frames' in data and data['num_frames'] > 0:
            n = data['num_frames']
            data['time_normalized'] = np.linspace(0, 1, n)
        else:
            data['time_normalized'] = np.array([])
        
        # Also create normalized time for cartesian if it has different length
        if 'cartesian_frames' in data:
            n_cart = data['cartesian_frames']
            data['time_normalized_cartesian'] = np.linspace(0, 1, n_cart)
        normalized.append(data)
    return normalized


def create_comparison_plot(data_dict, output_path):
    """Create comprehensive comparison visualization."""
    experiments = list(data_dict.keys())
    n_experiments = len(experiments)
    
    # Normalize time for all experiments
    data_list = [data_dict[exp] for exp in experiments]
    data_list = normalize_time(data_list)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, n_experiments))
    
    # 1. Gripper Position
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (exp_name, data) in enumerate(zip(experiments, data_list)):
        if 'gripper_position' in data and len(data['gripper_position']) > 0:
            time = data['time_normalized']
            ax1.plot(time, data['gripper_position'], label=exp_name, 
                    color=colors[i], linewidth=2, alpha=0.8)
    ax1.set_xlabel('Normalized Time')
    ax1.set_ylabel('Gripper Position')
    ax1.set_title('Gripper Position Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Target vs Actual Gripper (if available)
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (exp_name, data) in enumerate(zip(experiments, data_list)):
        if 'gripper_position' in data and 'target_gripper' in data:
            time = data['time_normalized']
            ax2.plot(time, data['gripper_position'], label=f'{exp_name} (actual)', 
                    color=colors[i], linewidth=2, alpha=0.8, linestyle='-')
            ax2.plot(time, data['target_gripper'], label=f'{exp_name} (target)', 
                    color=colors[i], linewidth=1.5, alpha=0.5, linestyle='--')
    ax2.set_xlabel('Normalized Time')
    ax2.set_ylabel('Gripper Position')
    ax2.set_title('Target vs Actual Gripper Position')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Cartesian X Position
    ax3 = fig.add_subplot(gs[1, 0])
    for i, (exp_name, data) in enumerate(zip(experiments, data_list)):
        if 'cartesian_x' in data and len(data['cartesian_x']) > 0:
            time = data.get('time_normalized_cartesian', data['time_normalized'])
            if len(time) == len(data['cartesian_x']):
                ax3.plot(time, data['cartesian_x'], label=exp_name, 
                        color=colors[i], linewidth=2, alpha=0.8)
    ax3.set_xlabel('Normalized Time')
    ax3.set_ylabel('X Position (m)')
    ax3.set_title('Cartesian X Position')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cartesian Y Position
    ax4 = fig.add_subplot(gs[1, 1])
    for i, (exp_name, data) in enumerate(zip(experiments, data_list)):
        if 'cartesian_y' in data and len(data['cartesian_y']) > 0:
            time = data.get('time_normalized_cartesian', data['time_normalized'])
            if len(time) == len(data['cartesian_y']):
                ax4.plot(time, data['cartesian_y'], label=exp_name, 
                        color=colors[i], linewidth=2, alpha=0.8)
    ax4.set_xlabel('Normalized Time')
    ax4.set_ylabel('Y Position (m)')
    ax4.set_title('Cartesian Y Position')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Cartesian Z Position
    ax5 = fig.add_subplot(gs[2, 0])
    for i, (exp_name, data) in enumerate(zip(experiments, data_list)):
        if 'cartesian_z' in data and len(data['cartesian_z']) > 0:
            time = data.get('time_normalized_cartesian', data['time_normalized'])
            if len(time) == len(data['cartesian_z']):
                ax5.plot(time, data['cartesian_z'], label=exp_name, 
                        color=colors[i], linewidth=2, alpha=0.8)
    ax5.set_xlabel('Normalized Time')
    ax5.set_ylabel('Z Position (m)')
    ax5.set_title('Cartesian Z Position')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Force Prediction (if available)
    ax6 = fig.add_subplot(gs[2, 1])
    has_force = False
    for i, (exp_name, data) in enumerate(zip(experiments, data_list)):
        if 'force_prediction' in data and len(data['force_prediction']) > 0:
            time = data['time_normalized']
            ax6.plot(time, data['force_prediction'], label=exp_name, 
                    color=colors[i], linewidth=2, alpha=0.8)
            has_force = True
    if has_force:
        ax6.set_xlabel('Normalized Time')
        ax6.set_ylabel('Force Prediction')
        ax6.set_title('Force Prediction Over Time')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Force prediction\nnot available', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Force Prediction')
    
    # 7. Tactile Magnitude (if available)
    ax7 = fig.add_subplot(gs[3, 0])
    has_tactile = False
    for i, (exp_name, data) in enumerate(zip(experiments, data_list)):
        if 'tactile_magnitude' in data and len(data['tactile_magnitude']) > 0:
            time = data['time_normalized']
            ax7.plot(time, data['tactile_magnitude'], label=exp_name, 
                    color=colors[i], linewidth=2, alpha=0.8)
            has_tactile = True
    if has_tactile:
        ax7.set_xlabel('Normalized Time')
        ax7.set_ylabel('Tactile Magnitude')
        ax7.set_title('Tactile Magnitude Over Time')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'Tactile data\nnot available', 
                ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Tactile Magnitude')
    
    # 8. Statistics Table
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.axis('off')
    
    # Create statistics table
    stats_data = []
    stats_data.append(['Experiment', 'Frames', 'Duration (est.)'])
    
    for exp_name, data in zip(experiments, data_list):
        num_frames = data.get('num_frames', 0)
        # Estimate duration (assuming ~50Hz)
        duration_est = num_frames / 50.0 if num_frames > 0 else 0
        stats_data.append([exp_name, str(num_frames), f'{duration_est:.1f}s'])
    
    table = ax8.table(cellText=stats_data[1:], colLabels=stats_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax8.set_title('Trajectory Statistics', pad=20)
    
    plt.suptitle('Experiment Comparison', fontsize=16, y=0.995)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Compare multiple trajectory experiments')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Parent directory containing experiment subdirectories')
    parser.add_argument('--output', type=str, default=None,
                       help='Output image path (default: <data_dir>/comparison.png)')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                       help='Specific experiment names to compare (default: all subdirectories)')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Find experiment directories
    if args.experiments:
        exp_dirs = [data_dir / exp for exp in args.experiments]
    else:
        exp_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    exp_dirs = [d for d in exp_dirs if (d / 'trajectory.h5').exists()]
    
    if len(exp_dirs) == 0:
        print(f"ERROR: No trajectory.h5 files found in subdirectories of {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(exp_dirs)} experiments:")
    for exp_dir in exp_dirs:
        print(f"  - {exp_dir.name}")
    
    # Load all trajectory data
    data_dict = {}
    for exp_dir in exp_dirs:
        h5_path = exp_dir / 'trajectory.h5'
        print(f"\nLoading {exp_dir.name}...")
        data = load_trajectory_data(h5_path)
        if data is not None:
            data_dict[exp_dir.name] = data
            print(f"  Loaded {data.get('num_frames', 0)} frames")
        else:
            print(f"  Failed to load data")
    
    if len(data_dict) == 0:
        print("ERROR: No data loaded successfully")
        sys.exit(1)
    
    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / 'comparison.png'
    
    # Create comparison plot
    print(f"\nCreating comparison visualization...")
    create_comparison_plot(data_dict, output_path)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
