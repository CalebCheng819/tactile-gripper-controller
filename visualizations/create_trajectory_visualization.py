#!/usr/bin/env python3
"""
Create synchronized visualization of trajectory data with camera feeds and plots.
Left: Wrist camera (top) and external camera (bottom)
Right: Joint positions, gripper position, force prediction, tactile values
"""

import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FFMpegWriter
from pathlib import Path
import json
import sys
import argparse

# Camera IDs
WRIST_CAM = "17225336"
EXTERNAL_CAM = "24013089"  # or 24395123


def load_trajectory_data(h5_path):
    """Load all necessary data from HDF5 file."""
    data = {}

    with h5py.File(h5_path, 'r') as f:
        # Joint positions (121, 7)
        data['joint_positions'] = f['observation/robot_state/joint_positions'][:]

        # Gripper position (121,)
        data['gripper_position'] = f['observation/robot_state/gripper_position'][:]

        # Force prediction (121, 500, 1) - take last value
        force_pred = f['observation/robot_state/force_prediction'][:]
        data['force_prediction'] = force_pred[:, -1, 0]  # Shape: (121,)

        # Tactile values (121, 500, 6) - take last value
        tactile = f['observation/robot_state/tactile_values'][:]
        data['tactile_values'] = tactile[:, -1, :]  # Shape: (121, 6)

        # Gripper command (from action). Prefer velocity, fall back to position.
        if 'action/gripper_velocity' in f:
            data['gripper_command'] = f['action/gripper_velocity'][:]
            data['gripper_command_type'] = 'velocity'
        elif 'action/gripper_position' in f:
            data['gripper_command'] = f['action/gripper_position'][:]
            data['gripper_command_type'] = 'position'
        else:
            data['gripper_command'] = None
            data['gripper_command_type'] = None

        # Timestamps for reference
        data['num_frames'] = len(data['joint_positions'])

    return data


def load_video_frames(recordings_dir, camera_id):
    """Load all frames from a camera video (left view only)."""
    # Some datasets save as "<id>.mp4" while others use "<id>..mp4"; try both.
    candidates = [
        recordings_dir / f"{camera_id}..mp4",
        recordings_dir / f"{camera_id}.mp4",
    ]
    video_path = next((p for p in candidates if p.exists()), None)
    if video_path is None:
        raise FileNotFoundError(f"Video not found for camera {camera_id}")

    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract left camera view only (first half of the width)
        # Videos are side-by-side stereo (2560x720), we want left half (1280x720)
        height, width = frame_rgb.shape[:2]
        left_frame = frame_rgb[:, :width//2]

        frames.append(left_frame)

    cap.release()
    print(f"Loaded {len(frames)} frames from {camera_id} (left view only)")
    return frames


def create_visualization_frame(fig, gs, data, wrist_frames, external_frames, frame_idx):
    """Create a single visualization frame."""
    fig.clear()

    # Left column: Video frames
    ax_wrist = fig.add_subplot(gs[0:3, 0])
    ax_external = fig.add_subplot(gs[3:6, 0])

    # Right column: Plots (6 rows)
    ax_joints = fig.add_subplot(gs[0, 1])
    ax_gripper = fig.add_subplot(gs[1, 1])
    ax_force = fig.add_subplot(gs[2, 1])
    ax_tactile = fig.add_subplot(gs[3, 1])
    ax_gripper_cmd = fig.add_subplot(gs[4, 1])
    ax_gripper_cmd_disp = fig.add_subplot(gs[5, 1])

    # === LEFT SIDE: VIDEOS ===
    # Wrist camera
    if frame_idx < len(wrist_frames):
        ax_wrist.imshow(wrist_frames[frame_idx])
    ax_wrist.set_title(f'Wrist Camera ({WRIST_CAM})', fontsize=10, fontweight='bold')
    ax_wrist.axis('off')

    # External camera
    if frame_idx < len(external_frames):
        ax_external.imshow(external_frames[frame_idx])
    ax_external.set_title(f'External Camera ({EXTERNAL_CAM})', fontsize=10, fontweight='bold')
    ax_external.axis('off')

    # === RIGHT SIDE: PLOTS ===
    current_time = np.arange(frame_idx + 1)
    total_time = np.arange(data['num_frames'])

    # 1. Joint Positions (7 joints)
    for joint_idx in range(7):
        ax_joints.plot(current_time,
                      data['joint_positions'][:frame_idx+1, joint_idx],
                      label=f'Joint {joint_idx}',
                      linewidth=1.5)
    ax_joints.axvline(frame_idx, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax_joints.set_xlim(0, data['num_frames'])
    ax_joints.set_ylabel('Position (rad)', fontsize=8)
    ax_joints.set_title('Joint Positions', fontsize=9, fontweight='bold')
    ax_joints.legend(loc='upper right', fontsize=6, ncol=2)
    ax_joints.grid(True, alpha=0.3)
    ax_joints.tick_params(labelsize=7)

    # 2. Gripper Position
    ax_gripper.plot(current_time,
                    data['gripper_position'][:frame_idx+1],
                    color='purple',
                    linewidth=2)
    ax_gripper.axvline(frame_idx, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax_gripper.set_xlim(0, data['num_frames'])
    ax_gripper.set_ylim(-0.05, max(data['gripper_position']) + 0.05)
    ax_gripper.set_ylabel('Position', fontsize=8)
    ax_gripper.set_title('Gripper Position', fontsize=9, fontweight='bold')
    ax_gripper.grid(True, alpha=0.3)
    ax_gripper.tick_params(labelsize=7)

    # 3. Force Prediction
    ax_force.plot(current_time,
                  data['force_prediction'][:frame_idx+1],
                  color='orange',
                  linewidth=2)
    ax_force.axvline(frame_idx, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax_force.set_xlim(0, data['num_frames'])
    ax_force.set_ylabel('Force', fontsize=8)
    ax_force.set_title('Force Prediction', fontsize=9, fontweight='bold')
    ax_force.grid(True, alpha=0.3)
    ax_force.tick_params(labelsize=7)

    # 4. Tactile Values (6 channels)
    for channel_idx in range(6):
        ax_tactile.plot(current_time,
                       data['tactile_values'][:frame_idx+1, channel_idx],
                       label=f'Ch {channel_idx}',
                       linewidth=1.5)
    ax_tactile.axvline(frame_idx, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax_tactile.set_xlim(0, data['num_frames'])
    ax_tactile.set_xlabel('Frame', fontsize=8)
    ax_tactile.set_ylabel('Value', fontsize=8)
    ax_tactile.set_title('Tactile Values', fontsize=9, fontweight='bold')
    ax_tactile.legend(loc='upper right', fontsize=6, ncol=3)
    ax_tactile.grid(True, alpha=0.3)
    ax_tactile.tick_params(labelsize=7)

    # 5. Gripper Command (from action dict)
    if data['gripper_command'] is not None:
        cmd = data['gripper_command']
        ax_gripper_cmd.plot(current_time, cmd[:frame_idx+1], color='teal', linewidth=2)
        ax_gripper_cmd.axvline(frame_idx, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax_gripper_cmd.set_xlim(0, data['num_frames'])
        cmd_min = float(np.min(cmd))
        cmd_max = float(np.max(cmd))
        pad = max(0.05, 0.1 * (cmd_max - cmd_min))
        ax_gripper_cmd.set_ylim(cmd_min - pad, cmd_max + pad)
        label = 'Velocity' if data['gripper_command_type'] == 'velocity' else 'Position'
        ax_gripper_cmd.set_ylabel(label, fontsize=8)
    else:
        ax_gripper_cmd.text(0.5, 0.5, 'No gripper command logged',
                            ha='center', va='center', fontsize=8)
        ax_gripper_cmd.set_xlim(0, data['num_frames'])
        ax_gripper_cmd.set_ylim(-1, 1)
        ax_gripper_cmd.set_ylabel('Command', fontsize=8)
    ax_gripper_cmd.axvline(frame_idx, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax_gripper_cmd.set_xlabel('Frame', fontsize=8)
    ax_gripper_cmd.set_title('Gripper Command', fontsize=9, fontweight='bold')
    ax_gripper_cmd.grid(True, alpha=0.3)
    ax_gripper_cmd.tick_params(labelsize=7)

    # 6. Gripper Command Displacement (relative)
    if data['gripper_command'] is not None:
        cmd = data['gripper_command']
        if data['gripper_command_type'] == 'velocity':
            disp = np.cumsum(cmd)
            ylabel = 'Rel. Disp.'
        else:
            disp = cmd - cmd[0]
            ylabel = 'Rel. Position'
        ax_gripper_cmd_disp.plot(current_time, disp[:frame_idx+1], color='slateblue', linewidth=2)
        ax_gripper_cmd_disp.axvline(frame_idx, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax_gripper_cmd_disp.set_xlim(0, data['num_frames'])
        disp_min = float(np.min(disp))
        disp_max = float(np.max(disp))
        pad = max(0.05, 0.1 * (disp_max - disp_min))
        ax_gripper_cmd_disp.set_ylim(disp_min - pad, disp_max + pad)
        ax_gripper_cmd_disp.set_ylabel(ylabel, fontsize=8)
    else:
        ax_gripper_cmd_disp.text(0.5, 0.5, 'No gripper command logged',
                                 ha='center', va='center', fontsize=8)
        ax_gripper_cmd_disp.set_xlim(0, data['num_frames'])
        ax_gripper_cmd_disp.set_ylim(-1, 1)
        ax_gripper_cmd_disp.set_ylabel('Disp.', fontsize=8)
    ax_gripper_cmd_disp.axvline(frame_idx, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax_gripper_cmd_disp.set_xlabel('Frame', fontsize=8)
    ax_gripper_cmd_disp.set_title('Gripper Command Displacement', fontsize=9, fontweight='bold')
    ax_gripper_cmd_disp.grid(True, alpha=0.3)
    ax_gripper_cmd_disp.tick_params(labelsize=7)

    # Add overall frame counter
    fig.suptitle(f'Frame: {frame_idx + 1} / {data["num_frames"]}',
                fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.97])


def main():
    parser = argparse.ArgumentParser(description='Create trajectory visualization')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to data directory containing trajectory.h5 and recordings/')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (default: <data_dir>/visualization.mp4)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    trajectory_path = data_dir / "trajectory.h5"
    recordings_dir = data_dir / "recordings" / "MP4"

    if not trajectory_path.exists():
        print(f"ERROR: trajectory.h5 not found at {trajectory_path}")
        sys.exit(1)

    if not recordings_dir.exists():
        print(f"ERROR: recordings/MP4 directory not found at {recordings_dir}")
        sys.exit(1)

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        date_str = data_dir.parent.name
        global_vis_root = data_dir.parents[2] / "visualization_videos"
        exp_name = data_dir.name

        save_dir = global_vis_root / date_str
        save_dir.mkdir(parents=True, exist_ok=True)

        output_path = save_dir /f"{exp_name}_visualization.mp4"

    print("=" * 60)
    print("Creating Trajectory Visualization")
    print("=" * 60)
    print(f"Data dir: {data_dir}")
    print(f"Output: {output_path}")

    # Load trajectory data
    print("\n[1/4] Loading trajectory data...")
    data = load_trajectory_data(trajectory_path)
    print(f"  - Loaded {data['num_frames']} frames")
    print(f"  - Joint positions: {data['joint_positions'].shape}")
    print(f"  - Gripper position: {data['gripper_position'].shape}")
    print(f"  - Force prediction: {data['force_prediction'].shape}")
    print(f"  - Tactile values: {data['tactile_values'].shape}")

    # Load video frames
    print("\n[2/4] Loading video frames...")
    try:
        wrist_frames = load_video_frames(recordings_dir, WRIST_CAM)
        external_frames = load_video_frames(recordings_dir, EXTERNAL_CAM)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Setup figure
    print("\n[3/4] Setting up visualization...")
    fig = plt.figure(figsize=(18, 12), dpi=100)
    gs = GridSpec(6, 2, figure=fig, width_ratios=[1.8, 1], hspace=0.3, wspace=0.25)

    # Setup video writer
    fps = 15  # Adjust based on your preference
    print(f"\n[4/4] Generating video at {fps} FPS...")
    writer = FFMpegWriter(fps=fps, bitrate=5000)

    # Determine number of frames to render
    num_frames = min(data['num_frames'], len(wrist_frames), len(external_frames))
    print(f"  - Rendering {num_frames} frames...")

    with writer.saving(fig, str(output_path), dpi=100):
        for frame_idx in range(num_frames):
            if frame_idx % 10 == 0:
                print(f"    Frame {frame_idx + 1}/{num_frames}", end='\r')

            create_visualization_frame(fig, gs, data, wrist_frames, external_frames, frame_idx)
            writer.grab_frame()

    print(f"\n\n{'=' * 60}")
    print(f"✓ Visualization saved to: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
