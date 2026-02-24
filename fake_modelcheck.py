#!/usr/bin/env python3
"""
Run tactile_adapter on demo data and plot normalized delta over time.
"""

import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from r2d2.misc.parameters import hand_camera_id

cwd = os.getcwd()
sys.path.append(cwd)

# Add tactile_module to path
sys.path.insert(0, "/home/pi0/multi-modal/tactile_module")
from robot_inference_adapter import TactileGripperAdapter

DEMO_DATA_DIR = "/home/pi0/multi-modal/droid-multi-modal/data/success/2025-12-24/Wed_Dec_24_14_15_31_2025"
TACTILE_MODEL_CHECKPOINT = "/home/pi0/multi-modal/tactile_module/checkpoints/1224_paper_cup_two_img_delta_gripper_normalized.pt"
TACTILE_MODEL_CONFIG = "/home/pi0/multi-modal/tactile_module/configs/example_with_normalization.yaml"


def load_demo_sources(demo_dir, camera_id):
    demo_path = Path(demo_dir).expanduser()
    trajectory_path = demo_path / "trajectory.h5"
    recordings_dir = demo_path / "recordings" / "MP4"

    if not trajectory_path.exists():
        raise FileNotFoundError(f"trajectory.h5 not found at {trajectory_path}")
    if not recordings_dir.exists():
        raise FileNotFoundError(f"recordings/MP4 not found at {recordings_dir}")

    candidates = [
        recordings_dir / f"{camera_id}..mp4",
        recordings_dir / f"{camera_id}.mp4",
    ]
    video_path = next((p for p in candidates if p.exists()), None)
    if video_path is None:
        raise FileNotFoundError(f"Video not found for camera {camera_id} in {recordings_dir}")

    with h5py.File(trajectory_path, "r") as h5f:
        tactile_values = h5f["observation/robot_state/tactile_values"][:]
        joint_positions = h5f["observation/robot_state/joint_positions"][:]
        gripper_position = h5f["observation/robot_state/gripper_position"][:]
        force_pred = h5f["observation/robot_state/force_prediction"][:]
        force_prediction = force_pred[:, -1, 0]
        tactile_last = tactile_values[:, -1, :]

        if "action/gripper_velocity" in h5f:
            gripper_command = h5f["action/gripper_velocity"][:]
            gripper_command_type = "velocity"
        elif "action/gripper_position" in h5f:
            gripper_command = h5f["action/gripper_position"][:]
            gripper_command_type = "position"
        else:
            gripper_command = None
            gripper_command_type = None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")

    data = {
        "tactile_values": tactile_values,
        "tactile_last": tactile_last,
        "joint_positions": joint_positions,
        "gripper_position": gripper_position,
        "force_prediction": force_prediction,
        "gripper_command": gripper_command,
        "gripper_command_type": gripper_command_type,
        "num_frames": len(joint_positions),
    }

    return cap, data, trajectory_path, video_path


def plot_signals(data, normalized_deltas):
    num_frames = min(data["num_frames"], len(normalized_deltas))
    time_idx = np.arange(num_frames)

    fig = plt.figure(figsize=(12, 14), dpi=110)
    gs = GridSpec(7, 1, figure=fig, hspace=0.35)

    ax_norm = fig.add_subplot(gs[0, 0])
    ax_norm.plot(time_idx, normalized_deltas[:num_frames], color="red", linewidth=1.5)
    ax_norm.set_title("Normalized Delta", fontsize=10, fontweight="bold")
    ax_norm.set_ylabel("Value", fontsize=8)
    ax_norm.grid(True, alpha=0.3)

    ax_joints = fig.add_subplot(gs[1, 0])
    for joint_idx in range(7):
        ax_joints.plot(
            time_idx,
            data["joint_positions"][:num_frames, joint_idx],
            label=f"J{joint_idx}",
            linewidth=1.2,
        )
    ax_joints.set_title("Joint Positions", fontsize=10, fontweight="bold")
    ax_joints.set_ylabel("Rad", fontsize=8)
    ax_joints.legend(loc="upper right", fontsize=6, ncol=4)
    ax_joints.grid(True, alpha=0.3)

    ax_grip = fig.add_subplot(gs[2, 0])
    ax_grip.plot(time_idx, data["gripper_position"][:num_frames], color="purple", linewidth=1.5)
    ax_grip.set_title("Gripper Position", fontsize=10, fontweight="bold")
    ax_grip.set_ylabel("Position", fontsize=8)
    ax_grip.grid(True, alpha=0.3)

    ax_force = fig.add_subplot(gs[3, 0])
    ax_force.plot(time_idx, data["force_prediction"][:num_frames], color="orange", linewidth=1.5)
    ax_force.set_title("Force Prediction", fontsize=10, fontweight="bold")
    ax_force.set_ylabel("Force", fontsize=8)
    ax_force.grid(True, alpha=0.3)

    ax_tact = fig.add_subplot(gs[4, 0])
    for channel_idx in range(data["tactile_last"].shape[1]):
        ax_tact.plot(
            time_idx,
            data["tactile_last"][:num_frames, channel_idx],
            label=f"Ch {channel_idx}",
            linewidth=1.2,
        )
    ax_tact.set_title("Tactile Values (Last Sample)", fontsize=10, fontweight="bold")
    ax_tact.set_ylabel("Value", fontsize=8)
    ax_tact.legend(loc="upper right", fontsize=6, ncol=3)
    ax_tact.grid(True, alpha=0.3)

    ax_cmd = fig.add_subplot(gs[5, 0])
    if data["gripper_command"] is not None:
        ax_cmd.plot(time_idx, data["gripper_command"][:num_frames], color="teal", linewidth=1.5)
        label = "Velocity" if data["gripper_command_type"] == "velocity" else "Position"
        ax_cmd.set_ylabel(label, fontsize=8)
        ax_cmd.set_title("Gripper Command", fontsize=10, fontweight="bold")
    else:
        ax_cmd.text(0.5, 0.5, "No gripper command logged",
                    ha="center", va="center", fontsize=8)
        ax_cmd.set_title("Gripper Command", fontsize=10, fontweight="bold")
    ax_cmd.grid(True, alpha=0.3)

    ax_disp = fig.add_subplot(gs[6, 0])
    if data["gripper_command"] is not None:
        cmd = data["gripper_command"][:num_frames]
        if data["gripper_command_type"] == "velocity":
            disp = np.cumsum(cmd)
            ylabel = "Rel. Disp."
        else:
            disp = cmd - cmd[0]
            ylabel = "Rel. Position"
        ax_disp.plot(time_idx, disp, color="slateblue", linewidth=1.5)
        ax_disp.set_ylabel(ylabel, fontsize=8)
    else:
        ax_disp.text(0.5, 0.5, "No gripper command logged",
                     ha="center", va="center", fontsize=8)
    ax_disp.set_title("Gripper Command Displacement", fontsize=10, fontweight="bold")
    ax_disp.set_xlabel("Frame", fontsize=8)
    ax_disp.grid(True, alpha=0.3)

    fig.suptitle("Demo Signals + Tactile Model Output", fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()


def main():
    print("Loading tactile model...")
    tactile_adapter = TactileGripperAdapter(
        checkpoint_path=TACTILE_MODEL_CHECKPOINT,
        config_path=TACTILE_MODEL_CONFIG,
    )
    print(f"Model loaded on device: {tactile_adapter.device}")

    cap, data, trajectory_path, video_path = load_demo_sources(
        DEMO_DATA_DIR, hand_camera_id
    )
    num_frames = data["tactile_values"].shape[0]
    print(f"Demo trajectory: {trajectory_path}")
    print(f"Demo video: {video_path}")
    print(f"Frames available: {num_frames}")

    normalized_deltas = []
    frame_idx = 0

    while frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        half_width = width // 2
        left_bgr = frame[:, :half_width]
        right_bgr = frame[:, half_width:]

        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
        tactile_history = data["tactile_values"][frame_idx].astype(np.float32)

        normalized_delta = tactile_adapter.predict_delta(
            image_left=left_rgb,
            image_right=right_rgb,
            tactile_history=tactile_history,
            step_index=0,
        )
        try:
            normalized_delta = float(np.asarray(normalized_delta).reshape(-1)[0])
        except Exception:
            normalized_delta = float(normalized_delta)

        print(f"frame={frame_idx:04d} normalized_delta={normalized_delta:+.6f}")
        normalized_deltas.append(normalized_delta)
        frame_idx += 1

    cap.release()

    if not normalized_deltas:
        print("No frames processed; skipping plot.")
        return

    plot_signals(data, normalized_deltas)


if __name__ == "__main__":
    main()
