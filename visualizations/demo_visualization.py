#!/usr/bin/env python3
"""Visualize a demo trajectory by pairing the wrist camera with robot telemetry."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.axes import Axes

from r2d2.camera_utils.recording_readers.mp4_reader import MP4Reader
from r2d2.camera_utils.recording_readers.svo_reader import SVOReader


DEFAULT_DEMO = "/home/pi0/multi-modal/droid-multi-modal/data/success/2025-11-18/Fri_Nov_14_14_19_34_2025"
WRIST_CAMERA_ID = "17225336"  # wrist_left
SECONDARY_CAMERA_ID = "24395123"  # varied camera
CONTROL_FREQUENCY_FALLBACK_HZ = 15.0


def _combine_robot_timestamp(seconds: np.ndarray, nanos: np.ndarray) -> np.ndarray:
    return seconds.astype(np.float64) + nanos.astype(np.float64) * 1e-9


def _normalize_timestamps(ts: np.ndarray) -> np.ndarray:
    if ts.size == 0:
        return ts.astype(np.float64)
    ts = ts.astype(np.float64)
    return ts - ts[0]


def _resample_series(source_times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Resample `values` collected at `source_times` onto `target_times` via linear interpolation."""
    if values.shape[0] == 0 or target_times.size == 0:
        new_shape = (target_times.size,) + tuple(values.shape[1:])
        return np.zeros(new_shape, dtype=np.float64)

    if source_times.size != values.shape[0]:
        raise ValueError("Timestamp/value length mismatch when resampling telemetry.")

    src = source_times.astype(np.float64)
    tgt = target_times.astype(np.float64)

    # Ensure strictly non-decreasing timestamps for interpolation.
    np.maximum.accumulate(src, out=src)

    flat = values.reshape(values.shape[0], -1).astype(np.float64)
    out = np.empty((tgt.size, flat.shape[1]), dtype=np.float64)
    for idx in range(flat.shape[1]):
        column = flat[:, idx]
        out[:, idx] = np.interp(tgt, src, column, left=column[0], right=column[-1])
    return out.reshape((tgt.size,) + tuple(values.shape[1:]))


def _convert_camera_timestamp(raw: np.ndarray, normalize: bool = True) -> np.ndarray:
    if raw.size == 0:
        return raw.astype(np.float64)
    raw = raw.astype(np.float64)
    diffs = np.diff(raw)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        scale = 1.0
    else:
        mean_diff = float(diffs.mean())
        if mean_diff >= 1e6:
            scale = 1e-9  # assume timestamps are in nanoseconds
        elif mean_diff >= 1e3:
            scale = 1e-6  # assume microseconds
        elif mean_diff >= 1:
            scale = 1e-3  # assume milliseconds
        else:
            scale = 1.0   # assume seconds already
    scaled = raw * scale
    if normalize:
        scaled = scaled - scaled[0]
    return scaled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo visualization helper.")
    parser.add_argument(
        "--demo-folder",
        type=Path,
        default=Path(DEFAULT_DEMO),
        help="Directory that contains trajectory.h5 and recordings/.",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default=WRIST_CAMERA_ID,
        help="Camera serial to visualize (defaults to the wrist_left camera).",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Optional explicit path to the wrist video (mp4 or svo/svo2).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit the number of frames to play (defaults to the trajectory length).",
    )
    parser.add_argument(
        "--secondary-camera-id",
        type=str,
        default=SECONDARY_CAMERA_ID,
        help="Serial for the additional camera (empty string to disable).",
    )
    parser.add_argument(
        "--secondary-video-path",
        type=Path,
        default=None,
        help="Optional explicit path for the secondary camera video.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Optional path to save the visualization as an MP4 file.",
    )
    return parser.parse_args()


def load_timeseries(h5_path: Path, camera_id: str) -> Dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        cam_ts_raw = np.array(f[f"observation/timestamp/cameras/{camera_id}_frame_received"][:], dtype=np.float64)
        joint_pos = np.array(f["observation/robot_state/joint_positions"][:], dtype=np.float64)
        gripper_pos = np.array(f["observation/robot_state/gripper_position"][:], dtype=np.float64)
        tactile_values = np.array(f["observation/robot_state/tactile_values"][:, -1, :], dtype=np.float64)
        try:
            force_prediction = np.array(f["observation/robot_state/force_prediction"][:, -1, 0], dtype=np.float64)
        except KeyError:
            force_prediction = np.zeros(joint_pos.shape[0], dtype=np.float64)

        try:
            robot_secs = np.array(
                f["observation/timestamp/robot_state/robot_timestamp_seconds"][:], dtype=np.float64
            )
            robot_nanos = np.array(
                f["observation/timestamp/robot_state/robot_timestamp_nanos"][:], dtype=np.float64
            )
            robot_ts = _combine_robot_timestamp(robot_secs, robot_nanos)
        except KeyError:
            fallback = np.arange(joint_pos.shape[0], dtype=np.float64) / CONTROL_FREQUENCY_FALLBACK_HZ
            robot_ts = fallback

    cam_ts_absolute = _convert_camera_timestamp(cam_ts_raw, normalize=False)
    cam_ts = cam_ts_absolute - cam_ts_absolute[0]
    robot_ts = _normalize_timestamps(robot_ts)

    joint_pos = _resample_series(robot_ts, joint_pos, cam_ts)
    gripper_pos = _resample_series(robot_ts, gripper_pos.reshape(-1, 1), cam_ts).reshape(-1)
    tactile_values = _resample_series(robot_ts, tactile_values, cam_ts)
    force_prediction = _resample_series(robot_ts, force_prediction.reshape(-1, 1), cam_ts).reshape(-1)

    return dict(
        timestamps=cam_ts,
        camera_timestamps_absolute=cam_ts_absolute,
        joint_positions=joint_pos,
        gripper_position=gripper_pos,
        tactile_values=tactile_values,
        force_prediction=force_prediction,
    )


def _candidate_video_paths(recording_dir: Path, camera_id: str) -> List[Path]:
    base_names = [f"{camera_id}.mp4", f"{camera_id}..mp4", f"{camera_id}.svo", f"{camera_id}.svo2"]
    search_dirs = [
        recording_dir / "MP4",
        recording_dir,
        recording_dir / "SVO",
    ]
    candidates: List[Path] = []
    for directory in search_dirs:
        for name in base_names:
            candidate = directory / name
            if candidate.exists():
                candidates.append(candidate)
    return candidates


def resolve_video_path(demo_folder: Path, camera_id: str, override: Optional[Path]) -> Path:
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"Provided video path not found: {override}")
        return override

    recording_dir = demo_folder / "recordings"
    if not recording_dir.exists():
        raise FileNotFoundError(f"No recordings folder found under {demo_folder}")

    candidates = _candidate_video_paths(recording_dir, camera_id)
    if not candidates:
        raise FileNotFoundError(f"Could not locate a video for camera {camera_id} in {recording_dir}")
    return candidates[0]


def _reader_for_path(video_path: Path, camera_id: str):
    suffix = video_path.suffix.lower()
    if suffix == ".mp4":
        return MP4Reader(str(video_path), serial_number=camera_id)
    if suffix in {".svo", ".svo2"}:
        return SVOReader(str(video_path), serial_number=camera_id)
    raise ValueError(f"Unsupported video file: {video_path}")


def load_video_frames(video_path: Path, camera_id: str, max_count: Optional[int]) -> Tuple[List[np.ndarray], np.ndarray]:
    reader = _reader_for_path(video_path, camera_id)
    if isinstance(reader, MP4Reader):
        reader.set_reading_parameters(image=True, concatenate_images=False)
    else:
        reader.set_reading_parameters(image=True, depth=False, pointcloud=False, concatenate_images=False)
    frames: List[np.ndarray] = []
    frame_timestamps: List[float] = []
    timestamps_complete = True
    key = f"{camera_id}_left"
    try:
        while max_count is None or len(frames) < max_count:
            result = reader.read_camera(return_timestamp=True)
            if result is None:
                break
            frame_dict, timestamp_val = result
            image_dict = frame_dict.get("image", {})
            if key not in image_dict:
                raise KeyError(f"{key} not found in frame output")
            frame = image_dict[key]
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if timestamp_val is None:
                timestamps_complete = False
            else:
                frame_timestamps.append(float(timestamp_val))
    finally:
        reader.disable_camera()
    if timestamps_complete and frame_timestamps:
        timestamps = _convert_camera_timestamp(np.array(frame_timestamps, dtype=np.float64), normalize=False)
    else:
        timestamps = np.array([], dtype=np.float64)
    return frames, timestamps


def _axis_limits(series: np.ndarray, pad_ratio: float = 0.05) -> Tuple[float, float]:
    min_val = float(np.nanmin(series))
    max_val = float(np.nanmax(series))
    if np.isclose(min_val, max_val):
        return min_val - 0.5, max_val + 0.5
    pad = (max_val - min_val) * pad_ratio
    return min_val - pad, max_val + pad


def _slice_timeseries(data: Dict[str, np.ndarray], start: int, end: int) -> None:
    keys = [
        "timestamps",
        "camera_timestamps_absolute",
        "joint_positions",
        "gripper_position",
        "tactile_values",
        "force_prediction",
    ]
    for key in keys:
        data[key] = data[key][start:end]


def _renormalize_time_axis(data: Dict[str, np.ndarray]) -> None:
    timestamps = data.get("timestamps")
    if timestamps is None or timestamps.size == 0:
        return
    offset = timestamps[0]
    if offset != 0.0:
        data["timestamps"] = timestamps - offset


def _uniform_sample_frames(frames: Sequence[np.ndarray], desired_count: int) -> List[np.ndarray]:
    if desired_count <= 0 or len(frames) == 0:
        return []
    if len(frames) == desired_count:
        return list(frames)
    positions = np.linspace(0, len(frames) - 1, num=desired_count)
    indices = np.clip(np.round(positions).astype(int), 0, len(frames) - 1)
    return [frames[idx] for idx in indices]


def _select_frames_for_timestamps(
    frames: Sequence[np.ndarray],
    frame_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
) -> List[np.ndarray]:
    if target_timestamps.size == 0:
        return []
    if len(frames) == 0:
        return []
    if frame_timestamps.size != len(frames):
        frame_timestamps = frame_timestamps[: len(frames)]
    if frame_timestamps.size == 0:
        return _uniform_sample_frames(frames, target_timestamps.size)
    ts = frame_timestamps.astype(np.float64).copy()
    np.maximum.accumulate(ts, out=ts)
    idx_float = np.interp(
        target_timestamps,
        ts,
        np.arange(ts.size, dtype=np.float64),
        left=0.0,
        right=float(ts.size - 1),
    )
    indices = np.clip(np.round(idx_float).astype(int), 0, len(frames) - 1)
    return [frames[idx] for idx in indices]


def _crop_data_to_video_range(data: Dict[str, np.ndarray], video_timestamps: np.ndarray) -> None:
    if video_timestamps.size == 0:
        return
    camera_abs = data.get("camera_timestamps_absolute")
    if camera_abs is None or camera_abs.size == 0:
        return
    start_time = video_timestamps[0]
    end_time = video_timestamps[-1]
    start_idx = np.searchsorted(camera_abs, start_time, side="left")
    end_idx = np.searchsorted(camera_abs, end_time, side="right")
    if end_idx <= start_idx:
        raise ValueError("Video timestamps do not overlap with telemetry timestamps.")
    if start_idx == 0 and end_idx >= camera_abs.size:
        return
    _slice_timeseries(data, start_idx, end_idx)
    _renormalize_time_axis(data)


def build_layout(
    time_axis: np.ndarray,
    joint_positions: np.ndarray,
    gripper: np.ndarray,
    tactile: np.ndarray,
    force: np.ndarray,
    video_frames: Sequence[np.ndarray],
    primary_camera_id: str,
    secondary_video_frames: Optional[Sequence[np.ndarray]] = None,
    secondary_camera_id: Optional[str] = None,
):
    fig = plt.figure(figsize=(22, 14))
    video_rows = 2 if secondary_video_frames is not None else 1
    num_plots = 10

    outer = gridspec.GridSpec(1, 2, width_ratios=[3.8, 1.2], wspace=0.04, figure=fig)
    left_gs = outer[0].subgridspec(video_rows, 1, hspace=0.08)
    right_gs = outer[1].subgridspec(num_plots, 1, hspace=0.1)

    video_axes: List[Axes] = []
    video_artists: List = []

    primary_ax = fig.add_subplot(left_gs[0, 0])
    primary_ax.set_title(f"{primary_camera_id} (wrist)", fontsize=14)
    primary_ax.axis("off")
    primary_artist = primary_ax.imshow(video_frames[0])
    video_axes.append(primary_ax)
    video_artists.append(primary_artist)

    if secondary_video_frames is not None:
        secondary_ax = fig.add_subplot(left_gs[1, 0])
        title = secondary_camera_id or "Secondary Camera"
        secondary_ax.set_title(title, fontsize=14)
        secondary_ax.axis("off")
        secondary_artist = secondary_ax.imshow(secondary_video_frames[0])
        video_axes.append(secondary_ax)
        video_artists.append(secondary_artist)

    plot_axes: List[Axes] = []
    for i in range(num_plots):
        share_x = plot_axes[0] if plot_axes else None
        ax = fig.add_subplot(right_gs[i, 0], sharex=share_x)
        ax.set_xlim(time_axis[0], time_axis[-1])
        if i < num_plots - 1:
            ax.tick_params(labelbottom=False)
        plot_axes.append(ax)
    plot_axes[-1].set_xlabel("Time (s)")
    joint_lines = []
    cursor_lines = []
    for idx in range(7):
        plot_axes[idx].set_title(f"Joint {idx + 1} Position")
        plot_axes[idx].set_ylim(*_axis_limits(joint_positions[:, idx]))
        (line,) = plot_axes[idx].plot([], [], lw=1.5)
        joint_lines.append(line)
        cursor_lines.append(plot_axes[idx].axvline(time_axis[0], color="k", lw=0.8, ls="--"))

    plot_axes[7].set_title("Gripper Position")
    plot_axes[7].set_ylim(*_axis_limits(gripper))
    (gripper_line,) = plot_axes[7].plot([], [], color="tab:green", lw=1.5)
    cursor_lines.append(plot_axes[7].axvline(time_axis[0], color="k", lw=0.8, ls="--"))

    tactile_lines = []
    plot_axes[8].set_title("Tactile Values")
    plot_axes[8].set_ylim(*_axis_limits(tactile))
    for sensor_idx in range(tactile.shape[1]):
        (line,) = plot_axes[8].plot([], [], lw=1.0, label=f"T{sensor_idx + 1}")
        tactile_lines.append(line)
    plot_axes[8].legend(loc="upper right", fontsize="small", ncol=3)
    cursor_lines.append(plot_axes[8].axvline(time_axis[0], color="k", lw=0.8, ls="--"))

    plot_axes[9].set_title("Force Prediction")
    plot_axes[9].set_ylim(*_axis_limits(force))
    (force_line,) = plot_axes[9].plot([], [], color="tab:red", lw=1.5)
    cursor_lines.append(plot_axes[9].axvline(time_axis[0], color="k", lw=0.8, ls="--"))

    return dict(
        fig=fig,
        video_axes=video_axes,
        video_artists=video_artists,
        joint_lines=joint_lines,
        gripper_line=gripper_line,
        tactile_lines=tactile_lines,
        force_line=force_line,
        cursor_lines=cursor_lines,
        plot_axes=plot_axes,
    )


def animate(
    layout: Dict[str, object],
    video_frames: Sequence[np.ndarray],
    secondary_video_frames: Optional[Sequence[np.ndarray]],
    time_axis: np.ndarray,
    joint_positions: np.ndarray,
    gripper: np.ndarray,
    tactile: np.ndarray,
    force: np.ndarray,
):
    video_artists: List = layout["video_artists"]
    joint_lines: List = layout["joint_lines"]
    gripper_line = layout["gripper_line"]
    tactile_lines: List = layout["tactile_lines"]
    force_line = layout["force_line"]
    cursor_lines: List = layout["cursor_lines"]

    def _update(frame_idx: int):
        video_artists[0].set_data(video_frames[frame_idx])
        if secondary_video_frames is not None and len(video_artists) > 1:
            video_artists[1].set_data(secondary_video_frames[frame_idx])

        current_time = time_axis[frame_idx]
        time_slice = time_axis[: frame_idx + 1]

        for joint_idx, line in enumerate(joint_lines):
            line.set_data(time_slice, joint_positions[: frame_idx + 1, joint_idx])

        gripper_line.set_data(time_slice, gripper[: frame_idx + 1])

        for sensor_idx, line in enumerate(tactile_lines):
            line.set_data(time_slice, tactile[: frame_idx + 1, sensor_idx])

        force_line.set_data(time_slice, force[: frame_idx + 1])

        for cursor in cursor_lines:
            cursor.set_xdata([current_time, current_time])

        artists = list(video_artists)
        artists.extend([gripper_line, force_line])
        artists.extend(joint_lines)
        artists.extend(tactile_lines)
        artists.extend(cursor_lines)
        return artists

    average_dt = np.mean(np.diff(time_axis)) if len(time_axis) > 1 else 0.05
    interval_ms = max(1, int(average_dt * 1000))

    animation_obj = FuncAnimation(
        layout["fig"],
        _update,
        frames=len(video_frames),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    return animation_obj, interval_ms


def main():
    args = parse_args()
    h5_path = args.demo_folder / "trajectory.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"trajectory.h5 not found under {args.demo_folder}")

    data = load_timeseries(h5_path, args.camera_id)
    video_path = resolve_video_path(args.demo_folder, args.camera_id, args.video_path)

    video_frames_raw, video_timestamps_abs = load_video_frames(video_path, args.camera_id, None)
    if not video_frames_raw:
        raise RuntimeError(f"No frames could be read from {video_path}")

    if video_timestamps_abs.size > 0:
        _crop_data_to_video_range(data, video_timestamps_abs)

    if args.max_frames is not None:
        max_frames = min(args.max_frames, len(data["timestamps"]))
        _slice_timeseries(data, 0, max_frames)
        _renormalize_time_axis(data)

    if len(data["timestamps"]) == 0:
        raise RuntimeError("No telemetry samples available for visualization.")

    if video_timestamps_abs.size > 0:
        video_frames = _select_frames_for_timestamps(
            video_frames_raw, video_timestamps_abs, data["camera_timestamps_absolute"]
        )
    else:
        if len(video_frames_raw) < len(data["timestamps"]):
            print("Warning: video timestamps missing; truncating telemetry to video length.")
            _slice_timeseries(data, 0, len(video_frames_raw))
            _renormalize_time_axis(data)
        video_frames = _uniform_sample_frames(video_frames_raw, len(data["timestamps"]))

    secondary_frames = None
    secondary_cam_id = args.secondary_camera_id.strip()
    if secondary_cam_id:
        secondary_video_path = resolve_video_path(args.demo_folder, secondary_cam_id, args.secondary_video_path)
        secondary_raw, secondary_ts_abs = load_video_frames(secondary_video_path, secondary_cam_id, None)
        if not secondary_raw:
            raise RuntimeError(f"No frames could be read from {secondary_video_path}")
        if secondary_ts_abs.size > 0:
            secondary_frames = _select_frames_for_timestamps(
                secondary_raw, secondary_ts_abs, data["camera_timestamps_absolute"]
            )
        else:
            if len(secondary_raw) < len(data["timestamps"]):
                print("Warning: secondary video shorter than telemetry; repeating last available frame.")
            secondary_frames = _uniform_sample_frames(secondary_raw, len(data["timestamps"]))

    final_count = min(len(video_frames), len(data["timestamps"]))
    if final_count == 0:
        raise RuntimeError("No overlapping video frames and telemetry samples after alignment.")
    if final_count != len(data["timestamps"]):
        _slice_timeseries(data, 0, final_count)
        _renormalize_time_axis(data)
        video_frames = video_frames[:final_count]
        if secondary_frames is not None:
            secondary_frames = secondary_frames[:final_count]
    elif secondary_frames is not None and len(secondary_frames) != final_count:
        secondary_frames = secondary_frames[:final_count]

    layout = build_layout(
        time_axis=data["timestamps"],
        joint_positions=data["joint_positions"],
        gripper=data["gripper_position"],
        tactile=data["tactile_values"],
        force=data["force_prediction"],
        video_frames=video_frames,
        primary_camera_id=args.camera_id,
        secondary_video_frames=secondary_frames,
        secondary_camera_id=secondary_cam_id if secondary_cam_id else None,
    )

    anim, interval_ms = animate(
        layout,
        video_frames=video_frames,
        secondary_video_frames=secondary_frames,
        time_axis=data["timestamps"],
        joint_positions=data["joint_positions"],
        gripper=data["gripper_position"],
        tactile=data["tactile_values"],
        force=data["force_prediction"],
    )
    if args.output_video is not None:
        output_path = args.output_video
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = 15 if interval_ms == 0 else max(1, int(round(1000 / interval_ms)))
        writer = FFMpegWriter(fps=fps)
        print(f"Saving visualization to {output_path} (fps={fps})...")
        anim.save(str(output_path), writer=writer)
        print(f"Wrote {output_path}")

    plt.show()
    return anim


if __name__ == "__main__":
    main()
