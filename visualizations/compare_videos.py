#!/usr/bin/env python3
"""
Compare multiple experiment videos side-by-side in a 2x2 grid.
Loads videos from multiple directories and creates synchronized comparison video.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys
import os

# Add droid-multi-modal to path for SVOReader
script_dir = Path(__file__).parent
droid_root = script_dir.parent.parent
if str(droid_root) not in sys.path:
    sys.path.insert(0, str(droid_root))


def load_video_frames_from_svo(svo_path, camera_id):
    """Load frames directly from SVO file."""
    try:
        from r2d2.camera_utils.recording_readers.svo_reader import SVOReader
    except ImportError:
        raise ImportError("SVOReader not available. Install r2d2 package or convert SVO to MP4 first.")
    
    try:
        camera = SVOReader(str(svo_path), serial_number=camera_id)
        camera.set_reading_parameters(image=True, depth=False, pointcloud=False, concatenate_images=True)
        
        frames = []
        frame_count = camera.get_frame_count()
        
        for _ in range(frame_count):
            output = camera.read_camera(return_timestamp=False)
            if output is None:
                break
            
            data_dict = output
            sbs_frame = data_dict["image"][camera_id]
            
            # Convert BGRA to RGB
            if len(sbs_frame.shape) == 3 and sbs_frame.shape[2] == 4:
                frame_rgb = cv2.cvtColor(sbs_frame, cv2.COLOR_BGRA2RGB)
            else:
                frame_rgb = cv2.cvtColor(sbs_frame, cv2.COLOR_BGR2RGB)
            
            # Extract left camera view only (first half of the width)
            height, width = frame_rgb.shape[:2]
            left_frame = frame_rgb[:, :width//2]
            
            frames.append(left_frame)
        
        camera.disable_camera()
        return frames
    except Exception as e:
        raise RuntimeError(f"Failed to read SVO file (ZED SDK may not be set up): {e}. "
                          f"Try converting SVO to MP4 first using: "
                          f"python3 droid-multi-modal/scripts/convert/svo_to_mp4_v2.py")


def load_video_frames(recordings_dir, camera_id, auto_convert_svo=False):
    """Load all frames from a camera video (left view only)."""
    recordings_path = Path(recordings_dir)
    
    # Try MP4 directory first
    mp4_dir = recordings_path / "MP4"
    video_path = None
    if mp4_dir.exists():
        candidates = [
            mp4_dir / f"{camera_id}..mp4",
            mp4_dir / f"{camera_id}.mp4",
        ]
        video_path = next((p for p in candidates if p.exists()), None)
    
    # If MP4 not found, try SVO
    if video_path is None:
        svo_dir = recordings_path / "SVO"
        if svo_dir.exists():
            # Try .svo2 first, then .svo
            svo_file = svo_dir / f"{camera_id}.svo2"
            if not svo_file.exists():
                svo_file = svo_dir / f"{camera_id}.svo"
            
            if svo_file.exists():
                if auto_convert_svo:
                    print(f"  Converting SVO to MP4: {svo_file.name}")
                    try:
                        from r2d2.camera_utils.recording_readers.svo_reader import SVOReader
                        import json
                        
                        camera = SVOReader(str(svo_file), serial_number=camera_id)
                        camera.set_reading_parameters(image=True, depth=False, pointcloud=False, concatenate_images=True)
                        width, height = camera.get_frame_resolution()
                        
                        mp4_dir.mkdir(parents=True, exist_ok=True)
                        mp4_path = mp4_dir / f"{camera_id}.mp4"
                        video_codec = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(mp4_path), video_codec, 15, (width * 2, height))
                        
                        frame_count = camera.get_frame_count()
                        received_timestamps = []
                        
                        for _ in range(frame_count):
                            output = camera.read_camera(return_timestamp=True)
                            if output is None:
                                break
                            data_dict, timestamp = output
                            sbs_frame = data_dict["image"][camera_id]
                            received_timestamps.append(timestamp)
                            sbs_frame = cv2.cvtColor(sbs_frame, cv2.COLOR_BGRA2BGR)
                            writer.write(sbs_frame)
                        
                        camera.disable_camera()
                        writer.release()
                        
                        # Save timestamps
                        ts_path = mp4_dir / f"{camera_id}_timestamps.json"
                        with open(ts_path, "w") as f:
                            json.dump(received_timestamps, f)
                        
                        video_path = mp4_path
                        print(f"  Converted to: {mp4_path}")
                    except Exception as e:
                        print(f"  Conversion failed: {e}")
                        print(f"  Trying direct SVO reading...")
                        try:
                            return load_video_frames_from_svo(svo_file, camera_id)
                        except Exception as e2:
                            raise RuntimeError(f"Both conversion and direct reading failed. "
                                            f"Conversion error: {e}. Reading error: {e2}")
                else:
                    # Direct SVO reading (will fail if ZED SDK not set up)
                    try:
                        return load_video_frames_from_svo(svo_file, camera_id)
                    except Exception as e:
                        raise FileNotFoundError(
                            f"MP4 not found and SVO reading failed: {e}\n"
                            f"  Options:\n"
                            f"  1. Convert SVO to MP4 first using conversion script\n"
                            f"  2. Run with --auto-convert-svo flag (requires ZED SDK)\n"
                            f"  3. Ensure MP4 files exist in {mp4_dir}"
                        )
    
    if video_path is None or not video_path.exists():
        raise FileNotFoundError(f"Video not found for camera {camera_id} in {recordings_dir}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
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
    print(f"  Loaded {len(frames)} frames from {camera_id} (left view only)")
    return frames


def resample_frames(frames, target_count):
    """Resample frames to target count using linear interpolation."""
    if len(frames) == 0:
        return frames
    
    if len(frames) == target_count:
        return frames
    
    # Create normalized time axis for original frames
    original_time = np.linspace(0, 1, len(frames))
    target_time = np.linspace(0, 1, target_count)
    
    # Resample each frame (interpolate pixel values)
    resampled = []
    frame_shape = frames[0].shape
    h, w, c = frame_shape
    
    # For each target time point, find interpolated frame
    for t in target_time:
        # Find surrounding frames
        idx = np.searchsorted(original_time, t)
        if idx == 0:
            resampled.append(frames[0])
        elif idx >= len(frames):
            resampled.append(frames[-1])
        else:
            # Linear interpolation between frames
            t0 = original_time[idx - 1]
            t1 = original_time[idx]
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            
            frame0 = frames[idx - 1].astype(np.float32)
            frame1 = frames[idx].astype(np.float32)
            interp_frame = (frame0 * (1 - alpha) + frame1 * alpha).astype(np.uint8)
            resampled.append(interp_frame)
    
    return resampled


def synchronize_videos(video_frames_list, sync_mode='normalized', target_frames=None):
    """
    Synchronize videos of different lengths.
    
    Args:
        video_frames_list: List of frame lists (one per experiment)
        sync_mode: 'normalized' (resample to common length), 'shortest' (truncate), 'pad' (pad with last frame)
        target_frames: Target frame count for normalized mode (default: shortest video length)
    """
    if not video_frames_list:
        return []
    
    lengths = [len(frames) for frames in video_frames_list]
    min_length = min(lengths)
    max_length = max(lengths)
    
    print(f"\nVideo lengths: {lengths} (min={min_length}, max={max_length})")
    
    if sync_mode == 'shortest':
        # Truncate all to shortest
        print(f"Truncating all videos to {min_length} frames")
        return [frames[:min_length] for frames in video_frames_list]
    
    elif sync_mode == 'pad':
        # Pad shorter videos with last frame
        print(f"Padding shorter videos to {max_length} frames")
        synchronized = []
        for frames in video_frames_list:
            if len(frames) < max_length:
                padded = frames + [frames[-1]] * (max_length - len(frames))
                synchronized.append(padded)
            else:
                synchronized.append(frames)
        return synchronized
    
    elif sync_mode == 'normalized':
        # Resample all to common length
        if target_frames is None:
            target_frames = min_length  # Default to shortest
        
        print(f"Resampling all videos to {target_frames} frames (normalized time)")
        synchronized = []
        for frames in video_frames_list:
            resampled = resample_frames(frames, target_frames)
            synchronized.append(resampled)
        return synchronized
    
    else:
        raise ValueError(f"Unknown sync_mode: {sync_mode}")


def resize_frame(frame, target_size):
    """Resize frame to target size maintaining aspect ratio, then pad if needed."""
    h, w = target_size
    frame_h, frame_w = frame.shape[:2]
    
    # Calculate scaling to fit within target size
    scale = min(w / frame_w, h / frame_h)
    new_w = int(frame_w * scale)
    new_h = int(frame_h * scale)
    
    # Resize
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to exact target size (center)
    padded = np.zeros((h, w, 3), dtype=resized.dtype)
    y_offset = (h - new_h) // 2
    x_offset = (w - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded


def create_grid_frame(video_frames_synced, experiment_names, frame_idx, grid_size=(640, 360)):
    """
    Create a single 2x2 grid frame from synchronized videos.
    
    Args:
        video_frames_synced: List of synchronized frame lists
        experiment_names: List of experiment names
        frame_idx: Current frame index
        grid_size: Size of each video quadrant (width, height)
    """
    w, h = grid_size
    grid_frame = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    
    # Layout: 2x2 grid
    # [0] [1]
    # [2] [3]
    positions = [
        (0, 0),      # Top-left
        (w, 0),      # Top-right
        (0, h),      # Bottom-left
        (w, h),      # Bottom-right
    ]
    
    for i, (frames, name) in enumerate(zip(video_frames_synced, experiment_names)):
        if frame_idx < len(frames):
            frame = frames[frame_idx]
            # Resize frame to grid size
            resized = resize_frame(frame, (h, w))
            
            # Place in grid
            x_offset, y_offset = positions[i]
            grid_frame[y_offset:y_offset+h, x_offset:x_offset+w] = resized
            
            # Add text label
            cv2.putText(grid_frame, name, (x_offset + 10, y_offset + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            # Frame index out of range - show black with label
            x_offset, y_offset = positions[i]
            cv2.putText(grid_frame, f"{name} (end)", (x_offset + 10, y_offset + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
    
    return grid_frame


def create_comparison_video(data_dir, experiments, output_path, camera_id='17225336', 
                           sync_mode='normalized', fps=30, auto_convert_svo=False):
    """
    Create synchronized comparison video from multiple experiments.
    
    Args:
        data_dir: Parent directory containing experiment subdirectories
        experiments: List of experiment directory names
        output_path: Output video file path
        camera_id: Camera ID to use for video
        sync_mode: Synchronization mode ('normalized', 'shortest', 'pad')
        fps: Output video frame rate
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    
    print("=" * 60)
    print("Creating Four-Way Video Comparison")
    print("=" * 60)
    print(f"Data dir: {data_dir}")
    print(f"Experiments: {experiments}")
    print(f"Output: {output_path}")
    print(f"Camera ID: {camera_id}")
    print(f"Sync mode: {sync_mode}")
    
    # Load all videos
    print("\n[1/3] Loading videos...")
    video_frames_list = []
    valid_experiments = []
    
    for exp_name in experiments:
        exp_dir = data_dir / exp_name
        recordings_dir = exp_dir / "recordings"
        
        if not recordings_dir.exists():
            print(f"  WARNING: {exp_name} - recordings directory not found, skipping")
            continue
        
        try:
            print(f"  Loading {exp_name}...")
            frames = load_video_frames(recordings_dir, camera_id, auto_convert_svo=auto_convert_svo)
            if frames:
                video_frames_list.append(frames)
                valid_experiments.append(exp_name)
        except Exception as e:
            print(f"  ERROR: {exp_name} - {e}")
            continue
    
    if len(video_frames_list) == 0:
        print("\n" + "=" * 60)
        print("ERROR: No videos loaded successfully")
        print("=" * 60)
        print("\nPossible solutions:")
        print("1. Convert SVO to MP4 first:")
        print("   python3 droid-multi-modal/scripts/visualizations/convert_svos_for_comparison.py \\")
        print("       --data-dir /path/to/data/success/2026-02-18")
        print("\n2. If ZED SDK is available, use auto-convert:")
        print("   python3 compare_videos.py --data-dir /path/to/data --auto-convert-svo")
        print("\n3. Ensure MP4 files exist in recordings/MP4/ directories")
        print("=" * 60)
        sys.exit(1)
    
    if len(video_frames_list) != len(experiments):
        print(f"WARNING: Only {len(video_frames_list)}/{len(experiments)} videos loaded")
    
    # Synchronize videos
    print("\n[2/3] Synchronizing videos...")
    video_frames_synced = synchronize_videos(video_frames_list, sync_mode=sync_mode)
    
    # Determine output video length
    output_length = max(len(frames) for frames in video_frames_synced)
    print(f"Output video length: {output_length} frames ({output_length/fps:.1f} seconds @ {fps} fps)")
    
    # Create video writer
    grid_w, grid_h = 1280, 720  # 2x2 grid: 640x360 each
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (grid_w, grid_h))
    
    if not out.isOpened():
        print(f"ERROR: Failed to create video writer for {output_path}")
        sys.exit(1)
    
    # Write frames
    print("\n[3/3] Writing video frames...")
    for frame_idx in range(output_length):
        if (frame_idx + 1) % 50 == 0:
            print(f"  Frame {frame_idx + 1}/{output_length}")
        
        grid_frame = create_grid_frame(video_frames_synced, valid_experiments, frame_idx)
        
        # Convert RGB to BGR for OpenCV
        grid_frame_bgr = cv2.cvtColor(grid_frame, cv2.COLOR_RGB2BGR)
        out.write(grid_frame_bgr)
    
    out.release()
    print(f"\nSaved comparison video to: {output_path}")
    print(f"  Resolution: {grid_w}x{grid_h}")
    print(f"  Length: {output_length} frames ({output_length/fps:.1f} seconds)")
    print(f"  Frame rate: {fps} fps")


def main():
    parser = argparse.ArgumentParser(description='Create four-way video comparison')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Parent directory containing experiment subdirectories')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (default: <data_dir>/comparison_video.mp4)')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                       help='Specific experiment names (default: all subdirectories)')
    parser.add_argument('--camera-id', type=str, default='17225336',
                       help='Camera ID to use (default: 17225336 for wrist camera)')
    parser.add_argument('--sync-mode', type=str, default='normalized',
                       choices=['normalized', 'shortest', 'pad'],
                       help='Synchronization mode: normalized (resample), shortest (truncate), pad (pad with last frame)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Output video frame rate (default: 30)')
    parser.add_argument('--auto-convert-svo', action='store_true',
                       help='Automatically convert SVO to MP4 if MP4 not found')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Find experiment directories
    if args.experiments:
        experiments = args.experiments
    else:
        experiments = [d.name for d in data_dir.iterdir() if d.is_dir()]
    
    # Filter to only directories with recordings
    valid_experiments = []
    for exp in experiments:
        exp_dir = data_dir / exp
        recordings_dir = exp_dir / "recordings"
        if recordings_dir.exists():
            valid_experiments.append(exp)
        else:
            print(f"WARNING: {exp} has no recordings directory, skipping")
    
    if len(valid_experiments) == 0:
        print(f"ERROR: No valid experiment directories found in {data_dir}")
        sys.exit(1)
    
    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / 'comparison_video.mp4'
    
    # Create comparison video
    create_comparison_video(
        data_dir=data_dir,
        experiments=valid_experiments,
        output_path=output_path,
        camera_id=args.camera_id,
        sync_mode=args.sync_mode,
        fps=args.fps,
        auto_convert_svo=getattr(args, 'auto_convert_svo', False)
    )


if __name__ == '__main__':
    main()
