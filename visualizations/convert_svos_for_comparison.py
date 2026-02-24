#!/usr/bin/env python3
"""
Convert SVO files to MP4 for video comparison.
Converts wrist camera (17225336) SVO files in the specified data directory.
"""

import sys
from pathlib import Path

# Add droid-multi-modal to path
script_dir = Path(__file__).parent
droid_root = script_dir.parent.parent
if str(droid_root) not in sys.path:
    sys.path.insert(0, str(droid_root))

import cv2
import json
from tqdm import tqdm
from r2d2.camera_utils.recording_readers.svo_reader import SVOReader


def convert_svo_to_mp4(svo_path, recording_dir, camera_id):
    """Convert a single SVO file to MP4."""
    try:
        camera = SVOReader(str(svo_path), serial_number=camera_id)
        camera.set_reading_parameters(image=True, depth=False, pointcloud=False, concatenate_images=True)
        
        try:
            width, height = camera.get_frame_resolution()
        except AttributeError:
            print(f"  Warning: Could not get resolution, using defaults")
            width, height = 1280, 720
        
        # Create MP4 directory
        mp4_dir = recording_dir / "MP4"
        mp4_dir.mkdir(parents=True, exist_ok=True)
        
        mp4_path = mp4_dir / f"{camera_id}.mp4"
        ts_path = mp4_dir / f"{camera_id}_timestamps.json"
        
        # Skip if already exists
        if mp4_path.exists():
            print(f"  MP4 already exists: {mp4_path.name}")
            return True
        
        # Setup VideoWriter
        video_codec = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(mp4_path), video_codec, 15, (width * 2, height))
        
        if not video_writer.isOpened():
            print(f"  ERROR: Failed to create video writer")
            return False
        
        # Convert frames
        frame_count = camera.get_frame_count()
        received_timestamps = []
        
        print(f"  Converting {frame_count} frames...")
        for _ in tqdm(range(frame_count), desc=f"    {mp4_path.name}", leave=False):
            output = camera.read_camera(return_timestamp=True)
            if output is None:
                break
            data_dict, timestamp = output
            sbs_frame = data_dict["image"][camera_id]
            sbs_frame = cv2.cvtColor(sbs_frame, cv2.COLOR_BGRA2BGR)
            received_timestamps.append(timestamp)
            video_writer.write(sbs_frame)
        
        camera.disable_camera()
        video_writer.release()
        
        # Save timestamps
        with open(ts_path, "w") as f:
            json.dump(received_timestamps, f)
        
        print(f"  ✓ Converted: {mp4_path.name} ({frame_count} frames)")
        return True
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert SVO files to MP4 for comparison')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Parent directory containing experiment subdirectories')
    parser.add_argument('--camera-id', type=str, default='17225336',
                       help='Camera ID to convert (default: 17225336 for wrist camera)')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    camera_id = args.camera_id
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Find all experiment directories
    experiments = [d for d in data_dir.iterdir() if d.is_dir()]
    
    print("=" * 60)
    print("Converting SVO files to MP4")
    print("=" * 60)
    print(f"Data dir: {data_dir}")
    print(f"Camera ID: {camera_id}")
    print(f"Found {len(experiments)} experiments\n")
    
    success_count = 0
    for exp_dir in experiments:
        exp_name = exp_dir.name
        recordings_dir = exp_dir / "recordings"
        svo_dir = recordings_dir / "SVO"
        
        if not svo_dir.exists():
            print(f"{exp_name}: No SVO directory, skipping")
            continue
        
        # Find SVO file (try .svo2 first, then .svo)
        svo_file = svo_dir / f"{camera_id}.svo2"
        if not svo_file.exists():
            svo_file = svo_dir / f"{camera_id}.svo"
        
        if not svo_file.exists():
            print(f"{exp_name}: No SVO file for camera {camera_id}, skipping")
            continue
        
        print(f"{exp_name}:")
        if convert_svo_to_mp4(svo_file, recordings_dir, camera_id):
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"Conversion complete: {success_count}/{len(experiments)} successful")
    print("=" * 60)


if __name__ == '__main__':
    main()
