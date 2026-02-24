#!/usr/bin/env python3

import glob
import json
import os
import cv2
from pathlib import Path
from tqdm import tqdm

from r2d2.camera_utils.recording_readers.svo_reader import SVOReader


def convert_svo_to_mp4(filepath, recording_folderpath):
    """Convert a single SVO file to MP4."""
    try:
        serial_number = filepath.split("/")[-1][:-4]
        camera = SVOReader(filepath, serial_number=serial_number)
        camera.set_reading_parameters(image=True, depth=False, pointcloud=False, concatenate_images=True)
        width, height = camera.get_frame_resolution()

        # Create MP4 Writer
        video_output_path = os.path.join(recording_folderpath, "MP4", serial_number + ".mp4")
        timestamp_output_path = video_output_path[:-4] + "_timestamps.json"
        video_codec = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_output_path, fourcc=video_codec, fps=15, frameSize=(width * 2, height))

        # Convert To MP4
        frame_count = camera.get_frame_count()
        received_timestamps = []

        for _i in range(frame_count):
            output = camera.read_camera(return_timestamp=True)
            if output is None:
                break
            else:
                data_dict, timestamp = output

            sbs_frame = data_dict["image"][serial_number]
            received_timestamps.append(timestamp)
            sbs_frame = cv2.cvtColor(sbs_frame, cv2.COLOR_BGRA2BGR)
            video_writer.write(sbs_frame)

        # Close Everything
        camera.disable_camera()
        video_writer.release()

        with open(timestamp_output_path, "w") as jsonFile:
            json.dump(received_timestamps, jsonFile)
        
        return True
    except Exception as e:
        print(f"    ❌ ERROR converting {filepath}: {str(e)}")
        return False


def check_and_convert_svo(root_dir, output_file="conversion_log.txt"):
    """
    Check for unconverted SVO files and convert them to MP4.
    Only print directories with issues or unconverted files.
    Save output to a text file.
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        msg = f"Error: Directory '{root_dir}' does not exist."
        print(msg)
        return
    
    corrupted_traj = []
    
    # Open log file for writing
    with open(output_file, "w") as log:
        log_msg = f"Conversion Log - Started\n{'='*60}\n"
        log.write(log_msg)
        print(log_msg, end="")
        
        # Find all directories with recordings
        for dir_path in sorted(root_path.rglob("*")):
            if not dir_path.is_dir():
                continue
            
            if dir_path.name != "recordings":
                continue
            
            recording_folderpath = str(dir_path)
            msg = f"Checking: {recording_folderpath}\n"
            log.write(msg)
            print(msg, end="")
            mp4_folderpath = os.path.join(recording_folderpath, "MP4")
            svo_folderpath = os.path.join(recording_folderpath, "SVO")
            
            if not os.path.exists(mp4_folderpath):
                os.makedirs(mp4_folderpath)
            if not os.path.exists(svo_folderpath):
                os.makedirs(svo_folderpath)

            # Move Files To New Location
            svo_files_to_move = glob.glob(recording_folderpath + "/*.svo2")
            for f in svo_files_to_move:
                path_list = f.split("/")
                path_list.insert(len(path_list) - 1, "SVO")
                new_f = "/".join(path_list)
                os.rename(f, new_f)

            # Gather Files To Convert
            svo_filepaths = glob.glob(svo_folderpath + "/*.svo2")
            mp4_filepaths = glob.glob(mp4_folderpath + "/*.mp4")
            files_to_convert = []
            
            for f in svo_filepaths:
                serial_number = f.split("/")[-1][:-4]
                if not any([serial_number in f for f in mp4_filepaths]):
                    files_to_convert.append(f)

            # Check MP4 integrity (but don't re-convert them)
            for f in mp4_filepaths:
                timestamp_filepath = f[:-4] + "_timestamps.json"
                reader = cv2.VideoCapture(f)
                if not reader.isOpened():
                    # MP4 is corrupted, mark for re-conversion
                    serial_number = f.split("/")[-1][:-4]
                    svo_file = os.path.join(svo_folderpath, serial_number + ".svo2")
                    if os.path.exists(svo_file):
                        files_to_convert.append(svo_file)
                reader.release()

            # Convert Files
            if files_to_convert:
                msg = f"Converting files in: {recording_folderpath}\n"
                log.write(msg)
                print(msg, end="")
                for f in files_to_convert:
                    msg = f"  Converting: {f}\n"
                    log.write(msg)
                    print(msg, end="")
                    convert_svo_to_mp4(f, recording_folderpath)

            # Check Success
            num_mp4 = len(glob.glob(mp4_folderpath + "/*.mp4"))
            num_svo = len(svo_filepaths)

            if num_svo > num_mp4:
                msg = f"⚠️  ISSUE: {recording_folderpath}\n   SVO files: {num_svo}, MP4 files: {num_mp4}\n"
                log.write(msg)
                print(msg, end="")
                corrupted_traj.append(recording_folderpath)

            log.flush()

        if corrupted_traj:
            final_msg = "\n" + "="*60 + "\nThe following trajectories have issues:\n"
            for folderpath in corrupted_traj:
                final_msg += f"  - {folderpath}\n"
            log.write(final_msg)
            print(final_msg, end="")
        else:
            final_msg = "\n" + "="*60 + "\n✓ All files converted successfully!\n"
            log.write(final_msg)
            print(final_msg, end="")
        
        log.write("\n" + "="*60 + "\nConversion Log - Completed\n")


if __name__ == "__main__":
    # Hardcode the path here
    input_dir = "/media/pi0/D8685FAD685F88E0/2025-10-17"
    output_log = "conversion_log.txt"
    
    check_and_convert_svo(input_dir, output_file=output_log)
    print(f"\n✓ Log saved to: {output_log}")