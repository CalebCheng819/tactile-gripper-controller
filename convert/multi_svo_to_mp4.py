import os
import glob
import json
import cv2
from tqdm import tqdm
from r2d2.camera_utils.recording_readers.svo_reader import SVOReader

# === Modify this to point to your actual root folder ===
ROOTS = ["data/success/recollected_data"]
    
# ROOTS = [
#     os.path.expanduser("~/code/droid_mingyo/data/success/test_with_buffer")
# ]



def convert_svo_to_mp4(filepath, recording_folderpath):
    serial_number = os.path.basename(filepath)[:-4]  # remove .svo
    camera = SVOReader(filepath, serial_number=serial_number)
    camera.set_reading_parameters(image=True, depth=False, pointcloud=False, concatenate_images=True)
    # width, height = camera.get_frame_resolution()
    try:
        width, height = camera.get_frame_resolution()
    except AttributeError:
        print(f"Warning: Falling back to default resolution for {filepath}")


    # Create output folders
    mp4_dir = os.path.join(recording_folderpath, "MP4")
    os.makedirs(mp4_dir, exist_ok=True)
    video_output_path = os.path.join(mp4_dir, serial_number + ".mp4")
    timestamp_output_path = video_output_path[:-4] + "_timestamps.json"

    # Setup VideoWriter
    video_codec = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_output_path, fourcc=video_codec, fps=15, frameSize=(width * 2, height))

    # Write frames
    received_timestamps = []
    for _ in range(camera.get_frame_count()):
        output = camera.read_camera(return_timestamp=True)
        if output is None:
            break
        data_dict, timestamp = output
        sbs_frame = data_dict["image"][serial_number]
        sbs_frame = cv2.cvtColor(sbs_frame, cv2.COLOR_BGRA2BGR)
        received_timestamps.append(timestamp)
        video_writer.write(sbs_frame)

    camera.disable_camera()
    video_writer.release()

    with open(timestamp_output_path, "w") as f:
        json.dump(received_timestamps, f)


# === Crawl and convert ===
svo_files = []
for root in ROOTS:
    svo_files += glob.glob(os.path.join(root, "*", "recordings", "SVO", "*.svo"))

print(f"Found {len(svo_files)} .svo files.")

for svo_path in tqdm(svo_files):
    # folder = os.path.dirname(os.path.dirname(svo_path))
    folder = os.path.abspath(os.path.join(svo_path, "../../.."))  
    try:
        convert_svo_to_mp4(svo_path, folder)
    except Exception as e:
        print(f"Error converting {svo_path}: {e}")
