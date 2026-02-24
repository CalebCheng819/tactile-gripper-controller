import glob
import json
import os

import cv2
from tqdm import tqdm

from r2d2.camera_utils.recording_readers.svo_reader import SVOReader
from r2d2.data_loading.trajectory_sampler import collect_data_folderpaths

def convert_svo_to_mp4(svo_path, mp4_dir):
    serial_number = os.path.basename(svo_path)[:-4]
    camera = SVOReader(svo_path, serial_number=serial_number)
    camera.set_reading_parameters(image=True, depth=False, pointcloud=False, concatenate_images=True)
    # # width, height = camera.get_frame_resolution()

    # # Match with svo_reader.py in droid_mingyo --> changed to ac
    # width = camera_info.camera_resolution.width
    # height = camera_info.camera_resolution.height

    width, height = camera.get_frame_resolution()

    mp4_path = os.path.join(mp4_dir, serial_number + ".mp4")
    ts_path = mp4_path[:-4] + "_timestamps.json"
    video_writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (width * 2, height))
    timestamps = []
    for _ in range(camera.get_frame_count()):
        output = camera.read_camera(return_timestamp=True)
        if output is None:
            break
        data_dict, timestamp = output
        frame = data_dict["image"][serial_number]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        video_writer.write(frame)
        timestamps.append(timestamp)
    camera.disable_camera()
    video_writer.release()
    with open(ts_path, "w") as f:
        json.dump(timestamps, f)

base_dir = os.path.expanduser("~/code/droid_mingyo/data/success/pi0@100.75.164.93")
for root, dirs, files in os.walk(base_dir):
    if "SVO" in dirs:
        svo_dir = os.path.join(root, "SVO")
        mp4_dir = os.path.join(root, "MP4")
        os.makedirs(mp4_dir, exist_ok=True)
        svo_files = glob.glob(os.path.join(svo_dir, "*.svo"))
        for svo_file in svo_files:
            print(f"Converting: {svo_file}")
            convert_svo_to_mp4(svo_file, mp4_dir)


# import os
# from pathlib import Path
# import cv2
# import pyzed.sl as sl
# from tqdm import tqdm

# def export_mp4(svo_file: Path, mp4_dir: Path, stereo_view: str = "left", show_progress: bool = False) -> bool:
#     mp4_out = mp4_dir / f"{svo_file.stem}_{stereo_view}.mp4"
#     sdk_version = sl.Camera().get_sdk_version()
#     use_sdk_4 = sdk_version.startswith("4.0")

#     if not (sdk_version.startswith("4.0") or sdk_version.startswith("3.8")):
#         print(f"❌ Unsupported SDK version: {sdk_version}")
#         return False

#     init_params = sl.InitParameters()
#     init_params.set_from_svo_file(str(svo_file))
#     init_params.svo_real_time_mode = False
#     init_params.coordinate_units = sl.UNIT.MILLIMETER
#     init_params.camera_image_flip = sl.FLIP_MODE.OFF

#     zed = sl.Camera()
#     err = zed.open(init_params)
#     if err != sl.ERROR_CODE.SUCCESS:
#         print(f"❌ Failed to open SVO file: {svo_file.name}")
#         zed.close()
#         return False

#     if use_sdk_4:
#         info = zed.get_camera_information().camera_configuration
#         fps = info.fps
#         width, height = info.resolution.width, info.resolution.height
#     else:
#         info = zed.get_camera_information()
#         fps = info.camera_fps
#         width, height = info.camera_resolution.width, info.camera_resolution.height

#     img_container = sl.Mat()
#     video_writer = cv2.VideoWriter(
#         str(mp4_out),
#         cv2.VideoWriter_fourcc(*"mp4v"),
#         fps,
#         (width, height),
#     )

#     if not video_writer.isOpened():
#         print(f"❌ Failed to open VideoWriter for: {mp4_out}")
#         zed.close()
#         return False

#     n_frames = zed.get_svo_number_of_frames()
#     rt_params = sl.RuntimeParameters()
#     if show_progress:
#         pbar = tqdm(total=n_frames, desc=f"Exporting {svo_file.name}", leave=False)

#     while True:
#         grabbed = zed.grab(rt_params)
#         if grabbed == sl.ERROR_CODE.SUCCESS or (use_sdk_4 and grabbed == sl.ERROR_CODE.END_OF_SVOFILE_REACHED):
#             svo_pos = zed.get_svo_position()
#             view = {"left": sl.VIEW.LEFT, "right": sl.VIEW.RIGHT}[stereo_view]
#             zed.retrieve_image(img_container, view)
#             rgb = cv2.cvtColor(img_container.get_data(), cv2.COLOR_RGBA2RGB)
#             video_writer.write(rgb)

#             if show_progress:
#                 pbar.update()

#             if svo_pos >= (n_frames - 1) or (use_sdk_4 and grabbed == sl.ERROR_CODE.END_OF_SVOFILE_REACHED):
#                 break
#         else:
#             break

#     video_writer.release()
#     zed.close()
#     if show_progress:
#         pbar.close()

#     return True

# def batch_convert_all_svo(root_dir: Path):
#     for svo_dir in root_dir.rglob("recordings/SVO"):
#         mp4_dir = svo_dir.parent / "MP4"
#         mp4_dir.mkdir(exist_ok=True)
#         print(f"\n📁 處理資料夾: {svo_dir}")

#         for svo_file in svo_dir.glob("*.svo"):
#             print(f"   🎞️ 左畫面轉換: {svo_file.name}")
#             export_mp4(svo_file, mp4_dir, stereo_view="left", show_progress=True)

#             print(f"   🎞️ 右畫面轉換: {svo_file.name}")
#             export_mp4(svo_file, mp4_dir, stereo_view="right", show_progress=True)

# if __name__ == "__main__":
#     root_path = Path("~/code/droid_mingyo/data/success/test_with_buffer").expanduser()
#     batch_convert_all_svo(root_path)
