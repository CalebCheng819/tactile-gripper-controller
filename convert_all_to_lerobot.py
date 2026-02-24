#!/usr/bin/env python3
"""
One-shot pipeline: raw episodes -> LeRobot-style dataset folders.

Expected raw layout:
RAW_ROOT/
  Fri_Jan_23_16:02:01_2026/
    trajectory.h5
    recordings/
      17225336.svo2
      24013089.svo2
      24395123.svo2
  Fri_Jan_23_16:10:05_2026/
    ...

Output layout:
OUT_ROOT/
  Fri_Jan_23_16:02:01_2026/
    meta/{info.json, episodes.jsonl}
    data/chunk-000/file-000.parquet
    videos/observation.images.<serial>_left/chunk-000/file-000.mp4
    videos/observation.images.<serial>_right/chunk-000/file-000.mp4
    aux/frames_<serial>.csv
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import cv2
import h5py
import pandas as pd

# ZED
import pyzed.sl as sl
from typing import Optional



EP_NAME_RE = re.compile(r"^[A-Za-z]{3}_[A-Za-z]{3}__?\d{1,2}_\d{2}:\d{2}:\d{2}_\d{4}$")
# your example: Fri_Jan_23_16:02:01_2026  (one underscore between Jan and 23)
# some systems might produce Fri_Jan__23... (double underscore) -> accept both


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------
# 1) Export SVO2 -> left/right mp4 + frame timestamps CSV
# ---------------------------
def export_svo2_to_mp4_and_timestamps(
    svo_path: Path,
    out_left_mp4: Path,
    out_right_mp4: Path,
    out_csv: Path,
    resize_w: int = 0,
    resize_h: int = 0,
    fps_override: float = 0.0,
):
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO {svo_path}: {status}")

    cam_info = cam.get_camera_information()
    src_w = cam_info.camera_configuration.resolution.width
    src_h = cam_info.camera_configuration.resolution.height

    out_w = resize_w if resize_w > 0 else src_w
    out_h = resize_h if resize_h > 0 else src_h

    fps = fps_override if fps_override > 0 else float(cam_info.camera_configuration.fps)
    if fps <= 0:
        fps = 30.0

    ensure_dir(out_left_mp4.parent)
    ensure_dir(out_right_mp4.parent)
    ensure_dir(out_csv.parent)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_left = cv2.VideoWriter(str(out_left_mp4), fourcc, fps, (out_w, out_h))
    vw_right = cv2.VideoWriter(str(out_right_mp4), fourcc, fps, (out_w, out_h))

    mat_left = sl.Mat()
    mat_right = sl.Mat()
    rows = []
    frame_idx = 0

    try:
        while True:
            err = cam.grab()
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if err != sl.ERROR_CODE.SUCCESS:
                continue

            cam.retrieve_image(mat_left, sl.VIEW.LEFT)
            cam.retrieve_image(mat_right, sl.VIEW.RIGHT)

            ts = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)
            ts_ns = int(ts.get_nanoseconds())

            left = mat_left.get_data()
            right = mat_right.get_data()

            # BGRA->BGR for mp4 writer
            if left.shape[2] == 4:
                left_bgr = cv2.cvtColor(left, cv2.COLOR_BGRA2BGR)
            else:
                left_bgr = left
            if right.shape[2] == 4:
                right_bgr = cv2.cvtColor(right, cv2.COLOR_BGRA2BGR)
            else:
                right_bgr = right

            if (left_bgr.shape[1], left_bgr.shape[0]) != (out_w, out_h):
                left_bgr = cv2.resize(left_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
            if (right_bgr.shape[1], right_bgr.shape[0]) != (out_w, out_h):
                right_bgr = cv2.resize(right_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)

            vw_left.write(left_bgr)
            vw_right.write(right_bgr)

            rows.append((frame_idx, ts_ns))
            frame_idx += 1
    finally:
        vw_left.release()
        vw_right.release()
        cam.close()

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "timestamp_ns"])
        w.writerows(rows)

    return len(rows), fps


def load_frames_csv(csv_path: Path) -> np.ndarray:
    ts = []
    with csv_path.open("r") as f:
        r = csv.DictReader(f)
        for row in r:
            ts.append(int(row["timestamp_ns"]))
    if not ts:
        raise RuntimeError(f"Empty frames csv: {csv_path}")
    arr = np.asarray(ts, dtype=np.int64)
    # ensure sorted
    if not np.all(arr[:-1] <= arr[1:]):
        arr = np.sort(arr)
    return arr


def align_timestamps_to_frames(step_ts_ns: np.ndarray, frame_ts_ns: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(frame_ts_ns, step_ts_ns, side="left")
    idx0 = np.clip(idx - 1, 0, len(frame_ts_ns) - 1)
    idx1 = np.clip(idx, 0, len(frame_ts_ns) - 1)
    d0 = np.abs(frame_ts_ns[idx0] - step_ts_ns)
    d1 = np.abs(frame_ts_ns[idx1] - step_ts_ns)
    best = np.where(d1 < d0, idx1, idx0).astype(np.int32)
    return best


def as_f32(x):
    return np.asarray(x, dtype=np.float32)


# ---------------------------
# 2) Convert trajectory.h5 -> parquet (with tactile window + aligned frame_idx)
# ---------------------------
def convert_h5_to_parquet(
    h5_path: Path,
    out_root: Path,
    serials: list[str],
    camera_ts_field: str = "estimated_capture",  # or frame_received
):
    out_parquet = out_root / "data" / "chunk-000" / "file-000.parquet"
    ensure_dir(out_parquet.parent)

    # Load per-camera video timestamps from aux (exported from svo2)
    frame_ts = {}
    for s in serials:
        csv_path = out_root / "aux" / f"frames_{s}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing frames csv: {csv_path}")
        frame_ts[s] = load_frames_csv(csv_path)

    with h5py.File(h5_path, "r") as f:
        step_start_ns = np.asarray(f["observation/timestamp/control/step_start"], dtype=np.int64)
        T = step_start_ns.shape[0]

        joint_pos = as_f32(f["observation/robot_state/joint_positions"][...])      # (T,7)
        joint_vel = as_f32(f["observation/robot_state/joint_velocities"][...])     # (T,7)
        cart_pos = as_f32(f["observation/robot_state/cartesian_position"][...])    # (T,6)
        grip_pos = as_f32(f["observation/robot_state/gripper_position"][...])      # (T,)
        human_int = np.asarray(f["observation/robot_state/human_intervention"][...], dtype=bool)

        tactile = as_f32(f["observation/robot_state/tactile_values"][...])         # (T,500,6)
        force = as_f32(f["observation/robot_state/force_prediction"][...])         # (T,500,1)

        tactile_flat = tactile.reshape(T, -1)                                      # (T,3000)
        force_flat = force.reshape(T, -1)                                          # (T,500)

        act_jv = as_f32(f["action/joint_velocity"][...])                           # (T,7)
        act_gv = as_f32(f["action/gripper_velocity"][...])                         # (T,)

        cam_step_ts = {}
        for s in serials:
            key = f"observation/timestamp/cameras/{s}_{camera_ts_field}"
            if key not in f:
                raise KeyError(f"Missing h5 camera timestamp dataset: {key}")
            cam_step_ts[s] = np.asarray(f[key][...], dtype=np.int64)

    cam_frame_idx = {}
    for s in serials:
        cam_frame_idx[s] = align_timestamps_to_frames(cam_step_ts[s], frame_ts[s])

    data = {
        "episode_index": np.zeros((T,), dtype=np.int32),
        "frame_index": np.arange(T, dtype=np.int32),
        "timestamp.control.step_start_ns": step_start_ns,

        "observation.state.joint_positions": list(joint_pos),
        "observation.state.joint_velocities": list(joint_vel),
        "observation.state.cartesian_position": list(cart_pos),
        "observation.state.gripper_position": grip_pos,
        "observation.state.human_intervention": human_int,

        # flatten windows
        "observation.state.tactile_window": list(tactile_flat),  # (3000,) -> reshape (500,6)
        "observation.state.force_window": list(force_flat),      # (500,)  -> reshape (500,1)

        "action.joint_velocity": list(act_jv),
        "action.gripper_velocity": act_gv,
    }

    for s in serials:
        data[f"observation.image_frame.{s}_left"] = cam_frame_idx[s]
        data[f"observation.image_frame.{s}_right"] = cam_frame_idx[s]
        data[f"timestamp.camera.{s}_{camera_ts_field}_ns"] = cam_step_ts[s]

    df = pd.DataFrame(data)
    df.to_parquet(out_parquet, engine="pyarrow", compression="zstd", index=False)
    return T, out_parquet


# ---------------------------
# 3) Build meta: episodes.jsonl + info.json
# ---------------------------
def build_meta(out_root: Path, T: int, serials: list[str]):
    meta_dir = out_root / "meta"
    ensure_dir(meta_dir)

    episodes_path = meta_dir / "episodes.jsonl"
    with episodes_path.open("w") as f:
        f.write(json.dumps({
            "episode_index": 0,
            "start_index": 0,
            "end_index": T - 1,
            "length": T,
        }) + "\n")

    video_keys = []
    for s in serials:
        video_keys.append(f"observation.images.{s}_left")
        video_keys.append(f"observation.images.{s}_right")

    info = {
        "format": "lerobot",
        "version": "0.1",
        "num_episodes": 1,
        "num_frames": T,
        "timestamp_unit": "ns",
        "video_keys": video_keys,
        "state_keys": [
            "observation.state.joint_positions",
            "observation.state.joint_velocities",
            "observation.state.cartesian_position",
            "observation.state.gripper_position",
            "observation.state.human_intervention",
            "observation.state.tactile_window",  # (3000,) -> (500,6)
            "observation.state.force_window",    # (500,)  -> (500,1)
        ],
        "action_keys": [
            "action.joint_velocity",      # (7,)
            "action.gripper_velocity",    # scalar
        ],
        "notes": [
            "tactile_window is flattened: (500,6) -> (3000,)",
            "force_window is flattened: (500,1) -> (500,)",
            "observation.image_frame.<serial>_{left,right} stores aligned frame indices into corresponding mp4.",
        ],
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

def find_svo_file(rec_dir: Path, serial: str) -> Optional[Path]:

    # Most common: <serial>.svo2 or <serial>.svo
    cand = [
        rec_dir / f"{serial}.svo2",
        rec_dir / f"{serial}.svo",
    ]
    for p in cand:
        if p.exists():
            return p

    # fallback: any file containing serial and ending with .svo/.svo2
    hits = []
    for ext in ("*.svo2", "*.svo"):
        hits.extend(sorted(rec_dir.glob(ext)))
    for p in hits:
        if serial in p.name:
            return p

    return None

# ---------------------------
# 4) One episode end-to-end
# ---------------------------
def process_one_episode(ep_dir: Path, out_root: Path, serials: list[str], camera_ts_field: str,
                        resize_w: int, resize_h: int, fps_override: float, overwrite: bool):
    h5_path = ep_dir / "trajectory.h5"
    rec_dir = ep_dir / "recordings" / "SVO"


    if not h5_path.exists():
        return False, f"missing trajectory.h5"
    if not rec_dir.exists():
        return False, f"missing recordings/"

    # Prepare output episode folder
    out_ep = out_root / ep_dir.name
    ensure_dir(out_ep)

    # If already converted and not overwrite, skip
    out_parquet = out_ep / "data" / "chunk-000" / "file-000.parquet"
    if out_parquet.exists() and not overwrite:
        return True, f"skip (exists): {out_parquet}"

    # Export each serial
    for s in serials:
        svo_path = find_svo_file(rec_dir, s)
        if svo_path is None:
            return False, f"missing svo/svo2 for serial={s} in {rec_dir}"


        left_key = f"observation.images.{s}_left"
        right_key = f"observation.images.{s}_right"
        out_left = out_ep / "videos" / left_key / "chunk-000" / "file-000.mp4"
        out_right = out_ep / "videos" / right_key / "chunk-000" / "file-000.mp4"
        out_csv = out_ep / "aux" / f"frames_{s}.csv"

        # Export videos/csv (overwrite if requested)
        if overwrite or (not out_left.exists()) or (not out_right.exists()) or (not out_csv.exists()):
            nframes, fps = export_svo2_to_mp4_and_timestamps(
                svo_path=svo_path,
                out_left_mp4=out_left,
                out_right_mp4=out_right,
                out_csv=out_csv,
                resize_w=resize_w,
                resize_h=resize_h,
                fps_override=fps_override,
            )
        else:
            # still ensure csv exists
            nframes = len(load_frames_csv(out_csv))
            fps = None

    # Convert h5 -> parquet
    T, pq = convert_h5_to_parquet(
        h5_path=h5_path,
        out_root=out_ep,
        serials=serials,
        camera_ts_field=camera_ts_field,
    )

    # Meta
    build_meta(out_ep, T=T, serials=serials)

    # Quick sanity check: frame idx range
    df = pd.read_parquet(pq)
    for s in serials:
        idx = df[f"observation.image_frame.{s}_left"].to_numpy()
        if idx.min() < 0:
            return False, f"bad frame idx (<0) for serial {s}"
    return True, f"ok (steps={T}) -> {out_ep}"


# ---------------------------
# 5) Main: scan all episodes
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, required=True, help="root folder containing episode dirs")
    ap.add_argument("--out_root", type=str, required=True, help="output root folder for lerobot episodes")
    ap.add_argument("--serials", type=str, default="17225336,24013089,24395123", help="comma-separated serials")
    ap.add_argument("--camera_ts_field", type=str, default="estimated_capture",
                    choices=["estimated_capture", "frame_received"])
    ap.add_argument("--resize_w", type=int, default=0, help="resize video width (0=keep original)")
    ap.add_argument("--resize_h", type=int, default=0, help="resize video height (0=keep original)")
    ap.add_argument("--fps", type=float, default=0.0, help="override fps (0=use SVO metadata)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    ap.add_argument("--only", type=str, default="", help="process only one episode folder name")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    serials = [s.strip() for s in args.serials.split(",") if s.strip()]

    # Collect episodes
    episodes = []
    for p in sorted(raw_root.iterdir()):
        if not p.is_dir():
            continue
        if args.only and p.name != args.only:
            continue
        # If you want strict name matching, enable this:
        # if not EP_NAME_RE.match(p.name):
        #     continue
        episodes.append(p)

    if not episodes:
        print(f"[ERROR] No episode folders found under: {raw_root}")
        return

    ok_count = 0
    for ep in episodes:
        try:
            ok, msg = process_one_episode(
                ep_dir=ep,
                out_root=out_root,
                serials=serials,
                camera_ts_field=args.camera_ts_field,
                resize_w=args.resize_w,
                resize_h=args.resize_h,
                fps_override=args.fps,
                overwrite=args.overwrite,
            )
            if ok:
                ok_count += 1
                print(f"[OK] {ep.name}: {msg}")
            else:
                print(f"[FAIL] {ep.name}: {msg}")
        except Exception as e:
            print(f"[EXCEPTION] {ep.name}: {e}")

    print(f"\nDone. success={ok_count}/{len(episodes)}  out_root={out_root}")


if __name__ == "__main__":
    main()
