#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-shot pipeline:
1) Sanitize path names (replace ':' -> '_') under data_dir
2) For each trajectory folder, move *.svo2/*.svo into recordings/SVO/
3) Convert SVO -> MP4 side-by-side + timestamps json in recordings/MP4/
4) Report corrupted trajectories.

Usage:
  python preprocess_svo_pipeline.py --data_dir /path/to/data/success/2026-01-06-infer

Optional:
  --no_rename        Disable renaming ':' -> '_'
  --dry_run          Print operations without changing files
  --fps 15           Output MP4 fps (default 15)
  --ext svo2         Extensions to handle (default: svo2,svo)
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
from tqdm import tqdm

from r2d2.camera_utils.recording_readers.svo_reader import SVOReader
from r2d2.data_loading.trajectory_sampler import collect_data_folderpaths


# -----------------------------
# 1) sanitize (rename) utilities
# -----------------------------
def sanitize_name(name: str) -> str:
    return name.replace(":", "_")


def rename_special_characters(root_dir: Path, dry_run: bool = False) -> None:
    """
    Recursively rename files/dirs by replacing ':' with '_'.
    Walk bottom-up so children are renamed before parents.
    """
    root_dir = root_dir.resolve()
    for current_root, dirs, files in os.walk(str(root_dir), topdown=False):
        current_root_p = Path(current_root)

        # rename files
        for file_name in files:
            new_name = sanitize_name(file_name)
            if new_name != file_name:
                old_path = current_root_p / file_name
                new_path = current_root_p / new_name
                print(f"[RENAME] file: {old_path} -> {new_path}")
                if not dry_run:
                    old_path.rename(new_path)

        # rename directories
        for dir_name in dirs:
            new_name = sanitize_name(dir_name)
            if new_name != dir_name:
                old_path = current_root_p / dir_name
                new_path = current_root_p / new_name
                print(f"[RENAME] dir : {old_path} -> {new_path}")
                if not dry_run:
                    old_path.rename(new_path)


# -----------------------------
# 2) SVO -> MP4 conversion
# -----------------------------
def convert_svo_to_mp4(
    svo_path: Path,
    recording_folderpath: Path,
    fps: int = 15,
    dry_run: bool = False,
) -> Tuple[Path, Path]:
    """
    Convert one SVO/SVO2 file to MP4 (side-by-side) + timestamps JSON.
    Returns (mp4_path, timestamps_path).
    """
    serial = svo_path.stem  # 17225336 from 17225336.svo2

    mp4_dir = recording_folderpath / "MP4"
    mp4_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = mp4_dir / f"{serial}.mp4"
    ts_path = mp4_dir / f"{serial}_timestamps.json"

    if dry_run:
        print(f"[DRY] convert {svo_path} -> {mp4_path} (+ {ts_path})")
        return mp4_path, ts_path

    camera = SVOReader(str(svo_path), serial_number=serial)
    camera.set_reading_parameters(image=True, depth=False, pointcloud=False, concatenate_images=True)
    width, height = camera.get_frame_resolution()

    video_codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc=video_codec, fps=fps, frameSize=(width * 2, height))

    frame_count = camera.get_frame_count()
    received_timestamps: List[int] = []

    for _ in range(frame_count):
        output = camera.read_camera(return_timestamp=True)
        if output is None:
            break
        data_dict, timestamp = output

        sbs_frame = data_dict["image"][serial]
        received_timestamps.append(timestamp)

        # BGRA -> BGR for mp4
        sbs_frame = cv2.cvtColor(sbs_frame, cv2.COLOR_BGRA2BGR)
        writer.write(sbs_frame)

    camera.disable_camera()
    writer.release()

    with open(ts_path, "w") as f:
        json.dump(received_timestamps, f)

    return mp4_path, ts_path


def is_mp4_readable(mp4_path: Path) -> bool:
    cap = cv2.VideoCapture(str(mp4_path))
    ok = cap.isOpened()
    cap.release()
    return ok


def find_svo_files(recordings_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    """
    Find svo files under recordings_dir (supports nested folders like recordings/SVO/*.svo2).
    """
    files: List[Path] = []
    for ext in exts:
        files.extend(recordings_dir.rglob(f"*.{ext}"))
    return sorted(files)


def move_svo_to_archive(recordings_dir: Path, exts: Tuple[str, ...], dry_run: bool = False) -> None:
    """
    Move recordings/*.svo2 (or *.svo) into recordings/SVO/.
    Also tolerant if they already are in recordings/SVO or deeper.
    """
    svo_archive = recordings_dir / "SVO"
    svo_archive.mkdir(parents=True, exist_ok=True)

    # only move files directly under recordings/ (not already nested)
    for ext in exts:
        for f in recordings_dir.glob(f"*.{ext}"):
            dest = svo_archive / f.name
            print(f"[MOVE] {f} -> {dest}")
            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dest))


def build_svo_by_serial(recordings_dir: Path, exts: Tuple[str, ...]) -> Dict[str, Path]:
    """
    Map serial -> svo_path by scanning recordings_dir recursively.
    Prefer shallowest path if duplicates exist.
    """
    candidates = find_svo_files(recordings_dir, exts)
    by_serial: Dict[str, Path] = {}
    for p in candidates:
        serial = p.stem
        if serial not in by_serial:
            by_serial[serial] = p
        else:
            # choose shallower path (shorter string length)
            if len(str(p)) < len(str(by_serial[serial])):
                by_serial[serial] = p
    return by_serial


# -----------------------------
# 3) per-trajectory processing
# -----------------------------
def process_one_trajectory(folderpath: Path, exts: Tuple[str, ...], fps: int, dry_run: bool) -> Tuple[bool, List[str]]:
    """
    Returns (ok, messages). ok=True means conversion counts look consistent.
    """
    msgs: List[str] = []
    recordings_dir = folderpath / "recordings"
    if not recordings_dir.exists():
        return True, [f"[SKIP] No recordings/ in {folderpath}"]

    (recordings_dir / "MP4").mkdir(parents=True, exist_ok=True)
    (recordings_dir / "SVO").mkdir(parents=True, exist_ok=True)

    # move stray svo files into recordings/SVO
    move_svo_to_archive(recordings_dir, exts, dry_run=dry_run)

    # build serial -> svo
    svo_by_serial = build_svo_by_serial(recordings_dir, exts)

    # gather existing mp4
    mp4_dir = recordings_dir / "MP4"
    mp4_paths = sorted(mp4_dir.glob("*.mp4"))
    mp4_by_serial = {p.stem: p for p in mp4_paths}

    files_to_convert: Set[Path] = set()

    # rule A: if mp4 missing OR timestamps missing -> convert from svo
    for serial, svo_path in svo_by_serial.items():
        mp4_path = mp4_by_serial.get(serial, mp4_dir / f"{serial}.mp4")
        ts_path = mp4_dir / f"{serial}_timestamps.json"
        if (not mp4_path.exists()) or (not ts_path.exists()):
            files_to_convert.add(svo_path)

    # rule B: mp4 exists but unreadable -> re-convert from svo if available
    for serial, mp4_path in mp4_by_serial.items():
        if not is_mp4_readable(mp4_path):
            if serial in svo_by_serial:
                files_to_convert.add(svo_by_serial[serial])
            else:
                msgs.append(f"[WARN] mp4 unreadable but missing svo source: {mp4_path}")

    # convert
    for svo_path in sorted(files_to_convert):
        msgs.append(f"[CONVERT] {svo_path}")
        try:
            convert_svo_to_mp4(svo_path, recordings_dir, fps=fps, dry_run=dry_run)
        except Exception as e:
            msgs.append(f"[ERROR] convert failed for {svo_path}: {e}")

    # final check: count mp4 vs svo (by serial)
    mp4_count = len(list(mp4_dir.glob("*.mp4")))
    svo_count = len(svo_by_serial)

    ok = (mp4_count >= svo_count)
    if not ok:
        msgs.append(f"[CORRUPT?] svo_count={svo_count} mp4_count={mp4_count}")
    return ok, msgs


# -----------------------------
# 4) main entry
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Root directory containing many trajectory folders.")
    p.add_argument("--no_rename", action="store_true", help="Disable renaming ':' -> '_'")
    p.add_argument("--dry_run", action="store_true", help="Print operations without modifying files")
    p.add_argument("--fps", type=int, default=15, help="Output mp4 fps (default 15)")
    p.add_argument("--ext", type=str, default="svo2,svo", help="Comma-separated extensions to treat as SVO files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"[FAIL] data_dir does not exist: {data_dir}")

    exts = tuple([e.strip().lstrip(".") for e in args.ext.split(",") if e.strip()])
    print(f"[INFO] data_dir={data_dir}")
    print(f"[INFO] exts={exts}  fps={args.fps}  dry_run={args.dry_run}  rename={not args.no_rename}")

    # Step 1: sanitize names
    if not args.no_rename:
        rename_special_characters(data_dir, dry_run=args.dry_run)

    # Step 2: collect trajectory folders
    # collect_data_folderpaths expects the folder structure used by r2d2.
    # It returns a list[str], we cast to Path.
    all_folderpaths = [Path(p) for p in collect_data_folderpaths(data_dir=str(data_dir))]

    corrupted: List[Path] = []
    for folderpath in tqdm(all_folderpaths, desc="Trajectories"):
        ok, msgs = process_one_trajectory(folderpath, exts=exts, fps=args.fps, dry_run=args.dry_run)
        for m in msgs:
            print(m)
        if not ok:
            corrupted.append(folderpath)

    print("\n=== Summary ===")
    print(f"Total trajectories: {len(all_folderpaths)}")
    print(f"Corrupted(?)      : {len(corrupted)}")
    if corrupted:
        print("The following trajectories are corrupted (mp4_count < svo_count):")
        for p in corrupted:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
