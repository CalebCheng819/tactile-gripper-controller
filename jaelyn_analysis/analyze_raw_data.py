#!/usr/bin/env python3
"""
Analyze a raw DROID multi-modal capture (trajectory + recordings).

Reports:
  * Robot-state/observation frequency.
  * Control/action loop frequency.
  * Per-camera read frequency (when timestamps exist).
  * Image resolution for MP4/SVO recordings under the input directory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import pyzed.sl as sl  # type: ignore
except ImportError:  # pragma: no cover
    sl = None


FreqStats = Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        help="Trajectory .h5 file or directory containing the raw capture (trajectory + media).",
    )
    parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        help="Path to ffprobe (used to query MP4 metadata).",
    )
    return parser.parse_args()


def find_trajectory_file(base_path: Path) -> Path:
    if base_path.is_file() and base_path.suffix == ".h5":
        return base_path

    if base_path.is_file():
        base_path = base_path.parent

    candidates = sorted(base_path.rglob("trajectory.h5"))
    if not candidates:
        raise FileNotFoundError(f"Could not locate trajectory.h5 under {base_path}")
    return candidates[0]


def resolve_paths(input_arg: str) -> Tuple[Path, Path]:
    input_path = Path(input_arg).expanduser().resolve()
    trajectory = find_trajectory_file(input_path)
    data_root = trajectory.parent if input_path.is_file() else input_path
    return trajectory, data_root


def combine_robot_timestamps(secs: np.ndarray, nanos: np.ndarray) -> np.ndarray:
    return secs.astype(np.float64) + nanos.astype(np.float64) * 1e-9


def _infer_scale(mean_diff: float) -> float:
    if mean_diff >= 1e6:
        return 1e-9  # assume nanoseconds
    if mean_diff >= 1e3:
        return 1e-6  # assume microseconds
    if mean_diff >= 1:
        return 1e-3  # assume milliseconds
    return 1.0  # already seconds


def frequency_stats_from_series(
    series: np.ndarray, unit: Optional[str] = None
) -> Optional[FreqStats]:
    if series.size < 2:
        return None
    diffs = np.diff(series.astype(np.float64))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None

    if unit == "seconds":
        dt = diffs
    elif unit == "nanoseconds":
        dt = diffs * 1e-9
    elif unit == "microseconds":
        dt = diffs * 1e-6
    elif unit == "milliseconds":
        dt = diffs * 1e-3
    else:
        dt = diffs * _infer_scale(float(diffs.mean()))

    dt_mean = float(dt.mean())
    stats: FreqStats = {
        "hz": 1.0 / dt_mean,
        "dt_mean": dt_mean,
        "dt_std": float(dt.std()),
        "samples": float(series.size),
    }
    return stats


def compute_robot_state_frequency(h5_file: h5py.File) -> Optional[FreqStats]:
    try:
        secs = h5_file["observation"]["timestamp"]["robot_state"][
            "robot_timestamp_seconds"
        ][:]
        nanos = h5_file["observation"]["timestamp"]["robot_state"][
            "robot_timestamp_nanos"
        ][:]
    except KeyError:
        return None
    combined = combine_robot_timestamps(secs, nanos)
    return frequency_stats_from_series(combined, unit="seconds")


def compute_control_frequency(h5_file: h5py.File) -> Optional[FreqStats]:
    try:
        step_start = h5_file["observation"]["timestamp"]["control"]["step_start"][:]
    except KeyError:
        return None
    return frequency_stats_from_series(step_start)


def compute_camera_frequencies(h5_file: h5py.File) -> Dict[str, FreqStats]:
    out: Dict[str, FreqStats] = {}
    try:
        cam_group = h5_file["observation"]["timestamp"]["cameras"]
    except KeyError:
        return out
    for name, dataset in cam_group.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        if not name.endswith("_read_start"):
            continue
        stats = frequency_stats_from_series(dataset[:])
        if stats:
            out[name] = stats
    return out


def discover_media_files(base_path: Path) -> List[Path]:
    supported = {".mp4", ".mov", ".mkv", ".avi", ".svo", ".svo2"}
    return sorted(
        path for path in base_path.rglob("*") if path.is_file() and path.suffix.lower() in supported
    )


def probe_with_ffprobe(filepath: Path, ffprobe_bin: str) -> Optional[Tuple[int, int]]:
    try:
        proc = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(filepath),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(proc.stdout)
        stream = data["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except Exception:
        return None


def probe_with_opencv(filepath: Path) -> Optional[Tuple[int, int]]:
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    height, width = frame.shape[:2]
    return int(width), int(height)


def probe_svo(filepath: Path) -> Optional[Tuple[int, int]]:
    if sl is None:
        return None
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(filepath))
    cam = sl.Camera()
    if cam.open(init_params) != sl.ERROR_CODE.SUCCESS:
        return None
    mat = sl.Mat()
    status = cam.retrieve_image(mat, sl.VIEW.LEFT)
    cam.close()
    if status != sl.ERROR_CODE.SUCCESS:
        return None
    data = mat.get_data()
    if data is None:
        return None
    height, width = data.shape[:2]
    return int(width), int(height)


def probe_media_resolution(
    filepath: Path, ffprobe_bin: str
) -> Dict[str, Optional[str]]:
    info: Dict[str, Optional[str]] = {
        "width": None,
        "height": None,
        "source": None,
        "note": None,
    }
    ext = filepath.suffix.lower()
    source: Optional[str]
    if ext in {".mp4", ".mov", ".mkv", ".avi"}:
        wh = probe_with_ffprobe(filepath, ffprobe_bin)
        source = "ffprobe"
        if wh is None:
            wh = probe_with_opencv(filepath)
            source = "opencv" if wh else None
    else:
        wh = probe_svo(filepath)
        source = "pyzed" if wh else None

    if wh:
        width, height = wh
        info["width"] = str(width)
        info["height"] = str(height)
        info["source"] = source
        if width >= 2 * height:
            info["note"] = "Stereo side-by-side"
    else:
        info["note"] = "Unable to determine resolution"
    return info


def run_analysis(args: argparse.Namespace) -> None:
    trajectory, data_root = resolve_paths(args.input_path)
    print(f"Trajectory file: {trajectory}")

    with h5py.File(trajectory, "r") as handle:
        obs_stats = compute_robot_state_frequency(handle)
        ctrl_stats = compute_control_frequency(handle)
        camera_stats = compute_camera_frequencies(handle)

    print("\nFrequencies")
    if obs_stats:
        print(
            f"  Robot state: {obs_stats['hz']:.2f} Hz "
            f"(Δt mean {obs_stats['dt_mean']*1e3:.2f} ms, samples {int(obs_stats['samples'])})"
        )
    else:
        print("  Robot state: unavailable")

    if ctrl_stats:
        print(
            f"  Control/action loop: {ctrl_stats['hz']:.2f} Hz "
            f"(Δt mean {ctrl_stats['dt_mean']*1e3:.2f} ms)"
        )
    else:
        print("  Control/action loop: unavailable")

    if camera_stats:
        print("  Camera read frequencies:")
        for name, stats in sorted(camera_stats.items()):
            print(
                f"    {name}: {stats['hz']:.2f} Hz "
                f"(Δt mean {stats['dt_mean']*1e3:.2f} ms)"
            )
    else:
        print("  Camera read frequencies: unavailable")

    media_files = discover_media_files(data_root)
    print("\nMedia resolutions")
    if not media_files:
        print("  No MP4/SVO recordings found under input path.")
        return

    for media in media_files:
        info = probe_media_resolution(media, args.ffprobe)
        rel = media.relative_to(data_root)
        width = info["width"] or "?"
        height = info["height"] or "?"
        note = f" ({info['note']})" if info["note"] else ""
        source = f"[{info['source']}]" if info["source"] else "[unknown]"
        print(f"  {rel}: {width}x{height} {source}{note}")


def main() -> None:
    args = parse_args()
    try:
        run_analysis(args)
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
