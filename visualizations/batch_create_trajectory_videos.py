#!/usr/bin/env python3
"""
Batch-create trajectory visualization videos for all runs in a date folder.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _get_script_path(mode: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    if mode == "infer":
        return script_dir / "create_trajectory_visualization_infer.py"
    return script_dir / "create_trajectory_visualization.py"


def _iter_run_dirs(data_root: Path):
    for entry in sorted(data_root.iterdir()):
        if entry.is_dir():
            yield entry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-create trajectory visualization videos for a date folder."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to a date folder containing run subfolders.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root output directory (default: <repo>/outputs).",
    )
    parser.add_argument(
        "--mode",
        choices=["base", "infer"],
        default="base",
        help="Which visualization script to use.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output videos.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List actions without generating videos.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate prerequisites and data availability, then exit.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        print(f"ERROR: data-root not found: {data_root}")
        return 1

    repo_root = _resolve_repo_root()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else repo_root / "outputs"
    )
    date_name = data_root.name
    output_dir = output_root / date_name
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = _get_script_path(args.mode)
    if not script_path.is_file():
        print(f"ERROR: visualization script not found: {script_path}")
        return 1
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found in PATH (required by matplotlib FFMpegWriter)")
        return 1

    total = 0
    skipped = 0
    success = 0
    failed = 0
    ready = 0

    print("=" * 60)
    print("Batch Trajectory Visualization")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Mode: {args.mode}")
    if args.dry_run:
        print("Dry run: no videos will be rendered.")
    if args.check_only:
        print("Check-only: validating prerequisites and inputs only.")

    for run_dir in _iter_run_dirs(data_root):
        total += 1
        run_name = run_dir.name
        trajectory_path = run_dir / "trajectory.h5"
        recordings_dir = run_dir / "recordings" / "MP4"
        output_path = output_dir / f"{run_name}.mp4"

        print("")
        print(f"[{total}] {run_name}")

        if not trajectory_path.is_file():
            print("  - SKIP: trajectory.h5 not found")
            skipped += 1
            continue
        if not recordings_dir.is_dir():
            print("  - SKIP: recordings/MP4 not found")
            skipped += 1
            continue
        if output_path.exists() and not args.overwrite:
            print(f"  - SKIP: output exists ({output_path})")
            skipped += 1
            continue

        ready += 1
        if args.check_only:
            print("  - READY")
            continue
        if args.dry_run:
            print(f"  - WOULD RUN -> {output_path}")
            continue

        cmd = [
            sys.executable,
            str(script_path),
            "--data-dir",
            str(run_dir),
            "--output",
            str(output_path),
        ]
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print("  - OK")
            success += 1
        else:
            print(f"  - FAILED (exit {result.returncode})")
            failed += 1

    print("")
    print("=" * 60)
    print("Batch Processing Complete")
    print("=" * 60)
    print(f"Total runs: {total}")
    print(f"Successful: {success}")
    print(f"Skipped: {skipped}")
    print(f"Ready: {ready}")
    print(f"Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
