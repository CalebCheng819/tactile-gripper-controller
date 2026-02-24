#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def check_unconverted_svo(root_dir):
    """
    Check which SVO files haven't been converted to MP4.
    
    Args:
        root_dir: Path to the directory containing the recordings
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Error: Directory '{root_dir}' does not exist.")
        sys.exit(1)
    
    # Find all directories with recordings
    for dir_path in sorted(root_path.iterdir()):
        if not dir_path.is_dir():
            continue
        
        svo_dir = dir_path / "recordings" / "SVO"
        mp4_dir = dir_path / "recordings" / "MP4"
        
        if not svo_dir.exists() or not mp4_dir.exists():
            continue
        
        # Get list of SVO files (without extension)
        svo_files = sorted([f.stem for f in svo_dir.glob("*.svo2")])
        
        if not svo_files:
            continue
        
        # Check which SVO files don't have corresponding MP4
        unconverted = []
        
        for svo in svo_files:
            mp4_file = mp4_dir / f"{svo}..mp4"
            if not mp4_file.exists():
                unconverted.append(svo)
        
        # Only print if there are unconverted files or issues
        if unconverted:
            print(f"Checking: {dir_path.name}")
            print("-" * 60)
            print(f"Unconverted SVO files: {', '.join(unconverted)}")
            print()

if __name__ == "__main__":
    # Hardcode the path here
    input_dir = "/media/pi0/D8685FAD685F88E0/2025-10-17"
    
    check_unconverted_svo(input_dir)