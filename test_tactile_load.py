#!/usr/bin/env python3
"""Test script to check if tactile model can be loaded."""

import sys
import os

# Add tactile_module to path
sys.path.insert(0, "/home/pi0/multi-modal/tactile_module")

TACTILE_MODEL_CHECKPOINT = "/home/pi0/multi-modal/tactile_module/checkpoints/two_img_delta_gripper_normalized.pt"
TACTILE_MODEL_CONFIG = "/home/pi0/multi-modal/tactile_module/configs/example_with_normalization.yaml"

print("="*70)
print("Testing Tactile Model Loading")
print("="*70)
print(f"Checkpoint: {TACTILE_MODEL_CHECKPOINT}")
print(f"Config: {TACTILE_MODEL_CONFIG}")
print()

# Check if files exist
if not os.path.exists(TACTILE_MODEL_CHECKPOINT):
    print(f"✗ ERROR: Checkpoint file not found: {TACTILE_MODEL_CHECKPOINT}")
    sys.exit(1)
else:
    print(f"✓ Checkpoint file exists")

if not os.path.exists(TACTILE_MODEL_CONFIG):
    print(f"✗ ERROR: Config file not found: {TACTILE_MODEL_CONFIG}")
    sys.exit(1)
else:
    print(f"✓ Config file exists")

print()

# Try to import
try:
    print("Attempting to import TactileGripperAdapter...")
    from robot_inference_adapter import TactileGripperAdapter
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Try to load model
try:
    print("Attempting to load tactile model...")
    adapter = TactileGripperAdapter(
        checkpoint_path=TACTILE_MODEL_CHECKPOINT,
        config_path=TACTILE_MODEL_CONFIG,
    )
    print("✓ Model loaded successfully")
    print(f"  Device: {adapter.device}")
    print("="*70)
    print("SUCCESS: Tactile model can be loaded!")
    print("="*70)
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    print("="*70)
    print("FAILED: Tactile model cannot be loaded!")
    print("="*70)
    sys.exit(1)
