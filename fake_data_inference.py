

#!/usr/bin/env python3
"""
Teleoperation script with tactile model gripper control toggle.

Default: Manual teleop (SpaceMouse controls arm; keyboard for extra commands)
Press 't' key: Toggle tactile-model control for gripper delta
  - When enabled: gripper delta is produced by tactile model inference
  - When disabled: gripper controlled manually (current manual method)
Arm motion remains manual in both modes (only gripper delta is model-driven).
"""

import os
import sys
import time
import signal
import json
from pathlib import Path
from multiprocessing import Process, Event
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import hydra
import numpy as np
import cv2
import h5py
from omegaconf import DictConfig
import yaml

from FORTE.sensing.sensor import sensor_data_updater
from FORTE.scripts.sys_utils import SharedRingBuffer, ForceRingBuffer, opencv_visualizer
from FORTE.scripts.force_est_nn import force_estimator_update_loop_restartsafe, MLPInference

from r2d2.robot_env import RobotEnv
from r2d2.user_interface.data_collector import DataCollecter
from r2d2.user_interface.gui import RobotGUI
from r2d2.misc.parameters import hand_camera_id

cwd = os.getcwd()
sys.path.append(cwd)

from devices import SpaceMouse, Keyboard
from interfaces import HITLPolicy, HumanInterventionReader

# Add tactile_module to path
sys.path.insert(0, "/home/pi0/multi-modal/tactile_module")
from robot_inference_adapter import TactileGripperAdapter

# Model checkpoint path
# TACTILE_MODEL_CHECKPOINT = "/home/pi0/multi-modal/tactile_module/checkpoints/two_img_delta_gripper_normalized.pt"
# TACTILE_MODEL_CHECKPOINT = "/home/pi0/multi-modal/tactile_module/checkpoints/paper_cup_two_img_delta_gripper_normalized.pt"
TACTILE_MODEL_CHECKPOINT = "/home/pi0/multi-modal/tactile_module/checkpoints/1224_paper_cup_two_img_delta_gripper_normalized.pt"
TACTILE_MODEL_CONFIG = "/home/pi0/multi-modal/tactile_module/configs/example_with_normalization.yaml"

# Force estimation model (shared with other demos)
FORCE_MODEL_PATH = "/home/pi0/multi-modal/droid-multi-modal/FORTE/force_est_ckpts/034__sizes-256x256__do-0p3__ido-0p05__wd-0p0001__lr-0p0005__ns-0p01__norm-none__huber-1"
FORCE_DEVICE = "cuda:0"
FORCE_ESTIMATION_HZ = 100

def load_normalization_stats(config_path=None, stats_file_path=None):
    """
    Load normalization statistics from JSON file.
    
    Args:
        config_path: Path to YAML config file (will check for stats_file in config)
        stats_file_path: Direct path to JSON stats file (overrides config)
    
    Returns:
        dict: Normalization statistics dictionary with keys: mean, std, min, max, etc.
              Returns None if file cannot be loaded.
    """
    stats_path = None
    
    # If direct path provided, use it
    if stats_file_path:
        stats_path = Path(stats_file_path).expanduser()
        if not stats_path.is_absolute():
            # Resolve relative to tactile_module directory
            tactile_module_dir = Path("/home/pi0/multi-modal/tactile_module")
            stats_path = (tactile_module_dir / stats_path).resolve()
    elif config_path:
        # Try to load from config file
        try:
            config_file = Path(config_path).expanduser()
            if config_file.exists():
                with config_file.open("r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                
                # Check for stats_file in gripper_normalization section
                robomimic_cfg = config.get("robomimic", {})
                gripper_norm_cfg = robomimic_cfg.get("gripper_normalization", {})
                stats_file = gripper_norm_cfg.get("stats_file")
                
                if stats_file:
                    stats_path = Path(stats_file).expanduser()
                    if not stats_path.is_absolute():
                        # Resolve relative to config file directory (same as robot_inference_adapter.py)
                        config_dir = config_file.resolve().parent
                        stats_path = (config_dir / stats_path).resolve()
        except Exception as e:
            print(f"[WARNING] Failed to load config for normalization stats: {e}")
    
    # Load stats from JSON file
    if stats_path and stats_path.exists():
        try:
            with stats_path.open("r", encoding="utf-8") as f:
                stats_data = json.load(f)
            
            # Validate required fields
            if "mean" not in stats_data or "std" not in stats_data:
                print(f"[ERROR] Stats file missing required fields (mean/std): {stats_path}")
                return None
            
            print(f"[INFO] Loaded normalization stats from: {stats_path}")
            print(f"       mean: {stats_data['mean']}, std: {stats_data['std']}")
            if "min" in stats_data and "max" in stats_data:
                print(f"       min: {stats_data['min']}, max: {stats_data['max']}")
            
            return stats_data
        except Exception as e:
            print(f"[ERROR] Failed to load normalization stats from {stats_path}: {e}")
            return None
    
    print(f"[ERROR] Normalization stats file not found. Please ensure the stats file exists.")
    return None


# Load normalization parameters from config/stats file
_norm_stats = load_normalization_stats(TACTILE_MODEL_CONFIG)
if _norm_stats is None:
    raise FileNotFoundError(
        "Failed to load normalization stats file. "
        "Please ensure the stats file path is correct in the config file."
    )

GRIPPER_NORM_MEAN = float(_norm_stats["mean"])
GRIPPER_NORM_STD = float(_norm_stats["std"])

# Safety clamping for gripper delta - loaded from normalization stats file
# Using exact training data range (no margin)
GRIPPER_DELTA_MIN = float(_norm_stats.get("min", -float("inf")))
GRIPPER_DELTA_MAX = float(_norm_stats.get("max", float("inf")))
# Gripper position range: RobotEnv clips gripper_position to [0, 1] internally
# (see droid/droid/franka/robot.py line 207-209)
GRIPPER_POSITION_MIN = 0.0  # RobotEnv expects [0, 1] for gripper position
GRIPPER_POSITION_MAX = 1.0  # RobotEnv expects [0, 1] for gripper position

# Toggle key
TOGGLE_KEY = "t"

DEMO_DATA_DIR = "/home/pi0/multi-modal/droid-multi-modal/data/success/2025-12-24/Wed_Dec_24_14_15_31_2025"


class DemoDataReader:
    """Provides demo tactile history + wrist images for tactile_adapter inputs."""

    def __init__(self, demo_dir, camera_id):
        demo_path = Path(demo_dir).expanduser()
        trajectory_path = demo_path / "trajectory.h5"
        recordings_dir = demo_path / "recordings" / "MP4"

        if not trajectory_path.exists():
            raise FileNotFoundError(f"trajectory.h5 not found at {trajectory_path}")
        if not recordings_dir.exists():
            raise FileNotFoundError(f"recordings/MP4 not found at {recordings_dir}")

        candidates = [
            recordings_dir / f"{camera_id}..mp4",
            recordings_dir / f"{camera_id}.mp4",
        ]
        video_path = next((p for p in candidates if p.exists()), None)
        if video_path is None:
            raise FileNotFoundError(f"Video not found for camera {camera_id} in {recordings_dir}")

        with h5py.File(trajectory_path, "r") as h5f:
            self.tactile_values = h5f["observation/robot_state/tactile_values"][:]

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video {video_path}")

        self.frame_idx = 0
        self.num_frames = self.tactile_values.shape[0]
        print(f"[DEMO] Using {trajectory_path}")
        print(f"[DEMO] Using {video_path}")
        print(f"[DEMO] Frames available: {self.num_frames}")

    def read_next(self):
        if self.frame_idx >= self.num_frames:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        height, width = frame.shape[:2]
        half_width = width // 2
        left_bgr = frame[:, :half_width]
        right_bgr = frame[:, half_width:]

        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

        tactile_history = self.tactile_values[self.frame_idx].astype(np.float32)
        self.frame_idx += 1

        return left_rgb, right_rgb, tactile_history

    def close(self):
        if self.cap is not None:
            self.cap.release()


class babyFORTEReader:
    """Tactile sensor reader (from jaelyn_demo.py)."""

    def __init__(self, cfg, shutdown_event, force_buffer=None):
        self.shared_sensor_buffer = SharedRingBuffer(cfg.buffer.size, cfg.buffer.num_channels, 'd')
        self.shared_force_buffer = force_buffer or ForceRingBuffer()
        self._owns_force_buffer = force_buffer is None
        self.force_estimator_process = None

        # Load force estimator and spawn process to keep shared_force_buffer updated
        try:
            self._force_estimator = MLPInference(FORCE_MODEL_PATH, device=FORCE_DEVICE)
            self.force_estimator_process = Process(
                target=force_estimator_update_loop_restartsafe,
                args=(
                    self._force_estimator,
                    self.shared_sensor_buffer,
                    self.shared_force_buffer,
                    shutdown_event,
                ),
                kwargs={"hz": FORCE_ESTIMATION_HZ, "start_delay": 5.0},
            )
            self.force_estimator_process.start()
            print(f"[force-estimator] started with {FORCE_MODEL_PATH} on {FORCE_DEVICE}")
        except Exception as e:
            print(f"[WARNING] Force estimator unavailable: {e}")
            import traceback
            traceback.print_exc()

        self.sensor_process = Process(
            target=sensor_data_updater,
            args=(cfg.elvrgripper, self.shared_sensor_buffer),
            kwargs={'mode': 'process', 'shutdown_event': shutdown_event},
        )
        self.sensor_process.start()

        time.sleep(2)

        self.visualizer_process = Process(
            target=opencv_visualizer,
            args=(self.shared_sensor_buffer, cfg.buffer, self.shared_force_buffer, None, shutdown_event)
        )
        self.visualizer_process.start()

        self.shutdown_event = shutdown_event
        self.cfg = cfg

    def update_values(self):
        pass

    def read_values(self):
        if self.shared_sensor_buffer.is_empty():
            return np.zeros((self.shared_sensor_buffer.num_channels,))
        return self.shared_sensor_buffer.get_latest_freq_history()

    def reset(self):
        """Reset tactile reader (for compatibility with DataCollecter)."""
        # Note: The sensor processes continue running, this just clears any accumulated state
        # The buffer will continue to be updated by the sensor process
        pass

    def close(self):
        print("Shutting down processes...")
        self.shutdown_event.set()

        procs = [self.visualizer_process, self.sensor_process, self.force_estimator_process]
        procs = [p for p in procs if p is not None]

        for proc in procs:
            proc.terminate()

        time.sleep(2)
        for proc in procs:
            proc.kill()
            proc.join(timeout=1)

        self.shared_sensor_buffer.close()
        if self._owns_force_buffer:
            self.shared_force_buffer.close()
        print("Demo completed. Resources cleaned up.")


class TactileGripperController:
    """
    Wrapper around HITLPolicy that adds tactile model gripper control toggle.
    
    When tactile model is enabled, overrides gripper delta from manual input
    with model prediction. Arm motion always comes from manual teleop.
    """

    def __init__(self, hitl_policy, tactile_adapter, tactile_reader, keyboard, demo_reader=None):
        self.hitl_policy = hitl_policy
        self.tactile_adapter = tactile_adapter
        self.tactile_reader = tactile_reader
        self.keyboard = keyboard
        self.demo_reader = demo_reader
        
        self.model_enabled = False
        self._last_toggle_state = False
        self._emergency_stop = False
        self._print_counter = 0  # Counter to reduce print frequency
        self._missing_image_warned = False  # Throttle missing-image warnings
        self._normalized_delta_history = []
        self._plot_saved = False
        
        # Safety mechanism: Minimum delta threshold to prevent unwanted movements
        # If model predicts delta smaller than this, it's treated as "no action"
        # 
        # PROBLEM ANALYSIS:
        # The model was trained on data filtered with idle_epsilon=0.005, which removed
        # all "no action" samples (delta < 0.005). As a result, the model learned to
        # always predict grasping actions, even when there's no target object.
        # 
        # SOLUTION:
        # Set threshold to 0.008-0.01 to suppress unwanted grasping when no target object.
        # This compensates for the training data bias.
        # 
        # Based on testing without target object:
        # - Model predicts delta ~0.003-0.008 without target object
        # - Threshold 0.008 suppresses most unwanted motions
        # - Threshold 0.01 is very strict (may suppress legitimate small motions)
        self.DELTA_THRESHOLD = 0.003  # Minimum delta to apply gripper action
        # Alternative thresholds:
        # self.DELTA_THRESHOLD = 0.006  # Less strict - some unwanted motions may pass
        # self.DELTA_THRESHOLD = 0.01   # Very strict - suppresses delta < 0.01
        
        print(f"\n{'='*70}")
        print("🤖 TACTILE MODEL GRIPPER CONTROL")
        print(f"{'='*70}")
        print(f"📌 Press '{TOGGLE_KEY}' key during trajectory collection to toggle")
        print(f"   • When ENABLED: Tactile model controls gripper automatically")
        print(f"   • When DISABLED: Manual gripper control")
        print(f"   • Arm is always controlled by SpaceMouse")
        print(f"\n Press Ctrl+C for emergency stop")
        print(f"{'='*70}\n")

    def _check_toggle(self):
        """Check if toggle key was pressed."""
        if self.keyboard is None:
            return False
        
        # Check if 't' key is pressed
        current_state = self.keyboard.buttons.get(TOGGLE_KEY, False)
        
        # Detect rising edge (key just pressed)
        if current_state and not self._last_toggle_state:
            self.model_enabled = not self.model_enabled
            # Clear, prominent message
            print("\n" + "="*70)
            if self.model_enabled:
                print("TACTILE MODEL ENABLED - GRIPPER AUTO-CONTROL ACTIVE ")
                print("   The tactile model will now control the gripper automatically!")
                print("   You control the arm with SpaceMouse, model controls gripper.")
            else:
                print("TACTILE MODEL DISABLED - MANUAL CONTROL ")
                print("   Gripper control returned to manual mode.")
            print("="*70 + "\n")
            return True
        
        self._last_toggle_state = current_state
        return False


    def _denormalize_gripper_delta(self, normalized_delta):
        """
        De-normalize gripper delta from model output.
        
        During training: normalized = (delta - mean) / std
        At inference: delta = normalized * std + mean
        """
        return normalized_delta * GRIPPER_NORM_STD + GRIPPER_NORM_MEAN

    def _clamp_gripper_delta(self, delta):
        """Clamp gripper delta to safe range."""
        return np.clip(delta, GRIPPER_DELTA_MIN, GRIPPER_DELTA_MAX)

    def _save_normalized_delta_plot(self):
        if not self._normalized_delta_history or self._plot_saved:
            return
        history = np.asarray(self._normalized_delta_history, dtype=np.float32)
        out_dir = Path.home() / "Downloads"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "normalized_delta.png"

        width = 1200
        height = 400
        plot = np.full((height, width, 3), 255, dtype=np.uint8)

        x_vals = np.linspace(0, width - 1, num=len(history)).astype(np.int32)
        min_val = float(np.min(history))
        max_val = float(np.max(history))
        if min_val == max_val:
            min_val -= 1.0
            max_val += 1.0
        margin = 0.1 * (max_val - min_val)
        min_val -= margin
        max_val += margin

        y_vals = (height - 1) - ((history - min_val) / (max_val - min_val) * (height - 1))
        y_vals = y_vals.astype(np.int32)
        points = np.stack([x_vals, y_vals], axis=1)
        cv2.polylines(plot, [points], isClosed=False, color=(0, 0, 255), thickness=2)
        cv2.imwrite(str(out_path), plot)
        print(f"[DEMO] Saved normalized_delta plot to {out_path}")
        self._plot_saved = True

    def forward(self, obs_dict, include_info=False):
        """
        Forward pass with optional tactile model gripper override.
        
        Returns action with:
        - Arm joints: from manual teleop (HITLPolicy)
        - Gripper: from tactile model if enabled, else from manual teleop
        """
        # Emergency stop check
        if self._emergency_stop:
            stop_action = np.zeros(8, dtype=np.float32)
            if include_info:
                return stop_action, {"emergency_stop": True}
            return stop_action
        
        # Check for toggle
        self._check_toggle()
        
        # Get base action from HITLPolicy (includes manual gripper)
        base_action = self.hitl_policy.forward(obs_dict, include_info=include_info)
        
        if include_info:
            action, info = base_action
        else:
            action = base_action
        
        # If model is disabled, return action as-is
        if not self.model_enabled:
            return base_action
        
        # Model is enabled: override gripper with tactile model prediction
        try:
            # Get observations needed for tactile model
            robot_state = obs_dict.get("robot_state", {})
            
            # Get wrist images + tactile history
            if self.demo_reader is not None:
                demo_payload = self.demo_reader.read_next()
                if demo_payload is None:
                    if not self._missing_image_warned:
                        print("[WARNING] Demo data exhausted; using manual gripper control")
                        self._missing_image_warned = True
                    self._save_normalized_delta_plot()
                    return base_action
                wrist_left, wrist_right, tactile_history = demo_payload
            else:
                # Get wrist camera images from obs_dict["image"]
                # ZED cameras return images with keys: serial_number + "_left" and serial_number + "_right"
                # For hand camera: hand_camera_id + "_left" and hand_camera_id + "_right"
                image_dict = obs_dict.get("image", {})
                
                # Try to get wrist camera images using hand_camera_id
                wrist_left_key = f"{hand_camera_id}_left"
                wrist_right_key = f"{hand_camera_id}_right"
                
                # Debug: Print available image keys on first access (to verify correct camera ID)
                if not hasattr(self, '_image_keys_logged'):
                    available_keys = list(image_dict.keys())
                    print(f"\n[DEBUG] Available image keys: {available_keys[:10]}...")  # Show first 10 keys
                    print(f"[DEBUG] Looking for wrist camera images:")
                    print(f"        Left key:  '{wrist_left_key}'")
                    print(f"        Right key: '{wrist_right_key}'")
                    print(f"        hand_camera_id: {hand_camera_id}")
                    self._image_keys_logged = True
                
                wrist_left = image_dict.get(wrist_left_key)
                wrist_right = image_dict.get(wrist_right_key)
                
                # Fallback: try alternative key names (for compatibility)
                if wrist_left is None:
                    wrist_left = (obs_dict.get("wrist_image_left") or 
                                 obs_dict.get("wrist_image") or
                                 robot_state.get("wrist_image_left"))
                if wrist_right is None:
                    wrist_right = (obs_dict.get("wrist_image_right") or
                                  robot_state.get("wrist_image_right"))
                
                # If right image not available, use left as fallback
                if wrist_right is None:
                    wrist_right = wrist_left
                
                # Convert and preprocess images to match training data format
                # Training data preprocessing (from robomimic_delta_gripper_v2.py):
                # 1. Get image from obs["image"][cam_id + "_left/right"]
                # 2. Convert BGR->RGB via [..., ::-1] (or BGRA->RGB via cv2)
                # 3. Resize to (224, 224) with letterbox or direct resize
                # 4. Output format: (H, W, 3) RGB uint8
                
                def preprocess_image(img):
                    """Preprocess image to match training format: (H, W, 3) RGB uint8"""
                    if img is None:
                        return None
                    
                    # Handle different input formats
                    if len(img.shape) == 3:
                        if img.shape[0] == 4:
                            # (4, H, W) format - transpose to (H, W, 4)
                            img = np.transpose(img, (1, 2, 0))
                        
                        # Convert BGRA/BGR to RGB
                        if img.shape[2] == 4:
                            # BGRA -> RGB (remove alpha channel)
                            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                        elif img.shape[2] == 3:
                            # Check if it's BGR or RGB by checking if it needs conversion
                            # ZED cameras typically return BGRA, but if we get BGR, convert it
                            # We'll assume if it's 3-channel from ZED, it might be BGR
                            # Actually, let's check: if the image came from ZED and is 3-channel,
                            # it's likely already been processed. But to be safe, we'll convert BGR->RGB
                            # However, since we're getting BGRA from ZED, this case shouldn't happen
                            # But we'll keep it for safety
                            pass  # Assume RGB if already 3-channel
                    
                    # Ensure format is (H, W, 3) RGB uint8
                    if len(img.shape) != 3 or img.shape[2] != 3:
                        raise ValueError(f"Expected image with shape (H, W, 3); got {img.shape}")
                    
                    # Ensure uint8 dtype (model adapter will normalize to [0,1] if needed)
                    if img.dtype != np.uint8:
                        img = img.astype(np.uint8)
                    
                    return img
                
                wrist_left = preprocess_image(wrist_left)
                wrist_right = preprocess_image(wrist_right)
                
                # Get tactile history
                tactile_history = self.tactile_reader.read_values()
            
            # Get current gripper position
            gripper_pos = obs_dict.get("gripper_position") or robot_state.get("gripper_position", [0.0])
            if isinstance(gripper_pos, (list, np.ndarray)):
                current_gripper = float(gripper_pos[0])
            else:
                current_gripper = float(gripper_pos)
            
            # Run tactile model inference
            if wrist_left is not None and wrist_right is not None:
                # Clear missing image warning flag since images are now available
                self._missing_image_warned = False
                
                # Model outputs normalized delta (action chunk, we use first step)
                normalized_delta = self.tactile_adapter.predict_delta(
                    image_left=wrist_left,
                    image_right=wrist_right,
                    tactile_history=tactile_history,
                    step_index=0  # Use first step from action chunk
                )
                try:
                    self._normalized_delta_history.append(float(np.asarray(normalized_delta).reshape(-1)[0]))
                except Exception:
                    self._normalized_delta_history.append(float(normalized_delta))
                if self.demo_reader is not None and len(self._normalized_delta_history) % 50 == 0:
                    self._save_normalized_delta_plot()
                print("normalized_delta", normalized_delta)

                
                # De-normalize
                delta = self._denormalize_gripper_delta(normalized_delta)
                
                # Safety mechanism: If delta is very small (near zero), don't apply it
                # This helps prevent unwanted gripper movements when no target object is present
                # The model should predict near-zero delta when there's no object to grasp
                raw_delta = delta  # Save for debugging
                if abs(delta) < self.DELTA_THRESHOLD:
                    # Model predicts no action needed - keep gripper unchanged
                    # This is expected behavior when no target object is visible
                    delta = 0.0
                    # Debug: Print when delta is suppressed (every 100 steps to avoid spam)
                    if self._print_counter % 100 == 0:
                        print(f"[SAFETY] Suppressed small delta: {raw_delta:.6f} (threshold: {self.DELTA_THRESHOLD:.6f})")
                
                # Clamp delta
                delta = self._clamp_gripper_delta(delta)
                
                # Check action space to determine if we need velocity or position command
                # When action_space is "joint_velocity", gripper action is interpreted as velocity
                # When action_space is "joint_position", gripper action is interpreted as position
                action_space = self.hitl_policy.robot_env.action_space if hasattr(self.hitl_policy, 'robot_env') else "joint_velocity"
                
                if "velocity" in action_space:
                    # Convert delta to velocity command
                    # max_gripper_delta = 0.25 (from RobotIKSolver)
                    # velocity = delta / max_gripper_delta
                    max_gripper_delta = 0.25
                    gripper_velocity = 0.5* delta / max_gripper_delta
                    # print("delta:", delta * 0.5)
                    print("gripper_velocity", gripper_velocity)
                    # Clamp velocity to [-1, 1] range (standard action range)
                    gripper_velocity = np.clip(gripper_velocity, -1.0, 1.0)
                    
                    # Override gripper in action (last dimension) with velocity command
                    action = np.array(action, dtype=np.float32).copy()
                    action[-1] = gripper_velocity
                    
                    # Compute target for display
                    target_gripper = current_gripper + delta * 0.5
                    target_gripper = np.clip(target_gripper, GRIPPER_POSITION_MIN, GRIPPER_POSITION_MAX)
                    
                    # Print tactile model output (every 30 steps to reduce clutter)
                    self._print_counter += 1
                    if self._print_counter % 30 == 0:
                        # Show normalized delta for debugging (to understand model's raw output)
                        print(f"[TACTILE MODEL ACTIVE] Gripper Position: {current_gripper:.3f} → {target_gripper:.3f} "
                              f"(delta: {delta:+.4f}, norm_delta: {normalized_delta:+.4f}, vel: {gripper_velocity:+.3f})")
                        if abs(delta) < self.DELTA_THRESHOLD:
                            print(f"[INFO] Delta suppressed (below threshold {self.DELTA_THRESHOLD:.4f})")
                else:
                    # Position mode: use absolute position
                    target_gripper = current_gripper + delta
                    target_gripper = np.clip(target_gripper, GRIPPER_POSITION_MIN, GRIPPER_POSITION_MAX)
                    
                    # Override gripper in action (last dimension) with position command
                    action = np.array(action, dtype=np.float32).copy()
                    action[-1] = target_gripper
                    
                    # Print tactile model output (every 30 steps to reduce clutter)
                    self._print_counter += 1
                    if self._print_counter % 30 == 0:
                        # Show normalized delta for debugging (to understand model's raw output)
                        print(f"[TACTILE MODEL ACTIVE] Gripper: {current_gripper:.3f} → {target_gripper:.3f} "
                              f"(delta: {delta:+.4f}, norm_delta: {normalized_delta:+.4f})")
                        if abs(delta) < self.DELTA_THRESHOLD:
                            print(f"[INFO] Delta suppressed (below threshold {self.DELTA_THRESHOLD:.4f})")
            else:
                # Avoid spamming missing-image warnings
                if not self._missing_image_warned:
                    print("[WARNING] Missing wrist images, using manual gripper control")
                    self._missing_image_warned = True
        
        except KeyboardInterrupt:
            # Emergency stop requested
            print("\n[EMERGENCY STOP] Keyboard interrupt detected")
            self._emergency_stop = True
            stop_action = np.zeros(8, dtype=np.float32)
            if include_info:
                return stop_action, {"emergency_stop": True}
            return stop_action
        
        except Exception as e:
            print(f"[ERROR] Tactile model inference failed: {e}")
            print("         Falling back to manual gripper control")
            # Don't set emergency stop for inference errors, just fall back to manual
            import traceback
            traceback.print_exc()
        
        if include_info:
            return action, info
        else:
            return action

    def reset_state(self):
        """Reset controller state."""
        self.hitl_policy.reset_state()
        self.model_enabled = False
        self._last_toggle_state = False

    def set_instruction(self, instruction):
        """Set instruction (for compatibility with HITLPolicy)."""
        self.hitl_policy.set_instruction(instruction)

    @property
    def instruction(self):
        """Get instruction (for compatibility with HITLPolicy)."""
        return self.hitl_policy.instruction

    def get_info(self):
        """Get controller info (for compatibility with DataCollecter)."""
        return self.hitl_policy.get_info()


@hydra.main(config_path="../FORTE/config/", config_name="baby_FORTE")
def main(cfg: DictConfig):
    devices = {"spacemouse": SpaceMouse(reset_with_idle=False), "keyboard": Keyboard()}

    shutdown_event = Event()

    def handle_exit(signum, frame):
        print(f"\nSignal {signum} received. Exiting...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    force_buffer = ForceRingBuffer()

    try:
        # Initialize tactile sensor reader
        tactile_reader = babyFORTEReader(cfg, shutdown_event, force_buffer)

        # Initialize robot environment
        env = RobotEnv(
            action_space="joint_velocity",
            sensor_readers={
                "tactile_values": tactile_reader,
                "human_intervention": HumanInterventionReader(**devices),
            },
            experiment_name="tactile_infer_teleop"
        )

        # Load tactile model
        print(f"\n{'='*60}")
        print("Loading tactile model...")
        print(f"Checkpoint: {TACTILE_MODEL_CHECKPOINT}")
        print(f"Config: {TACTILE_MODEL_CONFIG}")
        print(f"{'='*60}\n")
        
        try:
            tactile_adapter = TactileGripperAdapter(
                checkpoint_path=TACTILE_MODEL_CHECKPOINT,
                config_path=TACTILE_MODEL_CONFIG,
            )
            print("Tactile model loaded successfully")
            print(f"   Device: {tactile_adapter.device}")
        except Exception as e:
            print(f"Failed to load tactile model: {e}")
            print("   Continuing with manual control only")
            import traceback
            traceback.print_exc()
            tactile_adapter = None

        # Create base HITLPolicy controller
        hitl_policy = HITLPolicy(devices, policy=None, robot_env=env)

        # Wrap with tactile gripper controller
        if tactile_adapter is not None:
            demo_reader = DemoDataReader(DEMO_DATA_DIR, hand_camera_id)
            controller = TactileGripperController(
                hitl_policy=hitl_policy,
                tactile_adapter=tactile_adapter,
                tactile_reader=tactile_reader,
                keyboard=devices["keyboard"],
                demo_reader=demo_reader
            )
        else:
            # Fallback to manual only if model failed to load
            controller = hitl_policy
            print("\n Running in manual-only mode (tactile model unavailable)")

        # Start GUI
        data_collector = DataCollecter(env=env, controller=controller)
        RobotGUI(robot=data_collector)

    finally:
        print("\nCleaning up resources...")
        tactile_reader.close()
        force_buffer.close()
        if "demo_reader" in locals():
            demo_reader.close()
        print("Resources cleaned up.")


if __name__ == "__main__":
    main()
