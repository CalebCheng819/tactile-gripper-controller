

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
import re
import atexit
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Event
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import hydra
import numpy as np
import cv2
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
# TACTILE_MODEL_CHECKPOINT = "/home/pi0/multi-modal/tactile_module/checkpoints/1224_paper_cup_two_img_delta_gripper_normalized.pt"
TACTILE_MODEL_CHECKPOINT = "/home/pi0/multi-modal/checkpoints/box_and_cup/checkpoints/ab_drop1_C1_seed42_drop_demo_20_20260220_203907_best.pt"
# Set to "auto" to infer matching config from checkpoint name (recommended).
# You can still set an explicit yaml path if needed.
TACTILE_MODEL_CONFIG = "auto"
# Fallback config when auto resolution fails.
TACTILE_MODEL_CONFIG_FALLBACK = "/home/pi0/multi-modal/tactile_module/configs/example_with_normalization.yaml"
# Optional roots to search for auto-matched experiment configs.
TACTILE_MODEL_CONFIG_SEARCH_ROOTS = [
    "/home/pi0/multi-modal/tactile_module/experiments",
    "/home/pi0/multi-modal/tactile_module/configs",
]

# Run log settings (stdout + stderr tee to file for post-run debugging)
TACTILE_SAVE_LOG = os.getenv("TACTILE_SAVE_LOG", "1") != "0"
TACTILE_RUN_LOG_DIR = "/home/pi0/multi-modal/droid-multi-modal/scripts/logs/tactile_infer_teleop"

# Force estimation model (shared with other demos)
FORCE_MODEL_PATH = "/home/pi0/multi-modal/droid-multi-modal/FORTE/force_est_ckpts/034__sizes-256x256__do-0p3__ido-0p05__wd-0p0001__lr-0p0005__ns-0p01__norm-none__huber-1"
FORCE_DEVICE = "cuda:0"
FORCE_ESTIMATION_HZ = 100


class _TeeStream:
    """Mirror writes to multiple streams."""

    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)

    def fileno(self):
        for stream in self._streams:
            fn = getattr(stream, "fileno", None)
            if callable(fn):
                return fn()
        raise OSError("No fileno available on tee streams")


_RUN_LOG_FILE_HANDLE = None
_RUN_LOG_PATH = None


def _sanitize_filename(name):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))
    return safe.strip("._-") or "run"


def _close_run_log():
    global _RUN_LOG_FILE_HANDLE
    if _RUN_LOG_FILE_HANDLE is not None:
        try:
            _RUN_LOG_FILE_HANDLE.flush()
        finally:
            _RUN_LOG_FILE_HANDLE.close()
        _RUN_LOG_FILE_HANDLE = None


def _setup_run_logging():
    """
    Enable per-run log file while keeping terminal output visible.

    Env controls:
      - TACTILE_SAVE_LOG=0 disables file logging.
      - TACTILE_RUN_LOG_DIR=/path/to/dir overrides default directory.
      - TACTILE_RUN_LOG_PATH=/path/to/file.log writes to an explicit file.
    """
    global _RUN_LOG_FILE_HANDLE, _RUN_LOG_PATH

    if not TACTILE_SAVE_LOG:
        return None

    explicit_log_path = os.getenv("TACTILE_RUN_LOG_PATH", "").strip()
    if explicit_log_path:
        log_path = Path(explicit_log_path).expanduser()
        if not log_path.is_absolute():
            log_path = (Path.cwd() / log_path).resolve()
    else:
        log_dir = Path(os.getenv("TACTILE_RUN_LOG_DIR", TACTILE_RUN_LOG_DIR)).expanduser()
        if not log_dir.is_absolute():
            log_dir = (Path.cwd() / log_dir).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_tag = _sanitize_filename(Path(TACTILE_MODEL_CHECKPOINT).stem)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{ts}__{checkpoint_tag}.log"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    _RUN_LOG_FILE_HANDLE = log_path.open("a", encoding="utf-8", buffering=1)
    _RUN_LOG_PATH = str(log_path)

    sys.stdout = _TeeStream(sys.stdout, _RUN_LOG_FILE_HANDLE)
    sys.stderr = _TeeStream(sys.stderr, _RUN_LOG_FILE_HANDLE)
    atexit.register(_close_run_log)

    print(f"[INFO] Run log enabled: {_RUN_LOG_PATH}")
    print(f"[INFO] TACTILE_MODEL_CHECKPOINT: {TACTILE_MODEL_CHECKPOINT}")
    return _RUN_LOG_PATH


_setup_run_logging()

def _parse_checkpoint_variant(checkpoint_path):
    """
    Parse variant tag from checkpoint filename.
    Example filename suffix:
      ...-tm_tactile_only__aux_on__pm_delta__chunk_10_best.pt
    """
    name = Path(checkpoint_path).name
    m = re.search(r"-(tm_[^/]+)_best\.pt$", name)
    if not m:
        return None, None

    variant = m.group(1)
    m2 = re.match(
        r"tm_(?P<tactile_mode>.+?)__aux_(?P<aux>on|off)__pm_(?P<prediction_mode>delta|multi_head|final_target)__chunk_(?P<chunk>\d+)$",
        variant,
    )
    if not m2:
        return variant, None

    meta = {
        "variant": variant,
        "tactile_mode": m2.group("tactile_mode"),
        "use_force_aux_head": m2.group("aux") == "on",
        "prediction_mode": m2.group("prediction_mode"),
        "action_chunk_size": int(m2.group("chunk")),
    }
    return variant, meta


def _yaml_matches_variant(yaml_path, meta):
    if not meta:
        return True
    try:
        with Path(yaml_path).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return False

    model = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    rob = cfg.get("robomimic", {}) if isinstance(cfg, dict) else {}

    tactile_mode = str(rob.get("tactile_mode", "")).strip()
    prediction_mode = str(model.get("prediction_mode", "")).strip()
    target_mode = str(rob.get("target_mode", prediction_mode)).strip()

    # action_chunk_size may be in robomimic or model section.
    chunk_raw = rob.get("action_chunk_size", model.get("action_chunk_size", None))
    try:
        chunk = int(chunk_raw) if chunk_raw is not None else None
    except Exception:
        chunk = None

    aux_on = bool(model.get("use_force_aux_head", False))

    if tactile_mode and tactile_mode != meta["tactile_mode"]:
        return False
    if prediction_mode and prediction_mode != meta["prediction_mode"]:
        return False
    if target_mode and target_mode != meta["prediction_mode"]:
        return False
    if chunk is not None and chunk != meta["action_chunk_size"]:
        return False
    if aux_on != meta["use_force_aux_head"]:
        return False
    return True


def resolve_tactile_model_config(checkpoint_path, config_value):
    """
    Resolve config path robustly:
      1) explicit config path (if exists)
      2) env override TACTILE_MODEL_CONFIG_OVERRIDE
      3) auto-infer from checkpoint variant in experiment config directories
      4) fallback config
    """
    # 0) environment override always wins when valid.
    env_override = os.getenv("TACTILE_MODEL_CONFIG_OVERRIDE", "").strip()
    if env_override:
        p = Path(env_override).expanduser()
        if p.exists():
            return str(p.resolve())
        print(f"[WARNING] TACTILE_MODEL_CONFIG_OVERRIDE does not exist: {p}")

    # 1) explicit path.
    cfg_value = str(config_value).strip() if config_value is not None else ""
    if cfg_value and cfg_value.lower() not in {"auto", "none", ""}:
        p = Path(cfg_value).expanduser()
        if p.exists():
            return str(p.resolve())
        print(f"[WARNING] Explicit TACTILE_MODEL_CONFIG not found: {p}. Falling back to auto.")

    # 2) auto infer from checkpoint name.
    variant, meta = _parse_checkpoint_variant(checkpoint_path)
    if variant:
        candidates = []
        for root in TACTILE_MODEL_CONFIG_SEARCH_ROOTS:
            rp = Path(root).expanduser()
            if not rp.exists():
                continue
            # Common layout: <root>/**/configs/<variant>.yaml
            for p in rp.glob(f"**/{variant}.yaml"):
                if _yaml_matches_variant(p, meta):
                    candidates.append(p)

        if candidates:
            # prefer newest file
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            chosen = candidates[0].resolve()
            print(f"[INFO] Auto-resolved config from checkpoint variant '{variant}': {chosen}")
            return str(chosen)

        print(f"[WARNING] Could not auto-match config for checkpoint variant: {variant}")

    # 3) fallback
    fb = Path(TACTILE_MODEL_CONFIG_FALLBACK).expanduser()
    if fb.exists():
        print(f"[INFO] Using fallback config: {fb}")
        return str(fb.resolve())

    raise FileNotFoundError(
        "Failed to resolve TACTILE_MODEL_CONFIG. Set explicit path or ensure fallback exists."
    )


def load_normalization_stats(config_path=None, stats_file_path=None):
    """
    Load normalization statistics from JSON file.
    
    Args:
        config_path: Path to YAML config file (will check for stats_file in config)
        stats_file_path: Direct path to JSON stats file (overrides config)
    
    Returns:
        dict: Normalization statistics dictionary (keys depend on normalization type,
              e.g. mean/std or min/max or abs_max). Returns None if file cannot be loaded.
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
            
            if not isinstance(stats_data, dict):
                print(f"[ERROR] Invalid stats file format (expected JSON object): {stats_path}")
                return None

            print(f"[INFO] Loaded normalization stats from: {stats_path}")
            if "mean" in stats_data and "std" in stats_data:
                print(f"       mean: {stats_data['mean']}, std: {stats_data['std']}")
            if "min" in stats_data and "max" in stats_data:
                print(f"       min: {stats_data['min']}, max: {stats_data['max']}")
            if "abs_max" in stats_data:
                print(f"       abs_max: {stats_data['abs_max']}")
            
            return stats_data
        except Exception as e:
            print(f"[ERROR] Failed to load normalization stats from {stats_path}: {e}")
            return None
    
    print(f"[ERROR] Normalization stats file not found. Please ensure the stats file exists.")
    return None


# Resolve model config automatically when requested, then load normalization stats.
TACTILE_MODEL_CONFIG = resolve_tactile_model_config(TACTILE_MODEL_CHECKPOINT, TACTILE_MODEL_CONFIG)
print(f"[INFO] Using tactile model config: {TACTILE_MODEL_CONFIG}")

# Load normalization parameters from config/stats file
_norm_stats = load_normalization_stats(TACTILE_MODEL_CONFIG)
if _norm_stats is None:
    raise FileNotFoundError(
        "Failed to load normalization stats file. "
        "Please ensure the stats file path is correct in the config file."
    )

# Safety clamping for gripper delta -- symmetric range based on abs_max
# Training data deltas are one-sided [0, 0.02], but at inference the force-safety
# layer can produce negative deltas (reopen). Use symmetric range so both signs pass.
_abs_max = float(_norm_stats.get("abs_max", _norm_stats.get("max", float("inf"))))
GRIPPER_DELTA_MIN = -_abs_max
GRIPPER_DELTA_MAX = _abs_max
# Gripper position range: RobotEnv clips gripper_position to [0, 1] internally
# (see droid/droid/franka/robot.py line 207-209)
GRIPPER_POSITION_MIN = 0.0
GRIPPER_POSITION_MAX = 1.0

# ---------- final_target fallback (capped-P + hysteresis) ----------
TARGET_FALLBACK_STEP_MAX = 0.005   # max delta per tick when far from target
TARGET_FALLBACK_STEP_MIN = 0.00085 # floor: survives velocity deadzone (0.0562/80 * 1.2)
TARGET_FALLBACK_P_GAIN = 0.1      # proportional gain for soft landing near target
TARGET_FALLBACK_EMA_ALPHA = 0.3   # EMA smoothing on raw final_target
TARGET_FALLBACK_DZ_ENTER = 0.008  # enter deadzone (stop) when |error| drops below this
TARGET_FALLBACK_DZ_EXIT = 0.015   # exit deadzone (resume) only when |error| exceeds this

# ---------- oscillation fixes ----------
FORCE_SAFETY_COOLDOWN_FRAMES = 10  # ~0.2s at 50Hz; hold after hard_stop
TARGET_FALLBACK_DUTY_PERIOD = 5    # emit step once per 5 frames, hold 0 between
GRASP_STABLE_FORCE_ENTER = 100.0   # freeze EMA when |force| exceeds this
GRASP_STABLE_FORCE_EXIT = 70.0     # unfreeze only when |force| drops below this
GRASP_STABLE_DELTA_EPS = 0.0005    # applied delta below this = "not moving"
GRASP_STABLE_FRAMES = 15           # consecutive stable frames to confirm freeze
GRASP_FREEZE_MIN_FRAMES = 30       # once frozen, hold >= 0.6s before allowing unfreeze

# Toggle key
TOGGLE_KEY = "t"


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

    def read_force_values(self, k=50):
        """
        Read latest force history from shared force ring buffer.
        Returns shape (k, 1) float32.
        """
        try:
            k = max(1, int(k))
            force = np.asarray(self.shared_force_buffer.get_latest(k=k), dtype=np.float32).reshape(-1, 1)
            return force
        except Exception as e:
            if not hasattr(self, "_force_read_warned"):
                print(f"[WARNING] Failed to read force history: {e}. Using zeros.")
                self._force_read_warned = True
            return np.zeros((max(1, int(k)), 1), dtype=np.float32)

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

    def __init__(self, hitl_policy, tactile_adapter, tactile_reader, keyboard):
        self.hitl_policy = hitl_policy
        self.tactile_adapter = tactile_adapter
        self.tactile_reader = tactile_reader
        self.keyboard = keyboard
        
        self.model_enabled = False
        self._last_toggle_state = False
        self._emergency_stop = False
        self._print_counter = 0  # Counter to reduce print frequency
        self._missing_image_warned = False  # Throttle missing-image warnings
        self._timing_enabled = os.getenv("TACTILE_PROFILE", "1") != "0"
        self._timing_every = max(1, int(os.getenv("TACTILE_PROFILE_EVERY", "1")))
        self._timing_counter = 0
        self._target_ema = None  # EMA-smoothed final_target for fallback
        self._target_in_deadzone = False  # hysteresis state for fallback deadzone
        self._force_cooldown_counter = 0
        self._fallback_duty_counter = 0
        self._grasp_stable_counter = 0
        self._ema_frozen = False
        self._ema_freeze_held = 0
        
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
        self.DELTA_THRESHOLD = 0.000356  # Minimum delta to apply gripper action
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
        try:
            mode = getattr(self.tactile_adapter.adapter_cfg, "tactile_mode", "tactile_only")
            print(f"   • Model tactile_mode: {mode}")
        except Exception:
            pass
        print(f"\n Press Ctrl+C for emergency stop")
        print(f"{'='*70}\n")

    def _check_toggle(self):
        """Check if toggle key was pressed (rising-edge detection)."""
        if self.keyboard is None:
            return False
        
        current_state = self.keyboard.buttons.get(TOGGLE_KEY, False)
        
        toggled = False
        if current_state and not self._last_toggle_state:
            self.model_enabled = not self.model_enabled
            print("\n" + "="*70)
            if self.model_enabled:
                print("TACTILE MODEL ENABLED - GRIPPER AUTO-CONTROL ACTIVE ")
                print("   The tactile model will now control the gripper automatically!")
                print("   You control the arm with SpaceMouse, model controls gripper.")
                self.tactile_adapter.reset_force_safety_state()
            else:
                print("TACTILE MODEL DISABLED - MANUAL CONTROL ")
                print("   Gripper control returned to manual mode.")
                self._target_ema = None
                self._target_in_deadzone = False
                self._force_cooldown_counter = 0
                self._fallback_duty_counter = 0
                self._grasp_stable_counter = 0
                self._ema_frozen = False
                self._ema_freeze_held = 0
            print("="*70 + "\n")
            toggled = True
        
        self._last_toggle_state = current_state
        return toggled


    def _clamp_gripper_delta(self, delta):
        """Clamp gripper delta to safe range."""
        return np.clip(delta, GRIPPER_DELTA_MIN, GRIPPER_DELTA_MAX)

    def forward(self, obs_dict, include_info=False):
        """
        Forward pass with optional tactile model gripper override.
        
        Returns action with:
        - Arm joints: from manual teleop (HITLPolicy)
        - Gripper: from tactile model if enabled, else from manual teleop
        """
        teleop_start = time.perf_counter()
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
        teleop_done = time.perf_counter()
        teleop_t0 = teleop_done  # Timeline origin: 0 ms at end of teleop
        teleop_ms = (teleop_done - teleop_start) * 1000.0
        
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
            
            preprocess_start = time.perf_counter()
            wrist_left = preprocess_image(wrist_left)
            wrist_right = preprocess_image(wrist_right)
            preprocess_done = time.perf_counter()
            image_preprocess_ms = (preprocess_done - preprocess_start) * 1000.0
            t_after_preprocess_ms = (preprocess_done - teleop_t0) * 1000.0
            
            # Get tactile history
            sensor_start = time.perf_counter()
            tactile_history = self.tactile_reader.read_values()
            sensor_done = time.perf_counter()
            sensor_read_ms = (sensor_done - sensor_start) * 1000.0
            t_after_tactile_ms = (sensor_done - teleop_t0) * 1000.0

            # Get force history when model mode needs it
            force_history = None
            force_read_ms = 0.0
            t_after_force_ms = t_after_tactile_ms
            adapter_cfg = getattr(self.tactile_adapter, "adapter_cfg", None)
            tactile_mode = getattr(adapter_cfg, "tactile_mode", "tactile_only")
            safety_enabled = bool(getattr(adapter_cfg, "force_safety_enabled", False))
            need_force_history = tactile_mode in {"force_only", "force_tactile"} or safety_enabled
            if need_force_history:
                force_start = time.perf_counter()
                force_history = self.tactile_reader.read_force_values(
                    k=getattr(self.tactile_adapter.adapter_cfg, "tactile_length", 50)
                )
                force_done = time.perf_counter()
                force_read_ms = (force_done - force_start) * 1000.0
                t_after_force_ms = (force_done - teleop_t0) * 1000.0
            
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
                
                # Adapter already returns delta in original (de-normalized) scale.
                model_start = time.perf_counter()
                model_delta = self.tactile_adapter.predict_delta(
                    image_left=wrist_left,
                    image_right=wrist_right,
                    tactile_history=tactile_history,
                    force_history=force_history,
                    step_index=0  # Use first step from action chunk
                )
                model_done = time.perf_counter()
                model_ms = (model_done - model_start) * 1000.0
                t_model_start_ms = (model_start - teleop_t0) * 1000.0
                t_model_end_ms = (model_done - teleop_t0) * 1000.0
                timings = getattr(self.tactile_adapter, "last_timings", {}) or {}

                
                delta = float(model_delta)
                
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

                # Retrieve auxiliary model outputs (force, final_target).
                predicted_force = None
                last_outputs = getattr(self.tactile_adapter, "last_model_outputs", {}) or {}
                force_chunk = last_outputs.get("force")
                final_target_arr = last_outputs.get("final_target")
                model_final_target = None
                if final_target_arr is not None:
                    _ft_val = float(final_target_arr[0])
                    if np.isfinite(_ft_val):
                        model_final_target = _ft_val
                if force_chunk is not None:
                    force_chunk = np.asarray(force_chunk).reshape(-1)
                    if force_chunk.size > 0:
                        predicted_force = float(force_chunk[0])
                delta_before_force_safety = float(delta)
                delta, force_safety_info = self.tactile_adapter.apply_force_safety_delta(
                    proposed_delta=delta,
                    current_gripper_position=current_gripper,
                    force_history=force_history,
                    predicted_force=predicted_force,
                )
                if not force_safety_info.get("baseline_init_done", True):
                    if self._print_counter % 10 == 0:
                        _raw = force_safety_info.get("raw_force")
                        _bl = force_safety_info.get("baseline")
                        _corr = force_safety_info.get("force_corrected")
                        _filt = force_safety_info.get("force")
                        print(
                            f"[FORCE BASELINE INIT] "
                            f"raw={_raw:.1f}, baseline={_bl}, "
                            f"corrected={_corr:.1f}, filtered={_filt:.1f}"
                        )
                elif force_safety_info.get("triggered") and self._print_counter % 30 == 0:
                    reason = force_safety_info.get("reason")
                    force_now = force_safety_info.get("force")
                    dforce_now = force_safety_info.get("dforce")
                    print(
                        "[FORCE SAFETY] "
                        f"reason={reason} delta {delta_before_force_safety:+.5f}->{delta:+.5f} "
                        f"force={force_now} dforce={dforce_now}"
                    )
                
                # ===== STEP 3: cooldown gate =====
                if force_safety_info.get("triggered"):
                    self._force_cooldown_counter = FORCE_SAFETY_COOLDOWN_FRAMES

                if self._force_cooldown_counter > 0:
                    self._force_cooldown_counter -= 1
                    delta = 0.0
                else:
                    # ===== STEP 4: main delta still useful? =====
                    fallback_active = (
                        abs(delta) < self.DELTA_THRESHOLD
                        and model_final_target is not None
                        and np.isfinite(current_gripper)
                    )

                    if not fallback_active:
                        self._fallback_duty_counter = 0

                    if fallback_active:
                        # ===== STEP 5: fallback =====
                        force_now = force_safety_info.get("force")

                        # --- stable grasp detection (freeze/unfreeze EMA) ---
                        if self._ema_frozen:
                            self._ema_freeze_held += 1
                            can_unfreeze = (
                                self._ema_freeze_held >= GRASP_FREEZE_MIN_FRAMES
                            )
                            if can_unfreeze and (
                                force_now is None
                                or abs(force_now) < GRASP_STABLE_FORCE_EXIT
                            ):
                                self._ema_frozen = False
                                self._ema_freeze_held = 0
                                self._grasp_stable_counter = 0
                        else:
                            if (
                                force_now is not None
                                and abs(force_now) > GRASP_STABLE_FORCE_ENTER
                                and abs(delta) < GRASP_STABLE_DELTA_EPS
                            ):
                                self._grasp_stable_counter += 1
                                if self._grasp_stable_counter >= GRASP_STABLE_FRAMES:
                                    self._ema_frozen = True
                                    self._ema_freeze_held = 0
                            else:
                                self._grasp_stable_counter = 0

                        # --- EMA update (gated by freeze) ---
                        if not self._ema_frozen:
                            if self._target_ema is None:
                                self._target_ema = model_final_target
                            else:
                                self._target_ema += TARGET_FALLBACK_EMA_ALPHA * (
                                    model_final_target - self._target_ema
                                )

                        error = self._target_ema - current_gripper

                        # --- hysteresis deadzone ---
                        if self._target_in_deadzone:
                            if abs(error) > TARGET_FALLBACK_DZ_EXIT:
                                self._target_in_deadzone = False
                        else:
                            if abs(error) < TARGET_FALLBACK_DZ_ENTER:
                                self._target_in_deadzone = True

                        if not self._target_in_deadzone:
                            raw_step = TARGET_FALLBACK_P_GAIN * abs(error)
                            clamped_step = max(
                                min(raw_step, TARGET_FALLBACK_STEP_MAX),
                                TARGET_FALLBACK_STEP_MIN,
                            )
                            fallback_delta = np.sign(error) * clamped_step

                            # --- duty cycle ---
                            self._fallback_duty_counter += 1
                            if self._fallback_duty_counter >= TARGET_FALLBACK_DUTY_PERIOD:
                                self._fallback_duty_counter = 0
                                # ===== STEP 6: fallback re-veto =====
                                if force_safety_info.get("hard_latched"):
                                    pass
                                else:
                                    delta = fallback_delta
                                    if self._print_counter % 30 == 0:
                                        print(
                                            f"[TARGET FALLBACK] ema={self._target_ema:.4f}, "
                                            f"err={error:+.4f}, step={fallback_delta:+.5f}, "
                                            f"frozen={self._ema_frozen}"
                                        )

                # ===== STEP 7: final clamp (covers both main delta and fallback) =====
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
                    
                    gripper_velocity = 20* delta / max_gripper_delta
                    if abs(gripper_velocity) < 0.056206943867595054:
                        gripper_velocity = 0.0

                    print("delta:", delta * 0.5)
                    print("gripper_velocity", gripper_velocity)
                    # Clamp velocity to [-1, 1] range (standard action range)
                    gripper_velocity = np.clip(gripper_velocity, -1.0, 1.0)
                    
                    # Override gripper in action (last dimension) with velocity command
                    action = np.array(action, dtype=np.float32).copy()
                    action[-1] = gripper_velocity
                    t_after_override_ms = (time.perf_counter() - teleop_t0) * 1000.0
                    
                    # Compute target for display
                    target_gripper = current_gripper + delta * 0.5
                    target_gripper = np.clip(target_gripper, GRIPPER_POSITION_MIN, GRIPPER_POSITION_MAX)
                    
                    # Print tactile model output (every 30 steps to reduce clutter)
                    self._print_counter += 1
                    if self._print_counter % 30 == 0:
                        ft_str = f", ft={model_final_target:.3f}" if model_final_target is not None else ""
                        ema_str = f", ema={self._target_ema:.3f}" if self._target_ema is not None else ""
                        print(f"[TACTILE MODEL ACTIVE] Gripper Position: {current_gripper:.3f} → {target_gripper:.3f} "
                              f"(delta: {delta:+.4f}, model_delta: {model_delta:+.4f}, vel: {gripper_velocity:+.3f}"
                              f"{ft_str}{ema_str})")
                else:
                    # Position mode: use absolute position
                    target_gripper = current_gripper + delta
                    target_gripper = np.clip(target_gripper, GRIPPER_POSITION_MIN, GRIPPER_POSITION_MAX)
                    
                    # Override gripper in action (last dimension) with position command
                    action = np.array(action, dtype=np.float32).copy()
                    action[-1] = target_gripper
                    t_after_override_ms = (time.perf_counter() - teleop_t0) * 1000.0
                    
                    # Print tactile model output (every 30 steps to reduce clutter)
                    self._print_counter += 1
                    if self._print_counter % 30 == 0:
                        ft_str = f", ft={model_final_target:.3f}" if model_final_target is not None else ""
                        ema_str = f", ema={self._target_ema:.3f}" if self._target_ema is not None else ""
                        print(f"[TACTILE MODEL ACTIVE] Gripper: {current_gripper:.3f} → {target_gripper:.3f} "
                              f"(delta: {delta:+.4f}, model_delta: {model_delta:+.4f}"
                              f"{ft_str}{ema_str})")

                if self._timing_enabled:
                    self._timing_counter += 1
                    if self._timing_counter % self._timing_every == 0:
                        timeline_parts = [
                            f"teleop=0.000ms (teleop_ms={teleop_ms:.3f}ms)",
                            f"after_image_preprocess={t_after_preprocess_ms:.3f}ms",
                            f"after_tactile_read={t_after_tactile_ms:.3f}ms",
                            f"after_force_read={t_after_force_ms:.3f}ms",
                            f"model_start={t_model_start_ms:.3f}ms",
                        ]

                        cumulative = 0.0
                        def add_stage(label, key):
                            nonlocal cumulative
                            if key in timings:
                                cumulative += float(timings[key])
                                timeline_parts.append(f"{label}={t_model_start_ms + cumulative:.3f}ms")

                        # Ordered model stages (if adapter provides them)
                        add_stage("img_prepare_left", "prepare_image_left_ms")
                        add_stage("img_prepare_right", "prepare_image_right_ms")
                        add_stage("tactile_prepare", "prepare_tactile_ms")
                        add_stage("image_encoder_left", "image_encoder_left_ms")
                        add_stage("image_encoder_right", "image_encoder_right_ms")
                        add_stage("tactile_encoder", "tactile_encoder_ms")
                        add_stage("transformer", "transformer_ms")
                        add_stage("action_head", "action_head_ms")

                        timeline_parts.append(f"model_end={t_model_end_ms:.3f}ms")
                        timeline_parts.append(f"gripper_override={t_after_override_ms:.3f}ms")
                        timeline_parts.append(f"image_preprocess_ms={image_preprocess_ms:.3f}ms")
                        timeline_parts.append(f"tactile_read_ms={sensor_read_ms:.3f}ms")
                        timeline_parts.append(f"force_read_ms={force_read_ms:.3f}ms")
                        timeline_parts.append(f"model_total_ms={model_ms:.3f}ms")
                        print("[TIMELINE] " + ", ".join(timeline_parts))
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
            controller = TactileGripperController(
                hitl_policy=hitl_policy,
                tactile_adapter=tactile_adapter,
                tactile_reader=tactile_reader,
                keyboard=devices["keyboard"]
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
        print("Resources cleaned up.")


if __name__ == "__main__":
    main()
