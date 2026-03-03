

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
from r2d2.misc.parameters import hand_camera_id, varied_camera_1_id, varied_camera_2_id

cwd = os.getcwd()
sys.path.append(cwd)

from devices import SpaceMouse, Keyboard
from interfaces import HITLPolicy, HumanInterventionReader

# Ensure openpi_client is importable (for pi0 policy + image tools).
try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy
except Exception:
    openpi_client_paths = [
        Path("/home/pi0/multi-modal/openpi/packages/openpi-client/src"),
        Path("/home/pi0/multi-modal/openpi-multi-modal/packages/openpi-client/src"),
    ]
    for _path in openpi_client_paths:
        if _path.exists():
            sys.path.insert(0, str(_path))
            break
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

# Add tactile_module to path
sys.path.insert(0, "/home/pi0/multi-modal/tactile_module")
from robot_inference_adapter import TactileGripperAdapter

# Model checkpoint path
# TACTILE_MODEL_CHECKPOINT = "/home/pi0/multi-modal/tactile_module/checkpoints/two_img_delta_gripper_normalized.pt"
# TACTILE_MODEL_CHECKPOINT = "/home/pi0/multi-modal/tactile_module/checkpoints/paper_cup_two_img_delta_gripper_normalized.pt"
# TACTILE_MODEL_CHECKPOINT = "/home/pi0/multi-modal/tactile_module/checkpoints/1224_paper_cup_two_img_delta_gripper_normalized.pt"
TACTILE_MODEL_CHECKPOINT = "/home/pi0/multi-modal/checkpoints/box_and_cup/checkpoints/ab_full_C4_seed42_drop_demo_20_20260220_203907_best.pt"
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

# Optional: auto-generate gripper normalization stats from a dataset.
# Set to None/"" to disable, or override via env TACTILE_GRIPPER_STATS_SOURCE_H5.
TACTILE_GRIPPER_STATS_SOURCE_H5 = os.getenv(
    "TACTILE_GRIPPER_STATS_SOURCE_H5",
    "/home/pi0/multi-modal/robomimic_output/action_target_gripper_position_50target_horizon_none_papercup_and_box_haiyi_100_delta.hdf5",
).strip()
if not TACTILE_GRIPPER_STATS_SOURCE_H5:
    TACTILE_GRIPPER_STATS_SOURCE_H5 = None
# Optional: override output stats path; "auto" writes next to the config file.
TACTILE_GRIPPER_STATS_OUTPUT = os.getenv("TACTILE_GRIPPER_STATS_OUTPUT", "auto").strip()
# Optional: "train" (default), "val", or "" for all data.
TACTILE_GRIPPER_STATS_SPLIT = os.getenv("TACTILE_GRIPPER_STATS_SPLIT", "train").strip().lower()
if TACTILE_GRIPPER_STATS_SPLIT in {"", "none", "all"}:
    TACTILE_GRIPPER_STATS_SPLIT = None
# Optional: recompute even if cached output exists.
TACTILE_GRIPPER_STATS_REFRESH = os.getenv("TACTILE_GRIPPER_STATS_REFRESH", "0") == "1"

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
        if TACTILE_GRIPPER_STATS_SOURCE_H5:
            print(f"[INFO] Using base config for auto-generated stats: {fb}")
        else:
            print(f"[INFO] Using fallback config: {fb}")
        return str(fb.resolve())

    raise FileNotFoundError(
        "Failed to resolve TACTILE_MODEL_CONFIG. Set explicit path or ensure fallback exists."
    )


def _resolve_gripper_stats_output_path(config_path, source_h5, output_spec):
    config_file = Path(config_path).expanduser().resolve()
    config_dir = config_file.parent
    if output_spec and str(output_spec).strip().lower() not in {"auto", ""}:
        out = Path(str(output_spec)).expanduser()
        if not out.is_absolute():
            out = (config_dir / out).resolve()
        return out

    dataset_tag = _sanitize_filename(Path(source_h5).stem)
    config_tag = _sanitize_filename(config_file.stem)
    return config_dir / f"{config_tag}__auto_{dataset_tag}_gripper_norm_stats.json"


def _autogenerate_gripper_stats_from_h5(source_h5, gripper_key, split, output_path, refresh=False):
    source_path = Path(source_h5).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Gripper stats source dataset not found: {source_path}")

    if output_path.exists() and not refresh:
        if output_path.stat().st_mtime >= source_path.stat().st_mtime:
            print(f"[INFO] Using cached gripper normalization stats: {output_path}")
            return output_path

    try:
        from compute_normalization_stats import compute_stats
    except Exception as e:
        raise ImportError(
            f"Failed to import compute_normalization_stats (requires h5py). Error: {e}"
        ) from e

    print(f"[INFO] Auto-generating gripper normalization stats from: {source_path}")
    stats = compute_stats(str(source_path), gripper_key=str(gripper_key), split=split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2))
    print(f"[INFO] Wrote gripper normalization stats to: {output_path}")
    return output_path


def _write_config_with_gripper_stats(config_path, stats_path):
    config_file = Path(config_path).expanduser().resolve()
    with config_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must be a YAML mapping: {config_file}")

    rob = cfg.setdefault("robomimic", {})
    rob["normalize_gripper"] = True
    gripper_norm_cfg = rob.setdefault("gripper_normalization", {})
    if "type" not in gripper_norm_cfg:
        print("[WARNING] gripper_normalization.type missing; defaulting to 'symmetric'")
        gripper_norm_cfg["type"] = "symmetric"

    config_dir = config_file.parent
    try:
        rel_stats = stats_path.relative_to(config_dir)
        gripper_norm_cfg["stats_file"] = str(rel_stats)
    except Exception:
        gripper_norm_cfg["stats_file"] = str(stats_path)

    derived_path = config_dir / f"{config_file.stem}__autostats.yaml"
    with derived_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[INFO] Wrote derived config with auto stats: {derived_path}")
    return str(derived_path)


def maybe_autogenerate_gripper_stats(config_path):
    if not TACTILE_GRIPPER_STATS_SOURCE_H5:
        return None

    config_file = Path(config_path).expanduser().resolve()
    with config_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    rob = cfg.get("robomimic", {}) if isinstance(cfg, dict) else {}
    gripper_key = rob.get("gripper_key", "obs/delta_gripper_position")

    output_path = _resolve_gripper_stats_output_path(
        config_path,
        TACTILE_GRIPPER_STATS_SOURCE_H5,
        TACTILE_GRIPPER_STATS_OUTPUT,
    )
    stats_path = _autogenerate_gripper_stats_from_h5(
        TACTILE_GRIPPER_STATS_SOURCE_H5,
        gripper_key,
        TACTILE_GRIPPER_STATS_SPLIT,
        output_path,
        refresh=TACTILE_GRIPPER_STATS_REFRESH,
    )
    return _write_config_with_gripper_stats(config_path, stats_path)


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
_auto_cfg = maybe_autogenerate_gripper_stats(TACTILE_MODEL_CONFIG)
if _auto_cfg:
    TACTILE_MODEL_CONFIG = _auto_cfg
TACTILE_GRIPPER_STATS_OVERRIDE = None
print(f"[INFO] Using tactile model config: {TACTILE_MODEL_CONFIG}")

# Load normalization parameters from config/stats file
_norm_stats = load_normalization_stats(
    TACTILE_MODEL_CONFIG,
    stats_file_path=TACTILE_GRIPPER_STATS_OVERRIDE,
)
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
# Force-zero calibration policy:
# collect initial force baseline for N steps, then enable normal gripper actions.
FORCE_ZERO_BASELINE_STEPS = max(1, int(os.getenv("FORCE_ZERO_BASELINE_STEPS", "150")))
FORCE_ZERO_BLOCK_ACTION_UNTIL_READY = os.getenv("FORCE_ZERO_BLOCK_ACTION_UNTIL_READY", "1") != "0"

# Toggle key
TOGGLE_KEY = "t"

# PI0 policy integration (arm control + gripper gating)
PI0_REMOTE_HOST = os.getenv("PI0_REMOTE_HOST", "127.0.1.1").strip()
PI0_REMOTE_PORT = int(os.getenv("PI0_REMOTE_PORT", "8000"))
PI0_OPEN_LOOP_HORIZON = int(os.getenv("PI0_OPEN_LOOP_HORIZON", "1"))
PI0_ARM_ACTION_MODE = os.getenv("PI0_ARM_ACTION_MODE", "joint_position").strip().lower()
PI0_ARM_SWITCH_SEC = float(os.getenv("PI0_ARM_SWITCH_SEC", "0.35"))
PI0_OVERRIDE_KEY = os.getenv("PI0_OVERRIDE_KEY", "p").strip() or "p"
PI0_GRIPPER_HYSTERESIS_EPS = float(os.getenv("PI0_GRIPPER_HYSTERESIS_EPS", "0.0075"))
PI0_IMAGE_SIZE = int(os.getenv("PI0_IMAGE_SIZE", "224"))
PI0_MAX_DT = float(os.getenv("PI0_MAX_DT", "0.1"))
PI0_DEFAULT_DT = 1.0 / float(os.getenv("PI0_DEFAULT_HZ", "100"))
PI0_JOINT_DELTA_MAX = float(os.getenv("PI0_JOINT_DELTA_MAX", "0.2"))
PI0_ARM_POS_EPS = float(os.getenv("PI0_ARM_POS_EPS", "0.003"))
PI0_ARM_EMA_ALPHA = float(os.getenv("PI0_ARM_EMA_ALPHA", "0.3"))
PI0_ARM_DA_MAX = float(os.getenv("PI0_ARM_DA_MAX", "0.12"))
PI0_GRIPPER_DELTA_MAX = float(os.getenv("PI0_GRIPPER_DELTA_MAX", "0.25"))
PI0_DEBUG_ACTION_PRINT = os.getenv("PI0_DEBUG_ACTION_PRINT", "1") != "0"
PI0_DEBUG_ACTION_EVERY = max(1, int(os.getenv("PI0_DEBUG_ACTION_EVERY", "1")))
PI0_DEBUG_ARM_MAP_PRINT = os.getenv("PI0_DEBUG_ARM_MAP_PRINT", "1") != "0"
PI0_DEBUG_ARM_MAP_EVERY = max(1, int(os.getenv("PI0_DEBUG_ARM_MAP_EVERY", "10")))

# ---------- generalized grasp state machine ----------
TACTILE_CONTACT_ABS_THR = float(os.getenv("TACTILE_CONTACT_ABS_THR", "0.08"))
TACTILE_CONTACT_Z_THR = float(os.getenv("TACTILE_CONTACT_Z_THR", "0.12"))
TACTILE_CONTACT_CONSEC = max(1, int(os.getenv("TACTILE_CONTACT_CONSEC", "4")))
TACTILE_PEAK_EMA_ALPHA = float(os.getenv("TACTILE_PEAK_EMA_ALPHA", "0.2"))
TACTILE_PEAK_THR = float(os.getenv("TACTILE_PEAK_THR", "0.012"))
TACTILE_PEAK_CONSEC = max(1, int(os.getenv("TACTILE_PEAK_CONSEC", "4")))
TACTILE_PEAK_MIN_PROGRESS = float(os.getenv("TACTILE_PEAK_MIN_PROGRESS", "0.5"))
TACTILE_TRAJ_NOMINAL_STEPS = max(1, int(os.getenv("TACTILE_TRAJ_NOMINAL_STEPS", "300")))
TACTILE_HARD_FORCE_ENTER = float(os.getenv("TACTILE_HARD_FORCE_ENTER", "300"))
TACTILE_HARD_FORCE_EXIT = float(os.getenv("TACTILE_HARD_FORCE_EXIT", "220"))
TACTILE_HARD_TACTILE_ENTER = float(os.getenv("TACTILE_HARD_TACTILE_ENTER", "0.28"))
TACTILE_HARD_TACTILE_EXIT = float(os.getenv("TACTILE_HARD_TACTILE_EXIT", "0.20"))
TACTILE_HARD_MIN_HOLD_FRAMES = max(1, int(os.getenv("TACTILE_HARD_MIN_HOLD_FRAMES", "15")))
TACTILE_HARD_RELEASE_DELTA = float(os.getenv("TACTILE_HARD_RELEASE_DELTA", "-0.002"))
TACTILE_FORCE_SOFT = float(os.getenv("TACTILE_FORCE_SOFT", "200"))
TACTILE_TACTILE_SOFT = float(os.getenv("TACTILE_TACTILE_SOFT", "0.25"))
TACTILE_SECURE_CONTACT_THR = float(os.getenv("TACTILE_SECURE_CONTACT_THR", "0.18"))
TACTILE_SECURE_FORCE_STD_THR = float(os.getenv("TACTILE_SECURE_FORCE_STD_THR", "15"))
TACTILE_SECURE_WINDOW = max(2, int(os.getenv("TACTILE_SECURE_WINDOW", "8")))
TACTILE_SECURE_MAX_CLOSE_DELTA = float(os.getenv("TACTILE_SECURE_MAX_CLOSE_DELTA", "0.0005"))
TACTILE_CONTACT_MAX_CLOSE_DELTA = float(os.getenv("TACTILE_CONTACT_MAX_CLOSE_DELTA", "0.0003"))
TACTILE_SECURE_BLOCK_POSITIVE = os.getenv("TACTILE_SECURE_BLOCK_POSITIVE", "1") != "0"
TACTILE_ROBUST_BASELINE_STEPS = max(1, int(os.getenv("TACTILE_ROBUST_BASELINE_STEPS", "30")))
TACTILE_STATE_LOG_ENABLE = os.getenv("TACTILE_STATE_LOG_ENABLE", "1") != "0"
TACTILE_STATE_LOG_EVERY = max(1, int(os.getenv("TACTILE_STATE_LOG_EVERY", "1")))


class _GraspStateMachineMixin:
    _GRASP_STATE_TO_CODE = {
        "APPROACH": 0,
        "CONTACT": 1,
        "SECURE": 2,
    }

    def _init_grasp_state_machine(self, label):
        self._grasp_label = label
        self._grasp_state = "APPROACH"
        self._contact_consec = 0
        self._peak_consec = 0
        self._hard_latched = False
        self._hard_hold_frames = 0
        self._robust_frozen = False
        self._traj_step = 0
        self._contact_peak_ema = 0.0
        self._contact_peak_raw = 0.0
        self._tactile_peak_for_protect = 0.0
        self._contact_abs = 0.0
        self._contact_z2 = 0.0
        self._force_mean = float("nan")
        self._force_max = float("nan")
        self._risk_score = 0.0
        self._secure_contact_hist = []
        self._secure_force_hist = []
        self._robust_abs_buffer = []
        self._robust_median = np.zeros(6, dtype=np.float32)
        self._robust_iqr = np.ones(6, dtype=np.float32)
        self._grasp_last_info = {}
        self._state_log_counter = 0
        self._state_log_path = self._resolve_state_log_path()

    def _reset_grasp_state_machine(self):
        self._grasp_state = "APPROACH"
        self._contact_consec = 0
        self._peak_consec = 0
        self._hard_latched = False
        self._hard_hold_frames = 0
        self._robust_frozen = False
        self._traj_step = 0
        self._contact_peak_ema = 0.0
        self._contact_peak_raw = 0.0
        self._tactile_peak_for_protect = 0.0
        self._contact_abs = 0.0
        self._contact_z2 = 0.0
        self._force_mean = float("nan")
        self._force_max = float("nan")
        self._risk_score = 0.0
        self._secure_contact_hist.clear()
        self._secure_force_hist.clear()
        self._robust_abs_buffer.clear()
        self._robust_median.fill(0.0)
        self._robust_iqr.fill(1.0)
        self._grasp_last_info = {}

    def _resolve_state_log_path(self):
        if not TACTILE_STATE_LOG_ENABLE:
            return None
        explicit = os.getenv("TACTILE_STATE_LOG_PATH", "").strip()
        if explicit:
            out = Path(explicit).expanduser()
            if not out.is_absolute():
                out = (Path.cwd() / out).resolve()
        elif _RUN_LOG_PATH:
            out = Path(_RUN_LOG_PATH).with_suffix(Path(_RUN_LOG_PATH).suffix + ".grasp_state.jsonl")
        else:
            log_dir = Path(os.getenv("TACTILE_RUN_LOG_DIR", TACTILE_RUN_LOG_DIR)).expanduser()
            if not log_dir.is_absolute():
                log_dir = (Path.cwd() / log_dir).resolve()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = log_dir / f"{ts}__{_sanitize_filename(self._grasp_label)}.grasp_state.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def _configure_force_baseline_warmup(self):
        adapter = getattr(self, "tactile_adapter", None)
        if adapter is None:
            return
        adapter_cfg = getattr(adapter, "adapter_cfg", None)
        if adapter_cfg is None:
            return
        if not bool(getattr(adapter_cfg, "force_safety_enabled", False)):
            return

        baseline_steps = int(FORCE_ZERO_BASELINE_STEPS)
        old_steps = int(max(1, getattr(adapter_cfg, "force_safety_baseline_init_samples", baseline_steps)))
        adapter_cfg.force_safety_baseline_init_samples = baseline_steps
        print(
            "[INFO] Force baseline warmup configured: "
            f"init_samples {old_steps} -> {baseline_steps}, "
            f"block_actions_until_ready={FORCE_ZERO_BLOCK_ACTION_UNTIL_READY}"
        )

    def _to_tactile_history_2d(self, tactile_history):
        arr = np.asarray(tactile_history, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((1, 6), dtype=np.float32)
        if arr.ndim == 1:
            if arr.size % 6 == 0:
                arr = arr.reshape(-1, 6)
            else:
                arr = np.pad(arr, (0, max(0, 6 - arr.size)), mode="constant")[:6].reshape(1, 6)
        elif arr.ndim >= 2:
            arr = arr.reshape(-1, arr.shape[-1])
            if arr.shape[1] == 6:
                pass
            elif arr.shape[0] == 6 and arr.shape[1] != 6:
                arr = arr.T
            elif arr.shape[1] > 6:
                arr = arr[:, :6]
            elif arr.shape[1] < 6:
                arr = np.pad(arr, ((0, 0), (0, 6 - arr.shape[1])), mode="constant")
        return arr.astype(np.float32, copy=False)

    def _to_force_history_1d(self, force_history):
        if force_history is None:
            return np.array([], dtype=np.float32)
        arr = np.asarray(force_history, dtype=np.float32).reshape(-1)
        return arr[np.isfinite(arr)]

    def _update_contact_and_force_metrics(self, tactile_history, force_history):
        tactile_2d = self._to_tactile_history_2d(tactile_history)
        abs_hist = np.abs(tactile_2d)

        if not self._robust_frozen and self._grasp_state == "APPROACH":
            self._robust_abs_buffer.append(abs_hist)
            if len(self._robust_abs_buffer) > TACTILE_ROBUST_BASELINE_STEPS:
                self._robust_abs_buffer.pop(0)

        if self._robust_abs_buffer:
            baseline = np.concatenate(self._robust_abs_buffer, axis=0)
        else:
            baseline = abs_hist

        q25 = np.percentile(baseline, 25.0, axis=0)
        q50 = np.percentile(baseline, 50.0, axis=0)
        q75 = np.percentile(baseline, 75.0, axis=0)
        self._robust_median = q50.astype(np.float32)
        self._robust_iqr = np.maximum((q75 - q25), 1e-6).astype(np.float32)

        z = (abs_hist - self._robust_median[None, :]) / self._robust_iqr[None, :]
        self._contact_abs = float(np.mean(abs_hist > 0.01))
        self._contact_z2 = float(np.mean(z > 2.0))
        self._contact_peak_raw = float(np.percentile(abs_hist, 99.0))
        self._contact_peak_ema = (
            TACTILE_PEAK_EMA_ALPHA * self._contact_peak_raw
            + (1.0 - TACTILE_PEAK_EMA_ALPHA) * float(self._contact_peak_ema)
        )
        self._tactile_peak_for_protect = float(self._contact_peak_ema)

        force_1d = self._to_force_history_1d(force_history)
        if force_1d.size:
            self._force_mean = float(np.mean(force_1d))
            self._force_max = float(np.max(force_1d))
        else:
            self._force_mean = float("nan")
            self._force_max = float("nan")

        self._traj_step += 1
        progress = min(1.0, self._traj_step / float(TACTILE_TRAJ_NOMINAL_STEPS))

        main_contact_now = (
            self._contact_abs >= TACTILE_CONTACT_ABS_THR
            and self._contact_z2 >= TACTILE_CONTACT_Z_THR
        )
        if main_contact_now:
            self._contact_consec += 1
        else:
            self._contact_consec = 0

        peak_contact_now = (
            progress >= TACTILE_PEAK_MIN_PROGRESS
            and self._contact_peak_ema >= TACTILE_PEAK_THR
        )
        if peak_contact_now:
            self._peak_consec += 1
        else:
            self._peak_consec = 0

        main_triggered = self._contact_consec >= TACTILE_CONTACT_CONSEC
        peak_triggered = self._peak_consec >= TACTILE_PEAK_CONSEC

        if self._grasp_state == "APPROACH" and (main_triggered or peak_triggered):
            self._grasp_state = "CONTACT"
            self._robust_frozen = True

        self._secure_contact_hist.append(self._contact_abs)
        self._secure_force_hist.append(self._force_mean)
        if len(self._secure_contact_hist) > TACTILE_SECURE_WINDOW:
            self._secure_contact_hist.pop(0)
            self._secure_force_hist.pop(0)

        if self._grasp_state == "CONTACT" and len(self._secure_contact_hist) >= TACTILE_SECURE_WINDOW:
            contact_mean = float(np.mean(self._secure_contact_hist))
            force_hist = np.asarray(self._secure_force_hist, dtype=np.float32)
            finite_force = force_hist[np.isfinite(force_hist)]
            force_std = float(np.std(finite_force)) if finite_force.size else 0.0
            if contact_mean >= TACTILE_SECURE_CONTACT_THR and force_std <= TACTILE_SECURE_FORCE_STD_THR:
                self._grasp_state = "SECURE"

    def _update_hard_protect_latch(self):
        force_hit = np.isfinite(self._force_max) and self._force_max >= TACTILE_HARD_FORCE_ENTER
        tactile_hit = self._tactile_peak_for_protect >= TACTILE_HARD_TACTILE_ENTER
        if not self._hard_latched and (force_hit or tactile_hit):
            self._hard_latched = True
            self._hard_hold_frames = 0

        if self._hard_latched:
            self._hard_hold_frames += 1
            force_clear = (not np.isfinite(self._force_max)) or (self._force_max <= TACTILE_HARD_FORCE_EXIT)
            tactile_clear = self._tactile_peak_for_protect <= TACTILE_HARD_TACTILE_EXIT
            if (
                self._hard_hold_frames >= TACTILE_HARD_MIN_HOLD_FRAMES
                and force_clear
                and tactile_clear
            ):
                self._hard_latched = False
                self._hard_hold_frames = 0

    def _apply_grasp_state_delta_postprocess(self, delta):
        delta = float(delta)
        force_ratio = 0.0
        if np.isfinite(self._force_max):
            force_ratio = max(0.0, self._force_max / max(TACTILE_FORCE_SOFT, 1e-6))
        tactile_ratio = max(0.0, self._tactile_peak_for_protect / max(TACTILE_TACTILE_SOFT, 1e-6))
        self._risk_score = float(np.clip(max(force_ratio, tactile_ratio), 0.0, 1.0))

        if self._grasp_state in {"CONTACT", "SECURE"} and delta > 0:
            delta *= (1.0 - self._risk_score)

        if self._grasp_state == "CONTACT" and delta > TACTILE_CONTACT_MAX_CLOSE_DELTA:
            delta = TACTILE_CONTACT_MAX_CLOSE_DELTA

        if self._grasp_state == "SECURE" and delta > 0.0:
            if TACTILE_SECURE_BLOCK_POSITIVE:
                delta = 0.0
            elif delta > TACTILE_SECURE_MAX_CLOSE_DELTA:
                delta = TACTILE_SECURE_MAX_CLOSE_DELTA

        self._update_hard_protect_latch()
        if self._hard_latched and delta > 0.0:
            delta = TACTILE_HARD_RELEASE_DELTA

        return float(delta)

    def _record_grasp_step(self, payload):
        self._grasp_last_info = {
            "grasp_state": self._grasp_state,
            "contact_abs": float(self._contact_abs),
            "contact_z2": float(self._contact_z2),
            "contact_peak_raw": float(self._contact_peak_raw),
            "contact_peak_ema": float(self._contact_peak_ema),
            "contact_consec_count": int(self._contact_consec),
            "peak_consec_count": int(self._peak_consec),
            "hard_protect_latched": bool(self._hard_latched),
            "hard_protect_hold_frames": int(self._hard_hold_frames),
            "force_mean": float(self._force_mean) if np.isfinite(self._force_mean) else None,
            "force_max": float(self._force_max) if np.isfinite(self._force_max) else None,
            "risk_score": float(self._risk_score),
            "delta_model_raw": float(payload.get("delta_model_raw", 0.0)),
            "delta_after_small_delta_gate": float(payload.get("delta_after_small_delta_gate", 0.0)),
            "delta_after_force_safety": float(payload.get("delta_after_force_safety", 0.0)),
            "delta_final_applied": float(payload.get("delta_final_applied", 0.0)),
        }

        if not self._state_log_path:
            return
        self._state_log_counter += 1
        if self._state_log_counter % TACTILE_STATE_LOG_EVERY != 0:
            return

        rec = dict(self._grasp_last_info)
        rec["step"] = int(self._traj_step)
        rec["controller"] = self._grasp_label
        rec["timestamp"] = time.time()
        try:
            with self._state_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        except Exception as e:
            if not hasattr(self, "_state_log_warned"):
                print(f"[WARNING] Failed to write tactile state log: {e}")
                self._state_log_warned = True

    def _append_grasp_info(self, info):
        if info is None:
            info = {}
        grasp_state_code = self._GRASP_STATE_TO_CODE.get(self._grasp_state, -1)
        info.update({
            "grasp_state_code": int(grasp_state_code),
            "contact_abs": float(self._contact_abs),
            "contact_z2": float(self._contact_z2),
            "contact_peak_ema": float(self._contact_peak_ema),
            "hard_protect_latched": bool(self._hard_latched),
            "contact_consec_count": int(self._contact_consec),
        })
        return info


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


class Pi0ArmTactileGripperController(_GraspStateMachineMixin):
    """
    PI0 policy controls arm motion with manual override; tactile model closes the gripper.

    - Arm: PI0 by default, manual while override key is held, soft blend.
    - Gripper: PI0 decides open/close; tactile model runs closed-loop when closing.
    """

    def __init__(self, hitl_policy, tactile_adapter, tactile_reader, keyboard, policy_client):
        self.hitl_policy = hitl_policy
        self.tactile_adapter = tactile_adapter
        self.tactile_reader = tactile_reader
        self.keyboard = keyboard
        self.policy_client = policy_client

        self._instruction = os.getenv("PI0_INSTRUCTION", "No instruction provided.")
        self._pred_action_chunk = None
        self._actions_from_chunk_completed = 0
        self._open_loop_horizon = max(1, int(PI0_OPEN_LOOP_HORIZON))

        self._last_step_time = None
        self._blend = 0.0
        self._last_override_pressed = False

        self._warned = set()
        self._missing_image_warned = False
        self._timing_enabled = os.getenv("TACTILE_PROFILE", "1") != "0"
        self._timing_every = max(1, int(os.getenv("TACTILE_PROFILE_EVERY", "1")))
        self._timing_counter = 0
        self._pi0_debug_counter = 0
        self._pi0_arm_map_counter = 0
        self._print_counter = 0
        self._pi0_arm_ema = np.zeros(7, dtype=np.float32)
        self._pi0_arm_prev = np.zeros(7, dtype=np.float32)

        # Tactile fallback/safety state
        self.tactile_enabled = False
        self._target_ema = None
        self._target_in_deadzone = False
        self._force_cooldown_counter = 0
        self._fallback_duty_counter = 0
        self._grasp_stable_counter = 0
        self._ema_frozen = False
        self._ema_freeze_held = 0

        # Same delta threshold used in the manual controller.
        self.DELTA_THRESHOLD = 0.000356
        self._init_grasp_state_machine("pi0_tactile")
        self._configure_force_baseline_warmup()

        print(f"\n{'='*70}")
        print("PI0 + TACTILE CONTROLLER")
        print(f"{'='*70}")
        print(f"Arm: PI0 by default, hold '{PI0_OVERRIDE_KEY}' for manual override.")
        if self.tactile_adapter is None:
            print("Gripper: PI0 only (tactile model unavailable).")
        else:
            print("Gripper: PI0 gating + tactile closed-loop when closing.")
        if PI0_DEBUG_ACTION_PRINT:
            print(f"PI0 action logging: enabled (every {PI0_DEBUG_ACTION_EVERY} step)")
        if PI0_DEBUG_ARM_MAP_PRINT:
            print(f"PI0 arm mapping logging: enabled (every {PI0_DEBUG_ARM_MAP_EVERY} step)")
        print(f"{'='*70}\n")

    def _warn_once(self, key, msg):
        if key in self._warned:
            return
        print(msg)
        self._warned.add(key)

    def _log_pi0_action(self, action, chunk_index, chunk_len):
        if not PI0_DEBUG_ACTION_PRINT:
            return
        self._pi0_debug_counter += 1
        if self._pi0_debug_counter % PI0_DEBUG_ACTION_EVERY != 0:
            return
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        arm = np.array2string(action[:7], precision=4, floatmode="fixed")
        gripper = float(action[7]) if action.size > 7 else float("nan")
        print(
            f"[PI0 ACTION] step={self._pi0_debug_counter} "
            f"chunk={chunk_index + 1}/{chunk_len} "
            f"mode={PI0_ARM_ACTION_MODE} "
            f"arm={arm} gripper={gripper:+.5f}"
        )

    def _reset_pi0_arm_filter(self):
        self._pi0_arm_ema.fill(0.0)
        self._pi0_arm_prev.fill(0.0)

    def _log_pi0_arm_mapping(self, err, a_raw, a_cmd):
        if not PI0_DEBUG_ARM_MAP_PRINT:
            return
        self._pi0_arm_map_counter += 1
        if self._pi0_arm_map_counter % PI0_DEBUG_ARM_MAP_EVERY != 0:
            return
        q_err_max = float(np.max(np.abs(err))) if err.size else 0.0
        sat_ratio = float(np.mean(np.abs(a_raw) >= 0.999)) if a_raw.size else 0.0
        cmd_max = float(np.max(np.abs(a_cmd))) if a_cmd.size else 0.0
        print(
            f"[PI0 ARM MAP] step={self._pi0_arm_map_counter} "
            f"q_err_max={q_err_max:.5f} sat_ratio={sat_ratio:.3f} cmd_max={cmd_max:.3f}"
        )

    def _map_pi0_position_to_velocity_action(self, pi0_pos, current_pos):
        """
        Convert PI0 absolute joint position target to normalized joint_velocity action.

        Action scale matches backend semantics:
        action in [-1, 1] corresponds to per-step joint delta in [-PI0_JOINT_DELTA_MAX, PI0_JOINT_DELTA_MAX].
        """
        err = np.asarray(pi0_pos - current_pos, dtype=np.float32)
        if PI0_ARM_POS_EPS > 0:
            err[np.abs(err) < PI0_ARM_POS_EPS] = 0.0

        delta_scale = max(float(PI0_JOINT_DELTA_MAX), 1e-6)
        a_raw = np.clip(err / delta_scale, -1.0, 1.0)

        alpha = float(np.clip(PI0_ARM_EMA_ALPHA, 0.0, 1.0))
        self._pi0_arm_ema = alpha * a_raw + (1.0 - alpha) * self._pi0_arm_ema

        da_max = max(0.0, float(PI0_ARM_DA_MAX))
        if da_max > 0:
            a_cmd = np.clip(self._pi0_arm_ema, self._pi0_arm_prev - da_max, self._pi0_arm_prev + da_max)
        else:
            a_cmd = self._pi0_arm_ema.copy()

        self._pi0_arm_prev = a_cmd.copy()
        self._log_pi0_arm_mapping(err, a_raw, a_cmd)
        return a_cmd

    def _map_pi0_gripper_position_to_velocity_action(self, pi0_gripper_cmd, current_gripper):
        target = float(np.clip(pi0_gripper_cmd, GRIPPER_POSITION_MIN, GRIPPER_POSITION_MAX))
        delta = target - float(current_gripper)
        if abs(delta) < PI0_GRIPPER_HYSTERESIS_EPS:
            return 0.0
        delta_scale = max(float(PI0_GRIPPER_DELTA_MAX), 1e-6)
        return float(np.clip(delta / delta_scale, -1.0, 1.0))

    def _get_dt(self):
        now = time.perf_counter()
        if self._last_step_time is None:
            dt = PI0_DEFAULT_DT
        else:
            dt = now - self._last_step_time
        self._last_step_time = now
        if not np.isfinite(dt) or dt <= 0:
            dt = PI0_DEFAULT_DT
        if dt > PI0_MAX_DT:
            dt = PI0_MAX_DT
        return dt

    def _update_blend(self, dt):
        override_pressed = False
        if self.keyboard is not None:
            override_pressed = bool(self.keyboard.buttons.get(PI0_OVERRIDE_KEY, False))
        if override_pressed != self._last_override_pressed:
            if override_pressed:
                print("ARM SOURCE: PI0 -> MANUAL")
            else:
                print("ARM SOURCE: MANUAL -> PI0")
            self._last_override_pressed = override_pressed
        target_blend = 1.0 if override_pressed else 0.0
        if PI0_ARM_SWITCH_SEC <= 0:
            self._blend = target_blend
        else:
            step = dt / PI0_ARM_SWITCH_SEC
            if target_blend > self._blend:
                self._blend = min(self._blend + step, target_blend)
            else:
                self._blend = max(self._blend - step, target_blend)
        self._blend = float(np.clip(self._blend, 0.0, 1.0))
        return self._blend

    def _get_joint_positions(self, obs_dict):
        robot_state = obs_dict.get("robot_state", {}) if isinstance(obs_dict, dict) else {}
        joint_pos = robot_state.get("joint_positions", obs_dict.get("joint_position") if isinstance(obs_dict, dict) else None)
        if joint_pos is None:
            self._warn_once("pi0_missing_joint", "[WARNING] Missing joint_positions; using zeros.")
            return np.zeros(7, dtype=np.float32)
        arr = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
        if arr.size < 7:
            arr = np.pad(arr, (0, 7 - arr.size))
        elif arr.size > 7:
            arr = arr[:7]
        return arr

    def _get_gripper_position(self, obs_dict):
        robot_state = obs_dict.get("robot_state", {}) if isinstance(obs_dict, dict) else {}
        gripper_pos = robot_state.get("gripper_position", obs_dict.get("gripper_position") if isinstance(obs_dict, dict) else None)
        if gripper_pos is None:
            self._warn_once("pi0_missing_gripper", "[WARNING] Missing gripper_position; using 0.")
            return 0.0
        if isinstance(gripper_pos, (list, np.ndarray)):
            return float(gripper_pos[0] if len(gripper_pos) > 0 else 0.0)
        return float(gripper_pos)

    def _preprocess_pi0_image(self, img):
        if img is None:
            return None
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(f"Expected HWC image with 3 or 4 channels; got {arr.shape}")
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        if hasattr(image_tools, "convert_to_uint8"):
            arr = image_tools.convert_to_uint8(arr)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr

    def _resize_for_pi0(self, img):
        if img is None:
            return None
        try:
            return image_tools.resize_with_pad(img, PI0_IMAGE_SIZE, PI0_IMAGE_SIZE)
        except Exception:
            return cv2.resize(img, (PI0_IMAGE_SIZE, PI0_IMAGE_SIZE))

    def build_pi0_obs(self, obs_dict):
        image_dict = obs_dict.get("image", {}) if isinstance(obs_dict, dict) else {}
        ext_left = image_dict.get(f"{varied_camera_1_id}_left")
        ext_right = image_dict.get(f"{varied_camera_2_id}_left")
        wrist_left = image_dict.get(f"{hand_camera_id}_left")

        if ext_left is None:
            self._warn_once("pi0_missing_ext_left", "[WARNING] Missing exterior left image; skipping PI0 inference.")
            return None
        if wrist_left is None:
            self._warn_once("pi0_missing_wrist_left", "[WARNING] Missing wrist left image; skipping PI0 inference.")
            return None
        if ext_right is None:
            self._warn_once("pi0_missing_ext_right", "[WARNING] Missing exterior right image (continuing).")

        try:
            ext_left = self._preprocess_pi0_image(ext_left)
            wrist_left = self._preprocess_pi0_image(wrist_left)
        except Exception as e:
            self._warn_once("pi0_preprocess_fail", f"[WARNING] PI0 image preprocess failed: {e}")
            return None

        ext_left = self._resize_for_pi0(ext_left)
        wrist_left = self._resize_for_pi0(wrist_left)

        joint_position = self._get_joint_positions(obs_dict)
        gripper_position = np.array([self._get_gripper_position(obs_dict)], dtype=np.float32)

        return {
            "observation/exterior_image_1_left": ext_left,
            "observation/wrist_image_left": wrist_left,
            "observation/joint_position": joint_position,
            "observation/gripper_position": gripper_position,
            "prompt": self.instruction,
        }

    def _get_pi0_action(self, obs_dict):
        if self.policy_client is None:
            return None

        chunk = self._pred_action_chunk
        chunk_len = len(chunk) if chunk is not None else 0
        horizon = min(self._open_loop_horizon, chunk_len) if chunk_len > 0 else 0

        need_query = (
            chunk is None
            or self._actions_from_chunk_completed >= horizon
            or self._actions_from_chunk_completed >= chunk_len
        )

        if need_query:
            request_data = self.build_pi0_obs(obs_dict)
            if request_data is None:
                # If we still have a previous chunk, keep using it.
                if chunk is None or self._actions_from_chunk_completed >= chunk_len:
                    return None
            else:
                try:
                    response = self.policy_client.infer(request_data)
                except Exception as e:
                    self._warn_once("pi0_infer_fail", f"[WARNING] PI0 infer failed: {e}")
                    return None
                actions = response.get("actions") if isinstance(response, dict) else None
                if actions is None:
                    self._warn_once("pi0_no_actions", "[WARNING] PI0 response missing 'actions'.")
                    return None
                self._pred_action_chunk = np.asarray(actions, dtype=np.float32)
                self._actions_from_chunk_completed = 0
                chunk = self._pred_action_chunk
                chunk_len = len(chunk)
                if chunk_len == 0:
                    self._warn_once("pi0_empty_chunk", "[WARNING] PI0 returned empty action chunk.")
                    return None

        if chunk is None or self._actions_from_chunk_completed >= len(chunk):
            return None
        chunk_index = self._actions_from_chunk_completed
        action = np.asarray(chunk[chunk_index], dtype=np.float32).reshape(-1)
        self._actions_from_chunk_completed += 1
        if action.size < 8:
            self._warn_once("pi0_action_shape", f"[WARNING] PI0 action has unexpected shape {action.shape}.")
            return None
        self._log_pi0_action(action, chunk_index, len(chunk))
        return action

    def _detect_opening_closing(self, pi_gripper_cmd, current_gripper):
        if pi_gripper_cmd < 0 or pi_gripper_cmd > 1:
            mode = "velocity"
            if self.tactile_enabled:
                is_opening = pi_gripper_cmd < -PI0_GRIPPER_HYSTERESIS_EPS
                is_closing = pi_gripper_cmd > PI0_GRIPPER_HYSTERESIS_EPS
            else:
                is_opening = pi_gripper_cmd < -PI0_GRIPPER_HYSTERESIS_EPS
                is_closing = pi_gripper_cmd > PI0_GRIPPER_HYSTERESIS_EPS
        else:
            mode = "position"
            if self.tactile_enabled:
                is_opening = pi_gripper_cmd < (current_gripper - PI0_GRIPPER_HYSTERESIS_EPS)
                is_closing = pi_gripper_cmd > (current_gripper + PI0_GRIPPER_HYSTERESIS_EPS)
            else:
                is_opening = pi_gripper_cmd < (current_gripper - PI0_GRIPPER_HYSTERESIS_EPS)
                is_closing = pi_gripper_cmd > (current_gripper + PI0_GRIPPER_HYSTERESIS_EPS)
        return is_opening, is_closing, mode

    def _reset_tactile_state(self):
        self._target_ema = None
        self._target_in_deadzone = False
        self._force_cooldown_counter = 0
        self._fallback_duty_counter = 0
        self._grasp_stable_counter = 0
        self._ema_frozen = False
        self._ema_freeze_held = 0
        self._reset_grasp_state_machine()

    def _set_tactile_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled == self.tactile_enabled:
            return
        self.tactile_enabled = enabled
        if enabled:
            self._reset_grasp_state_machine()
            if self.tactile_adapter is not None:
                self.tactile_adapter.reset_force_safety_state()
        else:
            self._reset_tactile_state()

    def _clamp_gripper_delta(self, delta):
        return np.clip(delta, GRIPPER_DELTA_MIN, GRIPPER_DELTA_MAX)

    def _apply_tactile_override(self, obs_dict, action, teleop_t0, teleop_ms):
        if not self.tactile_enabled or self.tactile_adapter is None:
            return action

        try:
            robot_state = obs_dict.get("robot_state", {})
            image_dict = obs_dict.get("image", {})

            wrist_left_key = f"{hand_camera_id}_left"
            wrist_right_key = f"{hand_camera_id}_right"

            wrist_left = image_dict.get(wrist_left_key)
            wrist_right = image_dict.get(wrist_right_key)

            if wrist_left is None:
                wrist_left = (obs_dict.get("wrist_image_left") or
                              obs_dict.get("wrist_image") or
                              robot_state.get("wrist_image_left"))
            if wrist_right is None:
                wrist_right = (obs_dict.get("wrist_image_right") or
                               robot_state.get("wrist_image_right"))
            if wrist_right is None:
                wrist_right = wrist_left

            def preprocess_image(img):
                if img is None:
                    return None
                if len(img.shape) == 3:
                    if img.shape[0] == 4:
                        img = np.transpose(img, (1, 2, 0))
                    if img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                if len(img.shape) != 3 or img.shape[2] != 3:
                    raise ValueError(f"Expected image with shape (H, W, 3); got {img.shape}")
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                return img

            preprocess_start = time.perf_counter()
            wrist_left = preprocess_image(wrist_left)
            wrist_right = preprocess_image(wrist_right)
            preprocess_done = time.perf_counter()
            image_preprocess_ms = (preprocess_done - preprocess_start) * 1000.0
            t_after_preprocess_ms = (preprocess_done - teleop_t0) * 1000.0

            sensor_start = time.perf_counter()
            tactile_history = self.tactile_reader.read_values()
            sensor_done = time.perf_counter()
            sensor_read_ms = (sensor_done - sensor_start) * 1000.0
            t_after_tactile_ms = (sensor_done - teleop_t0) * 1000.0

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

            self._update_contact_and_force_metrics(tactile_history, force_history)

            gripper_pos = obs_dict.get("gripper_position") or robot_state.get("gripper_position", [0.0])
            if isinstance(gripper_pos, (list, np.ndarray)):
                current_gripper = float(gripper_pos[0])
            else:
                current_gripper = float(gripper_pos)

            if wrist_left is not None and wrist_right is not None:
                self._missing_image_warned = False

                model_start = time.perf_counter()
                model_delta = self.tactile_adapter.predict_delta(
                    image_left=wrist_left,
                    image_right=wrist_right,
                    tactile_history=tactile_history,
                    force_history=force_history,
                    step_index=0,
                )
                model_done = time.perf_counter()
                model_ms = (model_done - model_start) * 1000.0
                t_model_start_ms = (model_start - teleop_t0) * 1000.0
                t_model_end_ms = (model_done - teleop_t0) * 1000.0
                timings = getattr(self.tactile_adapter, "last_timings", {}) or {}

                delta_model_raw = 0.0
                delta_after_small_gate = 0.0
                delta_after_force_safety = 0.0
                delta_final_applied = 0.0

                delta = float(model_delta)
                delta_model_raw = float(delta)

                raw_delta = delta
                if abs(delta) < self.DELTA_THRESHOLD:
                    delta = 0.0
                    if not hasattr(self, "_print_counter"):
                        self._print_counter = 0
                    if self._print_counter % 100 == 0:
                        print(f"[SAFETY] Suppressed small delta: {raw_delta:.6f} (threshold: {self.DELTA_THRESHOLD:.6f})")
                delta_after_small_gate = float(delta)

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
                delta_after_force_safety = float(delta)
                baseline_init_done = bool(force_safety_info.get("baseline_init_done", True))
                baseline_block_active = bool(
                    (not baseline_init_done) and FORCE_ZERO_BLOCK_ACTION_UNTIL_READY
                )
                if not baseline_init_done:
                    if not hasattr(self, "_print_counter"):
                        self._print_counter = 0
                    if self._print_counter % 10 == 0:
                        _raw = force_safety_info.get("raw_force")
                        _bl = force_safety_info.get("baseline")
                        _corr = force_safety_info.get("force_corrected")
                        _filt = force_safety_info.get("force")
                        _bc = force_safety_info.get("baseline_count")
                        _br = force_safety_info.get("baseline_required")
                        print(
                            f"[FORCE BASELINE INIT] "
                            f"raw={_raw:.1f}, baseline={_bl}, "
                            f"corrected={_corr:.1f}, filtered={_filt:.1f}, "
                            f"count={_bc}/{_br}"
                        )
                if baseline_block_active:
                    delta = 0.0
                    delta_after_force_safety = 0.0
                elif force_safety_info.get("triggered") and getattr(self, "_print_counter", 0) % 30 == 0:
                    reason = force_safety_info.get("reason")
                    force_now = force_safety_info.get("force")
                    dforce_now = force_safety_info.get("dforce")
                    print(
                        "[FORCE SAFETY] "
                        f"reason={reason} delta {delta_before_force_safety:+.5f}->{delta:+.5f} "
                        f"force={force_now} dforce={dforce_now}"
                    )

                if force_safety_info.get("triggered"):
                    self._force_cooldown_counter = FORCE_SAFETY_COOLDOWN_FRAMES

                if baseline_block_active:
                    delta = 0.0
                elif self._force_cooldown_counter > 0:
                    self._force_cooldown_counter -= 1
                    delta = 0.0
                else:
                    fallback_active = (
                        abs(delta) < self.DELTA_THRESHOLD
                        and model_final_target is not None
                        and np.isfinite(current_gripper)
                    )

                    if not fallback_active:
                        self._fallback_duty_counter = 0

                    if fallback_active:
                        force_now = force_safety_info.get("force")

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

                        if not self._ema_frozen:
                            if self._target_ema is None:
                                self._target_ema = model_final_target
                            else:
                                self._target_ema += TARGET_FALLBACK_EMA_ALPHA * (
                                    model_final_target - self._target_ema
                                )

                        error = self._target_ema - current_gripper

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

                            self._fallback_duty_counter += 1
                            if self._fallback_duty_counter >= TARGET_FALLBACK_DUTY_PERIOD:
                                self._fallback_duty_counter = 0
                                if not force_safety_info.get("hard_latched"):
                                    delta = fallback_delta
                                    if not hasattr(self, "_print_counter"):
                                        self._print_counter = 0
                                    if self._print_counter % 30 == 0:
                                        print(
                                            f"[TARGET FALLBACK] ema={self._target_ema:.4f}, "
                                            f"err={error:+.4f}, step={fallback_delta:+.5f}, "
                                            f"frozen={self._ema_frozen}"
                                        )

                delta = self._apply_grasp_state_delta_postprocess(delta)
                delta = self._clamp_gripper_delta(delta)
                delta_final_applied = float(delta)

                action_space = self.hitl_policy.robot_env.action_space if hasattr(self.hitl_policy, "robot_env") else "joint_velocity"

                action = np.array(action, dtype=np.float32).copy()
                if "velocity" in action_space:
                    max_gripper_delta = 0.25
                    gripper_velocity = 20 * delta / max_gripper_delta
                    if abs(gripper_velocity) < 0.056206943867595054:
                        gripper_velocity = 0.0
                    gripper_velocity = np.clip(gripper_velocity, -1.0, 1.0)
                    action[-1] = gripper_velocity
                    t_after_override_ms = (time.perf_counter() - teleop_t0) * 1000.0
                    target_gripper = current_gripper + delta * 0.5
                    target_gripper = np.clip(target_gripper, GRIPPER_POSITION_MIN, GRIPPER_POSITION_MAX)
                    if not hasattr(self, "_print_counter"):
                        self._print_counter = 0
                    self._print_counter += 1
                    if self._print_counter % 30 == 0:
                        ft_str = f", ft={model_final_target:.3f}" if model_final_target is not None else ""
                        ema_str = f", ema={self._target_ema:.3f}" if self._target_ema is not None else ""
                        print(f"[TACTILE MODEL ACTIVE] Gripper Position: {current_gripper:.3f} -> {target_gripper:.3f} "
                              f"(delta: {delta:+.4f}, model_delta: {model_delta:+.4f}, vel: {gripper_velocity:+.3f}"
                              f"{ft_str}{ema_str})")
                else:
                    target_gripper = current_gripper + delta
                    target_gripper = np.clip(target_gripper, GRIPPER_POSITION_MIN, GRIPPER_POSITION_MAX)
                    action[-1] = target_gripper
                    t_after_override_ms = (time.perf_counter() - teleop_t0) * 1000.0
                    if not hasattr(self, "_print_counter"):
                        self._print_counter = 0
                    self._print_counter += 1
                    if self._print_counter % 30 == 0:
                        ft_str = f", ft={model_final_target:.3f}" if model_final_target is not None else ""
                        ema_str = f", ema={self._target_ema:.3f}" if self._target_ema is not None else ""
                        print(f"[TACTILE MODEL ACTIVE] Gripper: {current_gripper:.3f} -> {target_gripper:.3f} "
                              f"(delta: {delta:+.4f}, model_delta: {model_delta:+.4f}"
                              f"{ft_str}{ema_str})")

                self._record_grasp_step({
                    "delta_model_raw": delta_model_raw,
                    "delta_after_small_delta_gate": delta_after_small_gate,
                    "delta_after_force_safety": delta_after_force_safety,
                    "delta_final_applied": delta_final_applied,
                })

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
                if not self._missing_image_warned:
                    print("[WARNING] Missing wrist images, using PI0 gripper control")
                    self._missing_image_warned = True
        except KeyboardInterrupt:
            stop_action = np.zeros(8, dtype=np.float32)
            return stop_action
        except Exception as e:
            print(f"[ERROR] Tactile model inference failed: {e}")
            print("         Falling back to PI0 gripper control")
            import traceback
            traceback.print_exc()
        return action

    def forward(self, obs_dict, include_info=False):
        teleop_start = time.perf_counter()

        manual_action = self.hitl_policy.forward(obs_dict, include_info=include_info)
        teleop_done = time.perf_counter()
        teleop_t0 = teleop_done
        teleop_ms = (teleop_done - teleop_start) * 1000.0

        if include_info:
            manual_action, info = manual_action

        manual_action = np.asarray(manual_action, dtype=np.float32).reshape(-1)
        if manual_action.size < 8:
            manual_action = np.pad(manual_action, (0, 8 - manual_action.size))

        dt = self._get_dt()
        blend = self._update_blend(dt)

        pi0_action = self._get_pi0_action(obs_dict)
        pi0_available = pi0_action is not None

        manual_arm = manual_action[:7]
        pi0_arm = manual_arm
        pi0_gripper_cmd = manual_action[7]
        action_space = self.hitl_policy.robot_env.action_space if hasattr(self.hitl_policy, "robot_env") else "joint_velocity"

        if pi0_available:
            pi0_gripper_cmd = float(pi0_action[7])
            if PI0_ARM_ACTION_MODE == "joint_position":
                current_pos = self._get_joint_positions(obs_dict)
                pi0_pos = np.asarray(pi0_action[:7], dtype=np.float32)
                if "velocity" in action_space:
                    pi0_arm = self._map_pi0_position_to_velocity_action(pi0_pos, current_pos)
                else:
                    self._reset_pi0_arm_filter()
                    pi0_arm = pi0_pos
            elif PI0_ARM_ACTION_MODE == "joint_velocity":
                self._reset_pi0_arm_filter()
                pi0_arm = np.asarray(pi0_action[:7], dtype=np.float32)
            else:
                self._warn_once("pi0_action_mode", f"[WARNING] Unknown PI0_ARM_ACTION_MODE={PI0_ARM_ACTION_MODE}; using joint_velocity.")
                self._reset_pi0_arm_filter()
                pi0_arm = np.asarray(pi0_action[:7], dtype=np.float32)
        else:
            self._reset_pi0_arm_filter()

        blended_arm = blend * manual_arm + (1.0 - blend) * pi0_arm
        blended_arm = np.clip(blended_arm, -1.0, 1.0)

        current_gripper = self._get_gripper_position(obs_dict)
        if pi0_available:
            is_opening, is_closing, _mode = self._detect_opening_closing(pi0_gripper_cmd, current_gripper)
            if is_closing and self.tactile_adapter is not None:
                self._set_tactile_enabled(True)
            elif is_opening:
                self._set_tactile_enabled(False)
        else:
            self._set_tactile_enabled(False)

        gripper_action = pi0_gripper_cmd
        if pi0_available and "velocity" in action_space:
            if 0.0 <= pi0_gripper_cmd <= 1.0:
                gripper_action = self._map_pi0_gripper_position_to_velocity_action(
                    pi0_gripper_cmd, current_gripper
                )

        action = np.zeros(8, dtype=np.float32)
        action[:7] = blended_arm
        action[7] = gripper_action

        action = self._apply_tactile_override(obs_dict, action, teleop_t0, teleop_ms)

        if include_info:
            return action, self._append_grasp_info(info)
        return action

    def reset_state(self):
        self.hitl_policy.reset_state()
        self._pred_action_chunk = None
        self._actions_from_chunk_completed = 0
        self._last_step_time = None
        self._blend = 0.0
        self._last_override_pressed = False
        self._reset_pi0_arm_filter()
        self._reset_tactile_state()
        self.tactile_enabled = False
        self._reset_grasp_state_machine()

    def set_instruction(self, instruction):
        self._instruction = instruction

    @property
    def instruction(self):
        return self._instruction

    def get_info(self):
        return self.hitl_policy.get_info()


class TactileGripperController(_GraspStateMachineMixin):
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
        self._init_grasp_state_machine("manual_tactile")
        self._configure_force_baseline_warmup()
        
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
                self._reset_grasp_state_machine()
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
                self._reset_grasp_state_machine()
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
            if include_info:
                return action, self._append_grasp_info(info)
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

            self._update_contact_and_force_metrics(tactile_history, force_history)
            
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

                delta_model_raw = 0.0
                delta_after_small_gate = 0.0
                delta_after_force_safety = 0.0
                delta_final_applied = 0.0

                
                delta = float(model_delta)
                delta_model_raw = float(delta)
                
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
                delta_after_small_gate = float(delta)

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
                delta_after_force_safety = float(delta)
                baseline_init_done = bool(force_safety_info.get("baseline_init_done", True))
                baseline_block_active = bool(
                    (not baseline_init_done) and FORCE_ZERO_BLOCK_ACTION_UNTIL_READY
                )
                if not baseline_init_done:
                    if self._print_counter % 10 == 0:
                        _raw = force_safety_info.get("raw_force")
                        _bl = force_safety_info.get("baseline")
                        _corr = force_safety_info.get("force_corrected")
                        _filt = force_safety_info.get("force")
                        _bc = force_safety_info.get("baseline_count")
                        _br = force_safety_info.get("baseline_required")
                        print(
                            f"[FORCE BASELINE INIT] "
                            f"raw={_raw:.1f}, baseline={_bl}, "
                            f"corrected={_corr:.1f}, filtered={_filt:.1f}, "
                            f"count={_bc}/{_br}"
                        )
                if baseline_block_active:
                    delta = 0.0
                    delta_after_force_safety = 0.0
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

                if baseline_block_active:
                    delta = 0.0
                elif self._force_cooldown_counter > 0:
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
                delta = self._apply_grasp_state_delta_postprocess(delta)
                delta = self._clamp_gripper_delta(delta)
                delta_final_applied = float(delta)

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

                self._record_grasp_step({
                    "delta_model_raw": delta_model_raw,
                    "delta_after_small_delta_gate": delta_after_small_gate,
                    "delta_after_force_safety": delta_after_force_safety,
                    "delta_final_applied": delta_final_applied,
                })

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
            return action, self._append_grasp_info(info)
        else:
            return action

    def reset_state(self):
        """Reset controller state."""
        self.hitl_policy.reset_state()
        self.model_enabled = False
        self._last_toggle_state = False
        self._reset_grasp_state_machine()

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

        # Initialize PI0 policy client
        policy_client = None
        try:
            policy_client = websocket_client_policy.WebsocketClientPolicy(
                PI0_REMOTE_HOST, PI0_REMOTE_PORT
            )
            print(f"[INFO] PI0 policy client: {PI0_REMOTE_HOST}:{PI0_REMOTE_PORT}")
        except Exception as e:
            print(f"[WARNING] Failed to initialize PI0 policy client: {e}")
            policy_client = None

        # Controller selection
        if policy_client is not None:
            controller = Pi0ArmTactileGripperController(
                hitl_policy=hitl_policy,
                tactile_adapter=tactile_adapter,
                tactile_reader=tactile_reader,
                keyboard=devices["keyboard"],
                policy_client=policy_client,
            )
        elif tactile_adapter is not None:
            controller = TactileGripperController(
                hitl_policy=hitl_policy,
                tactile_adapter=tactile_adapter,
                tactile_reader=tactile_reader,
                keyboard=devices["keyboard"]
            )
        else:
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
