#!/usr/bin/env python3
"""
Hybrid control: PI0 arm + tactile gripper with staged PI0 provider migration.

Phase A (default):
  - Keep using interfaces.OpenPIWrapper as primary PI0 provider.
  - Add observability, safety mapping, and degradation controls.

Phase B (optional):
  - Enable Pi0ActionBridge provider and shadow A/B comparison via env vars.
"""

import contextlib
import atexit
import json
import logging
import multiprocessing as mp
import os
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import yaml
from omegaconf import DictConfig

from FORTE.sensing.sensor import sensor_data_updater
from FORTE.scripts.force_est_nn import MLPInference, force_estimator_update_loop_restartsafe
from FORTE.scripts.sys_utils import ForceRingBuffer, SharedRingBuffer, opencv_visualizer
from r2d2.robot_env import RobotEnv
from util.openpi import OpenPIConfigs, extract_observation

# Prefer fork to avoid semaphore permissions issues on some systems; ignore if already set.
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

repo_root = Path(__file__).resolve().parents[2]  # /home/pi0/multi-modal
tactile_module_dir = repo_root / "tactile_module"
scripts_dir = Path(__file__).resolve().parent

if not tactile_module_dir.exists():
    raise ImportError(f"tactile_module not found at expected path: {tactile_module_dir}")

for extra_path in (repo_root, tactile_module_dir, scripts_dir):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

# Ensure openpi_client is importable.
try:
    from openpi_client import image_tools, websocket_client_policy
except Exception:
    openpi_client_paths = [
        repo_root / "openpi" / "packages" / "openpi-client" / "src",
        repo_root / "openpi-multi-modal" / "packages" / "openpi-client" / "src",
        scripts_dir / "src" / "openpi-client" / "packages" / "openpi-client" / "src",
    ]
    for path in openpi_client_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from openpi_client import image_tools, websocket_client_policy

# Ensure oculus_reader is in path (for editable installs).
oculus_reader_path = scripts_dir / "src" / "oculus_reader"
if oculus_reader_path.exists() and str(oculus_reader_path) not in sys.path:
    sys.path.insert(0, str(oculus_reader_path))

from tactile_module.robot_inference_adapter import TactileGripperAdapter
from devices import Keyboard, SpaceMouse
from interfaces import HITLPolicy, OpenPIWrapper


FORCE_MODEL_PATH = "/home/pi0/multi-modal/droid-multi-modal/FORTE/force_est_ckpts/034__sizes-256x256__do-0p3__ido-0p05__wd-0p0001__lr-0p0005__ns-0p01__norm-none__huber-1"
FORCE_DEVICE = os.getenv("FORCE_DEVICE", "cuda:0")
FORCE_ESTIMATION_HZ = int(os.getenv("FORCE_ESTIMATION_HZ", "100"))
FORCE_START_DELAY_SEC = float(os.getenv("FORCE_START_DELAY_SEC", "5.0"))
FORCE_ESTIMATOR_ENABLED = os.getenv("FORCE_ESTIMATOR_ENABLED", "1") != "0"

PI0_PROVIDER = os.getenv("PI0_PROVIDER", "wrapper").strip().lower()
PI0_SHADOW_COMPARE = os.getenv("PI0_SHADOW_COMPARE", "0") != "0"
PI0_SHADOW_LOG_EVERY = max(1, int(os.getenv("PI0_SHADOW_LOG_EVERY", "50")))
PI0_REMOTE_HOST = os.getenv("PI0_REMOTE_HOST", "127.0.1.1").strip()
PI0_REMOTE_PORT = int(os.getenv("PI0_REMOTE_PORT", "8000"))
PI0_OPEN_LOOP_HORIZON = max(1, int(os.getenv("PI0_OPEN_LOOP_HORIZON", "4")))
PI0_INCLUDE_TACTILE_VALUES = os.getenv("PI0_INCLUDE_TACTILE_VALUES", "1") != "0"
PI0_INCLUDE_FORCE_PREDICTION = os.getenv("PI0_INCLUDE_FORCE_PREDICTION", "0") != "0"
PI0_ARM_ACTION_MODE = os.getenv("PI0_ARM_ACTION_MODE", "joint_velocity").strip().lower()
PI0_JOINT_DELTA_MAX = float(os.getenv("PI0_JOINT_DELTA_MAX", "0.2"))
PI0_ARM_POS_EPS = float(os.getenv("PI0_ARM_POS_EPS", "0.003"))
PI0_ARM_EMA_ALPHA = float(os.getenv("PI0_ARM_EMA_ALPHA", "0.3"))
PI0_ARM_DA_MAX = float(os.getenv("PI0_ARM_DA_MAX", "0.12"))
PI0_DEBUG_ACTION_EVERY = max(1, int(os.getenv("PI0_DEBUG_ACTION_EVERY", "20")))
PI0_DEBUG_ARM_MAP_EVERY = max(1, int(os.getenv("PI0_DEBUG_ARM_MAP_EVERY", "10")))
PI0_DEBUG_GRIPPER_EVERY = max(1, int(os.getenv("PI0_DEBUG_GRIPPER_EVERY", "20")))
PI0_DEBUG_STATE_EVERY = max(1, int(os.getenv("PI0_DEBUG_STATE_EVERY", "20")))
PI0_ARM_SAT_ALERT_RATIO = float(os.getenv("PI0_ARM_SAT_ALERT_RATIO", "0.95"))
PI0_ARM_SAT_ALERT_STEPS = max(1, int(os.getenv("PI0_ARM_SAT_ALERT_STEPS", "20")))
PI0_ARM_SAT_STOP_STEPS = max(0, int(os.getenv("PI0_ARM_SAT_STOP_STEPS", "0")))
PI0_GRASP_THRESHOLD = float(os.getenv("PI0_GRASP_THRESHOLD", "0.20"))
PI0_GRASP_HYSTERESIS = float(os.getenv("PI0_GRASP_HYSTERESIS", "0.03"))
PI0_GRIPPER_BIN_THRESHOLD = float(
    os.getenv("PI0_GRIPPER_BIN_THRESHOLD", str(PI0_GRASP_THRESHOLD))
)
PI0_EE_Z_WARN = float(os.getenv("PI0_EE_Z_WARN", "0.22"))
_PI0_EE_Z_HARD_MIN_RAW = os.getenv("PI0_EE_Z_HARD_MIN", "0.19").strip()
PI0_EE_Z_HARD_MIN = float(_PI0_EE_Z_HARD_MIN_RAW) if _PI0_EE_Z_HARD_MIN_RAW else None
PI0_EE_Z_GUARD_ACTION = os.getenv("PI0_EE_Z_GUARD_ACTION", "freeze").strip().lower()
PI0_EE_Z_GUARD_REQUIRE_OPEN = os.getenv("PI0_EE_Z_GUARD_REQUIRE_OPEN", "1") != "0"
PI0_EE_Z_SOFT_SCALE = os.getenv("PI0_EE_Z_SOFT_SCALE", "1") != "0"
PI0_EE_Z_FREEZE_TIMEOUT_STEPS = max(1, int(os.getenv("PI0_EE_Z_FREEZE_TIMEOUT_STEPS", "30")))
PI0_EE_Z_RETREAT_SCALE = float(os.getenv("PI0_EE_Z_RETREAT_SCALE", "0.5"))
PI0_EE_Z_RETREAT_MIN_NORM = float(os.getenv("PI0_EE_Z_RETREAT_MIN_NORM", "0.08"))
PI0_LOW_Z_HARD_GUARD_ALERT_STEPS = max(
    1, int(os.getenv("PI0_LOW_Z_HARD_GUARD_ALERT_STEPS", "40"))
)

FAILURE_MODE = os.getenv("FAILURE_MODE", "degrade_then_stop").strip().lower()
PI0_MAX_CONSEC_FAIL = max(1, int(os.getenv("PI0_MAX_CONSEC_FAIL", "8")))
TACTILE_MAX_CONSEC_FAIL = max(1, int(os.getenv("TACTILE_MAX_CONSEC_FAIL", "8")))

TACTILE_MODEL_CHECKPOINT = os.getenv(
    "TACTILE_MODEL_CHECKPOINT",
    "/home/pi0/multi-modal/checkpoints/box_and_cup/checkpoints/ab_full_C4_seed42_drop_demo_20_20260220_203907_best.pt",
).strip()
TACTILE_MODEL_CONFIG = os.getenv("TACTILE_MODEL_CONFIG", "auto").strip()
TACTILE_MODEL_CONFIG_FALLBACK = os.getenv(
    "TACTILE_MODEL_CONFIG_FALLBACK",
    "/home/pi0/multi-modal/tactile_module/configs/default.yaml",
).strip()
TACTILE_MODEL_CONFIG_SEARCH_ROOTS = [
    p.strip()
    for p in os.getenv(
        "TACTILE_MODEL_CONFIG_SEARCH_ROOTS",
        "/home/pi0/multi-modal/tactile_module/experiments:/home/pi0/multi-modal/tactile_module/configs",
    ).split(":")
    if p.strip()
]
TACTILE_GRIPPER_STATS_SOURCE_H5 = os.getenv(
    "TACTILE_GRIPPER_STATS_SOURCE_H5",
    "/home/pi0/multi-modal/robomimic_output/action_target_gripper_position_50target_horizon_none_papercup_and_box_haiyi_100_delta.hdf5",
).strip() or None
TACTILE_GRIPPER_STATS_OUTPUT = os.getenv("TACTILE_GRIPPER_STATS_OUTPUT", "auto").strip()
TACTILE_GRIPPER_STATS_SPLIT = os.getenv("TACTILE_GRIPPER_STATS_SPLIT", "train").strip().lower()
if TACTILE_GRIPPER_STATS_SPLIT in {"", "none", "all"}:
    TACTILE_GRIPPER_STATS_SPLIT = None
TACTILE_GRIPPER_STATS_REFRESH = os.getenv("TACTILE_GRIPPER_STATS_REFRESH", "0") == "1"

PI0_BRIDGE_DEBUG = os.getenv("PI0_BRIDGE_DEBUG", "1") != "0"
TACTILE_SAVE_LOG = os.getenv("TACTILE_SAVE_LOG", "1") != "0"
TACTILE_RUN_LOG_DIR = os.getenv(
    "TACTILE_RUN_LOG_DIR",
    "/home/pi0/multi-modal/droid-multi-modal/scripts/logs/tactile_module_inference",
).strip()


def _sanitize_filename(name):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))
    return safe.strip("._-") or "run"


class _TeeStream:
    """Mirror writes to multiple file-like streams."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

    def fileno(self):
        for stream in self._streams:
            fn = getattr(stream, "fileno", None)
            if callable(fn):
                return fn()
        raise OSError("No fileno available on tee streams")


_RUN_LOG_FILE_HANDLE = None
_RUN_LOG_PATH = None


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
        log_dir = Path(TACTILE_RUN_LOG_DIR).expanduser()
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

if FAILURE_MODE not in {"degrade_then_stop", "immediate_stop"}:
    print(f"[WARNING] Unknown FAILURE_MODE={FAILURE_MODE}, using degrade_then_stop")
    FAILURE_MODE = "degrade_then_stop"

if PI0_PROVIDER not in {"wrapper", "bridge"}:
    print(f"[WARNING] Unknown PI0_PROVIDER={PI0_PROVIDER}, using wrapper")
    PI0_PROVIDER = "wrapper"

if PI0_GRASP_HYSTERESIS < 0.0:
    print(f"[WARNING] PI0_GRASP_HYSTERESIS={PI0_GRASP_HYSTERESIS} < 0, clamping to 0.")
    PI0_GRASP_HYSTERESIS = 0.0

if not 0.0 <= PI0_GRASP_THRESHOLD <= 1.0:
    print(f"[WARNING] PI0_GRASP_THRESHOLD={PI0_GRASP_THRESHOLD} outside [0,1], clamping.")
    PI0_GRASP_THRESHOLD = float(np.clip(PI0_GRASP_THRESHOLD, 0.0, 1.0))

if not 0.0 <= PI0_GRIPPER_BIN_THRESHOLD <= 1.0:
    print(f"[WARNING] PI0_GRIPPER_BIN_THRESHOLD={PI0_GRIPPER_BIN_THRESHOLD} outside [0,1], clamping.")
    PI0_GRIPPER_BIN_THRESHOLD = float(np.clip(PI0_GRIPPER_BIN_THRESHOLD, 0.0, 1.0))

if PI0_EE_Z_WARN < 0.0:
    print(f"[WARNING] PI0_EE_Z_WARN={PI0_EE_Z_WARN} < 0, clamping to 0.")
    PI0_EE_Z_WARN = 0.0

if PI0_EE_Z_HARD_MIN is not None and PI0_EE_Z_HARD_MIN < 0.0:
    print(f"[WARNING] PI0_EE_Z_HARD_MIN={PI0_EE_Z_HARD_MIN} < 0, disabling hard min.")
    PI0_EE_Z_HARD_MIN = None

if PI0_EE_Z_HARD_MIN is not None and PI0_EE_Z_HARD_MIN >= PI0_EE_Z_WARN:
    adjusted_warn = PI0_EE_Z_HARD_MIN + 0.01
    print(
        "[WARNING] PI0_EE_Z_HARD_MIN must be below PI0_EE_Z_WARN. "
        f"Adjusting warn from {PI0_EE_Z_WARN:.4f} to {adjusted_warn:.4f}."
    )
    PI0_EE_Z_WARN = adjusted_warn

if PI0_EE_Z_GUARD_ACTION not in {"warn", "freeze", "freeze_then_retreat", "retreat", "stop"}:
    print(f"[WARNING] Unknown PI0_EE_Z_GUARD_ACTION={PI0_EE_Z_GUARD_ACTION}, using freeze.")
    PI0_EE_Z_GUARD_ACTION = "freeze"

if PI0_EE_Z_FREEZE_TIMEOUT_STEPS < 1:
    print(
        f"[WARNING] PI0_EE_Z_FREEZE_TIMEOUT_STEPS={PI0_EE_Z_FREEZE_TIMEOUT_STEPS} < 1, clamping to 1."
    )
    PI0_EE_Z_FREEZE_TIMEOUT_STEPS = 1

if PI0_EE_Z_RETREAT_SCALE <= 0.0:
    print(
        f"[WARNING] PI0_EE_Z_RETREAT_SCALE={PI0_EE_Z_RETREAT_SCALE} <= 0, using 0.5."
    )
    PI0_EE_Z_RETREAT_SCALE = 0.5

if PI0_EE_Z_RETREAT_MIN_NORM < 0.0:
    print(
        f"[WARNING] PI0_EE_Z_RETREAT_MIN_NORM={PI0_EE_Z_RETREAT_MIN_NORM} < 0, clamping to 0."
    )
    PI0_EE_Z_RETREAT_MIN_NORM = 0.0


def _parse_checkpoint_variant(checkpoint_path):
    """
    Parse variant tag from checkpoint filename.
    Example suffix:
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
    env_override = os.getenv("TACTILE_MODEL_CONFIG_OVERRIDE", "").strip()
    if env_override:
        p = Path(env_override).expanduser()
        if p.exists():
            return str(p.resolve())
        print(f"[WARNING] TACTILE_MODEL_CONFIG_OVERRIDE does not exist: {p}")

    cfg_value = str(config_value).strip() if config_value is not None else ""
    if cfg_value and cfg_value.lower() not in {"auto", "none", ""}:
        p = Path(cfg_value).expanduser()
        if p.exists():
            return str(p.resolve())
        print(f"[WARNING] Explicit TACTILE_MODEL_CONFIG not found: {p}. Falling back to auto.")

    variant, meta = _parse_checkpoint_variant(checkpoint_path)
    if variant:
        candidates = []
        for root in TACTILE_MODEL_CONFIG_SEARCH_ROOTS:
            rp = Path(root).expanduser()
            if not rp.exists():
                continue
            for p in rp.glob(f"**/{variant}.yaml"):
                if _yaml_matches_variant(p, meta):
                    candidates.append(p)
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            chosen = candidates[0].resolve()
            print(f"[INFO] Auto-resolved config from checkpoint variant '{variant}': {chosen}")
            return str(chosen)
        print(f"[WARNING] Could not auto-match config for checkpoint variant: {variant}")

    fallback = Path(TACTILE_MODEL_CONFIG_FALLBACK).expanduser()
    if fallback.exists():
        print(f"[INFO] Using fallback config: {fallback}")
        return str(fallback.resolve())

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
    except Exception as exc:
        raise ImportError(
            f"Failed to import compute_normalization_stats (requires h5py). Error: {exc}"
        ) from exc

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
        return config_path

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


def validate_gripper_stats_file(config_path):
    config_file = Path(config_path).expanduser().resolve()
    with config_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config format: {config_file}")

    rob = cfg.get("robomimic", {})
    if not isinstance(rob, dict):
        return

    if not bool(rob.get("normalize_gripper", False)):
        return

    gripper_norm_cfg = rob.get("gripper_normalization", {}) or {}
    stats_file = gripper_norm_cfg.get("stats_file")
    if not stats_file:
        # Direct inline mean/std or min/max is allowed.
        return

    stats_path = Path(stats_file).expanduser()
    if not stats_path.is_absolute():
        stats_path = (config_file.parent / stats_path).resolve()

    if not stats_path.exists():
        raise FileNotFoundError(f"Gripper normalization stats file not found: {stats_path}")

    with stats_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Gripper stats JSON must be an object: {stats_path}")


class babyFORTEReader:
    """Spin up tactile (and optional force-estimator) processes."""

    def __init__(self, cfg, shutdown_event):
        self.shutdown_event = shutdown_event
        self.shared_sensor_buffer = SharedRingBuffer(cfg.buffer.size, cfg.buffer.num_channels, "d")
        self.shared_force_buffer = ForceRingBuffer()
        self._force_estimator_process = None

        self.sensor_process = mp.Process(
            target=sensor_data_updater,
            args=(cfg.elvrgripper, self.shared_sensor_buffer),
            kwargs={"mode": "process", "shutdown_event": shutdown_event},
        )
        self.sensor_process.start()
        time.sleep(2)

        if FORCE_ESTIMATOR_ENABLED:
            try:
                force_device = FORCE_DEVICE
                start_method = mp.get_start_method(allow_none=True)
                if str(force_device).lower().startswith("cuda") and start_method == "fork":
                    print(
                        "[WARNING] FORCE_DEVICE=cuda with multiprocessing start method 'fork' "
                        "causes CUDA re-init failures. Falling back to FORCE_DEVICE=cpu for force estimator."
                    )
                    force_device = "cpu"

                force_model = MLPInference(FORCE_MODEL_PATH, device=force_device)
                self._force_estimator_process = mp.Process(
                    target=force_estimator_update_loop_restartsafe,
                    args=(
                        force_model,
                        self.shared_sensor_buffer,
                        self.shared_force_buffer,
                        shutdown_event,
                    ),
                    kwargs={"hz": FORCE_ESTIMATION_HZ, "start_delay": FORCE_START_DELAY_SEC},
                )
                self._force_estimator_process.start()
                print(f"[force-estimator] started ({FORCE_MODEL_PATH}, device={force_device})")
            except Exception as e:
                print(f"[WARNING] Force estimator unavailable: {e}")
        else:
            print("[INFO] Force estimator disabled by FORCE_ESTIMATOR_ENABLED=0")

        self.visualizer_process = mp.Process(
            target=opencv_visualizer,
            args=(self.shared_sensor_buffer, cfg.buffer, self.shared_force_buffer, None, shutdown_event),
        )
        self.visualizer_process.start()

    def read_values(self):
        try:
            if self.shared_sensor_buffer.is_empty():
                return np.zeros((self.shared_sensor_buffer.num_channels,), dtype=np.float32)
            return self.shared_sensor_buffer.get_latest_freq_history()
        except Exception as e:
            print(f"[WARNING] Failed to read tactile values: {e}")
            return np.zeros((self.shared_sensor_buffer.num_channels,), dtype=np.float32)

    def read_force_values(self, k=50):
        try:
            k = max(1, int(k))
            force = np.asarray(self.shared_force_buffer.get_latest(k=k), dtype=np.float32).reshape(-1, 1)
            return force
        except Exception as e:
            if not hasattr(self, "_force_warned"):
                print(f"[WARNING] Failed to read force values: {e}. Using zeros.")
                self._force_warned = True
            return np.zeros((max(1, int(k)), 1), dtype=np.float32)

    def close(self):
        if self.shutdown_event is not None:
            self.shutdown_event.set()

        procs = [self.visualizer_process, self.sensor_process, self._force_estimator_process]
        procs = [p for p in procs if p is not None]

        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        time.sleep(1)
        for proc in procs:
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=1)

        self.shared_sensor_buffer.close()
        self.shared_force_buffer.close()


@contextlib.contextmanager
def _delay_keyboard_interrupt():
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


class Pi0ActionBridge:
    """Direct PI0 websocket provider mirroring OpenPIWrapper request semantics."""

    def __init__(self, config: OpenPIConfigs):
        self.config = config
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(config.remote_host, config.remote_port)
        self.reset()

    def reset(self):
        self._actions_from_chunk_completed = 0
        self._pred_action_chunk = None
        self._instruction = "No instruction provided."
        self._start_time = time.time()
        self._last_request_keys = []
        self._last_has_tactile_values = False
        self._last_prompt_len = 0

    def set_instruction(self, instruction):
        self._instruction = instruction

    @property
    def instruction(self):
        return self._instruction

    def _build_request_data(self, obs_dict):
        robot_states = obs_dict["robot_state"]
        curr_obs = extract_observation(self.config, obs_dict)

        request_data = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(
                curr_obs[f"{self.config.external_camera}_image"], 224, 224
            ),
            "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
            "observation/joint_position": curr_obs["joint_position"],
            "observation/gripper_position": curr_obs["gripper_position"],
            "prompt": self.instruction,
        }

        if PI0_INCLUDE_TACTILE_VALUES and "tactile_values" in robot_states:
            request_data["observation/tactile_values"] = np.array(robot_states["tactile_values"]).flatten()

        if PI0_INCLUDE_FORCE_PREDICTION:
            force_prediction = robot_states.get("force_prediction")
            if force_prediction is not None:
                request_data["observation/force_prediction"] = np.array(force_prediction).flatten()

        return request_data

    def forward(self, obs_dict, include_info=False):
        start_time = time.time()
        request_keys = self._last_request_keys
        has_tactile_values = self._last_has_tactile_values
        prompt_len = self._last_prompt_len
        chunk_reused = False
        chunk_index = -1

        chunk = self._pred_action_chunk
        chunk_len = len(chunk) if chunk is not None else 0
        horizon = min(self.config.open_loop_horizon, chunk_len) if chunk_len > 0 else 0
        need_query = (
            chunk is None
            or self._actions_from_chunk_completed == 0
            or self._actions_from_chunk_completed >= horizon
            or self._actions_from_chunk_completed >= chunk_len
        )

        if need_query:
            request_data = self._build_request_data(obs_dict)
            request_keys = sorted(request_data.keys())
            has_tactile_values = "observation/tactile_values" in request_data
            prompt_len = len(str(request_data.get("prompt", "")))

            try:
                response = self.policy_client.infer(request_data)
                actions = response.get("actions") if isinstance(response, dict) else None
                if actions is None:
                    raise KeyError("PI0 bridge response missing 'actions'.")
                self._pred_action_chunk = np.asarray(actions, dtype=np.float32)
                self._actions_from_chunk_completed = 0
                self._last_request_keys = request_keys
                self._last_has_tactile_values = has_tactile_values
                self._last_prompt_len = prompt_len
            except Exception:
                # Reuse remaining chunk when available.
                if chunk is not None and self._actions_from_chunk_completed < chunk_len:
                    chunk_reused = True
                else:
                    raise
        else:
            chunk_reused = True

        chunk = self._pred_action_chunk
        if chunk is None:
            raise RuntimeError("PI0 bridge has no action chunk available.")

        chunk_len = len(chunk)
        if self._actions_from_chunk_completed >= chunk_len:
            raise RuntimeError("PI0 bridge chunk exhausted without refresh.")

        chunk_index = self._actions_from_chunk_completed
        action = np.asarray(chunk[chunk_index], dtype=np.float32).reshape(-1)
        self._actions_from_chunk_completed += 1
        action = np.clip(action, -1, 1)

        if include_info:
            info = {
                "action_chunk": chunk,
                "actions_from_chunk_completed": self._actions_from_chunk_completed,
                "time": time.time() - start_time,
                "request_keys": request_keys,
                "chunk_len": chunk_len,
                "chunk_index": chunk_index,
                "chunk_reused": chunk_reused,
                "open_loop_horizon": int(self.config.open_loop_horizon),
                "prompt_len": int(prompt_len),
                "has_tactile_values": bool(has_tactile_values),
            }
            return action, info
        return action


class HybridPolicy:
    """
    Hybrid policy:
      - Arm: PI0 output (with optional position->velocity mapping)
      - Gripper: PI0 gate + tactile override when grasping
      - Failure policy: degrade_then_stop (default) or immediate_stop
    """

    def __init__(
        self,
        primary_provider,
        tactile_adapter,
        tactile_reader,
        openpi_config,
        grasp_threshold=0.5,
        hysteresis=0.1,
        gripper_bin_threshold=0.5,
        shadow_provider=None,
    ):
        self.primary_provider = primary_provider
        self.shadow_provider = shadow_provider
        self.tactile_adapter = tactile_adapter
        self.tactile_reader = tactile_reader
        self.openpi_config = openpi_config

        self.grasp_threshold = grasp_threshold
        self.hysteresis = hysteresis
        self.gripper_bin_threshold = gripper_bin_threshold

        self.tactile_active = False
        self.last_gripper_action = 0.0
        self.should_stop = False

        self._instruction = "No instruction provided."
        self._safe_action = np.zeros(8, dtype=np.float32)
        self._safe_action_valid = False

        self._pi0_fail_count = 0
        self._tactile_fail_count = 0

        self._pi0_action_counter = 0
        self._pi0_arm_map_counter = 0
        self._pi0_arm_ema = np.zeros(7, dtype=np.float32)
        self._pi0_arm_prev = np.zeros(7, dtype=np.float32)
        self._pi0_arm_high_sat_streak = 0
        self._pi0_gripper_counter = 0
        self._pi0_gripper_high_seen = False
        self._state_counter = 0
        self._prev_ee_z = None
        self._prev_ee_t = None
        self._low_z_warn_counter = 0
        self._low_z_hard_guard_counter = 0
        self._pos_mode_mismatch_streak = 0

        self._shadow_count = 0
        self._shadow_linf_samples = []
        self._shadow_l2_samples = []
        self._shadow_gripper_sign_match = 0
        self._shadow_chunk_len_match = 0
        self._shadow_request_key_match = 0

    def _maybe_stop_on_failure(self, context, fail_count, fail_limit, exc):
        if FAILURE_MODE == "immediate_stop":
            self.should_stop = True
            print(f"[CRITICAL] {context} failed (immediate_stop): {exc}")
            return
        if fail_count >= fail_limit:
            self.should_stop = True
            print(f"[CRITICAL] {context} failed {fail_count}/{fail_limit}: {exc}")

    def _get_joint_positions(self, obs_dict):
        robot_state = obs_dict.get("robot_state", {}) if isinstance(obs_dict, dict) else {}
        joint_pos = robot_state.get("joint_positions")
        if joint_pos is None:
            return np.zeros(7, dtype=np.float32)
        arr = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
        if arr.size < 7:
            arr = np.pad(arr, (0, 7 - arr.size))
        elif arr.size > 7:
            arr = arr[:7]
        return arr

    def _get_ee_z(self, obs_dict):
        robot_state = obs_dict.get("robot_state", {}) if isinstance(obs_dict, dict) else {}
        cart_pos = robot_state.get("cartesian_position")
        if cart_pos is None:
            return None
        arr = np.asarray(cart_pos, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            return None
        return float(arr[2])

    def _log_state(self, ee_z, arm_cmd, pi0_gripper_raw):
        self._state_counter += 1
        now = time.time()
        dz = None
        if ee_z is not None and self._prev_ee_z is not None and self._prev_ee_t is not None:
            dt = max(now - self._prev_ee_t, 1e-6)
            dz = (ee_z - self._prev_ee_z) / dt

        if self._state_counter % PI0_DEBUG_STATE_EVERY != 0:
            self._prev_ee_z = ee_z
            self._prev_ee_t = now
            return

        arm_cmd = np.asarray(arm_cmd, dtype=np.float32).reshape(-1)
        arm_norm = float(np.linalg.norm(arm_cmd))
        arm_max = float(np.max(np.abs(arm_cmd))) if arm_cmd.size else 0.0
        z_str = "NA" if ee_z is None else f"{ee_z:.4f}"
        dz_str = "NA" if dz is None else f"{dz:+.4f}"
        print(
            f"[STATE] step={self._state_counter} ee_z={z_str} dz={dz_str} "
            f"arm_norm={arm_norm:.4f} arm_max={arm_max:.4f} "
            f"gripper_raw={pi0_gripper_raw:+.4f} tactile={int(self.tactile_active)}"
        )

        self._prev_ee_z = ee_z
        self._prev_ee_t = now

    def _build_low_z_retreat_cmd(self, arm_cmd):
        arm = np.asarray(arm_cmd, dtype=np.float32).reshape(-1)
        retreat = -arm * float(PI0_EE_Z_RETREAT_SCALE)
        retreat_norm = float(np.linalg.norm(retreat))
        if retreat_norm <= 1e-8:
            return np.zeros_like(arm, dtype=np.float32), retreat_norm
        if retreat_norm < PI0_EE_Z_RETREAT_MIN_NORM:
            retreat *= float(PI0_EE_Z_RETREAT_MIN_NORM / retreat_norm)
        retreat = np.clip(retreat, -1.0, 1.0)
        retreat_norm = float(np.linalg.norm(retreat))
        return retreat.astype(np.float32), retreat_norm

    def _guard_low_z(self, ee_z, arm_cmd, pi0_gripper_raw):
        if ee_z is None:
            return arm_cmd

        guard_eligible = True
        if PI0_EE_Z_GUARD_REQUIRE_OPEN:
            guard_eligible = (pi0_gripper_raw <= self.gripper_bin_threshold) and (not self.tactile_active)
        if not guard_eligible:
            return arm_cmd

        if ee_z <= PI0_EE_Z_WARN:
            self._low_z_warn_counter += 1
            if self._low_z_warn_counter <= 3 or self._low_z_warn_counter % PI0_DEBUG_STATE_EVERY == 0:
                print(
                    f"[LOW_Z WARN] ee_z={ee_z:.4f} <= warn={PI0_EE_Z_WARN:.4f}, "
                    f"gripper_raw={pi0_gripper_raw:+.4f}, action={PI0_EE_Z_GUARD_ACTION}"
                )

        # In freeze mode, taper arm command as z approaches hard_min to reduce impact risk.
        if (
            PI0_EE_Z_GUARD_ACTION in {"freeze", "freeze_then_retreat"}
            and PI0_EE_Z_SOFT_SCALE
            and PI0_EE_Z_HARD_MIN is not None
            and PI0_EE_Z_HARD_MIN < ee_z <= PI0_EE_Z_WARN
        ):
            denom = max(PI0_EE_Z_WARN - PI0_EE_Z_HARD_MIN, 1e-6)
            scale = float(np.clip((ee_z - PI0_EE_Z_HARD_MIN) / denom, 0.0, 1.0))
            if self._low_z_warn_counter <= 3 or self._low_z_warn_counter % PI0_DEBUG_STATE_EVERY == 0:
                print(
                    f"[LOW_Z SOFT] ee_z={ee_z:.4f}, scale={scale:.3f} "
                    f"(warn={PI0_EE_Z_WARN:.4f}, hard_min={PI0_EE_Z_HARD_MIN:.4f})"
                )
            return np.asarray(arm_cmd, dtype=np.float32) * scale

        if PI0_EE_Z_HARD_MIN is None or ee_z > PI0_EE_Z_HARD_MIN:
            self._low_z_hard_guard_counter = 0
            return arm_cmd

        self._low_z_hard_guard_counter += 1

        if PI0_EE_Z_GUARD_ACTION == "freeze":
            if (
                self._low_z_hard_guard_counter == PI0_LOW_Z_HARD_GUARD_ALERT_STEPS
                or self._low_z_hard_guard_counter % PI0_DEBUG_STATE_EVERY == 0
            ):
                print(
                    "[WARNING] LOW_Z hard guard has been active for "
                    f"{self._low_z_hard_guard_counter} consecutive steps "
                    f"(ee_z={ee_z:.4f}, hard_min={PI0_EE_Z_HARD_MIN:.4f}). "
                    "This can deadlock arm motion near grasp. "
                    "Consider PI0_EE_Z_GUARD_REQUIRE_OPEN=1 and/or reducing PI0_EE_Z_HARD_MIN "
                    "if hardware safety allows."
                )
            print(
                f"[LOW_Z GUARD] ee_z={ee_z:.4f} <= hard_min={PI0_EE_Z_HARD_MIN:.4f}; "
                "freezing arm command for this step."
            )
            return np.zeros_like(arm_cmd, dtype=np.float32)

        if PI0_EE_Z_GUARD_ACTION == "freeze_then_retreat":
            if self._low_z_hard_guard_counter <= PI0_EE_Z_FREEZE_TIMEOUT_STEPS:
                if self._low_z_hard_guard_counter == PI0_EE_Z_FREEZE_TIMEOUT_STEPS:
                    print(
                        "[LOW_Z GUARD] freeze_then_retreat reached timeout "
                        f"({PI0_EE_Z_FREEZE_TIMEOUT_STEPS} steps). Next step will retreat."
                    )
                return np.zeros_like(arm_cmd, dtype=np.float32)
            retreat_cmd, retreat_norm = self._build_low_z_retreat_cmd(arm_cmd)
            print(
                f"[LOW_Z RETREAT] ee_z={ee_z:.4f} <= hard_min={PI0_EE_Z_HARD_MIN:.4f}; "
                f"retreat_norm={retreat_norm:.4f}, scale={PI0_EE_Z_RETREAT_SCALE:.3f}"
            )
            return retreat_cmd

        if PI0_EE_Z_GUARD_ACTION == "retreat":
            retreat_cmd, retreat_norm = self._build_low_z_retreat_cmd(arm_cmd)
            print(
                f"[LOW_Z RETREAT] ee_z={ee_z:.4f} <= hard_min={PI0_EE_Z_HARD_MIN:.4f}; "
                f"retreat_norm={retreat_norm:.4f}, scale={PI0_EE_Z_RETREAT_SCALE:.3f}"
            )
            return retreat_cmd

        if PI0_EE_Z_GUARD_ACTION == "stop":
            self.should_stop = True
            print(
                f"[CRITICAL] ee_z={ee_z:.4f} <= hard_min={PI0_EE_Z_HARD_MIN:.4f}; "
                "setting should_stop=True."
            )
            return np.zeros_like(arm_cmd, dtype=np.float32)

        return arm_cmd

    def _map_pi0_position_to_velocity_action(self, pi0_pos, current_pos):
        err = np.asarray(pi0_pos - current_pos, dtype=np.float32)
        q_err_max = float(np.max(np.abs(err))) if err.size else 0.0
        pi0_abs_max = float(np.max(np.abs(pi0_pos))) if np.size(pi0_pos) else 0.0
        curr_abs_max = float(np.max(np.abs(current_pos))) if np.size(current_pos) else 0.0

        # Heuristic: PI0 position-mode is likely mismatched when predicted "positions"
        # stay tiny but q_err stays huge (common when model actually outputs velocities).
        if q_err_max > 1.0 and pi0_abs_max < 0.8 and curr_abs_max > 1.0:
            self._pos_mode_mismatch_streak += 1
        else:
            self._pos_mode_mismatch_streak = 0
        if self._pos_mode_mismatch_streak == 10:
            print(
                "[WARNING] Possible PI0 arm mode mismatch: configured as joint_position "
                f"but q_err_max={q_err_max:.3f}, pi0_abs_max={pi0_abs_max:.3f}. "
                "If this checkpoint outputs joint_velocity, set PI0_ARM_ACTION_MODE=joint_velocity."
            )

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
        self._pi0_arm_map_counter += 1
        sat_ratio = float(np.mean(np.abs(a_raw) >= 0.999)) if a_raw.size else 0.0
        if sat_ratio >= PI0_ARM_SAT_ALERT_RATIO:
            self._pi0_arm_high_sat_streak += 1
        else:
            self._pi0_arm_high_sat_streak = 0

        if self._pi0_arm_high_sat_streak == PI0_ARM_SAT_ALERT_STEPS:
            print(
                "[WARNING] PI0 arm mapping has high saturation for "
                f"{PI0_ARM_SAT_ALERT_STEPS} consecutive steps "
                f"(sat_ratio={sat_ratio:.3f}). "
                "This usually means PI0 action mode/scale mismatch."
            )
        if PI0_ARM_SAT_STOP_STEPS > 0 and self._pi0_arm_high_sat_streak >= PI0_ARM_SAT_STOP_STEPS:
            self.should_stop = True
            print(
                "[CRITICAL] PI0 arm mapping saturation exceeded stop threshold "
                f"({self._pi0_arm_high_sat_streak}/{PI0_ARM_SAT_STOP_STEPS}). Stopping robot."
            )

        if self._pi0_arm_map_counter % PI0_DEBUG_ARM_MAP_EVERY == 0:
            cmd_max = float(np.max(np.abs(a_cmd))) if a_cmd.size else 0.0
            print(
                f"[PI0 ARM MAP] step={self._pi0_arm_map_counter} "
                f"q_err_max={q_err_max:.5f} sat_ratio={sat_ratio:.3f} cmd_max={cmd_max:.3f}"
            )

        return a_cmd

    def _log_pi0_action(self, action, info):
        self._pi0_action_counter += 1
        if self._pi0_action_counter % PI0_DEBUG_ACTION_EVERY != 0:
            return

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        chunk_index = int(info.get("chunk_index", -1)) + 1
        chunk_len = int(info.get("chunk_len", 0))
        request_keys = info.get("request_keys", [])
        print(
            f"[PI0 ACTION] step={self._pi0_action_counter} "
            f"chunk={chunk_index}/{chunk_len} mode={PI0_ARM_ACTION_MODE} "
            f"arm={np.array2string(action[:7], precision=4, floatmode='fixed')} "
            f"gripper={float(action[7]):+.5f} keys={request_keys}"
        )

    def _run_shadow_compare(self, obs_dict, primary_action, primary_info):
        if not PI0_SHADOW_COMPARE or self.shadow_provider is None:
            return

        try:
            shadow_action, shadow_info = self.shadow_provider.forward(obs_dict, include_info=True)
            shadow_action = np.asarray(shadow_action, dtype=np.float32).reshape(-1)
            if shadow_action.size < 8:
                return

            diff = primary_action[:8] - shadow_action[:8]
            linf = float(np.max(np.abs(diff)))
            l2 = float(np.linalg.norm(diff))

            self._shadow_count += 1
            self._shadow_linf_samples.append(linf)
            self._shadow_l2_samples.append(l2)
            self._shadow_linf_samples = self._shadow_linf_samples[-2000:]
            self._shadow_l2_samples = self._shadow_l2_samples[-2000:]

            primary_sign = np.sign(primary_action[7])
            shadow_sign = np.sign(shadow_action[7])
            if primary_sign == shadow_sign:
                self._shadow_gripper_sign_match += 1

            primary_chunk_len = int(primary_info.get("chunk_len", 0))
            shadow_chunk_len = int(shadow_info.get("chunk_len", 0))
            if primary_chunk_len == shadow_chunk_len:
                self._shadow_chunk_len_match += 1

            primary_keys = tuple(primary_info.get("request_keys", []))
            shadow_keys = tuple(shadow_info.get("request_keys", []))
            if primary_keys == shadow_keys:
                self._shadow_request_key_match += 1

            if self._shadow_count % PI0_SHADOW_LOG_EVERY == 0:
                linf_arr = np.asarray(self._shadow_linf_samples, dtype=np.float32)
                l2_arr = np.asarray(self._shadow_l2_samples, dtype=np.float32)
                sign_match_ratio = self._shadow_gripper_sign_match / max(1, self._shadow_count)
                chunk_len_match_ratio = self._shadow_chunk_len_match / max(1, self._shadow_count)
                request_key_match_ratio = self._shadow_request_key_match / max(1, self._shadow_count)
                print(
                    "[PI0 SHADOW] "
                    f"n={self._shadow_count} "
                    f"linf_mean={linf_arr.mean():.5f} linf_p95={np.percentile(linf_arr,95):.5f} linf_p99={np.percentile(linf_arr,99):.5f} "
                    f"l2_mean={l2_arr.mean():.5f} l2_p95={np.percentile(l2_arr,95):.5f} l2_p99={np.percentile(l2_arr,99):.5f} "
                    f"gripper_sign_match={sign_match_ratio:.3f} "
                    f"chunk_len_match={chunk_len_match_ratio:.3f} "
                    f"request_key_match={request_key_match_ratio:.3f} "
                    f"chunk_primary={primary_chunk_len} chunk_shadow={shadow_chunk_len}"
                )
        except Exception as e:
            if PI0_BRIDGE_DEBUG:
                print(f"[WARNING] PI0 shadow compare failed: {e}")

    def _log_pi0_gripper(self, pi0_gripper_raw, is_grasping):
        self._pi0_gripper_counter += 1
        if pi0_gripper_raw > (self.grasp_threshold + self.hysteresis):
            self._pi0_gripper_high_seen = True

        if self._pi0_gripper_counter % PI0_DEBUG_GRIPPER_EVERY != 0:
            return

        activation_threshold = self.grasp_threshold + self.hysteresis
        deactivation_threshold = self.grasp_threshold - self.hysteresis
        print(
            f"[PI0 GRIPPER] step={self._pi0_gripper_counter} "
            f"raw={pi0_gripper_raw:+.5f} active={int(is_grasping)} "
            f"activate>{activation_threshold:.3f} deactivate<{deactivation_threshold:.3f} "
            f"bin>{self.gripper_bin_threshold:.3f}"
        )
        if not self._pi0_gripper_high_seen and self._pi0_gripper_counter >= 100:
            print(
                "[WARNING] PI0 gripper raw signal has never crossed activation threshold "
                f"({activation_threshold:.3f}) in {self._pi0_gripper_counter} steps."
            )

    def _fallback_action(self):
        if self._safe_action_valid:
            return self._safe_action.copy()
        return np.zeros(8, dtype=np.float32)

    def _try_reuse_provider_chunk(self):
        chunk = getattr(self.primary_provider, "_pred_action_chunk", None)
        if chunk is None:
            return None, None

        completed = int(getattr(self.primary_provider, "_actions_from_chunk_completed", 0))
        chunk_len = len(chunk)
        if completed < 0 or completed >= chunk_len:
            return None, None

        action = np.asarray(chunk[completed], dtype=np.float32).reshape(-1)
        if action.size < 8:
            return None, None

        try:
            setattr(self.primary_provider, "_actions_from_chunk_completed", completed + 1)
        except Exception:
            pass

        info = {
            "chunk_reused": True,
            "chunk_len": int(chunk_len),
            "chunk_index": int(completed),
            "request_keys": getattr(self.primary_provider, "_last_request_keys", []),
            "prompt_len": int(getattr(self.primary_provider, "_last_prompt_len", 0)),
            "has_tactile_values": bool(getattr(self.primary_provider, "_last_has_tactile_values", False)),
        }
        return action[:8], info

    def _provider_forward(self, obs_dict):
        try:
            action, info = self.primary_provider.forward(obs_dict, include_info=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.size < 8:
                raise ValueError(f"Expected PI0 action length >= 8, got {action.size}")
            action = action[:8]
            self._pi0_fail_count = 0
            return action, info, None
        except Exception as e:
            self._pi0_fail_count += 1
            self._maybe_stop_on_failure("PI0 inference", self._pi0_fail_count, PI0_MAX_CONSEC_FAIL, e)
            reused_action, reused_info = self._try_reuse_provider_chunk()
            if reused_action is not None:
                reused_info["provider_error"] = str(e)
                print(
                    f"[WARNING] PI0 inference failed (count={self._pi0_fail_count}/{PI0_MAX_CONSEC_FAIL}), "
                    "reusing cached chunk action."
                )
                return reused_action, reused_info, e
            print(
                f"[WARNING] PI0 inference failed (count={self._pi0_fail_count}/{PI0_MAX_CONSEC_FAIL}), "
                "falling back to last safe action."
            )
            return None, {"provider_error": str(e), "chunk_reused": True}, e

    def _is_grasping(self, pi0_gripper_raw):
        if self.tactile_active:
            return pi0_gripper_raw > (self.grasp_threshold - self.hysteresis)
        return pi0_gripper_raw > (self.grasp_threshold + self.hysteresis)

    def _build_arm_command(self, pi0_action, obs_dict, action_space):
        arm_raw = np.asarray(pi0_action[:7], dtype=np.float32)
        if PI0_ARM_ACTION_MODE == "joint_position" and "velocity" in action_space:
            current_pos = self._get_joint_positions(obs_dict)
            return self._map_pi0_position_to_velocity_action(arm_raw, current_pos)

        # Treat as direct velocity command.
        if PI0_ARM_ACTION_MODE not in {"joint_position", "joint_velocity"}:
            print(f"[WARNING] Unknown PI0_ARM_ACTION_MODE={PI0_ARM_ACTION_MODE}, using joint_velocity")
        return np.clip(arm_raw, -1.0, 1.0)

    def _maybe_run_tactile_override(self, obs_dict, action):
        if not self.tactile_active or self.tactile_adapter is None:
            return action, None

        try:
            curr_obs = extract_observation(self.openpi_config, obs_dict, save_to_disk=False)
            wrist_image_left = curr_obs.get("wrist_image_left") or curr_obs.get("wrist_image")
            wrist_image_right = curr_obs.get("wrist_image_right") or curr_obs.get("wrist_image")
            if wrist_image_left is None:
                raise ValueError("No wrist image available for tactile override.")
            if wrist_image_right is None:
                wrist_image_right = wrist_image_left

            current_gripper = float(curr_obs["gripper_position"][0])
            tactile_hist = self.tactile_reader.read_values()

            force_history = None
            adapter_cfg = getattr(self.tactile_adapter, "adapter_cfg", None)
            tactile_mode = getattr(adapter_cfg, "tactile_mode", "tactile_only")
            safety_enabled = bool(getattr(adapter_cfg, "force_safety_enabled", False))
            need_force = tactile_mode in {"force_only", "force_tactile"} or safety_enabled
            if need_force:
                force_history = self.tactile_reader.read_force_values(
                    k=getattr(adapter_cfg, "tactile_length", 50)
                )

            merged_action, tactile_delta = self.tactile_adapter.override_gripper(
                np.asarray(action, dtype=np.float32).copy(),
                wrist_image_left,
                wrist_image_right,
                tactile_hist,
                current_gripper,
                force_history=force_history,
                pi0_gate=0.0,
                absolute_clip=(0.0, 1.0),
                step_index=0,
            )
            merged_action = np.asarray(merged_action, dtype=np.float32).reshape(-1)
            if merged_action.size < 8:
                raise ValueError(f"Invalid merged action shape: {merged_action.shape}")

            self._tactile_fail_count = 0
            return merged_action[:8], float(tactile_delta)
        except Exception as e:
            self._tactile_fail_count += 1
            self._maybe_stop_on_failure("Tactile inference", self._tactile_fail_count, TACTILE_MAX_CONSEC_FAIL, e)
            print(
                f"[WARNING] Tactile inference failed (count={self._tactile_fail_count}/{TACTILE_MAX_CONSEC_FAIL}), "
                "falling back to PI0 gripper."
            )
            return action, None

    def forward(self, obs_dict, include_info=False):
        if self.should_stop:
            action = np.zeros(8, dtype=np.float32)
            if include_info:
                return action, {"robot_stopped": True, "reason": "policy_stop"}
            return action

        pi0_action, pi0_info, pi0_error = self._provider_forward(obs_dict)
        if pi0_action is None:
            action = self._fallback_action()
            if include_info:
                info = {
                    "tactile_active": False,
                    "pi0_error": str(pi0_error),
                    "pi0_fail_count": self._pi0_fail_count,
                    "tactile_fail_count": self._tactile_fail_count,
                    "robot_stopped": self.should_stop,
                    "pi0_provider": PI0_PROVIDER,
                }
                return action, info
            return action

        self._log_pi0_action(pi0_action, pi0_info)
        self._run_shadow_compare(obs_dict, pi0_action, pi0_info)

        action_space = "joint_velocity"
        if hasattr(obs_dict, "get"):
            # Try to infer from env state key if present.
            action_space = "joint_velocity"

        arm_cmd = self._build_arm_command(pi0_action, obs_dict, action_space)

        pi0_gripper_raw = float(pi0_action[7])
        is_grasping = self._is_grasping(pi0_gripper_raw)
        self._log_pi0_gripper(pi0_gripper_raw, is_grasping)
        ee_z = self._get_ee_z(obs_dict)
        self._log_state(ee_z, arm_cmd, pi0_gripper_raw)
        arm_cmd = self._guard_low_z(ee_z, arm_cmd, pi0_gripper_raw)

        if is_grasping and self.tactile_adapter is not None and not self.tactile_active:
            z_msg = "NA" if ee_z is None else f"{ee_z:.4f}"
            print(f"[TACTILE] Activated - pi0 grasping signal: {pi0_gripper_raw:.3f}, ee_z={z_msg}")
        if (not is_grasping) and self.tactile_active:
            z_msg = "NA" if ee_z is None else f"{ee_z:.4f}"
            print(f"[TACTILE] Deactivated - pi0 not grasping: {pi0_gripper_raw:.3f}, ee_z={z_msg}")

        self.tactile_active = bool(is_grasping and self.tactile_adapter is not None)
        self.last_gripper_action = 1.0 if pi0_gripper_raw > self.gripper_bin_threshold else 0.0

        # Position-space gripper for RobotEnv(gripper_action_space="position").
        pi0_gripper_cmd = 1.0 if pi0_gripper_raw > self.gripper_bin_threshold else 0.0

        action = np.zeros(8, dtype=np.float32)
        action[:7] = np.clip(arm_cmd, -1.0, 1.0)
        action[7] = float(np.clip(pi0_gripper_cmd, 0.0, 1.0))

        tactile_delta = None
        action, tactile_delta = self._maybe_run_tactile_override(obs_dict, action)

        action[:7] = np.clip(action[:7], -1.0, 1.0)
        action[7] = float(np.clip(action[7], 0.0, 1.0))

        self._safe_action = action.copy()
        self._safe_action_valid = True

        if include_info:
            info = {
                "tactile_active": self.tactile_active,
                "tactile_delta": tactile_delta,
                "pi0_gripper_raw": pi0_gripper_raw,
                "ee_z": ee_z,
                "pi0_provider": PI0_PROVIDER,
                "pi0_info": pi0_info,
                "pi0_fail_count": self._pi0_fail_count,
                "tactile_fail_count": self._tactile_fail_count,
                "robot_stopped": self.should_stop,
            }
            return action, info
        return action

    def reset(self):
        self.tactile_active = False
        self.last_gripper_action = 0.0
        self.should_stop = False
        self._safe_action = np.zeros(8, dtype=np.float32)
        self._safe_action_valid = False
        self._pi0_fail_count = 0
        self._tactile_fail_count = 0
        self._pi0_action_counter = 0
        self._pi0_arm_map_counter = 0
        self._pi0_arm_ema.fill(0.0)
        self._pi0_arm_prev.fill(0.0)
        self._pi0_arm_high_sat_streak = 0
        self._pi0_gripper_counter = 0
        self._pi0_gripper_high_seen = False
        self._state_counter = 0
        self._prev_ee_z = None
        self._prev_ee_t = None
        self._low_z_warn_counter = 0
        self._low_z_hard_guard_counter = 0
        self._pos_mode_mismatch_streak = 0
        self._shadow_count = 0
        self._shadow_linf_samples = []
        self._shadow_l2_samples = []
        self._shadow_gripper_sign_match = 0
        self._shadow_chunk_len_match = 0
        self._shadow_request_key_match = 0

        if self.primary_provider is not None:
            self.primary_provider.reset()
            self.primary_provider.set_instruction(self._instruction)
        if self.shadow_provider is not None:
            self.shadow_provider.reset()
            self.shadow_provider.set_instruction(self._instruction)

    def set_instruction(self, instruction):
        self._instruction = instruction
        if self.primary_provider is not None:
            self.primary_provider.set_instruction(instruction)
        if self.shadow_provider is not None:
            self.shadow_provider.set_instruction(instruction)

    @property
    def instruction(self):
        return self._instruction


def _setup_tactile_config():
    config_path = resolve_tactile_model_config(TACTILE_MODEL_CHECKPOINT, TACTILE_MODEL_CONFIG)
    config_path = maybe_autogenerate_gripper_stats(config_path)
    validate_gripper_stats_file(config_path)
    print(f"[INFO] Tactile checkpoint: {TACTILE_MODEL_CHECKPOINT}")
    print(f"[INFO] Tactile config: {config_path}")
    return config_path


@hydra.main(version_base=None, config_path="../FORTE/config/", config_name="baby_FORTE")
def main(cfg: DictConfig):
    shutdown_event = mp.Event() if hasattr(mp, "Event") else None

    if _RUN_LOG_PATH:
        log_file = Path(_RUN_LOG_PATH)
    else:
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"tactile_module_{int(time.time())}.log"

    handlers = [logging.StreamHandler()]
    if not _RUN_LOG_PATH:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)

    openpi_config = OpenPIConfigs()
    openpi_config.left_camera_id = "24395123"
    openpi_config.right_camera_id = "24013089"
    openpi_config.wrist_camera_id = "17225336"
    openpi_config.max_timesteps = 600
    openpi_config.remote_host = PI0_REMOTE_HOST
    openpi_config.remote_port = PI0_REMOTE_PORT
    openpi_config.open_loop_horizon = PI0_OPEN_LOOP_HORIZON

    print("Initializing components...")
    print(f"Runtime log: {log_file}")
    print(f"PI0 provider: {PI0_PROVIDER}")
    print(f"PI0 shadow compare: {PI0_SHADOW_COMPARE}")
    print(f"PI0 arm action mode: {PI0_ARM_ACTION_MODE}")
    print(f"PI0 joint delta max: {PI0_JOINT_DELTA_MAX}")
    print(f"PI0 open-loop horizon: {PI0_OPEN_LOOP_HORIZON}")
    if PI0_OPEN_LOOP_HORIZON == 1:
        print(
            "[WARNING] PI0_OPEN_LOOP_HORIZON=1 will execute only the first action of each chunk; "
            "safe but often slower/noisier for approach behaviors."
        )
    print(
        f"PI0 gripper thresholds: grasp={PI0_GRASP_THRESHOLD:.3f}, "
        f"hysteresis={PI0_GRASP_HYSTERESIS:.3f}, bin={PI0_GRIPPER_BIN_THRESHOLD:.3f}"
    )
    print(
        f"PI0 ee_z guard: warn={PI0_EE_Z_WARN:.4f}, hard_min="
        f"{'disabled' if PI0_EE_Z_HARD_MIN is None else f'{PI0_EE_Z_HARD_MIN:.4f}'}, "
        f"action={PI0_EE_Z_GUARD_ACTION}, require_open={PI0_EE_Z_GUARD_REQUIRE_OPEN}, "
        f"soft_scale={PI0_EE_Z_SOFT_SCALE}"
    )
    if PI0_EE_Z_GUARD_ACTION in {"retreat", "freeze_then_retreat"}:
        print(
            f"PI0 ee_z retreat config: freeze_timeout={PI0_EE_Z_FREEZE_TIMEOUT_STEPS}, "
            f"retreat_scale={PI0_EE_Z_RETREAT_SCALE:.3f}, "
            f"retreat_min_norm={PI0_EE_Z_RETREAT_MIN_NORM:.3f}"
        )
    print("[INFO] PI0 action semantics: 8-D arm+gripper action chunk (not end-effector pose).")
    if PI0_ARM_ACTION_MODE == "joint_velocity":
        print(
            "[INFO] PI0_ARM_ACTION_MODE=joint_velocity. "
            "If you observe weak target tracking, try joint_position."
        )
    else:
        print(
            "[INFO] PI0_ARM_ACTION_MODE=joint_position. "
            "If you observe high q_err/saturation, switch to joint_velocity."
        )
    if TACTILE_GRIPPER_STATS_SOURCE_H5:
        print(f"TACTILE_GRIPPER_STATS_SOURCE_H5: {TACTILE_GRIPPER_STATS_SOURCE_H5}")

    tactile_config_path = _setup_tactile_config()

    tactile_reader = None
    env = None
    try:
        tactile_reader = babyFORTEReader(cfg, shutdown_event)
        print("✓ Tactile/force reader initialized")

        adapter = TactileGripperAdapter(
            checkpoint_path=TACTILE_MODEL_CHECKPOINT,
            config_path=tactile_config_path,
        )
        print("✓ Tactile module adapter initialized")

        openpi_wrapper = OpenPIWrapper(openpi_config)
        openpi_wrapper.reset()
        print("✓ OpenPI wrapper initialized")

        bridge_provider = None
        if PI0_PROVIDER == "bridge" or PI0_SHADOW_COMPARE:
            bridge_provider = Pi0ActionBridge(openpi_config)
            bridge_provider.reset()
            print("✓ PI0 bridge initialized")

        if PI0_PROVIDER == "bridge":
            if bridge_provider is None:
                raise RuntimeError("PI0 provider is bridge but bridge initialization failed.")
            primary_provider = bridge_provider
            shadow_provider = openpi_wrapper if PI0_SHADOW_COMPARE else None
        else:
            primary_provider = openpi_wrapper
            shadow_provider = bridge_provider if PI0_SHADOW_COMPARE else None

        hybrid_policy = HybridPolicy(
            primary_provider=primary_provider,
            shadow_provider=shadow_provider,
            tactile_adapter=adapter,
            tactile_reader=tactile_reader,
            openpi_config=openpi_config,
            grasp_threshold=PI0_GRASP_THRESHOLD,
            hysteresis=PI0_GRASP_HYSTERESIS,
            gripper_bin_threshold=PI0_GRIPPER_BIN_THRESHOLD,
        )

        env = RobotEnv(
            action_space="joint_velocity",
            gripper_action_space="position",
            experiment_name="tactile_module_demo_v2",
            sensor_readers={"tactile_values": tactile_reader},
        )
        print("✓ Robot environment initialized")

        devices = {
            "spacemouse": SpaceMouse(),
            "keyboard": Keyboard(),
        }
        controller = HITLPolicy(devices, policy=hybrid_policy, robot_env=env)

        print("=" * 70)
        print("Hybrid Control System Ready")
        print("=" * 70)
        print("Arm: PI0 provider + mapping safeguards")
        print("Gripper: PI0 gate + tactile override on grasp")
        print(f"Failure mode: {FAILURE_MODE} (pi0={PI0_MAX_CONSEC_FAIL}, tactile={TACTILE_MAX_CONSEC_FAIL})")
        print("=" * 70)

        instruction = input("Enter instruction: ").strip()
        if not instruction:
            instruction = "No instruction provided."
        hybrid_policy.set_instruction(instruction)
        controller.set_instruction(instruction)
        print(f"Instruction set: {instruction}")

        step_count = 0
        loop_error_count = 0
        max_loop_errors = 20

        while True:
            loop_start = time.time()
            try:
                if hybrid_policy.should_stop:
                    print("[CRITICAL] Policy requested stop. Sending zero command and exiting.")
                    stop_action = np.zeros(8, dtype=np.float32)
                    try:
                        env.step(stop_action)
                    except Exception:
                        pass
                    break

                obs = env.get_observation()
                action = controller.forward(obs)
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                if action.size < 8:
                    raise ValueError(f"Invalid action shape {action.shape}, expected at least 8")
                action = action[:8]

                with _delay_keyboard_interrupt():
                    env.step(action)

                loop_error_count = 0

                if step_count % 20 == 0:
                    tactile_status = "ACTIVE" if hybrid_policy.tactile_active else "INACTIVE"
                    print(
                        f"[{step_count:04d}] Tactile: {tactile_status:8s} | "
                        f"pi0_gripper_bin={hybrid_policy.last_gripper_action:.0f} | "
                        f"cmd_gripper={action[-1]:+.3f}"
                    )

                step_count += 1
                elapsed = time.time() - loop_start
                target_period = 1.0 / max(float(env.control_hz), 1e-6)
                if elapsed < target_period:
                    time.sleep(target_period - elapsed)

            except KeyboardInterrupt:
                print("\n[INFO] Keyboard interrupt received")
                break
            except Exception as e:
                loop_error_count += 1
                logger.exception("[ERROR] Step %s failed: %s", step_count, e)
                print(f"[ERROR] Step {step_count} failed: {e}")

                if loop_error_count >= max_loop_errors:
                    print(f"[ERROR] Too many consecutive loop errors ({max_loop_errors}), stopping")
                    break
                time.sleep(0.1)

    finally:
        print("\nShutting down processes...")
        if tactile_reader is not None:
            try:
                tactile_reader.close()
            except Exception as e:
                print(f"[WARNING] Error closing tactile reader: {e}")
        if shutdown_event is not None:
            shutdown_event.set()
        print("Done.")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
