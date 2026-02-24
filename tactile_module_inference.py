#!/usr/bin/env python3
"""
Hybrid control: OpenPI inference + Tactile module gripper control.

What it does:
  - Uses OpenPI inference (pi0) for arm control via serve_policy.py
  - When pi0 initiates grasping (gripper action > threshold), switches to tactile module for gripper control
  - Arm always controlled by pi0, gripper switches between pi0 and tactile module

Usage:
  python tactile_module_demo_v2_improved.py

Notes:
  - Requires serve_policy.py to be running
  - Tactile module activates when pi0 gripper action > GRASP_THRESHOLD
  - Camera IDs are set to the same values used in 1-pi0.py; adjust if yours differ.
"""


# [1040] Tactile: INACTIVE | pi0_gripper: +0.521 | cmd_gripper: +0.521
# [TACTILE] Activated - pi0 grasping signal: 0.620
# [1060] Tactile: ACTIVE   | pi0_gripper: +0.484 | cmd_gripper: +0.298
# [TACTILE] Deactivated - pi0 not grasping: 0.240
# [1080] Tactile: INACTIVE | pi0_gripper: +0.001 | cmd_gripper: +0.001
# [1100] Tactile: INACTIVE | pi0_gripper: +0.316 | cmd_gripper: +0.316
# [TACTILE] Activated - pi0 grasping signal: 0.606
# [TACTILE] Deactivated - pi0 not grasping: 0.008
# [1120] Tactile: INACTIVE | pi0_gripper: +0.008 | cmd_gripper: +0.008

import contextlib
import logging
import os
import signal
import sys
import time
import multiprocessing as mp
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from FORTE.sensing.sensor import sensor_data_updater
from FORTE.scripts.sys_utils import SharedRingBuffer, opencv_visualizer

from r2d2.robot_env import RobotEnv

from util.openpi import DROID_CONTROL_FREQUENCY, OpenPIConfigs, extract_observation

# Prefer fork to avoid semaphore permissions issues on some systems; ignore if already set.
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

# Make sure we can import the tactile module
repo_root = Path(__file__).resolve().parents[2]  # /home/pi0/multi-modal
tactile_module_dir = repo_root / "tactile_module"
scripts_dir = Path(__file__).resolve().parent  # scripts directory
if tactile_module_dir.exists():
    for extra_path in (repo_root, tactile_module_dir, scripts_dir):
        extra_path_str = str(extra_path)
        if extra_path_str not in sys.path:
            sys.path.insert(0, extra_path_str)
else:
    raise ImportError(f"tactile_module not found at expected path: {tactile_module_dir}")

# Patch transformers to support dinov3_vit in older versions (Python 3.8 compatibility)
# import transformers_dinov3_patch  # noqa: F401  # NOT NEEDED in droid-tact (Python 3.9+)

from tactile_module.robot_inference_adapter import TactileGripperAdapter

cwd = os.getcwd()
sys.path.append(cwd)

# Ensure openpi_client is in path (for editable installs)
openpi_client_path = repo_root.parent / "openpi" / "packages" / "openpi-client" / "src"
if openpi_client_path.exists() and str(openpi_client_path) not in sys.path:
    sys.path.insert(0, str(openpi_client_path))

# Ensure oculus_reader is in path (for editable installs)
oculus_reader_path = scripts_dir / "src" / "oculus_reader"
if oculus_reader_path.exists() and str(oculus_reader_path) not in sys.path:
    sys.path.insert(0, str(oculus_reader_path))

from devices import SpaceMouse, Keyboard
from interfaces import HITLPolicy, OpenPIWrapper


class babyFORTEReader:
    """Spin up the tactile sensor process and expose recent tactile history."""

    def __init__(self, cfg, shutdown_event):
        self.shutdown_event = shutdown_event
        self.shared_sensor_buffer = SharedRingBuffer(cfg.buffer.size, cfg.buffer.num_channels, "d")

        self.sensor_process = mp.Process(
            target=sensor_data_updater,
            args=(cfg.elvrgripper, self.shared_sensor_buffer),
            kwargs={"mode": "process", "shutdown_event": shutdown_event},
        )
        self.sensor_process.start()
        time.sleep(2)

        self.visualizer_process = mp.Process(
            target=opencv_visualizer,
            args=(self.shared_sensor_buffer, cfg.buffer, None, None, shutdown_event),
        )
        self.visualizer_process.start()

    def read_values(self):
        """Read tactile values with error handling."""
        try:
            if self.shared_sensor_buffer.is_empty():
                return np.zeros((self.shared_sensor_buffer.num_channels,))
            return self.shared_sensor_buffer.get_latest_freq_history()
        except Exception as e:
            print(f"[WARNING] Failed to read tactile values: {e}")
            return np.zeros((self.shared_sensor_buffer.num_channels,))

    def close(self):
        if self.shutdown_event:
            self.shutdown_event.set()
        for proc in [self.visualizer_process, self.sensor_process]:
            if proc.is_alive():
                proc.terminate()
        time.sleep(1)
        for proc in [self.visualizer_process, self.sensor_process]:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1)
        self.shared_sensor_buffer.close()


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


class HybridPolicy:
    """
    Hybrid policy that combines OpenPI inference with tactile module.
    
    - Arm control: Always from OpenPI (pi0)
    - Gripper control: Switches between OpenPI and tactile module based on grasping signal
    """
    
    def __init__(self, openpi_policy, tactile_adapter, tactile_reader, openpi_config, grasp_threshold=0.5, hysteresis=0.1):
        self.openpi_policy = openpi_policy
        self.tactile_adapter = tactile_adapter
        self.tactile_reader = tactile_reader
        self.openpi_config = openpi_config
        self.grasp_threshold = grasp_threshold
        self.hysteresis = hysteresis  # Buffer to prevent rapid switching
        self.tactile_active = False
        self.last_gripper_action = 0.0
        self._cached_obs = None  # Cache observation to avoid double extraction
        self.should_stop = False  # Flag to signal robot should stop
        
    def forward(self, obs_dict, include_info=False):
        """
        Forward pass: Get action from OpenPI, override gripper with tactile if grasping.
        
        Returns:
            action: [7 arm joints, 1 gripper] - arm from pi0, gripper from pi0 or tactile
        """
        try:
            # Get action from OpenPI (pi0 inference)
            # Note: OpenPIWrapper.forward() will call extract_observation internally,
            # so we cache it here to avoid double extraction when tactile is active
            if include_info:
                pi0_action, info = self.openpi_policy.forward(obs_dict, include_info=True)
            else:
                pi0_action = self.openpi_policy.forward(obs_dict, include_info=False)
                info = {}
            
            # Validate action shape
            if len(pi0_action) != 8:
                raise ValueError(f"Expected action of length 8, got {len(pi0_action)}")
            
            # Get raw gripper action from pi0
            pi0_gripper_raw = pi0_action[-1]
            
            # Apply hysteresis to RAW value before binarization to prevent rapid switching
            # Use higher threshold to activate, lower threshold to deactivate
            if self.tactile_active:
                # Currently active: need to drop below (threshold - hysteresis) to deactivate
                # This prevents deactivation when signal briefly dips below 0.5
                is_grasping_raw = pi0_gripper_raw > (self.grasp_threshold - self.hysteresis)
            else:
                # Currently inactive: need to rise above (threshold + hysteresis) to activate
                # This prevents activation when signal briefly rises above 0.5
                is_grasping_raw = pi0_gripper_raw > (self.grasp_threshold + self.hysteresis)
            
            # Binarize based on hysteresis-filtered decision
            if is_grasping_raw:
                pi0_gripper_action = 1.0
            else:
                pi0_gripper_action = 0.0
            
            self.last_gripper_action = pi0_gripper_action
            is_grasping = is_grasping_raw
            
            if is_grasping:
                # pi0 decided to grasp -> activate tactile module for gripper control
                if not self.tactile_active:
                    print(f"[Module] Activated - pi0 grasping signal (raw: {pi0_gripper_raw:.3f}, binarized: {pi0_gripper_action:.0f})")
                    self.tactile_active = True
                
                # Get tactile history and wrist image
                # Extract observation only once (OpenPIWrapper already did it, but we need it here)
                # Cache it to avoid redundant extraction
                if self._cached_obs is None:
                    try:
                        curr_obs = extract_observation(
                            self.openpi_config,
                            obs_dict,
                            save_to_disk=False,
                        )
                        self._cached_obs = curr_obs
                    except Exception as e:
                        import traceback
                        error_msg = f"[CRITICAL ERROR] Failed to extract observation: {e}\n{traceback.format_exc()}"
                        print(error_msg)
                        import logging
                        logging.critical(error_msg)
                        # Set flag to stop robot
                        self.should_stop = True
                        # Stop robot immediately
                        stop_action = np.zeros(8, dtype=np.float32)
                        if include_info:
                            info["tactile_active"] = False
                            info["tactile_error"] = str(e)
                            info["robot_stopped"] = True
                            return stop_action, info
                        return stop_action
                else:
                    curr_obs = self._cached_obs
                
                # Get wrist images (left and right) - new architecture requires dual cameras
                wrist_image_left = curr_obs.get("wrist_image_left") or curr_obs.get("wrist_image")
                wrist_image_right = curr_obs.get("wrist_image_right") or curr_obs.get("wrist_image")
                
                # Use same image for both if only one camera available
                if wrist_image_left is None:
                    import traceback
                    error_msg = f"[CRITICAL ERROR] No wrist image found in observation - cannot use tactile module\n{traceback.format_exc()}"
                    print(error_msg)
                    import logging
                    logging.critical(error_msg)
                    # Set flag to stop robot
                    self.should_stop = True
                    # Stop robot immediately
                    stop_action = np.zeros(8, dtype=np.float32)
                    if include_info:
                        info["tactile_active"] = False
                        info["tactile_error"] = "No wrist image available"
                        info["robot_stopped"] = True
                        return stop_action, info
                    return stop_action
                
                if wrist_image_right is None:
                    wrist_image_right = wrist_image_left  # Use same image for both cameras
                
                current_gripper = float(curr_obs["gripper_position"][0])
                
                # Read tactile values with error handling
                try:
                    tactile_hist = self.tactile_reader.read_values()
                except Exception as e:
                    import traceback
                    error_msg = f"[CRITICAL ERROR] Failed to read tactile history: {e}\n{traceback.format_exc()}"
                    print(error_msg)
                    import logging
                    logging.critical(error_msg)
                    # Set flag to stop robot
                    self.should_stop = True
                    # Stop robot immediately
                    stop_action = np.zeros(8, dtype=np.float32)
                    if include_info:
                        info["tactile_active"] = False
                        info["tactile_error"] = f"Tactile sensor read failed: {e}"
                        info["robot_stopped"] = True
                        return stop_action, info
                    return stop_action
                
                # Override gripper with tactile module prediction
                try:
                    base_action = pi0_action.copy()  # Use pi0 action as base (arm control)
                    merged_action, tactile_delta = self.tactile_adapter.override_gripper(
                        base_action,
                        wrist_image_left,  # Left wrist camera image
                        wrist_image_right,  # Right wrist camera image
                        tactile_hist,
                        current_gripper,
                        pi0_gate=0.0,  # Always use tactile delta when active
                        absolute_clip=(-1.0, 1.0),
                        step_index=0,  # Use first step from action chunk
                    )
                    
                    # Validate merged action
                    if len(merged_action) != 8:
                        raise ValueError(f"Tactile adapter returned action of length {len(merged_action)}, expected 8")
                    
                    # Return action: arm from pi0, gripper from tactile module
                    if include_info:
                        info["tactile_active"] = True
                        info["tactile_delta"] = tactile_delta
                        info["pi0_gripper"] = pi0_gripper_action
                        return merged_action, info
                    else:
                        return merged_action
                        
                except Exception as e:
                    import traceback
                    error_msg = f"[CRITICAL ERROR] Tactile module failed: {e}\n{traceback.format_exc()}"
                    print(error_msg)
                    # Log to file if possible
                    import logging
                    logging.critical(error_msg)
                    # Set flag to stop robot
                    self.should_stop = True
                    # Stop robot immediately - return zero velocity command
                    stop_action = np.zeros(8, dtype=np.float32)
                    if include_info:
                        info["tactile_active"] = False
                        info["tactile_error"] = str(e)
                        info["robot_stopped"] = True
                        return stop_action, info
                    return stop_action
            else:
                # pi0 not grasping -> use pi0 gripper control
                if self.tactile_active:
                    print(f"[Module] Deactivated - pi0 not grasping (raw: {pi0_gripper_raw:.3f}, binarized: {pi0_gripper_action:.0f})")
                    self.tactile_active = False
                    self._cached_obs = None  # Clear cache when deactivating
                
                # Binarize gripper action (like 1-pi0.py does) before sending to robot
                # pi0 outputs continuous gripper values, but robot expects binary (0.0 or 1.0)
                if pi0_action[-1].item() > 0.5:
                    binarized_action = np.concatenate([pi0_action[:-1], np.ones((1,))])
                else:
                    binarized_action = np.concatenate([pi0_action[:-1], np.zeros((1,))])
                
                # Use pi0 action with binarized gripper (both arm and gripper from pi0)
                if include_info:
                    info["tactile_active"] = False
                    info["pi0_gripper"] = pi0_gripper_action
                    return binarized_action, info
                else:
                    return binarized_action
                    
        except Exception as e:
            print(f"[ERROR] Policy forward pass failed: {e}")
            # Return zero action as safe fallback
            zero_action = np.zeros(8, dtype=np.float32)
            if include_info:
                info = {"error": str(e), "tactile_active": False}
                return zero_action, info
            return zero_action
    
    def reset(self):
        """Reset policy state."""
        self.tactile_active = False
        self.last_gripper_action = 0.0
        self._cached_obs = None
        self.should_stop = False
        if self.openpi_policy is not None:
            self.openpi_policy.reset()
    
    def set_instruction(self, instruction):
        """Set instruction for OpenPI policy."""
        if self.openpi_policy is not None:
            self.openpi_policy.set_instruction(instruction)
    
    @property
    def instruction(self):
        """Get current instruction."""
        if self.openpi_policy is not None:
            return self.openpi_policy.instruction
        return "No instruction provided."


@hydra.main(config_path="../FORTE/config/", config_name="baby_FORTE")
def main(cfg: DictConfig):
    shutdown_event = mp.Event() if hasattr(mp, 'Event') else None

    # Setup logging for tactile module errors
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"tactile_module_{int(time.time())}.log"
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    logger = logging.getLogger(__name__)

    # Camera IDs: adjust to your hardware if needed.
    openpi_config = OpenPIConfigs()
    openpi_config.left_camera_id = "24395123"
    openpi_config.right_camera_id = "24013089"
    openpi_config.wrist_camera_id = "17225336"
    openpi_config.max_timesteps = 600

    print("Initializing components...")
    print(f"Error logs will be saved to: {log_file}")
    
    # Initialize tactile sensor reader
    try:
        tactile_reader = babyFORTEReader(cfg, shutdown_event)
        print("✓ Tactile sensor reader initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize tactile reader: {e}")
        raise

    # Initialize tactile module adapter
    try:
        adapter = TactileGripperAdapter(
            checkpoint_path="/home/pi0/multi-modal/tactile_module/checkpoints/delta_gripper.pt",
            config_path="/home/pi0/multi-modal/tactile_module/configs/default.yaml",
        )
        print("✓ Tactile module adapter initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize tactile adapter: {e}")
        raise

    # Initialize OpenPI policy wrapper
    try:
        openpi_policy = OpenPIWrapper(openpi_config)
        # IMPORTANT: Reset OpenPI policy to initialize internal state
        openpi_policy.reset()
        print("✓ OpenPI policy wrapper initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize OpenPI policy: {e}")
        print("[INFO] Make sure serve_policy.py is running")
        raise

    # Create hybrid policy
    hybrid_policy = HybridPolicy(
        openpi_policy=openpi_policy,
        tactile_adapter=adapter,
        tactile_reader=tactile_reader,
        openpi_config=openpi_config,
        grasp_threshold=0.5,  # Threshold for raw gripper value: >0.5 = grasping, <=0.5 = not grasping
        hysteresis=0.1,  # Hysteresis buffer applied to RAW value: activate when raw > 0.6, deactivate when raw < 0.4
    )

    # Initialize robot environment
    try:
        env = RobotEnv(
            action_space="joint_velocity",
            gripper_action_space="position",
            experiment_name="tactile_module_demo_v2",
            sensor_readers={"tactile_values": tactile_reader},
        )
        print("✓ Robot environment initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize robot environment: {e}")
        raise

    # Initialize devices (SpaceMouse, Keyboard)
    try:
        devices = {
            "spacemouse": SpaceMouse(),
            "keyboard": Keyboard(),
        }
        print("✓ Input devices initialized")
    except Exception as e:
        print(f"[WARNING] Failed to initialize some devices: {e}")
        devices = {}

    # Create HITL controller with hybrid policy
    controller = HITLPolicy(devices, policy=hybrid_policy, robot_env=env)

    print("=" * 60)
    print("Hybrid Control System Ready")
    print("=" * 60)
    print(f"Arm control: OpenPI (pi0) inference")
    print(f"Gripper control: Switches between OpenPI and Tactile Module")
    print(f"  - When pi0 gripper > {hybrid_policy.grasp_threshold + hybrid_policy.hysteresis:.2f} (grasping): Use Tactile Module")
    print(f"  - When pi0 gripper <= {hybrid_policy.grasp_threshold - hybrid_policy.hysteresis:.2f} (not grasping): Use OpenPI")
    print(f"  - Hysteresis: ±{hybrid_policy.hysteresis:.2f} to prevent rapid switching")
    print("=" * 60)
    print("Starting control loop. Press Ctrl+C to stop.")
    print()
    
    # Prompt for instruction before starting (like 1-pi0.py does)
    instruction = input("Enter instruction: ")
    hybrid_policy.set_instruction(instruction)
    print(f"Instruction set: {instruction}")
    print()

    try:
        step_count = 0
        error_count = 0
        max_errors = 10
        
        while True:
            start = time.time()
            try:
                # Check if tactile module requested stop
                if hybrid_policy.should_stop:
                    print("\n[CRITICAL] Tactile module error detected - stopping robot and exiting")
                    logger.critical("Tactile module error - stopping robot and exiting control loop")
                    # Send final stop command
                    stop_action = np.zeros(8, dtype=np.float32)
                    try:
                        env.step(stop_action)
                    except:
                        pass
                    break
                
                # Get observation from robot
                obs = env.get_observation()
                
                # Get action from hybrid policy
                action = controller.forward(obs)
                
                # Check again after getting action (in case error occurred during forward pass)
                if hybrid_policy.should_stop:
                    print("\n[CRITICAL] Tactile module error detected - stopping robot and exiting")
                    logger.critical("Tactile module error detected during forward pass - stopping robot")
                    # Send final stop command
                    stop_action = np.zeros(8, dtype=np.float32)
                    try:
                        env.step(stop_action)
                    except:
                        pass
                    break
                
                # Validate action before execution
                if len(action) != 8:
                    raise ValueError(f"Invalid action shape: {len(action)}, expected 8")
                
                # Execute action
                with _delay_keyboard_interrupt():
                    env.step(action)
                
                # Reset error count on successful step
                error_count = 0
                
                # Print status every 20 steps
                if step_count % 20 == 0:
                    pi0_gripper = hybrid_policy.last_gripper_action  # This is now binarized (0 or 1)
                    tactile_status = "ACTIVE" if hybrid_policy.tactile_active else "INACTIVE"
                    print(
                        f"[{step_count:04d}] Tactile Module : {tactile_status:8s} | "
                        f"pi0_gripper: {pi0_gripper:.0f} | "
                        f"cmd_gripper: {action[-1]:+.3f}"
                    )
                
                step_count += 1

                # Maintain control frequency
                elapsed = time.time() - start
                if elapsed < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed)
                    
            except KeyboardInterrupt:
                print("\n[INFO] Keyboard interrupt received")
                break
            except Exception as e:
                error_count += 1
                import traceback
                error_msg = f"[ERROR] Step {step_count} failed: {e}\n{traceback.format_exc()}"
                print(error_msg)
                logger.error(error_msg)
                
                # If tactile module error, stop immediately
                if "tactile" in str(e).lower() or "CRITICAL ERROR" in str(e):
                    print(f"[CRITICAL] Tactile module error detected - stopping robot immediately")
                    logger.critical(f"Tactile module error - stopping robot: {e}")
                    # Send stop command to robot
                    try:
                        stop_action = np.zeros(8, dtype=np.float32)
                        env.step(stop_action)
                    except:
                        pass
                    break
                
                if error_count >= max_errors:
                    print(f"[ERROR] Too many consecutive errors ({max_errors}), shutting down")
                    logger.error(f"Too many consecutive errors - shutting down")
                    # Send stop command before shutdown
                    try:
                        stop_action = np.zeros(8, dtype=np.float32)
                        env.step(stop_action)
                    except:
                        pass
                    break
                # Brief pause before retrying
                time.sleep(0.1)
                
    finally:
        print("\nShutting down processes...")
        try:
            tactile_reader.close()
        except Exception as e:
            print(f"[WARNING] Error closing tactile reader: {e}")
        if shutdown_event is not None:
            shutdown_event.set()
        print("Done.")


if __name__ == "__main__":
    # Ensure working directory is script dir so Hydra finds config
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
