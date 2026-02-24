#!/usr/bin/env python3
"""
Naive grasping: Tactile module only for gripper control with teleoperation.

What it does:
  - Uses tactile module for gripper control (continuous gripper commands)
  - Arm control: Manual teleoperation via SpaceMouse/GUI (like jaelyn_demo.py)
  - No pi0/OpenPI inference - pure tactile-based grasping
  - No instruction functionality - direct teleoperation only

Usage:
  python naive_grasping.py

Notes:
  - Camera IDs are set to the same values used in 1-pi0.py; adjust if yours differ.
  - Arm can be controlled manually via SpaceMouse/GUI
  - Gripper is always controlled by tactile module
"""

import os
import sys
import signal
import time
import multiprocessing as mp
from pathlib import Path
from multiprocessing import Event

import hydra
import numpy as np
from omegaconf import DictConfig

from FORTE.sensing.sensor import sensor_data_updater
from FORTE.scripts.sys_utils import SharedRingBuffer, opencv_visualizer

from r2d2.robot_env import RobotEnv
from r2d2.user_interface.data_collector import DataCollecter
from r2d2.user_interface.gui import RobotGUI

from util.openpi import OpenPIConfigs, extract_observation


class NaiveGraspingGUI(RobotGUI):
    """Custom GUI for naive grasping that initializes with dummy tasks to prevent crashes."""
    
    def __init__(self, robot=None, fullscreen=False):
        super().__init__(robot, fullscreen)
        # Initialize with dummy task to prevent empty task list errors
        self.info["fixed_tasks"] = ["Naive grasping"]
        self.info["new_tasks"] = []

# Prefer spawn to match jaelyn_demo.py
try:
    mp.set_start_method("spawn", force=True)
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

from tactile_module.robot_inference_adapter import TactileGripperAdapter

cwd = os.getcwd()
sys.path.append(cwd)

# Ensure oculus_reader is in path (for editable installs)
oculus_reader_path = scripts_dir / "src" / "oculus_reader"
if oculus_reader_path.exists() and str(oculus_reader_path) not in sys.path:
    sys.path.insert(0, str(oculus_reader_path))

from devices import SpaceMouse, Keyboard
from interfaces import HITLPolicy, HumanInterventionReader


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

    def reset(self):
        """Reset tactile reader (for compatibility with DataCollecter)."""
        pass


class TactileGripperController:
    """
    Wrapper controller that combines HITLPolicy arm control with tactile module gripper control.
    This allows teleoperation via GUI while tactile module controls the gripper.
    
    Features:
    - Toggle tactile gripper control on/off using SpaceMouse button B
    - When disabled: gripper maintains current position (arm still teleoperated)
    - When enabled: tactile module controls gripper automatically
    """

    def __init__(self, hitl_controller, tactile_adapter, tactile_reader, openpi_config):
        self.hitl_controller = hitl_controller
        self.tactile_adapter = tactile_adapter
        self.tactile_reader = tactile_reader
        self.openpi_config = openpi_config
        self.tactile_enabled = False  # Start with tactile control DISABLED (enable after positioning)
        self.last_gripper_position = None
        self.last_button_b_state = False  # Track button B state for toggle detection
        self.status_print_counter = 0  # Counter for periodic status messages
        self.first_run = True  # Flag to print initial status

    def forward(self, obs_dict, include_info=False):
        """
        Forward pass: Get arm from HITL controller, gripper from tactile module (if enabled).
        
        Returns:
            action: [7 arm joints, 1 gripper] - arm from teleoperation, gripper from tactile module (if enabled) or hold position
        """
        try:
            # Print initial status on first run
            if self.first_run:
                print(f"\n{'='*60}")
                print("[TACTILE CONTROL] Initial state: DISABLED")
                print("Gripper will hold current position until you enable tactile control.")
                print("Press SpaceMouse Button B to enable tactile gripper control.")
                print(f"{'='*60}\n")
                self.first_run = False
            
            # Check if tactile control should be toggled (SpaceMouse button B or keyboard 'b')
            button_b_pressed = False
            
            # Try accessing button state directly from HIDReader
            try:
                if hasattr(self.hitl_controller, 'oculus_reader'):
                    _, buttons = self.hitl_controller.oculus_reader.get_transformations_and_buttons()
                    # Check SpaceMouse button B
                    button_b_pressed = buttons.get("B", False)
                    # Also check for keyboard 'b' key (if keyboard buttons are available)
                    if not button_b_pressed:
                        # Check common keyboard button names
                        for key_name in ["b", "B", "key_b"]:
                            if key_name in buttons:
                                button_b_pressed = buttons.get(key_name, False)
                                if button_b_pressed:
                                    break
                else:
                    # Fallback to get_info method
                    controller_info = self.hitl_controller.get_info()
                    button_b_pressed = controller_info.get("failure", False)
            except Exception as e:
                # Fallback to get_info method if direct access fails
                try:
                    controller_info = self.hitl_controller.get_info()
                    button_b_pressed = controller_info.get("failure", False)
                except:
                    button_b_pressed = False
            
            # Debug: Print button state and available buttons (only first few times)
            if self.status_print_counter < 5:
                if hasattr(self.hitl_controller, 'oculus_reader'):
                    try:
                        _, buttons = self.hitl_controller.oculus_reader.get_transformations_and_buttons()
                        if self.status_print_counter == 0:
                            print(f"[DEBUG] Available buttons: {list(buttons.keys())}")
                            print(f"[DEBUG] Button B value: {buttons.get('B', 'NOT FOUND')}")
                    except:
                        pass
            
            # Detect button B press (transition from False to True)
            if button_b_pressed and not self.last_button_b_state:
                self.tactile_enabled = not self.tactile_enabled
                status = "ENABLED" if self.tactile_enabled else "DISABLED"
                print(f"\n{'='*60}")
                print(f"[TACTILE CONTROL] {status}")
                print(f"{'='*60}")
                if self.tactile_enabled:
                    print("✓ Tactile module is now controlling the gripper automatically.")
                    print("  The gripper will close when it detects contact with objects.")
                else:
                    print("✗ Tactile module DISABLED - gripper will hold current position.")
                    print("  Use SpaceMouse to position arm, then press Button B to enable grasping.")
                print(f"{'='*60}\n")
            
            self.last_button_b_state = button_b_pressed
            
            # Print status every 100 steps (~3 seconds at 30Hz)
            self.status_print_counter += 1
            if self.status_print_counter % 100 == 0:
                status = "ACTIVE" if self.tactile_enabled else "DISABLED (hold position)"
                print(f"[Status] Tactile gripper control: {status}")
            
            # Get arm action from HITL controller (teleoperation)
            arm_action_full = self.hitl_controller.forward(obs_dict, include_info=False)
            # Extract arm joints (first 7) - gripper will be overridden by tactile module (if enabled)
            arm_action = arm_action_full[:7]

            # Extract observation for wrist images and gripper position
            try:
                curr_obs = extract_observation(
                    self.openpi_config,
                    obs_dict,
                    save_to_disk=False,
                )
            except Exception as e:
                print(f"[ERROR] Failed to extract observation: {e}")
                # Fallback: return arm action with zero gripper
                fallback_action = np.concatenate([arm_action, np.array([0.0])])
                if include_info:
                    return fallback_action, {"error": str(e)}
                return fallback_action

            # Get wrist images (left and right)
            wrist_image_left = curr_obs.get("wrist_image_left") or curr_obs.get("wrist_image")
            wrist_image_right = curr_obs.get("wrist_image_right") or curr_obs.get("wrist_image")

            if wrist_image_left is None:
                print("[WARNING] No wrist image found - using zero gripper")
                fallback_action = np.concatenate([arm_action, np.array([0.0])])
                if include_info:
                    return fallback_action, {"warning": "No wrist image available"}
                return fallback_action

            if wrist_image_right is None:
                wrist_image_right = wrist_image_left  # Use same image for both cameras

            current_gripper = float(curr_obs["gripper_position"][0])
            
            # If tactile control is disabled, maintain current gripper position
            if not self.tactile_enabled:
                # Hold current gripper position (zero velocity command)
                gripper_action = current_gripper  # Position command, not velocity
                final_action = np.concatenate([arm_action, np.array([gripper_action])])
                if include_info:
                    info = {
                        "tactile_active": False,
                        "tactile_delta": 0.0,
                        "arm_source": "teleoperation",
                        "gripper_mode": "hold_position",
                    }
                    return final_action, info
                else:
                    return final_action

            # Tactile control is enabled - use tactile module for gripper
            # Read tactile values
            try:
                tactile_hist = self.tactile_reader.read_values()
            except Exception as e:
                print(f"[WARNING] Failed to read tactile history: {e}")
                fallback_action = np.concatenate([arm_action, np.array([current_gripper])])
                if include_info:
                    return fallback_action, {"warning": f"Tactile sensor read failed: {e}"}
                return fallback_action

            # Create base action: arm from teleoperation + dummy gripper (will be overridden)
            base_action = np.concatenate([arm_action, np.array([0.0])])

            # Get gripper command from tactile module
            try:
                merged_action, tactile_delta = self.tactile_adapter.override_gripper(
                    base_action,
                    wrist_image_left,
                    wrist_image_right,
                    tactile_hist,
                    current_gripper,
                    pi0_gate=0.0,  # Always use tactile delta (no pi0)
                    absolute_clip=(-1.0, 1.0),
                    step_index=0,
                )

                # Validate merged action
                if len(merged_action) != 8:
                    raise ValueError(f"Tactile adapter returned action of length {len(merged_action)}, expected 8")

                # Return action: arm from teleoperation, gripper from tactile module
                if include_info:
                    info = {
                        "tactile_active": True,
                        "tactile_delta": tactile_delta,
                        "arm_source": "teleoperation",
                        "gripper_mode": "tactile_control",
                    }
                    return merged_action, info
                else:
                    return merged_action

            except Exception as e:
                print(f"[WARNING] Tactile module failed: {e}")
                # Fallback: return arm action with current gripper position
                fallback_action = np.concatenate([arm_action, np.array([current_gripper])])
                if include_info:
                    return fallback_action, {"warning": str(e)}
                return fallback_action

        except Exception as e:
            print(f"[ERROR] Controller forward pass failed: {e}")
            zero_action = np.zeros(8, dtype=np.float32)
            if include_info:
                return zero_action, {"error": str(e)}
            return zero_action

    def reset_state(self):
        """Reset controller state."""
        self.hitl_controller.reset_state()

    def get_info(self):
        """Get controller info (for compatibility with DataCollecter)."""
        return self.hitl_controller.get_info()

    def reset(self):
        """Reset controller (alias for reset_state for compatibility)."""
        self.reset_state()

    def set_instruction(self, instruction):
        """Dummy method - no instruction functionality in naive grasping."""
        # Do nothing - we don't use instructions
        pass


@hydra.main(config_path="../FORTE/config/", config_name="baby_FORTE")
def main(cfg: DictConfig):
    shutdown_event = Event()

    def handle_exit(signum, frame):
        print(f"Signal {signum} received. Exiting...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Camera IDs: adjust to your hardware if needed.
    openpi_config = OpenPIConfigs()
    openpi_config.left_camera_id = "24395123"
    openpi_config.right_camera_id = "24013089"
    openpi_config.wrist_camera_id = "17225336"
    openpi_config.max_timesteps = 600

    print("Initializing components...")

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

    # Initialize devices (SpaceMouse, Keyboard) - like jaelyn_demo.py
    devices = {"spacemouse": SpaceMouse(reset_with_idle=False), "keyboard": Keyboard()}

    # Initialize robot environment
    try:
        env = RobotEnv(
            action_space="joint_velocity",
            gripper_action_space="position",
            experiment_name="naive_grasping",
            sensor_readers={
                "tactile_values": tactile_reader,
                "human_intervention": HumanInterventionReader(**devices),
            },
        )
        print("✓ Robot environment initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize robot environment: {e}")
        raise

    # Create HITL controller for teleoperation (no policy - direct teleoperation)
    hitl_controller = HITLPolicy(devices, policy=None, robot_env=env)

    # Create wrapper controller that combines teleoperation with tactile gripper control
    controller = TactileGripperController(
        hitl_controller=hitl_controller,
        tactile_adapter=adapter,
        tactile_reader=tactile_reader,
        openpi_config=openpi_config,
    )

    print("=" * 60)
    print("Naive Grasping System Ready")
    print("=" * 60)
    print("Arm control: Teleoperation via SpaceMouse")
    print("Gripper control: Tactile Module (toggle with SpaceMouse Button B)")
    print("")
    print("IMPORTANT: Tactile control starts DISABLED")
    print("")
    print("Usage:")
    print("  1. Start trajectory - gripper will HOLD current position")
    print("  2. Teleoperate arm to position using SpaceMouse")
    print("  3. Press SpaceMouse Button B to ENABLE tactile gripper control")
    print("  4. Tactile module will automatically control gripper to grasp")
    print("  5. Press Button B again to DISABLE if needed")
    print("")
    print("Initial state: ✗ Tactile control DISABLED (gripper holds position)")
    print("=" * 60)

    # Create DataCollecter and launch GUI (like jaelyn_demo.py)
    try:
        data_collector = DataCollecter(env=env, controller=controller, policy=None, save_data=False)
        # Use custom GUI that initializes with dummy tasks to prevent crashes
        NaiveGraspingGUI(robot=data_collector)
    finally:
        print("\nCleaning up resources...")
        tactile_reader.close()
        print("Resources cleaned up.")


if __name__ == "__main__":
    main()
