import os
import sys
import time
import signal
from multiprocessing import Process, Event
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import hydra
import numpy as np
from omegaconf import DictConfig

from FORTE.sensing.sensor import sensor_data_updater
from FORTE.scripts.force_est_nn import force_estimator_update_loop_restartsafe, MLPInference
from FORTE.scripts.sys_utils import SharedRingBuffer, ForceRingBuffer, opencv_visualizer

from r2d2.robot_env import RobotEnv
from r2d2.user_interface.data_collector import DataCollecter
from r2d2.user_interface.gui import RobotGUI

cwd = os.getcwd()
sys.path.append(cwd)

from devices import SpaceMouse, Keyboard
from interfaces import HITLPolicy, HumanInterventionReader

FORCE_MODEL_PATH = "/home/pi0/multi-modal/droid-multi-modal/FORTE/force_est_ckpts/034__sizes-256x256__do-0p3__ido-0p05__wd-0p0001__lr-0p0005__ns-0p01__norm-none__huber-1"
FORCE_DEVICE = "cuda:0"
FORCE_ESTIMATION_HZ = 100
FORCE_HISTORY_LENGTH = 500


class ForceHistoryReader:
    """Expose a fixed history of samples from a ForceRingBuffer."""

    def __init__(self, force_buffer, history_length=FORCE_HISTORY_LENGTH):
        self.force_buffer = force_buffer
        self.history_length = history_length

    def update_values(self):
        pass

    def read_values(self):
        values = self.force_buffer.get_latest(k=self.history_length)
        if values is None or values.size == 0:
            return np.zeros((self.history_length, 1))
        if values.shape[0] < self.history_length:
            pad_len = self.history_length - values.shape[0]
            values = np.concatenate((np.zeros(pad_len), values))
        return values.reshape(self.history_length, 1)

    def close(self):
        pass


class babyFORTEReader:
    def __init__(self, cfg, shutdown_event, force_buffer=None):
        self.shared_sensor_buffer = SharedRingBuffer(cfg.buffer.size, cfg.buffer.num_channels, 'd')
        self.shared_force_buffer = force_buffer or ForceRingBuffer()
        self._owns_force_buffer = force_buffer is None
        self.MLP_inf = MLPInference(FORCE_MODEL_PATH, device=FORCE_DEVICE)
        # time.sleep(2)
        self.sensor_process = Process(
            target=sensor_data_updater,
            args=(cfg.elvrgripper, self.shared_sensor_buffer),
            kwargs={'mode': 'process', 'shutdown_event': shutdown_event},
        )
        self.sensor_process.start()

        self.force_estimator_process = Process(
            target=force_estimator_update_loop_restartsafe,
            args=(
                self.MLP_inf,
                self.shared_sensor_buffer,
                self.shared_force_buffer,
                shutdown_event,
            ),
            kwargs={'hz': FORCE_ESTIMATION_HZ, 'start_delay': 5.0},
        )
        self.force_estimator_process.start()

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

    def close(self):
        print("Shutting down processes...")
        self.shutdown_event.set()

        for proc in [self.visualizer_process, self.sensor_process, self.force_estimator_process]:
            proc.terminate()

        time.sleep(2)
        for proc in [self.visualizer_process, self.sensor_process, self.force_estimator_process]:
            proc.kill()
            proc.join(timeout=1)

        self.shared_sensor_buffer.close()
        if self._owns_force_buffer:
            self.shared_force_buffer.close()
        print("Demo completed. Resources cleaned up.")
    
    def reset(self):
        """
        Fully reset the reader:
        - Kill all running processes
        - Recreate shared buffers
        - Restart processes exactly as __init__ does
        """
        print("Resetting babyFORTEReader...")

        # 1. Stop running processes
        self.shutdown_event.set()
        for proc in [self.visualizer_process, self.sensor_process, self.force_estimator_process]:
            if proc.is_alive():
                proc.terminate()

        time.sleep(2)

        for proc in [self.visualizer_process, self.sensor_process, self.force_estimator_process]:
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=1)

        # 2. Close and recreate shared buffers
        self.shared_sensor_buffer.reset()
        self.shared_force_buffer.reset()

        # Clear shutdown event for restart
        self.shutdown_event.clear()

        # 4. Restart processes (same as __init__)
        time.sleep(2)
        self.sensor_process = Process(
            target=sensor_data_updater,
            args=(self.cfg.elvrgripper, self.shared_sensor_buffer),
            kwargs={'mode': 'process', 'shutdown_event': self.shutdown_event},
        )
        self.sensor_process.start()

        self.force_estimator_process = Process(
            target=force_estimator_update_loop_restartsafe,
            args=(  
                self.MLP_inf,
                self.shared_sensor_buffer,
                self.shared_force_buffer,
                self.shutdown_event,
            ),
            kwargs={'hz': FORCE_ESTIMATION_HZ, 'start_delay': 5.0},
        )
        self.force_estimator_process.start()

        time.sleep(1)

        self.visualizer_process = Process(
            target=opencv_visualizer,
            args=(
                self.shared_sensor_buffer, 
                self.cfg.buffer,
                self.shared_force_buffer,
                None,
                self.shutdown_event
            )
        )
        self.visualizer_process.start()

        print("babyFORTEReader reset complete.")



@hydra.main(config_path="../FORTE/config/", config_name="baby_FORTE")
def main(cfg: DictConfig):
    devices = {"spacemouse": SpaceMouse(reset_with_idle=False), "keyboard": Keyboard()}

    shutdown_event = Event()

    def handle_exit(signum, frame):
        print(f"Signal {signum} received. Exiting...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    force_buffer = ForceRingBuffer()

    try:
        # env = RobotEnv(
        #     action_space="joint_velocity",
        #     sensor_readers={
        #     },
        
        tactile_reader = babyFORTEReader(cfg, shutdown_event, force_buffer)
        force_reader = ForceHistoryReader(force_buffer)

        env = RobotEnv(
            action_space="joint_velocity",
            sensor_readers={
                "tactile_values": tactile_reader,
                "force_prediction": force_reader,
                "human_intervention": HumanInterventionReader(**devices),
            },
            experiment_name="tactile_module_collection"
        )
        # Get gripper position safely (ServerInterface doesn't have get_gripper_position)
        try:
            if hasattr(env._robot, 'get_gripper_position'):
                gripper_pos = env._robot.get_gripper_position()
                print(f"Gripper value: {gripper_pos}")
            elif hasattr(env._robot, 'get_gripper_state'):
                # ServerInterface has get_gripper_state instead
                gripper_state = env._robot.get_gripper_state()
                print(f"Gripper state: {gripper_state}")
            else:
                # Get from observation instead
                obs = env.get_observation()
                gripper_pos = obs.get("robot_state", {}).get("gripper_position", "N/A")
                print(f"Gripper value (from observation): {gripper_pos}")
        except Exception as e:
            print(f"Could not get gripper position: {e}")
        controller = HITLPolicy(devices, policy=None, robot_env=env)

        data_collector = DataCollecter(env=env, controller=controller)
        RobotGUI(robot=data_collector)

    finally:
        print("Cleaning up resources...")
        tactile_reader.close()
        force_reader.close()
        force_buffer.close()
        print("Resources cleaned up.")


if __name__ == "__main__":
    main()