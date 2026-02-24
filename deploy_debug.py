# ruff: noqa
import hydra
from omegaconf import DictConfig
from FORTE.sensing.sensor import sensor_data_updater
from FORTE.scripts.sys_utils import SharedRingBuffer, opencv_visualizer
from multiprocessing import Process, Event, Lock

################################33

import contextlib
import faulthandler
import os, sys
import signal
import time
import numpy as np
from r2d2.robot_env import RobotEnv
import tqdm

cwd = os.getcwd()
sys.path.append(cwd)

from util import geom
from devices import SpaceMouse, Keyboard
from util.openpi import OpenPIConfigs
from interfaces import HITLPolicy, OpenPIWrapper, HumanInterventionReader

faulthandler.enable()

# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
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

############################

class babyFORTEReader:
    def __init__(self, cfg, shutdown_event) -> None:

        # Shared memory configuration
        BUFFER_SIZE = 50000  # Number of samples in the ring buffer
        NUM_CHANNELS = 6    # 6 sensor channels

        self.shared_sensor_buffer = SharedRingBuffer(BUFFER_SIZE, NUM_CHANNELS, 'd')

        self.sensor_process = Process(
            target=sensor_data_updater,
            args=(cfg.elvrgripper, self.shared_sensor_buffer),
            kwargs={'mode': 'process', 'shutdown_event': shutdown_event},
        )
        self.sensor_process.start()
        time.sleep(2)  # Ensure the sensor process starts before reading

        self.visualizer_process = Process(
            target=opencv_visualizer,
            args=(self.shared_sensor_buffer, cfg.buffer, None, None, shutdown_event)
        )
        self.visualizer_process.start()

        self.shutdown_event = shutdown_event
        self._test_cnt = 1

    def update_values(self):
        self._test_cnt += 1

    def read_values(self):
        if self.shared_sensor_buffer.is_empty():
            return np.zeros((self.shared_sensor_buffer.num_channels,))

        # Read the latest sensor values from the shared buffer
        # sensor_values = self.shared_sensor_buffer.get_latest()


        # Read the sensor data of the last 5 seconds
        sensor_values = self.shared_sensor_buffer.get_latest_freq_history()

        # print(sensor_values.shape)
        # sensor_values = None
        return sensor_values

    def close(self):
        print("Shutting down processes...")
        self.shutdown_event.set()
         
        self.visualizer_process.terminate()
        self.visualizer_process.join()
        self.sensor_process.terminate()
        # force_estimator_process.terminate()
        # slip_predictor_process.terminate()
        # gripper_process.terminate()

        time.sleep(2)
        print("Cleaning up resources...")
        self.sensor_process.kill()
        # force_estimator_process.kill()
        # slip_predictor_process.kill()
        # gripper_process.kill()
        self.sensor_process.join(timeout=1)
        # force_estimator_process.join(timeout=1)
        # slip_predictor_process.join(timeout=1)
        # gripper_process.join(timeout=1)
        self.shared_sensor_buffer.close()
        # force_buffer.close()
        # slip_buffer.close()
        print("Demo completed. Resources cleaned up.")

############################


@hydra.main(config_path="../FORTE/config/", config_name="baby_FORTE")
def main(cfg: DictConfig):

    devices = {"spacemouse": SpaceMouse(reset_with_idle=False), "keyboard": Keyboard()}

    openpi_config = OpenPIConfigs()
    openpi_config.remote_port = 8000  # Replace with your policy server port
    # Cameras in GDC
    openpi_config.left_camera_id = "24395123"  # Replace with your left camera ID
    openpi_config.right_camera_id = "24013089"  # Replace with your right camera ID
    openpi_config.wrist_camera_id = "17225336"  # Replace with your wrist camera ID
    assert (
        openpi_config.external_camera is not None and openpi_config.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {openpi_config.external_camera}"
    # Original, Jaelyn changed 
    openpi_config.remote_host = "127.0.0.1"  # Replace with your policy server host
    openpi_config.max_timesteps = 600
    openpi_policy = OpenPIWrapper(openpi_config)

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    shutdown_event = Event()
    def handle_exit(signum, frame):
        print(f"Signal {signum} received. Exiting...")
        shutdown_event.set()
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    shutdown_event = Event()
    tactile_reader = babyFORTEReader(cfg, shutdown_event)

    env = RobotEnv(action_space="joint_velocity", sensor_readers={"tactile_values":tactile_reader, "human_intervention": HumanInterventionReader(**devices)})
    # env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    controller = HITLPolicy(devices, policy=openpi_policy, robot_env=env)
    print("Created the droid env!")

    while True:
        env.reset(randomize=True)
        controller.reset_state()
        instruction = input("Enter instruction: ")
        # instruction = "grasp the can"
        controller.set_instruction(instruction)
        bar = tqdm.tqdm(range(openpi_config.max_timesteps))
        for t_step in bar:
            try:
                action = controller.forward(env.get_observation(), include_info=False)
                env.step(action)
            except KeyboardInterrupt:
                break
        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
    
    env.close()
    tactile_reader.close()


if __name__ == "__main__":
    main()
