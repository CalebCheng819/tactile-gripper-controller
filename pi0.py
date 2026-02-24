# ruff: noqa


import hydra
from omegaconf import DictConfig
from FORTE.sensing.sensor import sensor_data_updater
from FORTE.scripts.sys_utils import SharedRingBuffer, opencv_visualizer
from multiprocessing import Process, Event
from multiprocessing import Process, Lock

################################33

import contextlib
import dataclasses
import datetime
import faulthandler
import os, sys
import signal
import time
# from moviepy.editor import ImageSeImageSequenceClipquenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from r2d2.robot_env import RobotEnv
import tqdm
import tyro

from moviepy.editor import ImageSequenceClip


cwd = os.getcwd()
sys.path.append(cwd)

from util import geom
from util.openpi import OpenPIConfigs, extract_observation, DROID_CONTROL_FREQUENCY
from devices import SpaceMouse, Keyboard

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

    openpi_config = OpenPIConfigs()
    openpi_config.remote_port = 8000  # Replace with your policy server port
    # Cameras in GDC
    openpi_config.left_camera_id = "24395123"  # Replace with your left camera ID
    openpi_config.right_camera_id = "24013089"  # Replace with your right camera ID
    openpi_config.wrist_camera_id = "17225336"  # Replace with your wrist camera ID
    openpi_config.max_timesteps=600

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")

    shutdown_event = Event()
    tactile_reader = babyFORTEReader(cfg, shutdown_event)


    # ### To-do 1 
    # inf = READ_INFERENCE_MODEL_FOR_FORCE_ESTIMATION
    # shared_force_buffer = ForceRingBuffer()
    # ###

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(openpi_config.remote_host, openpi_config.remote_port)
    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    while True:
        instruction = input("Enter instruction: ")

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        bar = tqdm.tqdm(range(openpi_config.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")

        # ## To-do 2 
        # pi_model_input_buffer_for_force_history = np.zeros(10)
        ## 

        for t_step in bar:
            start_time = time.time()
            try:

                tactile_buffer = tactile_reader.read_values()
                ## To-do 3 
                # function from force_estr_nn.py
                # est_force(inf, tactile_buffer, shared_force_buffer) # --> gives force_buffer
                # current_force = READ_VALUE_FROM shared_force_buffer # --> force buffer gives current force 
                # pi_model_input_buffer_for_force_history = np.concatenate([current_force], pi_model_input_buffer_for_force_history[:-1]) # --> remove the tail
                # ##

                # Get the current observation
                curr_obs = extract_observation(
                    openpi_config,
                    env.get_observation(),
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )

                video.append(curr_obs[f"{openpi_config.external_camera}_image"])

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= openpi_config.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs[f"{openpi_config.external_camera}_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                        "observation/tactile_values": np.array(tactile_reader.read_values()).flatten(),
                        ## To-do 4 
                        # "observation/force_prediction": np.array(pi_model_input_buffer_for_force_history).flatten(),

                    }

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    assert pred_action_chunk.shape == (10, 8)

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # # Binarize gripper action
                # if action[-1].item() > 0.5:
                #     # action[-1] = 1.0
                #     action = np.concatenate([action[:-1], np.ones((1,))])
                # else:
                #     # action[-1] = 0.0
                #     action = np.concatenate([action[:-1], np.zeros((1,))])

                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)

                env.step(action)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        video = np.stack(video)
        save_filename = "video_" + timestamp
        ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")

        df = df.append(
            {
                "success": success,
                "duration": t_step,
                "video_filename": save_filename,
            },
            ignore_index=True,
        )

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


if __name__ == "__main__":
    main()
