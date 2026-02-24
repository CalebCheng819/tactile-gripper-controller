import os
import sys
import time
import hydra
import signal
import numpy as np

from omegaconf import DictConfig
from FORTE.sensing.sensor import sensor_data_updater
from FORTE.scripts.sys_utils import SharedRingBuffer, opencv_visualizer, ForceRingBuffer
from FORTE.scripts.force_est_nn import force_estimator_update_loop
from multiprocessing import Process, Event, Lock

from r2d2.robot_env import RobotEnv
from r2d2.user_interface.data_collector import DataCollecter
from r2d2.user_interface.gui import RobotGUI

cwd = os.getcwd()
sys.path.append(cwd)

from devices import SpaceMouse, Keyboard
from util.openpi import OpenPIConfigs
from interfaces import HITLPolicy, OpenPIWrapper, HumanInterventionReader

class babyFORTEReader:
    def __init__(self, cfg, shutdown_event) -> None:

        self.shared_sensor_buffer = SharedRingBuffer(cfg.buffer.size, cfg.buffer.num_channels, 'd')
        self.shared_force_buffer = ForceRingBuffer()

        self.sensor_process = Process(
            target=sensor_data_updater,
            args=(cfg.elvrgripper, self.shared_sensor_buffer),
            kwargs={'mode': 'process', 'shutdown_event': shutdown_event},
        )
        self.sensor_process.start()

        # Path and device used by the estimator subprocess
        run_dir_ckpt = "/home/pi0/multi-modal/droid-multi-modal/FORTE/force_est_ckpts/034__sizes-256x256__do-0p3__ido-0p05__wd-0p0001__lr-0p0005__ns-0p01__norm-none__huber-1"
        device_str = "cuda:0"

        # -------------------- UPDATED: start force estimator process --------------------
        self.force_estimator_process = Process(
            target=force_estimator_update_loop,
            args=(run_dir_ckpt, device_str, self.shared_sensor_buffer, self.shared_force_buffer, shutdown_event),
            kwargs={'hz': 100, 'start_delay': 5.0},   # <-- 5s pause before starting
        )
        self.force_estimator_process.start()
        # ------------------------------------------------------------------------------



        # self.sensor_process = Process(
        #     target=sensor_data_updater,
        #     args=(cfg.elvrgripper, self.shared_sensor_buffer),
        #     kwargs={'mode': 'process', 'shutdown_event': shutdown_event},
        # )
        # self.sensor_process.start()
        time.sleep(2)  # Ensure the sensor process starts before reading

        self.visualizer_process = Process(
            target=opencv_visualizer,
            args=(self.shared_sensor_buffer, cfg.buffer, self.shared_force_buffer, None, shutdown_event)
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
    
    def read_force(self):
        if self.shared_force_buffer.is_empty():
            return 0
        
        force_est = self.shared_force_buffer.get_data()[-1]

        return force_est

    def close(self):
        print("Shutting down processes...")
        self.shutdown_event.set()
         
        self.visualizer_process.terminate()
        self.visualizer_process.join()
        self.sensor_process.terminate()
        self.force_estimator_process.terminate()
        # force_estimator_process.terminate()
        # slip_predictor_process.terminate()
        # gripper_process.terminate()

        time.sleep(2)
        print("Cleaning up resources...")
        self.sensor_process.kill()
        self.force_estimator_process.kill()
        # slip_predictor_process.kill()
        # gripper_process.kill()
        self.sensor_process.join(timeout=1)
        self.force_estimator_process.join(timeout=1)
        # slip_predictor_process.join(timeout=1)
        # gripper_process.join(timeout=1)
        self.shared_sensor_buffer.close()
        self.shared_force_buffer.close()
        # slip_buffer.close()
        print("Demo completed. Resources cleaned up.")



@hydra.main(config_path="../FORTE/config/", config_name="baby_FORTE")
def main(cfg: DictConfig):

    devices = {"spacemouse": SpaceMouse(reset_with_idle=False), "keyboard": Keyboard()}
    openpi_config = OpenPIConfigs()

    openpi_config.remote_port = 8000  # Replace with your policy server port
    # Cameras in GDC
    openpi_config.left_camera_id = "24395123"  # Replace with your left camera ID
    openpi_config.right_camera_id = "24013089"  # Replace with your right camera ID
    openpi_config.wrist_camera_id = "17225336"  # Replace with your wrist camera ID

    # Original, Jaelyn changed 
    openpi_config.remote_host = "127.0.0.1"  # Replace with your policy server host
    openpi_policy = OpenPIWrapper(openpi_config)

    shutdown_event = Event()
    def handle_exit(signum, frame):
        print(f"Signal {signum} received. Exiting...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:        

        tactile_reader = babyFORTEReader(cfg, shutdown_event)
        # env = RobotEnv(action_space="joint_velocity", sensor_readers={"tactile_values":DummyTactileReader(), "human_intervention": HumanInterventionReader(**devices)})
        env = RobotEnv(action_space="joint_velocity", sensor_readers={"tactile_values":tactile_reader, "human_intervention": HumanInterventionReader(**devices)})
        controller = HITLPolicy(devices, policy=openpi_policy, robot_env=env)

        # Make the data collector
        data_collector = DataCollecter(env=env, controller=controller)

        # Make the GUI
        user_interface = RobotGUI(robot=data_collector)
    
    finally:
        print("Cleaning up resources...")
        tactile_reader.close()
        print("Resources cleaned up.")

if __name__ == "__main__":
    main()  
