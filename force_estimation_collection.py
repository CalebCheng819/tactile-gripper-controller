import hydra
from omegaconf import DictConfig
from FORTE.sensing.sensor import sensor_data_updater
from FORTE.scripts.sys_utils import SharedRingBuffer, opencv_visualizer
from multiprocessing import Process, Event
from multiprocessing import Process, Lock

from r2d2.controllers.oculus_controller import VRPolicy
from r2d2.robot_env import RobotEnv
from r2d2.user_interface.data_collector import DataCollecter
from r2d2.user_interface.gui import RobotGUI
from r2d2.misc.subprocess_utils import run_threaded_command
import pinocchio as pin

from openpi_client import image_tools
from openpi_client import websocket_client_policy

import os
import sys

import numpy as np
import copy
import time

cwd = os.getcwd()
sys.path.append(cwd)

from util import geom
from util.openpi import (OpenPIConfigs, DROID_CONTROL_FREQUENCY, prevent_keyboard_interrupt, extract_observation)
from devices import SpaceMouse, Keyboard
import signal


def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X


class HIDReader:
    
    def __init__(self, spacemouse=None, keyboard=None, **kwargs) -> None:
        self._spacemouse = spacemouse
        self._keyboard = keyboard

        if self._spacemouse is not None:
            self._spacemouse.start()
        if self._keyboard is not None:
            self._keyboard.start()

        self._hid_values = {
            "poses": {"r": np.eye(4)},
            "buttons": {"A": False, "B": False, "X": False, "Y": False, "RG": False, "RJ": False, "rightTrig": [0.0]},
        }

    def get_transformations_and_buttons(self):

        if self._spacemouse is not None:
            inp = self._spacemouse.control
            rot_mat = np.eye(4)
            rot_mat[:3, :3] = geom.euler_to_rot(np.array([inp[4], inp[3], inp[5]]))
            rot_mat[:3, 3]= np.array([inp[0], inp[1], inp[2]])
            self._hid_values["poses"]["r"] = rot_mat
            self._hid_values["buttons"].update({key: [value] for key, value in self._keyboard.scalars.items()})
        if self._keyboard is not None:
            self._hid_values["buttons"].update(self._keyboard.buttons)

        # print("Current HID Values:", self._hid_values)
        return copy.copy(self._hid_values["poses"]), copy.copy(self._hid_values["buttons"])

    @property
    def idle(self):
        if self._spacemouse is None:
            return True
        else:
            return self._spacemouse.idle
    
    def reset(self):
        if self._spacemouse is not None:
            self._spacemouse.reset()
        if self._keyboard is not None:
            self._keyboard.reset()

    def foce_set_grasping(self, grasping):
        self._keyboard._scalars["rightTrig"] = grasping

class SimpleGripperPolicy:
    """
    Returns zero joint/cartesian velocities and only drives the gripper.
    It toggles between open(1.0) and close(0.0) every `period_s` seconds.
    """
    def __init__(self, period_s: float = 10.0):
        self.period_s = float(period_s)
        self._t0 = time.time()
        self._instruction = "No instruction provided."

    def reset(self):
        self._t0 = time.time()

    def set_instruction(self, instruction: str):
        self._instruction = instruction

    @property
    def instruction(self):
        return self._instruction

    def forward(self, obs_dict, include_info=False):
        # Toggle open/close every period
        elapsed = time.time() - self._t0
        phase = int(elapsed // self.period_s) % 2
        gripper = .1 if phase == 0 else -0.1  # 1.0=open, 0.0=close this should be velocity control

        # Build action matching env.action_space lengths:
        # joint_velocity -> 8 (7 joints + gripper)
        # cartesian_velocity -> 7 (vx,vy,vz,wx,wy,wz + gripper)
        # (Your controller expects 8 for joint, 7 for cartesian.)
        if "robot_state" in obs_dict and "action_space" in getattr(self, "__dict__", {}):
            pass  # not used, HITLPolicy handles branching
        # We will return an 8D by default; HITLPolicy will handle mapping when needed.
        action = np.zeros(8, dtype=float)
        action[-1] = gripper

        print("SimpleGripperPolicy action:", action)

        if include_info:
            return action, {"elapsed_s": elapsed, "gripper": gripper}
        else:
            return action


class HITLPolicy(VRPolicy):
    def __init__(
        self,
        devices,
        policy=None,
        robot_env=None,
        right_controller: bool = True,
        max_lin_vel: float = 1,
        max_rot_vel: float = 1,
        max_gripper_vel: float = 1,
        spatial_coeff: float = 1,
        pos_action_gain: float = 5,
        rot_action_gain: float = 2,
        gripper_action_gain: float = 3,
        rmat_reorder: list = [-2, -1, -3, 4],
        **kwargs
    ):
        self.oculus_reader = HIDReader(**devices)
        self.policy = policy
        self.robot_env = robot_env
        self.vr_to_global_mat = np.eye(4)
        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.max_gripper_vel = max_gripper_vel
        self.spatial_coeff = spatial_coeff
        self.pos_action_gain = pos_action_gain
        self.rot_action_gain = rot_action_gain
        self.gripper_action_gain = gripper_action_gain
        self.global_to_env_mat = vec_to_reorder_mat(rmat_reorder)
        self.controller_id = "r" if right_controller else "l"
        self.reset_orientation = True
        self._instruction = "No instruction provided."
        self.reset_state()

        # Start State Listening Thread #
        run_threaded_command(self._update_internal_state)

        # self._model = pin.buildModelFromUrdf("/home/soroush/code/droid_mingyo/model/panda.urdf", pin.JointModelFreeFlyer())
        self._model = pin.buildModelFromUrdf("/home/pi0/multi-modal/droid-multi-modal/model/panda.urdf", pin.JointModelFreeFlyer())
        self._data = self._model.createData()
        self._link_idx_hand = self._model.getFrameId("panda_link8")
        self._previous_action = None

    def forward(self, obs_dict, include_info=False):

        assert self.robot_env.action_space in ["cartesian_velocity", "joint_velocity"]

        if self.oculus_reader.idle:
            self.reset_origin = True
            if self.policy is None:
                out = self.get_dummy_action(obs_dict, include_info=include_info)
            else:
                print(obs_dict.keys())
                out = self.policy.forward(obs_dict, include_info=include_info)
            grasping = out[-1] if not include_info else out[0][-1]
            self.oculus_reader.foce_set_grasping(grasping)
            return out
        else:
            if self.robot_env.action_space == "cartesian_velocity":
                return super().forward(obs_dict, include_info=include_info)
            else:
                out = super().forward(obs_dict, include_info=include_info)
                robot_state = obs_dict["robot_state"]
                if include_info:
                    cartesian_action, info = out
                    joint_action = self.cartesian_velocity_to_joint_velocity(
                        cartesian_action, robot_state
                    ).tolist()
                    joint_action = joint_action + [cartesian_action[-1]]
                    return np.clip(joint_action, -1, 1), info
                else:
                    cartesian_action = out
                    joint_action = self.cartesian_velocity_to_joint_velocity(
                        cartesian_action, robot_state
                    ).tolist()
                    joint_action = joint_action + [cartesian_action[-1]]
                    return np.clip(joint_action, -1, 1)

    def cartesian_velocity_to_joint_velocity(self, cartesian_action, robot_state):
        cur_q = np.array([0] * 7 + robot_state["joint_positions"])
        pin.forwardKinematics(self._model, self._data, cur_q)
        J_hand = pin.computeFrameJacobian(self._model, self._data, cur_q, self._link_idx_hand, pin.LOCAL_WORLD_ALIGNED)[:,6:]
        local_hand_vel = np.array(cartesian_action[:-1])
        joint_action = np.linalg.pinv(J_hand) @ local_hand_vel
        return joint_action  # Exclude gripper action

    def get_dummy_action(self, obs_dict, include_info=False):
        print(self.robot_env.action_space)
        if self.robot_env.action_space == "cartesian_velocity":
            dummy_action = np.zeros(7)
        else:
            dummy_action = np.zeros(8)
        if include_info:
            return dummy_action, {}
        else:
            return dummy_action

    def reset_state(self):
        super().reset_state()
        self.oculus_reader.reset()
        if self.policy is not None:
            self.policy.reset()
            self.policy.set_instruction(self.instruction)

    def set_instruction(self, instruction):
        self._instruction = instruction


    @property
    def instruction(self):
        return self._instruction

class LoadCellReader:
    def __init__(self, cfg, shutdown_event) -> None:

        # Shared memory configuration
        # BUFFER_SIZE = 50000  # Number of samples in the ring buffer
        BUFFER_SIZE = cfg.loadcell_buffer.size  # Number of samples in the ring buffer
        NUM_CHANNELS = cfg.loadcell_buffer.num_channels    # 6 sensor channels

        self.shared_lc_buffer = SharedRingBuffer(BUFFER_SIZE, NUM_CHANNELS, 'd')

        self.lc_process = Process(
            target=sensor_data_updater,
            args=(cfg.loadcell, self.shared_lc_buffer),
            kwargs={'mode': 'process', 'sensor_type': 'loadcell', 'shutdown_event': shutdown_event},
        )
        self.lc_process.start()
        # time.sleep(2)  # Ensure the sensor process starts before reading

        self.visualizer_process = Process(
            target=opencv_visualizer,
            args=(self.shared_lc_buffer, cfg.loadcell_buffer, None, None, shutdown_event, 'loadcell')
        )
        self.visualizer_process.start()

        self.shutdown_event = shutdown_event
        self._test_cnt = 1

    def update_values(self):
        self._test_cnt += 1

    def read_values(self):
        if self.shared_sensor_buffer.is_empty():
            return np.zeros((self.shared_lc_buffer.num_channels,))

        # Read the latest sensor values from the shared buffer
        # sensor_values = self.shared_sensor_buffer.get_latest()


        # Read the sensor data of the last 5 seconds
        sensor_values = self.shared_lc_buffer.get_latest_freq_history()

        # print(sensor_values.shape)
        # sensor_values = None
        return sensor_values

    def close(self):
        print("Shutting down processes...")
        self.shutdown_event.set()
         
        self.visualizer_process.terminate()
        self.visualizer_process.join()
        self.lc_process.terminate()
        # force_estimator_process.terminate()
        # slip_predictor_process.terminate()
        # gripper_process.terminate()

        time.sleep(2)
        print("Cleaning up resources...")
        self.lc_process.kill()
        # force_estimator_process.kill()
        # slip_predictor_process.kill()
        # gripper_process.kill()
        self.lc_process.join(timeout=1)
        # force_estimator_process.join(timeout=1)
        # slip_predictor_process.join(timeout=1)
        # gripper_process.join(timeout=1)
        self.shared_lc_buffer.close()
        # force_buffer.close()
        # slip_buffer.close()
        print("Demo completed. Resources cleaned up.")

class babyFORTEReader:
    def __init__(self, cfg, shutdown_event) -> None:

        # Shared memory configuration
        BUFFER_SIZE = cfg.tactile_buffer.size  # Number of samples in the ring buffer
        NUM_CHANNELS = cfg.tactile_buffer.num_channels    # 32 sensor channels

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
            args=(self.shared_sensor_buffer, cfg.tactile_buffer, None, None, shutdown_event, 'elvrgripper')
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


class HumanInterventionReader:
    def __init__(self, spacemouse=None, keyboard=None):
        self._spacemouse = spacemouse
        self._keyboard = keyboard
        self._idle = self._spacemouse.idle

    def update_values(self):
        self._idle = self._spacemouse.idle

    def read_values(self):
        self.update_values()
        return self._idle


# class OpenPIWrapper:
#     def __init__(self, config: OpenPIConfigs):
#         self.config = config
#         self.policy_client = websocket_client_policy.WebsocketClientPolicy(config.remote_host, config.remote_port)

#     def reset(self):
#         self._actions_from_chunk_completed = 0
#         self._pred_action_chunk = None
#         self._instruction = "No instruction provided."
#         self._start_time = time.time()


#     def set_instruction(self, instruction):
#         self._instruction = instruction

#     @property
#     def instruction(self):
#         return self._instruction

#     def forward(self, obs_dict, include_info=False):
#         start_time = time.time()

#         try:
#             # Get the current observation
#             robot_states = obs_dict["robot_state"]
#             curr_obs = extract_observation(self.config, obs_dict)

#             # Send websocket request to policy server if it's time to predict a new chunk
#             if self._actions_from_chunk_completed == 0 or self._actions_from_chunk_completed >= self.config.open_loop_horizon:
#                 self._actions_from_chunk_completed = 0

#                 # We resize images on the robot laptop to minimize the amount of data sent to the policy server
#                 # and improve latency.
#                 request_data = {
#                     "observation/exterior_image_1_left": image_tools.resize_with_pad(
#                         curr_obs[f"{self.config.external_camera}_image"], 224, 224
#                     ),
#                     "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
#                     "observation/joint_position": curr_obs["joint_position"],
#                     "observation/gripper_position": curr_obs["gripper_position"],
#                     "observation/tactile_values": np.array(robot_states["tactile_values"]).flatten(),
#                     "prompt": self.instruction,
#                 }
#                 # for key, value in request_data.items():
#                 #     if key != "prompt":
#                 #         print(key, ": ", value.shape)
#                 #     else:
#                 #         print(key, ": ", value)
#                 # print("#############################################")

#                 # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
#                 self._pred_action_chunk = self.policy_client.infer(request_data)["actions"]
#                 # print(self._pred_action_chunk.shape)
#                 # assert self._pred_action_chunk.shape == (10, 8)

#             # Select current action to execute from chunk
#             action = self._pred_action_chunk[self._actions_from_chunk_completed]
#             self._actions_from_chunk_completed += 1

#             print("grasping action", action[-1])
#             # Binarize gripper action
#             # if action[-1].item() > 0.5:
#             #     action = np.concatenate([action[:-1], np.ones((1,))])
#             # else:
#             #     action = np.concatenate([action[:-1], np.zeros((1,))])

#             action = np.clip(action, -1, 1)

#         except KeyboardInterrupt:
#             action = np.zeros(8)
    
#         if include_info:
#             info = {
#                 "action_chunk": self._pred_action_chunk,
#                 "actions_from_chunk_completed": self._actions_from_chunk_completed,
#                 "time": time.time() - start_time,
#             }
#             return action, info
#         else:
#             return action

class PromptReader:
    def __init__(self, controller=None):
        self._controller = controller

    def update_values(self):
        self._prompts = copy.copy(self._controller.instruction)

    def read_values(self):
        self.update_values()
        return self._prompts


@hydra.main(config_path="../FORTE/config/", config_name="force_estimation")
def main(cfg: DictConfig):

    devices = {"spacemouse": SpaceMouse(reset_with_idle=True), "keyboard": Keyboard()}
    # devices = {"keyboard": Keyboard()}
    # Super-simple gripper-only policy: toggles open/close every 2 seconds
    # openpi_policy = SimpleGripperPolicy(period_s=10.0)

    tactile_shutdown_event = Event()
    loadcell_shutdown_event = Event()

    def handle_exit(signum, frame):
        print(f"Signal {signum} received. Exiting...")
        tactile_shutdown_event.set()
        loadcell_shutdown_event.set()

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:        

        tactile_reader = babyFORTEReader(cfg, tactile_shutdown_event)
        loadcell_reader = LoadCellReader(cfg, loadcell_shutdown_event)
        devices = {"spacemouse": SpaceMouse(reset_with_idle=False), "keyboard": Keyboard()}

        # env = RobotEnv(action_space="joint_velocity", sensor_readers={"tactile_values":DummyTactileReader(), "human_intervention": HumanInterventionReader(**devices)})
        # env = RobotEnv(action_space="joint_velocity", sensor_readers={"tactile_values":tactile_reader, "human_intervention": HumanInterventionReader(**devices)})
        env = RobotEnv(action_space="joint_velocity",  sensor_readers={"tactile_values":tactile_reader, "loadcell_values":loadcell_reader, "human_intervention": HumanInterventionReader(**devices)})
        controller = HITLPolicy(devices, policy=None, robot_env=env)

        # Make the data collector
        data_collector = DataCollecter(env=env, controller=controller)

        # Make the GUI
        user_interface = RobotGUI(robot=data_collector)
    
    finally:
        print("Cleaning up resources...")
        tactile_reader.close()
        loadcell_reader.close()
        print("Resources cleaned up.")

if __name__ == "__main__":
    main()  