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

        self._model = pin.buildModelFromUrdf("/home/soroush/code/droid_mingyo/model/panda.urdf", pin.JointModelFreeFlyer())
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
    

class DummyTactileReader:
    def __init__(self):
        self._dummy_data = np.zeros(6)
        self._test_cnt = 1

    def update_values(self):
        self._test_cnt += 1

    def read_values(self):
        self.update_values()
        self._dummy_data += 0.001* self._test_cnt * np.ones(6)
        return self._dummy_data


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


class OpenPIWrapper:
    def __init__(self, config: OpenPIConfigs):
        self.config = config
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(config.remote_host, config.remote_port)

    def reset(self):
        self._actions_from_chunk_completed = 0
        self._pred_action_chunk = None
        self._instruction = "No instruction provided."
        self._start_time = time.time()


    def set_instruction(self, instruction):
        self._instruction = instruction

    @property
    def instruction(self):
        return self._instruction

    def forward(self, obs_dict, include_info=False):
        start_time = time.time()

        try:
            # Get the current observation
            curr_obs = extract_observation(self.config, obs_dict)

            # Send websocket request to policy server if it's time to predict a new chunk
            if self._actions_from_chunk_completed == 0 or self._actions_from_chunk_completed >= self.config.open_loop_horizon:
                self._actions_from_chunk_completed = 0

                # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                # and improve latency.
                request_data = {
                    "observation/exterior_image_1_left": image_tools.resize_with_pad(
                        curr_obs[f"{self.config.external_camera}_image"], 224, 224
                    ),
                    "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                    "observation/joint_position": curr_obs["joint_position"],
                    "observation/gripper_position": curr_obs["gripper_position"],
                    "prompt": self.instruction,
                }

                # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                self._pred_action_chunk = self.policy_client.infer(request_data)["actions"]
                assert self._pred_action_chunk.shape == (10, 8)

            # Select current action to execute from chunk
            action = self._pred_action_chunk[self._actions_from_chunk_completed]
            self._actions_from_chunk_completed += 1

            # Binarize gripper action
            if action[-1].item() > 0.5:
                action = np.concatenate([action[:-1], np.ones((1,))])
            else:
                action = np.concatenate([action[:-1], np.zeros((1,))])

            action = np.clip(action, -1, 1)

        except KeyboardInterrupt:
            action = np.zeros(8)
    
        if include_info:
            info = {
                "action_chunk": self._pred_action_chunk,
                "actions_from_chunk_completed": self._actions_from_chunk_completed,
                "time": time.time() - start_time,
            }
            return action, info
        else:
            return action

def main():

    devices = {"spacemouse": SpaceMouse(reset_with_idle=True), "keyboard": Keyboard()}
    openpi_config = OpenPIConfigs()
    openpi_config.left_camera_id = "25047636"  # Replace with your left camera ID
    openpi_config.right_camera_id = "24013089"  # Replace with your right camera ID
    openpi_config.wrist_camera_id = "17225336"  # Replace with your wrist camera ID
    openpi_config.remote_host = "172.16.0.111"  # Replace with your policy server host
    openpi_config.remote_port = 8000  # Replace with your policy server port

    openpi_policy = OpenPIWrapper(openpi_config)

    env = RobotEnv(action_space="joint_velocity", sensor_readers={"tactile_values":DummyTactileReader(), "human_intervention": HumanInterventionReader(**devices)})
    # env = RobotEnv(action_space="joint_velocity", sensor_readers={"tactile_values":DummyTactileReader()})
    controller = HITLPolicy(devices, policy=openpi_policy, robot_env=env)

    # Make the data collector
    data_collector = DataCollecter(env=env, controller=controller)

    # Make the GUI
    user_interface = RobotGUI(robot=data_collector)

if __name__ == "__main__":
    main()  