import os
import sys
import copy
import time
import re

import numpy as np
import pinocchio as pin

from openpi_client import image_tools
from openpi_client import websocket_client_policy

from r2d2.controllers.oculus_controller import VRPolicy
from r2d2.misc.subprocess_utils import run_threaded_command

cwd = os.getcwd()
sys.path.append(cwd)

from util import geom
from util.openpi import OpenPIConfigs, extract_observation


PI0_IO_CONTRACT_CHOICES = ("auto", "base", "tactile", "force", "tactile_force")
PI0_IO_DEFAULT_FALLBACK_ORDER = ("base", "tactile", "tactile_force", "force")


class PI0ProtocolError(RuntimeError):
    """Raised when PI0 request/response contract is violated."""


class PI0ContractInputError(PI0ProtocolError):
    """Raised when request data does not satisfy the active I/O contract."""


def _warn_once(obj, key: str, message: str):
    attr = f"_warned_{key}"
    if not hasattr(obj, attr):
        print(message)
        setattr(obj, attr, True)


def _normalize_contract_order(raw_order):
    if isinstance(raw_order, str):
        parts = [p.strip().lower() for p in raw_order.split(",") if p.strip()]
    elif isinstance(raw_order, (tuple, list)):
        parts = [str(p).strip().lower() for p in raw_order if str(p).strip()]
    else:
        parts = []

    if not parts:
        parts = list(PI0_IO_DEFAULT_FALLBACK_ORDER)

    normalized = []
    for contract in parts:
        if contract not in PI0_IO_CONTRACT_CHOICES or contract == "auto":
            continue
        if contract not in normalized:
            normalized.append(contract)

    if not normalized:
        normalized = list(PI0_IO_DEFAULT_FALLBACK_ORDER)
    return tuple(normalized)


def resolve_pi0_contracts(
    io_contract,
    io_fallback_order,
    include_tactile_values=False,
    include_force_prediction=False,
):
    requested = str(io_contract or "auto").strip().lower()
    if requested not in PI0_IO_CONTRACT_CHOICES:
        requested = "auto"

    fallback = list(_normalize_contract_order(io_fallback_order))

    if requested != "auto":
        order = [requested] + [c for c in fallback if c != requested]
        return requested, tuple(order)

    preferred = None
    if include_tactile_values and include_force_prediction:
        preferred = "tactile_force"
    elif include_tactile_values:
        preferred = "tactile"
    elif include_force_prediction:
        preferred = "force"

    if preferred in fallback:
        fallback = [preferred] + [c for c in fallback if c != preferred]
    return fallback[0], tuple(fallback)


def build_pi0_request_data(contract, curr_obs, robot_states, instruction, external_camera):
    contract = str(contract).strip().lower()
    if contract not in PI0_IO_CONTRACT_CHOICES or contract == "auto":
        raise PI0ContractInputError(f"Invalid active I/O contract: {contract}")

    request_data = {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(
            curr_obs[f"{external_camera}_image"], 224, 224
        ),
        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
        "observation/joint_position": curr_obs["joint_position"],
        "observation/gripper_position": curr_obs["gripper_position"],
        "prompt": instruction,
    }

    need_tactile = contract in {"tactile", "tactile_force"}
    need_force = contract in {"force", "tactile_force"}

    has_tactile_values = False
    has_force_prediction = False

    if need_tactile:
        tactile_values = robot_states.get("tactile_values")
        if tactile_values is None:
            raise PI0ContractInputError(
                "I/O contract requires observation/tactile_values but robot_state is missing tactile_values."
            )
        request_data["observation/tactile_values"] = np.array(tactile_values, dtype=np.float32).flatten()
        has_tactile_values = True

    if need_force:
        force_prediction = robot_states.get("force_prediction")
        if force_prediction is None:
            raise PI0ContractInputError(
                "I/O contract requires observation/force_prediction but robot_state is missing force_prediction."
            )
        request_data["observation/force_prediction"] = np.array(force_prediction, dtype=np.float32).flatten()
        has_force_prediction = True

    return request_data, has_tactile_values, has_force_prediction


def normalize_pi0_actions_response(response, action_dim=8):
    if not isinstance(response, dict):
        raise PI0ProtocolError(f"PI0 response must be dict, got {type(response).__name__}.")

    if "actions" not in response:
        raise PI0ProtocolError("PI0 response missing 'actions'.")

    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    elif actions.ndim != 2:
        raise PI0ProtocolError(f"PI0 actions must be 1D/2D, got shape={actions.shape}.")

    horizon, width = actions.shape
    if horizon < 1:
        raise PI0ProtocolError("PI0 actions horizon must be >=1.")
    if width < action_dim:
        raise PI0ProtocolError(
            f"PI0 actions width must be >= {action_dim}, got {width} (shape={actions.shape})."
        )
    if not np.isfinite(actions).all():
        raise PI0ProtocolError("PI0 actions contain NaN/Inf.")

    clipped = actions[:, :action_dim]
    return clipped, tuple(actions.shape), tuple(clipped.shape), sorted(response.keys())


def is_pi0_contract_error(err):
    text = str(err)
    patterns = (
        r"observation/tactile_values",
        r"observation/force_prediction",
        r"KeyError:\s*['\"]observation/",
        r"missing.*observation/",
    )
    return any(re.search(pat, text, flags=re.IGNORECASE) for pat in patterns)


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
            "buttons": {"A": False, "B": False, "X": False, "Y": False, "RG": False, "RJ": False, "rightTrig": [0.0], "Tac_Reset": False},
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

    def force_set_grasping(self, grasping):
        self._keyboard._scalars["rightTrig"] = grasping


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
        self.policy_client = None
        self.server_metadata = {}
        self._warned_action_width_clip = False
        self._warned_horizon_clip = False
        self._legacy_include_tactile = bool(getattr(self.config, "include_tactile_values", False))
        self._legacy_include_force = bool(getattr(self.config, "include_force_prediction", False))
        self._requested_contract = str(getattr(self.config, "io_contract", "auto")).strip().lower()
        self._protocol_max_retries = max(0, int(getattr(self.config, "protocol_max_retries", 2)))
        self._protocol_reconnect_on_error = bool(getattr(self.config, "protocol_reconnect_on_error", True))

        active, order = resolve_pi0_contracts(
            io_contract=self._requested_contract,
            io_fallback_order=getattr(self.config, "io_fallback_order", PI0_IO_DEFAULT_FALLBACK_ORDER),
            include_tactile_values=self._legacy_include_tactile,
            include_force_prediction=self._legacy_include_force,
        )
        self._io_contract_active = active
        self._io_contract_order = order
        self._io_contract_switch_count = 0
        self._io_contract_attempts = 0

        if self._requested_contract != "auto" and (self._legacy_include_tactile or self._legacy_include_force):
            _warn_once(
                self,
                "manual_contract_legacy_switches",
                "[PI0 CONTRACT] PI0_IO_CONTRACT is manual; legacy include switches are ignored for contract selection.",
            )

        self._connect_policy_client(reason="init")
        self.reset()

    def reset(self):
        self._actions_from_chunk_completed = 0
        self._pred_action_chunk = None
        self._instruction = "No instruction provided."
        self._start_time = time.time()
        self._last_request_keys = []
        self._last_has_tactile_values = False
        self._last_has_force_prediction = False
        self._last_prompt_len = 0
        self._last_response_action_shape = None
        self._last_response_keys = []
        self._last_protocol_error = None
        self._last_effective_open_loop_horizon = 0


    def set_instruction(self, instruction):
        self._instruction = instruction

    @property
    def instruction(self):
        return self._instruction

    @property
    def io_contract_active(self):
        return self._io_contract_active

    @property
    def io_contract_order(self):
        return self._io_contract_order

    @property
    def protocol_max_retries(self):
        return self._protocol_max_retries

    def _connect_policy_client(self, reason="reconnect"):
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            self.config.remote_host,
            self.config.remote_port,
        )
        metadata = {}
        try:
            metadata = self.policy_client.get_server_metadata() or {}
        except Exception:
            metadata = {}
        self.server_metadata = metadata
        if reason != "init":
            print(f"[PI0 CONTRACT] Reconnected websocket client ({reason}).")

    def _reconnect_policy_client(self, reason):
        try:
            ws = getattr(self.policy_client, "_ws", None)
            if ws is not None and hasattr(ws, "close"):
                ws.close()
        except Exception:
            pass
        self._connect_policy_client(reason=reason)

    def _contract_candidates(self):
        order = list(self._io_contract_order)
        if not order:
            order = list(PI0_IO_DEFAULT_FALLBACK_ORDER)

        if self._io_contract_active in order:
            idx = order.index(self._io_contract_active)
            order = order[idx:] + order[:idx]

        max_attempts = min(len(order), max(1, 1 + self._protocol_max_retries))
        return order[:max_attempts]

    def _query_action_chunk(self, curr_obs, robot_states):
        last_error = None
        candidates = self._contract_candidates()

        for attempt_idx, contract in enumerate(candidates, start=1):
            self._io_contract_attempts = attempt_idx
            try:
                request_data, has_tactile_values, has_force_prediction = build_pi0_request_data(
                    contract=contract,
                    curr_obs=curr_obs,
                    robot_states=robot_states,
                    instruction=self.instruction,
                    external_camera=self.config.external_camera,
                )
                response = self.policy_client.infer(request_data)
                actions, raw_shape, clipped_shape, response_keys = normalize_pi0_actions_response(response)

                if raw_shape[-1] > clipped_shape[-1]:
                    _warn_once(
                        self,
                        "response_action_dim_clip",
                        f"[PI0 CONTRACT] PI0 returned action width {raw_shape[-1]} > {clipped_shape[-1]}; clipping to first {clipped_shape[-1]} dims.",
                    )

                prev_contract = self._io_contract_active
                self._io_contract_active = contract
                if prev_contract != contract:
                    self._io_contract_switch_count += 1
                    print(
                        f"[PI0 CONTRACT] Switched contract {prev_contract} -> {contract} "
                        f"(attempt {attempt_idx}/{len(candidates)})."
                    )

                self._last_request_keys = sorted(request_data.keys())
                self._last_has_tactile_values = has_tactile_values
                self._last_has_force_prediction = has_force_prediction
                self._last_prompt_len = len(str(request_data.get("prompt", "")))
                self._last_response_action_shape = raw_shape
                self._last_response_keys = response_keys
                self._last_protocol_error = None
                return actions
            except Exception as exc:
                last_error = exc
                self._last_protocol_error = str(exc)
                is_contract_issue = isinstance(exc, PI0ProtocolError) or is_pi0_contract_error(exc)
                msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                print(
                    f"[PI0 CONTRACT] contract={contract} failed "
                    f"(attempt {attempt_idx}/{len(candidates)}): {msg}"
                )
                if self._protocol_reconnect_on_error:
                    try:
                        self._reconnect_policy_client(reason=f"contract={contract} failure")
                    except Exception as reconnect_exc:
                        self._last_protocol_error = (
                            f"{self._last_protocol_error} | reconnect failed: {reconnect_exc}"
                        )
                        print(f"[PI0 CONTRACT] reconnect failed: {reconnect_exc}")

                should_continue = attempt_idx < len(candidates) and (
                    is_contract_issue or self._protocol_reconnect_on_error
                )
                if not should_continue:
                    break

        raise RuntimeError(
            f"PI0 contract negotiation failed after {len(candidates)} attempt(s). "
            f"active={self._io_contract_active}, order={self._io_contract_order}, "
            f"last_error={last_error}"
        )

    def forward(self, obs_dict, include_info=False):
        start_time = time.time()
        request_keys = self._last_request_keys
        has_tactile_values = self._last_has_tactile_values
        has_force_prediction = self._last_has_force_prediction
        prompt_len = self._last_prompt_len
        chunk_reused = False
        chunk_index = -1
        response_action_shape = self._last_response_action_shape
        response_keys = self._last_response_keys
        effective_horizon = self._last_effective_open_loop_horizon

        try:
            # Get the current observation
            robot_states = obs_dict["robot_state"]
            curr_obs = extract_observation(self.config, obs_dict)

            chunk = self._pred_action_chunk
            chunk_len = len(chunk) if chunk is not None else 0
            configured_horizon = max(1, int(self.config.open_loop_horizon))
            if chunk_len > 0 and configured_horizon > chunk_len:
                _warn_once(
                    self,
                    "open_loop_horizon_clip",
                    f"[PI0 CONTRACT] open_loop_horizon={configured_horizon} > chunk_len={chunk_len}; clipping to {chunk_len}.",
                )
            horizon = min(configured_horizon, chunk_len) if chunk_len > 0 else 0
            effective_horizon = max(1, horizon) if chunk_len > 0 else 0
            self._last_effective_open_loop_horizon = effective_horizon
            need_query = (
                chunk is None
                or self._actions_from_chunk_completed == 0
                or self._actions_from_chunk_completed >= max(1, effective_horizon)
                or self._actions_from_chunk_completed >= chunk_len
            )

            # Send websocket request to policy server if it's time to predict a new chunk
            if need_query:
                prev_chunk = self._pred_action_chunk
                prev_completed = self._actions_from_chunk_completed
                try:
                    self._pred_action_chunk = self._query_action_chunk(curr_obs, robot_states)
                    self._actions_from_chunk_completed = 0
                    request_keys = self._last_request_keys
                    has_tactile_values = self._last_has_tactile_values
                    has_force_prediction = self._last_has_force_prediction
                    prompt_len = self._last_prompt_len
                    response_action_shape = self._last_response_action_shape
                    response_keys = self._last_response_keys
                except Exception:
                    if prev_chunk is not None and prev_completed < len(prev_chunk):
                        self._pred_action_chunk = prev_chunk
                        self._actions_from_chunk_completed = prev_completed
                        chunk_reused = True
                    else:
                        raise
            if not need_query:
                chunk_reused = True

            # Select current action to execute from chunk
            chunk_len = len(self._pred_action_chunk) if self._pred_action_chunk is not None else 0
            if self._pred_action_chunk is None:
                raise RuntimeError("PI0 action chunk is not available.")
            if self._actions_from_chunk_completed >= chunk_len:
                raise RuntimeError("PI0 action chunk exhausted without refresh.")
            chunk_index = self._actions_from_chunk_completed
            action = self._pred_action_chunk[self._actions_from_chunk_completed]
            self._actions_from_chunk_completed += 1

            # print("grasping action", action[-1])
            # Binarize gripper action

            # Original:
            # if action[-1].item() > 0.5:
            #     action = np.concatenate([action[:-1], np.ones((1,))])
            # else:
            #     action = np.concatenate([action[:-1], np.zeros((1,))])

            action = np.clip(action, -1, 1)

        except KeyboardInterrupt:
            action = np.zeros(8)
    
        if include_info:
            chunk_len = len(self._pred_action_chunk) if self._pred_action_chunk is not None else 0
            info = {
                "action_chunk": self._pred_action_chunk,
                "actions_from_chunk_completed": self._actions_from_chunk_completed,
                "time": time.time() - start_time,
                "request_keys": request_keys,
                "chunk_len": chunk_len,
                "chunk_index": chunk_index,
                "chunk_reused": chunk_reused,
                "open_loop_horizon": int(self.config.open_loop_horizon),
                "effective_open_loop_horizon": int(effective_horizon),
                "prompt_len": int(prompt_len),
                "has_tactile_values": bool(has_tactile_values),
                "has_force_prediction": bool(has_force_prediction),
                "response_action_shape": response_action_shape,
                "response_keys": response_keys,
                "io_contract_active": self._io_contract_active,
                "io_contract_attempts": int(self._io_contract_attempts),
                "io_contract_switch_count": int(self._io_contract_switch_count),
                "last_protocol_error": self._last_protocol_error,
            }
            return action, info
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

        # Jaelyn solving urdf error -- matching with mingyo_test
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
                out = self.policy.forward(obs_dict, include_info=include_info)
            grasping = out[-1] if not include_info else out[0][-1]
            self.oculus_reader.force_set_grasping(grasping)
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
        print("reset")
        print("Buttons", self._state["buttons"])

    def set_instruction(self, instruction):
        self._instruction = instruction

    # def get_buttons(self):


    @property
    def instruction(self):
        return self._instruction
    
