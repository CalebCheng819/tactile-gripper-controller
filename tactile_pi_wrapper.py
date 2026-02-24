"""
Tactile PI Policy Wrapper

Wraps PI policy to conditionally override gripper control with tactile model:
- Enable tactile control when PI policy indicates gripper is closing/grasping
- Disable tactile control when PI policy indicates gripper is opening
- Uses hysteresis to prevent jitter
"""

import sys
import os
import numpy as np
import cv2

# Add tactile_module to path
sys.path.insert(0, "/home/pi0/multi-modal/tactile_module")
# Import will be done inside __init__ to handle errors gracefully

# Model checkpoint path
TACTILE_MODEL_CHECKPOINT = "/home/pi0/multi-modal/tactile_module/checkpoints/two_img_delta_gripper_normalized.pt"
TACTILE_MODEL_CONFIG = "/home/pi0/multi-modal/tactile_module/configs/example_with_normalization.yaml"

# Normalization parameters (from training config)
GRIPPER_NORM_MEAN = 0.015909739212206692
GRIPPER_NORM_STD = 0.021398755184405233

# Safety clamping for gripper delta
GRIPPER_DELTA_MIN = -0.03964751958847046
GRIPPER_DELTA_MAX = 0.09691628813743591
GRIPPER_POSITION_MIN = 0.0
GRIPPER_POSITION_MAX = 1.0

# Hysteresis threshold for opening/closing detection
DEFAULT_HYSTERESIS_EPS = 0.0075  # 0.005-0.01 range as suggested


class TactilePIPolicyWrapper:
    """
    Wrapper around PI policy that conditionally overrides gripper with tactile model.
    
    - Arm action: Always from PI policy
    - Gripper action: 
      * When PI policy indicates closing/grasping → use tactile model
      * When PI policy indicates opening → use PI policy gripper command
    
    Opening/closing detection:
    - Position-based: closing if target_gripper > current_gripper + eps
    - Velocity-based: closing if cmd > +eps
    - Uses hysteresis to prevent jitter
    """
    
    def __init__(
        self,
        policy_client,
        tactile_reader,
        env,
        hysteresis_eps=DEFAULT_HYSTERESIS_EPS,
        tactile_model_checkpoint=TACTILE_MODEL_CHECKPOINT,
        tactile_model_config=TACTILE_MODEL_CONFIG,
    ):
        """
        Args:
            policy_client: PI policy client (websocket_client_policy.WebsocketClientPolicy)
            tactile_reader: Tactile sensor reader (babyFORTEReader)
            env: RobotEnv instance
            hysteresis_eps: Hysteresis threshold for opening/closing detection
            tactile_model_checkpoint: Path to tactile model checkpoint
            tactile_model_config: Path to tactile model config
        """
        self.policy_client = policy_client
        self.tactile_reader = tactile_reader
        self.env = env
        self.hysteresis_eps = hysteresis_eps
        
        # State tracking
        self.current_gripper_position = None
        self.tactile_enabled = False
        self._last_gripper_target = None
        self._print_counter = 0
        
        # Load tactile model
        print(f"\n{'='*70}")
        print("Loading tactile model for PI policy wrapper...")
        print(f"Checkpoint: {tactile_model_checkpoint}")
        print(f"Config: {tactile_model_config}")
        print(f"{'='*70}")
        
        # Verify files exist before attempting to load
        import os
        if not os.path.exists(tactile_model_checkpoint):
            print(f"\n✗ ERROR: Checkpoint file not found: {tactile_model_checkpoint}")
            self.tactile_adapter = None
        elif not os.path.exists(tactile_model_config):
            print(f"\n✗ ERROR: Config file not found: {tactile_model_config}")
            self.tactile_adapter = None
        else:
            print(f"✓ Files exist, attempting to load model...\n")

        # Try to import and load model
        try:
            # Import here to handle import errors gracefully
            from robot_inference_adapter import TactileGripperAdapter
            
            print("  Loading model... (this may take a few seconds)")
            self.tactile_adapter = TactileGripperAdapter(
                checkpoint_path=tactile_model_checkpoint,
                config_path=tactile_model_config,
            )
            print("✓ Tactile model loaded successfully")
            print(f"  Device: {self.tactile_adapter.device}")
            print(f"  Model ready for inference")
        except ImportError as e:
            print(f"\n✗✗✗ FAILED TO IMPORT TACTILE MODULE ✗✗✗")
            print(f"Import Error: {e}")
            print(f"  Make sure tactile_module is installed and accessible")
            import traceback
            traceback.print_exc()
            print(f"\n⚠️  Continuing with PI policy gripper control only")
            print(f"   Tactile gripper control will be DISABLED\n")
            self.tactile_adapter = None
        except Exception as e:
            print(f"\n✗✗✗ FAILED TO LOAD TACTILE MODEL ✗✗✗")
            print(f"Error: {e}")
            print(f"\nFull traceback:")
            import traceback
            traceback.print_exc()
            print(f"\n⚠️  Continuing with PI policy gripper control only")
            print(f"   Tactile gripper control will be DISABLED\n")
            self.tactile_adapter = None
        
        print(f"\n{'='*70}")
        print("🤖 TACTILE PI POLICY WRAPPER")
        print(f"{'='*70}")
        print("Gripper control logic:")
        print("  • Closing/Grasping (target > current) → Tactile model active")
        print("  • Opening (target < current - eps) → PI policy active")
        print(f"  • Hysteresis threshold: {hysteresis_eps:.4f}")
        if self.tactile_adapter is None:
            print("⚠️  WARNING: Tactile model is NOT loaded - tactile control will be disabled")
        else:
            print("✓ Tactile model is ready - tactile control will be enabled when grasping detected")
        print(f"{'='*70}\n")
    
    def _denormalize_gripper_delta(self, normalized_delta):
        """De-normalize gripper delta from model output."""
        return normalized_delta * GRIPPER_NORM_STD + GRIPPER_NORM_MEAN
    
    def _clamp_gripper_delta(self, delta):
        """Clamp gripper delta to safe range."""
        return np.clip(delta, GRIPPER_DELTA_MIN, GRIPPER_DELTA_MAX)
    
    def _detect_opening_closing(self, pi_gripper_cmd, current_gripper):
        """
        Detect if gripper is opening or closing based on PI policy gripper command.
        
        Handles both absolute position and velocity/delta semantics:
        - Absolute position: PI policy outputs target position in [0, 1]
          * Opening: target < current - eps
          * Closing: target > current + eps
        - Velocity/delta: PI policy outputs velocity/delta (can be negative/positive)
          * Opening: cmd < -eps
          * Closing: cmd > +eps
        
        Uses hysteresis to prevent jitter.
        
        Args:
            pi_gripper_cmd: Raw gripper command from PI policy
            current_gripper: Current gripper position [0, 1]
        
        Returns:
            (is_opening, is_closing, mode) tuple where mode is 'position' or 'velocity'
        """
        # Detect mode: if cmd is outside [0, 1] range, it's likely velocity/delta
        # Also check if cmd is negative (velocity/delta can be negative)
        if pi_gripper_cmd < 0 or pi_gripper_cmd > 1:
            # Velocity/delta mode
            mode = 'velocity'
            if self.tactile_enabled:
                # Currently using tactile - need significant closing signal to switch back
                is_opening = pi_gripper_cmd < -self.hysteresis_eps  # Significant opening
                is_closing = pi_gripper_cmd > self.hysteresis_eps  # Significant closing
            else:
                # Currently using PI policy - need opening signal to switch to tactile
                is_opening = pi_gripper_cmd < -self.hysteresis_eps  # Significant opening
                is_closing = pi_gripper_cmd > self.hysteresis_eps  # Significant closing
        else:
            # Absolute position mode
            mode = 'position'
            if self.tactile_enabled:
                # Currently using tactile - need significant closing signal to switch back
                is_opening = pi_gripper_cmd < (current_gripper - self.hysteresis_eps)  # Significant opening
                is_closing = pi_gripper_cmd > (current_gripper + self.hysteresis_eps)  # Significant closing
            else:
                # Currently using PI policy - need opening signal to switch to tactile
                is_opening = pi_gripper_cmd < (current_gripper - self.hysteresis_eps)  # Significant opening
                is_closing = pi_gripper_cmd > (current_gripper + self.hysteresis_eps)  # Significant closing
        
        return is_opening, is_closing, mode
    
    def _get_tactile_gripper_prediction(self, obs_dict):
        """
        Get gripper prediction from tactile model.
        
        Returns:
            target_gripper_position: Target gripper position [0.0, 1.0]
        """
        if self.tactile_adapter is None:
            return None
        
        try:
            # Get wrist camera images
            # Camera images are at top level of obs_dict with keys: "{camera_id}_left" and "{camera_id}_right"
            robot_state = obs_dict.get("robot_state", {})
            
            # Get wrist camera ID
            from r2d2.misc.parameters import hand_camera_id
            wrist_left_key = f"{hand_camera_id}_left"
            wrist_right_key = f"{hand_camera_id}_right"
            
            # Camera images usually live under obs_dict["image"] (see MultiCameraWrapper)
            image_dict = obs_dict.get("image", {})
            # Fix: Use explicit None checks instead of 'or' to avoid numpy array truth-value error
            wrist_left = image_dict.get(wrist_left_key)
            if wrist_left is None:
                wrist_left = obs_dict.get(wrist_left_key)
            wrist_right = image_dict.get(wrist_right_key)
            if wrist_right is None:
                wrist_right = obs_dict.get(wrist_right_key)
            if not hasattr(self, "_image_keys_logged"):
                print(f"[DEBUG] Available image keys: {list(image_dict.keys())[:10]}...")
                print(f"[DEBUG] Looking for wrist keys: {wrist_left_key}, {wrist_right_key}")
                self._image_keys_logged = True
            
            # Fallback: try alternative key names (for compatibility)
            if wrist_left is None:
                wrist_left = obs_dict.get("wrist_image_left")
                if wrist_left is None:
                    wrist_left = obs_dict.get("wrist_image")
            if wrist_right is None:
                wrist_right = obs_dict.get("wrist_image_right")
            
            # If right image not available, use left as fallback
            if wrist_right is None:
                wrist_right = wrist_left
            
            if wrist_left is None or wrist_right is None:
                return None
            
            # Preprocess images to match training format: (H, W, 3) RGB uint8
            def preprocess_image(img):
                """Preprocess image to match training format."""
                if img is None:
                    return None
                
                # Handle different input formats
                if len(img.shape) == 3:
                    if img.shape[0] == 4:
                        # (4, H, W) format - transpose to (H, W, 4)
                        img = np.transpose(img, (1, 2, 0))
                    
                    # Convert BGRA/BGR to RGB
                    if img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    elif img.shape[2] == 3:
                        # Assume RGB if already 3-channel
                        pass
                
                # Ensure format is (H, W, 3) RGB uint8
                if len(img.shape) != 3 or img.shape[2] != 3:
                    return None
                
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                
                return img
            
            wrist_left = preprocess_image(wrist_left)
            wrist_right = preprocess_image(wrist_right)
            
            if wrist_left is None or wrist_right is None:
                # Debug: print why images are None
                if not hasattr(self, '_image_warning_printed'):
                    print(f"[DEBUG] Tactile model: Missing wrist images (left={wrist_left is None}, right={wrist_right is None})")
                    print(f"        Available keys in obs_dict: {list(obs_dict.keys())[:10]}...")
                    self._image_warning_printed = True
                return None
            
            # Get tactile history
            tactile_history = self.tactile_reader.read_values()
            
            # Get current gripper position from robot_state
            gripper_pos = robot_state.get("gripper_position")
            
            if gripper_pos is None:
                # Fallback: try to get from env if available
                if hasattr(self.env, '_robot') and hasattr(self.env._robot, 'get_gripper_position'):
                    try:
                        gripper_pos = self.env._robot.get_gripper_position()
                    except:
                        gripper_pos = 0.0
                else:
                    gripper_pos = 0.0
            
            # Handle different formats
            if isinstance(gripper_pos, (list, np.ndarray)):
                current_gripper = float(gripper_pos[0] if len(gripper_pos) > 0 else 0.0)
            else:
                current_gripper = float(gripper_pos)
            
            # Run tactile model inference
            normalized_delta = self.tactile_adapter.predict_delta(
                image_left=wrist_left,
                image_right=wrist_right,
                tactile_history=tactile_history,
                step_index=0  # Use first step from action chunk
            )
            
            # De-normalize
            delta = self._denormalize_gripper_delta(normalized_delta)
            
            # Clamp delta
            delta = self._clamp_gripper_delta(delta)
            
            # Compute target gripper position
            target_gripper = current_gripper + delta
            target_gripper = np.clip(target_gripper, GRIPPER_POSITION_MIN, GRIPPER_POSITION_MAX)
            
            return target_gripper
        
        except Exception as e:
            # Print error but don't spam (only first few times)
            if not hasattr(self, '_error_count'):
                self._error_count = 0
            self._error_count += 1
            if self._error_count <= 3:
                print(f"[ERROR] Tactile model inference failed: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def infer(self, request_data):
        """
        Wrapper around policy_client.infer() that conditionally overrides gripper.
        
        This method should be called instead of policy_client.infer() directly.
        However, since we need access to the observation dict to run tactile model,
        we'll need to modify the calling code to pass obs_dict as well.
        
        For now, this is a placeholder. The actual integration will happen in
        the main loop where we have access to both the action chunk and obs_dict.
        """
        # Just pass through to policy client
        # The actual gripper override happens in get_action_with_tactile()
        return self.policy_client.infer(request_data)
    
    def get_action_with_tactile(self, action_chunk, action_index, obs_dict):
        """
        Get action from chunk with conditional tactile gripper override.
        
        Args:
            action_chunk: Action chunk from PI policy [horizon, 8]
            action_index: Index into action chunk
            obs_dict: Observation dictionary with images and robot state
        
        Returns:
            action: Action array [8] with potentially overridden gripper
        """
        # Get base action from PI policy
        action = action_chunk[action_index].copy()
        
        # Get current gripper position from observation
        # obs_dict structure: {"robot_state": {"gripper_position": ...}, "image": {...}, ...}
        robot_state = obs_dict.get("robot_state", {})
        gripper_pos = robot_state.get("gripper_position")
        
        if gripper_pos is None:
            # Fallback: try to get from env if available
            if hasattr(self.env, '_robot') and hasattr(self.env._robot, 'get_gripper_position'):
                try:
                    gripper_pos = self.env._robot.get_gripper_position()
                except:
                    gripper_pos = 0.0
            else:
                gripper_pos = 0.0

        # Handle different formats
        if isinstance(gripper_pos, (list, np.ndarray)):
            current_gripper = float(gripper_pos[0] if len(gripper_pos) > 0 else 0.0)
        else:
            current_gripper = float(gripper_pos)

        self.current_gripper_position = current_gripper
        
        # Get PI policy gripper target (before binarization)
        pi_gripper_target = float(action[-1])
        
        # Detect opening vs closing
        is_opening, is_closing, mode = self._detect_opening_closing(
            pi_gripper_target, current_gripper
        )
        
        # One-time debug print showing PI gripper raw output, current position, and interpreted mode
        if not hasattr(self, '_one_time_debug_printed'):
            print(f"\n{'='*70}")
            print("🔍 [ONE-TIME DEBUG] PI Policy Gripper Semantics:")
            print(f"   PI gripper raw output: {pi_gripper_target:.6f}")
            print(f"   Current gripper position: {current_gripper:.6f}")
            print(f"   Interpreted mode: {mode} ({'absolute position' if mode == 'position' else 'velocity/delta'})")
            print(f"   Detection: opening={is_opening}, closing={is_closing}")
            print(f"{'='*70}\n")
            self._one_time_debug_printed = True
        
        # Debug: print detection results occasionally
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter <= 5:
            print(f"[DEBUG] Detection: current={current_gripper:.3f}, target={pi_gripper_target:.3f}, "
                  f"mode={mode}, is_opening={is_opening}, is_closing={is_closing}, "
                  f"adapter={'available' if self.tactile_adapter is not None else 'None'}")
        
        # Update tactile_enabled flag based on opening/closing detection
        # Enable tactile when PI0 wants to grasp (closing), disable when opening
        # Only enable tactile if adapter is available
        if is_closing and self.tactile_adapter is not None:
            self.tactile_enabled = True
        elif is_opening:
            self.tactile_enabled = False
        # If neither (within hysteresis band), keep current state
        
        # Apply tactile model if enabled
        if self.tactile_enabled and self.tactile_adapter is not None:
            tactile_target = self._get_tactile_gripper_prediction(obs_dict)
            
            if tactile_target is not None:
                # Override gripper with tactile model prediction
                action[-1] = tactile_target
                
                # Print status (every 30 steps to reduce clutter)
                self._print_counter += 1
                if self._print_counter % 30 == 0:
                    print(f"🎯 [TACTILE ACTIVE] Gripper: {current_gripper:.3f} → {tactile_target:.3f} "
                          f"(PI target: {pi_gripper_target:.3f}, grasping detected)")
            else:
                # Fallback to PI policy if tactile model fails
                # Print warning more frequently to help debug
                self._print_counter += 1
                if self._print_counter % 10 == 0:
                    print(f"⚠️  [TACTILE FAILED] Falling back to PI policy gripper (tactile_target=None)")
        else:
            # Use PI policy gripper (will be binarized later in main loop)
            self._print_counter += 1
            if self._print_counter % 30 == 0:
                if self.tactile_enabled:
                    print(f"📌 [PI POLICY] Gripper: {current_gripper:.3f} → {pi_gripper_target:.3f} "
                          f"(tactile model unavailable)")
                elif is_opening:
                    print(f"📌 [PI POLICY] Gripper: {current_gripper:.3f} → {pi_gripper_target:.3f} "
                          f"(opening detected)")
                elif is_closing:
                    print(f"📌 [PI POLICY] Gripper: {current_gripper:.3f} → {pi_gripper_target:.3f} "
                          f"(grasping detected, but tactile model unavailable)")
                else:
                    print(f"📌 [PI POLICY] Gripper: {current_gripper:.3f} → {pi_gripper_target:.3f} "
                          f"(neutral/hysteresis band)")
        
        return action
