import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
from gripper_digit_sensor import GripperDIGITSensor
import time
import warnings
warnings.filterwarnings("ignore")

class HexagonPegInHoleRL(gym.Env):
    """
    Reinforcement Learning Environment for Hexagon Peg-in-Hole Task
    
    Task Description:
    1. Initial: Robot starts with peg grasped at position (0.629, 0.0, 0.875)
    2. Phase 1: Approach hole entrance at (0.629, 0.029, 0.8)
    3. Phase 2: Align peg to hole and insert
    4. Phase 3: Release gripper when peg is successfully inserted
    
    State Space (16 DOF):
    - NPE sensors: force (normal) + contact point (x,y local) for both sides -> 6 DOF
    - Peg: position + orientation (quaternion) -> 7 DOF
    - Peg to hole entrance: 3D vector -> 3 DOF
    
    Action Space (7 DOF):
    - 6 DOF joint angles (UR5e arm)
    - 1 DOF gripper width (0=closed, 1=open)
    
    Reward Structure:
    - Phase 1: Reduce distance to hole entrance
    - Phase 2: Align peg and maintain grasp stability (NPE feedback)
    - Phase 3: Successful release and insertion
    - Penalties: NPE deviation, abrupt movements, workspace violations, dropping peg
    """
    
    def __init__(self, xml_path="ur5e_with_DIGIT_primitive_hexagon.xml", 
                 render_mode=None, max_episode_steps=1000):
        super().__init__()
        
        self.xml_path = xml_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        
        # MuJoCo simulation
        self.model = None
        self.data = None
        self.viewer = None
        
        # DIGIT sensors
        self.digit_left_sensor = None
        self.digit_right_sensor = None
        
        # Episode tracking
        self.step_count = 0
        self.initial_peg_pos = np.array([0.629, 0.0, 0.875])
        self.hole_entrance_pos = np.array([0.629, 0.029, 0.8])
        self.initial_eef_pos = None
        self.workspace_center = None
        self.workspace_bounds = 0.1  # 100mm boundary box
        
        # Task phase tracking
        self.task_phase = 1  # 1: approach, 2: align/insert, 3: release
        self.peg_grasped = True
        self.success = False
        
        # Previous state for smoothness penalty
        self.prev_joint_angles = None
        self.prev_contacts_left = None
        self.prev_contacts_right = None
        
        # Load environment
        self._load_model()
        self._setup_spaces()
        self._initialize_sensors()
    
    def _load_model(self):
        """Load MuJoCo model and initialize"""
        try:
            self.model = mujoco.MjModel.from_xml_path(self.xml_path)
            self.data = mujoco.MjData(self.model)
            
            # Set initial joint positions (from demo)
            joint_names = [
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
            ]
            
            # Natural pose to reach peg
            initial_angles = [0.26, -1.66, -1.67, -1.04, 1.57, -2.29]
            
            for joint_name, angle in zip(joint_names, initial_angles):
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                self.data.qpos[joint_id] = angle
            
            # Close gripper initially (grasp peg)
            left_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rh_p12_rn_left")
            right_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rh_p12_rn_right")
            self.data.qpos[left_joint_id] = 1.1  # Closed
            self.data.qpos[right_joint_id] = 1.1  # Closed
            
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)
            
            # Store initial EE position for workspace bounds
            eef_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
            self.initial_eef_pos = self.data.site_xpos[eef_site_id].copy()
            self.workspace_center = self.hole_entrance_pos.copy()
            
            print(f"✓ Loaded {self.xml_path}")
            print(f"✓ Initial EE position: {self.initial_eef_pos}")
            print(f"✓ Workspace center: {self.workspace_center}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model: {e}")
    
    def _setup_spaces(self):
        """Define observation and action spaces"""
        
        # State space: 16 DOF
        # NPE: 2 sensors × (1 force + 2 contact coords) = 6 DOF
        # Peg: 3 position + 4 quaternion = 7 DOF  
        # Peg-to-hole vector: 3 DOF
        # Total: 6 + 7 + 3 = 16 DOF
        
        state_low = np.array([
            # NPE Left: force, x_contact, y_contact
            0.0, -10.0, -10.0,
            # NPE Right: force, x_contact, y_contact  
            0.0, -10.0, -10.0,
            # Peg position (global)
            0.4, -0.2, 0.7,
            # Peg orientation (quaternion w,x,y,z)
            -1.0, -1.0, -1.0, -1.0,
            # Peg to hole vector
            -0.5, -0.5, -0.5
        ])
        
        state_high = np.array([
            # NPE Left: force, x_contact, y_contact
            10.0, 10.0, 10.0,
            # NPE Right: force, x_contact, y_contact
            10.0, 10.0, 10.0,
            # Peg position (global)
            0.8, 0.2, 1.0,
            # Peg orientation (quaternion w,x,y,z)
            1.0, 1.0, 1.0, 1.0,
            # Peg to hole vector
            0.5, 0.5, 0.5
        ])
        
        self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32)
        
        # Action space: 7 DOF
        # 6 joint angles + 1 gripper (0=closed, 1=open)
        action_low = np.array([
            -3.14, -3.14, -3.14, -3.14, -3.14, -3.14,  # Joint limits
            0.0  # Gripper closed
        ])
        action_high = np.array([
            3.14, 3.14, 3.14, 3.14, 3.14, 3.14,  # Joint limits
            1.1  # Gripper open
        ])
        
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        
        print(f"✓ State space: {self.observation_space.shape}")
        print(f"✓ Action space: {self.action_space.shape}")
    
    def _initialize_sensors(self):
        """Initialize DIGIT sensors"""
        try:
            self.digit_left_sensor = GripperDIGITSensor(
                model=self.model,
                sensor_body_name="digit_geltip_left",
                sensor_type="left"
            )
            
            self.digit_right_sensor = GripperDIGITSensor(
                model=self.model,
                sensor_body_name="digit_geltip_right", 
                sensor_type="right"
            )
            
            print("✓ DIGIT sensors initialized")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize sensors: {e}")
    
    def _get_npe_data(self):
        """Get NPE (Normal Pressure Element) data from DIGIT sensors"""
        
        # Get contacts from both sensors
        left_contacts = self.digit_left_sensor.detect_proximity_contacts(self.data)
        right_contacts = self.digit_right_sensor.detect_proximity_contacts(self.data)
        
        # Left sensor data
        if left_contacts:
            left_force = len(left_contacts) * 0.01  # Simple force approximation
            left_x = np.mean([c['x_mm'] for c in left_contacts])
            left_y = np.mean([c['y_mm'] for c in left_contacts])
        else:
            left_force, left_x, left_y = 0.0, 0.0, 0.0
        
        # Right sensor data
        if right_contacts:
            right_force = len(right_contacts) * 0.01
            right_x = np.mean([c['x_mm'] for c in right_contacts])
            right_y = np.mean([c['y_mm'] for c in right_contacts])
        else:
            right_force, right_x, right_y = 0.0, 0.0, 0.0
        
        npe_data = np.array([left_force, left_x, left_y, right_force, right_x, right_y])
        
        # Store for smoothness penalty
        self.prev_contacts_left = left_contacts
        self.prev_contacts_right = right_contacts
        
        return npe_data
    
    def _get_peg_pose(self):
        """Get peg global position and orientation"""
        try:
            peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hexagon_peg_body")
            
            # Position
            pos = self.data.xpos[peg_body_id].copy()
            
            # Orientation (rotation matrix to quaternion)
            rot_mat = self.data.xmat[peg_body_id].reshape(3, 3)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rot_mat.flatten())
            
            return np.concatenate([pos, quat])  # [x,y,z,w,qx,qy,qz]
            
        except Exception as e:
            print(f"Warning: Could not get peg pose: {e}")
            return np.zeros(7)
    
    def _get_peg_to_hole_vector(self):
        """Get 3D vector from peg to hole entrance"""
        try:
            peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hexagon_peg_body")
            peg_pos = self.data.xpos[peg_body_id].copy()
            
            # Vector from peg to hole entrance
            peg_to_hole = self.hole_entrance_pos - peg_pos
            
            return peg_to_hole
            
        except Exception as e:
            print(f"Warning: Could not compute peg-to-hole vector: {e}")
            return np.zeros(3)
    
    def _get_observation(self):
        """Get full state observation"""
        
        # NPE data (6 DOF)
        npe_data = self._get_npe_data()
        
        # Peg pose (7 DOF)
        peg_pose = self._get_peg_pose()
        
        # Peg to hole vector (3 DOF)
        peg_to_hole = self._get_peg_to_hole_vector()
        
        # Concatenate all (16 DOF total)
        observation = np.concatenate([npe_data, peg_pose, peg_to_hole]).astype(np.float32)
        
        return observation
    
    def _apply_action(self, action):
        """Apply action to robot"""
        
        # Joint actions (first 6 DOF)
        joint_actions = action[:6]
        gripper_action = action[6]
        
        # Apply joint actions
        joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        
        actuator_names = [
            "shoulder_pan_actuator", "shoulder_lift_actuator", "elbow_actuator", 
            "wrist_1_actuator", "wrist_2_actuator", "wrist_3_actuator"
        ]
        
        for actuator_name, joint_angle in zip(actuator_names, joint_actions):
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            self.data.ctrl[actuator_id] = joint_angle
        
        # Apply gripper action
        try:
            left_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rh_p12_rn_left_actuator")
            right_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rh_p12_rn_right_actuator")
            self.data.ctrl[left_actuator_id] = gripper_action
            self.data.ctrl[right_actuator_id] = gripper_action
        except:
            pass
        
        # Store for smoothness penalty
        self.prev_joint_angles = joint_actions.copy()
    
    def _compute_reward(self, observation, action):
        """Compute reward based on task phases and constraints"""
        
        total_reward = 0.0
        info = {}
        
        # Extract data from observation
        npe_data = observation[:6]
        peg_pose = observation[6:13]
        peg_to_hole = observation[13:16]
        
        peg_pos = peg_pose[:3]
        peg_quat = peg_pose[3:]
        
        # Distance to hole
        distance_to_hole = np.linalg.norm(peg_to_hole)
        
        # ========== PHASE-BASED REWARDS ==========
        
        # Phase 1: Approach hole entrance
        if self.task_phase == 1:
            # Main reward: reduce distance to hole (increased weight)
            distance_reward = -distance_to_hole * 20.0
            total_reward += distance_reward
            info['distance_reward'] = distance_reward
            
            # Transition to phase 2 when close enough
            if distance_to_hole < 0.05:  # 50mm
                self.task_phase = 2
                total_reward += 50.0  # Bonus for reaching hole
                info['phase_transition'] = True
        
        # Phase 2: Align and insert peg
        elif self.task_phase == 2:
            # Distance reward (smaller weight)
            distance_reward = -distance_to_hole * 5.0
            total_reward += distance_reward
            
            # Alignment reward (peg should be vertical)
            # Ideal quaternion for vertical peg: [1, 0, 0, 0] or similar
            alignment_reward = -np.linalg.norm(peg_quat - np.array([1, 0, 0, 0])) * 20.0
            total_reward += alignment_reward
            info['alignment_reward'] = alignment_reward
            
            # Check if peg is inserted (Z position lower)
            if peg_pos[2] < 0.81:  # Below hole entrance
                self.task_phase = 3
                total_reward += 100.0  # Major bonus for insertion
                info['insertion_success'] = True
        
        # Phase 3: Release peg
        elif self.task_phase == 3:
            # Reward for opening gripper
            gripper_open_reward = action[6] * 30.0  # Encourage opening
            total_reward += gripper_open_reward
            info['gripper_reward'] = gripper_open_reward
            
            # Success if gripper is open and peg is in hole
            if action[6] > 0.8 and peg_pos[2] < 0.82:
                self.success = True
                total_reward += 200.0  # Success bonus
                info['success'] = True
        
        # ========== CONSTRAINT PENALTIES ==========
        
        # 1. NPE deviation penalty (grasp stability)
        if self.peg_grasped:
            left_force, left_x, left_y = npe_data[0], npe_data[1], npe_data[2]
            right_force, right_x, right_y = npe_data[3], npe_data[4], npe_data[5]
            
            # Penalty for contact point deviation from center
            left_deviation = np.sqrt(left_x**2 + left_y**2)
            right_deviation = np.sqrt(right_x**2 + right_y**2)
            npe_penalty = -(left_deviation + right_deviation) * 2.0
            total_reward += npe_penalty
            info['npe_penalty'] = npe_penalty
            
            # Penalty for force imbalance
            force_imbalance = abs(left_force - right_force) * 10.0
            total_reward -= force_imbalance
            info['force_imbalance'] = force_imbalance
        
        # 2. Smooth movement penalty (reduced)
        if self.prev_joint_angles is not None:
            joint_velocity = np.linalg.norm(action[:6] - self.prev_joint_angles)
            smoothness_penalty = -joint_velocity * 2.0
            total_reward += smoothness_penalty
            info['smoothness_penalty'] = smoothness_penalty
        
        # 3. Workspace boundary penalty (reduced)
        try:
            eef_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
            eef_pos = self.data.site_xpos[eef_site_id]
            workspace_violation = np.linalg.norm(eef_pos - self.workspace_center) - self.workspace_bounds
            if workspace_violation > 0:
                boundary_penalty = -workspace_violation * 20.0
                total_reward += boundary_penalty
                info['boundary_penalty'] = boundary_penalty
        except:
            pass
        
        # 4. Peg drop detection (major failure)
        if self.peg_grasped and (left_force + right_force) < 0.01:
            # Peg might have been dropped
            if action[6] < 0.8:  # Gripper should be closed
                total_reward -= 100.0  # Major penalty
                self.peg_grasped = False
                info['peg_dropped'] = True
        
        info['total_reward'] = total_reward
        info['task_phase'] = self.task_phase
        info['distance_to_hole'] = distance_to_hole
        
        return total_reward, info
    
    def _check_termination(self, observation):
        """Check if episode should terminate"""
        
        # Success termination
        if self.success:
            return True, False  # (done, truncated)
        
        # Max steps reached
        if self.step_count >= self.max_episode_steps:
            return True, True  # (done, truncated)
        
        # Failure conditions
        peg_pos = observation[6:9]
        
        # Peg fell too far
        if peg_pos[2] < 0.7:  # Below reasonable level
            return True, False  # Failed
        
        # Peg moved too far from target area
        if np.linalg.norm(peg_pos[:2] - self.hole_entrance_pos[:2]) > 0.2:
            return True, False  # Failed
        
        return False, False  # Continue
    
    def step(self, action):
        """Execute one environment step"""
        
        self.step_count += 1
        
        # Apply action
        self._apply_action(action)
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation
        observation = self._get_observation()
        
        # Compute reward
        reward, info = self._compute_reward(observation, action)
        
        # Check termination
        terminated, truncated = self._check_termination(observation)
        
        info['step'] = self.step_count
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode with IK-based grasp and NPE contact check"""
        import simple_ik
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.step_count = 0
        self.task_phase = 1
        self.peg_grasped = True
        self.success = False
        self.prev_joint_angles = None
        self.prev_contacts_left = None
        self.prev_contacts_right = None
        self.initial_npe_left = None
        self.initial_npe_right = None
        # Reset peg position (with small randomization)
        peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hexagon_peg_body")
        if options and options.get('randomize_peg', True):
            offset = np.random.normal(0, 0.01, 3)
            peg_pos = self.initial_peg_pos + offset
        else:
            peg_pos = self.initial_peg_pos.copy()
        try:
            peg_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hexagon_peg_body")
            joint_qpos_addr = self.model.jnt_qposadr[peg_joint_id]
            self.data.qpos[joint_qpos_addr:joint_qpos_addr+3] = peg_pos
            self.data.qpos[joint_qpos_addr+3:joint_qpos_addr+7] = [1, 0, 0, 0]
        except:
            print("Warning: Could not reset peg position")
        # Use IK to move EEF above peg (z+0.005), top-down orientation
        target_pos = peg_pos.copy(); target_pos[2] += 0.005
        target_quat = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg X, gripper down
        preferred_joints = np.array([0.26, -1.66, -1.67, -1.04, 1.57, -2.29])
        try:
            simple_ik.move_to_target_pose(
                self.model, self.data,
                target_pos, target_quat,
                max_iterations=500, step_size=0.1, pos_tolerance=5e-3, ori_tolerance=0.1,
                initial_joints=preferred_joints, joint_weight=0.02
            )
        except Exception as e:
            print(f"IK failed: {e}")
        # Close gripper
        left_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rh_p12_rn_left")
        right_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rh_p12_rn_right")
        self.data.qpos[left_joint_id] = 1.1
        self.data.qpos[right_joint_id] = 1.1
        mujoco.mj_forward(self.model, self.data)
        # Wait for NPE contact (simulate until both sensors have contact)
        max_wait_steps = 200
        for _ in range(max_wait_steps):
            mujoco.mj_step(self.model, self.data)
            npe = self._get_npe_data()
            if npe[0] > 0.0 and npe[3] > 0.0:
                self.initial_npe_left = npe[:3].copy()
                self.initial_npe_right = npe[3:6].copy()
                break
        # Enforce EEF workspace boundary (0.1m box from entrance)
        eef_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
        eef_pos = self.data.site_xpos[eef_site_id]
        if np.any(np.abs(eef_pos - np.array([0.629, 0.029, 0.875])) > 0.1):
            print("Warning: EEF out of workspace bounds after reset!")
        mujoco.mj_forward(self.model, self.data)
        observation = self._get_observation()
        info = {
            'initial_peg_pos': peg_pos.copy(),
            'hole_entrance_pos': self.hole_entrance_pos.copy(),
            'task_phase': self.task_phase,
            'initial_npe_left': self.initial_npe_left,
            'initial_npe_right': self.initial_npe_right
        }
        return observation, info
    
    def render(self):
        """Render environment (if render_mode is set)"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.distance = 1.5
                self.viewer.cam.lookat = [0.6, 0.0, 0.85]
                self.viewer.cam.elevation = -30
                self.viewer.cam.azimuth = 135
            
            if self.viewer.is_running():
                self.viewer.sync()
                time.sleep(0.01)
    
    def close(self):
        """Clean up resources"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

# Test the environment
if __name__ == "__main__":
    print("Testing HexagonPegInHoleRL Environment")
    
    env = HexagonPegInHoleRL(render_mode="human")
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}: reward={reward:.3f}, phase={info.get('task_phase', 1)}, "
              f"distance={info.get('distance_to_hole', 0):.3f}")
        
        env.render()
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    env.close()
    print("Environment test completed!")