"""
Simple and robust IK for UR5e robot
Using Levenberg-Marquardt method for better convergence
"""
import numpy as np
import mujoco


# Default initial joint configuration (from user's image)
# shoulder_pan: 0.26, shoulder_lift: -1.66, elbow: -1.67, 
# wrist_1: -1.04, wrist_2: 1.57, wrist_3: -2.29
DEFAULT_INITIAL_JOINTS = np.array([0.0, -1.57, -1.57, -1.57, 1.57, -1.57])


def move_to_target_pose(model, data, target_pos, target_quat=None, 
                        max_iterations=1000, lambda_init=0.001, 
                        pos_tolerance=1e-3, ori_tolerance=0.02,
                        initial_joints=None, joint_weight=0.0):
    """
    Move UR5e end effector to target position and orientation using Levenberg-Marquardt
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        target_pos: [x, y, z] target position in meters
        target_quat: [w, x, y, z] target quaternion (None = no orientation constraint)
        max_iterations: Maximum number of IK iterations
        lambda_init: Initial damping parameter for LM method (smaller = closer to Gauss-Newton)
        pos_tolerance: Position error tolerance in meters
        ori_tolerance: Orientation error tolerance in radians
        initial_joints: Initial joint configuration to start from (6D array, default from image)
        joint_weight: Weight for staying close to initial_joints (0 = no penalty)
    
    Returns:
        success: True if converged
        pos_error: Final position error
    """
    
    # Get end effector site
    try:
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
    except:
        print("❌ Error: 'eef_site' not found")
        return False, float('inf')
    
    # Get UR5e joint IDs
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint", 
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint"
    ]
    
    joint_ids = []
    for name in joint_names:
        try:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            joint_ids.append(jid)
        except:
            print(f"❌ Error: Joint '{name}' not found")
            return False, float('inf')
    
    # Set initial joint configuration
    if initial_joints is None:
        # Use default configuration and reset robot
        initial_joints = DEFAULT_INITIAL_JOINTS
        data.qpos[joint_ids] = initial_joints
        mujoco.mj_forward(model, data)
        print(f"✓ Using default initial joint configuration")
    else:
        # Use provided configuration - assume robot is already there
        initial_joints = np.array(initial_joints)
        # Do NOT reset qpos - continue from current state
        # print(f"✓ Continuing from current joint configuration")
    
    # Forward kinematics to ensure state is current
    mujoco.mj_forward(model, data)
    
    # Levenberg-Marquardt parameters
    lambda_param = lambda_init  # Damping parameter
    lambda_factor = 10.0  # Factor to increase/decrease lambda
    
    # Track best solution
    best_error = float('inf')
    best_q = data.qpos[joint_ids].copy()
    prev_error = float('inf')
    
    for iteration in range(max_iterations):
        # Forward kinematics
        mujoco.mj_forward(model, data)
        
        # Get current end effector pose
        current_pos = data.site_xpos[ee_site_id].copy()
        
        # Position error
        pos_error = target_pos - current_pos
        pos_error_norm = np.linalg.norm(pos_error)
        
        # Orientation error (if target_quat is specified)
        if target_quat is not None:
            current_mat = data.site_xmat[ee_site_id].reshape(3, 3)
            
            # Convert target quat to rotation matrix
            target_mat = np.zeros(9)
            mujoco.mju_quat2Mat(target_mat, target_quat)
            target_mat = target_mat.reshape(3, 3)
            
            # Orientation error (axis-angle)
            R_error = target_mat @ current_mat.T
            ori_error = np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0],
                R_error[1, 0] - R_error[0, 1]
            ]) * 0.5
            ori_error_norm = np.linalg.norm(ori_error)
        else:
            ori_error = np.zeros(3)
            ori_error_norm = 0.0
        
        # Total error
        total_error = pos_error_norm + ori_error_norm
        
        # Check convergence
        if pos_error_norm < pos_tolerance and ori_error_norm < ori_tolerance:
            print(f"✓ IK converged in {iteration} iterations")
            print(f"  Position error: {pos_error_norm*1000:.2f}mm")
            if target_quat is not None:
                print(f"  Orientation error: {np.degrees(ori_error_norm):.2f}°")
            return True, pos_error_norm
        
        # Track best solution
        if total_error < best_error:
            best_error = total_error
            best_q = data.qpos[joint_ids].copy()
        
        # Compute Jacobian
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
        
        # Extract Jacobian for arm joints only
        J_pos = jacp[:, joint_ids]
        J_ori = jacr[:, joint_ids]
        
        # Construct full Jacobian and error vector
        if target_quat is not None:
            # 6DOF: position + orientation
            J = np.vstack([J_pos, J_ori])
            error = np.concatenate([pos_error, ori_error])
            n_constraints = 6
        else:
            # 3DOF: position only
            J = J_pos
            error = pos_error
            n_constraints = 3
        
        # Levenberg-Marquardt update
        # (J^T J + lambda * I) dq = J^T error
        JTJ = J.T @ J
        JTe = J.T @ error
        
        # Add damping (Levenberg-Marquardt)
        damping_matrix = lambda_param * np.eye(6)
        
        # Add joint limit cost if needed
        if joint_weight > 0:
            q_current = data.qpos[joint_ids]
            joint_error = initial_joints - q_current
            JTe += joint_weight * joint_error
        
        # Solve for dq
        try:
            dq = np.linalg.solve(JTJ + damping_matrix, JTe)
        except np.linalg.LinAlgError:
            print(f"⚠️  Singular matrix at iteration {iteration}, increasing damping")
            lambda_param *= lambda_factor
            continue
        
        # Try the update
        q_new = data.qpos[joint_ids].copy() + dq
        
        # Clamp to joint limits
        for i, jid in enumerate(joint_ids):
            qpos_adr = model.jnt_qposadr[jid]
            joint_range = model.jnt_range[jid]
            q_new[i] = np.clip(q_new[i], joint_range[0], joint_range[1])
        
        # Evaluate new configuration
        data.qpos[joint_ids] = q_new
        mujoco.mj_forward(model, data)
        
        new_pos = data.site_xpos[ee_site_id].copy()
        new_pos_error = np.linalg.norm(target_pos - new_pos)
        
        if target_quat is not None:
            new_mat = data.site_xmat[ee_site_id].reshape(3, 3)
            target_mat_check = np.zeros(9)
            mujoco.mju_quat2Mat(target_mat_check, target_quat)
            target_mat_check = target_mat_check.reshape(3, 3)
            R_error_new = target_mat_check @ new_mat.T
            new_ori_error = np.array([
                R_error_new[2, 1] - R_error_new[1, 2],
                R_error_new[0, 2] - R_error_new[2, 0],
                R_error_new[1, 0] - R_error_new[0, 1]
            ]) * 0.5
            new_ori_error_norm = np.linalg.norm(new_ori_error)
        else:
            new_ori_error_norm = 0.0
        
        new_total_error = new_pos_error + new_ori_error_norm
        
        # Levenberg-Marquardt adaptation
        if new_total_error < total_error:
            # Accept step, decrease damping (move towards Gauss-Newton)
            lambda_param = max(lambda_param / lambda_factor, 1e-7)
            prev_error = new_total_error
        else:
            # Reject step, restore previous q, increase damping (move towards gradient descent)
            data.qpos[joint_ids] = best_q
            lambda_param = min(lambda_param * lambda_factor, 1e6)
    
    # Did not converge - restore best solution
    print(f"⚠️  IK did not fully converge after {max_iterations} iterations")
    data.qpos[joint_ids] = best_q
    mujoco.mj_forward(model, data)
    
    current_pos = data.site_xpos[ee_site_id].copy()
    final_error = np.linalg.norm(target_pos - current_pos)
    print(f"  Best position error: {final_error*1000:.2f}mm")
    
    return False, final_error


def stabilize_robot(model, data, num_steps=50):
    """
    Stabilize robot by setting actuator controls to current joint positions
    and running simulation for a few steps
    """
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint"
    ]
    
    actuator_names = [
        "shoulder_pan_actuator",
        "shoulder_lift_actuator",
        "elbow_actuator",
        "wrist_1_actuator",
        "wrist_2_actuator",
        "wrist_3_actuator"
    ]
    
    # Set actuator controls to current positions
    for joint_name, actuator_name in zip(joint_names, actuator_names):
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            
            # Set control target
            data.ctrl[actuator_id] = data.qpos[joint_id]
            
            # Zero out velocity
            data.qvel[joint_id] = 0.0
            
        except Exception as e:
            print(f"Warning: Could not stabilize {joint_name}: {e}")
    
    # Run simulation to let actuators reach target
    for _ in range(num_steps):
        mujoco.mj_step(model, data)
    
    print(f"✓ Robot stabilized after {num_steps} simulation steps")


def print_joint_state(model, data):
    """Print current joint positions"""
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint"
    ]
    
    print("\n📊 Current Joint State:")
    for name in joint_names:
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            pos = data.qpos[joint_id]
            vel = data.qvel[joint_id]
            print(f"  {name:20s}: {pos:7.4f} rad ({np.degrees(pos):7.2f}°) | vel: {vel:7.4f}")
        except:
            pass


class TaskSpaceController:
    """
    7-DOF Task Space Controller for UR5e + Gripper
    Control interface: [x, y, z, qw, qx, qy, qz, gripper_width]
    
    This allows intuitive control of end-effector pose (6-DOF) + gripper (1-DOF)
    while the underlying joint-space actuators are automatically computed via IK.
    """
    
    def __init__(self, model, data):
        """
        Initialize task space controller
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data
        
        # Get joint and actuator IDs
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint", 
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        
        self.actuator_names = [
            "shoulder_pan_actuator",
            "shoulder_lift_actuator",
            "elbow_actuator",
            "wrist_1_actuator",
            "wrist_2_actuator",
            "wrist_3_actuator"
        ]
        
        # Get IDs
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                         for name in self.joint_names]
        self.actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) 
                            for name in self.actuator_names]
        
        # Gripper joint and actuator IDs
        try:
            self.gripper_left_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rh_p12_rn_left")
            self.gripper_right_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rh_p12_rn_right")
            self.gripper_left_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rh_p12_rn_left_actuator")
            self.gripper_right_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rh_p12_rn_right_actuator")
            self.has_gripper = True
        except:
            print("⚠️  Warning: Gripper not found in model")
            self.has_gripper = False
        
        # End effector site
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
        
        # Current target (initialize to current pose)
        mujoco.mj_forward(model, data)
        self.target_pos = data.site_xpos[self.ee_site_id].copy()
        
        # Get current orientation as quaternion
        current_mat = data.site_xmat[self.ee_site_id].reshape(3, 3)
        self.target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(self.target_quat, current_mat.flatten())
        
        # Current gripper width (0 = closed, max angle = open)
        if self.has_gripper:
            self.target_gripper_angle = data.qpos[self.gripper_left_joint_id]
        else:
            self.target_gripper_angle = 0.0
        
        # Control parameters
        self.ik_tolerance = 5e-3  # 5mm position tolerance
        self.ik_max_iterations = 100  # Max IK iterations per control step
        
        # Gripper parameters (RH-P12-RN specifications)
        # Joint angle 0 = closed, ~0.87 rad (50°) = fully open (~86mm stroke)
        self.gripper_angle_to_width_scale = 86.0 / 0.87  # mm per radian
        self.gripper_max_angle = 0.87  # Max opening angle
        
        print("✓ Task Space Controller initialized")
        print(f"  Control DOF: 6 (pose) + 1 (gripper) = 7 total")
        print(f"  Current EE position: [{self.target_pos[0]:.3f}, {self.target_pos[1]:.3f}, {self.target_pos[2]:.3f}]")
        print(f"  Current gripper angle: {self.target_gripper_angle:.3f} rad ({np.degrees(self.target_gripper_angle):.1f}°)")
    
    def get_current_task_space_state(self):
        """
        Get current 7-DOF task space state
        
        Returns:
            dict with keys:
                'position': [x, y, z] in meters
                'quaternion': [w, x, y, z]
                'gripper_angle': radians (0 = closed, 0.87 = open)
                'gripper_width': mm (0 = closed, 86 = open)
        """
        mujoco.mj_forward(self.model, self.data)
        
        # Get EE pose
        pos = self.data.site_xpos[self.ee_site_id].copy()
        mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        
        # Get gripper state
        if self.has_gripper:
            gripper_angle = self.data.qpos[self.gripper_left_joint_id]
            gripper_width = gripper_angle * self.gripper_angle_to_width_scale
        else:
            gripper_angle = 0.0
            gripper_width = 0.0
        
        return {
            'position': pos,
            'quaternion': quat,
            'gripper_angle': gripper_angle,
            'gripper_width': gripper_width
        }
    
    def set_target_task_space(self, position=None, quaternion=None, gripper_angle=None, gripper_width=None):
        """
        Set target 7-DOF task space configuration
        
        Args:
            position: [x, y, z] target position in meters (None = keep current)
            quaternion: [w, x, y, z] target orientation (None = keep current)
            gripper_angle: Target gripper angle in radians (None = keep current)
            gripper_width: Target gripper width in mm (alternative to gripper_angle)
        """
        if position is not None:
            self.target_pos = np.array(position)
        
        if quaternion is not None:
            self.target_quat = np.array(quaternion)
        
        if gripper_width is not None:
            # Convert width to angle
            self.target_gripper_angle = np.clip(
                gripper_width / self.gripper_angle_to_width_scale,
                0.0,
                self.gripper_max_angle
            )
        elif gripper_angle is not None:
            self.target_gripper_angle = np.clip(gripper_angle, 0.0, self.gripper_max_angle)
    
    def update_control(self, step_ik=True):
        """
        Update joint-space actuator controls to achieve task-space target
        
        Args:
            step_ik: If True, perform IK to compute joint angles for EE target
        
        Returns:
            success: True if IK converged (or if step_ik=False)
        """
        success = True
        
        # 1. Solve IK for arm (6-DOF EE pose → 6 joint angles)
        if step_ik:
            success, error = move_to_target_pose(
                self.model, self.data,
                self.target_pos, self.target_quat,
                max_iterations=self.ik_max_iterations,
                pos_tolerance=self.ik_tolerance,
                ori_tolerance=0.1
            )
        
        # 2. Set arm actuator controls to current joint positions
        for i, (joint_id, actuator_id) in enumerate(zip(self.joint_ids, self.actuator_ids)):
            self.data.ctrl[actuator_id] = self.data.qpos[joint_id]
        
        # 3. Set gripper actuator controls
        if self.has_gripper:
            self.data.ctrl[self.gripper_left_actuator_id] = self.target_gripper_angle
            self.data.ctrl[self.gripper_right_actuator_id] = self.target_gripper_angle
        
        return success
    
    def incremental_move(self, delta_pos=None, delta_euler=None, delta_gripper=None):
        """
        Incremental motion in task space (useful for keyboard/GUI control)
        
        Args:
            delta_pos: [dx, dy, dz] position increment in meters
            delta_euler: [droll, dpitch, dyaw] orientation increment in radians
            delta_gripper: Gripper angle increment in radians
        
        Returns:
            success: True if IK converged
        """
        # Update position
        if delta_pos is not None:
            self.target_pos += np.array(delta_pos)
        
        # Update orientation (apply incremental rotation)
        if delta_euler is not None:
            # Convert current quat to matrix
            current_mat = np.zeros(9)
            mujoco.mju_quat2Mat(current_mat, self.target_quat)
            current_mat = current_mat.reshape(3, 3)
            
            # Create incremental rotation matrix from euler angles
            roll, pitch, yaw = delta_euler
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(roll), -np.sin(roll)],
                          [0, np.sin(roll), np.cos(roll)]])
            Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                          [0, 1, 0],
                          [-np.sin(pitch), 0, np.cos(pitch)]])
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0],
                          [0, 0, 1]])
            delta_R = Rz @ Ry @ Rx
            
            # Apply incremental rotation
            new_mat = current_mat @ delta_R
            
            # Convert back to quaternion
            mujoco.mju_mat2Quat(self.target_quat, new_mat.flatten())
        
        # Update gripper
        if delta_gripper is not None:
            self.target_gripper_angle = np.clip(
                self.target_gripper_angle + delta_gripper,
                0.0,
                self.gripper_max_angle
            )
        
        # Apply control
        return self.update_control(step_ik=True)
    
    def print_status(self):
        """Print current task space state"""
        state = self.get_current_task_space_state()
        
        print("\n" + "="*60)
        print("📊 TASK SPACE STATE (7-DOF)")
        print("="*60)
        print(f"Position (XYZ):    [{state['position'][0]:7.4f}, {state['position'][1]:7.4f}, {state['position'][2]:7.4f}] m")
        print(f"Quaternion (WXYZ): [{state['quaternion'][0]:7.4f}, {state['quaternion'][1]:7.4f}, "
              f"{state['quaternion'][2]:7.4f}, {state['quaternion'][3]:7.4f}]")
        print(f"Gripper Angle:     {state['gripper_angle']:7.4f} rad ({np.degrees(state['gripper_angle']):6.2f}°)")
        print(f"Gripper Width:     {state['gripper_width']:7.2f} mm")
        print("="*60)


if __name__ == "__main__":
    # Test the IK and Task Space Controller
    model = mujoco.MjModel.from_xml_path("ur5e_with_DIGIT_primitive_hexagon.xml")
    data = mujoco.MjData(model)
    
    # Initialize
    mujoco.mj_forward(model, data)
    
    print("\n" + "="*70)
    print("TESTING TASK SPACE CONTROLLER (7-DOF)")
    print("="*70)
    
    # Create task space controller
    controller = TaskSpaceController(model, data)
    controller.print_status()
    
    # Test 1: Move to target position
    print("\n🎯 Test 1: Moving to target position [0.5, 0.1, 0.85]")
    target_pos = np.array([0.5, 0.1, 0.85])
    target_quat = np.array([0.0, 1.0, 0.0, 0.0])  # Gripper pointing down
    
    controller.set_target_task_space(position=target_pos, quaternion=target_quat)
    success = controller.update_control(step_ik=True)
    
    if success:
        print("✅ IK converged")
        stabilize_robot(model, data, num_steps=50)
        controller.print_status()
    else:
        print("❌ IK did not fully converge (but may be close enough)")
        controller.print_status()
    
    # Test 2: Incremental gripper control
    print("\n🤏 Test 2: Opening gripper by 20mm")
    controller.incremental_move(delta_gripper=0.2)  # 0.2 rad ≈ 20mm
    stabilize_robot(model, data, num_steps=20)
    controller.print_status()
    
    # Test 3: Incremental position control
    print("\n📏 Test 3: Moving +5cm in Z direction")
    controller.incremental_move(delta_pos=[0.0, 0.0, 0.05])
    stabilize_robot(model, data, num_steps=50)
    controller.print_status()
    
    print("\n✅ All tests completed!")
