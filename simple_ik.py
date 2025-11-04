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
    
    # High-precision Levenberg-Marquardt parameters
    lambda_param = lambda_init  # Very low initial damping for precision
    lambda_min = 1e-12  # Minimum damping for numerical stability
    lambda_max = 1e3    # Maximum damping before giving up
    lambda_factor = 2.5  # Gentler adaptation factor
    
    # Track best solution
    best_error = float('inf')
    best_q = data.qpos[joint_ids].copy()
    best_pos_error = float('inf')
    
    # Convergence tracking with more patience for high precision
    error_history = []
    stagnation_threshold = 20   # More iterations before declaring stagnation
    stagnation_count = 0
    improvement_threshold = 1e-6  # Smaller threshold for detecting improvement
    
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
        
        # Weighted total error (prioritize position for manipulation tasks)
        pos_weight = 10.0  # Higher weight for position precision
        ori_weight = 1.0   # Lower weight for orientation
        total_error = pos_weight * pos_error_norm + ori_weight * ori_error_norm
        
        # Track error history for stagnation detection
        error_history.append(pos_error_norm)
        if len(error_history) > stagnation_threshold:
            recent_errors = error_history[-stagnation_threshold:]
            if abs(max(recent_errors) - min(recent_errors)) < improvement_threshold:
                stagnation_count += 1
            else:
                stagnation_count = 0
        
        # High-precision convergence check (1mm pos, 0.02 rad ori)
        if pos_error_norm < pos_tolerance and ori_error_norm < ori_tolerance:
            print(f"✓ IK converged in {iteration} iterations")
            print(f"  Position error: {pos_error_norm*1000:.3f}mm")
            if target_quat is not None:
                print(f"  Orientation error: {ori_error_norm:.4f} rad ({np.degrees(ori_error_norm):.3f}°)")
            return True, pos_error_norm
        
        # Track best solution based on position error (primary criterion)
        if pos_error_norm < best_pos_error:
            best_pos_error = pos_error_norm
            best_error = total_error
            best_q = data.qpos[joint_ids].copy()
        
        # Early exit if truly stagnating (prevents infinite loops) 
        if stagnation_count > 50:  # Much higher threshold for high precision
            print(f"⚠️  Prolonged stagnation detected at iteration {iteration}, using best solution")
            break
        
        # High-precision Jacobian computation
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
        
        # Extract Jacobian for arm joints only (6 DOF)
        J_pos = jacp[:, joint_ids]  # 3x6 position Jacobian
        J_ori = jacr[:, joint_ids]  # 3x6 orientation Jacobian
        
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
        
        # High-precision Levenberg-Marquardt with adaptive step sizing
        JTJ = J.T @ J
        JTe = J.T @ error
        
        # Improved damping with better numerical conditioning
        # Use diagonal of JTJ for better scaling (Marquardt modification)
        diag_JTJ = np.diag(np.diag(JTJ))
        damping_matrix = lambda_param * (diag_JTJ + 1e-6 * np.eye(6))
        
        # Add joint regularization for better conditioning
        if joint_weight > 0:
            q_current = data.qpos[joint_ids]
            joint_error = initial_joints - q_current
            JTe += joint_weight * joint_error
            # Add joint regularization to damping
            damping_matrix += joint_weight * np.eye(6)
        
        # Solve with improved numerical stability
        A = JTJ + damping_matrix
        try:
            # Use Cholesky decomposition for better numerical stability
            L = np.linalg.cholesky(A)
            y = np.linalg.solve(L, JTe)
            dq = np.linalg.solve(L.T, y)
        except np.linalg.LinAlgError:
            # Fallback to SVD for robustness
            try:
                dq = np.linalg.lstsq(A, JTe, rcond=1e-12)[0]
            except np.linalg.LinAlgError:
                print(f"⚠️  Matrix conditioning issue at iteration {iteration}")
                lambda_param = min(lambda_param * lambda_factor, lambda_max)
                continue
        
        # Adaptive step size limitation for stability
        dq_norm = np.linalg.norm(dq)
        max_step = 0.2  # Maximum joint change per iteration (radians)
        if dq_norm > max_step:
            dq = dq * (max_step / dq_norm)
        
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
        
        new_total_error = pos_weight * new_pos_error + ori_weight * new_ori_error_norm
        
        # Advanced Levenberg-Marquardt adaptation for high precision
        error_reduction = total_error - new_total_error
        predicted_reduction = 0.5 * dq.T @ (lambda_param * damping_matrix @ dq + JTe)
        
        if predicted_reduction > 0:
            gain_ratio = error_reduction / predicted_reduction
        else:
            gain_ratio = 0
        
        if gain_ratio > 0.75:
            # Excellent step - reduce damping significantly
            lambda_param = max(lambda_param / (lambda_factor * 2), lambda_min)
        elif gain_ratio > 0.25:
            # Good step - reduce damping moderately  
            lambda_param = max(lambda_param / lambda_factor, lambda_min)
        elif gain_ratio > 0:
            # Accept step but don't change damping
            pass
        else:
            # Bad step - reject and increase damping
            data.qpos[joint_ids] = data.qpos[joint_ids] - dq  # Undo the step
            lambda_param = min(lambda_param * lambda_factor, lambda_max)
            continue  # Skip to next iteration without accepting the step
        
        # Precision-adaptive damping strategy
        if pos_error_norm < 2e-3:  # Within 2mm - ultra-precision mode
            lambda_param = max(lambda_param, 0.0001)  # Very low damping for fine adjustments
        elif pos_error_norm < 5e-3:  # Within 5mm - high-precision mode  
            lambda_param = max(lambda_param, 0.001)   # Low damping for precision
        
        # Special handling when very close to target
        if pos_error_norm < 1.5e-3 and ori_error_norm < 0.03:
            # We're close - try micro-adjustments
            max_step = 0.05  # Smaller steps when very close
    
    # Did not converge to required precision - restore best solution
    print(f"⚠️  IK did not achieve required precision after {max_iterations} iterations")
    data.qpos[joint_ids] = best_q
    mujoco.mj_forward(model, data)
    
    current_pos = data.site_xpos[ee_site_id].copy()
    final_pos_error = np.linalg.norm(target_pos - current_pos)
    
    if target_quat is not None:
        current_mat = data.site_xmat[ee_site_id].reshape(3, 3)
        target_mat = np.zeros(9)
        mujoco.mju_quat2Mat(target_mat, target_quat)
        target_mat = target_mat.reshape(3, 3)
        R_error = target_mat @ current_mat.T
        ori_error = np.array([
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0], 
            R_error[1, 0] - R_error[0, 1]
        ]) * 0.5
        final_ori_error = np.linalg.norm(ori_error)
        
        print(f"  Best position error: {final_pos_error*1000:.3f}mm (target: {pos_tolerance*1000:.3f}mm)")
        print(f"  Best orientation error: {final_ori_error:.4f} rad (target: {ori_tolerance:.4f} rad)")
        print(f"  Position {'✓' if final_pos_error < pos_tolerance else '✗'} | Orientation {'✓' if final_ori_error < ori_tolerance else '✗'}")
    else:
        print(f"  Best position error: {final_pos_error*1000:.3f}mm (target: {pos_tolerance*1000:.3f}mm)")
    
    return False, final_pos_error


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


if __name__ == "__main__":
    # Test the IK
    model = mujoco.MjModel.from_xml_path("ur5e_with_DIGIT.xml")
    data = mujoco.MjData(model)
    
    # Initialize
    mujoco.mj_forward(model, data)
    
    # Get current EE position
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
    current_pos = data.site_xpos[ee_site_id].copy()
    print(f"Current EE position: {current_pos}")
    
    # Target position
    target_pos = np.array([0.5, 0.1, 0.85])
    target_quat = np.array([0.0, 1.0, 0.0, 0.0])  # Gripper pointing down
    
    print(f"\n🎯 Target position: {target_pos}")
    
    # Perform IK with Levenberg-Marquardt
    success, error = move_to_target_pose(
        model, data, target_pos, target_quat,
        max_iterations=500, lambda_init=0.01
    )
    
    if success or error < 0.01:  # Accept if close enough
        print("\n✅ IK completed")
        stabilize_robot(model, data, num_steps=50)
        
        # Verify final position
        final_pos = data.site_xpos[ee_site_id].copy()
        print(f"Final EE position: {final_pos}")
        print(f"Final error: {np.linalg.norm(target_pos - final_pos)*1000:.2f}mm")
        
        print_joint_state(model, data)
    else:
        print("\n❌ IK failed")
