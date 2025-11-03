"""
Helper functions to move UR5e robot to desired end effector pose using MuJoCo IK
"""

import numpy as np
import mujoco


def move_to_ee_pose_with_orientation(model, data, target_pos, target_quat=None, max_iterations=100, tolerance_pos=1e-3, tolerance_ori=1e-2):

    
    # Get end effector site ID
    try:
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
    except:
        print("❌ Error: 'eef_site' not found in model")
        return False, float('inf')
    
    # If no target orientation specified, use downward-facing gripper (z-axis down)
    if target_quat is None:
        # Quaternion for gripper pointing straight down (identity rotation then rotate to point down)
        target_quat = np.array([0.7071, 0.7071, 0, 0])  # 90° rotation around X-axis to point down
    
    # Get joint IDs for the UR5e arm (6 DOF)
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
            joint_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        except:
            print(f"❌ Error: Joint '{name}' not found")
            return False, float('inf')
    
    # IK loop
    for iteration in range(max_iterations):
        # Forward kinematics to get current EE pose
        mujoco.mj_forward(model, data)
        current_pos = data.site_xpos[ee_site_id].copy()
        current_mat = data.site_xmat[ee_site_id].reshape(3, 3).copy()
        
        # Calculate position error
        pos_error = target_pos - current_pos
        pos_error_norm = np.linalg.norm(pos_error)
        
        # Calculate orientation error using rotation matrix
        target_mat = np.zeros((3, 3))
        mujoco.mju_quat2Mat(target_mat.flatten(), target_quat)
        target_mat = target_mat.reshape(3, 3)
        
        # Orientation error: difference between current and target rotation matrices
        # Use axis-angle representation
        error_mat = target_mat @ current_mat.T
        ori_error = np.array([
            error_mat[2, 1] - error_mat[1, 2],
            error_mat[0, 2] - error_mat[2, 0],
            error_mat[1, 0] - error_mat[0, 1]
        ]) * 0.5
        ori_error_norm = np.linalg.norm(ori_error)
        
        # Check if converged
        if pos_error_norm < tolerance_pos and ori_error_norm < tolerance_ori:
            print(f"✓ IK converged in {iteration} iterations, pos_error = {pos_error_norm*1000:.3f}mm, ori_error = {np.degrees(ori_error_norm):.2f}°")
            return True, pos_error_norm
        
        # Compute Jacobian for the end effector site
        jacp = np.zeros((3, model.nv))  # Position jacobian
        jacr = np.zeros((3, model.nv))  # Rotation jacobian
        mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
        
        # Extract columns for the arm joints only
        jacp_arm = jacp[:, joint_ids]
        jacr_arm = jacr[:, joint_ids]
        
        # Combine position and orientation Jacobians
        jac_full = np.vstack([jacp_arm, jacr_arm])  # 6x6 matrix
        error_full = np.concatenate([pos_error, ori_error])  # 6D error vector
        
        # Compute pseudo-inverse with damping
        damping = 1e-4
        jac_pinv = jac_full.T @ np.linalg.inv(jac_full @ jac_full.T + damping * np.eye(6))
        
        # Compute joint velocity
        dq = jac_pinv @ error_full
        
        # Update joint positions with step size
        step_size = 0.3  # Smaller step for stability with orientation
        for i, joint_id in enumerate(joint_ids):
            data.qpos[joint_id] += step_size * dq[i]
    
    # Did not converge
    mujoco.mj_forward(model, data)
    current_pos = data.site_xpos[ee_site_id].copy()
    final_error = np.linalg.norm(target_pos - current_pos)
    print(f"⚠️  IK did not converge after {max_iterations} iterations, pos_error = {final_error*1000:.3f}mm")
    
    return False, final_error


def move_to_ee_position(model, data, target_pos, max_iterations=100, tolerance=1e-3):
    """
    Move robot end effector to target position using MuJoCo's inverse kinematics
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        target_pos: Target position [x, y, z] in world coordinates
        max_iterations: Maximum IK iterations
        tolerance: Position error tolerance in meters
    
    Returns:
        success: True if target reached within tolerance
        final_error: Final position error
    """
    
    # Get end effector site ID
    try:
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
    except:
        print("❌ Error: 'eef_site' not found in model")
        return False, float('inf')
    
    # Get joint IDs for the UR5e arm (6 DOF)
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
            joint_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        except:
            print(f"❌ Error: Joint '{name}' not found")
            return False, float('inf')
    
    # IK loop
    for iteration in range(max_iterations):
        # Forward kinematics to get current EE position
        mujoco.mj_forward(model, data)
        current_pos = data.site_xpos[ee_site_id].copy()
        
        # Calculate position error
        error = target_pos - current_pos
        error_norm = np.linalg.norm(error)
        
        # Check if converged
        if error_norm < tolerance:
            print(f"✓ IK converged in {iteration} iterations, error = {error_norm*1000:.3f}mm")
            return True, error_norm
        
        # Compute Jacobian for the end effector site
        jacp = np.zeros((3, model.nv))  # Position jacobian
        jacr = np.zeros((3, model.nv))  # Rotation jacobian
        mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
        
        # Extract columns for the arm joints only
        jacp_arm = jacp[:, joint_ids]
        
        # Compute pseudo-inverse
        # Add damping for numerical stability
        damping = 1e-4
        jacp_pinv = jacp_arm.T @ np.linalg.inv(jacp_arm @ jacp_arm.T + damping * np.eye(3))
        
        # Compute joint velocity
        dq = jacp_pinv @ error
        
        # Update joint positions with step size
        step_size = 0.5
        for i, joint_id in enumerate(joint_ids):
            data.qpos[joint_id] += step_size * dq[i]
    
    # Did not converge
    mujoco.mj_forward(model, data)
    current_pos = data.site_xpos[ee_site_id].copy()
    final_error = np.linalg.norm(target_pos - current_pos)
    print(f"⚠️  IK did not converge after {max_iterations} iterations, error = {final_error*1000:.3f}mm")
    
    return False, final_error


def set_actuator_controls_to_current_pose(model, data):
    """
    Set all actuator controls to match current joint positions
    This makes the robot hold its current pose
    Also resets velocities to prevent drift
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
    
    for joint_name, actuator_name in zip(joint_names, actuator_names):
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            
            # Set control to current position
            data.ctrl[actuator_id] = data.qpos[joint_id]
            
            # Reset velocity to zero to prevent drift
            data.qvel[joint_id] = 0.0
            
        except Exception as e:
            print(f"Warning: Could not set control for {joint_name}: {e}")
            pass
    
    # Apply the controls for a few steps to stabilize
    for _ in range(10):
        mujoco.mj_step(model, data)


# Example usage
if __name__ == "__main__":
    import mujoco.viewer
    
    # Load model
    model = mujoco.MjModel.from_xml_path("ur5e_with_DIGIT.xml")
    data = mujoco.MjData(model)
    
    # Set initial pose
    mujoco.mj_forward(model, data)
    
    # Get current EE position
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
    current_ee_pos = data.site_xpos[ee_site_id].copy()
    print(f"Current EE position: {current_ee_pos}")
    
    # Define target position (example: move to cube position)
    target_pos = np.array([-0.6, 0.0, 0.8])
    
    print(f"\n🎯 Moving to target: {target_pos}")
    
    # Perform IK
    success, error = move_to_ee_position(model, data, target_pos)
    
    if success:
        # Set actuator controls to hold this position
        set_actuator_controls_to_current_pose(model, data)
        
        # Verify final position
        mujoco.mj_forward(model, data)
        final_pos = data.site_xpos[ee_site_id].copy()
        print(f"Final EE position: {final_pos}")
        
        # Print joint angles
        print("\nJoint angles (radians):")
        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                      "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        for name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            angle = data.qpos[joint_id]
            print(f"  {name}: {angle:.4f} rad ({np.degrees(angle):.2f}°)")
    else:
        print("❌ IK failed to reach target")
