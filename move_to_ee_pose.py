"""
Helper functions to move UR5e robot to desired end effector pose using MuJoCo IK
"""

import numpy as np
import mujoco


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
            data.ctrl[actuator_id] = data.qpos[joint_id]
        except:
            pass


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
