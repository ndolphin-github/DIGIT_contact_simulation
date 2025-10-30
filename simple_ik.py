"""
Simple and robust IK for UR5e robot
Completely rewritten from scratch for stability
"""

import numpy as np
import mujoco


def move_to_target_pose(model, data, target_pos, target_quat=None, 
                        max_iterations=500, step_size=0.1, 
                        pos_tolerance=5e-3, ori_tolerance=0.1,
                        initial_joints=None, joint_weight=0.01):
    """
    Move UR5e end effector to target position and orientation
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        target_pos: [x, y, z] target position in meters
        target_quat: [w, x, y, z] target quaternion (None = no orientation constraint)
        max_iterations: Maximum number of IK iterations
        step_size: Step size for gradient descent (smaller = more stable)
        pos_tolerance: Position error tolerance in meters
        ori_tolerance: Orientation error tolerance in radians
        initial_joints: Initial joint configuration to stay close to (6D array)
        joint_weight: Weight for staying close to initial_joints (higher = prefer initial pose)
    
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
    
    # Save initial joint configuration
    q_init = data.qpos[joint_ids].copy()
    
    # Use provided initial joints if given, otherwise use current
    if initial_joints is not None:
        q_preferred = np.array(initial_joints)
        print(f"✓ Using preferred joint configuration as starting point")
        data.qpos[joint_ids] = q_preferred
    else:
        q_preferred = q_init.copy()
    
    # IK loop
    best_error = float('inf')
    best_q = data.qpos[joint_ids].copy()
    
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
        
        # Check convergence
        if pos_error_norm < pos_tolerance and ori_error_norm < ori_tolerance:
            print(f"✓ IK converged in {iteration} iterations")
            print(f"  Position error: {pos_error_norm*1000:.2f}mm")
            if target_quat is not None:
                print(f"  Orientation error: {np.degrees(ori_error_norm):.2f}°")
            return True, pos_error_norm
        
        # Track best solution
        total_error = pos_error_norm + ori_error_norm
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
        
        # Compute joint update
        if target_quat is not None:
            # 6DOF: position + orientation
            J = np.vstack([J_pos, J_ori])
            error = np.concatenate([pos_error, ori_error])
            
            # Damped least squares
            damping = 1e-4
            dq = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6)) @ error
        else:
            # 3DOF: position only
            damping = 1e-4
            dq = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + damping * np.eye(3)) @ pos_error
        
        # Add joint configuration cost to prefer staying near initial pose
        if initial_joints is not None:
            q_current = data.qpos[joint_ids]
            joint_error = q_preferred - q_current
            dq += joint_weight * joint_error  # Pull towards preferred configuration
        
        # Update joint positions with step size
        for i, jid in enumerate(joint_ids):
            data.qpos[jid] += step_size * dq[i]
            
            # Joint limit clamping
            qpos_adr = model.jnt_qposadr[jid]
            joint_range = model.jnt_range[jid]
            data.qpos[jid] = np.clip(data.qpos[jid], joint_range[0], joint_range[1])
    
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
    
    # Perform IK
    success, error = move_to_target_pose(
        model, data, target_pos, target_quat,
        max_iterations=500, step_size=0.1
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
