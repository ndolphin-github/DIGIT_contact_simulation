"""
Example: Move UR5e robot to desired end effector positions using IK
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from move_to_ee_pose import move_to_ee_position, set_actuator_controls_to_current_pose

def main():
    print("🤖 UR5e End Effector Position Control Example")
    print("="*60)
    
    # Load the robot model
    model = mujoco.MjModel.from_xml_path("ur5e_with_DIGIT.xml")
    data = mujoco.MjData(model)
    
    # Get EE site ID for monitoring
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
    
    # Initialize
    mujoco.mj_forward(model, data)
    
    print(f"\n📍 Current EE position: {data.site_xpos[ee_site_id]}")
    
    # Example 1: Move to cube position
    print("\n" + "="*60)
    print("Example 1: Moving EE to cube position [-0.6, 0, 0.8]")
    print("="*60)
    
    target_pos_1 = np.array([-0.6, 0.0, 0.8])
    success, error = move_to_ee_position(model, data, target_pos_1)
    
    if success:
        # IMPORTANT: Set actuator controls to match the IK solution
        set_actuator_controls_to_current_pose(model, data)
        print(f"✅ Moved to: {data.site_xpos[ee_site_id]}")
        print(f"   Error: {error*1000:.3f}mm")
        
        # Print the joint angles you need to use
        print("\n📋 Joint angles to reach this position:")
        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                      "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        for name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            angle = data.qpos[joint_id]
            print(f"   {name:25s}: {angle:7.4f} rad ({np.degrees(angle):7.2f}°)")
    
    # Example 2: Move to different position
    print("\n" + "="*60)
    print("Example 2: Moving EE to position [-0.5, 0.1, 0.9]")
    print("="*60)
    
    target_pos_2 = np.array([-0.5, 0.1, 0.9])
    success, error = move_to_ee_position(model, data, target_pos_2)
    
    if success:
        set_actuator_controls_to_current_pose(model, data)
        print(f"✅ Moved to: {data.site_xpos[ee_site_id]}")
        print(f"   Error: {error*1000:.3f}mm")
    
    # Example 3: Interactive viewer with IK
    print("\n" + "="*60)
    print("Example 3: Launching interactive viewer")
    print("="*60)
    print("💡 You can modify the code to add keyboard controls for IK")
    
    # Reset to initial pose for viewer
    target_pos_1 = np.array([-0.6, 0.0, 0.8])
    move_to_ee_position(model, data, target_pos_1)
    set_actuator_controls_to_current_pose(model, data)
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.5
        viewer.cam.lookat = [-0.4, 0.0, 1.0]
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 135
        
        print("✓ Viewer launched - robot should be at EE position [-0.6, 0, 0.8]")
        print("  Close viewer to exit")
        
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
    
    print("\n✅ Demo complete!")
    print("\n💡 TIP: Copy the joint angles from above into ur5e_digit_demo.py")
    print("   to make the robot start at that position")

if __name__ == "__main__":
    main()
