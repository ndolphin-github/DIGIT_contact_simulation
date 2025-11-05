"""
Task Space Control Demo - Simple Keyboard Control
Command EEF pose (x, y, z, roll, pitch, yaw) via keyboard → IK → Visualize
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import msvcrt

from simple_ik_legacy import move_to_target_pose, DEFAULT_INITIAL_JOINTS


class TaskSpaceController:
    """Simple task space controller - keyboard → target pose → IK → visualize"""
    
    def __init__(self, xml_path="ur5e_with_DIGIT_primitive_hexagon.xml"):
        """Initialize controller"""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Get IDs
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
        
        # Get peg body ID
        try:
            self.peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hexagon_peg_body")
            self.has_peg = True
        except:
            self.has_peg = False
            print("⚠️ Peg not found in model")
        
        # Get arm joint IDs
        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                      "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        
        # Get actuator IDs
        actuator_names = ["shoulder_pan_actuator", "shoulder_lift_actuator", "elbow_actuator",
                         "wrist_1_actuator", "wrist_2_actuator", "wrist_3_actuator"]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in actuator_names]
        
        # Get gripper actuator IDs
        try:
            gripper_names = ["rh_p12_rn_right_actuator", "rh_p12_rn_left_actuator"]
            self.gripper_actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) 
                                         for n in gripper_names]
            self.has_gripper = True
            print(f"   Gripper actuator IDs: {self.gripper_actuator_ids}")
        except:
            self.has_gripper = False
            self.gripper_actuator_ids = []
        
        # Movement step size (더 작게 = 더 부드럽게)
        self.pos_step = 0.001  # 0.2mm (더 작은 스텝)
        self.rot_step = np.deg2rad(0.5)  # 0.5도
        
        # Current target pose - will be set after stabilization
        self.target_pos = None
        self.target_rpy = None
        
        # Gripper state
        self.gripper_value = 0.0
        self.gripper_step = 0.05
        
        # CRITICAL: Cache last successful joint configuration to avoid IK oscillation
        self.last_joint_config = None
        
        print("✅ Task Space Controller initialized")
        print(f"   Position step: {self.pos_step*1000:.2f}mm")
        print(f"   Rotation step: {np.rad2deg(self.rot_step):.2f}°")
        print(f"   Gripper range: 0.0 to 1.6, step: {self.gripper_step}")
    
    def set_initial_pose(self):
        """Set robot to initial joint configuration"""
        # First, set peg to known safe position BEFORE moving robot
        if self.has_peg:
            peg_qpos_addr = self.model.body_jntadr[self.peg_body_id]
            # Position: X=0.4, Y=0.2, Z=0.81 (AWAY from robot, on table surface)
            self.data.qpos[peg_qpos_addr:peg_qpos_addr+3] = [0.4, 0.2, 0.81]
            # Orientation: identity quaternion [w, x, y, z]
            self.data.qpos[peg_qpos_addr+3:peg_qpos_addr+7] = [1, 0, 0, 0]
            # Zero velocity
            peg_qvel_addr = self.model.body_dofadr[self.peg_body_id]
            self.data.qvel[peg_qvel_addr:peg_qvel_addr+6] = 0.0
        
        # Now set robot joints
        for i, jid in enumerate(self.joint_ids):
            self.data.qpos[jid] = DEFAULT_INITIAL_JOINTS[i]
            self.data.ctrl[self.actuator_ids[i]] = DEFAULT_INITIAL_JOINTS[i]
            self.data.qvel[jid] = 0.0  # Zero velocity
        
        if self.has_gripper:
            for grip_id in self.gripper_actuator_ids:
                self.data.ctrl[grip_id] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
    
    def rpy_to_quat(self, roll, pitch, yaw):
        """Convert roll-pitch-yaw to quaternion [w, x, y, z]"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    def move_to_target(self):
        """Move to current target pose - IK on SEPARATE data to preserve contacts!"""
        # Convert RPY to quaternion
        target_quat = self.rpy_to_quat(self.target_rpy[0], self.target_rpy[1], self.target_rpy[2])
        
        # CRITICAL: Use cached joint config if available (prevents IK oscillation)
        if self.last_joint_config is not None:
            current_joints = self.last_joint_config.copy()
        else:
            # First time: get from actuators
            current_joints = np.array([self.data.ctrl[aid] for aid in self.actuator_ids])
        
        # **CREATE SEPARATE MjData FOR IK CALCULATION!**
        # This way IK doesn't touch the real simulation state at all!
        ik_data = mujoco.MjData(self.model)
        
        # Copy only joint positions for IK
        for j, jid in enumerate(self.joint_ids):
            ik_data.qpos[jid] = current_joints[j]
        mujoco.mj_forward(self.model, ik_data)
        
        # Run IK on the SEPARATE data
        success, error = move_to_target_pose(
            self.model, ik_data, self.target_pos, target_quat,
            max_iterations=500,
            lambda_init=0.01,
            pos_tolerance=0.0001,
            ori_tolerance=0.0001,
            initial_joints=current_joints
        )
        
        if not success and error > 0.010:
            return False
        
        # Get target joints from IK solution (from SEPARATE data!)
        target_joints = np.array([ik_data.qpos[jid] for jid in self.joint_ids])
        
        # Original simulation data is COMPLETELY UNTOUCHED!
        # Peg position, contacts, everything preserved!
        
        # Calculate joint differences
        joint_diff = target_joints - current_joints
        max_diff = np.max(np.abs(joint_diff))
        
        if max_diff < 1e-6:
            return True  # Already at target
        
        # LIMIT maximum joint change per step to prevent large jumps
        MAX_JOINT_CHANGE = np.deg2rad(2.0)  # 2도로 줄임 (더 부드럽게)
        if max_diff > MAX_JOINT_CHANGE:
            # Scale down the joint change
            scale = MAX_JOINT_CHANGE / max_diff
            target_joints = current_joints + scale * joint_diff
            joint_diff = target_joints - current_joints
            max_diff = MAX_JOINT_CHANGE
        
        # 더 많은 interpolation steps = 더 부드러운 움직임 (V2는 500-1000!)
        num_steps = max(100, int(max_diff * 500))  # 훨씬 더 많은 스텝
        num_steps = min(num_steps, 500)  # Cap을 500으로 증가 (V2 스타일)
        
        # Track peg position before movement
        peg_held_before = False
        if self.has_peg:
            ee_pos_before = self.data.site_xpos[self.ee_site_id].copy()
            peg_pos_before = self.data.xpos[self.peg_body_id].copy()
            dist_before = np.linalg.norm(peg_pos_before - ee_pos_before)
            peg_held_before = dist_before < 0.050
        
        for step in range(num_steps):
            # Sinusoidal easing for smooth motion
            t = (step + 1) / num_steps
            alpha = (1 - np.cos(t * np.pi)) / 2  # Smooth S-curve
            
            interp_joints = current_joints + alpha * joint_diff
            
            # GUI SLIDER METHOD: Set actuator CTRL, not qpos!
            # Let the actuators move the robot naturally (like GUI)
            for j, aid in enumerate(self.actuator_ids):
                self.data.ctrl[aid] = interp_joints[j]
            
            # MAINTAIN GRIPPER POSITION
            if self.has_gripper:
                for grip_id in self.gripper_actuator_ids:
                    self.data.ctrl[grip_id] = self.gripper_value
            
            # Multiple physics steps - actuators drive the motion
            for _ in range(5):  # More steps for actuator-based control
                mujoco.mj_step(self.model, self.data)
            
            # Forward kinematics to update state
            mujoco.mj_forward(self.model, self.data)
        
        # Check if peg was dropped during movement
        if self.has_peg and peg_held_before:
            ee_pos_after = self.data.site_xpos[self.ee_site_id].copy()
            peg_pos_after = self.data.xpos[self.peg_body_id].copy()
            dist_after = np.linalg.norm(peg_pos_after - ee_pos_after)
            if dist_after > 0.050:
                print(f"⚠️  PEG DROPPED! Distance: {dist_before*1000:.1f}mm → {dist_after*1000:.1f}mm")
        
        # CACHE final joint configuration for next iteration (prevents oscillation)
        self.last_joint_config = target_joints.copy()
        
        # AFTER interpolation, sync actuators to final position
        for j, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = target_joints[j]
        
        return True
    
    def print_status(self):
        """Print current target pose with peg tracking"""
        ee_pos = self.data.site_xpos[self.ee_site_id]
        
        print(f"\n📊 Target Pose:")
        print(f"   Position: X={self.target_pos[0]:.4f} Y={self.target_pos[1]:.4f} Z={self.target_pos[2]:.4f}")
        print(f"   EE Actual: X={ee_pos[0]:.4f} Y={ee_pos[1]:.4f} Z={ee_pos[2]:.4f}")
        print(f"   Rotation: R={np.rad2deg(self.target_rpy[0]):.1f}° P={np.rad2deg(self.target_rpy[1]):.1f}° Y={np.rad2deg(self.target_rpy[2]):.1f}°")
        print(f"   Gripper: {self.gripper_value:.1f}")
        
        # Print peg position and grasp status
        if self.has_peg:
            peg_pos = self.data.xpos[self.peg_body_id]
            peg_ee_dist = np.linalg.norm(peg_pos - ee_pos)
            grasp_status = "HELD" if peg_ee_dist < 0.050 else "DROPPED"
            print(f"   Peg Position: X={peg_pos[0]:.4f} Y={peg_pos[1]:.4f} Z={peg_pos[2]:.4f}")
            print(f"   Peg-EE Distance: {peg_ee_dist*1000:.1f}mm [{grasp_status}]")
    
    def run_manual_control(self):
        """Continuous key press control: hold keys → update target → IK → visualize"""
        self.set_initial_pose()
        
        # Stabilize physics simulation BEFORE any movement
        print("Stabilizing physics simulation...")
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
        print("✓ Physics stabilized")
        
        # NOW set target to current EE position (don't move yet)
        mujoco.mj_forward(self.model, self.data)
        
        # Get peg position and set EE target 50mm above it
        if self.has_peg:
            peg_pos = self.data.xpos[self.peg_body_id].copy()
            self.target_pos = peg_pos + np.array([0.0, 0.0, 0.03])  # 50mm above peg
            print(f"✓ Peg at: [{peg_pos[0]:.4f}, {peg_pos[1]:.4f}, {peg_pos[2]:.4f}]")
            print(f"✓ Target set 50mm above peg: [{self.target_pos[0]:.4f}, {self.target_pos[1]:.4f}, {self.target_pos[2]:.4f}]")
        else:
            # No peg, use current position
            current_pos = self.data.site_xpos[self.ee_site_id].copy()
            self.target_pos = current_pos.copy()
            print(f"✓ Starting at current position: [{self.target_pos[0]:.4f}, {self.target_pos[1]:.4f}, {self.target_pos[2]:.4f}]")
        
        self.target_rpy = np.array([0.0, np.pi, 0.0])  # Downward-facing
        
        # Initialize joint cache with current configuration
        self.last_joint_config = np.array([self.data.qpos[jid] for jid in self.joint_ids])
        
        # Now move to target position above peg
        print("Moving to position above peg...")
        if self.move_to_target():
            print("✓ Ready at position above peg")
        else:
            print("⚠️  Could not reach target position")
        
        print(f"✓ Starting at current position: [{self.target_pos[0]:.4f}, {self.target_pos[1]:.4f}, {self.target_pos[2]:.4f}]")
        
        print("\n" + "="*70)
        print("CONTINUOUS KEY PRESS CONTROL")
        print("="*70)
        print("Controls (HOLD keys for continuous movement):")
        print("  Position: W/S (X±)  A/D (Y±)  Q/E (Z±)")
        print("  Rotation: I/K (Roll±)  J/L (Pitch±)  U/O (Yaw±)")
        print("  Gripper:  C (close +0.1)  V (open -0.1)  [Range: 0.0-1.6]")
        print("  Utility:  H (home)  P (print status)  T (test grasp)  X (exit)")
        print("="*70 + "\n")
        
        self.print_status()
        
        # Launch passive viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("\n✅ Viewer started. Press and HOLD keys in console!\n")
            
            while viewer.is_running():
                moved = False
                
                # Check for key press (non-blocking)
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                    
                    # Position commands
                    if key == 'w':
                        self.target_pos[0] += self.pos_step
                        moved = True
                    elif key == 's':
                        self.target_pos[0] -= self.pos_step
                        moved = True
                    elif key == 'a':
                        self.target_pos[1] -= self.pos_step
                        moved = True
                    elif key == 'd':
                        self.target_pos[1] += self.pos_step
                        moved = True
                    elif key == 'q':
                        self.target_pos[2] += self.pos_step
                        moved = True
                    elif key == 'e':
                        self.target_pos[2] -= self.pos_step
                        moved = True
                    
                    # Rotation commands
                    elif key == 'i':
                        self.target_rpy[0] += self.rot_step
                        moved = True
                    elif key == 'k':
                        self.target_rpy[0] -= self.rot_step
                        moved = True
                    elif key == 'j':
                        self.target_rpy[1] -= self.rot_step
                        moved = True
                    elif key == 'l':
                        self.target_rpy[1] += self.rot_step
                        moved = True
                    elif key == 'u':
                        self.target_rpy[2] -= self.rot_step
                        moved = True
                    elif key == 'o':
                        self.target_rpy[2] += self.rot_step
                        moved = True
                    
                    # Gripper commands (continuous 0.0 to 1.6)
                    elif key == 'c':
                        self.gripper_value = min(1.6, self.gripper_value + self.gripper_step)
                        if self.has_gripper:
                            for grip_id in self.gripper_actuator_ids:
                                self.data.ctrl[grip_id] = self.gripper_value
                        print(f"Gripper: {self.gripper_value:.1f}")
                    elif key == 'v':
                        self.gripper_value = max(0.0, self.gripper_value - self.gripper_step)
                        if self.has_gripper:
                            for grip_id in self.gripper_actuator_ids:
                                self.data.ctrl[grip_id] = self.gripper_value
                        print(f"Gripper: {self.gripper_value:.1f}")
                    
                    # Utility commands
                    elif key == 'h':
                        self.target_pos = np.array([0.629, 0.0, 0.885])
                        self.target_rpy = np.array([0.0, np.pi, 0.0])
                        print("Home position (safe)")
                        moved = True
                    elif key == 'p':
                        self.print_status()
                    elif key == 't':
                        # Test grasp: close gripper on peg without moving
                        print("\n🧪 Testing grasp...")
                        print("  1. Opening gripper...")
                        self.gripper_value = 0.0
                        for _ in range(50):
                            if self.has_gripper:
                                for grip_id in self.gripper_actuator_ids:
                                    self.data.ctrl[grip_id] = 0.0
                            mujoco.mj_step(self.model, self.data)
                            viewer.sync()
                        
                        print("  2. Descending to peg...")
                        if self.has_peg:
                            peg_pos = self.data.xpos[self.peg_body_id].copy()
                            self.target_pos = peg_pos + np.array([0.0, 0.0, 0.005])  # 5mm above
                            self.move_to_target()
                        
                        print("  3. Closing gripper to 1.0...")
                        self.gripper_value = 1.0
                        for _ in range(100):
                            if self.has_gripper:
                                for grip_id in self.gripper_actuator_ids:
                                    self.data.ctrl[grip_id] = 1.0
                            mujoco.mj_step(self.model, self.data)
                            viewer.sync()
                        
                        print("  4. Testing lift (10mm up)...")
                        if self.has_peg:
                            self.target_pos[2] += 0.010  # Lift 10mm
                            self.move_to_target()
                        
                        # Check result
                        self.print_status()
                        
                    elif key == 'x':
                        print("Exiting...")
                        break
                
                # If pose changed, run IK
                if moved:
                    self.move_to_target()
                    # move_to_target() already does physics steps with gripper maintained
                    # NO extra mj_step() needed here!
                else:
                    # Only step simulation if no movement command
                    # MAINTAIN GRIPPER even during idle steps
                    if self.has_gripper:
                        for grip_id in self.gripper_actuator_ids:
                            self.data.ctrl[grip_id] = self.gripper_value
                    mujoco.mj_step(self.model, self.data)
                
                viewer.sync()
                time.sleep(0.01)
        
        print("\n✅ Controller stopped")


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("CONTINUOUS KEY PRESS CONTROL")
    print("="*70)
    print("• HOLD keys to continuously control EEF pose")
    print("• Gripper: 0.0 to 1.6 with 0.1 resolution (C/V keys)")
    print("• IK updates automatically, visualization in real-time")
    print("="*70 + "\n")
    
    controller = TaskSpaceController()
    controller.run_manual_control()


if __name__ == "__main__":
    main()
