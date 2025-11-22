"""
Task Space Control Demo V2 - With DIGIT Sensor Data Tracking
• Uses original actuator-based control (stable with adjusted friction)
• Records DIGIT sensor data (2552×1 distance field per sensor)
• Saves timestamped sensor data to CSV files
• X,Y positions fixed in filtered_FEM_grid.csv, only distances saved
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import msvcrt
import csv
from datetime import datetime
import os

from simple_ik_legacy import move_to_target_pose, DEFAULT_INITIAL_JOINTS
from gripper_digit_sensor import GripperDIGITSensor

Initial_peg_position=[0.5, 0.2, 0.81] ###


class TaskSpaceControllerV2:
    """V2: Original actuator-based control + DIGIT sensor data tracking"""
    
    def __init__(self, xml_path="ur5e_with_DIGIT_primitive_hexagon.xml"):
        """Initialize controller with DIGIT sensors"""
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
            print("Peg not found")
        
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
        
        # Initialize DIGIT sensors
        try:
            self.digit_left = GripperDIGITSensor(self.model, "digit_geltip_left", "left")
            self.digit_right = GripperDIGITSensor(self.model, "digit_geltip_right", "right")
            self.has_digit_sensors = True
            
            # Load FEM grid for high-resolution data (2552 nodes)
            self.fem_grid = self.digit_left.load_fem_grid('filtered_FEM_grid.csv')
            if self.fem_grid is not None:
                print(f" FEM grid loaded: {len(self.fem_grid)} nodes")
            else:
                print(" FEM grid not loaded - will use sparse contact data only")
                
        except Exception as e:
            self.has_digit_sensors = False
            self.fem_grid = None
            print(f" DIGIT sensors not initialized: {e}")
        
        # Movement step size
        self.pos_step = 0.001  # 1mm
        self.rot_step = np.deg2rad(0.5)  # 0.5 degrees
        
        # Current target pose
        self.target_pos = None
        self.target_rpy = None
        
        # Gripper state
        self.gripper_value = 0.0
        self.gripper_step = 0.01
        
        # Cache last joint configuration
        self.last_joint_config = None
        
        # Data recording - ONE CSV file per session
        self.recording_enabled = False
        self.sensor_data_rows = []  # List of rows for final CSV
        self.output_dir = "Teleoperation_sensor_data"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate session filename
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_filename = os.path.join(self.output_dir, f"session_{self.session_timestamp}.csv")
        
        print(" Task Space Controller V2 initialized")
        print(f"   Position step: {self.pos_step*1000:.2f}mm")
        print(f"   Rotation step: {np.rad2deg(self.rot_step):.2f}°")
        print(f"   Gripper range: 0.0 to 1.6, step: {self.gripper_step}")
        print(f"   Session data will be saved to: {self.session_filename}")
    
    def set_initial_pose(self):
        """Set robot to initial joint configuration"""
        # First, set peg to known safe position BEFORE moving robot
        if self.has_peg:
            peg_qpos_addr = self.model.body_jntadr[self.peg_body_id]
            # Position: X=0.4, Y=0.2, Z=0.81 (AWAY from robot, on table surface)
            self.data.qpos[peg_qpos_addr:peg_qpos_addr+3] = Initial_peg_position   ### Initial peg position###
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
    
    def capture_sensor_data(self):
        """
        Capture current DIGIT sensor data from both sensors.
        Returns dict with timestamped data including joint angles.
        """
        if not self.has_digit_sensors:
            return None
        
        timestamp = self.data.time
        
        # Get contact data from both sensors
        left_contacts = self.digit_left.detect_proximity_contacts(self.data)
        right_contacts = self.digit_right.detect_proximity_contacts(self.data)
        
        # Capture joint angles (6 values in radians)
        joint_angles = np.array([self.data.qpos[jid] for jid in self.joint_ids])
        
        sensor_data = {
            'timestamp': timestamp,
            'gripper_value': self.gripper_value,
            'joint_angles': joint_angles,  # 6 joint angles (radians)
            'ee_position': self.data.site_xpos[self.ee_site_id].copy(),
            'left_sensor': {
                'num_contacts': len(left_contacts),
                'distance_field': None  # Will be 2552×1 array (just distances)
            },
            'right_sensor': {
                'num_contacts': len(right_contacts),
                'distance_field': None  # Will be 2552×1 array (just distances)
            }
        }
        
        # If FEM grid is available, interpolate to 2552 nodes
        if self.fem_grid is not None:
            # Left sensor: interpolate to FEM grid (only save distance values)
            if len(left_contacts) > 0:
                left_distance_field = self.digit_left.interpolate_to_fem_grid(
                    left_contacts, self.fem_grid, influence_radius_mm=0.2
                )
                sensor_data['left_sensor']['distance_field'] = left_distance_field
            else:
                # No contacts: zeros
                sensor_data['left_sensor']['distance_field'] = np.zeros(len(self.fem_grid))
            
            # Right sensor: interpolate to FEM grid (only save distance values)
            if len(right_contacts) > 0:
                right_distance_field = self.digit_right.interpolate_to_fem_grid(
                    right_contacts, self.fem_grid, influence_radius_mm=0.2
                )
                sensor_data['right_sensor']['distance_field'] = right_distance_field
            else:
                # No contacts: zeros
                sensor_data['right_sensor']['distance_field'] = np.zeros(len(self.fem_grid))
        
        return sensor_data
    
    def add_data_row(self, sensor_data):
        """
        Add sensor data to the list of rows for final CSV.
        Row format: [timestamp, gripper_value, joint1...joint6, left_sensor_0...left_sensor_2551, right_sensor_0...right_sensor_2551]
        Total columns: 1 + 1 + 6 + 2552 + 2552 = 5112 columns
        """
        if sensor_data is None:
            return
        
        if sensor_data['left_sensor']['distance_field'] is None:
            return
        
        # Build row: timestamp, gripper, 6 joints, 2552 left values, 2552 right values
        row = [sensor_data['timestamp'], sensor_data['gripper_value']]
        row.extend(sensor_data['joint_angles'])  # Add 6 joint angles
        row.extend(sensor_data['left_sensor']['distance_field'])  # Add 2552 left sensor values
        row.extend(sensor_data['right_sensor']['distance_field'])  # Add 2552 right sensor values
        
        self.sensor_data_rows.append(row)
    
    def save_session_to_csv(self):
        """
        Save all recorded data to ONE CSV file for the entire session.
        Format: Each row is one timestep with 5112 columns:
        - Column 0: timestamp (seconds)
        - Column 1: gripper_value (0.0-1.6)
        - Columns 2-7: joint angles (6 values, radians)
        - Columns 8-2559: left sensor distance field (2552 values, mm)
        - Columns 2560-5111: right sensor distance field (2552 values, mm)
        """
        if len(self.sensor_data_rows) == 0:
            print(" No data recorded in this session")
            return
        
        print(f"\n Saving session data...")
        print(f"   Rows: {len(self.sensor_data_rows)}")
        print(f"   Columns: 5112 (timestamp + gripper + 6 joints + 2×2552 sensors)")
        
        # Create header
        header = ['timestamp', 'gripper_value']
        header.extend([f'joint{i+1}_rad' for i in range(6)])
        header.extend([f'left_sensor_{i}' for i in range(2552)])
        header.extend([f'right_sensor_{i}' for i in range(2552)])
        
        # Write to CSV
        with open(self.session_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.sensor_data_rows)
        
        print(f" Session data saved: {self.session_filename}")
        print(f"   File size: {os.path.getsize(self.session_filename) / 1024 / 1024:.2f} MB")
    
    def toggle_recording(self):
        """Toggle sensor data recording"""
        self.recording_enabled = not self.recording_enabled
        status = "ENABLED" if self.recording_enabled else "DISABLED"
        print(f"\n Sensor Data Recording: {status}")
        
        if self.recording_enabled:
            self.sensor_data_rows.clear()
            print(f"   Data will be saved to: {self.session_filename}")
        else:
            if len(self.sensor_data_rows) > 0:
                print(f"   Captured {len(self.sensor_data_rows)} frames")
    
    def save_snapshot(self):
        """Save current sensor data as a single snapshot (manual capture)"""
        print("\n📸 Capturing sensor snapshot...")
        sensor_data = self.capture_sensor_data()
        if sensor_data is not None:
            self.add_data_row(sensor_data)
            print(f" Snapshot added to session data ({len(self.sensor_data_rows)} total frames)")
    
    def print_sensor_status(self):
        """Print current DIGIT sensor status"""
        if not self.has_digit_sensors:
            print(" DIGIT sensors not available")
            return
        
        sensor_data = self.capture_sensor_data()
        if sensor_data is None:
            return
        
        print(f"\n DIGIT Sensor Status (t={sensor_data['timestamp']:.3f}s):")
        print(f"   LEFT sensor:  {sensor_data['left_sensor']['num_contacts']} contacts")
        print(f"   RIGHT sensor: {sensor_data['right_sensor']['num_contacts']} contacts")
        print(f"   Gripper: {self.gripper_value:.3f}")
        print(f"   Joint angles (deg): [{', '.join([f'{np.rad2deg(a):.1f}' for a in sensor_data['joint_angles']])}]")
        
        if sensor_data['left_sensor']['distance_field'] is not None:
            left_max = np.max(sensor_data['left_sensor']['distance_field'])
            left_active = sensor_data['left_sensor']['distance_field'][sensor_data['left_sensor']['distance_field'] > 0]
            left_mean = np.mean(left_active) if len(left_active) > 0 else 0.0
            print(f"   LEFT max: {left_max:.3f}mm, mean (active): {left_mean:.3f}mm")
        
        if sensor_data['right_sensor']['distance_field'] is not None:
            right_max = np.max(sensor_data['right_sensor']['distance_field'])
            right_active = sensor_data['right_sensor']['distance_field'][sensor_data['right_sensor']['distance_field'] > 0]
            right_mean = np.mean(right_active) if len(right_active) > 0 else 0.0
            print(f"   RIGHT max: {right_max:.3f}mm, mean (active): {right_mean:.3f}mm")
    
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
        
        # Peg position, contacts, everything preserved
        
        # Calculate joint differences
        joint_diff = target_joints - current_joints
        max_diff = np.max(np.abs(joint_diff))
        
        if max_diff < 1e-6:
            return True  # Already at target
        
        # LIMIT maximum joint change per step to prevent large jumps
        MAX_JOINT_CHANGE = np.deg2rad(2.0)  # 2 degrees
        if max_diff > MAX_JOINT_CHANGE:
            # Scale down the joint change
            scale = MAX_JOINT_CHANGE / max_diff
            target_joints = current_joints + scale * joint_diff
            joint_diff = target_joints - current_joints
            max_diff = MAX_JOINT_CHANGE
        
        # More interpolation steps = smoother movement
        num_steps = max(100, int(max_diff * 100))
        num_steps = min(num_steps, 50)  # Cap at 50
        
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
                print(f"  PEG DROPPED! Distance: {dist_before*1000:.1f}mm → {dist_after*1000:.1f}mm")
        
        # CACHE final joint configuration for next iteration (prevents oscillation)
        self.last_joint_config = target_joints.copy()
        
        # AFTER interpolation, sync actuators to final position
        for j, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = target_joints[j]
        
        # Record sensor data if enabled
        if self.recording_enabled and self.has_digit_sensors:
            sensor_data = self.capture_sensor_data()
            if sensor_data is not None:
                self.add_data_row(sensor_data)
        
        return True
    
    def print_status(self):
        """Print current target pose with peg tracking"""
        ee_pos = self.data.site_xpos[self.ee_site_id]
        
        print(f"\n Target Pose:")
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
        
        # Print recording status
        if self.recording_enabled:
            print(f"   🔴 RECORDING: {len(self.sensor_data_rows)} frames captured")
    
    def run_manual_control(self):
        """Continuous key press control: hold keys → update target → IK → visualize"""
        self.set_initial_pose()
        
        # Stabilize physics simulation
        print("Stabilizing physics simulation...")
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
        print("✓ Physics stabilized")
        
        # Set target to position above peg
        mujoco.mj_forward(self.model, self.data)
        
        if self.has_peg:
            peg_pos = self.data.xpos[self.peg_body_id].copy()
            self.target_pos = peg_pos + np.array([0.0, 0.0, 0.05])  # 50mm above peg
            print(f"✓ Peg at: [{peg_pos[0]:.4f}, {peg_pos[1]:.4f}, {peg_pos[2]:.4f}]")
            print(f"✓ Target set 50mm above peg: [{self.target_pos[0]:.4f}, {self.target_pos[1]:.4f}, {self.target_pos[2]:.4f}]")
        else:
            current_pos = self.data.site_xpos[self.ee_site_id].copy()
            self.target_pos = current_pos.copy()
        
        self.target_rpy = np.array([0.0, np.pi, 0.0])  # Downward-facing
        
        # Initialize joint cache
        self.last_joint_config = np.array([self.data.qpos[jid] for jid in self.joint_ids])
        
        # Move to target position above peg
        print("Moving to position above peg...")
        if self.move_to_target():
            print("✓ Ready at position above peg")
        else:
            print("  Could not reach target position")
        
        print("\n" + "="*70)
        print("TASK SPACE CONTROLLER V2 - WITH DIGIT SENSOR DATA TRACKING")
        print("="*70)
        print("Controls (HOLD keys for continuous movement):")
        print("  Position: W/S (X±)  A/D (Y±)  Q/E (Z±)")
        print("  Rotation: I/K (Roll±)  J/L (Pitch±)  U/O (Yaw±)")
        print("  Gripper:  C (close +0.05)  V (open -0.05)  [Range: 0.0-1.6]")
        print("  Utility:  H (home)  P (print status)  T (test grasp)  X (exit)")
        print("  Sensor:   R (toggle recording)  M (save snapshot)  G (sensor status)")
        print("="*70 + "\n")
        
        self.print_status()
        
        # Launch passive viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("\n Viewer started. Press and HOLD keys in console!\n")
            
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
                    
                    # Gripper commands
                    elif key == 'c':
                        self.gripper_value = min(1.6, self.gripper_value + self.gripper_step)
                        if self.has_gripper:
                            for grip_id in self.gripper_actuator_ids:
                                self.data.ctrl[grip_id] = self.gripper_value
                        print(f"Gripper: {self.gripper_value:.2f}")
                    elif key == 'v':
                        self.gripper_value = max(0.0, self.gripper_value - self.gripper_step)
                        if self.has_gripper:
                            for grip_id in self.gripper_actuator_ids:
                                self.data.ctrl[grip_id] = self.gripper_value
                        print(f"Gripper: {self.gripper_value:.2f}")
                    
                    # Sensor data commands
                    elif key == 'r':
                        self.toggle_recording()
                    elif key == 'm':
                        # Save single snapshot (add to session data)
                        self.save_snapshot()
                    elif key == 'g':
                        # Print sensor status
                        self.print_sensor_status()
                    
                    # Utility commands
                    elif key == 'h':
                        self.target_pos = np.array([0.629, 0.0, 0.885])
                        self.target_rpy = np.array([0.0, np.pi, 0.0])
                        print("Home position (safe)")
                        moved = True
                    elif key == 'p':
                        self.print_status()
                    elif key == 't':
                        # Test grasp sequence
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
                            self.target_pos = peg_pos + np.array([0.0, 0.0, 0.005])
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
                            self.target_pos[2] += 0.010
                            self.move_to_target()
                        
                        self.print_status()
                        self.print_sensor_status()
                        
                    elif key == 'x':
                        print("Exiting...")
                        break
                
                # If pose changed, run IK
                if moved:
                    self.move_to_target()
                else:
                    # Maintain gripper during idle
                    if self.has_gripper:
                        for grip_id in self.gripper_actuator_ids:
                            self.data.ctrl[grip_id] = self.gripper_value
                    mujoco.mj_step(self.model, self.data)
                    
                    # Record sensor data if enabled (even during idle)
                    if self.recording_enabled and self.has_digit_sensors:
                        # Record every 10th frame to avoid too much data
                        if len(self.sensor_data_rows) == 0 or self.data.time - self.sensor_data_rows[-1][0] > 0.1:
                            sensor_data = self.capture_sensor_data()
                            if sensor_data is not None:
                                self.add_data_row(sensor_data)
                
                viewer.sync()
                time.sleep(0.01)
        
        print("\n✅ Controller V2 stopped")
        
        # Save session data to ONE CSV file on exit (X button pressed)
        if len(self.sensor_data_rows) > 0:
            self.save_session_to_csv()
        else:
            print("⚠️ No data recorded in this session")


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("TASK SPACE CONTROLLER V2 - WITH DIGIT SENSOR DATA TRACKING")
    print("="*70)
    print("• Uses actuator-based control (same as original - stable with friction)")
    print("• Records: timestamp, gripper, 6 joints, 2×2552 sensor values")
    print("• ONE CSV file per session: 5112 columns × N rows")
    print("• Press 'R' to toggle recording, 'M' for snapshot, 'X' to exit & save")
    print("="*70 + "\n")
    
    controller = TaskSpaceControllerV2()
    controller.run_manual_control()


if __name__ == "__main__":
    main()
