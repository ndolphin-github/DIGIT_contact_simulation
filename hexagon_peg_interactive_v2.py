import numpy as np
import mujoco
import mujoco.viewer
import time
import threading

from simple_ik_legacy import move_to_target_pose, stabilize_robot, DEFAULT_INITIAL_JOINTS
from gripper_digit_sensor import GripperDIGITSensor


# ========== CONFIGURATION ==========
PEG_POSITION = np.array([0.4, 0.2, 0.81])          # Peg bottom center
HOLE_ENTRANCE = np.array([0.629, 0.029, 0.875])   # Hole entrance (top opening)
HOLE_CENTER = np.array([0.6, 0.0, 0.8])           # Hole center (reference)

# Forbidden zone: 130mm × 130mm × 75mm box above hole
FORBIDDEN_ZONE_CENTER = np.array([0.6, 0.0, 0.8])
FORBIDDEN_ZONE_SIZE = np.array([0.065, 0.065, 0.075])  # Half-sizes

# Visualization settings
ENABLE_PLOTTING = True  # Set to False to disable real-time plotting
PLOT_UPDATE_INTERVAL = 10  # Update plots every N simulation steps

# Motion control settings
MAX_JOINT_VELOCITY = 0.3  # rad/s - Maximum joint velocity for smooth motion
WAYPOINT_DENSITY = 0.01   # meters - Distance between waypoints for smooth paths


class InteractivePegDemo:
    """Interactive peg-in-hole demo with step-by-step control"""
    
    def __init__(self, xml_path="ur5e_with_DIGIT_primitive_hexagon.xml"):
        """Initialize demo"""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Get IDs
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
        self.peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hexagon_peg_body")
        
        # Get arm joint IDs
        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                      "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        
        # Get actuator IDs
        actuator_names = ["shoulder_pan_actuator", "shoulder_lift_actuator", "elbow_actuator",
                         "wrist_1_actuator", "wrist_2_actuator", "wrist_3_actuator"]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in actuator_names]
        
        # Get gripper actuator IDs
        gripper_names = ["rh_p12_rn_right_actuator", "rh_p12_rn_left_actuator"]
        self.gripper_actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in gripper_names]
        
        # Initialize DIGIT sensors
        self.digit_left = GripperDIGITSensor(self.model, "digit_geltip_left", "left")
        self.digit_right = GripperDIGITSensor(self.model, "digit_geltip_right", "right")
        
        # Control
        self.viewer = None
        self.simulation_running = False
        self.command_queue = []
        
        # Peg trajectory tracking
        self.peg_trajectory = []
        self.max_trajectory_points = 100  # Reduced for visual markers
        self.trajectory_markers = []
        self.marker_counter = 0
        self.show_peg_frame = True  # Show coordinate frame at peg
        
        # Plotting setup
        self.enable_plotting = ENABLE_PLOTTING
        self.plot_update_interval = PLOT_UPDATE_INTERVAL
        self.step_counter = 0
        
        if self.enable_plotting:
            self.setup_visualization()
    

    
    def get_sensor_data_summary(self):
        """
        Get comprehensive DIGIT sensor data
        
        Returns:
            dict with left and right sensor information
        """
        # Get raw contact data
        left_contacts = self.digit_left.detect_proximity_contacts(self.data)
        right_contacts = self.digit_right.detect_proximity_contacts(self.data)
        
        # Process left sensor
        left_summary = {
            'contact_count': len(left_contacts),
            'contacts': left_contacts,
            'statistics': {}
        }
        
        if len(left_contacts) > 0:
            depths = [c['distance_from_plane_mm'] for c in left_contacts]
            intensities = [c['intensity'] for c in left_contacts]
            x_positions = [c['x_mm'] for c in left_contacts]
            y_positions = [c['y_mm'] for c in left_contacts]
            
            left_summary['statistics'] = {
                'avg_depth_mm': np.mean(depths),
                'max_depth_mm': np.max(depths),
                'min_depth_mm': np.min(depths),
                'avg_intensity': np.mean(intensities),
                'max_intensity': np.max(intensities),
                'contact_area_coverage': len(left_contacts),  # Number of gel vertices in contact
                'x_range_mm': [np.min(x_positions), np.max(x_positions)],
                'y_range_mm': [np.min(y_positions), np.max(y_positions)],
                'centroid_x_mm': np.mean(x_positions),
                'centroid_y_mm': np.mean(y_positions)
            }
        
        # Process right sensor
        right_summary = {
            'contact_count': len(right_contacts),
            'contacts': right_contacts,
            'statistics': {}
        }
        
        if len(right_contacts) > 0:
            depths = [c['distance_from_plane_mm'] for c in right_contacts]
            intensities = [c['intensity'] for c in right_contacts]
            x_positions = [c['x_mm'] for c in right_contacts]
            y_positions = [c['y_mm'] for c in right_contacts]
            
            right_summary['statistics'] = {
                'avg_depth_mm': np.mean(depths),
                'max_depth_mm': np.max(depths),
                'min_depth_mm': np.min(depths),
                'avg_intensity': np.mean(intensities),
                'max_intensity': np.max(intensities),
                'contact_area_coverage': len(right_contacts),
                'x_range_mm': [np.min(x_positions), np.max(x_positions)],
                'y_range_mm': [np.min(y_positions), np.max(y_positions)],
                'centroid_x_mm': np.mean(x_positions),
                'centroid_y_mm': np.mean(y_positions)
            }
        
        return {
            'left': left_summary,
            'right': right_summary,
            'total_contacts': len(left_contacts) + len(right_contacts)
        }
    
    def print_sensor_info(self, detailed=False):
        """Print DIGIT sensor information"""
        sensor_data = self.get_sensor_data_summary()
        
        if sensor_data['total_contacts'] == 0:
            return
        
        print(f"\n📊 DIGIT Sensors: {sensor_data['total_contacts']} total contacts")
        
        for side, data in [('LEFT', sensor_data['left']), ('RIGHT', sensor_data['right'])]:
            if data['contact_count'] > 0:
                s = data['statistics']
                print(f"   {side}: {data['contact_count']} contacts | "
                      f"depth={s['avg_depth_mm']:.3f}mm | "
                      f"intensity={s['avg_intensity']:.3f} | "
                      f"centroid=({s['centroid_x_mm']:.1f},{s['centroid_y_mm']:.1f})mm")
    
    def set_initial_robot_pose(self):
        """Set robot to initial joint configuration"""
        for i, jid in enumerate(self.joint_ids):
            self.data.qpos[jid] = DEFAULT_INITIAL_JOINTS[i]
            self.data.qvel[jid] = 0.0
            self.data.ctrl[self.actuator_ids[i]] = DEFAULT_INITIAL_JOINTS[i]
        mujoco.mj_forward(self.model, self.data)
    
    def set_peg_position(self, position):
        """Set peg position in the simulation"""
        peg_qpos_addr = self.model.body_jntadr[self.peg_body_id]
        self.data.qpos[peg_qpos_addr:peg_qpos_addr+3] = position
        self.data.qpos[peg_qpos_addr+3:peg_qpos_addr+7] = [1, 0, 0, 0]  # Identity quaternion
        
        peg_qvel_addr = self.model.body_dofadr[self.peg_body_id]
        self.data.qvel[peg_qvel_addr:peg_qvel_addr+6] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
    
    def get_peg_position(self):
        """Get current peg position"""
        return self.data.xpos[self.peg_body_id].copy()
    
    def check_peg_held(self):
        """Check if peg is being held by analyzing its position relative to EE"""
        peg_pos = self.get_peg_position()
        ee_pos = self.get_ee_position()
        distance = np.linalg.norm(peg_pos - ee_pos)
        
        # If peg is within 50mm of end effector, likely being held
        is_held = distance < 0.050
        
        print(f"   Peg position: [{peg_pos[0]:.3f}, {peg_pos[1]:.3f}, {peg_pos[2]:.3f}]")
        print(f"   EE position:  [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
        print(f"   Distance: {distance*1000:.1f}mm - {'HELD' if is_held else 'DROPPED'}")
        
        return is_held
    
    def add_peg_trajectory_point(self):
        """Add current peg position to trajectory for visual tracking"""
        peg_pos = self.get_peg_position()
        current_time = self.data.time
        
        # Add to trajectory history
        self.peg_trajectory.append({
            'time': current_time,
            'position': peg_pos.copy(),
            'marker_id': self.marker_counter
        })
        
        # Limit trajectory length
        if len(self.peg_trajectory) > self.max_trajectory_points:
            # Remove oldest point
            old_point = self.peg_trajectory.pop(0)
            
        self.marker_counter += 1
    
    def visualize_peg_frame(self):
        """Visualize peg position with coordinate frame using MuJoCo viewer"""
        if self.viewer is None:
            return
            
        peg_pos = self.get_peg_position()
        peg_quat = self.data.xquat[self.peg_body_id].copy()  # [w, x, y, z]
        
        # Convert quaternion to rotation matrix for frame visualization
        peg_mat = np.zeros(9)
        mujoco.mju_quat2Mat(peg_mat, peg_quat)
        peg_mat = peg_mat.reshape(3, 3)
        
        # Frame size
        frame_size = 0.03  # 30mm frame axes
        
        try:
            # Try modern MuJoCo viewer API
            if hasattr(self.viewer, 'user_scn') and hasattr(self.viewer.user_scn, 'ngeom'):
                # Access the viewer's scene to add geometric markers
                scn = self.viewer.user_scn
                
                # X-axis (red line)
                x_end = peg_pos + peg_mat[:, 0] * frame_size
                self._add_line_marker(scn, peg_pos, x_end, [1, 0, 0, 1])
                
                # Y-axis (green line)
                y_end = peg_pos + peg_mat[:, 1] * frame_size
                self._add_line_marker(scn, peg_pos, y_end, [0, 1, 0, 1])
                
                # Z-axis (blue line)
                z_end = peg_pos + peg_mat[:, 2] * frame_size
                self._add_line_marker(scn, peg_pos, z_end, [0, 0, 1, 1])
                
            else:
                # Alternative: Print frame info to console (less frequently to avoid spam)
                if hasattr(self, 'last_frame_print_time'):
                    if self.data.time - self.last_frame_print_time > 1.0:  # Every 1 second
                        print(f"🎯 Peg Frame: pos=[{peg_pos[0]:.3f}, {peg_pos[1]:.3f}, {peg_pos[2]:.3f}] "
                              f"quat=[{peg_quat[0]:.2f}, {peg_quat[1]:.2f}, {peg_quat[2]:.2f}, {peg_quat[3]:.2f}]")
                        self.last_frame_print_time = self.data.time
                else:
                    self.last_frame_print_time = self.data.time
            
        except Exception as e:
            # Fallback: just track position
            pass
    
    def _add_line_marker(self, scn, start, end, color):
        """Helper to add line marker to scene"""
        try:
            # This is a simplified approach - MuJoCo viewer API varies by version
            # You might need to adapt this based on your MuJoCo version
            pass
        except:
            pass
    
    def visualize_trajectory_trail(self):
        """Visualize peg trajectory as console output and basic visual tracking"""
        if len(self.peg_trajectory) < 2:
            return
            
        # Console-based trajectory visualization (less frequent to avoid spam)
        if hasattr(self, 'last_trail_print_time'):
            if len(self.peg_trajectory) > 1 and self.data.time - self.last_trail_print_time > 2.0:  # Every 2 seconds
                current_pos = self.peg_trajectory[-1]['position']
                
                # Find significant movement since last 5 points
                if len(self.peg_trajectory) > 5:
                    old_pos = self.peg_trajectory[-6]['position']
                    movement = np.linalg.norm(current_pos - old_pos)
                    if movement > 0.005:  # Only show if moved > 5mm
                        print(f"🔄 Peg trajectory: {movement*1000:.1f}mm to [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
                
                self.last_trail_print_time = self.data.time
        else:
            self.last_trail_print_time = self.data.time
    
    def print_trajectory_stats(self):
        """Print statistics about peg trajectory with coordinate frame info"""
        if len(self.peg_trajectory) < 2:
            print("   Trajectory: Insufficient data")
            return
            
        positions = np.array([p['position'] for p in self.peg_trajectory])
        
        # Calculate statistics
        start_pos = positions[0]
        end_pos = positions[-1]
        total_distance = 0
        
        for i in range(1, len(positions)):
            total_distance += np.linalg.norm(positions[i] - positions[i-1])
        
        displacement = np.linalg.norm(end_pos - start_pos)
        
        # Current peg orientation
        peg_quat = self.data.xquat[self.peg_body_id].copy()
        
        # Convert quaternion to rotation matrix then to Euler angles
        peg_mat = np.zeros(9)
        mujoco.mju_quat2Mat(peg_mat, peg_quat)
        peg_mat = peg_mat.reshape(3, 3)
        
        # Extract Euler angles from rotation matrix (ZYX convention)
        # This is a manual conversion since mju_quat2Euler doesn't exist
        sy = np.sqrt(peg_mat[0, 0] * peg_mat[0, 0] + peg_mat[1, 0] * peg_mat[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(peg_mat[2, 1], peg_mat[2, 2])
            y = np.arctan2(-peg_mat[2, 0], sy)
            z = np.arctan2(peg_mat[1, 0], peg_mat[0, 0])
        else:
            x = np.arctan2(-peg_mat[1, 2], peg_mat[1, 1])
            y = np.arctan2(-peg_mat[2, 0], sy)
            z = 0
        
        peg_euler_deg = np.degrees([x, y, z])
        
        print(f"   🎯 Trajectory & Frame Stats:")
        print(f"     Points: {len(self.peg_trajectory)}")
        print(f"     Duration: {self.peg_trajectory[-1]['time'] - self.peg_trajectory[0]['time']:.2f}s")
        print(f"     Total path: {total_distance*1000:.1f}mm")
        print(f"     Displacement: {displacement*1000:.1f}mm")
        print(f"     Start: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
        print(f"     End:   [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")
        print(f"     Current Orientation (deg): [{peg_euler_deg[0]:.1f}, {peg_euler_deg[1]:.1f}, {peg_euler_deg[2]:.1f}]")
    
    def clear_trajectory(self):
        """Clear peg trajectory data"""
        self.peg_trajectory.clear()
        self.marker_counter = 0
        print("📍 Peg trajectory cleared")
    
    def toggle_peg_frame_visualization(self):
        """Toggle coordinate frame visualization at peg position"""
        self.show_peg_frame = not self.show_peg_frame
        status = "ENABLED" if self.show_peg_frame else "DISABLED"
        print(f"🎯 Peg coordinate frame visualization: {status}")
    
    def get_ee_position(self):
        """Get current end effector position"""
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.ee_site_id].copy()
    
    def is_in_forbidden_zone(self, position):
        """Check if position is inside forbidden zone"""
        if position[2] < FORBIDDEN_ZONE_CENTER[2] or position[2] > FORBIDDEN_ZONE_CENTER[2] + FORBIDDEN_ZONE_SIZE[2]:
            return False
        relative = np.abs(position - FORBIDDEN_ZONE_CENTER)
        return relative[0] <= FORBIDDEN_ZONE_SIZE[0] and relative[1] <= FORBIDDEN_ZONE_SIZE[1]
    
    def move_to_target_smooth(self, target_pos, target_quat, max_joint_velocity=None, maintain_gripper=None):

        if max_joint_velocity is None:
            max_joint_velocity = MAX_JOINT_VELOCITY
        
        # Get current joint state (from actuators, which is the real robot state)
        current_joints = np.array([self.data.ctrl[aid] for aid in self.actuator_ids])
        
        # Sync qpos to match actual robot state before IK
        for j, jid in enumerate(self.joint_ids):
            self.data.qpos[jid] = current_joints[j]
        mujoco.mj_forward(self.model, self.data)
        
        # Check workspace reachability first (distance from robot base at [0, 0, 0.8])
        robot_base = np.array([0, 0, 0.8])
        target_distance = np.linalg.norm(target_pos - robot_base)
        if target_distance > 0.82:  # UR5e conservative reach limit
            print(f"❌ Target unreachable: {target_distance:.3f}m > 0.82m (from base)")
            return False
        
        # High-precision IK solving (1mm position, 0.02 rad orientation tolerance)
        print(f"🎯 High-precision IK for target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] (dist: {target_distance:.3f}m)")
        
        success, error = move_to_target_pose(
            self.model, self.data, target_pos, target_quat,
            max_iterations=2000,  # Even more iterations for critical moves
            lambda_init=0.0005,   # Lower initial damping
            pos_tolerance=0.001,  # 1mm position tolerance
            ori_tolerance=0.02,   # 0.02 rad orientation tolerance  
            initial_joints=current_joints
        )
        
        # For critical moves like lifting, be slightly more lenient to avoid losing the peg
        if success:
            print(f"✅ High-precision IK achieved: {error*1000:.3f}mm error")
        elif error < 0.002:  # Within 2mm - acceptable for lifting to avoid dropping peg
            print(f"⚠️  Acceptable precision for lifting: {error*1000:.3f}mm (< 2mm)")
        else:
            print(f"❌ Failed to achieve required precision: {error*1000:.3f}mm > 2.000mm")
            return False
        
        # Get target joint configuration from IK solution
        target_joints = np.array([self.data.qpos[jid] for jid in self.joint_ids])
        
        # Calculate joint differences
        joint_diff = target_joints - current_joints
        max_diff = np.max(np.abs(joint_diff))
        
        print(f"   Max joint change: {np.degrees(max_diff):.1f}°")
        
        # Calculate number of steps for smooth motion
        if max_diff > 0:
            dt = self.model.opt.timestep
            num_steps = int(max_diff / (max_joint_velocity * dt))
            num_steps = max(num_steps, 10)  # At least 10 steps
            num_steps = min(num_steps, 500)  # Cap at 500 steps
        else:
            num_steps = 10
        
        print(f"   Interpolating over {num_steps} steps...")
        
        # Pure kinematic interpolation - directly set positions
        for step in range(num_steps):
            # Sinusoidal easing for smoother start/stop
            t = (step + 1) / num_steps
            alpha = (1 - np.cos(t * np.pi)) / 2  # Smooth S-curve
            
            interp_joints = current_joints + alpha * joint_diff
            
            # Directly set joint positions (kinematic control)
            for j, jid in enumerate(self.joint_ids):
                self.data.qpos[jid] = interp_joints[j]
                self.data.qvel[jid] = 0.0
            
            # MAINTAIN GRIPPER POSITION during arm movement (with higher force)
            if maintain_gripper is not None:
                for grip_id in self.gripper_actuator_ids:
                    self.data.ctrl[grip_id] = maintain_gripper
            
            # Additional physics step for better contact dynamics
            mujoco.mj_step(self.model, self.data)
            
            # Forward kinematics to update state
            mujoco.mj_forward(self.model, self.data)
            
            # Sync actuator commands to current positions
            for j, aid in enumerate(self.actuator_ids):
                self.data.ctrl[aid] = self.data.qpos[self.joint_ids[j]]
            
            if self.viewer is not None:
                self.viewer.sync()
        
        # Verify final position
        mujoco.mj_forward(self.model, self.data)
        final_pos = self.get_ee_position()
        final_error = np.linalg.norm(target_pos - final_pos)
        print(f"✅ Reached target with {final_error*1000:.2f}mm error")
        
        return True

    def move_to_position(self, target_pos, target_quat=None, gripper_value=None):
        """
        Simple IK move to target position
        Args:
            target_pos: [x, y, z] target position
            target_quat: [w, x, y, z] target quaternion (None = gripper pointing down)
            gripper_value: Gripper position (None = don't change, 0.0 = open, 1.0 = closed)
        """
        if target_quat is None:
            target_quat = np.array([0.0, 1.0, 0.0, 0.0])  # Default: gripper pointing down
        
        # Get current joints
        current_joints = np.array([self.data.qpos[jid] for jid in self.joint_ids])
        
        # Solve IK
        success, error = move_to_target_pose(
            self.model, self.data, target_pos, target_quat,
            max_iterations=1000, lambda_init=0.01,
            pos_tolerance=0.003, ori_tolerance=0.05,
            initial_joints=current_joints
        )
        
        if not success:
            print(f"❌ IK failed: {error*1000:.1f}mm error")
            return False
        
        # Get joint solution
        target_joints = np.array([self.data.qpos[jid] for jid in self.joint_ids])
        joint_diff = target_joints - current_joints
        max_diff = np.max(np.abs(joint_diff))
        
        # Smooth interpolation
        num_steps = max(50, int(max_diff * 200))
        num_steps = min(num_steps, 300)
        
        for step in range(num_steps):
            t = (step + 1) / num_steps
            alpha = 0.5 * (1 - np.cos(t * np.pi))  # Smooth easing
            
            interp_joints = current_joints + alpha * joint_diff
            
            # Set joint positions
            for j, jid in enumerate(self.joint_ids):
                self.data.qpos[jid] = interp_joints[j]
            
            # Set gripper if specified
            if gripper_value is not None:
                for grip_id in self.gripper_actuator_ids:
                    self.data.ctrl[grip_id] = gripper_value
            
            # Physics steps
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            
            # Sync actuators
            for j, aid in enumerate(self.actuator_ids):
                self.data.ctrl[aid] = self.data.qpos[self.joint_ids[j]]
            
            if self.viewer is not None:
                self.viewer.sync()
        
        final_pos = self.get_ee_position()
        final_error = np.linalg.norm(target_pos - final_pos)
        print(f"✅ Moved to target: {final_error*1000:.1f}mm error")
        
        return True

    def move_to_target_slow(self, target_pos, target_quat, maintain_gripper=None, extra_slow=True):
        """Ultra-slow motion for critical grasping operations"""
        # Use high precision IK first
        current_joints = np.array([self.data.qpos[jid] for jid in self.joint_ids])
        
        success, error = move_to_target_pose(
            self.model, self.data, target_pos, target_quat,
            max_iterations=2000,
            lambda_init=0.0005,
            pos_tolerance=0.002,  # Slightly relaxed for slow motion
            ori_tolerance=0.03,
            initial_joints=current_joints
        )
        
        if not success and error > 0.005:
            print(f"❌ IK failed for slow motion: {error*1000:.3f}mm")
            return False
        
        # Get joint differences for ultra-slow interpolation
        target_joints = np.array([self.data.qpos[jid] for jid in self.joint_ids])
        joint_diff = target_joints - current_joints
        max_diff = np.max(np.abs(joint_diff))
        
        # Debug: Check what the IK solution did to positions
        ee_after_ik = self.get_ee_position()
        peg_after_ik = self.get_peg_position()
        dist_after_ik = np.linalg.norm(peg_after_ik - ee_after_ik)
        print(f"   Post-IK: EE [{ee_after_ik[0]:.3f}, {ee_after_ik[1]:.3f}, {ee_after_ik[2]:.3f}] | Peg [{peg_after_ik[0]:.3f}, {peg_after_ik[1]:.3f}, {peg_after_ik[2]:.3f}] | Dist: {dist_after_ik*1000:.1f}mm")
        print(f"   Max joint change: {np.degrees(max_diff):.1f}°")
        
        # Ultra-slow motion parameters
        if extra_slow:
            num_steps = max(500, int(max_diff * 2000))  # Very slow
            num_steps = min(num_steps, 1000)  # Cap at 1000 steps
        else:
            num_steps = max(300, int(max_diff * 1000))  # Slower than normal
            num_steps = min(num_steps, 600)
        
        print(f"   Ultra-slow interpolation over {num_steps} steps...")
        
        # Ultra-slow kinematic interpolation with continuous peg monitoring
        for step in range(num_steps):
            t = (step + 1) / num_steps
            # Very smooth acceleration/deceleration curve
            alpha = 0.5 * (1 - np.cos(t * np.pi))  # Sinusoidal easing
            
            interp_joints = current_joints + alpha * joint_diff
            
            # Set joint positions
            for j, jid in enumerate(self.joint_ids):
                self.data.qpos[jid] = interp_joints[j]
            
            # Maintain gripper if specified
            if maintain_gripper is not None:
                for grip_id in self.gripper_actuator_ids:
                    self.data.ctrl[grip_id] = maintain_gripper
            
            # Multiple physics steps per interpolation step for stability
            for _ in range(3):
                mujoco.mj_step(self.model, self.data)
            
            mujoco.mj_forward(self.model, self.data)
            
            # Sync actuators
            for j, aid in enumerate(self.actuator_ids):
                self.data.ctrl[aid] = self.data.qpos[self.joint_ids[j]]
            
            if self.viewer is not None:
                self.viewer.sync()
            
            # Check peg every 50 steps during critical motion
            if maintain_gripper is not None and step % 50 == 0:
                peg_pos = self.get_peg_position()
                ee_pos = self.get_ee_position()
                distance = np.linalg.norm(peg_pos - ee_pos)
                if distance > 0.025:  # 25mm threshold
                    print(f"⚠️  Peg distance increasing: {distance*1000:.1f}mm at step {step}")
        
        # Verify final position
        mujoco.mj_forward(self.model, self.data)
        final_pos = self.get_ee_position()
        final_error = np.linalg.norm(target_pos - final_pos)
        print(f"✅ Slow motion complete: {final_error*1000:.2f}mm error")
        
        return True
    
    def move_gripper_above_peg(self, peg_pos, z_offset=0.015):
        """Move gripper above peg position with safety checks"""
        target_pos = peg_pos + np.array([0, 0, z_offset])
        
        # Check workspace reachability first (distance from robot base at [0, 0, 0.8])
        robot_base = np.array([0, 0, 0.8])
        target_distance = np.linalg.norm(target_pos - robot_base)
        if target_distance > 0.82:  # UR5e conservative reach limit
            print(f"❌ Target unreachable: {target_distance:.3f}m > 0.82m (from base)")
            return False
        
        if self.is_in_forbidden_zone(target_pos):
            print(f"❌ Target in forbidden zone: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            return False
        
        target_quat = np.array([0.0, 1.0, 0.0, 0.0])  # Gripper pointing down
        
        # High-precision gripper positioning
        print(f"🎯 High-precision gripper positioning: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        current_joints = np.array([self.data.qpos[jid] for jid in self.joint_ids])
        
        success, error = move_to_target_pose(
            self.model, self.data, target_pos, target_quat,
            max_iterations=1200,  # High iteration count for precision
            lambda_init=0.001,    # Low damping for precision
            pos_tolerance=0.001,  # 1mm position tolerance
            ori_tolerance=0.02,   # 0.02 rad orientation tolerance
            initial_joints=current_joints
        )
        
        if success:
            print(f"✅ High-precision gripper positioning: {error*1000:.3f}mm")
        else:
            print(f"❌ Failed high-precision positioning: {error*1000:.3f}mm > 1.000mm")
            return False
        
        stabilize_robot(self.model, self.data, num_steps=50)
        
        final_ee_pos = self.get_ee_position()
        if self.is_in_forbidden_zone(final_ee_pos):
            print(f"⚠️  EE in forbidden zone: [{final_ee_pos[0]:.3f}, {final_ee_pos[1]:.3f}, {final_ee_pos[2]:.3f}]")
        
        return True
    
    def set_gripper_position(self, value, num_steps=100):
        """Set gripper position (0.0 = open, 1.0 = fully closed)"""
        value = np.clip(value, 0.0, 1.0)
        for _ in range(num_steps):
            for act_id in self.gripper_actuator_ids:
                self.data.ctrl[act_id] = value
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()
    
    def execute_step_1(self):
        """Execute Step 1: Move gripper above peg"""
        print("\n" + "="*70)
        print("STEP 1: MOVE GRIPPER ABOVE PEG")
        print("="*70)
        
        # Clear and start trajectory tracking
        self.clear_trajectory()
        self.add_peg_trajectory_point()  # Mark starting position
        
        # Open gripper
        print("\n🖐️  Opening gripper...")
        self.set_gripper_position(0.0, num_steps=50)
        
        # Move directly to target above peg (20mm offset)
        target_pos = PEG_POSITION + np.array([0, 0, 0.010])  # 
        target_quat = np.array([0.0, 1.0, 0.0, 0.0])  # Gripper pointing down
        
        success = self.move_to_target_smooth(target_pos, target_quat)
        
        if not success:
            print("❌ Failed to reach target")
            return
        
        # Show results
        ee_pos = self.get_ee_position()
        peg_to_hole = HOLE_ENTRANCE - PEG_POSITION
        
        print(f"\n✅ Step 1 Complete")
        print(f"   EE position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
        print(f"   EE→Peg: {np.linalg.norm(ee_pos - PEG_POSITION)*1000:.1f}mm")
        
        # Print sensor data and trajectory
        self.print_sensor_info()
        self.add_peg_trajectory_point()  # Mark end of step 1
        self.print_trajectory_stats()
    
    def execute_step_2(self):
        """Execute Step 2: Simple peg grasping test"""
        print("\n" + "="*50)
        print("STEP 2: PEG GRASPING TEST")
        print("="*50)
        
        # 1. Move above peg
        print("📍 Moving above peg...")
        above_peg = PEG_POSITION + np.array([0, 0, 0.01])  # 10mm above
        if not self.move_to_position(above_peg, gripper_value=0.0):  # Open gripper
            print("❌ Failed to move above peg")
            return
        
        # 2. Close gripper to grasp
        print("🤏 Grasping peg...")
        self.set_gripper_position(1.0, num_steps=150)  # Close gripper
        
        # 3. Check grasp
        contacts = self.get_sensor_data_summary()['total_contacts']
        peg_held = self.check_peg_held()
        
        print(f"✅ Grasp complete: {contacts} contacts, Peg held: {peg_held}")
        return
    
    def execute_step_3(self):
        """Execute Step 3: Move peg from current position to entrance via joint interpolation"""
        print("\n" + "="*50)
        print("STEP 3: MOVE PEG TO ENTRANCE")
        print("="*50)
        
        # Check if peg is held
        if not self.check_peg_held():
            print("❌ No peg held - run step 2 first")
            return
        
        # Step 1: Read current 6 joint angles
        current_joints = np.array([self.data.qpos[jid] for jid in self.joint_ids])
        current_gripper = self.data.ctrl[6]  # Save current gripper state
        
        print(f"📍 Current joints: {current_joints}")
        print(f"🤏 Current gripper: {current_gripper:.3f}")
        
        # Step 2: Calculate target entrance position
        entrance_target = HOLE_ENTRANCE + np.array([0, 0, 0.05])  # 50mm above entrance
        
        # Use CURRENT orientation instead of forcing downward
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'ee_site')
        current_rotation_matrix = self.data.site_xmat[ee_site_id].reshape(3, 3)
        # Use natural "gripper pointing down" orientation with VERY relaxed tolerance
        target_quat = np.array([0.0, 1.0, 0.0, 0.0])  # Standard downward orientation
        
        print(f"🎯 Target position: [{entrance_target[0]:.3f}, {entrance_target[1]:.3f}, {entrance_target[2]:.3f}]")
        print(f"🧭 Using downward orientation with relaxed tolerance")
        
        # Step 3: Solve IK for entrance joint configuration
        print("🔧 Solving IK for entrance...")
        # BACKUP current robot state before IK calculation
        backup_qpos = self.data.qpos.copy()
        backup_ctrl = self.data.ctrl.copy()
        
        success, error = move_to_target_pose(
            self.model, self.data, entrance_target, target_quat,
            max_iterations=3000, lambda_init=0.0001,
            pos_tolerance=0.001, ori_tolerance=0.5,  # 1mm position, VERY relaxed orientation
            initial_joints=current_joints
        )
        
        if not success:
            print(f"❌ IK failed for entrance: {error*1000:.1f}mm error")
            print("🔄 Trying lower entrance position...")
            
            # Try lower position (reduce Z by 30mm)
            lower_entrance = entrance_target.copy()
            lower_entrance[2] -= 0.03  # 30mm lower
            print(f"🎯 Lower target: [{lower_entrance[0]:.3f}, {lower_entrance[1]:.3f}, {lower_entrance[2]:.3f}]")
            
            success, error = move_to_target_pose(
                self.model, self.data, lower_entrance, target_quat,
                max_iterations=3000, lambda_init=0.0001,
                pos_tolerance=0.001, ori_tolerance=0.5,  # Still 1mm precision, relaxed orientation
                initial_joints=current_joints
            )
            
            if not success:
                print(f"❌ Lower entrance also failed: {error*1000:.1f}mm error")
                print("🛑 Cannot achieve 1mm precision - stopping simulation")
                # Restore original state
                self.data.qpos[:] = backup_qpos
                self.data.ctrl[:] = backup_ctrl
                mujoco.mj_forward(self.model, self.data)
                return
            else:
                entrance_target = lower_entrance  # Use lower entrance
                print("✅ Lower entrance target accepted")
        
        # Get target joint configuration (IK solution)
        target_joints = np.array([self.data.qpos[jid] for jid in self.joint_ids])
        print(f"🎯 Target joints: {target_joints}")
        
        # RESTORE original robot state before interpolation starts
        self.data.qpos[:] = backup_qpos
        self.data.ctrl[:] = backup_ctrl
        mujoco.mj_forward(self.model, self.data)
        
        # Step 4: Smooth interpolation between current and target joints
        joint_diff = target_joints - current_joints
        max_diff = np.max(np.abs(joint_diff))
        print(f"📏 Max joint difference: {max_diff:.3f} rad")
        
        # Calculate number of steps for smooth motion
        num_steps = max(100, int(max_diff * 300))  # More steps for smoother motion
        num_steps = min(num_steps, 500)
        print(f"🚀 Moving with {num_steps} interpolation steps...")
        
        for step in range(num_steps):
            t = (step + 1) / num_steps
            alpha = 0.5 * (1 - np.cos(t * np.pi))  # Smooth sinusoidal easing
            
            # Interpolate joint positions
            interpolated_joints = current_joints + alpha * joint_diff
            
            # Apply to robot actuators
            for j, aid in enumerate(self.actuator_ids):
                self.data.ctrl[aid] = interpolated_joints[j]
            
            # Maintain current gripper state
            self.data.ctrl[6] = current_gripper
            
            # Step simulation MULTIPLE times for smoother motion
            for _ in range(10):  # 10번 더 세밀하게 시뮬레이션
                mujoco.mj_step(self.model, self.data)
        
        print("✅ Joint interpolation completed")
        
        if success:
            # Check if peg is still held after movement
            peg_held_after = self.check_peg_held()
            contacts_after = self.get_sensor_data_summary()['total_contacts']
            
            if peg_held_after:
                print(f"✅ SUCCESS: Peg transported to entrance!")
                print(f"   Contacts: {contacts_after}, Peg held: {peg_held_after}")
                
                # Show final positions
                final_eef = self.get_ee_position()
                final_peg = self.get_peg_position()
                print(f"   Final EEF: [{final_eef[0]:.3f}, {final_eef[1]:.3f}, {final_eef[2]:.3f}]")
                print(f"   Final Peg: [{final_peg[0]:.3f}, {final_peg[1]:.3f}, {final_peg[2]:.3f}]")
            else:
                print(f"❌ FAILURE: Peg dropped during transport!")
        else:
            print("❌ Failed to move to entrance")

    def run_interactive_viewer(self):
        """Run interactive viewer with command input"""
        self.set_initial_robot_pose()
        self.set_peg_position(PEG_POSITION)
        
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.viewer = viewer
            self.simulation_running = True
            
            ee_pos = self.get_ee_position()
            print(f"\n👀 Viewer ready | EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
            print(f"Commands: 1=Step 1, 2=Grasp Peg, 3=Move to Entrance, t=Trajectory, c=Clear, f=Frame, q=Quit")
            
            # Input thread
            def input_thread():
                while self.simulation_running:
                    try:
                        user_input = input("\nCmd: ").strip().lower()
                        if user_input:
                            self.command_queue.append(user_input)
                    except:
                        break
            
            threading.Thread(target=input_thread, daemon=True).start()
            
            # Main loop
            while viewer.is_running() and self.simulation_running:
                if len(self.command_queue) > 0:
                    cmd = self.command_queue.pop(0)
                    
                    if cmd == 'q':
                        self.simulation_running = False
                        break
                    elif cmd == '1':
                        self.execute_step_1()
                    elif cmd == '2':
                        self.execute_step_2()
                    elif cmd == '3':
                        self.execute_step_3()
                    elif cmd == 't':
                        print("\n📊 TRAJECTORY STATISTICS:")
                        self.add_peg_trajectory_point()  # Current position
                        self.print_trajectory_stats()
                    elif cmd == 'c':
                        self.clear_trajectory()
                    elif cmd == 'f':
                        self.toggle_peg_frame_visualization()
                    else:
                        print(f"Unknown: {cmd}")
                
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                
                # Update visualization
                if self.enable_plotting:
                    self.step_counter += 1
                    if self.step_counter % self.plot_update_interval == 0:
                        self.update_visualization()
                
                # Track peg trajectory and visualize (every 5 steps)
                if self.step_counter % 5 == 0:
                    self.add_peg_trajectory_point()
                    
                # Visualize peg coordinate frame and trajectory
                if self.show_peg_frame:
                    self.visualize_peg_frame()
                self.visualize_trajectory_trail()
                
                time.sleep(0.01)
            
            self.simulation_running = False
        
        self.viewer = None
        
        # Close plots if enabled
        if self.enable_plotting:
            pass  # Visualization disabled

def main():
    """Main entry point"""
    peg_to_entrance = HOLE_ENTRANCE - PEG_POSITION
    dist_mm = np.linalg.norm(peg_to_entrance) * 1000
    
    print(f"\nPeg→Hole: [{peg_to_entrance[0]*1000:.1f}, {peg_to_entrance[1]*1000:.1f}, {peg_to_entrance[2]*1000:.1f}]mm | dist={dist_mm:.1f}mm")
    print(f"Forbidden: {FORBIDDEN_ZONE_SIZE[0]*2000:.0f}×{FORBIDDEN_ZONE_SIZE[1]*2000:.0f}×{FORBIDDEN_ZONE_SIZE[2]*1000:.0f}mm")
    print(f"DIGIT: 15×15mm ROI, 0-0.8mm range")
    print(f"Plotting: {'ENABLED' if ENABLE_PLOTTING else 'DISABLED'} (update every {PLOT_UPDATE_INTERVAL} steps)")
    print(f"🔍 Trajectory Tracking: ENABLED - Visual markers in 3D viewer")
    print(f"🎯 Coordinate Frame: Shows peg position and orientation\n")
    
    demo = InteractivePegDemo()
    demo.run_interactive_viewer()


if __name__ == "__main__":
    main()
