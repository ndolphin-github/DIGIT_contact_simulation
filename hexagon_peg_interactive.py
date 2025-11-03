import numpy as np
import mujoco
import mujoco.viewer
import time
import threading
import matplotlib.pyplot as plt

from simple_ik import move_to_target_pose, stabilize_robot, DEFAULT_INITIAL_JOINTS
from gripper_digit_sensor import GripperDIGITSensor


# ========== CONFIGURATION ==========
PEG_POSITION = np.array([0.6, 0.2, 0.8])          # Peg bottom center
HOLE_ENTRANCE = np.array([0.629, 0.029, 0.875])   # Hole entrance (top opening)
HOLE_CENTER = np.array([0.6, 0.0, 0.8])           # Hole center (reference)

# Forbidden zone: 130mm × 130mm × 75mm box above hole
FORBIDDEN_ZONE_CENTER = np.array([0.6, 0.0, 0.8])
FORBIDDEN_ZONE_SIZE = np.array([0.065, 0.065, 0.075])  # Half-sizes

# Visualization settings
ENABLE_PLOTTING = False  # Set to False to disable real-time plotting
PLOT_UPDATE_INTERVAL = 10  # Update plots every N simulation steps

# Motion control settings
MAX_JOINT_VELOCITY = 0.5  # rad/s - Maximum joint velocity for smooth motion
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
        
        # Plotting setup
        self.enable_plotting = ENABLE_PLOTTING
        self.plot_update_interval = PLOT_UPDATE_INTERVAL
        self.step_counter = 0
        
        if self.enable_plotting:
            self.setup_visualization()
    
    def setup_visualization(self):
        """Setup matplotlib visualization for DIGIT sensors"""
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('DIGIT Sensor Visualization (NPE Format)', fontsize=14, fontweight='bold')
        
        # History data
        self.time_history = []
        self.left_contact_history = []
        self.right_contact_history = []
        self.left_depth_history = []
        self.right_depth_history = []
        
        plt.tight_layout()
        plt.show(block=False)
    
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
    
    def get_npe_sensor_data(self):
        """
        Get DIGIT sensor data in NPE-compatible format.
        
        Returns:
            dict with 'left' and 'right' sensor data in MuJoCo input format
        """
        left_npe_input = self.digit_left.get_npe_input_format(self.data)
        right_npe_input = self.digit_right.get_npe_input_format(self.data)
        
        return {
            'left': left_npe_input,
            'right': right_npe_input
        }
    
    def print_npe_sensor_info(self):
        """Print DIGIT sensor information in NPE format"""
        npe_data = self.get_npe_sensor_data()
        
        total_contacts = npe_data['left']['num_contacts'] + npe_data['right']['num_contacts']
        
        if total_contacts == 0:
            return
        
        print(f"\n📊 NPE Format: {total_contacts} total contacts")
        
        for side, data in [('LEFT', npe_data['left']), ('RIGHT', npe_data['right'])]:
            if data['num_contacts'] > 0:
                avg_depth = np.mean(data['distance_from_plane_mm'])
                avg_x = np.mean(data['contact_x_mm'])
                avg_y = np.mean(data['contact_y_mm'])
                print(f"   {side}: N={data['num_contacts']} | "
                      f"depth={avg_depth:.3f}mm | "
                      f"centroid=({avg_x:.1f},{avg_y:.1f})mm")
    
    def update_visualization(self):
        """Update real-time sensor visualization with NPE format data"""
        if not self.enable_plotting:
            return
        
        # Get NPE format data
        npe_data = self.get_npe_sensor_data()
        left_data = npe_data['left']
        right_data = npe_data['right']
        
        # Current time
        current_time = self.data.time
        
        # Calculate metrics
        left_contacts = left_data['num_contacts']
        right_contacts = right_data['num_contacts']
        
        left_avg_depth = np.mean(left_data['distance_from_plane_mm']) if left_contacts > 0 else 0.0
        right_avg_depth = np.mean(right_data['distance_from_plane_mm']) if right_contacts > 0 else 0.0
        
        # Update history
        self.time_history.append(current_time)
        self.left_contact_history.append(left_contacts)
        self.right_contact_history.append(right_contacts)
        self.left_depth_history.append(left_avg_depth)
        self.right_depth_history.append(right_avg_depth)
        
        # Keep reasonable history length
        if len(self.time_history) > 200:
            self.time_history.pop(0)
            self.left_contact_history.pop(0)
            self.right_contact_history.pop(0)
            self.left_depth_history.pop(0)
            self.right_depth_history.pop(0)
        
        # Update left sensor plots
        self._update_sensor_plots(
            left_data,
            self.axes[0, 0],
            self.axes[0, 1],
            self.left_contact_history,
            self.left_depth_history,
            left_contacts,
            left_avg_depth,
            "LEFT"
        )
        
        # Update right sensor plots
        self._update_sensor_plots(
            right_data,
            self.axes[1, 0],
            self.axes[1, 1],
            self.right_contact_history,
            self.right_depth_history,
            right_contacts,
            right_avg_depth,
            "RIGHT"
        )
        
        # Refresh display
        plt.draw()
        plt.pause(0.001)
    
    def _update_sensor_plots(self, npe_data, ax_contact, ax_history, 
                            contact_history, depth_history, 
                            current_contacts, current_depth, sensor_name):
        """Update plots for a single sensor (NPE format)"""
        
        # Update contact patch plot
        ax_contact.clear()
        ax_contact.set_xlim(-10, 10)
        ax_contact.set_ylim(0, 20)
        
        # Redraw ROI
        roi_rect = plt.Rectangle((-7.5, 0), 15, 15, 
                               linewidth=2, edgecolor='green', 
                               facecolor='none', linestyle='--')
        ax_contact.add_patch(roi_rect)
        
        if current_contacts > 0:
            contact_x = npe_data['contact_x_mm']
            contact_y = npe_data['contact_y_mm']
            distances = npe_data['distance_from_plane_mm']
            
            # Color by distance (closer = darker red)
            intensities = [1.0 - (d / 0.8) for d in distances]
            colors = plt.cm.Reds(intensities)
            
            ax_contact.scatter(contact_x, contact_y, c=colors, s=50, 
                             alpha=0.8, edgecolors='black')
            
            ax_contact.text(-9, 18, f'Contacts: {current_contacts}', 
                          fontweight='bold', fontsize=10)
            ax_contact.text(-9, 16, f'Avg Depth: {current_depth:.3f} mm', 
                          fontweight='bold', fontsize=10)
        else:
            ax_contact.text(0, 10, 'NO CONTACT', ha='center', 
                          fontsize=14, color='gray', fontweight='bold')
        
        ax_contact.set_xlabel('X (mm)')
        ax_contact.set_ylabel('Y (mm)')
        ax_contact.set_title(f'{sensor_name} DIGIT - Contact Patch (NPE Format)')
        ax_contact.grid(True, alpha=0.3)
        
        # Update history plots
        if len(self.time_history) > 1:
            ax_history.clear()
            
            # Plot contact count
            ax1 = ax_history
            color1 = 'tab:blue'
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Contact Count', color=color1)
            ax1.plot(self.time_history, contact_history, color=color1, linewidth=2, label='Contacts')
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, alpha=0.3)
            
            # Plot average depth on secondary axis
            ax2 = ax1.twinx()
            color2 = 'tab:red'
            ax2.set_ylabel('Avg Depth (mm)', color=color2)
            ax2.plot(self.time_history, depth_history, color=color2, linewidth=2, linestyle='--', label='Depth')
            ax2.tick_params(axis='y', labelcolor=color2)
            
            ax_history.set_title(f'{sensor_name} DIGIT - History')
    
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
    
    def generate_smooth_path(self, start_pos, end_pos, num_waypoints=2, safe_height=None):
        """
        Generate smooth path between two points avoiding forbidden zone.
        
        Args:
            start_pos: Starting 3D position [x, y, z]
            end_pos: Ending 3D position [x, y, z]
            num_waypoints: Number of waypoints including start and end
            safe_height: Optional safe height to arc over forbidden zone
        
        Returns:
            List of waypoint positions
        """
        waypoints = []
        
        # Check if direct path passes through forbidden zone
        needs_arc = False
        for alpha in np.linspace(0, 1, 20):
            test_pos = start_pos + alpha * (end_pos - start_pos)
            if self.is_in_forbidden_zone(test_pos):
                needs_arc = True
                break
        
        if needs_arc and safe_height is None:
            # Calculate safe height (top of forbidden zone + 20mm margin)
            safe_height = FORBIDDEN_ZONE_CENTER[2] + FORBIDDEN_ZONE_SIZE[2] + 0.02
        
        # Generate waypoints
        for i in range(num_waypoints):
            alpha = i / (num_waypoints - 1)
            
            if needs_arc:
                # Arc path: rise to safe height in middle, then descend
                # Use parabolic arc
                height_factor = 4 * alpha * (1 - alpha)  # Peak at alpha=0.5
                arc_height = safe_height if safe_height else max(start_pos[2], end_pos[2]) + 0.05
                
                # Linear interpolation in XY
                xy_interp = start_pos[:2] + alpha * (end_pos[:2] - start_pos[:2])
                
                # Z interpolation with arc
                z_linear = start_pos[2] + alpha * (end_pos[2] - start_pos[2])
                z_arc = z_linear + height_factor * (arc_height - max(start_pos[2], end_pos[2]))
                
                waypoint = np.array([xy_interp[0], xy_interp[1], z_arc])
            else:
                # Straight line interpolation
                waypoint = start_pos + alpha * (end_pos - start_pos)
            
            waypoints.append(waypoint)
        
        return waypoints
    
    def move_to_target_smooth(self, target_pos, target_quat, max_joint_velocity=None, maintain_gripper=None):

        if max_joint_velocity is None:
            max_joint_velocity = MAX_JOINT_VELOCITY
        
        # Get current joint state (from actuators, which is the real robot state)
        current_joints = np.array([self.data.ctrl[aid] for aid in self.actuator_ids])
        
        # Sync qpos to match actual robot state before IK
        for j, jid in enumerate(self.joint_ids):
            self.data.qpos[jid] = current_joints[j]
        mujoco.mj_forward(self.model, self.data)
        
        # Solve IK once for the target
        print(f"🎯 Solving IK for target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        success, error = move_to_target_pose(
            self.model, self.data, target_pos, target_quat,
            max_iterations=500, lambda_init=0.01,
            pos_tolerance=0.005, ori_tolerance=0.15,
            initial_joints=current_joints
        )
        
        if not success:
            if error < 0.02:  # Accept if within 20mm
                print(f"   Accepting solution with {error*1000:.2f}mm error")
            else:
                print(f"❌ IK failed with {error*1000:.2f}mm error")
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
            
            # MAINTAIN GRIPPER POSITION during arm movement
            if maintain_gripper is not None:
                for grip_id in self.gripper_actuator_ids:
                    self.data.ctrl[grip_id] = maintain_gripper
            
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
    
    def move_gripper_above_peg(self, peg_pos, z_offset=0.15):
        """Move gripper above peg position with safety checks"""
        target_pos = peg_pos + np.array([0, 0, z_offset])
        
        if self.is_in_forbidden_zone(target_pos):
            print(f"❌ Target in forbidden zone: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            return False
        
        target_quat = np.array([0.0, 1.0, 0.0, 0.0])  # Gripper pointing down
        success, error = move_to_target_pose(
            self.model, self.data, target_pos, target_quat,
            max_iterations=500, lambda_init=0.01,
            pos_tolerance=0.005, ori_tolerance=0.1
        )
        
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
        
        # Open gripper
        print("\n🖐️  Opening gripper...")
        self.set_gripper_position(0.0, num_steps=50)
        
        # Move directly to target above peg (20mm offset)
        target_pos = PEG_POSITION + np.array([0, 0, 0.020])  # 20mm above peg
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
        
        # Print sensor data
        self.print_sensor_info()
        self.print_npe_sensor_info()
    
    def execute_step_2(self):
        """Execute Step 2: Grab peg and move to hole entrance with Z offset"""
        print("\n" + "="*70)
        print("STEP 2: GRAB PEG AND MOVE TO ENTRANCE")
        print("="*70)
        
        target_quat = np.array([0.0, 1.0, 0.0, 0.0])
        
        # Step 2.1: Lower gripper onto peg
        print("\n📍 Lowering gripper onto peg...")
        peg_grasp_target = PEG_POSITION + np.array([0, 0, 0.020])  # 20mm above peg center  
        success = self.move_to_target_smooth(peg_grasp_target, target_quat)
        
        if not success:
            print("❌ Failed to lower to peg")
            return
        
        # Step 2.2: Close gripper to grasp peg
        print("🤏 Closing gripper to grasp peg...")
        self.set_gripper_position(1.0, num_steps=200)  # Smoothly close to 1.0
        
        # Wait for grasp to stabilize
        print("   Waiting for grasp to stabilize...")
        for _ in range(100):
            # Continue commanding gripper to stay closed
            for grip_id in self.gripper_actuator_ids:
                self.data.ctrl[grip_id] = 1.0
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()
        
        # Step 2.3: Lift peg (MAINTAIN GRIPPER at 1.0)
        print("⬆️  Lifting peg...")
        lift_target = PEG_POSITION + np.array([0, 0, 0.02])  # Lift 20mm
        success = self.move_to_target_smooth(lift_target, target_quat, maintain_gripper=1.0)
        
        if not success:
            print("❌ Failed to lift peg")
            return
        
        # Step 2.4: Move to hole entrance with 50mm Z offset (MAINTAIN GRIPPER at 1.0)
        print("🎯 Moving to hole entrance (Z offset +50mm)...")
        entrance_target = np.array([0.629, 0.029, 0.925])  # 50mm above entrance
        success = self.move_to_target_smooth(entrance_target, target_quat, maintain_gripper=1.0)
        
        if not success:
            print("❌ Failed to reach entrance")
            return
        
        # Final stabilization with gripper closed
        print("   Final stabilization...")
        for _ in range(100):
            for j, aid in enumerate(self.actuator_ids):
                self.data.ctrl[aid] = self.data.qpos[self.joint_ids[j]]
            # Keep gripper closed
            for grip_id in self.gripper_actuator_ids:
                self.data.ctrl[grip_id] = 1.0
            mujoco.mj_step(self.model, self.data)
        
        # Show results
        ee_pos = self.get_ee_position()
        peg_pos = self.data.xpos[self.peg_body_id].copy()
        entrance_to_peg = peg_pos - HOLE_ENTRANCE
        
        print(f"\n✅ Step 2 Complete")
        print(f"   EE position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
        print(f"   Peg position: [{peg_pos[0]:.3f}, {peg_pos[1]:.3f}, {peg_pos[2]:.3f}]")
        print(f"   Peg→Entrance: [{entrance_to_peg[0]*1000:.1f}, {entrance_to_peg[1]*1000:.1f}, {entrance_to_peg[2]*1000:.1f}]mm")
        print(f"   XY offset: {np.linalg.norm(entrance_to_peg[:2])*1000:.1f}mm | Z offset: {entrance_to_peg[2]*1000:.1f}mm")
        
        # Print sensor data
        self.print_sensor_info()
        self.print_npe_sensor_info()
    
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
            print(f"Commands: 1=Step 1, 2=Step 2, q=Quit")
            
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
                    else:
                        print(f"Unknown: {cmd}")
                
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                
                # Update visualization
                if self.enable_plotting:
                    self.step_counter += 1
                    if self.step_counter % self.plot_update_interval == 0:
                        self.update_visualization()
                
                time.sleep(0.01)
            
            self.simulation_running = False
        
        self.viewer = None
        
        # Close plots if enabled
        if self.enable_plotting:
            plt.close(self.fig)


def main():
    """Main entry point"""
    peg_to_entrance = HOLE_ENTRANCE - PEG_POSITION
    dist_mm = np.linalg.norm(peg_to_entrance) * 1000
    
    print(f"\nPeg→Hole: [{peg_to_entrance[0]*1000:.1f}, {peg_to_entrance[1]*1000:.1f}, {peg_to_entrance[2]*1000:.1f}]mm | dist={dist_mm:.1f}mm")
    print(f"Forbidden: {FORBIDDEN_ZONE_SIZE[0]*2000:.0f}×{FORBIDDEN_ZONE_SIZE[1]*2000:.0f}×{FORBIDDEN_ZONE_SIZE[2]*1000:.0f}mm")
    print(f"DIGIT: 15×15mm ROI, 0-0.8mm range")
    print(f"Plotting: {'ENABLED' if ENABLE_PLOTTING else 'DISABLED'} (update every {PLOT_UPDATE_INTERVAL} steps)\n")
    
    demo = InteractivePegDemo()
    demo.run_interactive_viewer()


if __name__ == "__main__":
    main()
