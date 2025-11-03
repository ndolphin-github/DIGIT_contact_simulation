import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import time
from gripper_digit_sensor import GripperDIGITSensor, UniversalMeshExtractor
from simple_ik import move_to_target_pose, stabilize_robot, print_joint_state

# ============================================================================
# CONFIGURATION
# ============================================================================
ENABLE_MATPLOTLIB = False  # Set to True to enable matplotlib visualization
# ============================================================================

class UR5eDIGITDemo:
    """UR5e robot with dual DIGIT sensors on gripper fingers - Hexagon peg-in-hole task"""
    
    def __init__(self, enable_visualization=ENABLE_MATPLOTLIB):
        self.model = None
        self.data = None
        
        # Visualization toggle
        self.enable_visualization = enable_visualization
        
        # DIGIT sensors (left and right)
        self.digit_left_sensor = None
        self.digit_right_sensor = None
        
        # Object extractor (for the peg)
        self.peg_extractor = None
        
        # Contact data history
        self.left_contact_history = []
        self.right_contact_history = []
        self.left_force_history = []
        self.right_force_history = []
        self.time_history = []
        self.start_time = time.time()
        
        # Matplotlib components
        self.fig = None
        self.axes = None
        
        # Update control
        self.plot_update_counter = 0
        
    def load_robot_scene(self):
        """Load the UR5e robot with DIGIT sensors"""
        
        xml_path = "ur5e_with_DIGIT_primitive_hexagon.xml"
        
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
            
            print(f"\n🎯 Loading scene with HEXAGON peg-in-hole task")
            
            # Set initial robot joint positions to reach near hexagon peg at [0.529, 0.129, 0.8]
            # These positions move the gripper closer to the peg (on the table)
            joint_names = [
                "shoulder_pan_joint",
                "shoulder_lift_joint", 
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint"
            ]
            
            # Initial joint angles (in radians) - safe starting pose
            initial_angles = [
                0.0,      # shoulder_pan: facing forward (+X direction) 
                -1.0,     # shoulder_lift: arm slightly forward
                1.5,      # elbow: moderately bent
                -2.0,     # wrist_1: wrist angled down
                -1.57,    # wrist_2: gripper vertical orientation
                0.0       # wrist_3: no rotation
            ]
            
            # Actuator names corresponding to joints
            actuator_names = [
                "shoulder_pan_actuator",
                "shoulder_lift_actuator",
                "elbow_actuator",
                "wrist_1_actuator",
                "wrist_2_actuator",
                "wrist_3_actuator"
            ]
            
            # Set joint positions AND actuator controls
            for joint_name, actuator_name, angle in zip(joint_names, actuator_names, initial_angles):
                try:
                    # Set joint position (qpos)
                    joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    self.data.qpos[joint_id] = angle
                    
                    # Set actuator control (ctrl) to match desired position
                    actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                    self.data.ctrl[actuator_id] = angle
                except Exception as e:
                    print(f"Warning: Could not set {joint_name}/{actuator_name}: {e}")
            
            # Set gripper to partially closed position (0.3 radians = ~17 degrees)
            try:
                # Set gripper joint positions
                left_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rh_p12_rn_left")
                right_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rh_p12_rn_right")
                self.data.qpos[left_joint_id] = 0.3
                self.data.qpos[right_joint_id] = 0.3
                
                # Set gripper actuator controls
                left_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rh_p12_rn_left_actuator")
                right_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rh_p12_rn_right_actuator")
                self.data.ctrl[left_actuator_id] = 0.3
                self.data.ctrl[right_actuator_id] = 0.3
            except Exception as e:
                print(f"Warning: Could not set gripper: {e}")
            
            # Forward dynamics to initialize
            mujoco.mj_forward(self.model, self.data)
            
            print(f"✓ UR5e robot scene loaded from {xml_path}")
            
        except Exception as e:
            print(f"❌ Failed to load robot scene: {e}")
            raise
    
    def print_body_pose(self, body_name):
        """Print global position and orientation of a body"""
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            
            # Get position
            pos = self.data.xpos[body_id].copy()
            
            # Get orientation (rotation matrix)
            rot_mat = self.data.xmat[body_id].reshape(3, 3).copy()
            
            # Convert rotation matrix to quaternion
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rot_mat.flatten())
            
            # Convert quaternion to euler angles for easier interpretation
            euler = np.zeros(3)
            mujoco.mju_quat2Vel(euler, quat, 1.0)
            
            print(f"\n📍 Body: {body_name}")
            print(f"   Position [x, y, z]: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            print(f"   Quaternion [w, x, y, z]: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
            print(f"   Rotation Matrix:")
            print(f"      X-axis: [{rot_mat[0,0]:.3f}, {rot_mat[1,0]:.3f}, {rot_mat[2,0]:.3f}]")
            print(f"      Y-axis: [{rot_mat[0,1]:.3f}, {rot_mat[1,1]:.3f}, {rot_mat[2,1]:.3f}]")
            print(f"      Z-axis: [{rot_mat[0,2]:.3f}, {rot_mat[1,2]:.3f}, {rot_mat[2,2]:.3f}]")
            
            return pos, quat, rot_mat
            
        except Exception as e:
            print(f"❌ Failed to get pose for {body_name}: {e}")
            return None, None, None
    
    def print_eef_pose(self):
        """Print end effector site position and orientation"""
        try:
            eef_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
            
            # Get position
            pos = self.data.site_xpos[eef_site_id].copy()
            
            # Get orientation (rotation matrix)
            rot_mat = self.data.site_xmat[eef_site_id].reshape(3, 3).copy()
            
            # Convert to quaternion
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rot_mat.flatten())
            
            print(f"\n🤖 End Effector (eef_site)")
            print(f"   Position [x, y, z]: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            print(f"   Quaternion [w, x, y, z]: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
            print(f"   Rotation Matrix:")
            print(f"      X-axis: [{rot_mat[0,0]:.3f}, {rot_mat[1,0]:.3f}, {rot_mat[2,0]:.3f}]")
            print(f"      Y-axis: [{rot_mat[0,1]:.3f}, {rot_mat[1,1]:.3f}, {rot_mat[2,1]:.3f}]")
            print(f"      Z-axis (gripper direction): [{rot_mat[0,2]:.3f}, {rot_mat[1,2]:.3f}, {rot_mat[2,2]:.3f}]")
            
            return pos, quat, rot_mat
            
        except Exception as e:
            print(f"❌ Failed to get EEF pose: {e}")
            return None, None, None
    
    def move_eef_to_peg(self):
        """Move end effector above peg with downward-facing orientation"""
        
        try:
            print("\n" + "="*70)
            print("STARTING IK - Moving gripper above peg")
            print("="*70)
            
            # Print initial state
            self.print_body_pose("hexagon_peg_body")
            self.print_eef_pose()
            
            # Get peg position
            peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hexagon_peg_body")
            peg_pos = self.data.xpos[peg_body_id].copy()
            
            # Target: 8cm above peg
            target_pos = peg_pos.copy()
            target_pos[2] += 0.005  #offset
            
            # Target orientation: gripper pointing straight down
            # This quaternion makes EEF Z-axis point in global -Z direction
            target_quat = np.array([0.0, 1.0, 0.0, 0.0])  # 180° rotation around X-axis
            
            # Preferred joint configuration (natural-looking pose from GUI)
            # Order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
            preferred_joints = np.array([
                0.26,   # shoulder_pan
                -1.66,  # shoulder_lift
                -1.67,  # elbow
                -1.04,  # wrist_1
                1.57,   # wrist_2
                -2.29   # wrist_3
            ])
            
            print(f"\n🎯 IK Target:")
            print(f"   Peg position:    [{peg_pos[0]:.4f}, {peg_pos[1]:.4f}, {peg_pos[2]:.4f}]")
            print(f"   Target (8cm up): [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
            print(f"   Orientation: Gripper pointing DOWN (global -Z)")
            print(f"\n   Preferred joint config: {np.degrees(preferred_joints).round(1)} degrees")
            
            # Perform IK starting from preferred configuration
            print(f"\n🔧 Running IK from natural pose...")
            success, error = move_to_target_pose(
                self.model, self.data, 
                target_pos, target_quat,
                max_iterations=500,
                step_size=0.1,
                pos_tolerance=5e-3,
                ori_tolerance=0.1,
                initial_joints=preferred_joints,
                joint_weight=0.02  # Small weight to stay near preferred config
            )
            
            # Check result
            if success or error < 0.01:  # Accept if position error < 10mm
                print(f"\n✅ IK COMPLETED (error: {error*1000:.2f}mm)")
                
                # Stabilize robot
                print(f"\n🔒 Stabilizing robot...")
                stabilize_robot(self.model, self.data, num_steps=100)
                
                # Verify final pose
                print("\n" + "="*70)
                print("AFTER IK - Final State")
                print("="*70)
                self.print_body_pose("hexagon_peg_body")
                self.print_eef_pose()
                print_joint_state(self.model, self.data)
                
                # Final position check
                eef_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
                final_pos = self.data.site_xpos[eef_site_id].copy()
                final_error = np.linalg.norm(final_pos - target_pos)
                
                print(f"\n✅ Final EE position: [{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}]")
                print(f"✅ Final error: {final_error*1000:.2f}mm")
                print("="*70)
                
            else:
                print(f"\n❌ IK FAILED - Error too large: {error*1000:.2f}mm")
                print("   Robot will remain in best found configuration")
                
        except Exception as e:
            print(f"\n❌ Exception during IK: {e}")
            import traceback
            traceback.print_exc()
    
    def initialize_digit_sensors(self):
        """Initialize both DIGIT sensors on gripper fingers"""
        
        try:
            # Using dedicated GripperDIGITSensor with fixed geometry
            # Geometry matches XML visualization: pos="0 ±0.004 0.03"
            
            # Left DIGIT sensor
            self.digit_left_sensor = GripperDIGITSensor(
                model=self.model,
                sensor_body_name="digit_geltip_left",
                sensor_type="left"
            )
            
            # Right DIGIT sensor  
            self.digit_right_sensor = GripperDIGITSensor(
                model=self.model,
                sensor_body_name="digit_geltip_right",
                sensor_type="right"
            )
            
            # Initialize hexagon peg extractor
            self.peg_extractor = UniversalMeshExtractor(
                model=self.model,
                object_geom_name="hexagon_peg_collision"
            )
            print(f"✓ Hexagon peg mesh extractor initialized")
            
            # Print sensor positions for verification
            left_pos = self.data.xpos[self.digit_left_sensor.sensor_body_id]
            right_pos = self.data.xpos[self.digit_right_sensor.sensor_body_id]
            
            print(f"  Left sensor position: [{left_pos[0]:.4f}, {left_pos[1]:.4f}, {left_pos[2]:.4f}]")
            print(f"  Right sensor position: [{right_pos[0]:.4f}, {right_pos[1]:.4f}, {right_pos[2]:.4f}]")
            
        except Exception as e:
            print(f"❌ Failed to initialize sensors: {e}")
            raise
    

    def setup_visualization(self):
        """Setup matplotlib visualization for both sensors"""
        
        if not self.enable_visualization:
            print("✓ Matplotlib visualization disabled (set ENABLE_MATPLOTLIB=True to enable)")
            return
        
        # Create 2x2 subplot grid (removed 3D plots)
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Left sensor plots (top row)
        ax_left_contact = self.axes[0, 0]
        ax_left_force = self.axes[0, 1]
        
        # Right sensor plots (bottom row)
        ax_right_contact = self.axes[1, 0]
        ax_right_force = self.axes[1, 1]
        
        # Configure left contact patch
        ax_left_contact.set_xlim(-10, 10)
        ax_left_contact.set_ylim(0, 20)
        ax_left_contact.set_xlabel('X (mm)')
        ax_left_contact.set_ylabel('Y (mm)')
        ax_left_contact.set_title('LEFT DIGIT - Contact Patch')
        ax_left_contact.grid(True, alpha=0.3)
        roi_left = plt.Rectangle((-7.5, 0), 15, 15, 
                                linewidth=2, edgecolor='green', 
                                facecolor='none', linestyle='--')
        ax_left_contact.add_patch(roi_left)
        
        # Configure right contact patch
        ax_right_contact.set_xlim(-10, 10)
        ax_right_contact.set_ylim(0, 20)
        ax_right_contact.set_xlabel('X (mm)')
        ax_right_contact.set_ylabel('Y (mm)')
        ax_right_contact.set_title('RIGHT DIGIT - Contact Patch')
        ax_right_contact.grid(True, alpha=0.3)
        roi_right = plt.Rectangle((-7.5, 0), 15, 15, 
                                 linewidth=2, edgecolor='green', 
                                 facecolor='none', linestyle='--')
        ax_right_contact.add_patch(roi_right)
        
        # Configure force plots
        for ax_force, title in [(ax_left_force, 'LEFT DIGIT - Force'), 
                                (ax_right_force, 'RIGHT DIGIT - Force')]:
            ax_force.set_xlabel('Time (s)')
            ax_force.set_ylabel('Contact Force (N)')
            ax_force.set_title(title)
            ax_force.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.ion()
        plt.show(block=False)
        
        print("✓ Dual sensor visualization setup complete")
    
    def get_sensor_contacts(self):
        """Get contact data from both sensors"""
        
        # Detect contacts for both sensors (they use gel tip vertices internally)
        left_contacts = self.digit_left_sensor.detect_proximity_contacts(self.data)
        right_contacts = self.digit_right_sensor.detect_proximity_contacts(self.data)
        
        return left_contacts, right_contacts
    
    def update_visualization(self):
        """Update all plots with current sensor data"""
        
        if not self.enable_visualization:
            return  # Skip visualization if disabled
        
        # Get contacts
        left_contacts, right_contacts = self.get_sensor_contacts()
        
        # Calculate forces (simple approximation)
        left_force = len(left_contacts) * 0.01 if left_contacts else 0.0
        right_force = len(right_contacts) * 0.01 if right_contacts else 0.0
        
        # Update history
        current_time = time.time() - self.start_time
        self.left_contact_history.append(len(left_contacts))
        self.right_contact_history.append(len(right_contacts))
        self.left_force_history.append(left_force)
        self.right_force_history.append(right_force)
        self.time_history.append(current_time)
        
        # Keep reasonable history length
        if len(self.time_history) > 100:
            self.left_contact_history = self.left_contact_history[-100:]
            self.right_contact_history = self.right_contact_history[-100:]
            self.left_force_history = self.left_force_history[-100:]
            self.right_force_history = self.right_force_history[-100:]
            self.time_history = self.time_history[-100:]
        
        # Update left sensor plots
        self._update_sensor_plots(
            left_contacts, 
            self.axes[0, 0],  # contact patch
            self.axes[0, 1],  # force
            self.left_force_history,
            left_force,
            "LEFT"
        )
        
        # Update right sensor plots
        self._update_sensor_plots(
            right_contacts,
            self.axes[1, 0],  # contact patch
            self.axes[1, 1],  # force
            self.right_force_history,
            right_force,
            "RIGHT"
        )
        
        # Refresh display
        plt.draw()
        plt.pause(0.001)
    
    def _update_sensor_plots(self, contacts, ax_contact, ax_force, 
                            force_history, current_force, sensor_name):
        """Update plots for a single sensor"""
        
        # Update contact patch plot
        ax_contact.clear()
        ax_contact.set_xlim(-10, 10)
        ax_contact.set_ylim(0, 20)
        
        # Redraw ROI
        roi_rect = plt.Rectangle((-7.5, 0), 15, 15, 
                               linewidth=2, edgecolor='green', 
                               facecolor='none', linestyle='--')
        ax_contact.add_patch(roi_rect)
        
        if contacts:
            contact_x = [c['x_mm'] for c in contacts]
            contact_y = [c['y_mm'] for c in contacts]
            intensities = [c['intensity'] for c in contacts]
            
            # Color by intensity (proximity weight)
            colors = plt.cm.Reds([1.0 - i/0.5 for i in intensities])
            ax_contact.scatter(contact_x, contact_y, c=colors, s=50, 
                             alpha=0.8, edgecolors='black')
            
            # ax_contact.text(-9, 18, f'Contacts: {len(contacts)}', 
            #               fontweight='bold', fontsize=10)
            ax_contact.text(-9, 16, f'Force: {current_force:.3f} N', 
                          fontweight='bold', fontsize=10)
        else:
            ax_contact.text(0, 10, 'NO CONTACT', ha='center', 
                          fontsize=14, color='gray', fontweight='bold')
        
        ax_contact.set_xlabel('X (mm)')
        ax_contact.set_ylabel('Y (mm)')
        ax_contact.set_title(f'{sensor_name} DIGIT - Contact Patch')
        ax_contact.grid(True, alpha=0.3)
        
        # Update force plot
        if len(self.time_history) > 1:
            ax_force.clear()
            ax_force.plot(self.time_history, force_history, 'b-', linewidth=2)
            ax_force.fill_between(self.time_history, force_history, 
                                 alpha=0.3, color='blue')
            ax_force.set_xlabel('Time (s)')
            ax_force.set_ylabel('Force (N)')
            ax_force.set_title(f'{sensor_name} DIGIT - Force')
            ax_force.grid(True, alpha=0.3)
    
    def print_sensor_status(self):
        """Print current sensor status to console"""
        
        left_contacts, right_contacts = self.get_sensor_contacts()
        
        # Get gripper joint positions
        try:
            left_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 
                                             "rh_p12_rn_left")
            right_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 
                                              "rh_p12_rn_right")
            left_angle = self.data.qpos[left_joint_id]
            right_angle = self.data.qpos[right_joint_id]
        except:
            left_angle = 0.0
            right_angle = 0.0
        
        # Get end effector global pose
        try:
            eef_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_site")
            eef_pos = self.data.site_xpos[eef_site_id]
            eef_mat = self.data.site_xmat[eef_site_id].reshape(3, 3)
        except:
            eef_pos = np.zeros(3)
            eef_mat = np.eye(3)
        
        print("\n" + "="*60)
        print(f"⏱️  Time: {time.time() - self.start_time:.2f}s")
        
        # Print EE global pose
        print(f"📍 EE Global Pose:")
        print(f"   Position:    [{eef_pos[0]:7.4f}, {eef_pos[1]:7.4f}, {eef_pos[2]:7.4f}]")
 
        
        # Print gripper and sensor info
        print(f"🤖 Gripper: Left={np.degrees(left_angle):.1f}°, Right={np.degrees(right_angle):.1f}°")
        print(f"👈 LEFT DIGIT:  {len(left_contacts)} contacts, Force={len(left_contacts)*0.01:.3f}N")
        print(f"👉 RIGHT DIGIT: {len(right_contacts)} contacts, Force={len(right_contacts)*0.01:.3f}N")
        
        if left_contacts:
            print(f"   Left contact range: X=[{min(c['x_mm'] for c in left_contacts):.1f}, "
                  f"{max(c['x_mm'] for c in left_contacts):.1f}]mm")
        if right_contacts:
            print(f"   Right contact range: X=[{min(c['x_mm'] for c in right_contacts):.1f}, "
                  f"{max(c['x_mm'] for c in right_contacts):.1f}]mm")
    
    def run_demo(self):
        """Run the UR5e with DIGIT sensors demo"""
        
        print("🚀 Starting UR5e with Dual DIGIT Sensors Demo")
        print(f"   Using HEXAGON peg-in-hole task")
        print("="*60)
        
        try:
            # Load and initialize
            self.load_robot_scene()
            
            # Use IK to move end effector to peg position
            self.move_eef_to_peg()
            
            self.initialize_digit_sensors()
            self.setup_visualization()
            
            print("\n✅ Demo ready!")
            print("🎛️  Use MuJoCo GUI controls to:")
            print("   - Move the robot arm")
            print("   - Control gripper fingers")
            print(f"   - Observe hexagon peg falling into hole")
            if self.enable_visualization:
                print("📊 Watch dual sensor feedback in matplotlib")
            else:
                print("📊 Matplotlib disabled (set ENABLE_MATPLOTLIB=True to enable)")
            print("🎥 3D visualization in MuJoCo viewer")
            print(f"\n💡 To switch peg types, change PEG_NAME at top of file")
            print("   Options: 'cylinder', 'cube', 'hexagon'")
            print("\nPress Ctrl+C to stop\n")
            
            # Run simulation with viewer
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                print("✓ MuJoCo viewer launched")
                
                # Set camera for better view
                viewer.cam.distance = 1.5
                viewer.cam.lookat = [0.6, 0.0, 0.85]
                viewer.cam.elevation = -30
                viewer.cam.azimuth = 135
                
                step = 0
                last_status_print = time.time()
                
                while viewer.is_running():
                    step += 1
                    
                    # Step simulation
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    
                    # Update plots every 10 steps (only if visualization enabled)
                    if step % 10 == 0 and self.enable_visualization:
                        self.update_visualization()
                    
                    # Print status every 2 seconds
                    if time.time() - last_status_print > 2.0:
                        self.print_sensor_status()
                        last_status_print = time.time()
                    
                    time.sleep(0.002)
                
                print("\n✓ MuJoCo viewer closed")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Demo stopped by user")
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            plt.close('all')

def main():
    """Main function"""
    demo = UR5eDIGITDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
