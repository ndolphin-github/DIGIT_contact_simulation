import numpy as np
import mujoco
import mujoco.viewer
import time
import tkinter as tk
from tkinter import ttk
import threading
from simple_ik import TaskSpaceController, stabilize_robot

# ============================================================================
# CONTROL CONFIGURATION
# ============================================================================
# Control resolution (slider precision)
POSITION_RESOLUTION = 0.0001  # 0.1mm steps for fine control
ORIENTATION_RESOLUTION = 0.001  # 0.001 rad steps (~0.06 degrees)
GRIPPER_RESOLUTION = 0.001  # 0.001 rad steps

# Control ranges (for sliders) - workspace limits
X_RANGE = (0.4, 0.8)  # X: 0.4 to 0.8m (400-800mm)
Y_RANGE = (-0.4, 0.4)  # Y: -0.4 to 0.4m (±400mm)
Z_RANGE = (0.5, 1.0)  # Z: 0.5 to 1.0m (500-1000mm, always positive)
ORIENTATION_RANGE = np.pi  # ±180 degrees
GRIPPER_RANGE = (0.0, 1.3)  # 0 to 1.3 rad (matches controller actuation range)

# ============================================================================

class ControlGUI:
    """GUI with sliders for 7-DOF control"""
    
    def __init__(self, initial_pos, initial_quat, initial_gripper):
        """Initialize GUI window with sliders"""
        
        self.root = tk.Tk()
        self.root.title("7-DOF Task Space Controller")
        self.root.geometry("500x600")
        
        # Convert initial quaternion to euler angles for display
        self.initial_pos = initial_pos.copy()
        self.initial_euler = self._quat_to_euler(initial_quat)
        self.initial_gripper = initial_gripper
        
        # Target values (will be read by controller)
        self.target_pos = initial_pos.copy()
        self.target_euler = self.initial_euler.copy()
        self.target_gripper = initial_gripper
        
        # Flag to prevent status updates during widget creation
        self._widgets_ready = False
        
        # Create GUI
        self._create_widgets()
        
        # Now widgets are ready, update initial status
        self._widgets_ready = True
        self._update_status()
        
        # Flag to check if GUI is running
        self.running = True
        
    def _quat_to_euler(self, quat):
        """Convert quaternion to euler angles (roll, pitch, yaw)"""
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def _euler_to_quat(self, euler):
        """Convert euler angles to quaternion"""
        roll, pitch, yaw = euler
        
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
    
    def _create_widgets(self):
        """Create all GUI widgets"""
        
        # Title
        title = tk.Label(self.root, text="7-DOF Task Space Control", 
                        font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Position sliders
        pos_frame = ttk.LabelFrame(main_frame, text="Position (m)", padding=10)
        pos_frame.pack(fill=tk.X, pady=5)
        
        self.pos_sliders = {}
        self.pos_labels = {}
        
        # Define range for each axis
        pos_ranges = [X_RANGE, Y_RANGE, Z_RANGE]
        
        for i, axis in enumerate(['X', 'Y', 'Z']):
            frame = ttk.Frame(pos_frame)
            frame.pack(fill=tk.X, pady=2)
            
            label = ttk.Label(frame, text=f"{axis}:", width=3)
            label.pack(side=tk.LEFT)
            
            value_label = ttk.Label(frame, text=f"{self.initial_pos[i]:.4f}", width=10)
            value_label.pack(side=tk.RIGHT)
            self.pos_labels[axis] = value_label
            
            # Clamp initial position to valid range
            range_min, range_max = pos_ranges[i]
            clamped_initial = np.clip(self.initial_pos[i], range_min, range_max)
            
            slider = tk.Scale(frame, from_=range_min, to=range_max,
                            orient=tk.HORIZONTAL,
                            resolution=POSITION_RESOLUTION,
                            length=300,
                            command=lambda v, ax=axis, idx=i: self._update_position(ax, idx, v))
            slider.set(clamped_initial)
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.pos_sliders[axis] = slider
        
        # Orientation sliders (Euler angles)
        ori_frame = ttk.LabelFrame(main_frame, text="Orientation (rad)", padding=10)
        ori_frame.pack(fill=tk.X, pady=5)
        
        self.ori_sliders = {}
        self.ori_labels = {}
        for i, axis in enumerate(['Roll', 'Pitch', 'Yaw']):
            frame = ttk.Frame(ori_frame)
            frame.pack(fill=tk.X, pady=2)
            
            label = ttk.Label(frame, text=f"{axis}:", width=6)
            label.pack(side=tk.LEFT)
            
            value_label = ttk.Label(frame, text=f"{self.initial_euler[i]:.3f}", width=10)
            value_label.pack(side=tk.RIGHT)
            self.ori_labels[axis] = value_label
            
            slider = tk.Scale(frame, from_=-ORIENTATION_RANGE, 
                            to=ORIENTATION_RANGE,
                            orient=tk.HORIZONTAL,
                            resolution=ORIENTATION_RESOLUTION,
                            length=300,
                            command=lambda v, ax=axis, idx=i: self._update_orientation(ax, idx, v))
            slider.set(self.initial_euler[i])
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.ori_sliders[axis] = slider
        
        # Gripper slider
        grip_frame = ttk.LabelFrame(main_frame, text="Gripper", padding=10)
        grip_frame.pack(fill=tk.X, pady=5)
        
        grip_inner = ttk.Frame(grip_frame)
        grip_inner.pack(fill=tk.X, pady=2)
        
        grip_label = ttk.Label(grip_inner, text="Angle:", width=6)
        grip_label.pack(side=tk.LEFT)
        
        # Clamp initial gripper to valid range
        clamped_gripper = np.clip(self.initial_gripper, GRIPPER_RANGE[0], GRIPPER_RANGE[1])
        
        self.grip_value_label = ttk.Label(grip_inner, 
                                         text=f"{clamped_gripper:.3f}rad", 
                                         width=10)
        self.grip_value_label.pack(side=tk.RIGHT)
        
        self.grip_slider = tk.Scale(grip_inner, from_=GRIPPER_RANGE[0], 
                                    to=GRIPPER_RANGE[1],
                                    orient=tk.HORIZONTAL,
                                    resolution=GRIPPER_RESOLUTION,
                                    length=300,
                                    command=self._update_gripper)
        self.grip_slider.set(clamped_gripper)
        self.grip_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        reset_btn = ttk.Button(button_frame, text="Reset to Home", 
                              command=self._reset_to_home)
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        current_btn = ttk.Button(button_frame, text="Use Current Pose", 
                                command=self._use_current)
        current_btn.pack(side=tk.LEFT, padx=5)
        
        # Status display
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.status_text = tk.Text(status_frame, height=8, width=50, 
                                   font=("Courier", 9))
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        # Initial status
        self._update_status()
    
    def _update_position(self, axis, idx, value):
        """Update position target from slider"""
        self.target_pos[idx] = float(value)
        self.pos_labels[axis].config(text=f"{self.target_pos[idx]:.4f}")
        if self._widgets_ready:
            self._update_status()
    
    def _update_orientation(self, axis, idx, value):
        """Update orientation target from slider"""
        self.target_euler[idx] = float(value)
        self.ori_labels[axis].config(text=f"{self.target_euler[idx]:.3f}")
        if self._widgets_ready:
            self._update_status()
    
    def _update_gripper(self, value):
        """Update gripper target from slider"""
        self.target_gripper = float(value)
        self.grip_value_label.config(text=f"{self.target_gripper:.3f}rad")
        if self._widgets_ready:
            self._update_status()
    
    def _update_status(self):
        """Update status display"""
        quat = self._euler_to_quat(self.target_euler)
        
        status = f"Target Position:\n"
        status += f"  X: {self.target_pos[0]:7.4f} m ({self.target_pos[0]*1000:.1f} mm)\n"
        status += f"  Y: {self.target_pos[1]:7.4f} m ({self.target_pos[1]*1000:.1f} mm)\n"
        status += f"  Z: {self.target_pos[2]:7.4f} m ({self.target_pos[2]*1000:.1f} mm)\n\n"
        status += f"Target Orientation:\n"
        status += f"  Roll:  {self.target_euler[0]:7.4f} rad ({np.degrees(self.target_euler[0]):6.2f}°)\n"
        status += f"  Pitch: {self.target_euler[1]:7.4f} rad ({np.degrees(self.target_euler[1]):6.2f}°)\n"
        status += f"  Yaw:   {self.target_euler[2]:7.4f} rad ({np.degrees(self.target_euler[2]):6.2f}°)\n\n"
        status += f"Gripper: {self.target_gripper:.3f} rad"
        
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(1.0, status)
    
    def _reset_to_home(self):
        """Reset all sliders to initial values"""
        for i, axis in enumerate(['X', 'Y', 'Z']):
            self.pos_sliders[axis].set(self.initial_pos[i])
        for i, axis in enumerate(['Roll', 'Pitch', 'Yaw']):
            self.ori_sliders[axis].set(self.initial_euler[i])
        self.grip_slider.set(self.initial_gripper)
    
    def _use_current(self):
        """This will be set by external controller"""
        pass  # Placeholder for external callback
    
    def get_target_pose(self):
        """Get current target from GUI"""
        quat = self._euler_to_quat(self.target_euler)
        return self.target_pos.copy(), quat, self.target_gripper
    
    def update_current_pose(self, pos, quat, gripper):
        """Update display with current robot pose (optional)"""
        # Could add current pose display if needed
        pass
    
    def run(self):
        """Start GUI main loop (must be called from main thread)"""
        self.running = True
        self.root.mainloop()
        self.running = False  # Set to False when GUI closes


class PegInHoleTaskController:
    """Interactive 7-DOF controller for peg-in-hole task"""
    
    def __init__(self, xml_path="ur5e_with_DIGIT_primitive_hexagon.xml"):
        """Initialize the controller and scene"""
        
        print("🚀 Loading peg-in-hole scene...")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Set robot to default initial joint configuration
        from simple_ik import DEFAULT_INITIAL_JOINTS
        joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        for i, name in enumerate(joint_names):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.data.qpos[joint_id] = DEFAULT_INITIAL_JOINTS[i]
        
        # Initialize simulation
        mujoco.mj_forward(self.model, self.data)
        
        # Create task space controller
        print("\n🎮 Initializing 7-DOF Task Space Controller...")
        self.controller = TaskSpaceController(self.model, self.data)

        # Initialize GUI with current robot pose (so sliders start at current state)
        try:
            state = self.controller.get_current_task_space_state()
            init_pos = state['position']
            init_quat = state['quaternion']
            init_gripper_angle = state['gripper_angle']
        except Exception:
            # Fallback defaults if state query fails
            init_pos = np.array([0.5, 0.0, 0.85])
            init_quat = np.array([0.0, 1.0, 0.0, 0.0])
            init_gripper_angle = 0.3

        # Create GUI instance (ControlGUI defined above)
        self.gui = ControlGUI(init_pos, init_quat, init_gripper_angle)
        
        # Get peg body ID for monitoring
        try:
            self.peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hexagon_peg_body")
            self.has_peg = True
        except:
            print("⚠️  Warning: Hexagon peg not found in scene")
            self.has_peg = False
        
        # Control state
        self.running = True
        self.last_status_print = time.time()
        self.status_print_interval = 2.0  # Print status every 2 seconds
        

    

    def reset_to_home(self):
        """Reset robot to home position above peg"""
        print("\n🏠 Resetting to home position...")
        
        if self.has_peg:
            # Get peg position
            peg_pos = self.data.xpos[self.peg_body_id].copy()
            
            # Target: 8cm above peg, gripper pointing down
            home_pos = peg_pos.copy()
            home_pos[2] += 0.08  # 8cm above peg
            home_quat = np.array([0.0, 1.0, 0.0, 0.0])  # Gripper pointing down
            gripper_angle = 0.3  # Partially open
            
            print(f"  Target position: {home_pos}")
            print(f"  Peg position: {peg_pos}")
        else:
            # Default home position
            home_pos = np.array([0.5, 0.1, 0.85])
            home_quat = np.array([0.0, 1.0, 0.0, 0.0])
            gripper_angle = 0.3
        
        # Set target and execute
        self.controller.set_target_task_space(
            position=home_pos,
            quaternion=home_quat,
            gripper_angle=gripper_angle
        )
        success = self.controller.update_control(step_ik=True)
        
        if success:
            print("✅ Reset complete")
            stabilize_robot(self.model, self.data, num_steps=100)
        else:
            print("⚠️  Reset may not be exact, but robot moved")
        
        self.controller.print_status()
    
    def _simulation_loop(self):
        """Background thread: run MuJoCo simulation and update control"""
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            
            # Configure camera
            viewer.cam.distance = 1.5
            viewer.cam.lookat = [0.5, 0.0, 0.85]
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 135
            
            last_update = time.time()
            update_interval = 0.2  # Update control every 200ms (reduce shakiness)
            
            print("✓ MuJoCo viewer launched - use GUI to control robot")
            
            while viewer.is_running() and self.gui.running:
                
                # Update control from GUI at specified interval
                if time.time() - last_update > update_interval:
                    try:
                        # Get target from GUI
                        target_pos, target_quat, target_gripper = self.gui.get_target_pose()
                        
                        # Set target
                        self.controller.set_target_task_space(
                            position=target_pos,
                            quaternion=target_quat,
                            gripper_angle=target_gripper
                        )
                        
                        # Update control (IK + actuators)
                        self.controller.update_control(step_ik=True)
                    except Exception as e:
                        print(f"⚠️  Control update error: {e}")
                    
                    last_update = time.time()
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                
                time.sleep(0.001)
            
            print("\n✓ MuJoCo viewer closed")
            self.gui.running = False
    
    def run_interactive(self):
        """Run interactive control: GUI in main thread, simulation in background"""
        
        print("\n🎬 Starting interactive simulation...")
        print("   GUI will open shortly - use sliders to control robot\n")
        
        # Start simulation in background thread (NOT GUI - tkinter must be in main)
        sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        sim_thread.start()
        
        # Wait a moment for MuJoCo to initialize
        time.sleep(0.5)
        
        # Run GUI in main thread (tkinter requirement)
        self.gui.run()
        
        print("\n✓ GUI closed")
    
def main():
    """Main function"""
    
    print("\n" + "="*70)
    print("🎮 7-DOF TASK SPACE CONTROLLER")
    print("="*70)
    
    # Create controller
    demo = PegInHoleTaskController()
    demo.run_interactive()


if __name__ == "__main__":
    main()
