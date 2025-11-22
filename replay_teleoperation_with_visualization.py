"""
Replay Teleoperation Data with Real-time Sensor Visualization
• Loads CSV session data from Teleoperation_sensor_data
• Replays joint angles and gripper movements in simulation
• Displays LEFT and RIGHT DIGIT sensor contact patches in real-time
• Shows high-resolution FEM grid heatmaps for both sensors
• Synchronized timesteps with original recording
"""

import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import time
import csv
import os
import msvcrt
from datetime import datetime

from simple_ik_legacy import DEFAULT_INITIAL_JOINTS
from gripper_digit_sensor import GripperDIGITSensor


# ============================================================================
# CONFIGURATION: Set your CSV file path here
# ============================================================================
CSV_FILE_PATH = "Teleoperation_sensor_data/Trial1_peg_pos1/trial2.csv"
PLAYBACK_SPEED = 4.0  # 1.0 = real-time, 2.0 = 2x speed, 0.5 = slow motion

# Visualization settings
ENABLE_PLOT_VISUALIZATION =True  # True = Show matplotlib plots, False = No plots (faster)
PLOT_UPDATE_INTERVAL = 5  # Update plots every N simulation steps (lower = more frequent updates)
ENABLE_FEM_VISUALIZATION = True  # Show high-res FEM grid heatmaps
# ============================================================================


class TeleoperationReplayWithVisualization:
    """Replay recorded teleoperation data with real-time sensor visualization"""
    
    def __init__(self, xml_path="ur5e_with_DIGIT_primitive_hexagon.xml"):
        """Initialize replay system with visualization"""
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
            print(f"✅ Gripper actuator IDs: {self.gripper_actuator_ids}")
        except:
            self.has_gripper = False
            self.gripper_actuator_ids = []
            print("⚠️ Gripper actuators not found")
        
        # Initialize DIGIT sensors
        try:
            self.digit_left = GripperDIGITSensor(self.model, "digit_geltip_left", "left")
            self.digit_right = GripperDIGITSensor(self.model, "digit_geltip_right", "right")
            self.has_digit_sensors = True
            print("✅ DIGIT sensors initialized")
            
            # Load FEM grid for high-resolution visualization
            self.fem_grid = self.digit_left.load_fem_grid('filtered_FEM_grid.csv')
            if self.fem_grid is not None:
                print(f"✅ FEM grid loaded: {len(self.fem_grid)} nodes")
            else:
                print("⚠️ FEM grid not loaded - will use sparse contact data only")
                
        except Exception as e:
            self.has_digit_sensors = False
            self.fem_grid = None
            print(f"⚠️ DIGIT sensors not initialized: {e}")
        
        # Replay data
        self.replay_data = None
        self.current_frame = 0
        self.start_time = 0.0
        self.playback_speed = PLAYBACK_SPEED
        self.is_paused = False
        
        # Visualization settings
        self.plot_update_counter = 0
        self.update_interval = PLOT_UPDATE_INTERVAL
        self.enable_plot_visualization = ENABLE_PLOT_VISUALIZATION  # Control plot display
        self.enable_fem_visualization = ENABLE_FEM_VISUALIZATION  # Legacy FEM setting
        
        # Matplotlib components
        self.fig = None
        self.axes = {}  # Dictionary to store all axes
        self._colorbars = {}  # Store colorbars for updates
        
        print("✅ Teleoperation Replay System with Visualization initialized")
    
    def load_csv_data(self, csv_path):
        """Load recorded session data from CSV file"""
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            return False
        
        print(f"\n📂 Loading CSV data: {csv_path}")
        
        data_rows = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            for row in reader:
                timestamp = float(row[0])
                gripper_value = float(row[1])
                joint_angles = [float(row[i]) for i in range(2, 8)]
                
                # Parse sensor data if available
                left_sensor_data = None
                right_sensor_data = None
                if len(row) >= 5112:
                    left_sensor_data = np.array([float(row[i]) for i in range(8, 2560)])
                    right_sensor_data = np.array([float(row[i]) for i in range(2560, 5112)])
                
                data_rows.append({
                    'timestamp': timestamp,
                    'gripper_value': gripper_value,
                    'joint_angles': np.array(joint_angles),
                    'left_sensor': left_sensor_data,
                    'right_sensor': right_sensor_data
                })
        
        self.replay_data = data_rows
        print(f"✅ Loaded {len(data_rows)} frames")
        print(f"   Duration: {data_rows[-1]['timestamp'] - data_rows[0]['timestamp']:.2f} seconds")
        print(f"   First timestamp: {data_rows[0]['timestamp']:.3f}s")
        print(f"   Last timestamp: {data_rows[-1]['timestamp']:.3f}s")
        
        return True
    
    def setup_visualization(self):
        """Setup matplotlib window with 4 subplots (2x2 grid)"""
        if not self.enable_plot_visualization:
            print("📊 Plot visualization is DISABLED (ENABLE_PLOT_VISUALIZATION = False)")
            return
        
        self.fig = plt.figure(figsize=(14, 10))
        gs = self.fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3, height_ratios=[3, 1])
        
        # Top-left: LEFT sensor FEM grid heatmap
        self.axes['left_fem'] = self.fig.add_subplot(gs[0, 0])
        self._setup_fem_axis(self.axes['left_fem'], 'LEFT Sensor - FEM Grid (2552 nodes)')
        
        # Top-right: RIGHT sensor FEM grid heatmap
        self.axes['right_fem'] = self.fig.add_subplot(gs[0, 1])
        self._setup_fem_axis(self.axes['right_fem'], 'RIGHT Sensor - FEM Grid (2552 nodes)')
        
        # Bottom-left: LEFT sensor 1x2552 vector bar
        self.axes['left_vector'] = self.fig.add_subplot(gs[1, 0])
        self._setup_vector_axis(self.axes['left_vector'], 'LEFT Sensor - Distance Vector (2552 values)')
        
        # Bottom-right: RIGHT sensor 1x2552 vector bar
        self.axes['right_vector'] = self.fig.add_subplot(gs[1, 1])
        self._setup_vector_axis(self.axes['right_vector'], 'RIGHT Sensor - Distance Vector (2552 values)')
        
        plt.ion()  # Interactive mode
        plt.show(block=False)
        
        print("✅ Visualization setup complete:")
        print("   - TOP: LEFT & RIGHT sensor FEM grids")
        print("   - BOTTOM: LEFT & RIGHT sensor distance vectors (1x2552)")
    
    def _setup_contact_axis(self, ax, title):
        """Setup a contact patch axis"""
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, 20)
        ax.set_xlabel('X (mm)', fontsize=10)
        ax.set_ylabel('Y (mm)', fontsize=10)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Add sensor ROI boundary
        roi_rect = plt.Rectangle((-7.5, 0), 15, 20, 
                               linewidth=2, edgecolor='green', 
                               facecolor='none', linestyle='--')
        ax.add_patch(roi_rect)
    
    def _setup_fem_axis(self, ax, title):
        """Setup a FEM grid axis"""
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, 20)
        ax.set_xlabel('X (mm)', fontsize=10)
        ax.set_ylabel('Y (mm)', fontsize=10)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_aspect('equal', adjustable='box')
    
    def _setup_vector_axis(self, ax, title):
        """Setup a 1x2552 vector bar axis"""
        ax.set_xlim(0, 2552)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Node Index', fontsize=10)
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_aspect('auto')
    
    def update_visualization(self):
        """Update all sensor visualizations"""
        if not self.has_digit_sensors or not self.enable_plot_visualization:
            return
        
        # Get contact data from both sensors
        left_contacts = self.digit_left.detect_proximity_contacts(self.data)
        right_contacts = self.digit_right.detect_proximity_contacts(self.data)
        
        # Update LEFT sensor plots
        if self.enable_fem_visualization and self.fem_grid is not None:
            left_fem_field = self._update_fem_plot(
                self.axes['left_fem'],
                left_contacts,
                'left_fem',
                'LEFT'
            )
            # Update LEFT vector bar
            self._update_vector_plot(
                self.axes['left_vector'],
                left_fem_field,
                'left_vector',
                'LEFT'
            )
        
        # Update RIGHT sensor plots
        if self.enable_fem_visualization and self.fem_grid is not None:
            right_fem_field = self._update_fem_plot(
                self.axes['right_fem'],
                right_contacts,
                'right_fem',
                'RIGHT'
            )
            # Update RIGHT vector bar
            self._update_vector_plot(
                self.axes['right_vector'],
                right_fem_field,
                'right_vector',
                'RIGHT'
            )
        
        # Refresh display
        plt.draw()
        plt.pause(0.001)
    
    def _update_contact_plot(self, ax, contacts, ax_key, sensor_name):
        """Update contact patch plot for a sensor"""
        ax.clear()
        self._setup_contact_axis(ax, f'{sensor_name} Sensor - Contact Patch')
        
        if contacts and len(contacts) > 0:
            contact_x = [c['x_mm'] for c in contacts]
            contact_y = [c['y_mm'] for c in contacts]
            distances = [c['distance_from_plane_mm'] for c in contacts]
            
            scatter = ax.scatter(contact_x, contact_y,
                               c=distances, cmap='plasma',
                               s=30, alpha=0.8, edgecolors='black',
                               vmin=0, vmax=0.3)
            
            # Update or create colorbar
            if ax_key not in self._colorbars or self._colorbars[ax_key] is None:
                self._colorbars[ax_key] = plt.colorbar(scatter, ax=ax, label='Distance (mm)')
            else:
                self._colorbars[ax_key].update_normal(scatter)
            
            # Stats text
            min_dist = min(distances)
            ax.text(-9, 18.5, f'Contacts: {len(contacts)}', 
                   fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.text(-9, 17, f'Min: {min_dist:.4f}mm', 
                   fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0, 10, 'NO CONTACT', ha='center', fontsize=14, 
                   color='gray', fontweight='bold')
    
    def _update_fem_plot(self, ax, contacts, ax_key, sensor_name):
        """Update FEM grid heatmap for a sensor"""
        ax.clear()
        self._setup_fem_axis(ax, f'{sensor_name} Sensor - FEM Grid (2552 nodes)')
        
        fem_distance_field = None  # Return value for vector plot
        
        if contacts and len(contacts) > 0 and self.fem_grid is not None:
            # Interpolate to FEM grid
            if sensor_name == 'LEFT':
                fem_distance_field = self.digit_left.interpolate_to_fem_grid(
                    contacts, self.fem_grid, influence_radius_mm=0.2
                )
            else:
                fem_distance_field = self.digit_right.interpolate_to_fem_grid(
                    contacts, self.fem_grid, influence_radius_mm=0.2
                )
            
            # Create mask for active nodes
            contact_mask = fem_distance_field > 1e-6
            
            if np.any(contact_mask):
                scatter_fem = ax.scatter(
                    self.fem_grid['x'][contact_mask],
                    self.fem_grid['y'][contact_mask],
                    c=fem_distance_field[contact_mask],
                    cmap='plasma',
                    s=8,
                    vmin=0,
                    vmax=0.3,
                    alpha=1.0,
                    edgecolors='none'
                )
                
                # Update or create colorbar
                fem_key = f'{ax_key}_colorbar'
                if fem_key not in self._colorbars or self._colorbars[fem_key] is None:
                    self._colorbars[fem_key] = plt.colorbar(scatter_fem, ax=ax, label='Distance (mm)')
                else:
                    self._colorbars[fem_key].update_normal(scatter_fem)
                
                # Stats text
                active_nodes = np.sum(contact_mask)
                max_dist = np.max(fem_distance_field[contact_mask])
                ax.text(-9, 18.5, f'Active: {active_nodes}/{len(self.fem_grid)}',
                       fontweight='bold', fontsize=10, color='white',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                ax.text(-9, 17, f'Max: {max_dist:.4f}mm',
                       fontweight='bold', fontsize=10, color='white',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        else:
            ax.text(0, 10, 'NO CONTACT', ha='center', fontsize=14,
                   color='gray', fontweight='bold')
            # Return zeros if no contact
            if self.fem_grid is not None:
                fem_distance_field = np.zeros(len(self.fem_grid))
        
        return fem_distance_field
    
    def _update_vector_plot(self, ax, distance_vector, ax_key, sensor_name):
        """Update 1x2552 vector bar visualization"""
        ax.clear()
        self._setup_vector_axis(ax, f'{sensor_name} Sensor - Distance Vector (2552 values)')
        
        if distance_vector is not None and len(distance_vector) == 2552:
            # Create horizontal bar using imshow
            # Reshape to (1, 2552) for horizontal display
            vector_2d = distance_vector.reshape(1, -1)
            
            im = ax.imshow(vector_2d, 
                          cmap='plasma',
                          aspect='auto',
                          vmin=0,
                          vmax=0.3,
                          extent=[0, 2552, 0, 1],
                          interpolation='nearest')
            
            # Update or create colorbar
            vec_key = f'{ax_key}_colorbar'
            if vec_key not in self._colorbars or self._colorbars[vec_key] is None:
                self._colorbars[vec_key] = plt.colorbar(im, ax=ax, 
                                                        orientation='horizontal',
                                                        label='Distance (mm)',
                                                        pad=0.15)
            else:
                self._colorbars[vec_key].update_normal(im)
            
            # Stats
            active_count = np.sum(distance_vector > 1e-6)
            max_val = np.max(distance_vector)
            # ax.text(1276, 0.5, f'Active: {active_count}/2552 | Max: {max_val:.4f}mm',
            #        ha='center', va='center', fontweight='bold', fontsize=9,
            #        color='white' if max_val > 0.15 else 'black',
            #        bbox=dict(boxstyle='round', facecolor='black' if max_val > 0.15 else 'white', 
            #                alpha=0.7))
        else:
            ax.text(1276, 0.5, 'NO DATA', ha='center', va='center',
                   fontsize=12, color='gray', fontweight='bold')
    
    def set_initial_pose(self):
        """Set robot to initial joint configuration"""
        if self.has_peg:
            peg_qpos_addr = self.model.body_jntadr[self.peg_body_id]
            self.data.qpos[peg_qpos_addr:peg_qpos_addr+3] = [0.4, 0.2, 0.81]
            self.data.qpos[peg_qpos_addr+3:peg_qpos_addr+7] = [1, 0, 0, 0]
            peg_qvel_addr = self.model.body_dofadr[self.peg_body_id]
            self.data.qvel[peg_qvel_addr:peg_qvel_addr+6] = 0.0
        
        if self.replay_data is not None and len(self.replay_data) > 0:
            first_frame = self.replay_data[0]
            for i, jid in enumerate(self.joint_ids):
                self.data.qpos[jid] = first_frame['joint_angles'][i]
                self.data.ctrl[self.actuator_ids[i]] = first_frame['joint_angles'][i]
                self.data.qvel[jid] = 0.0
            
            if self.has_gripper:
                for grip_id in self.gripper_actuator_ids:
                    self.data.ctrl[grip_id] = first_frame['gripper_value']
        else:
            for i, jid in enumerate(self.joint_ids):
                self.data.qpos[jid] = DEFAULT_INITIAL_JOINTS[i]
                self.data.ctrl[self.actuator_ids[i]] = DEFAULT_INITIAL_JOINTS[i]
                self.data.qvel[jid] = 0.0
            
            if self.has_gripper:
                for grip_id in self.gripper_actuator_ids:
                    self.data.ctrl[grip_id] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
    
    def get_frame_at_time(self, elapsed_time):
        """Get the appropriate frame index for the given elapsed time"""
        if self.replay_data is None or len(self.replay_data) == 0:
            return None
        
        adjusted_time = elapsed_time * self.playback_speed
        target_timestamp = self.replay_data[0]['timestamp'] + adjusted_time
        
        if target_timestamp >= self.replay_data[-1]['timestamp']:
            return len(self.replay_data) - 1
        
        for i, frame in enumerate(self.replay_data):
            if frame['timestamp'] >= target_timestamp:
                return i
        
        return len(self.replay_data) - 1
    
    def apply_frame(self, frame_index):
        """Apply joint angles and gripper value from a specific frame"""
        if self.replay_data is None or frame_index >= len(self.replay_data):
            return
        
        frame = self.replay_data[frame_index]
        
        for i, jid in enumerate(self.joint_ids):
            target_angle = frame['joint_angles'][i]
            self.data.ctrl[self.actuator_ids[i]] = target_angle
        
        if self.has_gripper:
            for grip_id in self.gripper_actuator_ids:
                self.data.ctrl[grip_id] = frame['gripper_value']
        
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
        mujoco.mj_forward(self.model, self.data)
        self.current_frame = frame_index
    
    def run_replay(self, csv_path):
        """Main replay loop with visualization"""
        if not self.load_csv_data(csv_path):
            return
        
        self.set_initial_pose()
        
        print("Stabilizing physics simulation...")
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
        print("✓ Physics stabilized")
        
        # Setup visualization (only if enabled)
        if self.has_digit_sensors and self.enable_plot_visualization:
            self.setup_visualization()
        
        print("\n" + "="*70)
        print("TELEOPERATION REPLAY WITH SENSOR VISUALIZATION")
        print("="*70)
        print(f"Loaded: {csv_path}")
        print(f"Frames: {len(self.replay_data)}")
        print(f"Duration: {self.replay_data[-1]['timestamp'] - self.replay_data[0]['timestamp']:.2f}s")
        print(f"Playback Speed: {self.playback_speed}x")
        print(f"Plot Visualization: {'Enabled' if self.enable_plot_visualization else 'Disabled'}")
        print(f"Plot Update Interval: every {self.update_interval} steps")
        print(f"FEM Visualization: {'Enabled' if self.enable_fem_visualization else 'Disabled'}")
        print("="*70)
        print("\n🎬 Replay will start automatically in 1 second...")
        print("="*70 + "\n")
        
        self.current_frame = 0
        self.is_paused = True  # Start paused briefly
        self.plot_update_counter = 0
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("✅ Viewer started. Waiting 1 second before replay...\n")
            
            # Wait 1 second before starting
            time.sleep(10.0)
            
            self.is_paused = False
            self.start_time = time.time()
            print("▶️  Replay started!\n")
            
            last_update_time = 0
            
            while viewer.is_running():
                current_time = time.time()
                
                if not self.is_paused:
                    elapsed = current_time - self.start_time
                    target_frame = self.get_frame_at_time(elapsed)
                    
                    if target_frame is not None:
                        if target_frame >= len(self.replay_data) - 1:
                            if self.current_frame != len(self.replay_data) - 1:
                                self.apply_frame(len(self.replay_data) - 1)
                                print(f"\n✅ Replay finished!")
                                print(f"   Final frame: {len(self.replay_data)}/{len(self.replay_data)} (100%)")
                                print(f"   Final timestamp: {self.replay_data[-1]['timestamp']:.2f}s")
                                
                                # Final visualization update
                                if self.has_digit_sensors:
                                    self.update_visualization()
                                
                                print("   Close viewer to exit")
                            self.is_paused = True
                        else:
                            if target_frame != self.current_frame:
                                self.apply_frame(target_frame)
                                
                                # Update visualization periodically
                                self.plot_update_counter += 1
                                if self.has_digit_sensors and self.plot_update_counter >= self.update_interval:
                                    self.update_visualization()
                                    self.plot_update_counter = 0
                                
                                if current_time - last_update_time > 1.0:
                                    progress = (self.current_frame / len(self.replay_data)) * 100
                                    print(f"⏱️  Frame {self.current_frame}/{len(self.replay_data)} ({progress:.1f}%) - "
                                          f"Timestamp: {self.replay_data[self.current_frame]['timestamp']:.2f}s - "
                                          f"Speed: {self.playback_speed}x")
                                    last_update_time = current_time
                    else:
                        print("\n⚠️ Error: Could not determine target frame")
                        self.is_paused = True
                else:
                    mujoco.mj_step(self.model, self.data)
                
                viewer.sync()
                time.sleep(0.01)
        
        print("\n✅ Replay stopped")
        
        # Keep matplotlib window open (only if it was created)
        if self.has_digit_sensors and self.enable_plot_visualization:
            print("📊 Matplotlib window still open. Close it manually to exit.")
            plt.ioff()
            plt.show()


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("TELEOPERATION REPLAY WITH SENSOR VISUALIZATION")
    print("="*70)
    print(f"\n📂 CSV File: {CSV_FILE_PATH}")
    print(f"⏱️  Playback Speed: {PLAYBACK_SPEED}x")
    print(f"📊 Plot Visualization: {'Enabled' if ENABLE_PLOT_VISUALIZATION else 'Disabled'}")
    print(f"📊 Plot Update Interval: every {PLOT_UPDATE_INTERVAL} steps")
    print(f"📈 FEM Visualization: {'Enabled' if ENABLE_FEM_VISUALIZATION else 'Disabled'}")
    
    if not os.path.exists(CSV_FILE_PATH):
        print(f"\n❌ File not found: {CSV_FILE_PATH}")
        print("\n💡 Tip: Edit the CSV_FILE_PATH variable at the top of this script")
        return
    
    replay = TeleoperationReplayWithVisualization()
    replay.run_replay(CSV_FILE_PATH)


if __name__ == "__main__":
    main()
