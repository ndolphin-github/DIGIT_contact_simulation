import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import threading
import time
import pandas as pd
from scipy.interpolate import RBFInterpolator
from modular_digit_sensor import ModularDIGITSensor, UniversalMeshExtractor

# ===== PERFORMANCE CONFIGURATION =====
# Adjust these settings based on your system performance
PLOT_UPDATE_INTERVAL = 10  # Update plots every N simulation steps (10 = ~6fps plot updates at 60fps sim)
ENABLE_FIELD_VISUALIZATION = False  # Set False to skip expensive 7509-node field computation
FIELD_FAST_MODE = True  # Use fast vectorized computation (recommended: True)
# =====================================

class MuJoCoControlledDIGITDemo:
    """DIGIT GelTip demo using MuJoCo's built-in GUI controls"""
    
    def __init__(self):
        self.model = None
        self.data = None
        self.digit_sensor = None
        self.indenter_extractor = None
        
        # Indenter position (adjustable in script)
        self.indenter_x_mm = 0.0
        self.indenter_y_mm = 7.5
        
        # FEM grid for high-resolution visualization (loaded by sensor)
        self.fem_grid = None
        
        # Matplotlib components
        self.fig = None
        self.ax_contact = None
        self.ax_field = None
        
        # Colorbars (to update them properly)
        self._contact_colorbar = None
        self._field_colorbar = None
        
        # Update control (configured at top of file)
        self.plot_update_counter = 0
        self.update_interval = PLOT_UPDATE_INTERVAL
        self.enable_field_visualization = ENABLE_FIELD_VISUALIZATION
        self.field_fast_mode = FIELD_FAST_MODE
    
    def create_scene_with_controls(self):
        """Create scene with MuJoCo actuators for GUI control"""
        
        xml_content = f"""<?xml version="1.0" ?>
<mujoco model="digit_geltip_demo">
    <compiler angle="radian" meshdir="mesh/"/>
    <option timestep="0.002" gravity="0 0 0"/>
    
    <default>
        <geom contype="1" conaffinity="1" condim="6"/>
        <joint damping="10.0"/>
    </default>
    
    <asset>
        <mesh name="digit_geltip" file="DIGIT_GelTip.STL" scale="0.001 0.001 0.001"/>
        <mesh name="indenter" file="indenters/square.STL" scale="0.001 0.001 0.001"/>
    </asset>
    
    <worldbody>
        <light pos="0.02 0.02 0.05"/>
        <geom name="ground" type="plane" size="0.05 0.05 0.01" rgba="0.1 0.1 0.1 1" pos="0 0 -0.02"/>
        
        <!-- DIGIT GelTip with clear position control -->
        <body name="digit_sensor" pos="0 0 0">
            <joint name="sensor_x" type="slide" axis="1 0 0" range="-0.02 0.02" damping="10.0"/>
            <joint name="sensor_y" type="slide" axis="0 1 0" range="-0.02 0.02" damping="10.0"/>
            <joint name="sensor_z" type="slide" axis="0 0 1" range="-0.01 0.02" damping="10.0"/>
            <joint name="sensor_rx" type="hinge" axis="1 0 0" range="-0.5 0.5" damping="10.0"/>
            <joint name="sensor_ry" type="hinge" axis="0 1 0" range="-0.5 0.5" damping="10.0"/>
            <joint name="sensor_rz" type="hinge" axis="0 0 1" range="-0.5 0.5" damping="10.0"/>
            <geom name="digit_sensor_geom" type="mesh" mesh="digit_geltip" rgba="0.6 0.6 0.6 1.0"/>
        </body>
        
        <!-- Indenter with clear Z position control -->
        <body name="indenter_body" pos="{self.indenter_x_mm/1000} {self.indenter_y_mm/1000} 0.006">
            <joint name="indenter_z" type="slide" axis="0 0 1" range="-0.002 0.002" damping="10.0"/>
            <geom name="indenter_geom" type="mesh" mesh="indenter" rgba="0.8 0.2 0.2 0.8"/>
        </body>
    </worldbody>
    
    <actuator>
        <!-- Direct position control for sensor (no acceleration, direct movement) -->
        <position name="sensor_x_pos" joint="sensor_x" kp="1000" ctrlrange="-0.02 0.02"/>
        <position name="sensor_y_pos" joint="sensor_y" kp="1000" ctrlrange="-0.02 0.02"/>
        <position name="sensor_z_pos" joint="sensor_z" kp="1000" ctrlrange="-0.01 0.02"/>
        <position name="sensor_roll" joint="sensor_rx" kp="1000" ctrlrange="-0.5 0.5"/>
        <position name="sensor_pitch" joint="sensor_ry" kp="1000" ctrlrange="-0.5 0.5"/>
        <position name="sensor_yaw" joint="sensor_rz" kp="1000" ctrlrange="-0.5 0.5"/>
        
        <!-- Direct position control for indenter -->
        <position name="indenter_z_pos" joint="indenter_z" kp="1000" ctrlrange="-0.0022 0.002"/>
    </actuator>
</mujoco>"""
        
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.data = mujoco.MjData(self.model)
        
        # Set initial positions explicitly
        self.data.qpos[:] = 0.0  # All joints at zero position
        self.data.qvel[:] = 0.0  # All velocities at zero
        
        # Forward dynamics to update positions
        mujoco.mj_forward(self.model, self.data)
        
        print(f"✓ Scene created with direct position controls (no acceleration)")
        print(f"  DIGIT_GelTip.STL at origin 0,0,0")
        print(f"  Indenter at X={self.indenter_x_mm}mm, Y={self.indenter_y_mm}mm, Z=6mm")
        print(f"  Controls: sensor_x_pos, sensor_y_pos, sensor_z_pos")
        print(f"           sensor_roll, sensor_pitch, sensor_yaw, indenter_z_pos")
        
    def initialize_sensors(self):
        """Initialize DIGIT sensor with enhanced features and indenter extractor"""
        
        self.digit_sensor = ModularDIGITSensor(
            model=self.model,
            sensor_body_name="digit_sensor",
        )
        
        # CRITICAL: Initialize rest positions BEFORE any contact occurs
        print("Initializing gel rest positions...")
        self.digit_sensor.initialize_gel_rest_positions(self.data)
        
        self.indenter_extractor = UniversalMeshExtractor(
            model=self.model,
            object_geom_name="indenter_geom"
        )
        
        # Load FEM grid for high-resolution visualization
        self.fem_grid = self.digit_sensor.load_fem_grid()
        
    def setup_simple_plots(self):
        """Setup matplotlib window: contact patch and FEM grid heatmap side-by-side"""
        
        self.fig = plt.figure(figsize=(10, 5))
        gs = self.fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
        
        # Left: Contact patch with intensity coloring (sparse MuJoCo data)
        self.ax_contact = self.fig.add_subplot(gs[0, 0])
        self.ax_contact.set_xlim(-10, 10)
        self.ax_contact.set_ylim(0, 20)
        self.ax_contact.set_xlabel('X (mm)', fontsize=12)
        self.ax_contact.set_ylabel('Y (mm)', fontsize=12)
        self.ax_contact.set_title('Contact Patch (Sparse MuJoCo ~1000 nodes)', fontweight='bold', fontsize=12)
        self.ax_contact.grid(True, alpha=0.3)
        self.ax_contact.set_aspect('equal', adjustable='box')
        
        # Add sensor ROI boundary
        roi_rect = plt.Rectangle((-7.5, 0), 15, 20, 
                               linewidth=2, edgecolor='green', 
                               facecolor='none', linestyle='--')
        self.ax_contact.add_patch(roi_rect)
        
        # Right: High-resolution FEM grid heatmap (interpolated dense data)
        self.ax_field = self.fig.add_subplot(gs[0, 1])
        self.ax_field.set_xlim(-10, 10)
        self.ax_field.set_ylim(0, 20)
        self.ax_field.set_xlabel('X (mm)', fontsize=12)
        self.ax_field.set_ylabel('Y (mm)', fontsize=12)
        self.ax_field.set_title('High-Res Distance Field (FEM Grid 2553 nodes)', fontweight='bold', fontsize=12)
        self.ax_field.set_aspect('equal', adjustable='box')
        
        plt.ion()  # Interactive mode
        plt.show(block=False)
        
        print("✓ Enhanced plots setup:")
        print("  - Contact patch with sparse MuJoCo contacts (left)")
        print("  - High-resolution FEM grid heatmap (right)")
        
    def update_plots(self):
        """Update both plots with enhanced sensor data"""
        
        # Get contact data
        world_vertices = self.indenter_extractor.get_world_vertices(self.data)
        contacts = self.digit_sensor.detect_proximity_contacts(world_vertices, self.data)
        
        # Get enhanced sensor data (optimized - field disabled)
        stats = self.digit_sensor.get_contact_statistics(contacts)
        
        # ===== 1. Contact Patch Plot (colored by intensity) =====
        self.ax_contact.clear()
        self.ax_contact.set_xlim(-10, 10)
        self.ax_contact.set_ylim(0, 20)
        
        # Redraw ROI boundary
        roi_rect = plt.Rectangle((-7.5, 0), 15, 20, 
                               linewidth=2, edgecolor='green', 
                               facecolor='none', linestyle='--')
        self.ax_contact.add_patch(roi_rect)
        
        if contacts:
            contact_x = [c['x_mm'] for c in contacts]
            contact_y = [c['y_mm'] for c in contacts]
            intensities = [c['intensity'] for c in contacts]
            
            # Color by distance (same as FEM grid for consistency)
            distances = [c['distance_from_plane_mm'] for c in contacts]
            
            scatter = self.ax_contact.scatter(contact_x, contact_y, 
                                             c=distances, cmap='plasma',
                                             s=30, alpha=0.8, edgecolors='black',
                                             vmin=0, vmax=0.3)
            
            # Add colorbar
            if not hasattr(self, '_contact_colorbar') or self._contact_colorbar is None:
                self._contact_colorbar = plt.colorbar(scatter, ax=self.ax_contact, label='Distance (mm)')
            else:
                self._contact_colorbar.update_normal(scatter)
            
            self.ax_contact.text(-9, 18.5, f'Contacts: {len(contacts)}', fontweight='bold', fontsize=11,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            self.ax_contact.text(-9, 17, f'Min Dist: {stats["min_distance_from_plane_mm"]:.4f}mm', 
                                fontweight='bold', fontsize=11,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            self.ax_contact.text(0, 10, 'NO CONTACT', ha='center', fontsize=16, color='gray', fontweight='bold')
            
        self.ax_contact.set_xlabel('X (mm)', fontsize=12)
        self.ax_contact.set_ylabel('Y (mm)', fontsize=12)
        #self.ax_contact.set_title(f'Contact Patch (Sparse MuJoCo ~{len(contacts)} nodes)', fontweight='bold', fontsize=12)
        self.ax_contact.grid(True, alpha=0.3)
        self.ax_contact.set_aspect('equal', adjustable='box')
        
        # ===== 2. High-Resolution FEM Grid Heatmap =====
        self.ax_field.clear()
        self.ax_field.set_xlim(-10, 10)
        self.ax_field.set_ylim(0, 20)
        
        if contacts and self.fem_grid is not None:
            # Interpolate sparse contacts to dense FEM grid using sensor's method
            fem_distance_field = self.digit_sensor.interpolate_to_fem_grid(
                contacts, 
                self.fem_grid, 
                influence_radius_mm=0.2
            )
            
            # Create mask: only plot nodes with non-zero values (contact regions)
            contact_mask = fem_distance_field > 1e-6  # Threshold for "active" nodes
            
            if np.any(contact_mask):
                # Only plot contact regions (makes background truly transparent)
                scatter_fem = self.ax_field.scatter(
                    self.fem_grid['x'][contact_mask], 
                    self.fem_grid['y'][contact_mask],
                    c=fem_distance_field[contact_mask],
                    cmap='plasma',
                    s=8,  # Increased from 5 for smoother appearance
                    vmin=0,
                    vmax=0.3,
                    alpha=1.0,  # Full opacity for continuous appearance
                    edgecolors='none'  # Remove edges for smoother blending
                )
                
                # Add colorbar
                if not hasattr(self, '_field_colorbar') or self._field_colorbar is None:
                    self._field_colorbar = plt.colorbar(scatter_fem, ax=self.ax_field, label='Distance (mm)')
                else:
                    self._field_colorbar.update_normal(scatter_fem)
            
            self.ax_field.text(-9, 18.5, f'FEM Nodes: {len(self.fem_grid)}', 
                              fontweight='bold', fontsize=11, color='white',
                              bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        else:
            self.ax_field.text(0, 10, 'NO CONTACT' if contacts is not None and len(contacts) == 0 else 'FEM GRID NOT LOADED', 
                              ha='center', fontsize=14, color='gray', fontweight='bold')
        
        self.ax_field.set_xlabel('X (mm)', fontsize=12)
        self.ax_field.set_ylabel('Y (mm)', fontsize=12)
        #self.ax_field.set_title('High-Res Distance Field (FEM Grid 2553 nodes)', fontweight='bold', fontsize=12)
        self.ax_field.set_aspect('equal', adjustable='box')
        
        # Refresh display
        plt.draw()
        plt.pause(0.001)
        
    def run_demo(self):
        """Run the demo with MuJoCo GUI controls"""
        
        print("🚀 Starting DIGIT GelTip Demo with MuJoCo GUI Controls")
        print("=" * 55)
        
        # Customize indenter position here
        self.indenter_x_mm = 0.0    # X position in mm
        self.indenter_y_mm = 7.5    # Y position in mm
        
        try:
            # Setup scene and sensors
            self.create_scene_with_controls()
            self.initialize_sensors()
            
            # Setup simple plots
            self.setup_simple_plots()
            
            print("✅ Demo ready!")
            print("🎛️  Use MuJoCo GUI Control panel")
            print("📊 Watch contact patch shape and force level in matplotlib")
            print("🎥 3D visualization in MuJoCo viewer")
            print(f"\n⚙️  Performance Settings:")
            print(f"   - Plot update interval: every {self.update_interval} steps")
            print(f"   - Field visualization: {'Enabled' if self.enable_field_visualization else 'Disabled'}")
            print(f"   - Field computation: {'Fast (vectorized)' if self.field_fast_mode else 'Slow (loops)'}")
            if not self.enable_field_visualization:
                print(f"   ⚠️  Field visualization disabled for max performance")
            
            # Run with GUI controls
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                print("✓ MuJoCo viewer with GUI controls launched")
                
                # Set camera
                viewer.cam.distance = 0.06
                viewer.cam.lookat = [0.0, 0.005, 0.01]
                viewer.cam.elevation = -20
                viewer.cam.azimuth = 45
                
                step = 0
                
                while viewer.is_running():
                    step += 1
                    
                    # Step simulation
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    
                    # Update plots periodically (configurable interval)
                    if step % self.update_interval == 0:
                        self.update_plots()
                    
                    time.sleep(0.01)
                    
                print("MuJoCo viewer closed")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function"""
    demo = MuJoCoControlledDIGITDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
