import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import threading
import time
from modular_digit_sensor import ModularDIGITSensor, UniversalMeshExtractor

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
        
        # Data for plotting
        self.contact_history = []
        self.force_history = []
        self.time_history = []
        self.start_time = time.time()
        
        # Matplotlib components
        self.fig = None
        self.ax_contact = None
        self.ax_force = None
        
        # Update control
        self.plot_update_counter = 0
        
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
        <mesh name="indenter" file="indenters/Ball.STL" scale="0.001 0.001 0.001"/>
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
        """Initialize DIGIT sensor and indenter extractor"""
        
        self.digit_sensor = ModularDIGITSensor(
            model=self.model,
            sensor_body_name="digit_sensor"
        )
        
        self.indenter_extractor = UniversalMeshExtractor(
            model=self.model,
            object_geom_name="indenter_geom"
        )
        
        print("✓ Sensors initialized")
        
    def setup_simple_plots(self):
        """Setup simple matplotlib window with only contact patch and force"""
        
        self.fig, (self.ax_contact, self.ax_force) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Contact patch visualization
        self.ax_contact.set_xlim(-10, 10)
        self.ax_contact.set_ylim(0, 20)
        self.ax_contact.set_xlabel('X (mm)')
        self.ax_contact.set_ylabel('Y (mm)')
        self.ax_contact.set_title('Contact Patch Shape')
        self.ax_contact.grid(True, alpha=0.3)
        
        # Add sensor ROI boundary
        roi_rect = plt.Rectangle((-7.5, 0), 15, 15, 
                               linewidth=2, edgecolor='green', 
                               facecolor='none', linestyle='--')
        self.ax_contact.add_patch(roi_rect)
        self.ax_contact.text(0, 17, 'DIGIT ROI (15×15mm)', ha='center', fontweight='bold')
        
        # Force level plot
        self.ax_force.set_xlabel('Time (s)')
        self.ax_force.set_ylabel('Contact Force (N)')
        self.ax_force.set_title('Force Level')
        self.ax_force.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.ion()  # Interactive mode
        plt.show(block=False)
        
        print("✓ Simple plots setup: Contact patch shape + Force level")
        
    def update_plots(self):
        """Update contact patch and force plots"""
        
        # Get contact data
        world_vertices = self.indenter_extractor.get_world_vertices(self.data)
        contacts = self.digit_sensor.detect_proximity_contacts(world_vertices, self.data)
        
        # Calculate force
        total_force = len(contacts) * 0.01 if contacts else 0.0
        
        # Update history
        current_time = time.time() - self.start_time
        self.contact_history.append(len(contacts))
        self.force_history.append(total_force)
        self.time_history.append(current_time)
        
        # Keep reasonable history
        if len(self.time_history) > 100:
            self.contact_history = self.contact_history[-100:]
            self.force_history = self.force_history[-100:]
            self.time_history = self.time_history[-100:]
        
        # Update contact patch plot
        self.ax_contact.clear()
        self.ax_contact.set_xlim(-10, 10)
        self.ax_contact.set_ylim(0, 20)
        
        # Redraw ROI boundary
        roi_rect = plt.Rectangle((-7.5, 0), 15, 15, 
                               linewidth=2, edgecolor='green', 
                               facecolor='none', linestyle='--')
        self.ax_contact.add_patch(roi_rect)
        
        if contacts:
            contact_x = [c['x_mm'] for c in contacts]
            contact_y = [c['y_mm'] for c in contacts]
            proximities = [c['proximity'] for c in contacts]
            
            # Color by proximity
            colors = plt.cm.Reds([1.0 - p/0.5 for p in proximities])
            self.ax_contact.scatter(contact_x, contact_y, c=colors, s=50, alpha=0.8, edgecolors='black')
            
            self.ax_contact.text(-9, 18, f'Contacts: {len(contacts)}', fontweight='bold', fontsize=12)
            self.ax_contact.text(-9, 16, f'Force: {total_force:.3f} N', fontweight='bold', fontsize=12)
        else:
            self.ax_contact.text(0, 10, 'NO CONTACT', ha='center', fontsize=16, color='gray', fontweight='bold')
            
        self.ax_contact.set_xlabel('X (mm)')
        self.ax_contact.set_ylabel('Y (mm)')
        self.ax_contact.set_title('Contact Patch Shape')
        self.ax_contact.grid(True, alpha=0.3)
        
        # Update force plot
        if len(self.time_history) > 1:
            self.ax_force.clear()
            self.ax_force.plot(self.time_history, self.force_history, 'b-', linewidth=3)
            self.ax_force.fill_between(self.time_history, self.force_history, alpha=0.3, color='blue')
            self.ax_force.set_xlabel('Time (s)')
            self.ax_force.set_ylabel('Force (N)')
            self.ax_force.set_title('Force Level')
            self.ax_force.grid(True, alpha=0.3)
        
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
                    
                    # Update plots every 10 steps
                    if step % 10 == 0:
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
