"""
Visualize DIGIT Sensor Data from CSV (No Simulation)
• Loads CSV session data with sensor readings
• Displays LEFT and RIGHT DIGIT sensor FEM grids in real-time
• Shows 1x2552 distance vectors for both sensors
• Time-synchronized playback for screen recording
• No MuJoCo simulation - pure data visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import os
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================
CSV_FILE_PATH = "Teleoperation_sensor_data/Trial2_peg_pos1/trial2.csv"
PLAYBACK_SPEED = 5.0  # 1.0 = real-time, 2.0 = 2x speed, 0.5 = slow motion
PLOT_UPDATE_INTERVAL = 1  # Update plots every N frames (1 = every frame)
FEM_GRID_PATH = 'filtered_FEM_grid.csv'
# ============================================================================


class SensorDataVisualizer:
    """Visualize sensor data from CSV without simulation"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.replay_data = None
        self.fem_grid = None
        self.playback_speed = PLAYBACK_SPEED
        self.update_interval = PLOT_UPDATE_INTERVAL
        
        # Load FEM grid
        self.fem_grid = self.load_fem_grid(FEM_GRID_PATH)
        if self.fem_grid is not None:
            print(f"✅ FEM grid loaded: {len(self.fem_grid)} nodes")
        else:
            print("⚠️ FEM grid not loaded - visualization will be limited")
        
        # Matplotlib components
        self.fig = None
        self.axes = {}
        self._colorbars = {}
        
        print("✅ Sensor Data Visualizer initialized")
    
    def load_fem_grid(self, csv_path):
        """Load FEM grid from CSV file"""
        if not os.path.exists(csv_path):
            print(f"⚠️ FEM grid file not found: {csv_path}")
            return None
        
        try:
            import pandas as pd
            fem_grid = pd.read_csv(csv_path)
            print(f"✅ FEM grid loaded: {len(fem_grid)} nodes")
            print(f"   X range: [{fem_grid['x'].min():.3f}, {fem_grid['x'].max():.3f}] mm")
            print(f"   Y range: [{fem_grid['y'].min():.3f}, {fem_grid['y'].max():.3f}] mm")
            return fem_grid
        except Exception as e:
            print(f"⚠️ Error loading FEM grid: {e}")
            return None
    
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
                
                # Parse sensor data (2552 nodes each)
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
    
    def update_visualization(self, frame_data, frame_index):
        """Update all sensor visualizations with frame data"""
        if frame_data is None:
            return
        
        left_sensor_data = frame_data['left_sensor']
        right_sensor_data = frame_data['right_sensor']
        timestamp = frame_data['timestamp']
        
        # Update LEFT sensor plots
        if left_sensor_data is not None and len(left_sensor_data) == 2552:
            self._update_fem_plot(
                self.axes['left_fem'],
                left_sensor_data,
                'left_fem',
                'LEFT',
                timestamp,
                frame_index
            )
            self._update_vector_plot(
                self.axes['left_vector'],
                left_sensor_data,
                'left_vector',
                'LEFT'
            )
        
        # Update RIGHT sensor plots
        if right_sensor_data is not None and len(right_sensor_data) == 2552:
            self._update_fem_plot(
                self.axes['right_fem'],
                right_sensor_data,
                'right_fem',
                'RIGHT',
                timestamp,
                frame_index
            )
            self._update_vector_plot(
                self.axes['right_vector'],
                right_sensor_data,
                'right_vector',
                'RIGHT'
            )
        
        # Refresh display
        plt.draw()
        plt.pause(0.001)
    
    def _update_fem_plot(self, ax, distance_field, ax_key, sensor_name, timestamp, frame_index):
        """Update FEM grid heatmap for a sensor"""
        ax.clear()
        self._setup_fem_axis(ax, f'{sensor_name} Sensor - FEM Grid | Frame {frame_index} | t={timestamp:.2f}s')
        
        if distance_field is not None and len(distance_field) == 2552 and self.fem_grid is not None:
            # Create mask for active nodes
            contact_mask = distance_field > 1e-6
            
            if np.any(contact_mask):
                # Get x, y from FEM grid (pandas DataFrame)
                fem_x = self.fem_grid['x'].values if hasattr(self.fem_grid, 'values') else self.fem_grid['x']
                fem_y = self.fem_grid['y'].values if hasattr(self.fem_grid, 'values') else self.fem_grid['y']
                
                scatter_fem = ax.scatter(
                    fem_x[contact_mask],
                    fem_y[contact_mask],
                    c=distance_field[contact_mask],
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
                max_dist = np.max(distance_field[contact_mask])
                ax.text(-9, 18.5, f'Active: {active_nodes}/2552',
                       fontweight='bold', fontsize=10, color='white',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                ax.text(-9, 17, f'Max: {max_dist:.4f}mm',
                       fontweight='bold', fontsize=10, color='white',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            else:
                ax.text(0, 10, 'NO CONTACT', ha='center', fontsize=14,
                       color='gray', fontweight='bold')
        else:
            # Debug info
            if distance_field is None:
                ax.text(0, 10, 'NO SENSOR DATA', ha='center', fontsize=14,
                       color='red', fontweight='bold')
            elif self.fem_grid is None:
                ax.text(0, 10, 'NO FEM GRID', ha='center', fontsize=14,
                       color='red', fontweight='bold')
            else:
                ax.text(0, 10, f'DATA SIZE MISMATCH\n{len(distance_field)} values', 
                       ha='center', fontsize=12, color='orange', fontweight='bold')
    
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
        else:
            ax.text(1276, 0.5, 'NO DATA', ha='center', va='center',
                   fontsize=12, color='gray', fontweight='bold')
    
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
    
    def run_visualization(self, csv_path):
        """Main visualization loop - no simulation, just data playback"""
        if not self.load_csv_data(csv_path):
            return
        
        self.setup_visualization()
        
        print("\n" + "="*70)
        print("SENSOR DATA VISUALIZATION (NO SIMULATION)")
        print("="*70)
        print(f"Loaded: {csv_path}")
        print(f"Frames: {len(self.replay_data)}")
        print(f"Duration: {self.replay_data[-1]['timestamp'] - self.replay_data[0]['timestamp']:.2f}s")
        print(f"Playback Speed: {self.playback_speed}x")
        print(f"Plot Update Interval: every {self.update_interval} frames")
        print("="*70)
        print("\n🎬 Visualization will start in 2 seconds...")
        print("   Close the matplotlib window to stop.")
        print("="*70 + "\n")
        
        # Wait before starting
        time.sleep(2.0)
        
        start_time = time.time()
        current_frame = 0
        last_update_time = 0
        
        print("▶️  Visualization started!\n")
        
        try:
            while plt.fignum_exists(self.fig.number):
                current_time = time.time()
                elapsed = current_time - start_time
                target_frame = self.get_frame_at_time(elapsed)
                
                if target_frame is not None:
                    # Check if we've reached the end
                    if target_frame >= len(self.replay_data) - 1:
                        if current_frame != len(self.replay_data) - 1:
                            # Final frame
                            self.update_visualization(
                                self.replay_data[-1], 
                                len(self.replay_data) - 1
                            )
                            print(f"\n✅ Visualization finished!")
                            print(f"   Final frame: {len(self.replay_data)}/{len(self.replay_data)} (100%)")
                            print(f"   Final timestamp: {self.replay_data[-1]['timestamp']:.2f}s")
                            print("\n   Keeping window open. Close it manually to exit.")
                            current_frame = len(self.replay_data) - 1
                        
                        # Keep window open
                        plt.pause(0.1)
                        continue
                    
                    # Update if frame changed and update interval met
                    if target_frame != current_frame and (target_frame % self.update_interval == 0):
                        self.update_visualization(
                            self.replay_data[target_frame],
                            target_frame
                        )
                        current_frame = target_frame
                        
                        # Progress update
                        if current_time - last_update_time > 1.0:
                            progress = (current_frame / len(self.replay_data)) * 100
                            print(f"⏱️  Frame {current_frame}/{len(self.replay_data)} ({progress:.1f}%) - "
                                  f"Timestamp: {self.replay_data[current_frame]['timestamp']:.2f}s - "
                                  f"Speed: {self.playback_speed}x")
                            last_update_time = current_time
                else:
                    print("\n⚠️ Error: Could not determine target frame")
                    break
                
                plt.pause(0.001)
        
        except KeyboardInterrupt:
            print("\n⚠️ Visualization interrupted by user")
        
        print("\n✅ Visualization stopped")
        print("📊 Close the matplotlib window to exit completely.")
        plt.ioff()
        plt.show()


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("SENSOR DATA VISUALIZATION (NO SIMULATION)")
    print("="*70)
    print(f"\n📂 CSV File: {CSV_FILE_PATH}")
    print(f"⏱️  Playback Speed: {PLAYBACK_SPEED}x")
    print(f"📊 Plot Update Interval: every {PLOT_UPDATE_INTERVAL} frames")
    
    if not os.path.exists(CSV_FILE_PATH):
        print(f"\n❌ File not found: {CSV_FILE_PATH}")
        print("\n💡 Tip: Edit the CSV_FILE_PATH variable at the top of this script")
        return
    
    visualizer = SensorDataVisualizer()
    visualizer.run_visualization(CSV_FILE_PATH)


if __name__ == "__main__":
    main()
