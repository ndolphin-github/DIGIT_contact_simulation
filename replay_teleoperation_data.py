
import numpy as np
import mujoco
import mujoco.viewer
import time
import csv
import os
from datetime import datetime

from simple_ik_legacy import DEFAULT_INITIAL_JOINTS
from gripper_digit_sensor import GripperDIGITSensor


# ============================================================================
# CONFIGURATION: Set your CSV file path here
# ============================================================================
CSV_FILE_PATH = "Teleoperation_sensor_data/Trial2_peg_pos2/trial2.csv"
PLAYBACK_SPEED = 2.0  # 1.0 = real-time, 2.0 = 2x speed, 0.5 = slow motion
Initial_peg_position=[0.6, 0.2, 0.81] 
# ============================================================================


class TeleoperationReplay:
    """Replay recorded teleoperation data in MuJoCo simulation"""
    
    def __init__(self, xml_path="ur5e_with_DIGIT_primitive_hexagon.xml"):
        """Initialize replay system"""
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
        
        # Initialize DIGIT sensors (optional - for visual comparison)
        try:
            self.digit_left = GripperDIGITSensor(self.model, "digit_geltip_left", "left")
            self.digit_right = GripperDIGITSensor(self.model, "digit_geltip_right", "right")
            self.has_digit_sensors = True
            print("✅ DIGIT sensors initialized")
        except Exception as e:
            self.has_digit_sensors = False
            print(f"⚠️ DIGIT sensors not initialized: {e}")
        
        # Replay data
        self.replay_data = None
        self.current_frame = 0
        self.start_time = 0.0
        self.playback_speed = PLAYBACK_SPEED  # Use global configuration
        self.is_paused = False
        
        print("✅ Teleoperation Replay System initialized")
    
    def load_csv_data(self, csv_path):
        """
        Load recorded session data from CSV file.
        Returns: List of dicts with keys: timestamp, gripper_value, joint_angles
        """
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            return False
        
        print(f"\n📂 Loading CSV data: {csv_path}")
        
        data_rows = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            for row in reader:
                # Parse timestamp, gripper, and 6 joint angles (first 8 columns)
                timestamp = float(row[0])
                gripper_value = float(row[1])
                joint_angles = [float(row[i]) for i in range(2, 8)]  # Columns 2-7
                
                # Optionally parse sensor data (columns 8-2559 for left, 2560-5111 for right)
                # For replay, we only need joint angles and gripper, but we can store them for display
                left_sensor_data = None
                right_sensor_data = None
                if len(row) >= 5112:  # Full data with sensors
                    left_sensor_data = [float(row[i]) for i in range(8, 2560)]
                    right_sensor_data = [float(row[i]) for i in range(2560, 5112)]
                
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
    
    def set_initial_pose(self):
        """Set robot to initial joint configuration (from recording start)"""
        # Set peg to safe position
        if self.has_peg:
            peg_qpos_addr = self.model.body_jntadr[self.peg_body_id]
            self.data.qpos[peg_qpos_addr:peg_qpos_addr+3] = Initial_peg_position
            self.data.qpos[peg_qpos_addr+3:peg_qpos_addr+7] = [1, 0, 0, 0]
            peg_qvel_addr = self.model.body_dofadr[self.peg_body_id]
            self.data.qvel[peg_qvel_addr:peg_qvel_addr+6] = 0.0
        
        # Set robot to first frame's joint angles (if data is loaded)
        if self.replay_data is not None and len(self.replay_data) > 0:
            first_frame = self.replay_data[0]
            for i, jid in enumerate(self.joint_ids):
                self.data.qpos[jid] = first_frame['joint_angles'][i]
                self.data.ctrl[self.actuator_ids[i]] = first_frame['joint_angles'][i]
                self.data.qvel[jid] = 0.0
            
            # Set gripper
            if self.has_gripper:
                for grip_id in self.gripper_actuator_ids:
                    self.data.ctrl[grip_id] = first_frame['gripper_value']
        else:
            # Fall back to default if no data loaded
            for i, jid in enumerate(self.joint_ids):
                self.data.qpos[jid] = DEFAULT_INITIAL_JOINTS[i]
                self.data.ctrl[self.actuator_ids[i]] = DEFAULT_INITIAL_JOINTS[i]
                self.data.qvel[jid] = 0.0
            
            if self.has_gripper:
                for grip_id in self.gripper_actuator_ids:
                    self.data.ctrl[grip_id] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
    
    def get_frame_at_time(self, elapsed_time):
        """
        Get the appropriate frame index for the given elapsed time.
        Uses linear interpolation if needed.
        """
        if self.replay_data is None or len(self.replay_data) == 0:
            return None
        
        # Adjust for playback speed
        adjusted_time = elapsed_time * self.playback_speed
        
        # Calculate total duration
        first_timestamp = self.replay_data[0]['timestamp']
        last_timestamp = self.replay_data[-1]['timestamp']
        total_duration = last_timestamp - first_timestamp
        
        # Offset by first frame's timestamp
        target_timestamp = first_timestamp + adjusted_time
        
        # If we've exceeded total duration, return last frame
        if adjusted_time >= total_duration:
            return len(self.replay_data) - 1
        
        # Find closest frame
        for i, frame in enumerate(self.replay_data):
            if frame['timestamp'] >= target_timestamp:
                return i
        
        # Fallback: return last frame
        return len(self.replay_data) - 1
    
    def apply_frame(self, frame_index):
        """Apply joint angles and gripper value from a specific frame"""
        if self.replay_data is None or frame_index >= len(self.replay_data):
            return
        
        frame = self.replay_data[frame_index]
        
        # Set joint angles directly (position control)
        for i, jid in enumerate(self.joint_ids):
            target_angle = frame['joint_angles'][i]
            self.data.ctrl[self.actuator_ids[i]] = target_angle
        
        # Set gripper
        if self.has_gripper:
            for grip_id in self.gripper_actuator_ids:
                self.data.ctrl[grip_id] = frame['gripper_value']
        
        # Step physics to apply controls
        for _ in range(5):  # Multiple steps for smooth actuation
            mujoco.mj_step(self.model, self.data)
        
        mujoco.mj_forward(self.model, self.data)
        
        self.current_frame = frame_index
    
    def print_status(self):
        """Print current replay status"""
        if self.replay_data is None:
            print("⚠️ No data loaded")
            return
        
        frame = self.replay_data[self.current_frame]
        progress = (self.current_frame / len(self.replay_data)) * 100
        
        print(f"\n📊 Replay Status:")
        print(f"   Frame: {self.current_frame}/{len(self.replay_data)} ({progress:.1f}%)")
        print(f"   Timestamp: {frame['timestamp']:.3f}s")
        print(f"   Gripper: {frame['gripper_value']:.3f}")
        print(f"   Joint angles (deg): [{', '.join([f'{np.rad2deg(a):.1f}' for a in frame['joint_angles']])}]")
        print(f"   Playback speed: {self.playback_speed}x")
        print(f"   Paused: {self.is_paused}")
        
        # Peg status
        if self.has_peg:
            ee_pos = self.data.site_xpos[self.ee_site_id]
            peg_pos = self.data.xpos[self.peg_body_id]
            peg_ee_dist = np.linalg.norm(peg_pos - ee_pos)
            grasp_status = "HELD" if peg_ee_dist < 0.050 else "DROPPED"
            print(f"   Peg-EE Distance: {peg_ee_dist*1000:.1f}mm [{grasp_status}]")
    
    def run_replay(self, csv_path):
        """Main replay loop"""
        # Load data
        if not self.load_csv_data(csv_path):
            return
        
        # Set initial pose
        self.set_initial_pose()
        
        # Stabilize physics
        print("Stabilizing physics simulation...")
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
        print("✓ Physics stabilized")
        
        print("\n" + "="*70)
        print("TELEOPERATION REPLAY")
        print("="*70)
        print(f"Loaded: {csv_path}")
        print(f"Frames: {len(self.replay_data)}")
        print(f"Duration: {self.replay_data[-1]['timestamp'] - self.replay_data[0]['timestamp']:.2f}s")
        print("\nControls:")
        print("  SPACE  - Pause/Resume")
        print("  R      - Restart from beginning")
        print("  +/-    - Increase/Decrease playback speed")
        print("  P      - Print current status")
        print("  X      - Exit")
        print("="*70 + "\n")
        
        self.current_frame = 0
        self.is_paused = False
        
        # Launch viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("✅ Viewer started. Replaying data...\n")
            
            self.start_time = time.time()
            last_update_time = self.start_time
            
            while viewer.is_running():
                current_time = time.time()
                
                # Check for keyboard input (non-blocking is not available in standard mujoco.viewer)
                # For now, we'll just auto-play
                # In a full implementation, you'd need a separate thread for keyboard input
                
                if not self.is_paused:
                    # Calculate elapsed time
                    elapsed = current_time - self.start_time
                    
                    # Get target frame
                    target_frame = self.get_frame_at_time(elapsed)
                    
                    if target_frame is not None:
                        # ALWAYS apply frame, even if same
                        self.apply_frame(target_frame)
                        
                        # Print progress every 1 second
                        if current_time - last_update_time > 1.0:
                            progress = (self.current_frame / len(self.replay_data)) * 100
                            print(f"⏱️  Frame {self.current_frame}/{len(self.replay_data)} ({progress:.1f}%) - " 
                                  f"Timestamp: {self.replay_data[self.current_frame]['timestamp']:.2f}s - "
                                  f"Speed: {self.playback_speed}x")
                            last_update_time = current_time
                        
                        # Check if we've reached the last frame
                        if self.current_frame >= len(self.replay_data) - 1:
                            # Print completion message once
                            if not hasattr(self, '_completion_printed'):
                                print(f"\n✅ Replay finished!")
                                print(f"   Final frame: {len(self.replay_data)}/{len(self.replay_data)} (100.0%)")
                                print(f"   Final timestamp: {self.replay_data[-1]['timestamp']:.2f}s")
                                print("   Close viewer to exit")
                                self._completion_printed = True
                            self.is_paused = True
                    else:
                        # Error condition
                        print("\n⚠️ Error: Could not determine target frame")
                        self.is_paused = True
                else:
                    # When paused, still step physics to maintain visualization
                    mujoco.mj_step(self.model, self.data)
                
                viewer.sync()
                time.sleep(0.01)  # ~100 FPS
        
        print("\n✅ Replay stopped")


def list_available_sessions(data_dir="Teleoperation_sensor_data"):
    """List all available session CSV files"""
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return []
    
    sessions = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                sessions.append(full_path)
    
    return sessions


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("TELEOPERATION REPLAY SYSTEM")
    print("="*70)
    print(f"\n📂 CSV File: {CSV_FILE_PATH}")
    print(f"⏱️  Playback Speed: {PLAYBACK_SPEED}x")
    
    # Check if file exists
    if not os.path.exists(CSV_FILE_PATH):
        print(f"\n❌ File not found: {CSV_FILE_PATH}")
        print("\n💡 Tip: Edit the CSV_FILE_PATH variable at the top of this script")
        return
    
    # Create replay system and run
    replay = TeleoperationReplay()
    replay.run_replay(CSV_FILE_PATH)


if __name__ == "__main__":
    main()
