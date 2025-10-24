'''
UR5e MuJoCo Parser - Simplified and focused version for UR5e robot simulation
'''

import os
import mujoco.viewer
import numpy as np
import mujoco

from src.util import r2w, rpy2r, trim_scale

class UR5eSimulator(object):
    """
        UR5e MuJoCo Simulator class
    """
    def __init__(self, name='UR5e', xml_path=None, USE_VIEWER=True, VERBOSE=True):
        """
            Initialize UR5e simulator
        """
        self.name = name
        self.xml_path = xml_path
        self.VERBOSE = VERBOSE
        self.USE_VIEWER = USE_VIEWER
        
        # Time tracking
        self.tick = 0
        self.render_tick = 0
        
        # Load the model
        if self.xml_path is not None:
            self._load_model()
        
        # Initialize viewer if requested
        if self.USE_VIEWER:
            self.init_viewer()
        
        # Reset to initial state
        self.reset()
        
        # Print info
        if self.VERBOSE:
            self.print_info()
    
    def _load_model(self):
        """
            Load MuJoCo model from XML
        """
        self.full_xml_path = os.path.abspath(self.xml_path)
        self.model = mujoco.MjModel.from_xml_path(self.full_xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Get basic model info
        self.n_dof = self.model.nv  # degrees of freedom
        self.n_joints = self.model.njnt  # number of joints
        self.n_actuators = self.model.nu  # number of actuators
        
        # Joint information
        self.joint_names = []
        for i in range(self.n_joints):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                self.joint_names.append(joint_name)
        
        self.joint_types = self.model.jnt_type
        self.joint_ranges = self.model.jnt_range
        
        # Revolute joints (typical for UR5e)
        self.rev_joint_idxs = np.where(self.joint_types == mujoco.mjtJoint.mjJNT_HINGE)[0]
        self.rev_joint_names = [self.joint_names[i] for i in self.rev_joint_idxs if i < len(self.joint_names)]
        self.n_rev_joints = len(self.rev_joint_idxs)
        
        # UR5e specific joint names (expected)
        self.ur5e_joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Actuator information
        self.actuator_names = []
        for i in range(self.n_actuators):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name:
                self.actuator_names.append(actuator_name)
        
        self.actuator_ranges = self.model.actuator_ctrlrange
        
        # Body information
        self.body_names = []
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name:
                self.body_names.append(body_name)
        
        # Important UR5e bodies
        self.ur5e_body_names = [
            'base_link',
            'shoulder_link',
            'upper_arm_link',
            'forearm_link', 
            'wrist_1_link',
            'wrist_2_link',
            'wrist_3_link'
        ]
        
        # End effector
        self.end_effector_name = 'wrist_3_link'  # Can be changed based on your model
    
    def print_info(self):
        """
            Print model information
        """
        print(f"=== {self.name} Model Information ===")
        print(f"XML Path: {self.xml_path}")
        print(f"Degrees of Freedom: {self.n_dof}")
        print(f"Number of Joints: {self.n_joints}")
        print(f"Number of Actuators: {self.n_actuators}")
        print(f"Joint Names: {self.joint_names}")
        print(f"Revolute Joints: {self.rev_joint_names}")
        print(f"Actuator Names: {self.actuator_names}")
        print(f"Body Names: {self.body_names}")
        print("=" * 50)
    
    def init_viewer(self, title='UR5e Simulator', width=1200, height=800):
        """
            Initialize MuJoCo viewer
        """
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 2.0
        self.viewer.cam.azimuth = 45
        self.viewer.cam.elevation = -30
        self.viewer.cam.lookat = [0, 0, 0.5]
    
    def is_viewer_alive(self):
        """
            Check if viewer is alive
        """
        if not hasattr(self, 'viewer') or self.viewer is None:
            return False
        
        # Try different methods to check if viewer is alive
        try:
            # Method 1: Direct is_alive attribute/method
            if hasattr(self.viewer, 'is_alive'):
                if callable(self.viewer.is_alive):
                    return self.viewer.is_alive()
                else:
                    return self.viewer.is_alive
            
            # Method 2: Check if viewer handle is valid
            if hasattr(self.viewer, '_handle'):
                return self.viewer._handle is not None
            
            # Method 3: Assume viewer is alive if we have a viewer object
            return True
            
        except Exception:
            return False
    
    def reset(self):
        """
            Reset simulation to initial state
        """
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.tick = 0
        self.render_tick = 0
    
    def step(self, ctrl=None, nstep=1):
        """
            Step simulation forward
        """
        if ctrl is not None:
            self.data.ctrl[:] = ctrl
        
        for _ in range(nstep):
            mujoco.mj_step(self.model, self.data)
        
        self.tick += nstep
    
    def forward(self, joint_positions=None):
        """
            Forward kinematics - set joint positions and compute forward kinematics
        """
        if joint_positions is not None:
            self.data.qpos[:len(joint_positions)] = joint_positions
        
        mujoco.mj_forward(self.model, self.data)
        self.tick += 1
    
    def render(self, render_every=1):
        """
            Render simulation
        """
        if self.USE_VIEWER and hasattr(self, 'viewer') and self.viewer is not None:
            try:
                if (self.render_tick % render_every == 0) or (self.render_tick == 0):
                    self.viewer.sync()
                self.render_tick += 1
            except Exception as e:
                print(f"Render error: {e}")
                self.USE_VIEWER = False
    
    def get_joint_positions(self):
        """
            Get current joint positions
        """
        return self.data.qpos[:self.n_rev_joints].copy()
    
    def get_joint_velocities(self):
        """
            Get current joint velocities
        """
        return self.data.qvel[:self.n_rev_joints].copy()
    
    def get_end_effector_pose(self):
        """
            Get end effector position and orientation
        """
        if self.end_effector_name in self.body_names:
            body_id = self.model.body(self.end_effector_name).id
            position = self.data.xpos[body_id].copy()
            rotation_mat = self.data.xmat[body_id].reshape(3, 3).copy()
            return position, rotation_mat
        else:
            print(f"Warning: End effector '{self.end_effector_name}' not found")
            return None, None
    
    def get_jacobian(self, body_name=None):
        """
            Get Jacobian matrix for a body (default: end effector)
        """
        if body_name is None:
            body_name = self.end_effector_name
        
        if body_name in self.body_names:
            body_id = self.model.body(body_name).id
            J_pos = np.zeros((3, self.model.nv))
            J_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, J_pos, J_rot, body_id)
            
            # Return only the relevant columns for actuated joints
            J_pos = J_pos[:, :self.n_rev_joints]
            J_rot = J_rot[:, :self.n_rev_joints]
            J_full = np.vstack([J_pos, J_rot])
            
            return J_pos, J_rot, J_full
        else:
            print(f"Warning: Body '{body_name}' not found")
            return None, None, None
    
    def inverse_kinematics(self, target_pos, target_rot=None, body_name=None, 
                          max_iter=100, tol=1e-3, step_size=0.1):
        """
            Simple inverse kinematics solver
        """
        if body_name is None:
            body_name = self.end_effector_name
        
        current_q = self.get_joint_positions()
        
        for i in range(max_iter):
            # Get current pose
            current_pos, current_rot = self.get_end_effector_pose()
            if current_pos is None:
                return None
            
            # Calculate position error
            pos_error = target_pos - current_pos
            
            # Calculate rotation error (if target rotation is provided)
            if target_rot is not None:
                rot_error_mat = target_rot @ current_rot.T
                rot_error = r2w(rot_error_mat)
                error = np.concatenate([pos_error, rot_error])
            else:
                error = pos_error
            
            # Check convergence
            if np.linalg.norm(error) < tol:
                print(f"IK converged in {i} iterations")
                return current_q
            
            # Get Jacobian
            J_pos, J_rot, J_full = self.get_jacobian(body_name)
            if J_pos is None:
                return None
            
            # Use appropriate Jacobian
            if target_rot is not None:
                J = J_full
            else:
                J = J_pos
            
            # Calculate joint velocity using damped least squares
            lambda_reg = 1e-4
            dq = step_size * np.linalg.solve(J.T @ J + lambda_reg * np.eye(J.shape[1]), J.T @ error)
            
            # Update joint positions
            current_q += dq
            
            # Apply joint limits (simple clipping)
            for j in range(len(current_q)):
                if j < len(self.joint_ranges):
                    current_q[j] = np.clip(current_q[j], 
                                         self.joint_ranges[j, 0], 
                                         self.joint_ranges[j, 1])
            
            # Update simulation
            self.forward(current_q)
        
        print(f"IK did not converge in {max_iter} iterations")
        return current_q
    
    def set_joint_positions(self, joint_positions):
        """
            Set joint positions directly (bypasses physics - use for initialization only)
        """
        self.forward(joint_positions)
    
    def set_joint_targets(self, joint_targets):
        """
            Set joint target positions using actuators (respects physics and collisions)
        """
        # Ensure we have the right number of control inputs
        ctrl = np.zeros(self.n_actuators)
        n_joints = min(len(joint_targets), self.n_actuators)
        ctrl[:n_joints] = joint_targets[:n_joints]
        
        # Apply control and step simulation
        self.step(ctrl=ctrl, nstep=1)
    
    def get_simulation_time(self):
        """
            Get current simulation time
        """
        return self.data.time
    
    def close_viewer(self):
        """
            Close the viewer
        """
        if hasattr(self, 'viewer'):
            self.viewer.close()
            self.USE_VIEWER = False
    
    def add_visual_marker(self, position, size=0.02, color=[1, 0, 0, 1]):
        """
            Add visual marker to the scene (if supported by viewer)
        """
        try:
            if hasattr(self, 'viewer') and self.viewer is not None:
                if hasattr(self.viewer, 'add_marker'):
                    self.viewer.add_marker(
                        pos=position,
                        size=[size, size, size],
                        rgba=color,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE
                    )
                    return True
                else:
                    print(f"Visual marker requested at {position} but viewer doesn't support markers")
                    return False
        except Exception as e:
            print(f"Failed to add visual marker: {e}")
            return False
        return False
