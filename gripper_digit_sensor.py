"""
Gripper-mounted DIGIT sensor module for UR5e demo
Designed specifically for the RH-P12-RN gripper finger geometry

This module uses the same principles as modular_digit_sensor.py but with
fixed geometry for the gripper-mounted DIGIT sensors.
"""

import mujoco
import numpy as np


class GripperDIGITSensor:
    """
    DIGIT sensor for gripper fingers with fixed geometry.
    Uses gel tip vertices for contact detection (same principle as ModularDIGITSensor).
    
    Geometry (from XML visualization):
    - Sensor origin: At top of gel tip (red dot)
    - Sensing plane: 30mm down (local Z) and 4mm out (local Y)
    - ROI: 15mm × 15mm centered at sensing plane
    - Proximity threshold: 0.8mm
    
    Coordinate system (measured from debug):
    - Local X → World -X (horizontal)
    - Local Y → World -Z (vertical, pointing down)
    - Local Z → World -Y (horizontal, toward opposite finger)
    
    Variable meanings (changed from original):
    - 'distance_from_plane_mm': Distance from gel node to sensing plane
    - 'intensity': Proximity weight (1.0 = touching plane, 0.0 = at threshold)
    """
    
    def __init__(self, model, sensor_body_name, sensor_type="left"):
        """
        Initialize gripper-mounted DIGIT sensor
        
        Args:
            model: MuJoCo model
            sensor_body_name: Name of the sensor body (e.g., "digit_geltip_left")
            sensor_type: "left" or "right" to determine sensing direction
        """
        self.model = model
        self.sensor_type = sensor_type
        
        # Get sensor body ID
        try:
            self.sensor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sensor_body_name)
        except:
            raise ValueError(f"Sensor body '{sensor_body_name}' not found in model")
        
        # Fixed geometry parameters (in sensor local coordinates)
        # From XML: pos="0 ±0.004 0.03"
        self.sensing_distance = 0.004  # 4mm from gel surface to sensing plane
        self.sensing_plane_offset_z = 0.030  # 30mm down from origin
        
        # Sensing direction (outward from finger)
        self.sensing_direction = 1.0 if sensor_type == "left" else -1.0
        
        # ROI dimensions
        self.roi_half_width = 0.0075   # ±7.5mm in X
        self.roi_half_height = 0.0075  # ±7.5mm in Z
        
        # Proximity threshold (distance from sensing plane to consider contact)
        self.proximity_threshold = 0.0008  # 0.8mm (same as ModularDIGITSensor)
        
        # Sensing plane position in sensor local coordinates: [X, Y, Z]
        self.sensing_plane_pos = np.array([
            0.0,  # X = 0 (centered)
            self.sensing_distance * self.sensing_direction,  # Y = ±4mm
            self.sensing_plane_offset_z  # Z = 30mm
        ])
        
        # Extract gel tip mesh vertices for contact detection
        # Need to use the geometry name, not body name
        gel_geom_name = f"{sensor_body_name}_geom"
        self.gel_tip_extractor = UniversalMeshExtractor(model, gel_geom_name)
        
        print(f"✓ Gripper DIGIT sensor initialized: {sensor_body_name}")
        print(f"  Type: {sensor_type.upper()}")
        print(f"  Sensing distance: {self.sensing_distance*1000:.1f}mm")
        print(f"  Proximity threshold: {self.proximity_threshold*1000:.1f}mm")
    
    def get_sensor_pose(self, data):
        """Get current sensor pose in world coordinates"""
        sensor_pos = data.xpos[self.sensor_body_id].copy()
        sensor_rot = data.xmat[self.sensor_body_id].reshape(3, 3).copy()
        return sensor_pos, sensor_rot
    
    def get_sensing_plane_world(self, data):
        """Get sensing plane center position in world coordinates"""
        sensor_pos, sensor_rot = self.get_sensor_pose(data)
        
        # Sensing plane local position: [0, ±4mm, 30mm]
        sensing_plane_local = np.array([
            0.0,
            self.sensing_distance * self.sensing_direction,
            self.sensing_plane_offset_z
        ])
        
        # Transform sensing plane position to world coordinates
        sensing_plane_world = sensor_rot @ sensing_plane_local + sensor_pos
        
        return sensing_plane_world
    
    def detect_proximity_contacts(self, data):
        """
        Detect proximity-based contacts for gripper-mounted DIGIT sensor.
        
        Returns list of contact dictionaries with:
            - position_sensor_local: 3D position in sensor frame
            - x_mm, y_mm: 2D ROI coordinates
            - distance_from_plane_mm: Distance from sensing plane
            - intensity: Contact intensity (1.0 = touching, 0.0 = far)
        """
        # Check for collisions with this sensor's gel tip
        colliding_geoms = set()
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]
            
            # Check if either geom belongs to this sensor body
            if body1 == self.sensor_body_id:
                colliding_geoms.add(geom2)  # Store the OTHER geom
            elif body2 == self.sensor_body_id:
                colliding_geoms.add(geom1)  # Store the OTHER geom
        
        if not colliding_geoms:
            return []
        
        # Get sensor pose
        sensor_pos = data.xpos[self.sensor_body_id]
        sensor_rot = data.xmat[self.sensor_body_id].reshape(3, 3)
        
        contacts = []
        
        # For each colliding geometry, extract its vertices
        for geom_id in colliding_geoms:
            # Get the mesh/primitive associated with this geom
            geom_type = self.model.geom_type[geom_id]
            
            # Extract vertices based on geometry type
            object_vertices_world = []
            
            if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                # Get mesh data
                dataid = self.model.geom_dataid[geom_id]
                if dataid >= 0:
                    mesh_id = dataid
                    vert_start = self.model.mesh_vertadr[mesh_id]
                    vert_count = self.model.mesh_vertnum[mesh_id]
                    
                    # Get body pose for this geom
                    body_id = self.model.geom_bodyid[geom_id]
                    body_pos = data.xpos[body_id]
                    body_rot = data.xmat[body_id].reshape(3, 3)
                    
                    # Get geom local pose
                    geom_pos = self.model.geom_pos[geom_id]
                    geom_quat = self.model.geom_quat[geom_id]
                    geom_rot = np.zeros(9)
                    mujoco.mju_quat2Mat(geom_rot, geom_quat)
                    geom_rot = geom_rot.reshape(3, 3)
                    
                    # Transform mesh vertices to world frame
                    for v_idx in range(vert_start, vert_start + vert_count):
                        vert_local = self.model.mesh_vert[v_idx]
                        # Local geom frame -> body frame -> world frame
                        vert_geom = geom_rot @ vert_local + geom_pos
                        vert_world = body_rot @ vert_geom + body_pos
                        object_vertices_world.append(vert_world)
            
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                # Generate box vertices
                size = self.model.geom_size[geom_id]
                body_id = self.model.geom_bodyid[geom_id]
                body_pos = data.xpos[body_id]
                body_rot = data.xmat[body_id].reshape(3, 3)
                
                geom_pos = self.model.geom_pos[geom_id]
                geom_quat = self.model.geom_quat[geom_id]
                geom_rot = np.zeros(9)
                mujoco.mju_quat2Mat(geom_rot, geom_quat)
                geom_rot = geom_rot.reshape(3, 3)
                
                # 8 corners of box
                for dx in [-size[0], size[0]]:
                    for dy in [-size[1], size[1]]:
                        for dz in [-size[2], size[2]]:
                            corner_local = np.array([dx, dy, dz])
                            corner_geom = geom_rot @ corner_local + geom_pos
                            corner_world = body_rot @ corner_geom + body_pos
                            object_vertices_world.append(corner_world)
            
            # Now check which object vertices are near the sensing plane
            for vert_world in object_vertices_world:
                # Transform to sensor local coordinates
                relative_pos = vert_world - sensor_pos
                vert_local = sensor_rot.T @ relative_pos
                
                # Calculate distance from sensing plane
                # Sensing plane is at self.sensing_plane_pos in local coords
                distance_vector = vert_local - self.sensing_plane_pos
                
                # Distance along surface normal (Y-axis for gripper sensors)
                distance_along_normal = np.abs(distance_vector[1])  # Y component
                
                # Check if within proximity threshold (0.8mm)
                if distance_along_normal <= self.proximity_threshold:
                    # Check if within ROI bounds (X and Z relative to sensing plane)
                    # X: centered at 0, range [-7.5mm, +7.5mm]
                    # Z: symmetric around sensing plane, range [-7.5mm, +7.5mm]
                    x_relative = vert_local[0] - self.sensing_plane_pos[0]  # Relative to center
                    z_relative = vert_local[2] - self.sensing_plane_pos[2]  # Relative to sensing plane Z
                    
                    # ROI bounds: X in [-7.5, +7.5], Z in [-7.5, +7.5] (symmetric)
                    if (abs(x_relative) <= self.roi_half_width and 
                        abs(z_relative) <= self.roi_half_height):  # Symmetric around sensing plane
                        
                        # Calculate intensity (1.0 = touching, 0.0 = at threshold)
                        intensity = 1.0 - (distance_along_normal / self.proximity_threshold)
                        
                        # Convert to 2D ROI coordinates in mm
                        x_mm = x_relative * 1000  # [-7.5, +7.5] mm
                        # Shift Z to [0, 15] range for visualization: z_relative [-7.5, +7.5] -> [0, 15]
                        z_mm = (z_relative + self.roi_half_height) * 1000  # [0, 15] mm for visualization as Y
                        
                        contacts.append({
                            'position_sensor_local': vert_local,
                            'x_mm': x_mm,
                            'y_mm': z_mm,  # Map Z to Y for 2D visualization
                            'distance_from_plane_mm': distance_along_normal * 1000,
                            'intensity': intensity
                        })
        
        return contacts
    
    def get_geltip_distance_field(self, data):
        """
        Generate distance field based on gel tip ROI mesh nodes (fixed resolution).
        Each gel tip node measures distance to nearest object surface.
        
        Returns:
            list of dicts with:
                - position_sensor_local: gel tip node position in sensor frame
                - x_mm, y_mm: 2D ROI coordinates
                - distance_to_object_mm: distance to nearest object point
                - intensity: contact intensity (1.0 = touching, 0.0 = far)
        """
        # Get gel tip vertices in world frame
        gel_vertices_world = self.gel_tip_extractor.get_world_vertices(data)
        
        if len(gel_vertices_world) == 0:
            return []
        
        # Get sensor pose
        sensor_pos = data.xpos[self.sensor_body_id]
        sensor_rot = data.xmat[self.sensor_body_id].reshape(3, 3)
        
        # Check for colliding objects
        colliding_geoms = set()
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]
            
            if body1 == self.sensor_body_id:
                colliding_geoms.add(geom2)
            elif body2 == self.sensor_body_id:
                colliding_geoms.add(geom1)
        
        if not colliding_geoms:
            return []
        
        # Extract all object vertices
        object_vertices_world = []
        for geom_id in colliding_geoms:
            geom_type = self.model.geom_type[geom_id]
            
            if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                dataid = self.model.geom_dataid[geom_id]
                if dataid >= 0:
                    mesh_id = dataid
                    vert_start = self.model.mesh_vertadr[mesh_id]
                    vert_count = self.model.mesh_vertnum[mesh_id]
                    
                    body_id = self.model.geom_bodyid[geom_id]
                    body_pos = data.xpos[body_id]
                    body_rot = data.xmat[body_id].reshape(3, 3)
                    
                    geom_pos = self.model.geom_pos[geom_id]
                    geom_quat = self.model.geom_quat[geom_id]
                    geom_rot = np.zeros(9)
                    mujoco.mju_quat2Mat(geom_rot, geom_quat)
                    geom_rot = geom_rot.reshape(3, 3)
                    
                    for v_idx in range(vert_start, vert_start + vert_count):
                        vert_local = self.model.mesh_vert[v_idx]
                        vert_geom = geom_rot @ vert_local + geom_pos
                        vert_world = body_rot @ vert_geom + body_pos
                        object_vertices_world.append(vert_world)
            
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                size = self.model.geom_size[geom_id]
                body_id = self.model.geom_bodyid[geom_id]
                body_pos = data.xpos[body_id]
                body_rot = data.xmat[body_id].reshape(3, 3)
                
                geom_pos = self.model.geom_pos[geom_id]
                geom_quat = self.model.geom_quat[geom_id]
                geom_rot = np.zeros(9)
                mujoco.mju_quat2Mat(geom_rot, geom_quat)
                geom_rot = geom_rot.reshape(3, 3)
                
                for dx in [-size[0], size[0]]:
                    for dy in [-size[1], size[1]]:
                        for dz in [-size[2], size[2]]:
                            corner_local = np.array([dx, dy, dz])
                            corner_geom = geom_rot @ corner_local + geom_pos
                            corner_world = body_rot @ corner_geom + body_pos
                            object_vertices_world.append(corner_world)
        
        if len(object_vertices_world) == 0:
            return []
        
        object_vertices_world = np.array(object_vertices_world)
        
        # Now iterate over gel tip vertices and compute distance to nearest object
        distance_field = []
        
        for gel_vert_world in gel_vertices_world:
            # Transform gel vertex to sensor local frame
            relative_pos = gel_vert_world - sensor_pos
            gel_vert_local = sensor_rot.T @ relative_pos
            
            # Check if this gel vertex is within the ROI
            x_relative = gel_vert_local[0] - self.sensing_plane_pos[0]
            z_relative = gel_vert_local[2] - self.sensing_plane_pos[2]
            
            if (abs(x_relative) <= self.roi_half_width and 
                abs(z_relative) <= self.roi_half_height):
                
                # Compute distance to nearest object vertex
                distances = np.linalg.norm(object_vertices_world - gel_vert_world, axis=1)
                min_distance = np.min(distances)
                
                # Convert to mm
                min_distance_mm = min_distance * 1000
                
                # Compute intensity based on proximity threshold
                if min_distance_mm <= self.proximity_threshold * 1000:
                    intensity = 1.0 - (min_distance_mm / (self.proximity_threshold * 1000))
                else:
                    intensity = 0.0
                
                # 2D ROI coordinates
                x_mm = x_relative * 1000
                z_mm = (z_relative + self.roi_half_height) * 1000
                
                distance_field.append({
                    'position_sensor_local': gel_vert_local,
                    'x_mm': x_mm,
                    'y_mm': z_mm,
                    'distance_to_object_mm': min_distance_mm,
                    'intensity': intensity
                })
        
        return distance_field


class UniversalMeshExtractor:
    """Extract mesh vertices from any object for sensor interaction"""
    
    def __init__(self, model, object_geom_name):
        self.model = model
        
        # Get object geometry info
        try:
            self.object_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, object_geom_name)
            self.object_body_id = model.geom_bodyid[self.object_geom_id]
        except:
            raise ValueError(f"Object geometry '{object_geom_name}' not found")
        
        # Extract mesh vertices and geometry transformation
        self.raw_vertices = self._extract_mesh_vertices()
        self.geom_pos, self.geom_rot = self._get_geometry_transformation()
    
    def _extract_mesh_vertices(self):
        """Extract raw mesh vertices"""
        try:
            geom_type = self.model.geom_type[self.object_geom_id]
            if geom_type != mujoco.mjtGeom.mjGEOM_MESH:
                return None
            
            mesh_id = self.model.geom_dataid[self.object_geom_id]
            vert_start = self.model.mesh_vertadr[mesh_id]
            vert_num = self.model.mesh_vertnum[mesh_id]
            
            vertices = self.model.mesh_vert[vert_start:vert_start + vert_num].copy()
            return vertices
        except Exception as e:
            print(f"Error extracting mesh vertices: {e}")
            return None
    
    def _get_geometry_transformation(self):
        """Get geometry transformation"""
        geom_pos = self.model.geom_pos[self.object_geom_id].copy()
        geom_quat = self.model.geom_quat[self.object_geom_id].copy()
        
        # Convert quaternion to rotation matrix
        w, x, y, z = geom_quat[0], geom_quat[1], geom_quat[2], geom_quat[3]
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm == 0:
            geom_rot = np.eye(3)
        else:
            w, x, y, z = w/norm, x/norm, y/norm, z/norm
            geom_rot = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ])
        
        return geom_pos, geom_rot
    
    def get_world_vertices(self, data):
        """Get object vertices in world coordinates"""
        if self.raw_vertices is None:
            return np.array([])
        
        # Get current object pose
        object_pos = data.xpos[self.object_body_id].copy()
        object_rot = data.xmat[self.object_body_id].reshape(3, 3).copy()
        
        # Transform vertices: Raw -> Geometry -> Body -> World
        world_vertices = []
        for vertex in self.raw_vertices:
            # Apply geometry transformation
            geom_vertex = self.geom_rot @ vertex + self.geom_pos
            # Apply body transformation
            world_vertex = object_rot @ geom_vertex + object_pos
            world_vertices.append(world_vertex)
        
        return np.array(world_vertices)
