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
        Detect contacts using gel tip vertices (same principle as ModularDIGITSensor)
        
        Args:
            data: MuJoCo data
            
        Returns:
            List of contact dictionaries with distance_from_plane_mm, intensity, etc.
        """
        # Step 1: Check if MuJoCo detected any collision with this sensor
        has_collision = False
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]
            
            if body1 == self.sensor_body_id or body2 == self.sensor_body_id:
                has_collision = True
                break
        
        if not has_collision:
            return []
        
        # Step 2: Get gel tip vertices in world coordinates
        gel_vertices_world = self.gel_tip_extractor.get_world_vertices(data)
        
        if len(gel_vertices_world) == 0:
            return []
        
        # Step 3: Transform gel vertices to sensor local coordinates
        sensor_pos, sensor_rot = self.get_sensor_pose(data)
        
        gel_vertices_local = []
        for vertex_world in gel_vertices_world:
            relative_pos = vertex_world - sensor_pos
            vertex_local = sensor_rot.T @ relative_pos
            gel_vertices_local.append(vertex_local)
        gel_vertices_local = np.array(gel_vertices_local)
        
        # Step 4: Calculate distance from each gel vertex to the sensing plane
        # Sensing plane is at Y = sensing_distance (±4mm depending on sensor type)
        # Distance along Y-axis (sensing direction)
        distances_along_normal = gel_vertices_local[:, 1] * self.sensing_direction
        
        # Distance from sensing plane (how far gel node is from the ideal sensing plane)
        distance_from_plane = np.abs(distances_along_normal - self.sensing_distance)
        
        # Filter by proximity threshold
        nearby_mask = distance_from_plane <= self.proximity_threshold
        
        # Step 5: Filter by ROI bounds (X and Z)
        x_positions = gel_vertices_local[:, 0]
        z_positions = gel_vertices_local[:, 2] - self.sensing_plane_offset_z  # Relative to sensing plane center
        
        x_in_roi = (x_positions >= -self.roi_half_width) & (x_positions <= self.roi_half_width)
        z_in_roi = (z_positions >= -self.roi_half_height) & (z_positions <= self.roi_half_height)
        roi_mask = x_in_roi & z_in_roi
        
        # Combined contact detection
        contact_mask = nearby_mask & roi_mask
        
        if not np.any(contact_mask):
            return []
        
        # Step 6: Get contact data for detected gel nodes
        contact_vertices = gel_vertices_local[contact_mask]
        contact_distance_from_plane = distance_from_plane[contact_mask]
        contact_x = x_positions[contact_mask]
        contact_z = z_positions[contact_mask]
        
        # Calculate proximity weights (1.0 = touching plane, 0.0 = at threshold)
        proximity_weights = 1.0 - (contact_distance_from_plane / self.proximity_threshold)
        
        # Step 7: Build results (matching ModularDIGITSensor output format)
        results = []
        for i in range(len(contact_x)):
            results.append({
                'position_sensor_local': contact_vertices[i],
                'x_mm': contact_x[i] * 1000,  # X position in ROI (mm)
                'y_mm': contact_z[i] * 1000,  # Z position in ROI (mm) - using Z as "height"
                'distance_from_plane_mm': contact_distance_from_plane[i] * 1000,  # Distance from sensing plane
                'intensity': proximity_weights[i]  # Proximity weight
            })
        
        return results


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
