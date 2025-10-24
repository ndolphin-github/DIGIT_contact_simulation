
import mujoco
import numpy as np

class ModularDIGITSensor:
    """Modular DIGIT sensor for contact detection"""
    
    def __init__(self, model, sensor_body_name="digit_sensor"):
        self.model = model
        # DIGIT sensor specifications
        self.sensor_plane_z = 0.004  # 4mm sensing plane
        self.roi_x_range = (-0.0075, 0.0075)  # ±7.5mm in X
        self.roi_y_range = (0.0, 0.015)  # 0-15mm in Y  
        self.proximity_threshold = 0.2  # mm
        
        # Get sensor body ID
        try:
            self.sensor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sensor_body_name)
        except:
            raise ValueError(f"Sensor body '{sensor_body_name}' not found in model")
    
    def get_sensor_pose(self, data):
        """Get current sensor pose in world coordinates"""
        sensor_pos = data.xpos[self.sensor_body_id].copy()
        sensor_rot = data.xmat[self.sensor_body_id].reshape(3, 3).copy()
        return sensor_pos, sensor_rot
    
    def world_to_sensor_coordinates(self, world_points, data):
        """Transform world points to sensor local coordinates"""
        sensor_pos, sensor_rot = self.get_sensor_pose(data)
        
        local_points = []
        for world_point in world_points:
            relative_pos = world_point - sensor_pos
            local_point = sensor_rot.T @ relative_pos
            local_points.append(local_point)
        
        return np.array(local_points)
    
    def detect_proximity_contacts(self, object_vertices_world, data):
        """Detect contacts between object and sensor"""
        # Transform object vertices to sensor coordinates
        sensor_local_vertices = self.world_to_sensor_coordinates(object_vertices_world, data)
        
        # Check distance from sensor plane (Z=4mm in sensor coordinates)
        z_distances = np.abs(sensor_local_vertices[:, 2] - self.sensor_plane_z)
        nearby_mask = z_distances <= (self.proximity_threshold / 1000.0)
        
        # Filter by ROI (15mm x 15mm sensor area)
        x_in_roi = ((sensor_local_vertices[:, 0] >= self.roi_x_range[0]) & 
                    (sensor_local_vertices[:, 0] <= self.roi_x_range[1]))
        y_in_roi = ((sensor_local_vertices[:, 1] >= self.roi_y_range[0]) & 
                    (sensor_local_vertices[:, 1] <= self.roi_y_range[1]))
        roi_mask = x_in_roi & y_in_roi
        
        # Combined contact detection
        contact_mask = nearby_mask & roi_mask
        contact_vertices = sensor_local_vertices[contact_mask]
        contact_distances = z_distances[contact_mask]
        
        # Project onto sensor plane and return results
        projected_vertices = contact_vertices.copy()
        projected_vertices[:, 2] = self.sensor_plane_z
        
        proximity_weights = 1.0 - (contact_distances / (self.proximity_threshold / 1000.0))
        
        results = []
        for i, vertex in enumerate(projected_vertices):
            results.append({
                'position_sensor_local': vertex,
                'x_mm': vertex[0] * 1000,
                'y_mm': vertex[1] * 1000,
                'proximity': proximity_weights[i],
                'distance_mm': contact_distances[i] * 1000
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
