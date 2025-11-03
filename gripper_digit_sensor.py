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
    
    def get_npe_input_format(self, data):
        """
        Convert sensor output to NPE-compatible MuJoCo input format.
        
        NPE expects:
            - num_contacts: int
            - contact_x_mm: list of floats
            - contact_y_mm: list of floats
            - distance_from_plane_mm: list of floats
        
        Returns:
            dict: MuJoCo input format for NPE inference (3N values)
        """
        contacts = self.detect_proximity_contacts(data)
        
        if len(contacts) == 0:
            return {
                'num_contacts': 0,
                'contact_x_mm': [],
                'contact_y_mm': [],
                'distance_from_plane_mm': []
            }
        
        # Extract only the fields NPE needs (3N values)
        mujoco_input = {
            'num_contacts': len(contacts),
            'contact_x_mm': [c['x_mm'] for c in contacts],
            'contact_y_mm': [c['y_mm'] for c in contacts],
            'distance_from_plane_mm': [c['distance_from_plane_mm'] for c in contacts]
        }
        
        return mujoco_input
    
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
    
    def load_fem_grid(self, csv_path='filtered_FEM_grid.csv'):
        """
        Load FEM grid from CSV file for high-resolution visualization.
        
        Args:
            csv_path: Path to the FEM grid CSV file
            
        Returns:
            pandas DataFrame with 'x' and 'y' columns, or None if file not found
        """
        try:
            import pandas as pd
            fem_grid = pd.read_csv(csv_path)
            print(f"✓ [{self.sensor_type.upper()}] Loaded FEM grid: {len(fem_grid)} nodes")
            print(f"  X range: [{fem_grid['x'].min():.3f}, {fem_grid['x'].max():.3f}] mm")
            print(f"  Y range: [{fem_grid['y'].min():.3f}, {fem_grid['y'].max():.3f}] mm")
            return fem_grid
        except FileNotFoundError:
            print(f"⚠️  [{self.sensor_type.upper()}] Warning: {csv_path} not found. High-res visualization disabled.")
            return None
        except ImportError:
            print(f"⚠️  [{self.sensor_type.upper()}] Warning: pandas not available. Install with: pip install pandas")
            return None
    
    def interpolate_to_fem_grid(self, contacts, fem_grid, influence_radius_mm=0.2):
        """
        Interpolate sparse contact data to fine FEM grid using Gaussian kernel.
        Only nodes near contacts will have non-zero values (creates sparse heatmap).
        Uses adaptive weighting to preserve sharp edges and non-convex features.
        
        Args:
            contacts: List of contact dictionaries from detect_proximity_contacts()
            fem_grid: pandas DataFrame with 'x' and 'y' columns (FEM grid positions)
            influence_radius_mm: Smoothness parameter (default 0.2mm for tight boundaries)
        
        Returns:
            numpy array of distance values at each FEM grid node (length = len(fem_grid))
            Values are 0.0 for nodes far from any contact (creating transparency effect)
        """
        if fem_grid is None or len(contacts) == 0:
            return np.zeros(len(fem_grid)) if fem_grid is not None else np.array([])
        
        # Extract contact positions and distance values (sparse data)
        contact_positions = np.array([[c['x_mm'], c['y_mm']] for c in contacts])
        contact_distances = np.array([c['distance_from_plane_mm'] for c in contacts])
        
        # FEM grid positions (dense target grid)
        fem_positions = fem_grid[['x', 'y']].values
        
        # Compute pairwise distances: (n_fem_nodes, n_contacts)
        distances = np.linalg.norm(
            fem_positions[:, np.newaxis, :] - contact_positions[np.newaxis, :, :],
            axis=2
        )
        
        # Find nearest contact for each FEM node (for edge detection)
        nearest_contact_dist = np.min(distances, axis=1)
        
        # Adaptive influence radius based on local contact density
        # In dense regions (many nearby contacts), use smaller radius for sharper edges
        # In sparse regions, use larger radius for smoother interpolation
        
        # Count contacts within 2x influence radius for each FEM node
        contact_density = np.sum(distances < (2 * influence_radius_mm), axis=1)
        
        # Adaptive radius: smaller in dense regions, larger in sparse regions
        # This preserves sharp features in high-density areas (like donut edges)
        adaptive_radius = influence_radius_mm * np.where(
            contact_density > 5,  # Dense region threshold
            0.7,  # Tighter radius in dense regions (sharper edges)
            1.0   # Normal radius in sparse regions
        )[:, np.newaxis]
        
        # Gaussian weighting with adaptive radius
        influences = np.exp(-(distances / adaptive_radius) ** 2)
        
        # Edge preservation: reduce influence of far contacts
        # This prevents bleeding across gaps (like donut holes)
        edge_threshold = 0.5 * influence_radius_mm  # 0.1mm for sharp edges
        edge_mask = distances < (3 * influence_radius_mm)  # Only consider nearby contacts
        influences = influences * edge_mask
        
        # Weighted average for distance field
        weighted_distances = influences * contact_distances[np.newaxis, :]
        total_weights = np.sum(influences, axis=1)
        
        # Only interpolate where weights are significant
        fem_distance_field = np.zeros(len(fem_positions))
        weight_threshold = 1e-3  # Nodes with weight below this stay at 0 (transparent)
        mask = total_weights > weight_threshold
        fem_distance_field[mask] = np.sum(weighted_distances[mask], axis=1) / total_weights[mask]
        
        # Additional refinement: detect and preserve boundaries
        # Zero out nodes that are too far from any contact (prevents false positives)
        max_distance_threshold = 1.5 * influence_radius_mm  # 0.3mm max distance
        fem_distance_field[nearest_contact_dist > max_distance_threshold] = 0.0
        
        return fem_distance_field


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
