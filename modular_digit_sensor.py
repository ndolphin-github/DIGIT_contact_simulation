
import mujoco
import numpy as np

class ModularDIGITSensor:
    """Modular DIGIT sensor for contact detection that works with any gel tip orientation"""
    
    def __init__(self, model, sensor_body_name="digit_sensor", 
                 surface_normal=np.array([0, 0, 1]), 
                 roi_axis_x=np.array([1, 0, 0]),
                 roi_axis_y=np.array([0, 1, 0]),
                 sensing_distance=0.004,
                 roi_width=0.015,
                 roi_height=0.015,
                 roi_offset_y=0.0,
                 roi_center_offset=np.array([0, 0, 0]),
                 gel_thickness_mm=0.3,
                 num_field_nodes=7509,
                 field_influence_radius_mm=0.1):  # Reduced from 3.0 to 1.5 for sharper edges
   
        self.model = model
        
        # Normalize direction vectors
        self.surface_normal = np.array(surface_normal) / np.linalg.norm(surface_normal)
        self.roi_axis_x = np.array(roi_axis_x) / np.linalg.norm(roi_axis_x)
        self.roi_axis_y = np.array(roi_axis_y) / np.linalg.norm(roi_axis_y)
        
        # DIGIT sensor specifications
        self.sensing_distance = sensing_distance  # 4mm sensing plane distance
        self.roi_width = roi_width  # 15mm width
        self.roi_height = roi_height  # 15mm height
        self.roi_offset_y = roi_offset_y  # Y offset for ROI (0 means ROI at [0, roi_height])
        self.proximity_threshold = 0.0008  #  proximity threshold (increased for better gradient)
        self.roi_center_offset = np.array(roi_center_offset)  # Offset from sensor origin to ROI center
        
        # Enhanced deformation tracking parameters
        self.gel_thickness_mm = gel_thickness_mm  # Physical gel thickness for depth calculation
        self.num_field_nodes = num_field_nodes  # Number of nodes in continuous contact field (matches FEM)
        self.field_influence_radius_mm = field_influence_radius_mm  # Spatial influence radius for field interpolation
        
        # Get sensor body ID
        try:
            self.sensor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sensor_body_name)
        except:
            raise ValueError(f"Sensor body '{sensor_body_name}' not found in model")
        
        # Extract gel tip mesh vertices (used as contact nodes)
        self.gel_tip_extractor = UniversalMeshExtractor(model, sensor_body_name)
        
        # Store rest (undeformed) positions of gel nodes
        self.gel_rest_positions_local = None
        
        # Pre-generate field node positions for consistent field generation
        self._initialize_field_nodes()
    
    def _initialize_field_nodes(self):
        """Pre-generate field node positions for continuous contact field generation"""
        # Create a uniform grid covering the DIGIT sensor ROI
        # This ensures consistent field structure across all measurements
        grid_size = int(np.sqrt(self.num_field_nodes))
        
        x_nodes = np.linspace(-self.roi_width/2, self.roi_width/2, grid_size)
        y_nodes = np.linspace(self.roi_offset_y, self.roi_offset_y + self.roi_height, grid_size)
        
        # Generate 2D grid positions (in mm)
        self.field_node_positions = []
        for y in y_nodes:
            for x in x_nodes:
                if len(self.field_node_positions) < self.num_field_nodes:
                    self.field_node_positions.append([x * 1000, y * 1000])  # Convert to mm
        
        self.field_node_positions = np.array(self.field_node_positions)
    
    def initialize_gel_rest_positions(self, data):
        """
        Store initial gel node positions (undeformed state).
        Call this after creating the sensor with initial data.
        """
        # Get gel vertices in world coordinates
        gel_vertices_world = self.gel_tip_extractor.get_world_vertices(data)
        
        # Transform to sensor local coordinates and store as rest positions
        self.gel_rest_positions_local = self.world_to_sensor_coordinates(gel_vertices_world, data)
        
        print(f"✓ Initialized {len(self.gel_rest_positions_local)} gel rest positions")
        
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
            # Subtract the ROI center offset to work relative to sensing region
            local_point = local_point - self.roi_center_offset
            local_points.append(local_point)
        
        return np.array(local_points)
    
    def detect_proximity_contacts(self, object_vertices_world, data):
        """Detect contacts using gel tip vertices (not object vertices).
        Only activates gel tip nodes that are actually in contact (checked via MuJoCo collisions)."""
        
        # First check if there's any actual collision with this sensor
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
        
        # If no collision, return empty (no activated nodes)
        if not has_collision:
            return []
        
        if len(object_vertices_world) == 0:
            return []
        
        # Get gel tip vertices in world coordinates
        gel_vertices_world = self.gel_tip_extractor.get_world_vertices(data)
        if len(gel_vertices_world) == 0:
            return []
        
        # Transform gel tip vertices to sensor coordinates
        sensor_local_vertices = self.world_to_sensor_coordinates(gel_vertices_world, data)
        
        # ===== CONTACT DETECTION (unchanged): Find gel nodes near object =====
        # For each gel tip vertex, find minimum distance to any object vertex
        min_distances = []
        for gel_vertex_world in gel_vertices_world:
            distances = np.linalg.norm(object_vertices_world - gel_vertex_world, axis=1)
            min_dist = np.min(distances)
            min_distances.append(min_dist)
        min_distances = np.array(min_distances)
        
        # Project gel vertices onto ROI coordinate system
        # Distance along surface normal (positive = toward sensing plane)
        distances_along_normal = np.dot(sensor_local_vertices, self.surface_normal)
        
        # Calculate distance from sensing plane
        distance_from_plane = np.abs(distances_along_normal - self.sensing_distance)
        
        # Filter by proximity to object AND distance from plane
        # Only show nodes within proximity_threshold (e.g., 1mm) from sensing plane
        nearby_object_mask = min_distances <= self.proximity_threshold
        near_plane_mask = distance_from_plane <= self.proximity_threshold
        
        # Project vertices onto ROI plane
        x_coords = np.dot(sensor_local_vertices, self.roi_axis_x)
        y_coords = np.dot(sensor_local_vertices, self.roi_axis_y)
        
        # Filter by ROI boundaries (centered at origin for X, offset for Y)
        x_in_roi = (x_coords >= -self.roi_width/2) & (x_coords <= self.roi_width/2)
        y_in_roi = (y_coords >= self.roi_offset_y) & (y_coords <= self.roi_offset_y + self.roi_height)
        roi_mask = x_in_roi & y_in_roi
        
        # Combined contact detection: must be near object AND near plane AND in ROI
        contact_mask = nearby_object_mask & near_plane_mask & roi_mask
        
        if not np.any(contact_mask):
            return []
        
        # Get contact data
        contact_vertices = sensor_local_vertices[contact_mask]
        contact_distances = min_distances[contact_mask]
        contact_x = x_coords[contact_mask]
        contact_y = y_coords[contact_mask]
        contact_normal_dist = distances_along_normal[contact_mask]
        contact_distance_from_plane = distance_from_plane[contact_mask]
        
        # Calculate proximity weights based on DISTANCE FROM SENSING PLANE
        # Nodes closer to sensing plane = higher proximity (smaller distance)
        # proximity_threshold is the max distance from plane to consider (e.g., 1mm)
        proximity_weights = 1.0 - (contact_distance_from_plane / self.proximity_threshold)
        proximity_weights = np.clip(proximity_weights, 0.0, 1.0)
        
        # Return results with 2D ROI coordinates
        results = []
        for i in range(len(contact_x)):
            results.append({
                'position_sensor_local': contact_vertices[i],
                'x_mm': contact_x[i] * 1000,
                'y_mm': contact_y[i] * 1000,
                'proximity': proximity_weights[i],
                'distance_mm': contact_distances[i] * 1000,
                'distance_from_plane_mm': contact_distance_from_plane[i] * 1000,
                'intensity': proximity_weights[i]
            })
        
        return results
    
    def get_continuous_contact_field(self, contacts, fast_mode=True):
        """
        Generate continuous contact field (7509 nodes) from discrete contact points.
        This represents the deformation field matching FEM output structure.
        
        Args:
            contacts: List of contact dictionaries from detect_proximity_contacts()
            fast_mode: If True, use faster vectorized computation (recommended for real-time)
        
        Returns:
            numpy array of shape (num_field_nodes,) with continuous deformation values
        """
        # Initialize field with zeros
        contact_field = np.zeros(self.num_field_nodes)
        
        if len(contacts) == 0:
            return contact_field
        
        # Extract contact information (vectorized)
        contact_positions = np.array([[c['x_mm'], c['y_mm']] for c in contacts])  # (N_contacts, 2)
        contact_intensities = np.array([c['intensity'] for c in contacts])  # (N_contacts,)
        
        if fast_mode:
            # ===== FAST VECTORIZED VERSION =====
            # Compute all distances at once: (N_nodes, N_contacts)
            # This is much faster than nested loops
            
            # Only use actual field nodes (may be less than num_field_nodes)
            n_actual_nodes = len(self.field_node_positions)
            
            # Reshape for broadcasting: (N_nodes, 1, 2) - (1, N_contacts, 2) -> (N_nodes, N_contacts)
            distances = np.linalg.norm(
                self.field_node_positions[:, np.newaxis, :] - contact_positions[np.newaxis, :, :],
                axis=2
            )  # Shape: (N_actual_nodes, N_contacts)
            
            # Compute influence weights using Gaussian kernel (vectorized)
            influences = np.exp(-(distances / self.field_influence_radius_mm) ** 2)  # (N_actual_nodes, N_contacts)
            
            # Weighted sum: each node gets weighted average of contact intensities
            weighted_intensities = influences * contact_intensities[np.newaxis, :]  # (N_actual_nodes, N_contacts)
            total_weights = np.sum(influences, axis=1)  # (N_actual_nodes,)
            
            # Avoid division by zero
            mask = total_weights > 0
            contact_field[:n_actual_nodes][mask] = np.sum(weighted_intensities[mask], axis=1) / total_weights[mask]
            
        else:
            # ===== SLOW LOOP VERSION (for reference/debugging) =====
            for node_idx, node_pos in enumerate(self.field_node_positions):
                weighted_intensity = 0.0
                total_weight = 0.0
                
                for i, contact_pos in enumerate(contact_positions):
                    distance = np.linalg.norm(node_pos - contact_pos)
                    influence = np.exp(-(distance / self.field_influence_radius_mm) ** 2)
                    weighted_intensity += influence * contact_intensities[i]
                    total_weight += influence
                
                if total_weight > 0:
                    contact_field[node_idx] = weighted_intensity / total_weight
        
        return contact_field
    
    def get_contact_statistics(self, contacts):
        """
        Calculate summary statistics from contact data.
        Useful for quick analysis and force estimation.
        
        Args:
            contacts: List of contact dictionaries from detect_proximity_contacts()
        
        Returns:
            Dictionary with contact statistics
        """
        if len(contacts) == 0:
            return {
                'num_contacts': 0,
                'avg_distance_from_plane_mm': 0.0,
                'min_distance_from_plane_mm': 0.0,
                'estimated_force_N': 0.0,
                'contact_area_coverage': 0.0,
                'avg_intensity': 0.0
            }
        
        distances_from_plane = np.array([c['distance_from_plane_mm'] for c in contacts])
        intensities = np.array([c['intensity'] for c in contacts])
        
        # Estimate contact area (approximate)
        x_coords = [c['x_mm'] for c in contacts]
        y_coords = [c['y_mm'] for c in contacts]
        contact_area_mm2 = (np.ptp(x_coords) * np.ptp(y_coords)) if len(x_coords) > 1 else 0.0
        roi_area_mm2 = self.roi_width * 1000 * self.roi_height * 1000
        coverage = min(1.0, contact_area_mm2 / roi_area_mm2)
        
        # Force estimation based on number of contacts and intensity
        estimated_force = len(contacts) * np.mean(intensities) * 0.001  # N
        
        return {
            'num_contacts': len(contacts),
            'avg_distance_from_plane_mm': float(np.mean(distances_from_plane)),
            'min_distance_from_plane_mm': float(np.min(distances_from_plane)),
            'estimated_force_N': float(estimated_force),
            'contact_area_coverage': float(coverage),
            'avg_intensity': float(np.mean(intensities))
        }
    
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
            print(f"✓ Loaded FEM grid: {len(fem_grid)} nodes")
            print(f"  X range: [{fem_grid['x'].min():.3f}, {fem_grid['x'].max():.3f}] mm")
            print(f"  Y range: [{fem_grid['y'].min():.3f}, {fem_grid['y'].max():.3f}] mm")
            return fem_grid
        except FileNotFoundError:
            print(f"⚠️  Warning: {csv_path} not found. High-res visualization disabled.")
            return None
        except ImportError:
            print(f"⚠️  Warning: pandas not available. Install with: pip install pandas")
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
