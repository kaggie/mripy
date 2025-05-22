import torch
import numpy as np

# Physical constant
mu_0 = 4 * np.pi * 1e-7  # Magnetic permeability of free space

def translate_vector(vector: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """
    Translates a vector or a batch of vectors by a given offset.

    Args:
        vector (torch.Tensor): A 3D tensor (x, y, z) or an N x 3 tensor where N is the
                               number of vectors.
        offset (torch.Tensor): A 3D tensor (dx, dy, dz) representing the translation offset.

    Returns:
        torch.Tensor: The translated vector(s) with the same shape as the input 'vector'.
    """
    return vector + offset

def rotate_vector(vector: torch.Tensor, axis: torch.Tensor, angle: float) -> torch.Tensor:
    """
    Rotates a vector or a batch of vectors around a given axis by a specified angle
    using Rodrigues' rotation formula.

    Args:
        vector (torch.Tensor): A 3D tensor (x, y, z) or an N x 3 tensor where N is the
                               number of vectors.
        axis (torch.Tensor): A 3D tensor (ax, ay, az) representing the axis of rotation.
                             This vector should be normalized.
        angle (float): The angle of rotation in radians.

    Returns:
        torch.Tensor: The rotated vector(s) with the same shape as the input 'vector'.

    Raises:
        ValueError: If the axis vector is not normalized (norm is not close to 1).
    """
    if not torch.isclose(torch.linalg.norm(axis), torch.tensor(1.0)):
        # Normalize the axis vector if it's not already normalized.
        # This is important for the correctness of Rodrigues' formula.
        axis = axis / torch.linalg.norm(axis)
        # Alternatively, raise ValueError("Axis vector must be normalized.")

    cos_theta = torch.cos(torch.tensor(angle))
    sin_theta = torch.sin(torch.tensor(angle))
    
    # Ensure axis is in the correct shape for broadcasting if vector is N x 3
    if vector.ndim > 1 and axis.ndim == 1:
        axis = axis.unsqueeze(0) # Reshape axis from [3] to [1, 3] for broadcasting

    # Rodrigues' rotation formula:
    # v_rot = v * cos(theta) + (k x v) * sin(theta) + k * (k . v) * (1 - cos(theta))
    
    # Cross product (k x v)
    # torch.cross expects both tensors to have the same number of dimensions,
    # or one of them to be a 3-element vector.
    # If vector is N x 3 and axis is 1 x 3 (after unsqueeze), cross product works as expected.
    cross_product_kv = torch.cross(axis, vector, dim=-1)

    # Dot product (k . v)
    # For N x 3 vectors and a 1 x 3 axis, we want to compute the dot product for each vector.
    # (axis * vector) results in element-wise multiplication.
    # .sum(dim=-1) sums along the last dimension (the components x,y,z)
    # .unsqueeze(-1) adds a new dimension at the end to make it N x 1 for broadcasting with axis (1x3 or Nx3)
    dot_product_kv = torch.sum(axis * vector, dim=-1, keepdim=True)
    
    term1 = vector * cos_theta
    term2 = cross_product_kv * sin_theta
    term3 = axis * dot_product_kv * (1 - cos_theta)
    
    rotated_vector = term1 + term2 + term3
    
    return rotated_vector

import abc

class Coil(abc.ABC):
    """
    Abstract base class for different coil geometries.
    
    Subclasses must implement the get_segments method to define the coil's geometry
    as a series of straight line segments.
    """
    def __init__(self):
        """
        Initializes the Coil. This base implementation currently does nothing but
        can be extended by subclasses.
        """
        super().__init__()

    @abc.abstractmethod
    def get_segments(self) -> torch.Tensor:
        """
        Abstract method to get the line segments representing the coil's geometry.

        Returns:
            torch.Tensor: A tensor of shape (N, 2, 3) where N is the number of
                          segments. Each segment is represented by its start and
                          end points [[x1,y1,z1], [x2,y2,z2]].
        """
        pass

class CircularLoopCoil(Coil):
    """
    Represents a circular loop coil approximated by straight line segments.
    """
    def __init__(self, 
                 radius: float, 
                 num_segments: int, 
                 center: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]), 
                 normal: torch.Tensor = torch.tensor([0.0, 0.0, 1.0])):
        """
        Initializes a CircularLoopCoil.

        Args:
            radius (float): The radius of the circular loop.
            num_segments (int): The number of straight line segments to approximate the loop.
            center (torch.Tensor, optional): The center of the loop. 
                                             Defaults to torch.tensor([0.0, 0.0, 0.0]).
            normal (torch.Tensor, optional): The normal vector to the plane of the loop. 
                                             Defaults to torch.tensor([0.0, 0.0, 1.0]).
                                             This vector will be normalized.
        """
        super().__init__()
        self.radius = radius
        self.num_segments = num_segments
        self.center = center.float()
        
        if not torch.isclose(torch.linalg.norm(normal), torch.tensor(1.0)):
            normal = normal.float() / torch.linalg.norm(normal.float())
        self.normal = normal.float()

    def get_segments(self) -> torch.Tensor:
        """
        Generates the line segments approximating the circular loop.

        The loop is first created in the XY plane centered at the origin,
        then rotated to align with the specified normal and translated to the center.

        Returns:
            torch.Tensor: A tensor of shape (num_segments, 2, 3) representing the
                          coil segments.
        """
        segments = torch.zeros((self.num_segments, 2, 3))
        angles = torch.linspace(0, 2 * np.pi, self.num_segments + 1)

        # Points for a loop in XY plane, centered at origin
        x_coords = self.radius * torch.cos(angles)
        y_coords = self.radius * torch.sin(angles)
        z_coords = torch.zeros_like(x_coords)
        
        points_xy = torch.stack((x_coords, y_coords, z_coords), dim=1) # Shape: (num_segments+1, 3)

        # Define segments from these points
        start_points_xy = points_xy[:-1, :]
        end_points_xy = points_xy[1:, :]
        
        segments_xy = torch.stack((start_points_xy, end_points_xy), dim=1) # Shape: (num_segments, 2, 3)

        # Rotate and translate the segments
        # Rotation: align default normal (0,0,1) with self.normal
        default_normal = torch.tensor([0.0, 0.0, 1.0])
        if not torch.allclose(self.normal, default_normal):
            # Calculate rotation axis and angle
            rot_axis = torch.cross(default_normal, self.normal)
            if torch.linalg.norm(rot_axis) < 1e-6: # Normals are collinear
                if torch.dot(default_normal, self.normal) < 0: # Normals are opposite
                    rot_axis = torch.tensor([1.0, 0.0, 0.0]) # Rotate by pi around x-axis
                    rot_angle = np.pi
                else: # Normals are same, no rotation needed
                    rot_axis = None 
                    rot_angle = 0.0
            else:    
                rot_axis = rot_axis / torch.linalg.norm(rot_axis)
                cos_angle = torch.dot(default_normal, self.normal)
                rot_angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))

            if rot_axis is not None and rot_angle != 0.0:
                start_points_rotated = rotate_vector(start_points_xy, rot_axis, rot_angle)
                end_points_rotated = rotate_vector(end_points_xy, rot_axis, rot_angle)
            else:
                start_points_rotated = start_points_xy
                end_points_rotated = end_points_xy
        else:
            start_points_rotated = start_points_xy
            end_points_rotated = end_points_xy
            
        # Translate
        start_points_final = translate_vector(start_points_rotated, self.center)
        end_points_final = translate_vector(end_points_rotated, self.center)
        
        segments = torch.stack((start_points_final, end_points_final), dim=1)
        return segments

class SolenoidCoil(Coil):
    """
    Represents a solenoid coil approximated by straight line segments.
    The solenoid is formed by a helix.
    """
    def __init__(self, 
                 radius: float, 
                 length: float, 
                 num_turns: int, 
                 segments_per_turn: int,
                 center: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]), 
                 axis: torch.Tensor = torch.tensor([0.0, 0.0, 1.0])):
        """
        Initializes a SolenoidCoil.

        Args:
            radius (float): Radius of the solenoid.
            length (float): Length of the solenoid along its axis.
            num_turns (int): Number of turns in the solenoid.
            segments_per_turn (int): Number of straight line segments to approximate each turn.
            center (torch.Tensor, optional): Center of the solenoid. 
                                             Defaults to torch.tensor([0.0, 0.0, 0.0]).
            axis (torch.Tensor, optional): Central axis of the solenoid. 
                                           Defaults to torch.tensor([0.0, 0.0, 1.0]).
                                           This vector will be normalized.
        """
        super().__init__()
        self.radius = radius
        self.length = length
        self.num_turns = num_turns
        self.segments_per_turn = segments_per_turn
        self.total_segments = num_turns * segments_per_turn
        self.center = center.float()
        
        if not torch.isclose(torch.linalg.norm(axis), torch.tensor(1.0)):
            axis = axis.float() / torch.linalg.norm(axis.float())
        self.axis = axis.float()

    def get_segments(self) -> torch.Tensor:
        """
        Generates the line segments approximating the solenoid.

        The solenoid is first created along the Z-axis, centered at the origin,
        then rotated to align with the specified axis and translated to the center.

        Returns:
            torch.Tensor: A tensor of shape (num_turns * segments_per_turn, 2, 3) 
                          representing the coil segments.
        """
        total_angle = 2 * np.pi * self.num_turns
        angles = torch.linspace(0, total_angle, self.total_segments + 1)
        
        # Points for a helix along Z-axis, centered at origin
        x_coords = self.radius * torch.cos(angles)
        y_coords = self.radius * torch.sin(angles)
        # z_coords go from -length/2 to length/2
        z_coords = torch.linspace(-self.length / 2.0, self.length / 2.0, self.total_segments + 1)
        
        points_helix_z = torch.stack((x_coords, y_coords, z_coords), dim=1)

        start_points_helix_z = points_helix_z[:-1, :]
        end_points_helix_z = points_helix_z[1:, :]

        # Rotate and translate
        default_axis = torch.tensor([0.0, 0.0, 1.0])
        if not torch.allclose(self.axis, default_axis):
            rot_axis_vec = torch.cross(default_axis, self.axis)
            if torch.linalg.norm(rot_axis_vec) < 1e-6: # Axes are collinear
                if torch.dot(default_axis, self.axis) < 0: # Axes are opposite
                    rot_axis_vec = torch.tensor([1.0, 0.0, 0.0]) 
                    rot_angle = np.pi
                else: # Axes are same
                    rot_axis_vec = None
                    rot_angle = 0.0
            else:
                rot_axis_vec = rot_axis_vec / torch.linalg.norm(rot_axis_vec)
                cos_angle = torch.dot(default_axis, self.axis)
                rot_angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))

            if rot_axis_vec is not None and rot_angle != 0.0:
                start_points_rotated = rotate_vector(start_points_helix_z, rot_axis_vec, rot_angle)
                end_points_rotated = rotate_vector(end_points_helix_z, rot_axis_vec, rot_angle)
            else:
                start_points_rotated = start_points_helix_z
                end_points_rotated = end_points_helix_z
        else:
            start_points_rotated = start_points_helix_z
            end_points_rotated = end_points_helix_z
            
        start_points_final = translate_vector(start_points_rotated, self.center)
        end_points_final = translate_vector(end_points_rotated, self.center)
        
        segments = torch.stack((start_points_final, end_points_final), dim=1)
        return segments

class SingleRungBirdcageCoil(Coil):
    """
    Represents a single rung of a birdcage coil, including small connecting arcs
    at each end to conceptual end-rings.
    """
    def __init__(self, 
                 radius: float, 
                 length: float, 
                 rung_angle_deg: float,
                 num_segments_per_rung: int, 
                 num_segments_per_arc: int,
                 arc_span_deg: float = 10.0, # Total span of each arc (e.g., 10 deg means +/- 5 deg from rung)
                 center: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]), 
                 axis: torch.Tensor = torch.tensor([0.0, 0.0, 1.0])):
        """
        Initializes a SingleRungBirdcageCoil.

        Args:
            radius (float): Radius of the birdcage cylinder.
            length (float): Length of the birdcage rung (straight part).
            rung_angle_deg (float): Angular position of the rung in degrees
                                    (e.g., 0 degrees is along the x-axis in the XY plane
                                    if axis is Z).
            num_segments_per_rung (int): Number of segments for the straight rung.
            num_segments_per_arc (int): Number of segments for each connecting arc.
            arc_span_deg (float, optional): The angular span of each arc segment at the ends
                                            of the rung, in degrees. Defaults to 10.0 degrees.
                                            This means the arc will span from 
                                            rung_angle_deg - arc_span_deg/2 to 
                                            rung_angle_deg + arc_span_deg/2.
            center (torch.Tensor, optional): Center of the birdcage cylinder.
                                             Defaults to torch.tensor([0.0, 0.0, 0.0]).
            axis (torch.Tensor, optional): Central axis of the birdcage cylinder.
                                           Defaults to torch.tensor([0.0, 0.0, 1.0]).
                                           This vector will be normalized.
        """
        super().__init__()
        self.radius = radius
        self.length = length
        self.rung_angle_rad = np.deg2rad(rung_angle_deg)
        self.num_segments_per_rung = num_segments_per_rung
        self.num_segments_per_arc = num_segments_per_arc
        self.arc_span_rad = np.deg2rad(arc_span_deg)
        self.center = center.float()

        if not torch.isclose(torch.linalg.norm(axis), torch.tensor(1.0)):
            axis = axis.float() / torch.linalg.norm(axis.float())
        self.axis = axis.float()

    def get_segments(self) -> torch.Tensor:
        """
        Generates segments for the single rung and its two connecting arcs.

        The structure is first created with its axis along Z, centered at the origin,
        then rotated and translated. The rung is placed at the specified `rung_angle_rad`.

        Returns:
            torch.Tensor: A tensor of shape 
                          (num_segments_per_rung + 2 * num_segments_per_arc, 2, 3)
                          representing the coil segments.
        """
        all_segments = []

        # 1. Create the straight rung (along Z-axis initially, then rotated to self.axis)
        # Rung endpoints in the XY plane (before considering length along Z)
        rung_x = self.radius * np.cos(self.rung_angle_rad)
        rung_y = self.radius * np.sin(self.rung_angle_rad)
        
        # Rung points along Z, from -length/2 to +length/2
        z_points_rung = torch.linspace(-self.length / 2.0, self.length / 2.0, self.num_segments_per_rung + 1)
        
        # Start and end points for each segment of the rung (in XY plane at rung_angle, extended along Z)
        rung_segment_starts_z = torch.zeros((self.num_segments_per_rung, 3))
        rung_segment_ends_z = torch.zeros((self.num_segments_per_rung, 3))

        rung_segment_starts_z[:, 0] = rung_x
        rung_segment_starts_z[:, 1] = rung_y
        rung_segment_starts_z[:, 2] = z_points_rung[:-1]
        
        rung_segment_ends_z[:, 0] = rung_x
        rung_segment_ends_z[:, 1] = rung_y
        rung_segment_ends_z[:, 2] = z_points_rung[1:]
        
        rung_segments_z = torch.stack((rung_segment_starts_z, rung_segment_ends_z), dim=1)
        all_segments.append(rung_segments_z)

        # 2. Create the two connecting arcs (in XY plane initially, then placed at ends of rung)
        # Arc 1 (bottom, at z = -length/2)
        # Arc 2 (top, at z = +length/2)
        arc_angles_start = self.rung_angle_rad - self.arc_span_rad / 2.0
        arc_angles_end = self.rung_angle_rad + self.arc_span_rad / 2.0
        
        # For simplicity, one arc goes from start_angle to rung_angle, other from rung_angle to end_angle
        # Or, consider each arc as centered on the rung_angle_rad, spanning arc_span_rad / 2 on each side.
        # Let's define the arc points relative to the rung's angular position.
        # Arc points are at self.radius.
        
        # Angles for one side of the arc (e.g., counter-clockwise from rung)
        angles1 = torch.linspace(self.rung_angle_rad, arc_angles_end, self.num_segments_per_arc + 1)
        # Angles for other side of the arc (e.g., clockwise, so rung_angle to arc_angles_start)
        angles2 = torch.linspace(self.rung_angle_rad, arc_angles_start, self.num_segments_per_arc + 1)

        # We need two distinct arcs, one at each end.
        # Let's make each arc span half the arc_span_rad, one clockwise, one counter-clockwise from the rung's connection point.
        # To make two small arcs that connect to the rung ends and extend outwards:
        # Arc 1: from rung_angle_rad to rung_angle_rad + arc_span_rad / 2
        # Arc 2: from rung_angle_rad to rung_angle_rad - arc_span_rad / 2
        # This seems more like a Y-shape rather than connecting to end-rings.

        # Correct approach: The arcs are part of the end-rings.
        # The rung connects to point (self.radius * cos(self.rung_angle_rad), self.radius * sin(self.rung_angle_rad), +/- self.length/2)
        # The arcs should extend from this point along the circumference.
        
        # Arc at z = -length/2
        arc1_angles = torch.linspace(self.rung_angle_rad - self.arc_span_rad / 2.0, 
                                     self.rung_angle_rad + self.arc_span_rad / 2.0, 
                                     self.num_segments_per_arc + 1)
        
        arc1_x = self.radius * torch.cos(arc1_angles)
        arc1_y = self.radius * torch.sin(arc1_angles)
        arc1_z = torch.full_like(arc1_x, -self.length / 2.0)
        points_arc1_z = torch.stack((arc1_x, arc1_y, arc1_z), dim=1)
        segments_arc1_z = torch.stack((points_arc1_z[:-1, :], points_arc1_z[1:, :]), dim=1)
        all_segments.append(segments_arc1_z)

        # Arc at z = +length/2
        arc2_angles = torch.linspace(self.rung_angle_rad - self.arc_span_rad / 2.0, 
                                     self.rung_angle_rad + self.arc_span_rad / 2.0, 
                                     self.num_segments_per_arc + 1)
        arc2_x = self.radius * torch.cos(arc2_angles)
        arc2_y = self.radius * torch.sin(arc2_angles)
        arc2_z = torch.full_like(arc2_x, self.length / 2.0)
        points_arc2_z = torch.stack((arc2_x, arc2_y, arc2_z), dim=1)
        segments_arc2_z = torch.stack((points_arc2_z[:-1, :], points_arc2_z[1:, :]), dim=1)
        all_segments.append(segments_arc2_z)

        combined_segments_z = torch.cat(all_segments, dim=0)

        # Rotate and translate all segments
        default_axis = torch.tensor([0.0, 0.0, 1.0])
        start_points_z = combined_segments_z[:, 0, :]
        end_points_z = combined_segments_z[:, 1, :]

        if not torch.allclose(self.axis, default_axis):
            rot_axis_vec = torch.cross(default_axis, self.axis)
            if torch.linalg.norm(rot_axis_vec) < 1e-6:
                if torch.dot(default_axis, self.axis) < 0:
                    rot_axis_vec = torch.tensor([1.0, 0.0, 0.0])
                    rot_angle = np.pi
                else:
                    rot_axis_vec = None
                    rot_angle = 0.0
            else:
                rot_axis_vec = rot_axis_vec / torch.linalg.norm(rot_axis_vec)
                cos_angle = torch.dot(default_axis, self.axis)
                rot_angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
            
            if rot_axis_vec is not None and rot_angle != 0.0:
                start_points_rotated = rotate_vector(start_points_z, rot_axis_vec, rot_angle)
                end_points_rotated = rotate_vector(end_points_z, rot_axis_vec, rot_angle)
            else:
                start_points_rotated = start_points_z
                end_points_rotated = end_points_z
        else:
            start_points_rotated = start_points_z
            end_points_rotated = end_points_z
            
        start_points_final = translate_vector(start_points_rotated, self.center)


        end_points_final = translate_vector(end_points_rotated, self.center)
        
        final_segments = torch.stack((start_points_final, end_points_final), dim=1)
        return final_segments




def main():
    """
    Main function to demonstrate Biot-Savart simulation for a circular loop coil.
    """
    print("Starting Biot-Savart simulation demonstration...")

    # 1. Setup Simulation Parameters
    print("Setting up simulation parameters...")
    # Coil parameters
    radius = 0.1  # meters (10 cm)
    num_segments_coil = 100
    current = 1.0  # Amperes
    coil_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    coil_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)  # Loop in XY plane

    # B-field map parameters
    # Define ranges for a slice visualization (e.g., XY plane at Z=0)
    # And for a line plot along Z-axis
    map_x_range = [-0.05, 0.05]  # meters
    map_y_range = [-0.05, 0.05]
    map_z_value_for_xy_slice = 0.0 
    
    # For the 3D map to get the slice from, we need full 3D coordinates.
    # Let's define a small 3D region for visualization if needed, or just the slice.
    # The task asks for B_xy_magnitude in the central XY plane.
    # We can generate a 2D map or a 3D map and take a slice.
    # The current generate_b_field_map is 3D.
    
    # For plot_b_field_slice, we need 1D coordinate tensors for X, Y, Z
    # that formed the basis of the b_field_map.
    resolution = 31 # Odd number to ensure a point at the center for slice_index
    
    x_coords = torch.linspace(map_x_range[0], map_x_range[1], resolution, dtype=torch.float32)
    y_coords = torch.linspace(map_y_range[0], map_y_range[1], resolution, dtype=torch.float32)
    # For the XY slice at Z=0, we need a z_coords array that includes 0.
    # Let's make a z_coords that is suitable for a small 3D volume around z=0.
    z_coords_3d_map = torch.linspace(-0.05, 0.05, resolution, dtype=torch.float32)


    # 2. Create Coil Object
    print("Creating coil object...")
    loop_coil = CircularLoopCoil(
        radius=radius,
        num_segments=num_segments_coil,
        center=coil_center,
        normal=coil_normal
    )

    # 3. Simulate B-Field Map (for Central XY Plane visualization)
    print("Simulating B-field map for XY slice...")
    # Generate a 3D B-field map
    b_field_map_3d = generate_b_field_map(
        loop_coil,
        x_coords,
        y_coords,
        z_coords_3d_map, # Use the 3D z_coords here
        current
    )
    
    # Find the index for the central XY plane (z=0)
    # We made z_coords_3d_map centered around 0 with an odd number of points.
    z_slice_idx = resolution // 2 
    # Verify that z_coords_3d_map[z_slice_idx] is close to 0
    print(f"Central Z-slice for XY plane plot is at Z = {z_coords_3d_map[z_slice_idx]:.3e} m (index {z_slice_idx})")


    # Plot B_xy_magnitude in the central XY plane (Z=0)
    print("Plotting B_xy_magnitude in central XY plane...")
    fig_xy_slice, ax_xy_slice = plt.subplots(figsize=(8, 7))
    plot_b_field_slice(
        b_field_map=b_field_map_3d,
        x_coords=x_coords,
        y_coords=y_coords,
        z_coords=z_coords_3d_map, # Pass the z_coords used for the map
        slice_axis='z',
        slice_index=z_slice_idx,
        component='xy_magnitude', # Magnitude of Bx and By components
        ax=ax_xy_slice,
        show_plot=False # Will show at the end
    )
    ax_xy_slice.set_title(f"$B_{{xy}}$ magnitude in XY plane (Z={z_coords_3d_map[z_slice_idx]:.2e}m)")


    # 4. Simulate and Plot Bz along the Z-axis
    print("Simulating and plotting Bz along the Z-axis...")
    # Define points along the Z-axis
    z_line_plot_range = [-0.15, 0.15] # meters
    num_points_z_line = 100
    z_axis_points_single_dim = torch.linspace(z_line_plot_range[0], z_line_plot_range[1], num_points_z_line, dtype=torch.float32)
    
    # Create 3D points for calculate_magnetic_field_at_point (X=0, Y=0)
    z_axis_points_3d = torch.stack([
        torch.zeros_like(z_axis_points_single_dim),
        torch.zeros_like(z_axis_points_single_dim),
        z_axis_points_single_dim
    ], dim=1)

    # Get coil segments (can be done once)
    coil_segments = loop_coil.get_segments()

    # Calculate Bz at each point on the Z-axis
    bz_values_on_axis = torch.zeros(num_points_z_line, dtype=torch.float32)
    for i, point_on_axis in enumerate(z_axis_points_3d):
        b_field_at_point = calculate_magnetic_field_at_point(
            segments=coil_segments,
            point=point_on_axis,
            current_magnitude=current
        )
        bz_values_on_axis[i] = b_field_at_point[2]  # Z-component of B

    # Plot Bz vs Z-coordinate
    fig_bz_line, ax_bz_line = plt.subplots(figsize=(8, 6))
    ax_bz_line.plot(z_axis_points_single_dim.numpy(), bz_values_on_axis.numpy())
    ax_bz_line.set_xlabel("Z-coordinate (m) along coil axis")
    ax_bz_line.set_ylabel("$B_z$ component (Tesla)")
    ax_bz_line.set_title("$B_z$ along the axis of a circular loop coil (R=0.1m, I=1A)")
    ax_bz_line.grid(True)

    print("Simulation and plotting complete.")
    plt.show() # Show all plots
