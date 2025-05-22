import torch
import numpy as np
import matplotlib.pyplot as plt
import abc

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
    """
    if not torch.isclose(torch.linalg.norm(axis), torch.tensor(1.0)):
        axis = axis / torch.linalg.norm(axis)

    cos_theta = torch.cos(torch.tensor(angle))
    sin_theta = torch.sin(torch.tensor(angle))
    
    if vector.ndim > 1 and axis.ndim == 1:
        axis = axis.unsqueeze(0) 

    cross_product_kv = torch.cross(axis, vector, dim=-1)
    dot_product_kv = torch.sum(axis * vector, dim=-1, keepdim=True)
    
    term1 = vector * cos_theta
    term2 = cross_product_kv * sin_theta
    term3 = axis * dot_product_kv * (1 - cos_theta)
    
    rotated_vector = term1 + term2 + term3
    
    return rotated_vector

class Coil(abc.ABC):
    """
    Abstract base class for different coil geometries.
    """
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_segments(self) -> torch.Tensor:
        """
        Abstract method to get the line segments representing the coil's geometry.
        Returns:
            torch.Tensor: A tensor of shape (N, 2, 3).
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
        super().__init__()
        self.radius = radius
        self.num_segments = num_segments
        self.center = center.float()
        
        if not torch.isclose(torch.linalg.norm(normal), torch.tensor(1.0)):
            normal = normal.float() / torch.linalg.norm(normal.float())
        self.normal = normal.float()

    def get_segments(self) -> torch.Tensor:
        angles = torch.linspace(0, 2 * np.pi, self.num_segments + 1)
        x_coords = self.radius * torch.cos(angles)
        y_coords = self.radius * torch.sin(angles)
        z_coords = torch.zeros_like(x_coords)
        points_xy = torch.stack((x_coords, y_coords, z_coords), dim=1)
        start_points_xy = points_xy[:-1, :]
        end_points_xy = points_xy[1:, :]
        
        default_normal = torch.tensor([0.0, 0.0, 1.0], device=self.normal.device, dtype=self.normal.dtype)
        if not torch.allclose(self.normal, default_normal):
            rot_axis = torch.cross(default_normal, self.normal)
            if torch.linalg.norm(rot_axis) < 1e-6:
                rot_axis = torch.tensor([1.0, 0.0, 0.0], device=self.normal.device, dtype=self.normal.dtype) if torch.dot(default_normal, self.normal) < 0 else None
                rot_angle = np.pi if rot_axis is not None else 0.0
            else:    
                rot_axis = rot_axis / torch.linalg.norm(rot_axis)
                cos_angle = torch.dot(default_normal, self.normal)
                rot_angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0)).item()

            if rot_axis is not None and rot_angle != 0.0:
                start_points_rotated = rotate_vector(start_points_xy, rot_axis, rot_angle)
                end_points_rotated = rotate_vector(end_points_xy, rot_axis, rot_angle)
            else:
                start_points_rotated = start_points_xy
                end_points_rotated = end_points_xy
        else:
            start_points_rotated = start_points_xy
            end_points_rotated = end_points_xy
            
        start_points_final = translate_vector(start_points_rotated, self.center)
        end_points_final = translate_vector(end_points_rotated, self.center)
        
        segments = torch.stack((start_points_final, end_points_final), dim=1)
        return segments

class SolenoidCoil(Coil):
    """
    Represents a solenoid coil approximated by straight line segments.
    """
    def __init__(self, 
                 radius: float, 
                 length: float, 
                 num_turns: int, 
                 segments_per_turn: int,
                 center: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]), 
                 axis: torch.Tensor = torch.tensor([0.0, 0.0, 1.0])):
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
        total_angle = 2 * np.pi * self.num_turns
        angles = torch.linspace(0, total_angle, self.total_segments + 1)
        x_coords = self.radius * torch.cos(angles)
        y_coords = self.radius * torch.sin(angles)
        z_coords = torch.linspace(-self.length / 2.0, self.length / 2.0, self.total_segments + 1)
        points_helix_z = torch.stack((x_coords, y_coords, z_coords), dim=1)
        start_points_helix_z = points_helix_z[:-1, :]
        end_points_helix_z = points_helix_z[1:, :]

        default_axis = torch.tensor([0.0, 0.0, 1.0], device=self.axis.device, dtype=self.axis.dtype)
        if not torch.allclose(self.axis, default_axis):
            rot_axis_vec = torch.cross(default_axis, self.axis)
            if torch.linalg.norm(rot_axis_vec) < 1e-6:
                rot_axis_vec = torch.tensor([1.0, 0.0, 0.0], device=self.axis.device, dtype=self.axis.dtype) if torch.dot(default_axis, self.axis) < 0 else None
                rot_angle = np.pi if rot_axis_vec is not None else 0.0
            else:
                rot_axis_vec = rot_axis_vec / torch.linalg.norm(rot_axis_vec)
                cos_angle = torch.dot(default_axis, self.axis)
                rot_angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0)).item()
            
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
    Represents a single rung of a birdcage coil.
    """
    def __init__(self, 
                 radius: float, 
                 length: float, 
                 rung_angle_deg: float,
                 num_segments_per_rung: int, 
                 num_segments_per_arc: int,
                 arc_span_deg: float = 10.0,
                 center: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]), 
                 axis: torch.Tensor = torch.tensor([0.0, 0.0, 1.0])):
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
        all_segments = []
        rung_x = self.radius * np.cos(self.rung_angle_rad)
        rung_y = self.radius * np.sin(self.rung_angle_rad)
        z_points_rung = torch.linspace(-self.length / 2.0, self.length / 2.0, self.num_segments_per_rung + 1)
        
        rung_segment_starts_z = torch.zeros((self.num_segments_per_rung, 3))
        rung_segment_ends_z = torch.zeros((self.num_segments_per_rung, 3))
        rung_segment_starts_z[:, 0] = rung_x
        rung_segment_starts_z[:, 1] = rung_y
        rung_segment_starts_z[:, 2] = z_points_rung[:-1]
        rung_segment_ends_z[:, 0] = rung_x
        rung_segment_ends_z[:, 1] = rung_y
        rung_segment_ends_z[:, 2] = z_points_rung[1:]
        all_segments.append(torch.stack((rung_segment_starts_z, rung_segment_ends_z), dim=1))

        for z_level in [-self.length / 2.0, self.length / 2.0]:
            arc_angles = torch.linspace(self.rung_angle_rad - self.arc_span_rad / 2.0, 
                                         self.rung_angle_rad + self.arc_span_rad / 2.0, 
                                         self.num_segments_per_arc + 1)
            arc_x = self.radius * torch.cos(arc_angles)
            arc_y = self.radius * torch.sin(arc_angles)
            arc_z = torch.full_like(arc_x, z_level)
            points_arc_z = torch.stack((arc_x, arc_y, arc_z), dim=1)
            all_segments.append(torch.stack((points_arc_z[:-1, :], points_arc_z[1:, :]), dim=1))

        combined_segments_z = torch.cat(all_segments, dim=0)
        start_points_z = combined_segments_z[:, 0, :]
        end_points_z = combined_segments_z[:, 1, :]

        default_axis = torch.tensor([0.0, 0.0, 1.0], device=self.axis.device, dtype=self.axis.dtype)
        if not torch.allclose(self.axis, default_axis):
            rot_axis_vec = torch.cross(default_axis, self.axis)
            if torch.linalg.norm(rot_axis_vec) < 1e-6:
                rot_axis_vec = torch.tensor([1.0, 0.0, 0.0], device=self.axis.device, dtype=self.axis.dtype) if torch.dot(default_axis, self.axis) < 0 else None
                rot_angle = np.pi if rot_axis_vec is not None else 0.0
            else:
                rot_axis_vec = rot_axis_vec / torch.linalg.norm(rot_axis_vec)
                cos_angle = torch.dot(default_axis, self.axis)
                rot_angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0)).item()
            
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

def calculate_magnetic_field_at_point(
    segments: torch.Tensor, 
    point: torch.Tensor, 
    current_magnitude: float,
    epsilon: float = 1e-12
) -> torch.Tensor:
    B_total = torch.zeros(3, dtype=point.dtype, device=point.device)
    P0 = point
    for i in range(segments.shape[0]):
        P1, P2 = segments[i, 0, :], segments[i, 1, :]
        r1, r2, r21 = P0 - P1, P0 - P2, P2 - P1
        norm_r21 = torch.linalg.norm(r21)
        if norm_r21 < epsilon: continue
        r1_cross_r2 = torch.cross(r1, r2)
        norm_sq_r1_cross_r2 = torch.sum(r1_cross_r2**2)
        if norm_sq_r1_cross_r2 < epsilon**2: continue
        r21_unit = r21 / norm_r21
        scalar_term = torch.dot(r1, r21_unit) - torch.dot(r2, r21_unit)
        B_segment = (mu_0 * current_magnitude * scalar_term) / (4 * torch.pi * norm_sq_r1_cross_r2) * r1_cross_r2
        B_total += B_segment
    return B_total

def generate_b_field_map(
    coil_obj: 'Coil', 
    x_coords: torch.Tensor, 
    y_coords: torch.Tensor, 
    z_coords: torch.Tensor, 
    current_magnitude: float,
    epsilon_b_calc: float = 1e-12
) -> torch.Tensor:
    segments = coil_obj.get_segments()
    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
    output_dtype, output_device = x_coords.dtype, x_coords.device
    b_field_map = torch.zeros((nx, ny, nz, 3), dtype=output_dtype, device=output_device)
    for ix, x_val in enumerate(x_coords):
        for iy, y_val in enumerate(y_coords):
            for iz, z_val in enumerate(z_coords):
                point = torch.tensor([x_val, y_val, z_val], dtype=output_dtype, device=output_device)
                b_field_map[ix, iy, iz, :] = calculate_magnetic_field_at_point(
                    segments, point, current_magnitude, epsilon=epsilon_b_calc
                )
    return b_field_map

def plot_b_field_slice(
    b_field_map: torch.Tensor, 
    x_coords: torch.Tensor, 
    y_coords: torch.Tensor, 
    z_coords: torch.Tensor, 
    slice_axis: str, 
    slice_index: int, 
    component: str = 'magnitude', 
    ax=None, 
    show_plot: bool = True
):
    """
    Visualizes a 2D slice of a 3D B-field map.
    Args:
        b_field_map (torch.Tensor): 4D tensor (Nx, Ny, Nz, 3).
        x_coords, y_coords, z_coords (torch.Tensor): 1D tensors for grid coordinates.
        slice_axis (str): 'x', 'y', or 'z'.
        slice_index (int): Index for the slice.
        component (str): 'magnitude', 'x', 'y', 'z', 'xy_magnitude', 'yz_magnitude', 'xz_magnitude'.
        ax (matplotlib.axes.Axes, optional): Matplotlib Axes.
        show_plot (bool, optional): If True, calls plt.show().
    """
    if slice_axis not in ['x', 'y', 'z']:
        raise ValueError("slice_axis must be one of 'x', 'y', or 'z'.")
    valid_components = ['magnitude', 'x', 'y', 'z', 'xy_magnitude', 'yz_magnitude', 'xz_magnitude']
    if component not in valid_components:
        raise ValueError(f"component must be one of {valid_components}.")

    b_map_np = b_field_map.detach().cpu().numpy()
    x_np, y_np, z_np = x_coords.detach().cpu().numpy(), y_coords.detach().cpu().numpy(), z_coords.detach().cpu().numpy()

    slice_data_comps, plot_dim1_coords, plot_dim2_coords, dim1_label, dim2_label, slice_val_str = None, None, None, "", "", ""

    if slice_axis == 'x':
        if not (0 <= slice_index < b_map_np.shape[0]): raise ValueError("slice_index out of bounds for x-axis.")
        slice_data_comps = b_map_np[slice_index, :, :, :]
        plot_dim1_coords, plot_dim2_coords = y_np, z_np
        dim1_label, dim2_label = "Y-coordinate", "Z-coordinate"
        slice_val_str = f"X = {x_np[slice_index]:.2e}"
        if component == 'yz_magnitude': data_to_plot = np.sqrt(slice_data_comps[:,:,1]**2 + slice_data_comps[:,:,2]**2)
        elif component == 'xy_magnitude' or component == 'xz_magnitude': raise ValueError(f"Component '{component}' not applicable for x-slice.")
    elif slice_axis == 'y':
        if not (0 <= slice_index < b_map_np.shape[1]): raise ValueError("slice_index out of bounds for y-axis.")
        slice_data_comps = b_map_np[:, slice_index, :, :]
        plot_dim1_coords, plot_dim2_coords = x_np, z_np
        dim1_label, dim2_label = "X-coordinate", "Z-coordinate"
        slice_val_str = f"Y = {y_np[slice_index]:.2e}"
        if component == 'xz_magnitude': data_to_plot = np.sqrt(slice_data_comps[:,:,0]**2 + slice_data_comps[:,:,2]**2)
        elif component == 'xy_magnitude' or component == 'yz_magnitude': raise ValueError(f"Component '{component}' not applicable for y-slice.")
    elif slice_axis == 'z':
        if not (0 <= slice_index < b_map_np.shape[2]): raise ValueError("slice_index out of bounds for z-axis.")
        slice_data_comps = b_map_np[:, :, slice_index, :]
        plot_dim1_coords, plot_dim2_coords = x_np, y_np
        dim1_label, dim2_label = "X-coordinate", "Y-coordinate"
        slice_val_str = f"Z = {z_np[slice_index]:.2e}"
        if component == 'xy_magnitude': data_to_plot = np.sqrt(slice_data_comps[:,:,0]**2 + slice_data_comps[:,:,1]**2)
        elif component == 'xz_magnitude' or component == 'yz_magnitude': raise ValueError(f"Component '{component}' not applicable for z-slice.")

    if component == 'magnitude': data_to_plot = np.linalg.norm(slice_data_comps, axis=-1)
    elif component == 'x': data_to_plot = slice_data_comps[:,:,0]
    elif component == 'y': data_to_plot = slice_data_comps[:,:,1]
    elif component == 'z': data_to_plot = slice_data_comps[:,:,2]

    manage_plot = ax is None
    if manage_plot: fig, ax = plt.subplots()

    # imshow expects data as (row, col), so (dim2, dim1) if dim1 is x-axis and dim2 is y-axis.
    # The data_to_plot is currently (dim1_coords_len, dim2_coords_len).
    # We want dim1_coords on x-axis and dim2_coords on y-axis.
    # So, imshow needs data.T
    im = ax.imshow(data_to_plot.T, aspect='auto', origin='lower', 
                   extent=[plot_dim1_coords[0], plot_dim1_coords[-1], plot_dim2_coords[0], plot_dim2_coords[-1]],
                   cmap='viridis')
    
    ax.set_xlabel(dim1_label)
    ax.set_ylabel(dim2_label)
    ax.set_title(f"B-field {component} | Slice at {slice_val_str}")
    plt.colorbar(im, ax=ax, label=f"B-field {component}")

    if manage_plot and show_plot: plt.show()
    return ax
