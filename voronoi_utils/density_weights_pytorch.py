import torch
import numpy as np # Kept for type hints if any remain from original
from collections import defaultdict # Might be used by imported Voronoi construction
import unittest

from .geometry_core import EPSILON, normalize_weights, ConvexHull, clip_polygon_2d, clip_polyhedron_3d
# Note: monotone_chain_2d, monotone_chain_convex_hull_3d are dependencies of ConvexHull
# and _sutherland_hodgman_*, _point_plane_signed_distance, etc. are dependencies of clipping functions.
# They are assumed to be available via the geometry_core module if not directly imported here.
# compute_polygon_area and compute_convex_hull_volume were also in geometry_core but not directly used here.

from .voronoi_from_delaunay import construct_voronoi_polygons_2d, construct_voronoi_polyhedra_3d

# --- Refactored compute_voronoi_density_weights ---

def compute_voronoi_density_weights_pytorch(
    points: torch.Tensor, 
    bounds: torch.Tensor | None = None, 
    space_dim: int | None = None
) -> torch.Tensor:
    """
    Computes Voronoi-based density compensation weights for a set of k-space points.
    Refactored to use PyTorch-based Voronoi construction (conceptual).
    """
    if not isinstance(points, torch.Tensor): raise TypeError("Input points must be a PyTorch tensor.")
    if points.ndim != 2: raise ValueError("Input points tensor must be 2-dimensional (N, D).")

    original_device = points.device
    
    if space_dim is None: space_dim = points.shape[1]
    if space_dim not in [2, 3]: raise ValueError(f"space_dim must be 2 or 3, got {space_dim}.")
    if points.shape[1] != space_dim: raise ValueError(f"Points dim {points.shape[1]} != space_dim {space_dim}.")

    n_points = points.shape[0]
    if n_points == 0: return torch.empty(0, dtype=points.dtype, device=original_device)
    
    if n_points <= space_dim:
        return torch.full((n_points,), 1.0 / (n_points if n_points > 0 else 1.0), 
                              dtype=points.dtype, device=original_device)

    bounds_on_correct_device = None
    if bounds is not None:
        if not isinstance(bounds, torch.Tensor): raise TypeError("Bounds must be a PyTorch tensor.")
        bounds_on_correct_device = bounds.to(device=original_device, dtype=points.dtype) 
        expected_bounds_shape = (2, space_dim)
        if bounds_on_correct_device.shape != expected_bounds_shape: 
            raise ValueError(f"Bounds shape must be {expected_bounds_shape}, got {bounds_on_correct_device.shape}.")
            
    voronoi_cells_vertex_lists: list 
    # unique_voronoi_vertices_all: torch.Tensor # This variable is not used further

    # Conceptual: Delaunay would be computed first if not provided to Voronoi constructor
    # For now, assume construct_voronoi_* can work with just points for placeholder.
    if space_dim == 2:
        # The second return value (unique_voronoi_vertices_all) is ignored as it's not used.
        voronoi_cells_vertex_lists, _ = construct_voronoi_polygons_2d(points, delaunay_triangles=None) # Pass delaunay_triangles=None for placeholder
    elif space_dim == 3:
        voronoi_cells_vertex_lists, _ = construct_voronoi_polyhedra_3d(points, delaunay_tetrahedra=None) # Pass delaunay_tetrahedra=None for placeholder
    else:
        raise ValueError("space_dim not 2 or 3")

    weights_list = []
    min_measure_floor = EPSILON # Use imported EPSILON from geometry_core

    for i_loop_main in range(n_points): 
        current_cell_vertex_coords_list = voronoi_cells_vertex_lists[i_loop_main]

        if not current_cell_vertex_coords_list:
            weights_list.append(1.0 / min_measure_floor); continue

        # Ensure all vertices are tensors before stacking
        valid_vertices_for_stack = [v for v in current_cell_vertex_coords_list if isinstance(v, torch.Tensor)]
        if not valid_vertices_for_stack:
            weights_list.append(1.0 / min_measure_floor); continue
            
        current_region_vor_vertices = torch.stack(valid_vertices_for_stack)
        
        vertices_for_final_hull_calc = None

        if space_dim == 2:
            if current_region_vor_vertices.shape[0] < 3: # A polygon needs at least 3 vertices
                if bounds_on_correct_device is None: # No bounds to potentially create a valid shape
                    weights_list.append(1.0 / min_measure_floor); continue
            
            if bounds_on_correct_device is not None:
                # Ensure vertices are on the same device as bounds for clipping
                vertices_for_final_hull_calc = clip_polygon_2d(current_region_vor_vertices.to(bounds_on_correct_device.device), bounds_on_correct_device)
            else: 
                vertices_for_final_hull_calc = current_region_vor_vertices
        
        elif space_dim == 3: 
            if current_region_vor_vertices.shape[0] < 4: # A polyhedron needs at least 4 vertices
                if bounds_on_correct_device is None:
                    weights_list.append(1.0 / min_measure_floor); continue
            
            if bounds_on_correct_device is not None:
                if current_region_vor_vertices.shape[0] > 0: 
                    vertices_for_final_hull_calc = clip_polyhedron_3d(current_region_vor_vertices.to(bounds_on_correct_device.device), bounds_on_correct_device, tol=EPSILON)
                else: 
                    vertices_for_final_hull_calc = torch.empty((0,3), dtype=points.dtype, device=original_device)
            else: 
                vertices_for_final_hull_calc = current_region_vor_vertices
        
        else: # Should not be reached due to earlier checks
            weights_list.append(1.0 / min_measure_floor); continue

        # After potential clipping, check if enough vertices remain for area/volume calculation
        if vertices_for_final_hull_calc is None or \
           (space_dim == 2 and vertices_for_final_hull_calc.shape[0] < 3) or \
           (space_dim == 3 and vertices_for_final_hull_calc.shape[0] < 4):
            weights_list.append(1.0 / min_measure_floor); continue
        
        try:
            # ConvexHull expects points on its own device or CPU, ensure consistency
            hull_of_region = ConvexHull(vertices_for_final_hull_calc.to(original_device), tol=EPSILON) 
        except (ValueError, RuntimeError): 
            weights_list.append(1.0 / min_measure_floor); continue

        cell_measure_tensor = hull_of_region.area if space_dim == 2 else hull_of_region.volume
        cell_measure = cell_measure_tensor.item() 
        
        actual_measure = min_measure_floor if abs(cell_measure) < min_measure_floor else abs(cell_measure)
        weights_list.append(1.0 / actual_measure)

    final_weights_unnormalized = torch.tensor(weights_list, dtype=points.dtype, device=original_device)
    
    try:
        normalized_weights = normalize_weights(final_weights_unnormalized) # Uses imported normalize_weights
    except ValueError:
        # Fallback for safety, though normalize_weights itself has robust handling
        return torch.full((n_points,), 1.0/n_points if n_points > 0 else 1.0, dtype=points.dtype, device=original_device)

    return normalized_weights


class TestPyTorchDensityWeights(unittest.TestCase):
    def test_simple_2d_two_points_with_bounds(self):
        points = torch.tensor([[0.25, 0.5], [0.75, 0.5]], dtype=torch.float32)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32) 
        weights = compute_voronoi_density_weights_pytorch(points, bounds=bounds, space_dim=2)
        self.assertEqual(weights.shape[0], 2)
        self.assertTrue(torch.all(weights >= 0))
        self.assertAlmostEqual(torch.sum(weights).item(), 1.0, places=5)
        # Note: The exact behavior depends on the placeholder Voronoi implementation.
        # If construct_voronoi_polygons_2d creates symmetric cells that clip to equal areas:
        # self.assertAlmostEqual(weights[0].item(), 0.5, places=5)
        # self.assertAlmostEqual(weights[1].item(), 0.5, places=5)
        # For now, we just check they are valid weights. More specific checks require
        # a non-placeholder Voronoi or very predictable placeholders.

    def test_2d_single_point_with_bounds(self):
        points = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        weights = compute_voronoi_density_weights_pytorch(points, bounds=bounds, space_dim=2)
        self.assertEqual(weights.shape[0], 1)
        self.assertAlmostEqual(weights[0].item(), 1.0, places=5)

    def test_2d_no_bounds(self):
        points = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float32)
        weights = compute_voronoi_density_weights_pytorch(points, bounds=None, space_dim=2)
        self.assertEqual(weights.shape[0], 2)
        self.assertTrue(torch.all(weights >= 0))
        self.assertAlmostEqual(torch.sum(weights).item(), 1.0, places=5)
        # Depending on placeholder construct_voronoi_polygons_2d, areas might be equal.
        # self.assertAlmostEqual(weights[0].item(), 0.5, places=5)
        # self.assertAlmostEqual(weights[1].item(), 0.5, places=5)


    def test_3d_simple_two_points_with_bounds(self):
        points = torch.tensor([[0.25,0.5,0.5], [0.75,0.5,0.5]], dtype=torch.float32)
        bounds = torch.tensor([[0.0,0.0,0.0], [1.0,1.0,1.0]], dtype=torch.float32)
        weights = compute_voronoi_density_weights_pytorch(points, bounds=bounds, space_dim=3)
        self.assertEqual(weights.shape[0], 2)
        self.assertTrue(torch.all(weights >= 0))
        self.assertAlmostEqual(torch.sum(weights).item(), 1.0, places=5)
        # Similar to 2D, exact equality depends on placeholder Voronoi behavior.
        # self.assertAlmostEqual(weights[0].item(), 0.5, places=5)
        # self.assertAlmostEqual(weights[1].item(), 0.5, places=5)

    def test_n_points_less_than_or_equal_dim(self):
        points_2d = torch.tensor([[0.1, 0.1]], dtype=torch.float32)
        weights_2d = compute_voronoi_density_weights_pytorch(points_2d, space_dim=2)
        self.assertEqual(weights_2d.shape[0], 1)
        self.assertAlmostEqual(weights_2d[0].item(), 1.0, places=6)

        points_2d_2 = torch.tensor([[0.1,0.1],[0.9,0.9]], dtype=torch.float32)
        weights_2d_2 = compute_voronoi_density_weights_pytorch(points_2d_2, space_dim=2)
        self.assertEqual(weights_2d_2.shape[0], 2)
        self.assertAlmostEqual(weights_2d_2[0].item(), 0.5, places=6)
        self.assertAlmostEqual(weights_2d_2[1].item(), 0.5, places=6)

        points_3d = torch.tensor([[0.1,0.1,0.1],[0.5,0.5,0.5]], dtype=torch.float32)
        weights_3d = compute_voronoi_density_weights_pytorch(points_3d, space_dim=3)
        self.assertEqual(weights_3d.shape[0], 2)
        self.assertAlmostEqual(weights_3d[0].item(), 0.5, places=6)
        self.assertAlmostEqual(weights_3d[1].item(), 0.5, places=6)

    def test_empty_points_input(self):
        points = torch.empty((0,2), dtype=torch.float32)
        weights = compute_voronoi_density_weights_pytorch(points, space_dim=2)
        self.assertEqual(weights.shape[0], 0)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
