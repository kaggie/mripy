import torch
import unittest

# Assume find_nearest_seed is available. For temporary standalone use, copy it here.
# In a final integrated version, this would be imported.
def find_nearest_seed(query_points: torch.Tensor, seeds: torch.Tensor, return_squared_distance: bool = True):
    if query_points.ndim == 1: query_points = query_points.unsqueeze(0)
    if seeds.ndim == 1: seeds = seeds.unsqueeze(0)
    if query_points.shape[1] != seeds.shape[1]:
        raise ValueError(f"Query dim ({query_points.shape[1]}) != seeds dim ({seeds.shape[1]}).")
    if seeds.shape[0] == 0:
        return (torch.full((query_points.shape[0],), -1, dtype=torch.long, device=query_points.device),
                torch.full((query_points.shape[0],), float('inf'), dtype=query_points.dtype, device=query_points.device))
    q_exp, s_exp = query_points.unsqueeze(1), seeds.unsqueeze(0)
    sq_dists = torch.sum((q_exp - s_exp)**2, dim=2)
    min_sq_dists, nearest_indices = torch.min(sq_dists, dim=1)
    return nearest_indices, min_sq_dists if return_squared_distance else torch.sqrt(min_sq_dists)

# --- ComputeVoronoiTessellation Function ---

def compute_voronoi_tessellation(
    shape: tuple[int, ...], 
    seeds: torch.Tensor, 
    voxel_size: tuple[float, ...] | torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Computes a rasterized Voronoi tessellation based on nearest seed assignment.

    Args:
        shape (tuple[int, ...]): Desired output shape of the grid (e.g., (H, W) for 2D, (D, H, W) for 3D).
        seeds (torch.Tensor): Seed points of shape (S, Dim), where S is the number of seeds
                              and Dim is the dimensionality (2 or 3).
        voxel_size (tuple[float, ...] | torch.Tensor): Size of a voxel in each dimension.
                                                       Must match Dim. e.g., (vs_x, vs_y) or (vs_z, vs_x, vs_y).
                                                       If tuple, order should match shape's last dimensions (x,y for 2D; z,y,x for 3D typical image order)
                                                       Conventionally, for shape (D,H,W), voxel_size might be (vz, vh, vw) or (vw,vh,vz)
                                                       Let's assume voxel_size corresponds to shape dimensions:
                                                       If shape is (H,W), voxel_size is (voxel_height, voxel_width)
                                                       If shape is (D,H,W), voxel_size is (voxel_depth, voxel_height, voxel_width)
        mask (torch.Tensor | None, optional): A boolean tensor of the same `shape`. 
                                              If provided, tessellation is only computed for `True` values.
                                              Defaults to None (tessellate all voxels).

    Returns:
        torch.Tensor: An integer tensor of the given `shape`. Each element contains the index
                      (0 to S-1) of the seed that the corresponding voxel is closest to.
                      Voxels outside the mask (if provided) or where no seeds are given
                      will have a value of -1.
    """
    num_dims = len(shape)
    if not (2 <= num_dims <= 3):
        raise ValueError(f"Shape must have 2 or 3 dimensions, got {num_dims}.")
    if seeds.ndim != 2 or seeds.shape[1] != num_dims:
        raise ValueError(f"Seeds must be a (S, {num_dims}) tensor, got {seeds.shape}.")
    
    if isinstance(voxel_size, tuple):
        voxel_size = torch.tensor(voxel_size, dtype=seeds.dtype, device=seeds.device)
    if voxel_size.shape[0] != num_dims:
        raise ValueError(f"voxel_size must have {num_dims} elements, got {voxel_size.shape[0]}.")

    output_tessellation = torch.full(shape, -1, dtype=torch.long, device=seeds.device)

    if seeds.shape[0] == 0: # No seeds, return empty tessellation
        return output_tessellation

    # Create grid of voxel center coordinates
    # The ranges for meshgrid should correspond to the number of voxels in each dimension
    grid_ranges = [torch.arange(s, device=seeds.device, dtype=seeds.dtype) for s in shape]
    
    # If shape is (H,W), grid_coords will be list of HxW tensors for Y, X indices
    # If shape is (D,H,W), grid_coords will be list of DxHxW tensors for Z, Y, X indices
    # Note: torch.meshgrid indexing default is 'ij' (matrix indexing)
    # For (H,W) -> shape_H_coords (H,W), shape_W_coords (H,W)
    # For (D,H,W) -> shape_D_coords (D,H,W), shape_H_coords (D,H,W), shape_W_coords (D,H,W)
    coord_grids_indices = torch.meshgrid(*grid_ranges, indexing='ij')

    # Stack to get (Dim, *shape) -> then flatten to (Dim, N_voxels) -> transpose to (N_voxels, Dim)
    voxel_indices_flat = torch.stack(coord_grids_indices, dim=0).view(num_dims, -1).transpose(0, 1)

    # Convert voxel indices to physical coordinates (centers of voxels)
    # (index + 0.5) * voxel_size
    # Voxel size needs to align with the dimensions from meshgrid.
    # If shape=(H,W), coord_grids_indices are (Y_indices, X_indices)
    # Voxel_size should be (voxel_size_H, voxel_size_W)
    # If shape=(D,H,W), coord_grids_indices are (Z_indices, Y_indices, X_indices)
    # Voxel_size should be (voxel_size_D, voxel_size_H, voxel_size_W)
    
    # Ensure voxel_size is broadcastable for element-wise multiplication with voxel_indices_flat
    # voxel_indices_flat is (N_voxels, Dim), voxel_size is (Dim)
    query_points_physical = (voxel_indices_flat.to(seeds.dtype) + 0.5) * voxel_size.view(1, num_dims)


    target_indices_flat = None
    if mask is not None:
        if mask.shape != shape:
            raise ValueError(f"Mask shape {mask.shape} must match output shape {shape}.")
        mask_flat = mask.flatten()
        if not torch.any(mask_flat): # All mask values are False
            return output_tessellation # No points to compute
        
        query_points_physical = query_points_physical[mask_flat]
        # Store the flat indices of True mask elements to place results back
        target_indices_flat = torch.where(mask_flat)[0]
    
    if query_points_physical.shape[0] == 0: # No query points after masking (or if shape was 0)
        return output_tessellation

    # Find nearest seed for all relevant voxel centers
    nearest_seed_indices, _ = find_nearest_seed(query_points_physical, seeds)

    # Populate the output tessellation tensor
    if mask is not None and target_indices_flat is not None:
        # Create a flat view of output_tessellation to assign masked values
        output_tessellation_flat_view = output_tessellation.flatten()
        output_tessellation_flat_view[target_indices_flat] = nearest_seed_indices
        output_tessellation = output_tessellation_flat_view.view(shape) # Reshape back
    else: # No mask, or mask was full
        output_tessellation = nearest_seed_indices.view(shape)
        
    return output_tessellation

# --- Unit Tests ---
class TestComputeVoronoiTessellation(unittest.TestCase):
    def test_tessellation_2d_simple(self):
        shape = (4, 4)
        seeds = torch.tensor([[1.0, 1.0], [3.0, 3.0]], dtype=torch.float32)
        voxel_size = torch.tensor([1.0, 1.0], dtype=torch.float32)
        
        tessellation = compute_voronoi_tessellation(shape, seeds, voxel_size)
        
        # Expected: Voxel (0,0) (center 0.5,0.5) is closer to seed 0 (1,1)
        # Voxel (3,3) (center 3.5,3.5) is closer to seed 1 (3,3)
        # Voxel (1,1) (center 1.5,1.5) closer to seed 0
        # Voxel (2,2) (center 2.5,2.5) closer to seed 1
        self.assertEqual(tessellation[0,0].item(), 0)
        self.assertEqual(tessellation[3,3].item(), 1)
        self.assertEqual(tessellation[1,1].item(), 0)
        self.assertEqual(tessellation[2,2].item(), 1)
        # Check a point on the boundary: (0,2) center (0.5, 2.5)
        # dist_sq to (1,1): (0.5-1)^2 + (2.5-1)^2 = (-0.5)^2 + (1.5)^2 = 0.25 + 2.25 = 2.5
        # dist_sq to (3,3): (0.5-3)^2 + (2.5-3)^2 = (-2.5)^2 + (-0.5)^2 = 6.25 + 0.25 = 6.5
        self.assertEqual(tessellation[0,2].item(), 0)


    def test_tessellation_3d_simple(self):
        shape = (2, 2, 2)
        seeds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
        # Voxel size such that physical coords are same as indices for simplicity of expectation
        voxel_size = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32) 
        
        tessellation = compute_voronoi_tessellation(shape, seeds, voxel_size)
        
        # Voxel (0,0,0) (center 0.5,0.5,0.5) is closer to seed 0 (0,0,0)
        # Voxel (1,1,1) (center 1.5,1.5,1.5) is closer to seed 1 (1,1,1)
        self.assertEqual(tessellation[0,0,0].item(), 0)
        self.assertEqual(tessellation[1,1,1].item(), 1)

    def test_tessellation_2d_with_mask(self):
        shape = (3, 3)
        seeds = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float32)
        voxel_size = torch.tensor([1.0, 1.0], dtype=torch.float32)
        mask = torch.tensor([
            [True,  True,  False],
            [True,  False, False],
            [False, False, True]
        ], dtype=torch.bool)
        
        tessellation = compute_voronoi_tessellation(shape, seeds, voxel_size, mask=mask)
        
        # Check masked areas
        # (0,0) center (0.5,0.5) -> seed 0
        # (0,1) center (0.5,1.5) -> seed 0 (dist_sq to s0: 0.25+2.25=2.5; dist_sq to s1: 2.25+0.25=2.5. Tie, can be 0 or 1 by min)
        # (1,0) center (1.5,0.5) -> seed 0
        # (2,2) center (2.5,2.5) -> seed 1
        self.assertEqual(tessellation[0,0].item(), 0)
        # self.assertEqual(tessellation[0,1].item(), 0) # Tie-breaking may vary. Test unmasked areas
        self.assertEqual(tessellation[1,0].item(), 0)
        self.assertEqual(tessellation[2,2].item(), 1)
        
        # Check unmasked areas (should remain -1)
        self.assertEqual(tessellation[0,2].item(), -1)
        self.assertEqual(tessellation[1,1].item(), -1)
        self.assertEqual(tessellation[1,2].item(), -1)
        self.assertEqual(tessellation[2,0].item(), -1)
        self.assertEqual(tessellation[2,1].item(), -1)

    def test_tessellation_no_seeds(self):
        shape = (2,2)
        seeds = torch.empty((0,2), dtype=torch.float32)
        voxel_size = torch.tensor([1.,1.])
        tessellation = compute_voronoi_tessellation(shape, seeds, voxel_size)
        self.assertTrue(torch.all(tessellation == -1))

    def test_tessellation_mask_all_false(self):
        shape = (2,2)
        seeds = torch.tensor([[0.,0.]], dtype=torch.float32)
        voxel_size = torch.tensor([1.,1.])
        mask = torch.zeros(shape, dtype=torch.bool)
        tessellation = compute_voronoi_tessellation(shape, seeds, voxel_size, mask=mask)
        self.assertTrue(torch.all(tessellation == -1))


# if __name__ == '__main__':
#    unittest.main(argv=['first-arg-is-ignored'], exit=False)
