import torch
import torch.nn.functional as F
import unittest

# --- Helper: Map physical seeds to voxel indices ---
def map_physical_coords_to_voxel_indices(
    physical_coords: torch.Tensor, 
    quality_map_shape: tuple[int, ...], 
    voxel_size: torch.Tensor
) -> torch.Tensor:
    """Converts physical coordinates to the nearest voxel indices."""
    # Voxel center for index (i,j,k) is ( (i+0.5)*vs_i, (j+0.5)*vs_j, (k+0.5)*vs_k )
    # So, index i = physical_coord_i / vs_i - 0.5
    indices_float = physical_coords / voxel_size.view(1, -1) - 0.5
    # Round to nearest integer index and clamp to be within grid bounds
    indices_long = torch.round(indices_float).long()
    
    for dim in range(indices_long.shape[1]):
        indices_long[:, dim] = torch.clamp(indices_long[:, dim], 0, quality_map_shape[dim] - 1)
    return indices_long

# --- Voronoi Region Growing Function ---
def voronoi_region_growing(
    initial_seeds_phys_coords: torch.Tensor, 
    quality_map: torch.Tensor, # Not used in this proximity-based version, but kept for signature
    voxel_size: tuple[float,...] | torch.Tensor,
    mask: torch.Tensor | None = None,
    max_iterations: int = 100,
    stop_threshold_fraction: float = 0.001
) -> torch.Tensor:
    """
    Performs Voronoi region growing based on proximity to initial seed physical locations.

    Args:
        initial_seeds_phys_coords (torch.Tensor): (S, Dim) tensor of physical seed coordinates.
        quality_map (torch.Tensor): (D1,...,Dk) tensor, defines shape and device. Not directly used for affinity in this version.
        voxel_size (tuple[float,...] | torch.Tensor): Physical size of a voxel.
        mask (torch.Tensor | None): Boolean tensor (D1,...,Dk). Growth confined to True areas.
        max_iterations (int): Max iterations for growth.
        stop_threshold_fraction (float): Stop if fraction of changed voxels is below this.

    Returns:
        torch.Tensor: Integer tensor (D1,...,Dk) with region IDs (0 to S-1) or -1.
    """
    num_seeds = initial_seeds_phys_coords.shape[0]
    num_dims = initial_seeds_phys_coords.shape[1]
    device = initial_seeds_phys_coords.device
    dtype = initial_seeds_phys_coords.dtype # For physical coordinate calculations

    if quality_map.ndim != num_dims:
        raise ValueError("quality_map dimensions must match seed dimensions.")

    if isinstance(voxel_size, tuple):
        voxel_size_tensor = torch.tensor(voxel_size, dtype=dtype, device=device)
    else:
        voxel_size_tensor = voxel_size.to(dtype=dtype, device=device)
    
    if voxel_size_tensor.shape[0] != num_dims:
        raise ValueError("voxel_size length must match seed dimensions.")

    segmentation_map = torch.full_like(quality_map, -1, dtype=torch.long, device=device)
    
    # Prepare mask: processable voxels are True
    if mask is not None:
        if mask.shape != quality_map.shape:
            raise ValueError("Mask shape must match quality_map shape.")
        if mask.dtype != torch.bool:
            raise TypeError("Mask must be boolean.")
        processable_mask = mask.clone()
    else:
        processable_mask = torch.ones_like(quality_map, dtype=torch.bool, device=device)
    
    num_total_processable_voxels = torch.sum(processable_mask).item()
    if num_total_processable_voxels == 0:
        return segmentation_map # No space to grow

    # 1. Initialization: Place initial seeds
    if num_seeds == 0:
        return segmentation_map # No seeds to grow from
        
    seed_voxel_indices = map_physical_coords_to_voxel_indices(
        initial_seeds_phys_coords, quality_map.shape, voxel_size_tensor
    )

    for seed_id in range(num_seeds):
        idx_tuple = tuple(seed_voxel_indices[seed_id].tolist())
        if processable_mask[idx_tuple]: # Only place seed if it's in a processable area
            segmentation_map[idx_tuple] = seed_id
        # else: seed is outside mask, it won't grow.

    # Define neighborhood kernel (e.g., 6-connectivity for 3D, 4-connectivity for 2D)
    # Using a simple N-dimensional max-pooling kernel approach to find neighbors
    # Create a kernel for dilation-like operation to find neighbors.
    # For 2D: [[0,1,0],[1,1,1],[0,1,0]] (cross shape + center) for 4-conn + self
    # For 3D: a 3x3x3 cube with center and 6 faces non-zero.
    if num_dims == 2:
        kernel = torch.tensor([[0,1,0],[1,1,1],[0,1,0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    elif num_dims == 3:
        kernel = torch.zeros((3,3,3), dtype=torch.float32, device=device)
        kernel[1,1,0] = 1; kernel[1,1,2] = 1 # x-axis
        kernel[1,0,1] = 1; kernel[1,2,1] = 1 # y-axis
        kernel[0,1,1] = 1; kernel[2,1,1] = 1 # z-axis
        kernel[1,1,1] = 1 # center
        kernel = kernel.unsqueeze(0).unsqueeze(0)
    else:
        raise NotImplementedError("Only 2D and 3D supported for now.")
    padding = kernel.shape[-1] // 2


    # Voxel centers grid for distance calculations
    grid_ranges = [torch.arange(s, device=device, dtype=dtype) for s in quality_map.shape]
    coord_grids_indices = torch.meshgrid(*grid_ranges, indexing='ij')
    voxel_indices_flat = torch.stack(coord_grids_indices, dim=0).view(num_dims, -1).transpose(0,1)
    voxel_centers_physical_flat = (voxel_indices_flat.to(dtype) + 0.5) * voxel_size_tensor.view(1, num_dims)
    # Reshape to match quality_map for easy indexing later if needed: (Dim, D1, D2, ...)
    # voxel_centers_physical_grid = voxel_centers_physical_flat.transpose(0,1).view(num_dims, *quality_map.shape)


    for iteration in range(max_iterations):
        changed_count_in_iter = 0
        
        # Create a map of current assigned regions (float for conv, +1 to handle -1)
        # Add 1 to map so -1 (unassigned) becomes 0, and seed_ids 0..S-1 become 1..S
        current_segmentation_plus_1 = (segmentation_map + 1).float().unsqueeze(0).unsqueeze(0)

        # Find potential candidates: unassigned voxels next to assigned ones
        # Dilate the assigned regions (where seg_map+1 > 0)
        # A voxel is part of the "active front" if it's -1 AND its dilated version is > 0
        # (meaning it's next to an assigned region).
        
        # Create a binary map of assigned areas (1 where assigned, 0 where -1)
        assigned_mask_binary = (segmentation_map != -1).float().unsqueeze(0).unsqueeze(0)
        dilated_assigned_mask = F.conv2d(assigned_mask_binary, kernel, padding=padding, stride=1) if num_dims == 2 else \
                                 F.conv3d(assigned_mask_binary, kernel, padding=padding, stride=1)
        dilated_assigned_mask = (dilated_assigned_mask > 0.5).squeeze(0).squeeze(0) # Back to boolean, original shape

        # Active front: currently unassigned (-1) AND processable AND next to an assigned region
        active_front_mask = (segmentation_map == -1) & processable_mask & dilated_assigned_mask
        
        if not torch.any(active_front_mask):
            break # No more voxels to grow into

        active_front_indices_tuple = torch.where(active_front_mask)
        if len(active_front_indices_tuple[0]) == 0: break

        # Get physical coords of active front voxels
        active_front_voxel_indices = torch.stack(active_front_indices_tuple, dim=1) # (N_front, Dim)
        active_front_phys_coords = (active_front_voxel_indices.to(dtype) + 0.5) * voxel_size_tensor.view(1, num_dims) # (N_front, Dim)

        # For each point in active_front, find its nearest *initial* seed
        # This implements the proximity-to-original-seed logic.
        # (N_front, S) distances
        dist_sq_to_seeds = torch.sum((active_front_phys_coords.unsqueeze(1) - initial_seeds_phys_coords.unsqueeze(0))**2, dim=2)
        
        # Find best seed and its ID for each active front voxel
        min_dist_sq, best_seed_ids_for_front = torch.min(dist_sq_to_seeds, dim=1) # (N_front,)

        # Update segmentation_map for these front voxels
        # Need to be careful: only update if the new assignment is valid (e.g. not overwriting a different seed from a parallel computation)
        # In this single-threaded CPU version, direct assignment is fine.
        newly_assigned_count_this_pass = 0
        for i in range(active_front_voxel_indices.shape[0]):
            voxel_idx_tuple = tuple(active_front_voxel_indices[i].tolist())
            # Double check it's still -1 (it should be due to active_front_mask)
            if segmentation_map[voxel_idx_tuple] == -1:
                segmentation_map[voxel_idx_tuple] = best_seed_ids_for_front[i]
                newly_assigned_count_this_pass += 1
        
        changed_count_in_iter = newly_assigned_count_this_pass

        if changed_count_in_iter == 0 and torch.any(active_front_mask): # No change but front exists (e.g. stuck)
             break # Or some other logic for handling stuck states
        if changed_count_in_iter < stop_threshold_fraction * num_total_processable_voxels:
            break
    
    return segmentation_map

class TestVoronoiRegionGrowing(unittest.TestCase):
    def test_simple_2d_growth_two_seeds(self):
        quality_map_shape_defining = torch.zeros((5, 5), dtype=torch.float32) # Quality not used by current logic
        voxel_size = (1.0, 1.0)
        # Seed 0 at (0.5,0.5) (voxel 0,0), Seed 1 at (4.5,4.5) (voxel 4,4)
        initial_seeds = torch.tensor([[0.5, 0.5], [4.5, 4.5]], dtype=torch.float32) 
        
        seg_map = voronoi_region_growing(initial_seeds, quality_map_shape_defining, voxel_size, max_iterations=10)

        # Check if corners belong to correct seeds
        self.assertEqual(seg_map[0,0].item(), 0)
        self.assertEqual(seg_map[4,4].item(), 1)
        
        # Check some points along the diagonal, expecting a split
        # Voxel (1,1) (center 1.5,1.5) should be seed 0
        # Voxel (3,3) (center 3.5,3.5) should be seed 1
        self.assertEqual(seg_map[1,1].item(), 0)
        self.assertEqual(seg_map[3,3].item(), 1)
        
        # Voxel (2,2) (center 2.5,2.5) is equidistant, behavior might depend on torch.min tie-breaking
        # but given iterative growth, it might be claimed by whoever gets there first or its neighbors.
        # For now, just check that it's assigned to one of them.
        self.assertIn(seg_map[2,2].item(), [0, 1])
        
        # Ensure all are assigned (no -1s)
        self.assertFalse(torch.any(seg_map == -1))

    def test_2d_growth_with_mask(self):
        quality_map_shape_defining = torch.zeros((5, 5), dtype=torch.float32)
        voxel_size = (1.0, 1.0)
        initial_seeds = torch.tensor([[0.5, 0.5]], dtype=torch.float32) # Seed 0 at (0,0)
        
        mask = torch.zeros((5,5), dtype=torch.bool)
        mask[0:3, 0:3] = True # Allow growth only in top-left 3x3 area
        
        seg_map = voronoi_region_growing(initial_seeds, quality_map_shape_defining, voxel_size, mask=mask, max_iterations=10)
        
        # Check that all cells within the 3x3 True mask area are assigned to seed 0
        for r in range(3):
            for c in range(3):
                self.assertEqual(seg_map[r,c].item(), 0, f"Cell ({r},{c}) not assigned to seed 0")
        
        # Check that cells outside the 3x3 True mask area remain -1
        for r in range(5):
            for c in range(5):
                if not (r < 3 and c < 3):
                    self.assertEqual(seg_map[r,c].item(), -1, f"Cell ({r},{c}) was assigned but is outside mask")

    def test_max_iterations_stop(self):
        quality_map_shape_defining = torch.zeros((10, 10), dtype=torch.float32)
        voxel_size = (1.0, 1.0)
        initial_seeds = torch.tensor([[0.5,0.5]], dtype=torch.float32) # Single seed
        
        # With very few iterations, not all cells should be filled
        seg_map_few_iter = voronoi_region_growing(initial_seeds, quality_map_shape_defining, voxel_size, max_iterations=2)
        self.assertTrue(torch.any(seg_map_few_iter == -1)) # Some should be unassigned
        self.assertTrue(torch.any(seg_map_few_iter == 0))  # Some should be assigned
        
        # With enough iterations, all should be filled
        seg_map_many_iter = voronoi_region_growing(initial_seeds, quality_map_shape_defining, voxel_size, max_iterations=20)
        self.assertFalse(torch.any(seg_map_many_iter == -1))


    def test_stop_threshold_fraction(self):
        quality_map_shape_defining = torch.zeros((20, 20), dtype=torch.float32) # Larger map
        voxel_size = (1.0, 1.0)
        initial_seeds = torch.tensor([[0.5,0.5]], dtype=torch.float32)
        
        # This should stop very early as many cells change initially
        seg_map_high_thresh = voronoi_region_growing(initial_seeds, quality_map_shape_defining, voxel_size, 
                                                     max_iterations=50, stop_threshold_fraction=0.5) 
        self.assertTrue(torch.any(seg_map_high_thresh == -1)) # Likely stopped before filling all

        # This should run longer / to completion
        seg_map_low_thresh = voronoi_region_growing(initial_seeds, quality_map_shape_defining, voxel_size, 
                                                    max_iterations=50, stop_threshold_fraction=0.0001)
        self.assertFalse(torch.any(seg_map_low_thresh == -1))


    def test_simple_3d_growth(self):
        quality_map_shape_defining = torch.zeros((3,3,3), dtype=torch.float32)
        voxel_size = (1.0,1.0,1.0)
        initial_seeds = torch.tensor([[0.5,0.5,0.5]], dtype=torch.float32) # Seed at center of voxel (0,0,0)
        
        seg_map = voronoi_region_growing(initial_seeds, quality_map_shape_defining, voxel_size, max_iterations=5)
        # All cells should be assigned to seed 0
        self.assertFalse(torch.any(seg_map == -1))
        self.assertTrue(torch.all(seg_map == 0))

    def test_no_seeds(self):
        quality_map_shape_defining = torch.zeros((3,3), dtype=torch.float32)
        voxel_size = (1.0,1.0)
        initial_seeds = torch.empty((0,2), dtype=torch.float32)
        seg_map = voronoi_region_growing(initial_seeds, quality_map_shape_defining, voxel_size)
        self.assertTrue(torch.all(seg_map == -1)) # Should remain all unassigned

    def test_mask_all_false(self):
        quality_map_shape_defining = torch.zeros((3,3), dtype=torch.float32)
        voxel_size = (1.0,1.0)
        initial_seeds = torch.tensor([[1.5,1.5]], dtype=torch.float32)
        mask = torch.zeros((3,3), dtype=torch.bool)
        seg_map = voronoi_region_growing(initial_seeds, quality_map_shape_defining, voxel_size, mask=mask)
        self.assertTrue(torch.all(seg_map == -1)) # Should remain all unassigned

    def test_seed_outside_mask(self):
        quality_map_shape_defining = torch.zeros((3,3), dtype=torch.float32)
        voxel_size = (1.0,1.0)
        initial_seeds = torch.tensor([[5.0,5.0]], dtype=torch.float32) # Seed well outside 3x3 grid
        mask = torch.ones((3,3), dtype=torch.bool) # Mask allows all in grid
        
        seg_map = voronoi_region_growing(initial_seeds, quality_map_shape_defining, voxel_size, mask=mask)
        # Seed is mapped to nearest voxel (2,2). So (2,2) should be 0, others grow from there.
        # This test actually verifies that map_physical_coords_to_voxel_indices clamps correctly
        # and growth proceeds from the clamped seed location.
        self.assertFalse(torch.any(seg_map == -1)) # All should be assigned if seed is clamped into processable area
        self.assertEqual(seg_map[2,2].item(), 0) # Clamped seed location

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
