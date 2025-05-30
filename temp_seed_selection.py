import torch
import unittest

# --- SelectVoronoiSeeds Function ---

def select_voronoi_seeds(
    quality_map: torch.Tensor, 
    voxel_size: tuple[float, ...] | torch.Tensor,
    quality_threshold: float, 
    min_seed_distance: float, 
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Selects seed points from a quality_map based on a quality threshold and minimum distance criterion.

    Args:
        quality_map (torch.Tensor): A tensor representing the quality at each point in a grid.
                                    Higher values are better. Shape (D1, D2, ..., Dk).
        voxel_size (tuple[float,...] | torch.Tensor): Physical size of a voxel/pixel in each dimension.
                                                      Order should match quality_map dimensions.
                                                      e.g., for (H,W) quality_map, (vs_h, vs_w).
        quality_threshold (float): Minimum quality value for a point to be considered a seed candidate.
        min_seed_distance (float): Minimum physical distance between any two selected seeds.
        mask (torch.Tensor | None, optional): A boolean tensor of the same shape as `quality_map`.
                                              If provided, only points where mask is True are considered.
                                              Defaults to None (consider all points).

    Returns:
        torch.Tensor: A tensor of shape (N_selected, Dim) containing the physical coordinates
                      of the selected seed points. Dim is the number of dimensions of quality_map.
                      Returns an empty tensor (0, Dim) if no seeds are selected.
    """
    num_dims = quality_map.ndim
    if not (2 <= num_dims <= 3): # Currently supporting 2D and 3D
        raise ValueError(f"quality_map must have 2 or 3 dimensions, got {num_dims}.")

    if isinstance(voxel_size, tuple):
        if len(voxel_size) != num_dims:
            raise ValueError(f"voxel_size tuple length ({len(voxel_size)}) must match quality_map ndim ({num_dims}).")
        voxel_size_tensor = torch.tensor(voxel_size, dtype=quality_map.dtype, device=quality_map.device)
    elif isinstance(voxel_size, torch.Tensor):
        if voxel_size.shape[0] != num_dims:
            raise ValueError(f"voxel_size tensor length ({voxel_size.shape[0]}) must match quality_map ndim ({num_dims}).")
        voxel_size_tensor = voxel_size.to(dtype=quality_map.dtype, device=quality_map.device)
    else:
        raise TypeError("voxel_size must be a tuple or a torch.Tensor.")

    if mask is not None:
        if mask.shape != quality_map.shape:
            raise ValueError(f"Mask shape {mask.shape} must match quality_map shape {quality_map.shape}.")
        if mask.dtype != torch.bool:
            raise TypeError("Mask must be a boolean tensor.")
        
        # Apply mask: set quality to a very low value where mask is False
        # Using a very low value instead of filtering coordinates early simplifies index mapping.
        # Ensure quality_map is float for this operation if it's not already.
        processed_quality_map = torch.where(mask, quality_map.float(), torch.tensor(float('-inf'), dtype=torch.float32, device=quality_map.device))
    else:
        processed_quality_map = quality_map.float()

    # Identify candidate points above threshold
    candidate_indices_tuple = torch.where(processed_quality_map > quality_threshold)
    
    if not candidate_indices_tuple or len(candidate_indices_tuple[0]) == 0:
        return torch.empty((0, num_dims), dtype=quality_map.dtype, device=quality_map.device)

    candidate_qualities = processed_quality_map[candidate_indices_tuple]
    
    # Convert N-dim indices to a (N_candidates, num_dims) tensor
    candidate_indices_tensor = torch.stack(candidate_indices_tuple, dim=1)

    # Convert candidate voxel indices to physical coordinates (centers of voxels)
    candidate_physical_coords = (candidate_indices_tensor.to(voxel_size_tensor.dtype) + 0.5) * voxel_size_tensor.view(1, num_dims)

    # Store candidates as (quality, x, y, [z]) list of dicts or custom objects for easier filtering
    # For now, use tensors and filter with masks.
    
    # Sort candidates by quality in descending order
    sorted_quality_indices = torch.argsort(candidate_qualities, descending=True)
    
    # Reorder candidates based on sorted quality
    sorted_candidate_physical_coords = candidate_physical_coords[sorted_quality_indices]
    # sorted_candidate_qualities = candidate_qualities[sorted_quality_indices] # Not strictly needed after sorting

    selected_seeds_coords_list = []
    
    # Boolean mask to keep track of available candidates (initially all True)
    # This operates on the sorted list of candidates.
    is_candidate_available = torch.ones(sorted_candidate_physical_coords.shape[0], dtype=torch.bool, device=quality_map.device)

    for i in range(sorted_candidate_physical_coords.shape[0]):
        if is_candidate_available[i]:
            current_best_seed_coord = sorted_candidate_physical_coords[i]
            selected_seeds_coords_list.append(current_best_seed_coord)
            
            # Mark this candidate as unavailable (though it's already processed)
            is_candidate_available[i] = False 
            
            # Remove other candidates within min_seed_distance from current_best_seed_coord
            if i + 1 < sorted_candidate_physical_coords.shape[0]: # If there are remaining candidates
                # remaining_candidate_coords = sorted_candidate_physical_coords[is_candidate_available] # Filter available ones
                # No need to filter "remaining_candidate_coords" here, just iterate through all and check `is_candidate_available`
                if sorted_candidate_physical_coords[is_candidate_available].shape[0] > 0 : # check if any candidates are still available
                    min_seed_distance_sq = min_seed_distance**2
                    for j in range(i + 1, sorted_candidate_physical_coords.shape[0]): # Only need to check candidates after the current one
                        if is_candidate_available[j]: 
                            dist_sq = torch.sum((sorted_candidate_physical_coords[j] - current_best_seed_coord)**2)
                            if dist_sq < min_seed_distance_sq:
                                is_candidate_available[j] = False # Mark for removal
    
    if not selected_seeds_coords_list:
        return torch.empty((0, num_dims), dtype=quality_map.dtype, device=quality_map.device)
        
    return torch.stack(selected_seeds_coords_list)


# --- Unit Tests ---
class TestSelectVoronoiSeeds(unittest.TestCase):
    def test_simple_2d_selection(self):
        quality_map = torch.tensor([
            [0.1, 0.2, 0.9], # Seed 1 at (0,2) coord (phys: 0.5, 2.5 if vs=1)
            [0.8, 0.3, 0.1], # Seed 2 at (1,0) coord (phys: 1.5, 0.5 if vs=1)
            [0.1, 0.7, 0.1]  # Seed 3 at (2,1) coord (phys: 2.5, 1.5 if vs=1)
        ], dtype=torch.float32)
        voxel_size = (1.0, 1.0)
        quality_threshold = 0.6
        min_seed_distance = 1.1 # Ensures (0,2) and (1,0) can be selected, but (2,1) might be too close to (1,0) or (0,2)
                                # (0.5,2.5) to (1.5,0.5): dx=1, dy=2. d^2=1+4=5. d=sqrt(5)~2.23
                                # (0.5,2.5) to (2.5,1.5): dx=2, dy=1. d^2=4+1=5. d=sqrt(5)~2.23
                                # (1.5,0.5) to (2.5,1.5): dx=1, dy=1. d^2=1+1=2. d=sqrt(2)~1.41
        
        # Expected order of processing based on quality: (0,2) (0.9), (1,0) (0.8), (2,1) (0.7)
        # 1. Select (0,2) [phys (0.5,2.5)]. Candidates left: (1,0), (2,1)
        # 2. (1,0) [phys (1.5,0.5)] is >1.1 away from (0.5,2.5). Select (1.5,0.5).
        #    Candidates left: (2,1)
        # 3. (2,1) [phys (2.5,1.5)]
        #    Dist to (0.5,2.5) is ~2.23 (ok)
        #    Dist to (1.5,0.5) is ~1.41 (ok, as 1.41 > 1.1)
        #    So, all 3 should be selected.
        
        selected_seeds = select_voronoi_seeds(quality_map, voxel_size, quality_threshold, min_seed_distance)
        self.assertEqual(selected_seeds.shape[0], 3)
        
        # If min_seed_distance = 1.5
        # 1. Select (0,2) [phys (0.5,2.5)]
        # 2. Select (1,0) [phys (1.5,0.5)] (dist to (0.5,2.5) is ~2.23 > 1.5)
        # 3. (2,1) [phys (2.5,1.5)]:
        #    Dist to (1.5,0.5) is ~1.41 (< 1.5). So (2,1) should be removed.
        min_seed_distance_strict = 1.5
        selected_seeds_strict = select_voronoi_seeds(quality_map, voxel_size, quality_threshold, min_seed_distance_strict)
        self.assertEqual(selected_seeds_strict.shape[0], 2)
        # Check that the two selected are (0.5,2.5) and (1.5,0.5) or vice-versa (order might depend on tie-breaking if qualities were equal)
        # The current implementation sorts by quality, so (0.5,2.5) then (1.5,0.5)
        expected_coords_strict = torch.tensor([[0.5, 2.5], [1.5, 0.5]], dtype=torch.float32)
        # Convert to list of sorted tuples for comparison to handle order invariance robustly
        selected_list_strict = sorted([tuple(coord.tolist()) for coord in selected_seeds_strict])
        expected_list_strict = sorted([tuple(coord.tolist()) for coord in expected_coords_strict])
        self.assertEqual(selected_list_strict, expected_list_strict)


    def test_with_mask(self):
        quality_map = torch.tensor([
            [0.9, 0.2, 0.8], 
            [0.1, 0.7, 0.3],
            [0.6, 0.1, 0.5] 
        ], dtype=torch.float32)
        voxel_size = (1.0, 1.0)
        mask = torch.tensor([
            [True, False, True],
            [False, True, False],
            [True, False, False]
        ], dtype=torch.bool)
        quality_threshold = 0.55 # Valid candidates by quality: (0,0) (0.9), (0,2) (0.8), (1,1) (0.7), (2,0) (0.6)
                                 # After mask: (0,0) (0.9), (0,2) (0.8), (1,1) (0.7), (2,0) (0.6)
        min_seed_distance = 1.1 
        # Coords: (0,0)->(0.5,0.5); (0,2)->(0.5,2.5); (1,1)->(1.5,1.5); (2,0)->(2.5,0.5)

        # 1. Select (0,0) (0.9) at (0.5,0.5)
        # 2. (0,2) (0.8) at (0.5,2.5). Dist to (0.5,0.5) is 2.0. OK. Select.
        # 3. (1,1) (0.7) at (1.5,1.5). Dist to (0.5,0.5) is sqrt(1^2+1^2)=1.41. OK.
        #                               Dist to (0.5,2.5) is sqrt(1^2+(-1)^2)=1.41. OK. Select.
        # 4. (2,0) (0.6) at (2.5,0.5). Dist to (0.5,0.5) is 2.0. OK.
        #                               Dist to (0.5,2.5) is sqrt(2^2+(-2)^2)=sqrt(8)~2.8. OK.
        #                               Dist to (1.5,1.5) is sqrt(1^2+(-1)^2)=1.41. OK. Select.
        # All 4 should be selected.
        
        selected_seeds = select_voronoi_seeds(quality_map, voxel_size, quality_threshold, min_seed_distance, mask=mask)
        self.assertEqual(selected_seeds.shape[0], 4)

    def test_quality_threshold(self):
        quality_map = torch.tensor([[0.1, 0.5], [0.9, 0.3]], dtype=torch.float32)
        voxel_size = (1.0, 1.0)
        quality_threshold = 0.6 # Only (0.9) should be selected
        min_seed_distance = 0.1
        selected_seeds = select_voronoi_seeds(quality_map, voxel_size, quality_threshold, min_seed_distance)
        self.assertEqual(selected_seeds.shape[0], 1)
        self.assertTrue(torch.allclose(selected_seeds[0], torch.tensor([1.5, 0.5]))) # Coord of (0.9)

    def test_no_points_above_threshold(self):
        quality_map = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
        voxel_size = (1.0, 1.0)
        quality_threshold = 0.5
        min_seed_distance = 1.0
        selected_seeds = select_voronoi_seeds(quality_map, voxel_size, quality_threshold, min_seed_distance)
        self.assertEqual(selected_seeds.shape[0], 0)

    def test_all_points_too_close(self):
        quality_map = torch.tensor([
            [0.9, 0.8], # (0.5,0.5), (0.5,1.5)
            [0.7, 0.6]  # (1.5,0.5), (1.5,1.5)
        ], dtype=torch.float32)
        voxel_size = (1.0, 1.0)
        quality_threshold = 0.5
        min_seed_distance = 2.0 # Any two points are sqrt(1^2)=1 or sqrt(1^2+1^2)=1.41 away, so only one seed selected.
        
        # 1. Select (0,0) (0.9) at (0.5,0.5)
        # All other points:
        # (0,1) at (0.5,1.5): dist to (0.5,0.5) is 1.0 (< 2.0) -> removed
        # (1,0) at (1.5,0.5): dist to (0.5,0.5) is 1.0 (< 2.0) -> removed
        # (1,1) at (1.5,1.5): dist to (0.5,0.5) is 1.41 (< 2.0) -> removed
        selected_seeds = select_voronoi_seeds(quality_map, voxel_size, quality_threshold, min_seed_distance)
        self.assertEqual(selected_seeds.shape[0], 1)
        # The selected seed should be the one with the highest quality
        # which corresponds to quality_map[0,0] -> physical coords (0.5,0.5)
        self.assertTrue(torch.allclose(selected_seeds[0], torch.tensor([0.5,0.5])))


    def test_3d_selection(self):
        quality_map = torch.rand((3,3,3), dtype=torch.float32) # Random quality
        # Ensure at least one point is high quality
        quality_map[1,1,1] = 0.95 # phys (0.75,0.75,0.75) with vs=0.5
        quality_map[0,0,0] = 0.85 # phys (0.25,0.25,0.25) with vs=0.5
        voxel_size = (0.5, 0.5, 0.5)
        quality_threshold = 0.8
        min_seed_distance = 0.6 # phys distance
        # (1,1,1) is phys (0.75,0.75,0.75) (center of voxel 1,1,1 if vs=0.5, index 0)
        # (0,0,0) is phys (0.25,0.25,0.25)
        # Distance between them: sqrt(3 * (0.5)^2) = sqrt(0.75) ~ 0.866 > 0.6
        # So both should be selected.
        
        selected_seeds = select_voronoi_seeds(quality_map, voxel_size, quality_threshold, min_seed_distance)
        self.assertEqual(selected_seeds.shape[0], 2)
        
        # Check if (0.75,0.75,0.75) and (0.25,0.25,0.25) are in the selected seeds
        # Order depends on quality: 0.95 is selected first.
        expected_s1 = torch.tensor([0.75,0.75,0.75]) # Corresponds to quality 0.95
        expected_s2 = torch.tensor([0.25,0.25,0.25]) # Corresponds to quality 0.85
        
        # Convert to list of sorted tuples for comparison
        selected_list = sorted([tuple(coord.round(decimals=4).tolist()) for coord in selected_seeds])
        expected_list = sorted([tuple(coord.round(decimals=4).tolist()) for coord in [expected_s1, expected_s2]])
        self.assertEqual(selected_list, expected_list)

    def test_empty_quality_map(self):
        quality_map = torch.empty((0,0), dtype=torch.float32)
        voxel_size = (1.0, 1.0)
        selected_seeds = select_voronoi_seeds(quality_map, voxel_size, 0.5, 1.0)
        self.assertEqual(selected_seeds.shape[0], 0)
        # The number of dimensions is derived from quality_map.ndim, which would be 2
        self.assertEqual(selected_seeds.shape[1], 2) 

    def test_empty_quality_map_3d(self):
        quality_map = torch.empty((0,0,0), dtype=torch.float32)
        voxel_size = (1.0, 1.0, 1.0)
        selected_seeds = select_voronoi_seeds(quality_map, voxel_size, 0.5, 1.0)
        self.assertEqual(selected_seeds.shape[0], 0)
        self.assertEqual(selected_seeds.shape[1], 3)


if __name__ == '__main__':
    # This allows running the tests directly if the file is executed.
    # Not strictly necessary for the agent but good for local testing.
    # Note: The `select_voronoi_seeds` function must be defined above this in the file.
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
