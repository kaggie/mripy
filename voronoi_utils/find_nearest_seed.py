import torch
import unittest

# --- findNearestSeed Function ---

def find_nearest_seed(query_points: torch.Tensor, seeds: torch.Tensor, return_squared_distance: bool = True):
    """
    Finds the nearest seed for each query point.

    Args:
        query_points (torch.Tensor): A tensor of shape (Q, Dim) representing Q query points
                                     in Dim dimensions. Or (Dim,) for a single query point.
        seeds (torch.Tensor): A tensor of shape (S, Dim) representing S seed points
                              in Dim dimensions.
        return_squared_distance (bool): If True, returns squared Euclidean distances.
                                       If False, returns Euclidean distances.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - nearest_seed_indices (torch.Tensor): Shape (Q,). Integer indices into the `seeds` tensor.
            - distances (torch.Tensor): Shape (Q,). Squared Euclidean or Euclidean distances
                                       to the nearest seed.
    """
    if query_points.ndim == 1:
        query_points = query_points.unsqueeze(0) # Reshape (Dim,) to (1, Dim)

    if seeds.ndim == 1: # Should ideally be (S, Dim) but handle (Dim,) as a single seed
         seeds = seeds.unsqueeze(0)

    if query_points.shape[1] != seeds.shape[1]:
        raise ValueError(f"Query points dimension ({query_points.shape[1]}) "
                         f"must match seeds dimension ({seeds.shape[1]}).")
    
    if seeds.shape[0] == 0: # No seeds to compare against
        num_queries = query_points.shape[0]
        return (torch.full((num_queries,), -1, dtype=torch.long, device=query_points.device),
                torch.full((num_queries,), float('inf'), dtype=query_points.dtype, device=query_points.device))


    # Expand dimensions for broadcasting:
    # query_points: (Q, 1, Dim)
    # seeds:        (1, S, Dim)
    q_expanded = query_points.unsqueeze(1)
    s_expanded = seeds.unsqueeze(0)

    # Calculate squared differences along the dimension axis
    # (q_expanded - s_expanded) has shape (Q, S, Dim)
    # sum over Dim axis -> squared Euclidean distances
    squared_distances = torch.sum((q_expanded - s_expanded)**2, dim=2) # Shape: (Q, S)

    # Find the minimum squared distance and its index for each query point
    # min_sq_distances will have shape (Q,)
    # nearest_seed_indices will have shape (Q,)
    min_sq_distances, nearest_seed_indices = torch.min(squared_distances, dim=1)

    if return_squared_distance:
        return nearest_seed_indices, min_sq_distances
    else:
        return nearest_seed_indices, torch.sqrt(min_sq_distances)

# --- Unit Tests ---

class TestFindNearestSeed(unittest.TestCase):
    def test_find_nearest_2d_single_query(self):
        query = torch.tensor([0.0, 0.0], dtype=torch.float32)
        seeds_tensor = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [0.0, 0.5]], dtype=torch.float32)
        # Expected: seed 2 (0.0, 0.5) is nearest. Sq_Dist = 0.25
        
        indices, sq_dists = find_nearest_seed(query, seeds_tensor, return_squared_distance=True)
        
        self.assertEqual(indices.item(), 2)
        self.assertAlmostEqual(sq_dists.item(), 0.25, places=6)

        indices_euc, dists_euc = find_nearest_seed(query, seeds_tensor, return_squared_distance=False)
        self.assertEqual(indices_euc.item(), 2)
        self.assertAlmostEqual(dists_euc.item(), 0.5, places=6)


    def test_find_nearest_2d_multiple_queries(self):
        queries = torch.tensor([[0.0, 0.0], [1.1, 1.1]], dtype=torch.float32)
        seeds_tensor = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [0.0, 0.5]], dtype=torch.float32)
        # Query 1 (0,0): nearest is seed 2 (0,0.5), sq_dist 0.25
        # Query 2 (1.1,1.1): nearest is seed 0 (1,1), sq_dist (0.1^2 + 0.1^2) = 0.01 + 0.01 = 0.02
        
        indices, sq_dists = find_nearest_seed(queries, seeds_tensor, return_squared_distance=True)
        
        expected_indices = torch.tensor([2, 0], dtype=torch.long)
        expected_sq_dists = torch.tensor([0.25, 0.02], dtype=torch.float32)
        
        self.assertTrue(torch.equal(indices, expected_indices))
        self.assertTrue(torch.allclose(sq_dists, expected_sq_dists, atol=1e-6))

    def test_find_nearest_3d(self):
        query = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        seeds_tensor = torch.tensor([[1,1,1], [-1,-1,-1], [0,0,0.5]], dtype=torch.float32)
        # Expected: seed 2 (0,0,0.5) is nearest. Sq_Dist = 0.25
        indices, sq_dists = find_nearest_seed(query, seeds_tensor)
        self.assertEqual(indices.item(), 2)
        self.assertAlmostEqual(sq_dists.item(), 0.25, places=6)

    def test_find_nearest_exact_match(self):
        query = torch.tensor([1.0, 1.0], dtype=torch.float32)
        seeds_tensor = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32)
        # Expected: seed 0 (1,1) is nearest. Sq_Dist = 0
        indices, sq_dists = find_nearest_seed(query, seeds_tensor)
        self.assertEqual(indices.item(), 0)
        self.assertAlmostEqual(sq_dists.item(), 0.0, places=6)

    def test_find_nearest_single_seed(self):
        query = torch.tensor([0.0, 0.0], dtype=torch.float32)
        single_seed = torch.tensor([1.0, 2.0], dtype=torch.float32) # Shape (2,)
        # Expected: seed 0. Sq_Dist = 1^2 + 2^2 = 5
        indices, sq_dists = find_nearest_seed(query, single_seed)
        self.assertEqual(indices.item(), 0)
        self.assertAlmostEqual(sq_dists.item(), 5.0, places=6)

        # Test with single seed explicitly (1, Dim)
        single_seed_exp = single_seed.unsqueeze(0)
        indices_exp, sq_dists_exp = find_nearest_seed(query, single_seed_exp)
        self.assertEqual(indices_exp.item(), 0)
        self.assertAlmostEqual(sq_dists_exp.item(), 5.0, places=6)


    def test_find_nearest_no_seeds(self):
        query = torch.tensor([0.0, 0.0], dtype=torch.float32)
        no_seeds = torch.empty((0,2), dtype=torch.float32)
        indices, sq_dists = find_nearest_seed(query, no_seeds)
        self.assertEqual(indices.item(), -1) # Expect -1 for index
        self.assertEqual(sq_dists.item(), float('inf')) # Expect inf for distance

    def test_find_nearest_dim_mismatch(self):
        query_2d = torch.tensor([0.0, 0.0], dtype=torch.float32)
        seeds_3d = torch.tensor([[1.,1.,1.]], dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "Query points dimension .* must match seeds dimension"):
            find_nearest_seed(query_2d, seeds_3d)
            
# if __name__ == '__main__':
#    unittest.main(argv=['first-arg-is-ignored'], exit=False)
