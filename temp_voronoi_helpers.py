import torch
import unittest # Keep unittest import here for when tests are added to this file
from collections import defaultdict # Added for get_cell_neighbors

# --- compute_cell_centroid Function ---

def compute_cell_centroid(cell_vertices_list: list[torch.Tensor]) -> torch.Tensor | None:
    """
    Computes the centroid of a Voronoi cell given its vertices.

    Args:
        cell_vertices_list (list[torch.Tensor]): A list of PyTorch Tensors, 
                                                 where each Tensor represents a vertex 
                                                 of the cell (e.g., shape (Dim,)).
                                                 Assumes Dim is consistent.

    Returns:
        torch.Tensor | None: A PyTorch Tensor of shape (Dim,) representing the centroid.
                             Returns None if the cell_vertices_list is empty,
                             or if vertices are not in a supported dimension (2D/3D),
                             or if insufficient points for a meaningful centroid in context
                             (e.g. <1 for average, though usually <Dim for geometric shapes).
    """
    if not cell_vertices_list:
        return None

    # Stack vertices into a single tensor (N_vertices, Dim)
    try:
        # Ensure all tensors in the list are on the same device and have compatible types for stacking.
        # If list is not empty, use the first tensor's device and dtype as reference.
        ref_device = cell_vertices_list[0].device
        ref_dtype = cell_vertices_list[0].dtype
        
        processed_vertices = []
        for v in cell_vertices_list:
            if not isinstance(v, torch.Tensor): # Basic type check
                return None 
            processed_vertices.append(v.to(device=ref_device, dtype=ref_dtype))
        
        vertices_tensor = torch.stack(processed_vertices)
    except Exception: 
        # Could happen if list contains tensors of different shapes that cannot be stacked,
        # or other unexpected types after the basic check.
        return None 

    if vertices_tensor.ndim != 2:
        # Expected (N_vertices, Dim). If not, input is malformed.
        return None 
        
    num_vertices, dim = vertices_tensor.shape

    if num_vertices == 0: # Should have been caught by the initial list check
        return None

    if dim not in [2, 3]:
        # print(f"Warning: compute_cell_centroid currently supports 2D/3D, got {dim}D.")
        return None # Or handle as simple average if desired for other Dims

    centroid = torch.mean(vertices_tensor.float(), dim=0) # Ensure float for mean calculation
    
    return centroid.to(vertices_tensor.dtype) # Cast back to original (stacked) dtype

# --- get_cell_neighbors Function (Conceptual - 2D Focus) ---

def get_cell_neighbors(
    target_cell_index: int, 
    all_cells_vertices_list: list[list[torch.Tensor]],
    shared_vertices_threshold: int = 2
) -> list[int]:
    """
    Finds neighboring cells for a target cell in a list of Voronoi cells (2D focus).

    Two cells are considered neighbors if they share at least `shared_vertices_threshold`
    vertices. For typical 2D Voronoi diagrams, sharing 2 vertices means sharing an edge.

    Args:
        target_cell_index (int): The index of the cell in `all_cells_vertices_list`
                                 for which to find neighbors.
        all_cells_vertices_list (list[list[torch.Tensor]]): A list where each element
                                 is a list of PyTorch Tensors (vertices) defining a cell.
                                 Each vertex tensor is expected to be 1D (e.g., shape (2,) for 2D).
        shared_vertices_threshold (int): Minimum number of shared vertices for two cells
                                         to be considered neighbors. Defaults to 2 (for 2D edges).

    Returns:
        list[int]: A list of integer indices of the cells that are neighbors
                   to the `target_cell_index`.
    
    Notes:
        - This function primarily focuses on 2D Voronoi cell adjacency. For 3D,
          sharing an edge (2 vertices) is different from sharing a face (>=3 coplanar vertices).
          A robust 3D version would need `shared_vertices_threshold >= 3` and potentially
          coplanarity checks for shared faces.
        - Vertex comparison relies on converting tensor coordinates to tuples of Python floats
          for hashing and set operations. Floating point precision issues might affect
          robustness if vertices that should be identical are not exactly equal.
          A tolerance-based comparison would be more robust for real-world float data.
    """
    if not (0 <= target_cell_index < len(all_cells_vertices_list)):
        # print(f"Warning: target_cell_index {target_cell_index} is out of bounds.")
        return []

    target_cell_v_list = all_cells_vertices_list[target_cell_index]
    if not target_cell_v_list or not isinstance(target_cell_v_list[0], torch.Tensor): # Target cell has no vertices defined or malformed
        return []

    # Convert target cell's vertices to a set of hashable tuples for efficient lookup
    try:
        # Rounding to a few decimal places before tupling can help with float precision issues.
        # Example: round to 5 decimal places. Adjust as needed.
        target_cell_v_tuples = set()
        for v_tensor in target_cell_v_list:
            if not isinstance(v_tensor, torch.Tensor) or v_tensor.ndim == 0: # Ensure it's a 1D tensor at least
                 # print(f"Warning: Invalid vertex tensor found in target cell {target_cell_index}.")
                 continue # Or handle error more strictly
            target_cell_v_tuples.add(
                tuple(round(coord.item(), 5) for coord in v_tensor)
            )
    except Exception: # Handle cases where v_tensor might not be as expected
        # print("Warning: Could not process vertices for target cell.")
        return []
    if not target_cell_v_tuples: return []


    neighbor_indices = []
    for i in range(len(all_cells_vertices_list)):
        if i == target_cell_index:
            continue # Don't compare a cell to itself

        current_cell_v_list = all_cells_vertices_list[i]
        if not current_cell_v_list or not isinstance(current_cell_v_list[0], torch.Tensor): # Current cell has no vertices or malformed
            continue
        
        # Check if dimensions match based on the first vertex
        if target_cell_v_list[0].shape != current_cell_v_list[0].shape:
            # print(f"Warning: Dimension mismatch between target cell and cell {i}")
            continue

        try:
            current_cell_v_tuples = set()
            for v_tensor in current_cell_v_list:
                if not isinstance(v_tensor, torch.Tensor) or v_tensor.ndim == 0:
                    # print(f"Warning: Invalid vertex tensor found in cell {i}.")
                    raise ValueError("Invalid vertex found") # Cause this cell to be skipped by outer try-except
                current_cell_v_tuples.add(
                     tuple(round(coord.item(), 5) for coord in v_tensor)
                )
        except Exception:
            # print(f"Warning: Could not process vertices for cell {i}.")
            continue # Skip this cell if its vertices are problematic
        
        if not current_cell_v_tuples: continue

        # Find common vertices
        common_vertices = target_cell_v_tuples.intersection(current_cell_v_tuples)
        
        if len(common_vertices) >= shared_vertices_threshold:
            neighbor_indices.append(i)
            
    return neighbor_indices

class TestComputeCellCentroid(unittest.TestCase):
    def test_centroid_empty_list(self):
        cell_vertices = []
        centroid = compute_cell_centroid(cell_vertices)
        self.assertIsNone(centroid)

    def test_centroid_single_point_2d(self):
        p1 = torch.tensor([1.0, 2.0], dtype=torch.float32)
        cell_vertices = [p1]
        centroid = compute_cell_centroid(cell_vertices)
        self.assertIsNotNone(centroid)
        if centroid is not None:
            self.assertTrue(torch.allclose(centroid, p1))

    def test_centroid_single_point_3d(self):
        p1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        cell_vertices = [p1]
        centroid = compute_cell_centroid(cell_vertices)
        self.assertIsNotNone(centroid)
        if centroid is not None:
            self.assertTrue(torch.allclose(centroid, p1))

    def test_centroid_line_segment_2d(self):
        # Centroid of a line segment is its midpoint
        p1 = torch.tensor([0.0, 0.0], dtype=torch.float32)
        p2 = torch.tensor([2.0, 2.0], dtype=torch.float32)
        cell_vertices = [p1, p2]
        centroid = compute_cell_centroid(cell_vertices)
        self.assertIsNotNone(centroid)
        if centroid is not None:
            self.assertTrue(torch.allclose(centroid, torch.tensor([1.0, 1.0])))
    
    def test_centroid_triangle_2d(self):
        # Centroid of a triangle is the average of its vertices
        p1 = torch.tensor([0.0, 0.0], dtype=torch.float32)
        p2 = torch.tensor([3.0, 0.0], dtype=torch.float32)
        p3 = torch.tensor([0.0, 3.0], dtype=torch.float32)
        cell_vertices = [p1, p2, p3]
        centroid = compute_cell_centroid(cell_vertices)
        self.assertIsNotNone(centroid)
        if centroid is not None:
            # Expected: ( (0+3+0)/3, (0+0+3)/3 ) = (1.0, 1.0)
            self.assertTrue(torch.allclose(centroid, torch.tensor([1.0, 1.0])))

    def test_centroid_square_2d(self):
        # Centroid of a square is its geometric center
        p1 = torch.tensor([0.0, 0.0], dtype=torch.float32)
        p2 = torch.tensor([2.0, 0.0], dtype=torch.float32)
        p3 = torch.tensor([2.0, 2.0], dtype=torch.float32)
        p4 = torch.tensor([0.0, 2.0], dtype=torch.float32)
        cell_vertices = [p1, p2, p3, p4] # Order doesn't matter for simple average
        centroid = compute_cell_centroid(cell_vertices)
        self.assertIsNotNone(centroid)
        if centroid is not None:
            # Expected: ( (0+2+2+0)/4, (0+0+2+2)/4 ) = (1.0, 1.0)
            self.assertTrue(torch.allclose(centroid, torch.tensor([1.0, 1.0])))

    def test_centroid_cube_3d(self):
        # Centroid of a cube (average of vertices) is its geometric center
        cell_vertices = [
            torch.tensor([0.,0.,0.]), torch.tensor([1.,0.,0.]), torch.tensor([0.,1.,0.]), torch.tensor([0.,0.,1.]),
            torch.tensor([1.,1.,0.]), torch.tensor([1.,0.,1.]), torch.tensor([0.,1.,1.]), torch.tensor([1.,1.,1.])
        ]
        # Convert to float32 for consistency if not already
        cell_vertices = [v.float() for v in cell_vertices]

        centroid = compute_cell_centroid(cell_vertices)
        self.assertIsNotNone(centroid)
        if centroid is not None:
            # Expected: (0.5, 0.5, 0.5)
            self.assertTrue(torch.allclose(centroid, torch.tensor([0.5, 0.5, 0.5])))
    
    def test_centroid_malformed_input_mixed_dims(self):
        # This should be caught by torch.stack if not perfectly handled, or return None from validation
        p1 = torch.tensor([0.0, 0.0])
        p2 = torch.tensor([1.0, 0.0, 0.0]) # 3D point
        cell_vertices = [p1, p2]
        centroid = compute_cell_centroid(cell_vertices)
        # The current compute_cell_centroid tries to stack, which would fail due to device/dtype processing of first element.
        # Or if that was aligned, stack would fail.
        self.assertIsNone(centroid)

    def test_centroid_unsupported_dim(self):
        p1 = torch.tensor([1.,2.,3.,4.]) # 4D point
        cell_vertices = [p1]
        centroid = compute_cell_centroid(cell_vertices)
        self.assertIsNone(centroid) # Function returns None for dims not 2 or 3

class TestGetCellNeighbors(unittest.TestCase):
    def _create_square_cell(self, x_offset, y_offset, size=1.0, dtype=torch.float32, device='cpu'):
        # Helper to create a list of vertex tensors for a square cell
        return [
            torch.tensor([x_offset, y_offset], dtype=dtype, device=device),
            torch.tensor([x_offset + size, y_offset], dtype=dtype, device=device),
            torch.tensor([x_offset + size, y_offset + size], dtype=dtype, device=device),
            torch.tensor([x_offset, y_offset + size], dtype=dtype, device=device)
        ]

    def test_simple_grid_neighbors(self):
        # Define a 3x1 grid of cells: C0 | C1 | C2
        # C0: (0,0)-(1,1)
        # C1: (1,0)-(2,1) (shares edge with C0 and C2)
        # C2: (2,0)-(3,1)
        all_cells = [
            self._create_square_cell(0,0), # Cell 0
            self._create_square_cell(1,0), # Cell 1
            self._create_square_cell(2,0)  # Cell 2
        ]
        # Expected neighbors (default threshold is 2 shared vertices):
        # Cell 0 neighbors: [1]
        # Cell 1 neighbors: [0, 2]
        # Cell 2 neighbors: [1]
        
        neighbors_c0 = get_cell_neighbors(0, all_cells)
        self.assertEqual(sorted(neighbors_c0), [1])
        
        neighbors_c1 = get_cell_neighbors(1, all_cells)
        self.assertEqual(sorted(neighbors_c1), [0, 2])
        
        neighbors_c2 = get_cell_neighbors(2, all_cells)
        self.assertEqual(sorted(neighbors_c2), [1])

    def test_no_neighbors(self):
        all_cells = [
            self._create_square_cell(0,0),   # Cell 0
            self._create_square_cell(10,10) # Cell 1 (far away)
        ]
        neighbors_c0 = get_cell_neighbors(0, all_cells)
        self.assertEqual(neighbors_c0, [])

    def test_target_index_out_of_bounds(self):
        all_cells = [self._create_square_cell(0,0)]
        neighbors = get_cell_neighbors(1, all_cells) # Index 1 is out of bounds
        self.assertEqual(neighbors, [])
        neighbors_neg = get_cell_neighbors(-1, all_cells) # Negative index
        self.assertEqual(neighbors_neg, [])
        
    def test_empty_cell_list(self):
        neighbors = get_cell_neighbors(0, [])
        self.assertEqual(neighbors, [])

    def test_cell_with_no_vertices(self):
        all_cells = [
            self._create_square_cell(0,0), # Cell 0
            []                            # Cell 1 (empty)
        ]
        neighbors_c0 = get_cell_neighbors(0, all_cells)
        self.assertEqual(neighbors_c0, []) # Cell 1 has no vertices to share

        neighbors_c1 = get_cell_neighbors(1, all_cells)
        self.assertEqual(neighbors_c1, []) # Target cell has no vertices

    def test_shared_vertex_threshold_1(self):
        # C0: (0,0)-(1,1)
        # C1: (1,1)-(2,2) (shares only one vertex (1,1) with C0)
        all_cells = [
            self._create_square_cell(0,0, size=1.0), # Cell 0: vertices (0,0),(1,0),(1,1),(0,1)
            self._create_square_cell(1,1, size=1.0)  # Cell 1: vertices (1,1),(2,1),(2,2),(1,2)
        ]
        # With threshold 2, no neighbors
        neighbors_thresh2 = get_cell_neighbors(0, all_cells, shared_vertices_threshold=2)
        self.assertEqual(neighbors_thresh2, [])
        
        # With threshold 1, they are neighbors
        neighbors_thresh1 = get_cell_neighbors(0, all_cells, shared_vertices_threshold=1)
        self.assertEqual(sorted(neighbors_thresh1), [1])
        
    def test_floating_point_precision_with_rounding(self):
        # get_cell_neighbors uses rounding to 5 decimal places for vertex comparison
        v1 = [torch.tensor([0.0, 0.0]), torch.tensor([1.000001, 0.0]), torch.tensor([1.0, 1.0]), torch.tensor([0.0, 1.0])]
        v2 = [torch.tensor([1.000004, 0.0]), torch.tensor([2.0, 0.0]), torch.tensor([2.0, 1.0]), torch.tensor([1.0, 1.0])]
        # Vertices (1.000001, 0.0) and (1.000004, 0.0) both round to (1.00000, 0.0)
        # Vertex (1.0, 1.0) is common. So, they should share two vertices after rounding.
        
        all_cells = [v1, v2]
        neighbors = get_cell_neighbors(0, all_cells, shared_vertices_threshold=2)
        self.assertEqual(sorted(neighbors), [1])

        # Test with values that won't match after rounding
        v3 = [torch.tensor([0.0, 0.0]), torch.tensor([1.0001, 0.0]), torch.tensor([1.0, 1.0]), torch.tensor([0.0, 1.0])]
        v4 = [torch.tensor([1.0002, 0.0]), torch.tensor([2.0, 0.0]), torch.tensor([2.0, 1.0]), torch.tensor([1.0, 1.0])]
        all_cells_no_match = [v3,v4]
        neighbors_no_match = get_cell_neighbors(0, all_cells_no_match, shared_vertices_threshold=2)
        self.assertEqual(neighbors_no_match, [])

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
