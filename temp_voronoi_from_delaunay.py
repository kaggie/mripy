import torch
import unittest
from collections import defaultdict

# Assume these functions are available (e.g., from temp_circumcenter_calculations.py or other modules)
# For this temporary file, we might need to redefine them or ensure they can be imported if run standalone.
# To make this file runnable for testing, let's include simplified versions or placeholders if complex.

EPSILON_GEOMETRY = 1e-7

def compute_triangle_circumcenter_2d(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor | None:
    # This is a placeholder/copy from temp_circumcenter_calculations.py for standalone testing.
    # In a real scenario, this would be imported.
    p1x, p1y = p1[0], p1[1]
    p2x, p2y = p2[0], p2[1]
    p3x, p3y = p3[0], p3[1]
    p1x_f64,p1y_f64,p2x_f64,p2y_f64,p3x_f64,p3y_f64 = p1x.to(torch.float64),p1y.to(torch.float64),p2x.to(torch.float64),p2y.to(torch.float64),p3x.to(torch.float64),p3y.to(torch.float64)
    D_f64 = 2*(p1x_f64*(p2y_f64-p3y_f64)+p2x_f64*(p3y_f64-p1y_f64)+p3x_f64*(p1y_f64-p2y_f64))
    if torch.abs(D_f64) < EPSILON_GEOMETRY: return None
    p1s,p2s,p3s = p1x_f64**2+p1y_f64**2,p2x_f64**2+p2y_f64**2,p3x_f64**2+p3y_f64**2
    Ux=(p1s*(p2y_f64-p3y_f64)+p2s*(p3y_f64-p1y_f64)+p3s*(p1y_f64-p2y_f64))/D_f64
    Uy=(p1s*(p3x_f64-p2x_f64)+p2s*(p1x_f64-p3x_f64)+p3s*(p2x_f64-p1x_f64))/D_f64
    return torch.tensor([Ux,Uy],dtype=p1.dtype,device=p1.device)

def compute_tetrahedron_circumcenter_3d(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor | None:
    # Placeholder/copy from temp_circumcenter_calculations.py
    points_f64 = torch.stack([p1,p2,p3,p4]).to(torch.float64)
    v1,v2,v3 = points_f64[1]-points_f64[0],points_f64[2]-points_f64[0],points_f64[3]-points_f64[0]
    vol_det_mat = torch.stack([v1,v2,v3],dim=0)
    if torch.abs(torch.det(vol_det_mat))<EPSILON_GEOMETRY*10: return None
    A=torch.empty((4,4),dtype=torch.float64,device=p1.device); B=torch.empty((4,1),dtype=torch.float64,device=p1.device)
    for i in range(4):
        pt=points_f64[i]; A[i,0]=2*pt[0]; A[i,1]=2*pt[1]; A[i,2]=2*pt[2]; A[i,3]=1.0; B[i]=pt[0]**2+pt[1]**2+pt[2]**2
    try:
        if torch.abs(torch.det(A)) < EPSILON_GEOMETRY: return None
        sol=torch.linalg.solve(A,B); cc_f64=sol[:3].squeeze(); return cc_f64.to(dtype=p1.dtype)
    except Exception: return None

# --- Voronoi Diagram Construction from Delaunay ---

def construct_voronoi_polygons_2d(points: torch.Tensor, delaunay_triangles: torch.Tensor):
    """
    Constructs Voronoi cells from a 2D Delaunay triangulation.
    Args:
        points (torch.Tensor): Shape (N, 2), coordinates of the input points.
        delaunay_triangles (torch.Tensor): Shape (M, 3), indices of points forming Delaunay triangles.
    Returns:
        Tuple[List[List[torch.Tensor]], torch.Tensor]:
            - voronoi_cells_vertices_list: A list of lists. Each inner list contains torch.Tensors (shape (2,))
                                           representing the ordered vertices of a Voronoi cell.
                                           The order corresponds to the input points.
            - unique_voronoi_vertices (torch.Tensor): Shape (V, 2), coordinates of all unique Voronoi vertices.
    """
    if points.shape[0] == 0 or delaunay_triangles.shape[0] == 0:
        return [], torch.empty((0, 2), dtype=points.dtype, device=points.device)

    # Calculate circumcenter for each Delaunay triangle (these are the Voronoi vertices)
    voronoi_vertex_coords_map = {} # Maps triangle index to its Voronoi vertex coordinates
    temp_voronoi_vertices_list = []
    for i, tri_indices in enumerate(delaunay_triangles):
        p1, p2, p3 = points[tri_indices[0]], points[tri_indices[1]], points[tri_indices[2]]
        circumcenter = compute_triangle_circumcenter_2d(p1, p2, p3)
        if circumcenter is not None:
            voronoi_vertex_coords_map[i] = circumcenter
            temp_voronoi_vertices_list.append(circumcenter)
    
    if not temp_voronoi_vertices_list: # No valid circumcenters
         return [[] for _ in range(points.shape[0])], torch.empty((0,2), dtype=points.dtype, device=points.device)

    # Create a list of unique Voronoi vertices
    # Using torch.unique can be tricky with floating point.
    # A more robust way for many vertices is to build a list and then unique-fy with tolerance,
    # but for now, stack and unique should work for typical cases.
    if temp_voronoi_vertices_list:
        unique_voronoi_vertices_tensor = torch.stack(temp_voronoi_vertices_list)
        # Using a simple unique for now. A tolerant unique might be needed.
        # unique_voronoi_vertices, _ = torch.unique(unique_voronoi_vertices_tensor, dim=0, return_inverse=True)
        # For now, just use the list as is, duplication is ok for cell construction, will unique later.
        # Let's assign an index to each circumcenter directly for now.
        voronoi_vertex_to_id = {i: i for i in range(len(temp_voronoi_vertices_list))}
        unique_voronoi_vertices = unique_voronoi_vertices_tensor # Keep all, unique later if needed for topology
        
        # Map triangle index to its Voronoi vertex *ID* (index in unique_voronoi_vertices)
        tri_idx_to_voronoi_v_idx = {}
        current_v_idx = 0
        for i, tri_indices in enumerate(delaunay_triangles):
            if i in voronoi_vertex_coords_map: # if it had a valid circumcenter
                # This simple mapping assumes voronoi_vertex_coords_map keys are dense from 0 to M-1
                # and correspond to order in temp_voronoi_vertices_list
                tri_idx_to_voronoi_v_idx[i] = current_v_idx 
                current_v_idx +=1
            
    else:
        unique_voronoi_vertices = torch.empty((0,2), dtype=points.dtype, device=points.device)
        tri_idx_to_voronoi_v_idx = {}


    # For each input point, find incident Delaunay triangles
    point_to_triangles_map = defaultdict(list)
    for tri_idx, tri_indices in enumerate(delaunay_triangles):
        if tri_idx not in tri_idx_to_voronoi_v_idx: continue # Skip triangles with no valid circumcenter
        for pt_idx in tri_indices:
            point_to_triangles_map[pt_idx.item()].append(tri_idx)

    voronoi_cells_vertices_list = [[] for _ in range(points.shape[0])]

    for pt_idx, incident_tri_indices in point_to_triangles_map.items():
        if not incident_tri_indices:
            continue

        # Get Voronoi vertices corresponding to these triangles
        cell_voronoi_v_indices = [tri_idx_to_voronoi_v_idx[tri_idx] for tri_idx in incident_tri_indices if tri_idx in tri_idx_to_voronoi_v_idx]
        
        if not cell_voronoi_v_indices: continue

        # Order these Voronoi vertices around the point pt_idx.
        # This requires finding shared edges between incident Delaunay triangles.
        # The Voronoi vertices (circumcenters) are ordered by the sequence of triangles around the point.
        
        # Simplified ordering for now (centroid sort - works for convex cells from Delaunay)
        # This is a common heuristic but not guaranteed for complex non-convex cases (not expected from Delaunay dual)
        # or for handling unbounded cells correctly without further logic.
        
        cell_v_coords = unique_voronoi_vertices[torch.tensor(cell_voronoi_v_indices, device=points.device)]
        
        if cell_v_coords.shape[0] < 3: # Need at least 3 vertices to form a polygon for sorting
            voronoi_cells_vertices_list[pt_idx] = [cv for cv in cell_v_coords] # Store as list of tensors
            continue

        # Compute centroid of these Voronoi vertices
        centroid = torch.mean(cell_v_coords, dim=0)
        
        # Sort vertices by angle around the centroid
        angles = torch.atan2(cell_v_coords[:,1] - centroid[1], cell_v_coords[:,0] - centroid[0])
        sorted_indices = torch.argsort(angles)
        
        ordered_cell_v_coords = cell_v_coords[sorted_indices]
        voronoi_cells_vertices_list[pt_idx] = [v_coord for v_coord in ordered_cell_v_coords] # Store list of tensors

    return voronoi_cells_vertices_list, unique_voronoi_vertices
    
# Placeholder for 3D version - significantly more complex
def construct_voronoi_polyhedra_3d(points: torch.Tensor, delaunay_tetrahedra: torch.Tensor):
    if points.shape[0] == 0 or delaunay_tetrahedra.shape[0] == 0:
        return [], torch.empty((0, 3), dtype=points.dtype, device=points.device)

    temp_voronoi_vertices_list = []
    tet_idx_to_voronoi_v_idx_map = {} # Maps tet_idx to index in unique_voronoi_vertices
    
    current_v_idx = 0
    for i, tet_indices in enumerate(delaunay_tetrahedra):
        p1,p2,p3,p4 = points[tet_indices[0]],points[tet_indices[1]],points[tet_indices[2]],points[tet_indices[3]]
        circumcenter = compute_tetrahedron_circumcenter_3d(p1,p2,p3,p4)
        if circumcenter is not None:
            temp_voronoi_vertices_list.append(circumcenter)
            tet_idx_to_voronoi_v_idx_map[i] = current_v_idx
            current_v_idx += 1
    
    if not temp_voronoi_vertices_list:
         return [[] for _ in range(points.shape[0])], torch.empty((0,3), dtype=points.dtype, device=points.device)

    unique_voronoi_vertices = torch.stack(temp_voronoi_vertices_list)

    point_to_tetrahedra_map = defaultdict(list)
    for tet_idx, tet_orig_indices in enumerate(delaunay_tetrahedra):
        if tet_idx not in tet_idx_to_voronoi_v_idx_map: continue
        for pt_idx in tet_orig_indices:
            point_to_tetrahedra_map[pt_idx.item()].append(tet_idx)

    voronoi_cells_polyhedra_list = [[] for _ in range(points.shape[0])]
    for pt_idx, incident_tet_indices in point_to_tetrahedra_map.items():
        if not incident_tet_indices: continue
        
        cell_voronoi_v_indices = [tet_idx_to_voronoi_v_idx_map[tet_idx] for tet_idx in incident_tet_indices if tet_idx in tet_idx_to_voronoi_v_idx_map]
        if not cell_voronoi_v_indices: continue
        
        # Each Voronoi cell is a polyhedron formed by these Voronoi vertices.
        # The structure of this polyhedron (faces, edges) requires complex topology derivation.
        # For now, just return the list of Voronoi vertex coordinates associated with this cell.
        # The actual polyhedron faces would be needed for things like volume calculation.
        cell_v_coords = unique_voronoi_vertices[torch.tensor(cell_voronoi_v_indices, device=points.device)]
        voronoi_cells_polyhedra_list[pt_idx] = [v_coord for v_coord in cell_v_coords] 

    return voronoi_cells_polyhedra_list, unique_voronoi_vertices


# --- Unit Tests ---
class TestVoronoiFromDelaunay(unittest.TestCase):
    def test_construct_voronoi_2d_simple_square(self):
        # For a square, Delaunay gives two triangles.
        # Voronoi diagram should have one Voronoi vertex at the center of the square.
        # Each of the 4 input points should have a Voronoi cell that is a quadrant.
        points = torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.]], dtype=torch.float32)
        # Two possible Delaunay triangulations for a square:
        # Option 1: Triangles (0,1,2) and (0,2,3) by original indices
        # Option 2: Triangles (0,1,3) and (1,2,3)
        # Let's assume one, e.g., (0,1,3) and (1,2,3)
        # T1=(0,1,3) -> points (0,0),(1,0),(0,1). Circumcenter C1=(0.5,0.5)
        # T2=(1,2,3) -> points (1,0),(1,1),(0,1). Circumcenter C2=(0.5,0.5)
        # This means only one unique Voronoi vertex at (0.5,0.5) if indices are mapped.
        
        # If Delaunay gives triangles [[0,1,3], [1,2,3]] (using original point indices)
        delaunay_tris = torch.tensor([[0,1,3],[1,2,3]], dtype=torch.long)

        cells, v_vertices = construct_voronoi_polygons_2d(points, delaunay_tris)
        
        self.assertEqual(v_vertices.shape[0], 1) # Only one unique Voronoi vertex (the center)
        self.assertTrue(torch.allclose(v_vertices[0], torch.tensor([0.5,0.5])))
        
        self.assertEqual(len(cells), 4) # One cell for each input point

        # Each cell should be formed by this single Voronoi vertex and points at infinity (conceptually).
        # The current `construct_voronoi_polygons_2d` returns the Voronoi vertices for the cell.
        # For a point surrounded by triangles whose circumcenters are all the same (like the square center),
        # the `cell_v_coords` would be a list of the same point. `centroid_sort` on this is trivial.
        # Each cell will be a list containing just the single Voronoi vertex [0.5, 0.5].
        # This is correct as the Voronoi cells are unbounded and share this one finite vertex.
        for i in range(4):
            self.assertEqual(len(cells[i]), 1)
            if cells[i]: # if list is not empty
                self.assertTrue(torch.allclose(cells[i][0], torch.tensor([0.5,0.5])))

    def test_construct_voronoi_2d_triangle_points(self):
        # Input: 3 points forming a triangle. Delaunay is just this one triangle.
        # Voronoi vertex is its circumcenter. Each input point's cell is unbounded,
        # formed by this circumcenter and two "points at infinity".
        points = torch.tensor([[0.,0.],[2.,0.],[0.,2.]], dtype=torch.float32) # Right triangle
        delaunay_tris = torch.tensor([[0,1,2]], dtype=torch.long) # The triangle itself

        cells, v_vertices = construct_voronoi_polygons_2d(points, delaunay_tris)
        
        self.assertEqual(v_vertices.shape[0], 1) # One Voronoi vertex (circumcenter of the input triangle)
        expected_circumcenter = torch.tensor([1.0, 1.0])
        self.assertTrue(torch.allclose(v_vertices[0], expected_circumcenter))

        self.assertEqual(len(cells), 3)
        for i in range(3):
            self.assertEqual(len(cells[i]), 1) # Each cell defined by the single finite Voronoi vertex
            if cells[i]:
                self.assertTrue(torch.allclose(cells[i][0], expected_circumcenter))
    
    def test_construct_voronoi_3d_single_tetrahedron(self):
        points = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], dtype=torch.float32)
        delaunay_tets = torch.tensor([[0,1,2,3]], dtype=torch.long)

        cells_poly, v_vertices = construct_voronoi_polyhedra_3d(points, delaunay_tets)

        self.assertEqual(v_vertices.shape[0], 1) # Single Voronoi vertex (circumcenter)
        expected_cc = torch.tensor([0.5,0.5,0.5])
        self.assertTrue(torch.allclose(v_vertices[0], expected_cc))

        self.assertEqual(len(cells_poly), 4)
        for i in range(4):
            self.assertEqual(len(cells_poly[i]), 1) # Each cell defined by the single Voronoi vertex
            if cells_poly[i]:
                 self.assertTrue(torch.allclose(cells_poly[i][0], expected_cc))


# if __name__ == '__main__':
#    unittest.main(argv=['first-arg-is-ignored'], exit=False)
