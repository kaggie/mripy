import torch
import unittest
from collections import defaultdict
from .circumcenter_calculations import compute_triangle_circumcenter_2d, compute_tetrahedron_circumcenter_3d

# EPSILON_GEOMETRY was used by the placeholder circumcenter functions.
# The imported functions from circumcenter_calculations use their own EPSILON_GEOMETRY.
# If functions in *this* file directly need an epsilon, it should be defined or imported.
# For now, let's assume EPSILON_GEOMETRY is not directly needed by construct_voronoi_polygons_2d itself,
# but rather by the circumcenter functions it calls (which are now imported).
# If any logic *within this file directly* needs an epsilon, it should be explicitly defined or imported.
# Let's define it here if it was used by the main functions in the temp file,
# otherwise, we can remove it if only the placeholders used it.
# Reviewing the temp file: EPSILON_GEOMETRY was used by the *placeholder* circumcenter functions.
# The main construct_voronoi_polygons_2d and construct_voronoi_polyhedra_3d do not use it directly.
# So, we might not need it here if the imported functions are self-contained with their epsilons.
# However, the original temp file *did* define it at the top level.
# For safety and consistency with the temp file's structure, I'll keep it defined.
EPSILON_GEOMETRY = 1e-7


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

    voronoi_vertex_coords_map = {} 
    temp_voronoi_vertices_list = []
    for i, tri_indices in enumerate(delaunay_triangles):
        p1, p2, p3 = points[tri_indices[0]], points[tri_indices[1]], points[tri_indices[2]]
        circumcenter = compute_triangle_circumcenter_2d(p1, p2, p3)
        if circumcenter is not None:
            voronoi_vertex_coords_map[i] = circumcenter
            temp_voronoi_vertices_list.append(circumcenter)
    
    if not temp_voronoi_vertices_list: 
         return [[] for _ in range(points.shape[0])], torch.empty((0,2), dtype=points.dtype, device=points.device)

    if temp_voronoi_vertices_list:
        unique_voronoi_vertices_tensor = torch.stack(temp_voronoi_vertices_list)
        unique_voronoi_vertices = unique_voronoi_vertices_tensor 
        
        tri_idx_to_voronoi_v_idx = {}
        current_v_idx = 0
        for i, tri_indices in enumerate(delaunay_triangles):
            if i in voronoi_vertex_coords_map: 
                tri_idx_to_voronoi_v_idx[i] = current_v_idx 
                current_v_idx +=1
            
    else:
        unique_voronoi_vertices = torch.empty((0,2), dtype=points.dtype, device=points.device)
        tri_idx_to_voronoi_v_idx = {}

    point_to_triangles_map = defaultdict(list)
    for tri_idx, tri_indices in enumerate(delaunay_triangles):
        if tri_idx not in tri_idx_to_voronoi_v_idx: continue 
        for pt_idx in tri_indices:
            point_to_triangles_map[pt_idx.item()].append(tri_idx)

    voronoi_cells_vertices_list = [[] for _ in range(points.shape[0])]

    for pt_idx, incident_tri_indices in point_to_triangles_map.items():
        if not incident_tri_indices:
            continue
        
        cell_voronoi_v_indices = [tri_idx_to_voronoi_v_idx[tri_idx] for tri_idx in incident_tri_indices if tri_idx in tri_idx_to_voronoi_v_idx]
        
        if not cell_voronoi_v_indices: continue
        
        cell_v_coords = unique_voronoi_vertices[torch.tensor(cell_voronoi_v_indices, device=points.device, dtype=torch.long)] # Ensure long for indexing
        
        if cell_v_coords.shape[0] < 3: 
            voronoi_cells_vertices_list[pt_idx] = [cv for cv in cell_v_coords] 
            continue

        centroid = torch.mean(cell_v_coords, dim=0)
        angles = torch.atan2(cell_v_coords[:,1] - centroid[1], cell_v_coords[:,0] - centroid[0])
        sorted_indices = torch.argsort(angles)
        
        ordered_cell_v_coords = cell_v_coords[sorted_indices]
        voronoi_cells_vertices_list[pt_idx] = [v_coord for v_coord in ordered_cell_v_coords] 

    return voronoi_cells_vertices_list, unique_voronoi_vertices
    
def construct_voronoi_polyhedra_3d(points: torch.Tensor, delaunay_tetrahedra: torch.Tensor):
    if points.shape[0] == 0 or delaunay_tetrahedra.shape[0] == 0:
        return [], torch.empty((0, 3), dtype=points.dtype, device=points.device)

    temp_voronoi_vertices_list = []
    tet_idx_to_voronoi_v_idx_map = {} 
    
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
        
        cell_v_coords = unique_voronoi_vertices[torch.tensor(cell_voronoi_v_indices, device=points.device, dtype=torch.long)] # Ensure long for indexing
        voronoi_cells_polyhedra_list[pt_idx] = [v_coord for v_coord in cell_v_coords] 

    return voronoi_cells_polyhedra_list, unique_voronoi_vertices


# --- Unit Tests ---
class TestVoronoiFromDelaunay(unittest.TestCase):
    def test_construct_voronoi_2d_simple_square(self):
        points = torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.]], dtype=torch.float32)
        delaunay_tris = torch.tensor([[0,1,3],[1,2,3]], dtype=torch.long)
        cells, v_vertices = construct_voronoi_polygons_2d(points, delaunay_tris)
        
        # With correct circumcenter (0.5,0.5) for both triangles from these points
        self.assertEqual(v_vertices.shape[0], 1) 
        self.assertTrue(torch.allclose(v_vertices[0], torch.tensor([0.5,0.5])))
        self.assertEqual(len(cells), 4) 
        for i in range(4):
            self.assertEqual(len(cells[i]), 1)
            if cells[i]: 
                self.assertTrue(torch.allclose(cells[i][0], torch.tensor([0.5,0.5])))

    def test_construct_voronoi_2d_triangle_points(self):
        points = torch.tensor([[0.,0.],[2.,0.],[0.,2.]], dtype=torch.float32) 
        delaunay_tris = torch.tensor([[0,1,2]], dtype=torch.long) 
        cells, v_vertices = construct_voronoi_polygons_2d(points, delaunay_tris)
        
        self.assertEqual(v_vertices.shape[0], 1) 
        expected_circumcenter = torch.tensor([1.0, 1.0])
        self.assertTrue(torch.allclose(v_vertices[0], expected_circumcenter))
        self.assertEqual(len(cells), 3)
        for i in range(3):
            self.assertEqual(len(cells[i]), 1) 
            if cells[i]:
                self.assertTrue(torch.allclose(cells[i][0], expected_circumcenter))
    
    def test_construct_voronoi_3d_single_tetrahedron(self):
        points = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], dtype=torch.float32)
        delaunay_tets = torch.tensor([[0,1,2,3]], dtype=torch.long)
        cells_poly, v_vertices = construct_voronoi_polyhedra_3d(points, delaunay_tets)

        self.assertEqual(v_vertices.shape[0], 1) 
        expected_cc = torch.tensor([0.5,0.5,0.5])
        self.assertTrue(torch.allclose(v_vertices[0], expected_cc))
        self.assertEqual(len(cells_poly), 4)
        for i in range(4):
            self.assertEqual(len(cells_poly[i]), 1) 
            if cells_poly[i]:
                 self.assertTrue(torch.allclose(cells_poly[i][0], expected_cc))

# if __name__ == '__main__':
#    unittest.main(argv=['first-arg-is-ignored'], exit=False)
