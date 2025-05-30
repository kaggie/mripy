import torch
import numpy as np # Only for type hints in future scipy.spatial.Voronoi replacement

EPSILON = 1e-9

def monotone_chain_2d(points: torch.Tensor, tol: float = 1e-6):
    """
    Computes the convex hull of 2D points using the Monotone Chain algorithm.
    Args:
        points (torch.Tensor): Tensor of shape (N, 2) representing N points in 2D.
        tol (float): Tolerance for floating point comparisons.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - hull_vertices_indices (torch.Tensor): Indices of points forming the convex hull, ordered.
            - hull_simplices (torch.Tensor): Pairs of indices forming the hull edges.
    """
    if not isinstance(points, torch.Tensor):
        raise ValueError("Input points must be a PyTorch tensor.")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Input points tensor must be 2-dimensional with shape (N, 2).")
    
    if points.shape[0] < 3:
        indices = torch.arange(points.shape[0], device=points.device)
        if points.shape[0] == 2: 
            return indices, torch.tensor([[0, 1]], device=points.device, dtype=torch.long)
        elif points.shape[0] == 1: 
            return indices, torch.empty((0, 2), device=points.device, dtype=torch.long)
        else: 
            return torch.empty((0,), device=points.device, dtype=torch.long),                    torch.empty((0, 2), device=points.device, dtype=torch.long)

    sorted_indices = torch.lexsort((points[:, 1], points[:, 0]))
    
    upper_hull = [] 
    lower_hull = [] 

    def cross_product_orientation(p1_orig_idx, p2_orig_idx, p3_orig_idx, pts_tensor):
        p1 = pts_tensor[p1_orig_idx]
        p2 = pts_tensor[p2_orig_idx]
        p3 = pts_tensor[p3_orig_idx]
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    for i in range(points.shape[0]):
        current_original_idx = sorted_indices[i] 
        while len(upper_hull) >= 2:
            orientation = cross_product_orientation(upper_hull[-2], upper_hull[-1], current_original_idx, points)
            if orientation >= -tol: 
                upper_hull.pop()
            else:
                break
        upper_hull.append(current_original_idx.item()) 

    for i in range(points.shape[0] - 1, -1, -1):
        current_original_idx = sorted_indices[i] 
        while len(lower_hull) >= 2:
            orientation = cross_product_orientation(lower_hull[-2], lower_hull[-1], current_original_idx, points)
            if orientation >= -tol: 
                lower_hull.pop()
            else:
                break
        lower_hull.append(current_original_idx.item()) 

    hull_vertices_indices_list = upper_hull[:-1] + lower_hull[:-1] 
    hull_vertices_indices_unique_ordered = list(dict.fromkeys(hull_vertices_indices_list))
    hull_vertices_indices = torch.tensor(hull_vertices_indices_unique_ordered, dtype=torch.long, device=points.device)

    num_hull_vertices = hull_vertices_indices.shape[0]
    if num_hull_vertices < 2: 
        simplices = torch.empty((0, 2), dtype=torch.long, device=points.device)
    elif num_hull_vertices == 2: 
        simplices = torch.tensor([[hull_vertices_indices[0], hull_vertices_indices[1]]], dtype=torch.long, device=points.device)
    else: 
        simplices_list = []
        for i in range(num_hull_vertices):
            simplices_list.append([hull_vertices_indices[i].item(), hull_vertices_indices[(i + 1) % num_hull_vertices].item()])
        simplices = torch.tensor(simplices_list, dtype=torch.long, device=points.device)

    return hull_vertices_indices, simplices


def monotone_chain_convex_hull_3d(points: torch.Tensor, tol: float = 1e-7):
    # This is a complex function. Using the user's version with added comments for clarity.
    # The core logic seems to be an incremental construction (like Quickhull or gift-wrapping variant).
    n, dim = points.shape
    device = points.device
    if dim != 3: raise ValueError("Points must be 3D.")
    if n < 4: 
        unique_indices = torch.unique(torch.arange(n, device=device))
        return unique_indices, torch.empty((0, 3), dtype=torch.long, device=device)

    p0_idx = torch.argmin(points[:, 0]) 
    p0 = points[p0_idx]
    
    dists_from_p0_sq = torch.sum((points - p0)**2, dim=1)
    dists_from_p0_sq[p0_idx] = -1 
    p1_idx = torch.argmax(dists_from_p0_sq) 
    p1 = points[p1_idx]
    dists_from_p0_sq[p0_idx] = 0 

    line_vec = p1 - p0
    if torch.norm(line_vec) < tol:
        for i in range(n):
            if i != p0_idx.item() and torch.norm(points[i] - p0) > tol:
                p1_idx = torch.tensor(i, device=device, dtype=torch.long); p1 = points[p1_idx]; line_vec = p1 - p0
                break
        if torch.norm(line_vec) < tol: 
            return torch.unique(torch.tensor([p0_idx.item(), p1_idx.item()], device=device, dtype=torch.long)),                    torch.empty((0,3),dtype=torch.long,device=device)

    ap = points - p0 
    t = torch.matmul(ap, line_vec) / (torch.dot(line_vec, line_vec) + EPSILON) 
    projections_on_line = p0.unsqueeze(0) + t.unsqueeze(1) * line_vec.unsqueeze(0)
    dists_sq_from_line = torch.sum((points - projections_on_line)**2, dim=1)
    dists_sq_from_line[p0_idx] = -1; dists_sq_from_line[p1_idx] = -1 
    p2_idx = torch.argmax(dists_sq_from_line)
    p2 = points[p2_idx]
    dists_sq_from_line[p0_idx] = 0; dists_sq_from_line[p1_idx] = 0 

    def compute_plane_normal(pt0, pt1, pt2): return torch.cross(pt1 - pt0, pt2 - pt0)
    
    normal_p0p1p2 = compute_plane_normal(p0, p1, p2)
    if torch.norm(normal_p0p1p2) < tol:
        found_non_collinear_for_plane = False
        for i in range(n):
            if i != p0_idx.item() and i != p1_idx.item(): 
                temp_normal = compute_plane_normal(p0, p1, points[i])
                if torch.norm(temp_normal) > tol: 
                    p2_idx = torch.tensor(i, device=device, dtype=torch.long); p2 = points[p2_idx]
                    normal_p0p1p2 = temp_normal
                    found_non_collinear_for_plane = True; break
        if not found_non_collinear_for_plane:
            return torch.unique(torch.tensor([p0_idx.item(), p1_idx.item(), p2_idx.item()], device=device, dtype=torch.long)),                    torch.empty((0,3), dtype=torch.long, device=device)

    signed_dists_from_plane = torch.matmul(points - p0.unsqueeze(0), normal_p0p1p2)
    signed_dists_from_plane[p0_idx] = 0; signed_dists_from_plane[p1_idx] = 0; signed_dists_from_plane[p2_idx] = 0
    
    p3_idx = torch.argmax(torch.abs(signed_dists_from_plane))
    p3 = points[p3_idx]

    if torch.abs(signed_dists_from_plane[p3_idx]) < tol:
        all_coplanar_indices = torch.tensor([p0_idx.item(), p1_idx.item(), p2_idx.item(), p3_idx.item()], device=device, dtype=torch.long)
        return torch.unique(all_coplanar_indices), torch.empty((0, 3), dtype=torch.long, device=device)

    initial_simplex_indices_list = [p0_idx.item(), p1_idx.item(), p2_idx.item(), p3_idx.item()]
    if torch.dot(p3 - p0, normal_p0p1p2) < 0: 
        initial_simplex_indices_list = [p0_idx.item(), p2_idx.item(), p1_idx.item(), p3_idx.item()] 

    s = initial_simplex_indices_list
    
    # Initial faces, ensuring outward orientation (example for one face)
    vtx_coords = points[torch.tensor(s, device=device)]
    
    faces_list_of_lists = [
        [s[0],s[1],s[2]], [s[0],s[3],s[1]], [s[1],s[3],s[2]], [s[0],s[2],s[3]]
    ]
    
    # Correct orientation for initial faces
    # Face [s0,s1,s2] vs s3
    n012 = compute_plane_normal(vtx_coords[0],vtx_coords[1],vtx_coords[2])
    if torch.dot(vtx_coords[3]-vtx_coords[0], n012) > tol : faces_list_of_lists[0] = [s[0],s[2],s[1]]
    # Face [s0,s3,s1] vs s2
    n031 = compute_plane_normal(vtx_coords[0],vtx_coords[3],vtx_coords[1])
    if torch.dot(vtx_coords[2]-vtx_coords[0], n031) > tol : faces_list_of_lists[1] = [s[0],s[1],s[3]]
    # Face [s1,s3,s2] vs s0
    n132 = compute_plane_normal(vtx_coords[1],vtx_coords[3],vtx_coords[2])
    if torch.dot(vtx_coords[0]-vtx_coords[1], n132) > tol : faces_list_of_lists[2] = [s[1],s[2],s[3]]
    # Face [s0,s2,s3] vs s1
    n023 = compute_plane_normal(vtx_coords[0],vtx_coords[2],vtx_coords[3])
    if torch.dot(vtx_coords[1]-vtx_coords[0], n023) > tol : faces_list_of_lists[3] = [s[0],s[3],s[2]]

    current_faces = torch.tensor(faces_list_of_lists, dtype=torch.long, device=device)
    hull_vertex_indices_set = set(initial_simplex_indices_list) 
    
    is_processed_mask = torch.zeros(n, dtype=torch.bool, device=device)
    for idx_val in hull_vertex_indices_set: is_processed_mask[idx_val] = True 
    
    candidate_points_original_indices = torch.arange(n, device=device)[~is_processed_mask]

    for pt_orig_idx_tensor in candidate_points_original_indices:
        pt_orig_idx = pt_orig_idx_tensor.item()
        current_point_coords = points[pt_orig_idx]
        
        visible_faces_indices_in_current_faces = [] 
        for i_face, face_v_orig_indices_tensor in enumerate(current_faces):
            # face_v_orig_indices are original indices into `points`
            p_f0, p_f1, p_f2 = points[face_v_orig_indices_tensor[0]], points[face_v_orig_indices_tensor[1]], points[face_v_orig_indices_tensor[2]]
            face_normal = compute_plane_normal(p_f0, p_f1, p_f2) 
            if torch.norm(face_normal) < EPSILON: continue # Degenerate face normal

            if torch.dot(current_point_coords - p_f0, face_normal) > tol:
                visible_faces_indices_in_current_faces.append(i_face)
        
        if not visible_faces_indices_in_current_faces: 
            continue 
            
        hull_vertex_indices_set.add(pt_orig_idx) 
        
        edge_count = {} 
        for i_face_idx in visible_faces_indices_in_current_faces:
            face_orig_indices_tensor = current_faces[i_face_idx]
            # Convert tensor elements to Python int for tuple keys
            face_orig_indices_list = [idx.item() for idx in face_orig_indices_tensor]
            edges_on_face = [
                tuple(sorted((face_orig_indices_list[0], face_orig_indices_list[1]))),
                tuple(sorted((face_orig_indices_list[1], face_orig_indices_list[2]))),
                tuple(sorted((face_orig_indices_list[2], face_orig_indices_list[0])))
            ]
            for edge_tuple in edges_on_face:
                edge_count[edge_tuple] = edge_count.get(edge_tuple, 0) + 1
                
        horizon_edges_orig_indices_tuples = [edge_orig_idx_tuple for edge_orig_idx_tuple, count in edge_count.items() if count == 1]
        
        faces_to_keep_mask = torch.ones(current_faces.shape[0], dtype=torch.bool, device=device)
        for i_face_idx in visible_faces_indices_in_current_faces:
            faces_to_keep_mask[i_face_idx] = False
        
        temp_new_faces_list_of_lists = [f.tolist() for f in current_faces[faces_to_keep_mask]] 
        
        for edge_orig_idx_tuple in horizon_edges_orig_indices_tuples:
            # New face: (pt_orig_idx, edge_p1_orig_idx, edge_p0_orig_idx) for consistent winding assuming edge is (p0,p1)
            # This ensures the new point is on one side of the new face, forming an outward normal.
            temp_new_faces_list_of_lists.append([pt_orig_idx, edge_orig_idx_tuple[1], edge_orig_idx_tuple[0]])

        if not temp_new_faces_list_of_lists:
             if current_faces.numel() == 0 and n >=4 : break 
        
        current_faces = torch.tensor(temp_new_faces_list_of_lists, dtype=torch.long, device=device) if temp_new_faces_list_of_lists else torch.empty((0,3),dtype=torch.long,device=device)
        if current_faces.numel() == 0 and n >=4 : break 

    final_hull_vertex_indices_tensor = torch.tensor(list(hull_vertex_indices_set), dtype=torch.long, device=device)
    
    # Final faces must consist of vertices from the final hull_vertex_indices_set
    # And ensure valid faces (3 unique vertices)
    valid_faces_list_final = []
    if current_faces.numel() > 0:
        all_final_hull_indices_list = final_hull_vertex_indices_tensor.tolist()
        for face_indices_tensor in current_faces:
            face_indices_list = face_indices_tensor.tolist()
            # Check if all vertices of the face are in the final hull set AND are unique
            if len(set(face_indices_list)) == 3 and all(idx in all_final_hull_indices_list for idx in face_indices_list):
                valid_faces_list_final.append(face_indices_list)
    
    final_faces_tensor = torch.tensor(valid_faces_list_final, dtype=torch.long, device=device) if valid_faces_list_final else torch.empty((0,3),dtype=torch.long,device=device)
    
    # The returned vertices should be only those that appear in the final faces
    if final_faces_tensor.numel() > 0:
        final_hull_vertex_indices_tensor = torch.unique(final_faces_tensor.flatten())
    # else: final_hull_vertex_indices_tensor remains as computed from hull_vertex_indices_set,
    # which is correct for degenerate cases (plane, line, point) where faces might be empty.

    return final_hull_vertex_indices_tensor, final_faces_tensor


class ConvexHull:
    def __init__(self, points: torch.Tensor, tol: float = 1e-6):
        if not isinstance(points, torch.Tensor): raise ValueError("Input points must be a PyTorch tensor.")
        if points.ndim != 2: raise ValueError("Input points tensor must be 2-dimensional (N, D).")
        
        self.points = points; self.device = points.device; self.dtype = points.dtype
        self.dim = points.shape[1]; self.tol = tol
        
        if self.dim not in [2, 3]: raise ValueError("Only 2D and 3D points are supported.")
        
        self.vertices: torch.Tensor | None = None 
        self.simplices: torch.Tensor | None = None 
        self._area: torch.Tensor | None = None   
        self._volume: torch.Tensor | None = None 
        
        self._compute_hull()

    def _compute_hull(self):
        if self.points.shape[0] == 0: 
            self.vertices = torch.empty((0,), dtype=torch.long, device=self.device)
            self.simplices = torch.empty((0,2 if self.dim==2 else 3), dtype=torch.long, device=self.device)
            self._area = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            self._volume = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            return

        if self.dim == 2: self._convex_hull_2d()
        else: self._convex_hull_3d()

    def _convex_hull_2d(self):
        self.vertices, self.simplices = monotone_chain_2d(self.points, self.tol)
        if self.vertices.shape[0] < 3: 
            self._area = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        else:
            hull_pts_coords = self.points[self.vertices] 
            x, y = hull_pts_coords[:,0], hull_pts_coords[:,1]
            self._area = (0.5 * torch.abs(torch.sum(x * torch.roll(y,-1) - torch.roll(x,-1) * y))).to(self.dtype)

    def _compute_face_normal(self,v0_coords,v1_coords,v2_coords): 
        return torch.cross(v1_coords-v0_coords, v2_coords-v0_coords)

    def _convex_hull_3d(self):
        self.vertices, self.simplices = monotone_chain_convex_hull_3d(self.points, self.tol)
        
        surface_area = torch.tensor(0.0, device=self.device, dtype=self.points.dtype)
        if self.simplices is not None and self.simplices.shape[0] > 0:
            if self.simplices.numel() > 0 and (torch.max(self.simplices) >= self.points.shape[0] or torch.min(self.simplices) < 0):
                self._area = torch.tensor(0.0, device=self.device, dtype=self.dtype) # Invalid indices
            else:
                for face_orig_indices in self.simplices: 
                    p0_c,p1_c,p2_c = self.points[face_orig_indices[0]], self.points[face_orig_indices[1]], self.points[face_orig_indices[2]]
                    face_normal = self._compute_face_normal(p0_c,p1_c,p2_c)
                    if torch.norm(face_normal) > EPSILON : # Ensure non-degenerate normal
                         surface_area += 0.5 * torch.norm(face_normal)
        self._area = surface_area

        if self.vertices.numel() < 4 or self.simplices is None or self.simplices.shape[0] == 0 or            (self.simplices.numel() > 0 and (torch.max(self.simplices) >= self.points.shape[0] or torch.min(self.simplices) < 0)):
            self._volume = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            return

        ref_pt_coords = self.points[self.vertices[0]]
        total_signed_volume = torch.tensor(0.0,device=self.device,dtype=self.points.dtype)
        for face_orig_indices in self.simplices: 
            p0_c,p1_c,p2_c = self.points[face_orig_indices[0]],self.points[face_orig_indices[1]],self.points[face_orig_indices[2]]
            total_signed_volume += torch.dot(p0_c-ref_pt_coords, torch.cross(p1_c-ref_pt_coords, p2_c-ref_pt_coords))
        self._volume = torch.abs(total_signed_volume) / 6.0

    @property
    def area(self) -> torch.Tensor: 
        return self._area if self._area is not None else torch.tensor(0.0,device=self.device,dtype=self.dtype)
    @property
    def volume(self) -> torch.Tensor: 
        if self.dim == 2: return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        return self._volume if self._volume is not None else torch.tensor(0.0,device=self.device,dtype=self.dtype)

def _sutherland_hodgman_is_inside(point: torch.Tensor, edge_type: str, clip_value: float) -> bool:
    if edge_type == 'left': return point[0] >= clip_value  
    elif edge_type == 'top': return point[1] <= clip_value   
    elif edge_type == 'right': return point[0] <= clip_value 
    elif edge_type == 'bottom': return point[1] >= clip_value 
    return False 

def _sutherland_hodgman_intersect(p1: torch.Tensor, p2: torch.Tensor, 
                                  clip_edge_p1: torch.Tensor, clip_edge_p2: torch.Tensor) -> torch.Tensor:
    x1, y1 = p1[0].to(torch.float64), p1[1].to(torch.float64)
    x2, y2 = p2[0].to(torch.float64), p2[1].to(torch.float64)
    x3, y3 = clip_edge_p1[0].to(torch.float64), clip_edge_p1[1].to(torch.float64)
    x4, y4 = clip_edge_p2[0].to(torch.float64), clip_edge_p2[1].to(torch.float64)
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if torch.abs(denominator) < EPSILON: return p2 
    t_numerator = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    intersect_x = x1 + (t_numerator / denominator) * (x2 - x1)
    intersect_y = y1 + (t_numerator / denominator) * (y2 - y1)
    return torch.tensor([intersect_x, intersect_y], dtype=p1.dtype, device=p1.device)

def clip_polygon_2d(polygon_vertices: torch.Tensor, clip_bounds: torch.Tensor) -> torch.Tensor:
    if not isinstance(polygon_vertices, torch.Tensor) or polygon_vertices.ndim != 2 or polygon_vertices.shape[1] != 2:
        raise ValueError("polygon_vertices must be a tensor of shape (N, 2).")
    if polygon_vertices.shape[0] == 0: 
        return torch.empty((0,2), dtype=polygon_vertices.dtype, device=polygon_vertices.device)
    if not isinstance(clip_bounds, torch.Tensor) or clip_bounds.shape != (2,2):
        raise ValueError("clip_bounds must be a tensor of shape (2, 2) [[min_x, min_y], [max_x, max_y]].")

    min_x, min_y = clip_bounds[0,0], clip_bounds[0,1]; max_x, max_y = clip_bounds[1,0], clip_bounds[1,1]
    if not (min_x <= max_x and min_y <= max_y): 
        raise ValueError("Clip bounds min must be less than or equal to max for each dimension.")

    device=polygon_vertices.device; dtype=polygon_vertices.dtype
    clip_edges_params = [
        ('left', min_x, torch.tensor([min_x,min_y],device=device,dtype=dtype), torch.tensor([min_x,max_y],device=device,dtype=dtype)),
        ('top', max_y, torch.tensor([min_x,max_y],device=device,dtype=dtype), torch.tensor([max_x,max_y],device=device,dtype=dtype)),
        ('right', max_x, torch.tensor([max_x,max_y],device=device,dtype=dtype), torch.tensor([max_x,min_y],device=device,dtype=dtype)),
        ('bottom', min_y, torch.tensor([max_x,min_y],device=device,dtype=dtype), torch.tensor([min_x,min_y],device=device,dtype=dtype))
    ]
    output_vertices_list_py = polygon_vertices.tolist() 

    for edge_type, clip_val, clip_e_p1, clip_e_p2 in clip_edges_params:
        if not output_vertices_list_py: break 
        input_verts_stage = [torch.tensor(v,dtype=dtype,device=device) for v in output_vertices_list_py]
        output_vertices_list_py = []
        if not input_verts_stage: break
        S_pt = input_verts_stage[-1] 
        for P_pt in input_verts_stage:
            S_in = _sutherland_hodgman_is_inside(S_pt,edge_type,clip_val)
            P_in = _sutherland_hodgman_is_inside(P_pt,edge_type,clip_val)
            if S_in and P_in: output_vertices_list_py.append(P_pt.tolist())
            elif S_in and not P_in: 
                output_vertices_list_py.append(_sutherland_hodgman_intersect(S_pt,P_pt,clip_e_p1,clip_e_p2).tolist())
            elif not S_in and P_in:
                output_vertices_list_py.append(_sutherland_hodgman_intersect(S_pt,P_pt,clip_e_p1,clip_e_p2).tolist())
                output_vertices_list_py.append(P_pt.tolist())
            S_pt = P_pt
    if not output_vertices_list_py: return torch.empty((0,2),dtype=dtype,device=device)
    
    final_clipped_verts_py = []
    if len(output_vertices_list_py) > 1:
        final_clipped_verts_py.append(output_vertices_list_py[0])
        for i in range(1,len(output_vertices_list_py)):
            if not torch.allclose(torch.tensor(output_vertices_list_py[i],device=device,dtype=dtype), 
                                  torch.tensor(output_vertices_list_py[i-1],device=device,dtype=dtype),atol=EPSILON*10): # Increased atol slightly for robust duplicate removal
                final_clipped_verts_py.append(output_vertices_list_py[i])
        if len(final_clipped_verts_py)>1 and torch.allclose(torch.tensor(final_clipped_verts_py[0],device=device,dtype=dtype), 
                                                            torch.tensor(final_clipped_verts_py[-1],device=device,dtype=dtype),atol=EPSILON*10):
            final_clipped_verts_py.pop()
        if not final_clipped_verts_py: return torch.empty((0,2),dtype=dtype,device=device)
        return torch.tensor(final_clipped_verts_py,dtype=dtype,device=device)
    elif output_vertices_list_py: return torch.tensor(output_vertices_list_py,dtype=dtype,device=device)
    else: return torch.empty((0,2),dtype=dtype,device=device)

def _point_plane_signed_distance(point_coords: torch.Tensor, plane_normal: torch.Tensor, plane_d_offset: torch.Tensor) -> torch.Tensor:
    return torch.matmul(point_coords, plane_normal) - plane_d_offset

def _segment_plane_intersection(p1_coords: torch.Tensor, p2_coords: torch.Tensor, 
                                plane_normal: torch.Tensor, plane_d_offset: torch.Tensor, 
                                tol: float = 1e-7) -> torch.Tensor | None:
    dp = p2_coords - p1_coords 
    den = torch.dot(dp, plane_normal) 
    if torch.abs(den) < tol: return None 
    t = (plane_d_offset - torch.dot(p1_coords, plane_normal)) / den
    if -tol <= t <= 1.0 + tol: 
        return p1_coords + t * dp
    return None

def clip_polyhedron_3d(input_poly_vertices_coords: torch.Tensor, 
                       bounding_box_minmax: torch.Tensor, 
                       tol: float = 1e-7) -> torch.Tensor:
    if not (isinstance(input_poly_vertices_coords, torch.Tensor) and input_poly_vertices_coords.ndim == 2 and input_poly_vertices_coords.shape[1] == 3):
        raise ValueError("input_poly_vertices_coords must be a tensor of shape (N, 3).")
    if input_poly_vertices_coords.shape[0] == 0: return torch.empty_like(input_poly_vertices_coords)

    if not (isinstance(bounding_box_minmax, torch.Tensor) and bounding_box_minmax.shape == (2,3)):
        raise ValueError("bounding_box_minmax must be a tensor of shape (2, 3).")

    device = input_poly_vertices_coords.device; dtype = input_poly_vertices_coords.dtype
    input_poly_v_cpu = input_poly_vertices_coords # Assuming ConvexHull handles device
    bbox_cpu = bounding_box_minmax
    min_coords,max_coords = bbox_cpu[0],bbox_cpu[1]
    if not (torch.all(min_coords <= max_coords)):
        raise ValueError("Bounding box min_coords must be less than or equal to max_coords.")

    if input_poly_v_cpu.shape[0] < 4: # Degenerate case: not enough for 3D hull
        is_inside_box = torch.ones(input_poly_v_cpu.shape[0], dtype=torch.bool, device=device)
        for dim_idx in range(3):
            is_inside_box &= (input_poly_v_cpu[:, dim_idx] >= min_coords[dim_idx] - tol)
            is_inside_box &= (input_poly_v_cpu[:, dim_idx] <= max_coords[dim_idx] + tol)
        return input_poly_v_cpu[is_inside_box]

    planes_params = [
        (torch.tensor([1,0,0],dtype=dtype,device=device),min_coords[0]), (torch.tensor([-1,0,0],dtype=dtype,device=device),-max_coords[0]),
        (torch.tensor([0,1,0],dtype=dtype,device=device),min_coords[1]), (torch.tensor([0,-1,0],dtype=dtype,device=device),-max_coords[1]),
        (torch.tensor([0,0,1],dtype=dtype,device=device),min_coords[2]), (torch.tensor([0,0,-1],dtype=dtype,device=device),-max_coords[2])
    ]
    candidate_v_list = []
    for v_idx in range(input_poly_v_cpu.shape[0]):
        v_c = input_poly_v_cpu[v_idx]
        if (v_c[0]>=min_coords[0]-tol and v_c[0]<=max_coords[0]+tol and             v_c[1]>=min_coords[1]-tol and v_c[1]<=max_coords[1]+tol and             v_c[2]>=min_coords[2]-tol and v_c[2]<=max_coords[2]+tol):
            candidate_v_list.append(v_c)
    try:
        initial_hull = ConvexHull(input_poly_v_cpu, tol=tol)
        if initial_hull.simplices is not None and initial_hull.simplices.numel() > 0:
            unique_edges_orig_indices = set()
            for face_orig_indices in initial_hull.simplices:
                fi_list = face_orig_indices.tolist()
                unique_edges_orig_indices.update([tuple(sorted((fi_list[0],fi_list[1]))), tuple(sorted((fi_list[1],fi_list[2]))), tuple(sorted((fi_list[2],fi_list[0])))])
            for edge_tpl in unique_edges_orig_indices:
                p1_c,p2_c = input_poly_v_cpu[edge_tpl[0]],input_poly_v_cpu[edge_tpl[1]]
                for pl_norm,pl_d_off in planes_params:
                    intersect_pt_c = _segment_plane_intersection(p1_c,p2_c,pl_norm,pl_d_off,tol)
                    if intersect_pt_c is not None:
                        if (intersect_pt_c[0]>=min_coords[0]-tol and intersect_pt_c[0]<=max_coords[0]+tol and                             intersect_pt_c[1]>=min_coords[1]-tol and intersect_pt_c[1]<=max_coords[1]+tol and                             intersect_pt_c[2]>=min_coords[2]-tol and intersect_pt_c[2]<=max_coords[2]+tol):
                            candidate_v_list.append(intersect_pt_c)
    except (ValueError, RuntimeError): pass
    if not candidate_v_list: return torch.empty((0,3),dtype=dtype,device=device)
    unique_final_cand = torch.unique(torch.stack(candidate_v_list),dim=0)
    if unique_final_cand.shape[0]<4: return unique_final_cand.to(device)
    try:
        clipped_hull = ConvexHull(unique_final_cand, tol=tol)
        if clipped_hull.vertices is None or clipped_hull.vertices.numel()<1:
            return unique_final_cand.to(device) if unique_final_cand.shape[0]>0 else torch.empty((0,3),dtype=dtype,device=device)
        if clipped_hull.simplices is not None and clipped_hull.simplices.numel()>0:
            final_v_indices = torch.unique(clipped_hull.simplices.flatten())
            return unique_final_cand[final_v_indices].to(device)
        else: return unique_final_cand[clipped_hull.vertices].to(device)
    except (ValueError,RuntimeError): return unique_final_cand.to(device) if unique_final_cand.shape[0]>0 else torch.empty((0,3),dtype=dtype,device=device)

def compute_polygon_area(points_coords: torch.Tensor) -> float:
    if not (isinstance(points_coords,torch.Tensor) and points_coords.ndim==2 and points_coords.shape[1]==2):
        raise ValueError("Input points_coords must be a PyTorch tensor of shape (N, 2).")
    if points_coords.shape[0]<3: return 0.0
    try: hull = ConvexHull(points_coords,tol=EPSILON)
    except ValueError: return 0.0
    area_val = hull.area.item()
    return 0.0 if abs(area_val)<EPSILON else area_val

def compute_convex_hull_volume(points_coords: torch.Tensor) -> float:
    if not (isinstance(points_coords,torch.Tensor) and points_coords.ndim==2 and points_coords.shape[1]==3):
        raise ValueError("Input points_coords must be a PyTorch tensor of shape (N, 3).")
    if points_coords.shape[0]<4: return 0.0
    try: hull = ConvexHull(points_coords,tol=EPSILON)
    except ValueError: return 0.0
    vol_val = hull.volume.item()
    return 0.0 if abs(vol_val)<EPSILON else vol_val

def normalize_weights(weights: torch.Tensor, target_sum: float = 1.0, tol: float = 1e-7) -> torch.Tensor:
    if not isinstance(weights, torch.Tensor): raise TypeError("Input weights must be a PyTorch tensor.")
    if weights.numel()==0: return torch.empty_like(weights)
    if weights.dim()!=1: raise AssertionError("Weights must be a 1D tensor.")
    if torch.any(weights<-tol): raise ValueError(f"Weights must be non-negative (or within -{tol} tolerance).")
    clamped_weights = torch.clamp(weights,min=0.0)
    weight_sum = torch.sum(clamped_weights)
    if weight_sum<tol: raise ValueError(f"Sum of weights ({weight_sum.item()}) after clamping is less than tolerance ({tol}); cannot normalize.")
    return clamped_weights*(target_sum/weight_sum)
