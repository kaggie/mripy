import torch

EPSILON_DELAUNAY_3D = 1e-7 # Epsilon for geometric predicates in 3D Delaunay

# User-provided helper functions for 3D Delaunay
def _orientation3d_pytorch(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor, tol: float) -> int:
    """
    Computes the orientation of point p4 relative to the plane defined by p1, p2, p3.
    Args:
        p1, p2, p3, p4: torch.Tensor of shape (3,) representing 3D points.
        tol: Tolerance for floating point comparisons.
    Returns:
        int: 
            0 if points are coplanar (within tolerance).
            1 if p4 is on one side of the plane (e.g., "positive" side).
           -1 if p4 is on the other side (e.g., "negative" side).
    """
    # Vector subtraction to get edges relative to p1
    v12 = p2 - p1
    v13 = p3 - p1
    v14 = p4 - p1

    # Matrix of these vectors (as rows or columns, determinant is the same up to sign)
    # mat = torch.stack((v12, v13, v14), dim=0) # If rows
    # For 3x3 matrix [a;b;c], det = a . (b x c)
    # Here, det = v12 . (v13 x v14) -- no, this is not the standard test.
    # Standard orientation test: sign of determinant of matrix:
    # | x2-x1  y2-y1  z2-z1 |
    # | x3-x1  y3-y1  z3-z1 |
    # | x4-x1  y4-y1  z4-z1 |
    # This is equivalent to (p2-p1) . cross(p3-p1, p4-p1) if p1 is origin.
    # Or more directly: det([p2-p1; p3-p1; p4-p1])

    mat = torch.stack((v12, v13, v14), dim=0) # Creates a 3x3 matrix
    
    # Use float64 for determinant calculation for precision
    det_val = torch.det(mat.to(dtype=torch.float64))

    if torch.abs(det_val) < tol:
        return 0  # Coplanar
    return 1 if det_val > 0 else -1

def _in_circumsphere3d_pytorch(p_check: torch.Tensor, 
                                   t1: torch.Tensor, t2: torch.Tensor, t3: torch.Tensor, t4: torch.Tensor, 
                                   tol: float) -> bool:
    """
    Checks if point p_check is strictly inside the circumsphere of the tetrahedron defined by t1, t2, t3, t4.
    Args:
        p_check: The point to check (3,).
        t1, t2, t3, t4: Vertices of the tetrahedron (3,).
        tol: Tolerance for determinant calculations.
    Returns:
        bool: True if p_check is strictly inside the circumsphere.
              False if on or outside, or if tetrahedron is degenerate.
    """
    # Robust InSphere predicate:
    # Sign of determinant of a 5x5 matrix:
    # | t1x  t1y  t1z  t1x^2+t1y^2+t1z^2  1 |
    # | t2x  t2y  t2z  t2x^2+t2y^2+t2z^2  1 |
    # | t3x  t3y  t3z  t3x^2+t3y^2+t3z^2  1 |
    # | t4x  t4y  t4z  t4x^2+t4y^2+t4z^2  1 |
    # | px   py   pz   px^2+py^2+pz^2    1 |
    # The sign depends on the orientation of (t1,t2,t3,t4).
    # If Orient3D(t1,t2,t3,t4) > 0, then p is inside if det(matrix) > 0.

    points_for_matrix = [t1, t2, t3, t4, p_check] # Order matters for final sign interpretation
    
    # Ensure all points are on the same device and dtype for matrix construction
    common_device = p_check.device
    common_dtype = p_check.dtype # Use original dtype for sum_sq, then cast matrix to float64 for det
    
    mat_rows = []
    for pt_i in points_for_matrix:
        pt_i_device_dtype = pt_i.to(device=common_device, dtype=common_dtype)
        sum_sq = torch.sum(pt_i_device_dtype**2) # ||pt_i||^2
        # Row: [x, y, z, x^2+y^2+z^2, 1]
        mat_rows.append(torch.cat((pt_i_device_dtype, sum_sq.unsqueeze(0), torch.tensor([1.0], device=common_device, dtype=common_dtype))))
    
    mat_5x5 = torch.stack(mat_rows, dim=0)
    
    # Calculate determinant using float64 for precision
    circumsphere_det_val = torch.det(mat_5x5.to(dtype=torch.float64))

    # Orientation of the tetrahedron (t1,t2,t3,t4)
    # This is crucial for interpreting the sign of circumsphere_det_val
    # Standard definition: Orient(t1,t2,t3,t4) > 0 if t4 is on positive side of plane (t1,t2,t3)
    orient_val = _orientation3d_pytorch(t1, t2, t3, t4, tol)

    if orient_val == 0: # Degenerate tetrahedron (coplanar points)
        return False # Cannot form a sphere, or point is effectively on a plane

    # Point is inside if (orient_val * circumsphere_det_val) > tol (for strict inside)
    # Using a small positive tolerance for strict "inside" check.
    return (orient_val * circumsphere_det_val) > tol


def delaunay_triangulation_3d(points: torch.Tensor, tol: float = EPSILON_DELAUNAY_3D) -> torch.Tensor:
    """
    Computes the 3D Delaunay triangulation of a set of points.
    Based on the incremental insertion algorithm (similar to Bowyer-Watson in 3D).
    Args:
        points (torch.Tensor): Tensor of shape (N, 3) representing N points in 3D.
        tol (float): Tolerance for geometric predicates.
    Returns:
        torch.Tensor: Tensor of shape (M, 4) representing M Delaunay tetrahedra.
                      Each row contains the original indices of the four points forming a tetrahedron.
                      Returns empty tensor (0,4) if N < 4.
    """
    n_input_points, dim = points.shape
    if dim != 3: raise ValueError("Input points must be 3-dimensional.")
    if n_input_points < 4: 
        return torch.empty((0, 4), dtype=torch.long, device=points.device)

    device = points.device
    original_dtype = points.dtype # Keep original dtype for most ops, use float64 for dets

    # Create a super-tetrahedron that encloses all input points
    min_coords, _ = torch.min(points, dim=0)
    max_coords, _ = torch.max(points, dim=0)
    center_coords = (min_coords + max_coords) / 2.0
    coord_range = max_coords - min_coords
    max_coord_range = torch.max(coord_range)
    if max_coord_range < tol : max_coord_range = 1.0 # Handle coincident input points

    # Scale factor for super-tetrahedron size
    # Make it significantly larger than the point cloud extent
    scale_factor_super = max(5.0 * max_coord_range, 10.0) 
    
    # Super-tetrahedron vertices (indices n_input_points to n_input_points+3)
    # Define them in a way that's unlikely to create degenerate initial conditions with input points
    sp_v0 = center_coords + torch.tensor([-scale_factor_super, -scale_factor_super * 0.5, -scale_factor_super * 0.25], device=device, dtype=original_dtype)
    sp_v1 = center_coords + torch.tensor([ scale_factor_super, -scale_factor_super * 0.5, -scale_factor_super * 0.25], device=device, dtype=original_dtype)
    sp_v2 = center_coords + torch.tensor([0.0,  scale_factor_super * 1.0, -scale_factor_super * 0.25], device=device, dtype=original_dtype)
    sp_v3 = center_coords + torch.tensor([0.0,  0.0,                       scale_factor_super * 1.25], device=device, dtype=original_dtype)
    
    super_tetra_vertices_coords = torch.stack([sp_v0, sp_v1, sp_v2, sp_v3], dim=0)
    
    # All points: original points followed by super-tetrahedron vertices
    all_points_coords = torch.cat([points, super_tetra_vertices_coords], dim=0)
    
    # Indices for super-tetrahedron vertices within all_points_coords
    st_idx_start = n_input_points
    st_indices_list = [st_idx_start, st_idx_start + 1, st_idx_start + 2, st_idx_start + 3]
    
    # Ensure initial super-tetrahedron has positive orientation
    p0_st, p1_st, p2_st, p3_st = all_points_coords[st_indices_list[0]], all_points_coords[st_indices_list[1]],                                      all_points_coords[st_indices_list[2]], all_points_coords[st_indices_list[3]]
    if _orientation3d_pytorch(p0_st, p1_st, p2_st, p3_st, tol) < 0:
        st_indices_list[1], st_indices_list[2] = st_indices_list[2], st_indices_list[1] # Swap two vertices to flip orientation

    # Initial triangulation: one tetrahedron (the super-tetrahedron)
    # Stores tetrahedra as lists of point indices into `all_points_coords`
    triangulation_tetra_indices = [st_indices_list] 

    # Optional: Process points in a shuffled order (can improve average performance)
    # permutation_orig_indices = torch.randperm(n_input_points, device=device)
    
    # Incremental insertion of each original point
    for i_point_loop_idx in range(n_input_points):
        # current_point_original_idx = permutation_orig_indices[i_point_loop_idx].item() # If using permutation
        current_point_original_idx = i_point_loop_idx # Index in original `points` tensor
        current_point_coords = all_points_coords[current_point_original_idx]

        bad_tetrahedra_indices_in_list = [] # List of indices of tetrahedra in `triangulation_tetra_indices`
        for tet_idx_in_list, tet_v_indices_list in enumerate(triangulation_tetra_indices):
            # Get coordinates of the vertices of the current tetrahedron
            v1_c,v2_c,v3_c,v4_c = all_points_coords[tet_v_indices_list[0]], all_points_coords[tet_v_indices_list[1]],                                       all_points_coords[tet_v_indices_list[2]], all_points_coords[tet_v_indices_list[3]]
            
            if _in_circumsphere3d_pytorch(current_point_coords, v1_c, v2_c, v3_c, v4_c, tol):
                bad_tetrahedra_indices_in_list.append(tet_idx_in_list)
        
        if not bad_tetrahedra_indices_in_list: # Point is not in any circumsphere (should be rare unless on boundary)
            continue

        # Find boundary faces of the polygonal cavity formed by bad tetrahedra
        boundary_faces_v_indices = [] # List of faces, each face is a list of 3 vertex indices
        face_counts = {} # Key: sorted tuple of 3 vertex indices (a face). Value: count.

        for tet_idx in bad_tetrahedra_indices_in_list:
            tet_v_orig_indices = triangulation_tetra_indices[tet_idx] # These are indices into all_points_coords
            # Faces of this tetrahedron (as sorted tuples of original indices)
            faces_of_this_tet = [
                tuple(sorted((tet_v_orig_indices[0], tet_v_orig_indices[1], tet_v_orig_indices[2]))),
                tuple(sorted((tet_v_orig_indices[0], tet_v_orig_indices[1], tet_v_orig_indices[3]))),
                tuple(sorted((tet_v_orig_indices[0], tet_v_orig_indices[2], tet_v_orig_indices[3]))),
                tuple(sorted((tet_v_orig_indices[1], tet_v_orig_indices[2], tet_v_orig_indices[3])))
            ]
            for face_tuple_sorted_indices in faces_of_this_tet:
                face_counts[face_tuple_sorted_indices] = face_counts.get(face_tuple_sorted_indices, 0) + 1
        
        for face_tuple_sorted_indices, count in face_counts.items():
            if count == 1: # This face is on the boundary of the cavity
                boundary_faces_v_indices.append(list(face_tuple_sorted_indices)) # Store as list [idx1,idx2,idx3]

        # Remove bad tetrahedra (iterate in reverse to preserve indices during pop)
        for tet_idx in sorted(bad_tetrahedra_indices_in_list, reverse=True):
            triangulation_tetra_indices.pop(tet_idx)

        # Retriangulate the cavity: form new tetrahedra by connecting current_point to each boundary face
        for face_v_indices_list in boundary_faces_v_indices:
            # New tetrahedron: (current_point_original_idx, face_v_idx1, face_v_idx2, face_v_idx3)
            # Ensure positive orientation for the new tetrahedron.
            # The face (v0,v1,v2) should be oriented such that current_point is on its "positive" side
            # if the face was part of an old tetrahedron that current_point was outside of.
            p_f0, p_f1, p_f2 = all_points_coords[face_v_indices_list[0]],                                    all_points_coords[face_v_indices_list[1]],                                    all_points_coords[face_v_indices_list[2]]
            
            # Check orientation of new tetrahedron: (p_f0, p_f1, p_f2, current_point_coords)
            current_orientation = _orientation3d_pytorch(p_f0, p_f1, p_f2, current_point_coords, tol)
            
            if current_orientation == 0: # Degenerate new tetrahedron (point coplanar with face)
                continue # Skip this degenerate tetrahedron

            new_tet_final_v_indices = [face_v_indices_list[0], face_v_indices_list[1], 
                                       face_v_indices_list[2], current_point_original_idx]
            if current_orientation < 0: # Flip two vertices of the base face to ensure positive orientation
                new_tet_final_v_indices = [face_v_indices_list[0], face_v_indices_list[2], # Swapped 1 and 2
                                           face_v_indices_list[1], current_point_original_idx]
            
            triangulation_tetra_indices.append(new_tet_final_v_indices)

    # Finalization: Remove all tetrahedra that include any vertex from the super-tetrahedron
    final_triangulation_list_of_lists = []
    for tet_v_indices_list in triangulation_tetra_indices:
        is_real_tetrahedron = True
        for v_idx_in_all_points in tet_v_indices_list:
            if v_idx_in_all_points >= n_input_points: # This vertex is part of the super-tetrahedron
                is_real_tetrahedron = False
                break
        if is_real_tetrahedron:
            final_triangulation_list_of_lists.append(tet_v_indices_list)
    
    if not final_triangulation_list_of_lists:
        return torch.empty((0, 4), dtype=torch.long, device=device)
        
    return torch.tensor(final_triangulation_list_of_lists, dtype=torch.long, device=device)
