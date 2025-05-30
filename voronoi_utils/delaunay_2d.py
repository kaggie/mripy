import torch

EPSILON_DELAUNAY = 1e-7 # Epsilon for geometric predicates in Delaunay

def get_triangle_circumcircle_details_2d(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor):
    """
    Computes the circumcenter and squared circumradius of a 2D triangle.
    Args:
        p1, p2, p3: torch.Tensor of shape (2,) representing the triangle vertices.
    Returns:
        Tuple[torch.Tensor, torch.Tensor] or Tuple[None, None]:
            - circumcenter (torch.Tensor (2,)): Coordinates of the circumcenter.
            - squared_radius (torch.Tensor (scalar)): Squared circumradius.
            Returns (None, None) if points are collinear (degenerate triangle).
    """
    # Using formulas from Wikipedia / mathworld.wolfram.com
    # Denominator D = 2 * (x1(y2-y3) + x2(y3-y1) + x3(y1-y2))
    # If D is close to zero, points are collinear.
    
    p1x, p1y = p1[0], p1[1]
    p2x, p2y = p2[0], p2[1]
    p3x, p3y = p3[0], p3[1]

    D = 2 * (p1x * (p2y - p3y) + p2x * (p3y - p1y) + p3x * (p1y - p2y))

    if torch.abs(D) < EPSILON_DELAUNAY: # Collinear points
        return None, None 

    p1_sq = p1x**2 + p1y**2
    p2_sq = p2x**2 + p2y**2
    p3_sq = p3x**2 + p3y**2

    # Circumcenter coordinates (Ux, Uy)
    Ux = (p1_sq * (p2y - p3y) + p2_sq * (p3y - p1y) + p3_sq * (p1y - p2y)) / D
    Uy = (p1_sq * (p3x - p2x) + p2_sq * (p1x - p3x) + p3_sq * (p2x - p1x)) / D
    
    circumcenter = torch.tensor([Ux, Uy], dtype=p1.dtype, device=p1.device)
    
    # Squared radius: (x1-Ux)^2 + (y1-Uy)^2
    squared_radius = (p1x - Ux)**2 + (p1y - Uy)**2
    
    return circumcenter, squared_radius

def is_point_in_circumcircle(point: torch.Tensor, 
                                 tri_p1: torch.Tensor, tri_p2: torch.Tensor, tri_p3: torch.Tensor) -> bool:
    """
    Checks if a point is strictly inside the circumcircle of a triangle.
    Args:
        point: torch.Tensor (2,) - The point to check.
        tri_p1, tri_p2, tri_p3: torch.Tensor (2,) - Vertices of the triangle.
    Returns:
        bool: True if the point is strictly inside the circumcircle.
              False if on or outside, or if triangle is degenerate.
    """
    circumcenter, squared_radius = get_triangle_circumcircle_details_2d(tri_p1, tri_p2, tri_p3)
    
    if circumcenter is None: # Degenerate triangle
        return False 
        
    dist_sq_to_center = torch.sum((point - circumcenter)**2)
    
    # Check if point is strictly inside (dist_sq < radius_sq)
    # Add a small tolerance for floating point comparisons if needed,
    # but Bowyer-Watson often relies on strict "in" for point location.
    return dist_sq_to_center < squared_radius - EPSILON_DELAUNAY


def delaunay_triangulation_2d(points: torch.Tensor) -> torch.Tensor:
    """
    Computes the 2D Delaunay triangulation of a set of points using the Bowyer-Watson algorithm.
    Args:
        points (torch.Tensor): Tensor of shape (N, 2) representing N points in 2D.
    Returns:
        torch.Tensor: Tensor of shape (M, 3) representing M Delaunay triangles.
                      Each row contains the original indices of the three points forming a triangle.
                      Returns empty tensor (0,3) if N < 3.
    """
    n_points = points.shape[0]
    if n_points < 3:
        return torch.empty((0, 3), dtype=torch.long, device=points.device)

    device = points.device
    dtype = points.dtype

    # 1. Initialization: Create a super-triangle that encompasses all input points.
    # Find min/max coordinates to define the super-triangle bounds.
    min_coords, _ = torch.min(points, dim=0)
    max_coords, _ = torch.max(points, dim=0)
    
    center = (min_coords + max_coords) / 2.0
    range_coords = max_coords - min_coords
    max_range = torch.max(range_coords)
    if max_range < EPSILON_DELAUNAY : max_range = 1.0 # Handle case where all points are nearly coincident

    # Define super-triangle vertices (make it large enough)
    # These points are temporary and their indices will be > n_points-1
    # Using a factor like 3 or 5 times max_range should be safe.
    offset_val = max_range * 5.0 
    # ST_p0 must be far from min_coords to avoid issues if points are near origin
    st_p0 = center + torch.tensor([-offset_val, -offset_val * 0.5], device=device, dtype=dtype) 
    st_p1 = center + torch.tensor([offset_val,  -offset_val * 0.5], device=device, dtype=dtype)
    st_p2 = center + torch.tensor([0.0,          offset_val * 1.5], device=device, dtype=dtype)


    # Combine original points with super-triangle vertices for processing
    # Original points: 0 to n_points-1
    # Super-triangle points: n_points, n_points+1, n_points+2
    all_points = torch.cat([points, st_p0.unsqueeze(0), st_p1.unsqueeze(0), st_p2.unsqueeze(0)], dim=0)
    
    idx_st_p0 = n_points
    idx_st_p1 = n_points + 1
    idx_st_p2 = n_points + 2

    # Initial triangulation: one triangle (the super-triangle)
    # Stores triangles as lists of point indices into `all_points`
    triangulation = [[idx_st_p0, idx_st_p1, idx_st_p2]]

    # 2. Incremental Point Insertion
    # Optional: Shuffle points for better average-case performance (not strictly necessary for correctness)
    # permutation = torch.randperm(n_points, device=device)

    for i in range(n_points):
        # current_point_original_idx = permutation[i].item() # if shuffling
        current_point_original_idx = i # Original index of the point in the input `points` tensor
        current_point_coords = all_points[current_point_original_idx]

        bad_triangles_indices = [] # Indices of triangles in `triangulation` list
        for tri_idx, tri_indices_list in enumerate(triangulation):
            p1_coords = all_points[tri_indices_list[0]]
            p2_coords = all_points[tri_indices_list[1]]
            p3_coords = all_points[tri_indices_list[2]]
            
            if is_point_in_circumcircle(current_point_coords, p1_coords, p2_coords, p3_coords):
                bad_triangles_indices.append(tri_idx)

        # Form polygonal cavity
        polygon_cavity_edges = [] # List of edges (tuples of 2 point indices)
        edge_counts = {} # To find edges that appear only once (boundary of bad triangles region)

        for tri_idx in bad_triangles_indices:
            tri_v_indices = triangulation[tri_idx] # Indices into all_points
            edges_of_tri = [
                tuple(sorted((tri_v_indices[0], tri_v_indices[1]))),
                tuple(sorted((tri_v_indices[1], tri_v_indices[2]))),
                tuple(sorted((tri_v_indices[2], tri_v_indices[0])))
            ]
            for edge_tuple in edges_of_tri:
                edge_counts[edge_tuple] = edge_counts.get(edge_tuple, 0) + 1
        
        for edge_tuple, count in edge_counts.items():
            if count == 1: # This edge is part of the polygonal cavity boundary
                polygon_cavity_edges.append(list(edge_tuple)) # Store as [idx1, idx2]

        # Remove bad triangles (iterate in reverse to maintain valid indices)
        for tri_idx in sorted(bad_triangles_indices, reverse=True):
            triangulation.pop(tri_idx)

        # Retriangulate cavity: Add new triangles by connecting current_point to cavity edges
        for edge_v_indices_list in polygon_cavity_edges:
            # New triangle: (current_point_original_idx, edge_v_idx1, edge_v_idx2)
            # Ensure consistent orientation if necessary, though Bowyer-Watson usually handles this.
            new_triangle = [current_point_original_idx, edge_v_indices_list[0], edge_v_indices_list[1]]
            triangulation.append(new_triangle)
    
    # 3. Finalization: Remove triangles that include vertices from the super-triangle.
    # These are triangles where any vertex index is >= n_points.
    final_triangulation_list = []
    for tri_v_indices_list in triangulation:
        is_real_triangle = True
        for v_idx in tri_v_indices_list:
            if v_idx >= n_points: # Vertex is part of the super-triangle
                is_real_triangle = False
                break
        if is_real_triangle:
            final_triangulation_list.append(tri_v_indices_list)
    
    if not final_triangulation_list:
        return torch.empty((0, 3), dtype=torch.long, device=device)
        
    return torch.tensor(final_triangulation_list, dtype=torch.long, device=device)
