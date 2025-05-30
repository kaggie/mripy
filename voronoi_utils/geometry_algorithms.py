import torch

EPSILON_GEOMETRY = 1e-7 # Epsilon for geometric predicates

def compute_triangle_circumcenter_2d(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor | None:
    """
    Computes the circumcenter of a 2D triangle.
    Args:
        p1, p2, p3: torch.Tensor of shape (2,) representing the triangle vertices.
    Returns:
        torch.Tensor (2,): Coordinates of the circumcenter.
        Returns None if points are collinear (degenerate triangle).
    """
    # Using formulas from Wikipedia / mathworld.wolfram.com
    # Denominator D = 2 * (x1(y2-y3) + x2(y3-y1) + x3(y1-y2))
    # If D is close to zero, points are collinear.
    
    p1x, p1y = p1[0], p1[1]
    p2x, p2y = p2[0], p2[1]
    p3x, p3y = p3[0], p3[1]

    # Use float64 for precision in intermediate calculations
    p1x_f64, p1y_f64 = p1x.to(torch.float64), p1y.to(torch.float64)
    p2x_f64, p2y_f64 = p2x.to(torch.float64), p2y.to(torch.float64)
    p3x_f64, p3y_f64 = p3x.to(torch.float64), p3y.to(torch.float64)

    D_f64 = 2 * (p1x_f64 * (p2y_f64 - p3y_f64) + 
                   p2x_f64 * (p3y_f64 - p1y_f64) + 
                   p3x_f64 * (p1y_f64 - p2y_f64))

    if torch.abs(D_f64) < EPSILON_GEOMETRY: # Collinear points
        return None 

    p1_sq_f64 = p1x_f64**2 + p1y_f64**2
    p2_sq_f64 = p2x_f64**2 + p2y_f64**2
    p3_sq_f64 = p3x_f64**2 + p3y_f64**2

    # Circumcenter coordinates (Ux, Uy)
    Ux_f64 = (p1_sq_f64 * (p2y_f64 - p3y_f64) + 
              p2_sq_f64 * (p3y_f64 - p1y_f64) + 
              p3_sq_f64 * (p1y_f64 - p2y_f64)) / D_f64
    Uy_f64 = (p1_sq_f64 * (p3x_f64 - p2x_f64) + 
              p2_sq_f64 * (p1x_f64 - p3x_f64) + 
              p3_sq_f64 * (p2x_f64 - p1x_f64)) / D_f64
    
    # Return circumcenter in the original dtype of input points
    return torch.tensor([Ux_f64, Uy_f64], dtype=p1.dtype, device=p1.device)


def compute_tetrahedron_circumcenter_3d(p1: torch.Tensor, p2: torch.Tensor, 
                                        p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor | None:
    """
    Computes the circumcenter of a 3D tetrahedron.
    Args:
        p1, p2, p3, p4: torch.Tensor of shape (3,) representing the tetrahedron vertices.
    Returns:
        torch.Tensor (3,): Coordinates of the circumcenter.
        Returns None if points are coplanar (degenerate tetrahedron).
    """
    # Algorithm from "Geometric Tools for Computer Graphics" by Schneider and Eberly, page 103 (Barycentric method)
    # or using a matrix inversion method based on the sphere equation.
    # (x-xc)^2 + (y-yc)^2 + (z-zc)^2 = r^2
    # x^2 - 2x*xc + xc^2 + ... = r^2
    # 2x*xc + 2y*yc + 2z*zc + (r^2 - xc^2 - yc^2 - zc^2) = x^2+y^2+z^2
    # This forms a linear system A * [xc, yc, zc, k]^T = B
    # Where k = r^2 - xc^2 - yc^2 - zc^2 (or similar constant term)
    #
    # Matrix A:
    # | 2x1  2y1  2z1  1 |   | xc |   | x1^2+y1^2+z1^2 |
    # | 2x2  2y2  2z2  1 | * | yc | = | x2^2+y2^2+z2^2 |
    # | 2x3  2y3  2z3  1 |   | zc |   | x3^2+y3^2+z3^2 |
    # | 2x4  2y4  2z4  1 |   | k' |   | x4^2+y4^2+z4^2 |
    # (where k' is related to r^2 and center norms)
    
    # Using float64 for matrix operations for precision
    points_f64 = torch.stack([p1, p2, p3, p4]).to(torch.float64)

    # Check for coplanarity first using orientation test (volume of tetrahedron)
    # Form matrix for determinant: [p2-p1; p3-p1; p4-p1]
    v1 = points_f64[1] - points_f64[0]
    v2 = points_f64[2] - points_f64[0]
    v3 = points_f64[3] - points_f64[0]
    vol_det_mat = torch.stack([v1, v2, v3], dim=0)
    volume_signed_x6 = torch.det(vol_det_mat) # Volume is 1/6 of this

    if torch.abs(volume_signed_x6) < EPSILON_GEOMETRY * 10: # Adjusted tolerance for volume check
         # Points are coplanar or nearly coplanar, no unique circumsphere/center for a 3D shape.
        return None

    A = torch.empty((4, 4), dtype=torch.float64, device=p1.device)
    B = torch.empty((4, 1), dtype=torch.float64, device=p1.device)

    for i in range(4):
        pt = points_f64[i]
        A[i, 0] = 2 * pt[0]
        A[i, 1] = 2 * pt[1]
        A[i, 2] = 2 * pt[2]
        A[i, 3] = 1.0
        B[i] = pt[0]**2 + pt[1]**2 + pt[2]**2
    
    try:
        # Solve Ax = B for x = [xc, yc, zc, k']
        # Using torch.linalg.solve for better stability than inv()
        solution = torch.linalg.solve(A, B)
        circumcenter_f64 = solution[:3].squeeze() # First 3 elements are xc, yc, zc
        return circumcenter_f64.to(dtype=p1.dtype) # Return in original dtype
    except Exception: 
        # Catch potential errors from solve (e.g., singular matrix if somehow missed by volume check)
        return None
