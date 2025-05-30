import torch
import unittest

# Assume EPSILON_GEOMETRY would be available from a core geometry file
# For this temporary file, define it directly.
EPSILON_GEOMETRY = 1e-7 

# --- Circumcenter Calculation Functions ---

def compute_triangle_circumcenter_2d(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor | None:
    """
    Computes the circumcenter of a 2D triangle.
    Args:
        p1, p2, p3: torch.Tensor of shape (2,) representing the triangle vertices.
    Returns:
        torch.Tensor (2,): Coordinates of the circumcenter.
        Returns None if points are collinear (degenerate triangle).
    """
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

    if torch.abs(D_f64) < EPSILON_GEOMETRY: 
        return None 

    p1_sq_f64 = p1x_f64**2 + p1y_f64**2
    p2_sq_f64 = p2x_f64**2 + p2y_f64**2
    p3_sq_f64 = p3x_f64**2 + p3y_f64**2

    Ux_f64 = (p1_sq_f64 * (p2y_f64 - p3y_f64) + 
              p2_sq_f64 * (p3y_f64 - p1y_f64) + 
              p3_sq_f64 * (p1y_f64 - p2y_f64)) / D_f64
    Uy_f64 = (p1_sq_f64 * (p3x_f64 - p2x_f64) + 
              p2_sq_f64 * (p1x_f64 - p3x_f64) + 
              p3_sq_f64 * (p2x_f64 - p1x_f64)) / D_f64
    
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
    points_f64 = torch.stack([p1, p2, p3, p4]).to(torch.float64)

    # Check for coplanarity using volume test
    v1 = points_f64[1] - points_f64[0]
    v2 = points_f64[2] - points_f64[0]
    v3 = points_f64[3] - points_f64[0]
    # Volume of tetrahedron is 1/6 * |det([v1, v2, v3])|
    vol_det_mat = torch.stack([v1, v2, v3], dim=0)
    volume_signed_x6 = torch.det(vol_det_mat) 

    # If 6*Volume is close to zero, points are coplanar.
    if torch.abs(volume_signed_x6) < EPSILON_GEOMETRY * 10: # Adjusted tolerance for volume check
        return None

    # System Ax = B for circumcenter [xc, yc, zc] and a constant k'
    # A_matrix * [xc, yc, zc, k']^T = B_vector
    A_matrix = torch.empty((4, 4), dtype=torch.float64, device=p1.device)
    B_vector = torch.empty((4, 1), dtype=torch.float64, device=p1.device)

    for i in range(4):
        pt = points_f64[i]
        A_matrix[i, 0] = 2 * pt[0]
        A_matrix[i, 1] = 2 * pt[1]
        A_matrix[i, 2] = 2 * pt[2]
        A_matrix[i, 3] = 1.0
        B_vector[i] = pt[0]**2 + pt[1]**2 + pt[2]**2
    
    try:
        # Check determinant of A_matrix before solving to avoid errors with singular matrices
        # This check should ideally be redundant given the volume check, but good for robustness.
        if torch.abs(torch.det(A_matrix)) < EPSILON_GEOMETRY:
             return None
        solution = torch.linalg.solve(A_matrix, B_vector)
        circumcenter_f64 = solution[:3].squeeze() # First 3 elements are xc, yc, zc
        return circumcenter_f64.to(dtype=p1.dtype) # Return in original dtype
    except Exception: 
        # Catch potential errors from solve (e.g., singular matrix if missed by checks)
        return None 

# --- Unit Tests ---

class TestCircumcenterCalculations(unittest.TestCase):
    def test_circumcenter_2d_right_triangle(self):
        p1 = torch.tensor([0.,0.])
        p2 = torch.tensor([2.,0.])
        p3 = torch.tensor([0.,2.])
        center = compute_triangle_circumcenter_2d(p1,p2,p3)
        self.assertIsNotNone(center, "Center should not be None for a right triangle")
        if center is not None:
            self.assertTrue(torch.allclose(center, torch.tensor([1.,1.])))

    def test_circumcenter_2d_equilateral_triangle(self):
        p1 = torch.tensor([0.,0.], dtype=torch.float32)
        p2 = torch.tensor([2.,0.], dtype=torch.float32)
        p3 = torch.tensor([1., torch.sqrt(torch.tensor(3.0))], dtype=torch.float32)
        center = compute_triangle_circumcenter_2d(p1,p2,p3)
        self.assertIsNotNone(center, "Center should not be None for an equilateral triangle")
        if center is not None:
            expected_center = torch.tensor([1.0, 1.0/torch.sqrt(torch.tensor(3.0))], dtype=torch.float32)
            self.assertTrue(torch.allclose(center, expected_center, atol=1e-6))

    def test_circumcenter_2d_collinear(self):
        p1 = torch.tensor([0.,0.])
        p2 = torch.tensor([1.,1.])
        p3 = torch.tensor([2.,2.])
        center = compute_triangle_circumcenter_2d(p1,p2,p3)
        self.assertIsNone(center, "Center should be None for collinear points")

    def test_circumcenter_3d_simple_tetrahedron(self):
        p1 = torch.tensor([0.,0.,0.])
        p2 = torch.tensor([1.,0.,0.])
        p3 = torch.tensor([0.,1.,0.])
        p4 = torch.tensor([0.,0.,1.])
        center = compute_tetrahedron_circumcenter_3d(p1,p2,p3,p4)
        self.assertIsNotNone(center, "Center should not be None for a simple tetrahedron")
        if center is not None:
            self.assertTrue(torch.allclose(center, torch.tensor([0.5,0.5,0.5])))
    
    def test_circumcenter_3d_regular_tetrahedron_origin_centered(self):
        # Vertices of a regular tetrahedron centered at origin if sqrt(3) is exact.
        # For robustness, use values that should make calculation stable.
        p1 = torch.tensor([1., 1., 1.]) 
        p2 = torch.tensor([1.,-1.,-1.])
        p3 = torch.tensor([-1.,1.,-1.])
        p4 = torch.tensor([-1.,-1.,1.])
        # This tetrahedron is centered at (0,0,0)
        center = compute_tetrahedron_circumcenter_3d(p1,p2,p3,p4)
        self.assertIsNotNone(center, "Center should not be None for this regular tetrahedron")
        if center is not None:
             self.assertTrue(torch.allclose(center, torch.tensor([0.,0.,0.]), atol=1e-6))

    def test_circumcenter_3d_coplanar_points(self):
        p1 = torch.tensor([0.,0.,0.])
        p2 = torch.tensor([1.,0.,0.])
        p3 = torch.tensor([0.,1.,0.])
        p4 = torch.tensor([1.,1.,0.]) # All points on z=0 plane
        center = compute_tetrahedron_circumcenter_3d(p1,p2,p3,p4)
        self.assertIsNone(center, "Center should be None for coplanar points")
        
    def test_circumcenter_3d_another_coplanar_set(self):
        # This set is also coplanar (on z=0) and includes collinear points.
        p1 = torch.tensor([0.0, 0.0, 0.0])
        p2 = torch.tensor([1.0, 0.0, 0.0])
        p3 = torch.tensor([2.0, 0.0, 0.0]) # p1,p2,p3 are collinear
        p4 = torch.tensor([0.0, 1.0, 0.0]) # p4 makes them coplanar but not all collinear
        center = compute_tetrahedron_circumcenter_3d(p1,p2,p3,p4)
        self.assertIsNone(center, "Center should be None for this coplanar/collinear set")

# To allow running tests if this file is executed directly (though not typical for agent)
# if __name__ == '__main__':
#    unittest.main(argv=['first-arg-is-ignored'], exit=False)
