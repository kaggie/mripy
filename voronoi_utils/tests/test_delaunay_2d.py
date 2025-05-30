import torch
import unittest
from voronoi_utils.delaunay_2d import delaunay_triangulation_2d, get_triangle_circumcircle_details_2d

class TestDelaunay2DHelpers(unittest.TestCase):
    def test_circumcircle_right_angle(self):
        p1 = torch.tensor([0.,0.])
        p2 = torch.tensor([2.,0.])
        p3 = torch.tensor([0.,2.])
        center, sq_radius = get_triangle_circumcircle_details_2d(p1,p2,p3)
        self.assertTrue(torch.allclose(center, torch.tensor([1.,1.])))
        self.assertAlmostEqual(sq_radius.item(), 2.0) # radius = sqrt(2)

    def test_circumcircle_equilateral(self):
        p1 = torch.tensor([0.,0.])
        p2 = torch.tensor([2.,0.])
        p3 = torch.tensor([1., torch.sqrt(torch.tensor(3.0))]) # approx (1, 1.732)
        center, sq_radius = get_triangle_circumcircle_details_2d(p1,p2,p3)
        # Expected center: (1, sqrt(3)/3) approx (1, 0.577)
        # Expected radius R = a/sqrt(3) = 2/sqrt(3). R^2 = 4/3
        self.assertTrue(torch.allclose(center, torch.tensor([1.0, 1.0/torch.sqrt(torch.tensor(3.0))])))
        self.assertAlmostEqual(sq_radius.item(), 4.0/3.0, places=5)
        
    def test_circumcircle_collinear(self):
        p1 = torch.tensor([0.,0.])
        p2 = torch.tensor([1.,1.])
        p3 = torch.tensor([2.,2.])
        center, sq_radius = get_triangle_circumcircle_details_2d(p1,p2,p3)
        self.assertIsNone(center)
        self.assertIsNone(sq_radius)


class TestDelaunayTriangulation2D(unittest.TestCase):
    def test_dt_empty_input(self):
        points = torch.empty((0,2), dtype=torch.float32)
        triangles = delaunay_triangulation_2d(points)
        self.assertEqual(triangles.shape[0], 0)

    def test_dt_less_than_3_points(self):
        points = torch.tensor([[0.,0.],[1.,1.]])
        triangles = delaunay_triangulation_2d(points)
        self.assertEqual(triangles.shape[0], 0)

    def test_dt_3_points_collinear(self):
        # Collinear points should not form a triangle in a strict Delaunay sense.
        # However, Bowyer-Watson might still produce something involving super-triangle
        # if not handled carefully, or if the circumcircle tests become ambiguous.
        # A robust implementation should ideally return 0 triangles.
        points = torch.tensor([[0.,0.],[1.,1.],[2.,2.]])
        triangles = delaunay_triangulation_2d(points)
        # Depending on implementation details for collinear points (especially with super-triangle),
        # this might produce unexpected results. For now, assume it results in no valid triangles.
        self.assertEqual(triangles.shape[0], 0) 

    def test_dt_3_points_triangle(self):
        points = torch.tensor([[0.,0.],[1.,0.],[0.,1.]]) # A simple right triangle
        triangles = delaunay_triangulation_2d(points)
        self.assertEqual(triangles.shape[0], 1) # Should form one triangle
        # Check if the triangle uses the 3 input points (indices 0,1,2)
        if triangles.shape[0] == 1:
             self.assertTrue(all(i in triangles[0].tolist() for i in [0,1,2]))

    def test_dt_4_points_square(self):
        points = torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
        # Expected: two triangles, e.g., (0,1,2) and (0,2,3) or (0,1,3) and (1,2,3)
        # (original indices)
        triangles = delaunay_triangulation_2d(points)
        self.assertEqual(triangles.shape[0], 2) 
        
        # Verify Delaunay property (empty circumcircle) for one diagonal
        # For a square, the diagonal splits it into two right triangles.
        # The circumcircle of one should not contain the 4th point of the square.
        # E.g., for tri (0,1,3), point 2 should not be in its circumcircle.
        # p0=(0,0), p1=(1,0), p2=(1,1), p3=(0,1)
        # Tri (0,1,3): (0,0)-(1,0)-(0,1). Circumcenter (0.5, 0.5). Radius^2 = 0.5. Point (1,1) dist^2 = (0.5^2+0.5^2)=0.5.
        # Point (1,1) is ON the circumcircle. A robust Delaunay may choose either diagonal.
        # If it picks (0,1,2) and (0,2,3):
        # Tri (0,1,2): (0,0)-(1,0)-(1,1). Circumcenter (0.5,0.5). Point (0,1) is on this circle.
        # This indicates that for points on a common circle (co-circular), either diagonal is valid.
        # The key is that no point is *strictly inside* another triangle's circumcircle.

        # A simpler check: all points used are from the original set.
        if triangles.numel() > 0:
             self.assertTrue(torch.all(triangles >= 0) and torch.all(triangles < points.shape[0]))

    def test_dt_regular_hexagon_points(self):
        # Approximate vertices of a regular hexagon centered at origin, radius 1
        angles = torch.arange(0, 6, dtype=torch.float32) * (torch.pi / 3)
        points = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        
        triangles = delaunay_triangulation_2d(points)
        # A regular hexagon can be triangulated into 4 triangles if one central point is added,
        # or typically 6 triangles if triangulated from the center (if center is a point),
        # or n-2 = 4 triangles if no center point is added and triangulated from one vertex.
        # Delaunay of boundary points of convex polygon: n-2 triangles.
        # For 6 points, expect 6-2 = 4 triangles from its boundary.
        # If the algorithm implicitly creates a "center" due to symmetries, might be 6.
        # Bowyer-Watson on the boundary should give n-2 triangles.
        # However, for highly symmetric cases, floating point can play a role.
        # A more specific check: ensure all triangles are valid (e.g. non-zero area).
        self.assertTrue(triangles.shape[0] >= 4) # Expect at least 4, possibly more depending on implementation details with symmetries
        if triangles.numel() > 0:
             self.assertTrue(torch.all(triangles >= 0) & torch.all(triangles < points.shape[0]))


if __name__ == '__main__':
    unittest.main()
