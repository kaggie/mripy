import torch
import unittest
from voronoi_utils.geometry_algorithms import compute_triangle_circumcenter_2d, compute_tetrahedron_circumcenter_3d

class TestCircumcenterCalculations(unittest.TestCase):
    def test_circumcenter_2d_right_triangle(self):
        p1 = torch.tensor([0.,0.])
        p2 = torch.tensor([2.,0.])
        p3 = torch.tensor([0.,2.])
        center = compute_triangle_circumcenter_2d(p1,p2,p3)
        self.assertIsNotNone(center)
        if center is not None:
            self.assertTrue(torch.allclose(center, torch.tensor([1.,1.])))

    def test_circumcenter_2d_equilateral_triangle(self):
        p1 = torch.tensor([0.,0.], dtype=torch.float32)
        p2 = torch.tensor([2.,0.], dtype=torch.float32)
        p3 = torch.tensor([1., torch.sqrt(torch.tensor(3.0))], dtype=torch.float32)
        center = compute_triangle_circumcenter_2d(p1,p2,p3)
        self.assertIsNotNone(center)
        if center is not None:
            expected_center = torch.tensor([1.0, 1.0/torch.sqrt(torch.tensor(3.0))], dtype=torch.float32)
            self.assertTrue(torch.allclose(center, expected_center, atol=1e-6))

    def test_circumcenter_2d_collinear(self):
        p1 = torch.tensor([0.,0.])
        p2 = torch.tensor([1.,1.])
        p3 = torch.tensor([2.,2.])
        center = compute_triangle_circumcenter_2d(p1,p2,p3)
        self.assertIsNone(center)

    def test_circumcenter_3d_simple_tetrahedron(self):
        p1 = torch.tensor([0.,0.,0.])
        p2 = torch.tensor([1.,0.,0.])
        p3 = torch.tensor([0.,1.,0.])
        p4 = torch.tensor([0.,0.,1.])
        center = compute_tetrahedron_circumcenter_3d(p1,p2,p3,p4)
        self.assertIsNotNone(center)
        if center is not None:
            self.assertTrue(torch.allclose(center, torch.tensor([0.5,0.5,0.5])))
    
    def test_circumcenter_3d_regular_tetrahedron_origin_centered_ish(self):
        p1 = torch.tensor([1., 1., 1.])
        p2 = torch.tensor([1.,-1.,-1.])
        p3 = torch.tensor([-1.,1.,-1.])
        p4 = torch.tensor([-1.,-1.,1.])
        center = compute_tetrahedron_circumcenter_3d(p1,p2,p3,p4)
        self.assertIsNotNone(center)
        if center is not None:
             self.assertTrue(torch.allclose(center, torch.tensor([0.,0.,0.]), atol=1e-6))

    def test_circumcenter_3d_coplanar_points(self):
        p1 = torch.tensor([0.,0.,0.])
        p2 = torch.tensor([1.,0.,0.])
        p3 = torch.tensor([0.,1.,0.])
        p4 = torch.tensor([1.,1.,0.]) 
        center = compute_tetrahedron_circumcenter_3d(p1,p2,p3,p4)
        self.assertIsNone(center)
        
    def test_circumcenter_3d_another_coplanar(self):
        # Test case that might have caused issues if A matrix is singular
        # but volume check should catch it.
        p1 = torch.tensor([0.0, 0.0, 0.0])
        p2 = torch.tensor([1.0, 0.0, 0.0])
        p3 = torch.tensor([2.0, 0.0, 0.0]) # Collinear with p1, p2
        p4 = torch.tensor([0.0, 1.0, 0.0]) # Forms a plane with p1,p2,p3
        center = compute_tetrahedron_circumcenter_3d(p1,p2,p3,p4)
        self.assertIsNone(center)

if __name__ == '__main__':
    unittest.main()
