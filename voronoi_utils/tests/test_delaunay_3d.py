import torch
import unittest
from voronoi_utils.delaunay_3d import delaunay_triangulation_3d, _orientation3d_pytorch, _in_circumsphere3d_pytorch
from voronoi_utils.geometry_core import ConvexHull # For volume calculation in tests

class TestDelaunay3DHelpers(unittest.TestCase):
    def test_orientation3d(self):
        p1 = torch.tensor([0.,0.,0.])
        p2 = torch.tensor([1.,0.,0.])
        p3 = torch.tensor([0.,1.,0.])
        p4_pos = torch.tensor([0.,0.,1.]) # Positive orientation
        p4_neg = torch.tensor([0.,0.,-1.])# Negative orientation
        p4_coplanar = torch.tensor([0.5,0.5,0.]) # Coplanar

        self.assertEqual(_orientation3d_pytorch(p1,p2,p3,p4_pos, 1e-7), 1)
        self.assertEqual(_orientation3d_pytorch(p1,p2,p3,p4_neg, 1e-7), -1)
        self.assertEqual(_orientation3d_pytorch(p1,p2,p3,p4_coplanar, 1e-7), 0)

    def test_in_circumsphere_tetrahedron(self):
        # Tetrahedron vertices (example: a regular tetrahedron's concept)
        t1 = torch.tensor([1.,0.,-1./torch.sqrt(torch.tensor(2.0))])
        t2 = torch.tensor([-1.,0.,-1./torch.sqrt(torch.tensor(2.0))])
        t3 = torch.tensor([0.,1.,1./torch.sqrt(torch.tensor(2.0))])
        t4 = torch.tensor([0.,-1.,1./torch.sqrt(torch.tensor(2.0))])
        # Approximate circumcenter of this specific tetrahedron should be origin (0,0,0)
        # Approximate radius: sqrt( (1-0)^2 + (0-0)^2 + (-1/sqrt(2)-0)^2 ) = sqrt(1 + 0.5) = sqrt(1.5)

        p_inside = torch.tensor([0.1, 0.1, 0.1]) # Clearly inside
        p_outside = torch.tensor([2., 2., 2.])   # Clearly outside
        #p_on_approx = torch.tensor([1.,0.,-1./torch.sqrt(torch.tensor(2.0))]) * 0.9 # Slightly inside one vertex scaling
        
        # Need to ensure (t1,t2,t3,t4) has positive orientation for standard interpretation
        # For this set, Orient(t1,t2,t3,t4) might be negative. Let's check.
        # If _orientation3d_pytorch(t1,t2,t3,t4,1e-7) < 0, then result of _in_circumsphere3d_pytorch is flipped.
        # The _in_circumsphere3d_pytorch handles this by multiplying by orientation.

        self.assertTrue(_in_circumsphere3d_pytorch(p_inside, t1,t2,t3,t4, 1e-7))
        self.assertFalse(_in_circumsphere3d_pytorch(p_outside, t1,t2,t3,t4, 1e-7))
        # Point on a vertex is considered "on boundary", not strictly inside.
        self.assertFalse(_in_circumsphere3d_pytorch(t1, t1,t2,t3,t4, 1e-7)) 


class TestDelaunayTriangulation3D(unittest.TestCase):
    def test_dt3d_empty_input(self):
        points = torch.empty((0,3), dtype=torch.float32)
        tetrahedra = delaunay_triangulation_3d(points)
        self.assertEqual(tetrahedra.shape[0], 0)

    def test_dt3d_less_than_4_points(self):
        points = torch.tensor([[0.,0.,0.],[1.,1.,1.],[2.,0.,0.]])
        tetrahedra = delaunay_triangulation_3d(points)
        self.assertEqual(tetrahedra.shape[0], 0)

    def test_dt3d_4_points_coplanar(self):
        # Coplanar points should not form a valid 3D tetrahedron for Delaunay.
        points = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
        tetrahedra = delaunay_triangulation_3d(points)
        # Expect 0 tetrahedra as they are coplanar, cannot form a 3D simplex.
        self.assertEqual(tetrahedra.shape[0], 0)

    def test_dt3d_single_tetrahedron(self):
        points = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        tetrahedra = delaunay_triangulation_3d(points)
        self.assertEqual(tetrahedra.shape[0], 1) # Should form one tetrahedron
        if tetrahedra.shape[0] == 1:
            # Check if the tetrahedron uses the 4 input points (indices 0,1,2,3)
            self.assertTrue(all(i in tetrahedra[0].tolist() for i in [0,1,2,3]))

    def test_dt3d_cube_points(self):
        # 8 points of a unit cube
        points = torch.tensor([
            [0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.],
            [1.,1.,0.], [1.,0.,1.], [0.,1.,1.], [1.,1.,1.]
        ])
        tetrahedra = delaunay_triangulation_3d(points)
        # A common triangulation of a cube results in 5 or 6 tetrahedra.
        # For example, Schönhardt polyhedron cannot be triangulated without Steiner points,
        # but a cube can. Standard triangulations yield 5 or 6.
        self.assertTrue(tetrahedra.shape[0] >= 5) 
        
        # Basic check: all points used are from the original set.
        if tetrahedra.numel() > 0:
            self.assertTrue(torch.all(tetrahedra >= 0) and torch.all(tetrahedra < points.shape[0]))
        
        # Volume check (sum of volumes of tetrahedra should equal volume of cube's convex hull)
        # This is a more involved check.
        total_volume_delaunay = 0.0
        if tetrahedra.numel() > 0:
            for tet_indices in tetrahedra:
                tet_points = points[tet_indices]
                # Using ConvexHull to get volume of a single tetrahedron for simplicity
                # Direct formula: 1/6 * |det(p1-p0, p2-p0, p3-p0)|
                vol_mat = torch.stack([tet_points[1]-tet_points[0], 
                                       tet_points[2]-tet_points[0], 
                                       tet_points[3]-tet_points[0]], dim=0).to(torch.float64)
                total_volume_delaunay += torch.abs(torch.det(vol_mat)) / 6.0
        
        cube_hull = ConvexHull(points) # Volume of the original cube
        self.assertAlmostEqual(total_volume_delaunay, cube_hull.volume.item(), places=5)

if __name__ == '__main__':
    unittest.main()
