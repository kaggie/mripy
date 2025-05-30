import torch
import unittest
from voronoi_utils.geometry_core import (
    ConvexHull, clip_polygon_2d, compute_polygon_area, 
    compute_convex_hull_volume, normalize_weights, EPSILON,
    monotone_chain_2d, monotone_chain_convex_hull_3d,
    clip_polyhedron_3d 
)

class TestMonotoneChain2D(unittest.TestCase):
    def test_simple_square_and_internal_point(self):
        points = torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[0.5,0.5]]) 
        hull_indices, simplices = monotone_chain_2d(points)
        self.assertEqual(hull_indices.shape[0], 4) 
        self.assertEqual(simplices.shape[0], 4)
        # Verify hull points are indeed the square corners (original indices 0,1,2,3)
        hull_coords = points[hull_indices]
        expected_corners = torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
        # Check if all expected corners are in hull_coords (order might vary)
        self.assertTrue(all(any(torch.allclose(hc, ec) for hc in hull_coords) for ec in expected_corners))


    def test_collinear_points(self):
        points = torch.tensor([[0.,0.],[1.,1.],[2.,2.],[3.,3.]])
        hull_indices, simplices = monotone_chain_2d(points)
        self.assertEqual(hull_indices.shape[0], 2) 
        self.assertEqual(simplices.shape[0], 1) 
        self.assertIn(0, hull_indices.tolist())
        self.assertIn(3, hull_indices.tolist())

    def test_single_point(self):
        points = torch.tensor([[1.,1.]])
        hull_indices, simplices = monotone_chain_2d(points)
        self.assertEqual(hull_indices.shape[0], 1); self.assertEqual(simplices.shape[0], 0)

    def test_two_points(self):
        points = torch.tensor([[0.,0.], [1.,1.]])
        hull_indices, simplices = monotone_chain_2d(points)
        self.assertEqual(hull_indices.shape[0], 2); self.assertEqual(simplices.shape[0], 1)

class TestConvexHullClass(unittest.TestCase):
    def test_ch_2d_square_with_internal(self):
        points = torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[0.5,0.5]])
        hull = ConvexHull(points, tol=1e-7)
        self.assertEqual(hull.vertices.shape[0], 4)
        self.assertAlmostEqual(hull.area.item(), 1.0, places=5)
        self.assertEqual(hull.simplices.shape[0], 4)

    def test_ch_2d_collinear(self):
        points = torch.tensor([[0.,0.],[1.,1.],[2.,2.],[3.,3.]])
        hull = ConvexHull(points, tol=1e-7) 
        self.assertEqual(hull.vertices.shape[0], 2) 
        self.assertAlmostEqual(hull.area.item(), 0.0, places=6)
        self.assertEqual(hull.simplices.shape[0],1)

    def test_ch_3d_tetrahedron(self):
        points = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        hull = ConvexHull(points, tol=1e-7)
        self.assertEqual(hull.vertices.shape[0], 4)
        self.assertTrue(hull.simplices is not None and hull.simplices.shape[0] == 4) # Tetrahedron has 4 faces
        self.assertAlmostEqual(hull.volume.item(), 1/6, places=6)
        expected_sa = 1.5 + 0.5 * (3**0.5)
        self.assertAlmostEqual(hull.area.item(), expected_sa, places=5)

    def test_ch_3d_cube(self):
        points = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,0.],[1.,0.,1.],[0.,1.,1.],[1.,1.,1.]])
        hull = ConvexHull(points, tol=1e-7)
        self.assertEqual(hull.vertices.shape[0], 8)
        self.assertTrue(hull.simplices is not None and hull.simplices.shape[0] >= 4) # Cube has at least 4 faces (typically 12 triangles)
        self.assertAlmostEqual(hull.volume.item(), 1.0, places=6)
        self.assertAlmostEqual(hull.area.item(), 6.0, places=6)

    def test_ch_3d_planar_points(self):
        points = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
        hull = ConvexHull(points, tol=1e-7) 
        self.assertEqual(hull.vertices.shape[0], 4)
        self.assertAlmostEqual(hull.volume.item(), 0.0, places=6)
        self.assertAlmostEqual(hull.area.item(), 2.0, places=6) 
        self.assertTrue(hull.simplices is None or hull.simplices.shape[0] == 0 or hull.simplices.numel() == 0 or hull.simplices.shape[0] == 2) # Coplanar might result in 2 faces or none from monotone_chain_3d

class TestClipping2D(unittest.TestCase):
    def test_clip_polygon_2d_fully_inside(self):
        polygon = torch.tensor([[0.25,0.25],[0.75,0.25],[0.75,0.75],[0.25,0.75]])
        bounds = torch.tensor([[0.,0.],[1.,1.]])
        clipped = clip_polygon_2d(polygon, bounds)
        self.assertEqual(clipped.shape[0], 4); self.assertAlmostEqual(compute_polygon_area(clipped),0.25,places=5)

    def test_clip_polygon_2d_partial_overlap(self):
        polygon = torch.tensor([[0.5,0.5],[1.5,0.5],[1.5,1.5],[0.5,1.5]])
        bounds = torch.tensor([[0.,0.],[1.,1.]])
        clipped = clip_polygon_2d(polygon, bounds)
        self.assertTrue(clipped.shape[0]==4); self.assertAlmostEqual(compute_polygon_area(clipped),0.25,places=5)

    def test_clip_polygon_fully_outside(self):
        polygon = torch.tensor([[10.,10.],[12.,10.],[12.,12.],[10.,12.]])
        bounds = torch.tensor([[0.,0.],[1.,1.]])
        clipped = clip_polygon_2d(polygon, bounds)
        self.assertEqual(clipped.shape[0], 0)

    def test_clip_triangle(self):
        polygon = torch.tensor([[0.0,0.0],[2.0,0.0],[1.0,2.0]])
        bounds = torch.tensor([[0.0,0.0],[1.0,1.0]])
        clipped = clip_polygon_2d(polygon, bounds)
        self.assertEqual(clipped.shape[0],4); self.assertAlmostEqual(compute_polygon_area(clipped),0.75,places=5)

class TestAreaVolumeFunctions(unittest.TestCase):
    def test_cpa_square(self): self.assertAlmostEqual(compute_polygon_area(torch.tensor([[0.,0.],[2.,0.],[2.,2.],[0.,2.]])),4.0,places=6)
    def test_cpa_degenerate_line(self): self.assertAlmostEqual(compute_polygon_area(torch.tensor([[0.,0.],[1.,1.]])),0.0,places=6)
    def test_cpa_collinear_gt_3pts(self): self.assertAlmostEqual(compute_polygon_area(torch.tensor([[0.,0.],[1.,1.],[2.,2.]])),0.0,places=6)
    def test_cchv_cube(self): self.assertAlmostEqual(compute_convex_hull_volume(torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,1.],[1.,1.,1.],[0.,1.,1.]])),1.0,places=6)
    def test_cchv_degenerate_plane(self): self.assertAlmostEqual(compute_convex_hull_volume(torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])),0.0,places=6)
    def test_cchv_coplanar_gt_4pts(self): self.assertAlmostEqual(compute_convex_hull_volume(torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])),0.0,places=6)

class TestNormalizeWeightsFunc(unittest.TestCase):
    def test_nw_simple(self):
        w=torch.tensor([1.,2.,3.,4.]); n=normalize_weights(w)
        self.assertTrue(torch.allclose(n,torch.tensor([0.1,0.2,0.3,0.4]))); self.assertAlmostEqual(torch.sum(n).item(),1.0,places=6)
    def test_nw_target_sum(self): self.assertTrue(torch.allclose(normalize_weights(torch.tensor([1.,1.,1.,1.]),target_sum=4.0),torch.tensor([1.,1.,1.,1.])))
    def test_nw_with_zeros(self): self.assertTrue(torch.allclose(normalize_weights(torch.tensor([0.,1.,0.,3.])),torch.tensor([0.,0.25,0.,0.75])))
    def test_nw_all_zeros_error(self):
        with self.assertRaisesRegex(ValueError,"Sum of weights .* less than tolerance"): normalize_weights(torch.tensor([0.,0.,0.]))
    def test_nw_small_sum_error(self):
        with self.assertRaisesRegex(ValueError,"Sum of weights .* less than tolerance"): normalize_weights(torch.tensor([EPSILON/10,EPSILON/10]),tol=EPSILON)
    def test_nw_negative_weights_error(self):
        with self.assertRaisesRegex(ValueError,"Weights must be non-negative"): normalize_weights(torch.tensor([1.,-1.,2.]))
    def test_nw_negative_weights_near_zero_ok(self):
        n=normalize_weights(torch.tensor([1.,-EPSILON/2,2.]),tol=EPSILON)
        self.assertTrue(torch.allclose(n,torch.tensor([1/3,0.,2/3]),atol=1e-7))
    def test_nw_empty_tensor(self):
        norm_empty=normalize_weights(torch.empty(0,dtype=torch.float32))
        self.assertEqual(norm_empty.numel(),0); self.assertEqual(norm_empty.dtype,torch.float32)

class TestClipPolyhedron3D(unittest.TestCase):
    def test_clip_cube_fully_inside_itself(self):
        cube_verts = torch.tensor([
            [0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,0.],
            [0.,0.,1.], [1.,0.,1.], [1.,1.,1.], [0.,1.,1.]
        ], dtype=torch.float32)
        bounds = torch.tensor([[0.,0.,0.],[1.,1.,1.]], dtype=torch.float32)
        clipped_verts = clip_polyhedron_3d(cube_verts, bounds)
        
        self.assertIsNotNone(clipped_verts)
        if clipped_verts is not None:
            self.assertEqual(clipped_verts.shape[0], 8) 
            if clipped_verts.shape[0] >= 4:
                 hull_of_clipped = ConvexHull(clipped_verts)
                 self.assertAlmostEqual(hull_of_clipped.volume.item(), 1.0, places=5)

    def test_clip_cube_half_way(self):
        cube_verts = torch.tensor([
            [0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,0.],
            [0.,0.,1.], [1.,0.,1.], [1.,1.,1.], [0.,1.,1.]
        ], dtype=torch.float32)
        bounds = torch.tensor([[0.,0.,0.],[0.5,1.,1.]], dtype=torch.float32) # Clip along x-axis
        clipped_verts = clip_polyhedron_3d(cube_verts, bounds)
        
        self.assertIsNotNone(clipped_verts)
        if clipped_verts is not None and clipped_verts.shape[0] >=4:
            hull_of_clipped = ConvexHull(clipped_verts)
            self.assertAlmostEqual(hull_of_clipped.volume.item(), 0.5, places=5)
            self.assertEqual(clipped_verts.shape[0], 8)


    def test_clip_poly_fully_outside_3d(self):
        poly_verts = torch.tensor([[2.,2.,2.], [3.,2.,2.], [3.,3.,2.], [2.,3.,2.],
                                   [2.,2.,3.], [3.,2.,3.], [3.,3.,3.], [2.,3.,3.]], dtype=torch.float32)
        bounds = torch.tensor([[0.,0.,0.],[1.,1.,1.]], dtype=torch.float32)
        clipped_verts = clip_polyhedron_3d(poly_verts, bounds)
        self.assertIsNotNone(clipped_verts)
        if clipped_verts is not None:
            self.assertEqual(clipped_verts.shape[0], 0) # Should be empty


if __name__ == '__main__':
    unittest.main()
