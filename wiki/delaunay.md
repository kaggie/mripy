# Delaunay Triangulation Utilities

This document outlines the Delaunay triangulation functions available in the `voronoi_utils` module.

## 2D Delaunay Triangulation

-   **Function:** `voronoi_utils.delaunay_2d.delaunay_triangulation_2d(points: torch.Tensor) -> torch.Tensor`
-   **Description:** Computes the 2D Delaunay triangulation of a set of input points using the Bowyer-Watson algorithm.
-   **Input:**
    -   `points`: A PyTorch tensor of shape `(N, 2)` representing N points in 2D.
-   **Output:**
    -   A PyTorch tensor of shape `(M, 3)` representing M Delaunay triangles. Each row contains the original indices of the three points forming a triangle. Returns an empty tensor `(0,3)` if N < 3.
-   **Algorithm Notes:**
    -   Uses a super-triangle to initialize the triangulation.
    -   Incrementally inserts points, identifies "bad" triangles (whose circumcircles contain the new point), removes them, and re-triangulates the resulting cavity.
    -   Removes triangles connected to the super-triangle in the final step.
-   **Dependencies:** PyTorch.

## 3D Delaunay Triangulation

-   **Function:** `voronoi_utils.delaunay_3d.delaunay_triangulation_3d(points: torch.Tensor, tol: float = 1e-7) -> torch.Tensor`
-   **Description:** Computes the 3D Delaunay triangulation of a set of input points. This implementation is based on an incremental insertion algorithm.
-   **Input:**
    -   `points`: A PyTorch tensor of shape `(N, 3)` representing N points in 3D.
    -   `tol` (optional): Tolerance for geometric predicate calculations (e.g., orientation, in-circumsphere tests). Defaults to `1e-7`.
-   **Output:**
    -   A PyTorch tensor of shape `(M, 4)` representing M Delaunay tetrahedra. Each row contains the original indices of the four points forming a tetrahedron. Returns an empty tensor `(0,4)` if N < 4.
-   **Algorithm Notes:**
    -   Initializes with a super-tetrahedron enclosing all input points.
    -   Points are added incrementally. For each point, tetrahedra whose circumspheres contain the point are identified and removed, forming a cavity.
    -   The cavity is re-triangulated by connecting the new point to the boundary faces of the cavity.
    -   Geometric predicates (`_orientation3d_pytorch`, `_in_circumsphere3d_pytorch`) are used with floating point arithmetic, and a tolerance parameter is available.
    -   Super-tetrahedron elements are removed at the end.
-   **Dependencies:** PyTorch.
-   **Source:** Based on user-provided code.
