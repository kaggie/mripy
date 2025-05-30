# Voronoi Diagram Construction

This document describes functions for constructing Voronoi diagrams from Delaunay triangulations.

## 2D Voronoi Polygons from 2D Delaunay Triangulation

-   **Function:** `temp_voronoi_from_delaunay.construct_voronoi_polygons_2d(points: torch.Tensor, delaunay_triangles: torch.Tensor)`
    (Note: Currently in a temporary file due to environment issues.)
-   **Description:** Constructs the Voronoi cells for a set of 2D input points, given their Delaunay triangulation.
-   **Input:**
    -   `points`: A PyTorch tensor of shape `(N, 2)` containing the coordinates of the N input points.
    -   `delaunay_triangles`: A PyTorch tensor of shape `(M, 3)` representing M Delaunay triangles. Each row contains point indices referring to the `points` tensor.
-   **Output:**
    -   `voronoi_cells_vertices_list`: A list of lists. Each inner list `voronoi_cells_vertices_list[i]` contains PyTorch Tensors (each of shape `(2,)`) representing the ordered vertices of the Voronoi cell for `points[i]`.
    -   `unique_voronoi_vertices`: A PyTorch tensor of shape `(V, 2)` containing the coordinates of all unique Voronoi vertices (circumcenters of the Delaunay triangles).
-   **Algorithm:**
    1.  Calculates the circumcenter for each Delaunay triangle. These are the potential Voronoi vertices.
    2.  Builds a map from each input point to the list of Delaunay triangles incident to it.
    3.  For each input point, its Voronoi cell is formed by the circumcenters of its incident triangles.
    4.  The Voronoi vertices for a cell are ordered using a centroid-based angle sort. This is a heuristic that works for convex cells typically resulting from Delaunay duals.
-   **Notes:**
    -   The ordering of Voronoi vertices for unbounded cells (points on the convex hull of the input set) will represent the finite part of the cell. Proper handling of "points at infinity" or clipping against a bounding box is a separate step if needed.
    -   The uniqueness of Voronoi vertices is handled simply at the moment; a more robust tolerance-based unique vertex identification might be needed for complex cases.

## 3D Voronoi Polyhedra from 3D Delaunay Triangulation (Conceptual)

-   **Function:** `temp_voronoi_from_delaunay.construct_voronoi_polyhedra_3d(points: torch.Tensor, delaunay_tetrahedra: torch.Tensor)`
    (Note: Currently in a temporary file.)
-   **Description:** Constructs Voronoi cells (polyhedra) for 3D input points from their Delaunay triangulation.
-   **Input:**
    -   `points`: `(N, 3)` tensor of input point coordinates.
    -   `delaunay_tetrahedra`: `(M, 4)` tensor of Delaunay tetrahedra (point indices).
-   **Output:**
    -   `voronoi_cells_polyhedra_list`: List of lists. Each inner list contains PyTorch Tensors (shape `(3,)`) representing the Voronoi vertices that form the cell for the corresponding input point. The full polyhedral structure (faces, edges) is not explicitly derived in this version.
    -   `unique_voronoi_vertices`: `(V, 3)` tensor of unique Voronoi vertex coordinates (circumcenters of Delaunay tetrahedra).
-   **Algorithm:**
    1.  Computes circumcenters for all Delaunay tetrahedra (Voronoi vertices).
    2.  Maps input points to their incident tetrahedra.
    3.  The Voronoi cell for a point consists of the circumcenters of its incident tetrahedra.
-   **Notes:**
    -   This version primarily identifies the set of Voronoi vertices belonging to each cell. Constructing the full topological information (faces and their correct orientation) for each Voronoi polyhedron is a more complex step and is not fully implemented in the current simplified version.
    -   Ordering of vertices for a 3D polyhedron is non-trivial and typically requires defining its faces.
