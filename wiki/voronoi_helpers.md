# Voronoi Helper Utilities

This document describes various helper functions that can be used in conjunction with Voronoi diagram construction and analysis.

## Compute Cell Centroid

-   **Function:** `temp_voronoi_helpers.compute_cell_centroid(cell_vertices_list: list[torch.Tensor]) -> torch.Tensor | None`
    (Note: Currently implemented in a temporary file `temp_voronoi_helpers.py`.)

-   **Description:** Calculates the centroid of a Voronoi cell given the list of its vertex coordinates.

-   **Input Parameters:**
    -   `cell_vertices_list (list[torch.Tensor])`: A list of PyTorch Tensors. Each tensor in the list represents a single vertex of the cell (e.g., a 1D tensor of shape `(Dim,)`, where `Dim` is typically 2 for 2D cells or 3 for 3D cells). All vertex tensors in the list should have consistent dimensionality.

-   **Algorithm & Notes:**
    -   The function first stacks the list of vertex tensors into a single tensor of shape `(N_vertices, Dim)`.
    -   It then computes the centroid as the arithmetic mean of these vertex coordinates along the `N_vertices` dimension.
    -   **For 2D convex polygons** (which Voronoi cells are), this method correctly calculates the geometric centroid.
    -   **For 3D polyhedra**, this method calculates the center of mass of the vertices. This serves as a common and simple approximation of the true geometric centroid (volumetric centroid), especially for reasonably symmetric shapes. For arbitrary polyhedra, the true geometric centroid calculation is more complex (e.g., requiring decomposition into tetrahedra).
    -   Handles cases with empty input list or malformed vertex data by returning `None`.
    -   Currently supports 2D and 3D vertices.

-   **Return Value:**
    -   A PyTorch Tensor of shape `(Dim,)` representing the coordinates of the centroid.
    -   Returns `None` if the input `cell_vertices_list` is empty, if vertices are not consistently shaped, or if the dimension is not 2 or 3.

## Get Cell Neighbors (Conceptual - 2D Focus)

-   **Function:** `temp_voronoi_helpers.get_cell_neighbors(target_cell_index: int, all_cells_vertices_list: list[list[torch.Tensor]], shared_vertices_threshold: int = 2) -> list[int]`
    (Note: Currently implemented in a temporary file `temp_voronoi_helpers.py`.)

-   **Description:** Identifies neighboring cells for a specified target cell from a list of all Voronoi cells. This implementation is primarily focused on and simplified for 2D Voronoi diagrams.

-   **Input Parameters:**
    -   `target_cell_index (int)`: The index of the cell (within `all_cells_vertices_list`) for which neighbors are to be found.
    -   `all_cells_vertices_list (list[list[torch.Tensor]])`: A list where each element is a Voronoi cell. Each cell, in turn, is represented as a list of PyTorch Tensors (its vertices).
    -   `shared_vertices_threshold (int, optional)`: The minimum number of shared vertices required for two cells to be considered neighbors. Defaults to `2`.
        -   For typical 2D Voronoi cells, sharing 2 vertices indicates a shared edge.
        -   Sharing 1 vertex indicates they meet at a point.

-   **Algorithm & Notes (2D Focus):**
    1.  Validates the `target_cell_index`.
    2.  Retrieves the vertices of the target cell. These vertex coordinate tensors are converted into a hashable format (tuples of Python floats, rounded to a fixed precision to aid in robust comparison despite potential minor floating-point variations).
    3.  Iterates through all other cells in `all_cells_vertices_list`.
    4.  For each other cell, its vertices are also converted into a set of hashable, rounded tuples.
    5.  The intersection of the two vertex sets (target cell's and current other cell's) is found.
    6.  If the number of vertices in this intersection (i.e., common vertices) is greater than or equal to `shared_vertices_threshold`, the index of the other cell is added to a list of neighbors.
    7.  **Floating-Point Precision**: Vertex comparison uses rounding to a fixed number of decimal places before converting to tuples. This is a heuristic to manage floating-point inaccuracies when trying to identify if vertices are "the same". A more advanced system might use an epsilon-based comparison or a robust geometric library.
    8.  **3D Adjacency**: This logic is simplified for 2D. In 3D, Voronoi cells are polyhedra, and neighbors share faces (which are polygons, defined by >=3 coplanar vertices). A simple count of shared vertices would not be sufficient to robustly determine shared faces in 3D; it would require checking for coplanarity and polygon matching of the shared vertices. The current function with `shared_vertices_threshold=2` would find cells sharing at least an edge in 3D, not necessarily a face.

-   **Return Value:**
    -   A list of integer indices representing the cells in `all_cells_vertices_list` that are considered neighbors to the cell at `target_cell_index`.

-   **Dependencies:** PyTorch.
