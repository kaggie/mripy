# Distance Utilities

This document outlines utility functions related to distance calculations.

## Find Nearest Seed

-   **Function:** `temp_find_nearest_seed.find_nearest_seed(query_points: torch.Tensor, seeds: torch.Tensor, return_squared_distance: bool = True)`
    (Note: Currently in a temporary file `temp_find_nearest_seed.py` due to environment issues.)
-   **Description:** For each query point, this function finds the seed point from a given set of seeds that is closest in terms of Euclidean distance.
-   **Input:**
    -   `query_points`: A PyTorch tensor of shape `(Q, Dim)` representing Q query points, or `(Dim,)` for a single query point. `Dim` is the dimensionality of the space.
    -   `seeds`: A PyTorch tensor of shape `(S, Dim)` representing S seed points in the same `Dim`-dimensional space.
    -   `return_squared_distance` (optional, bool): If `True` (default), the function returns squared Euclidean distances. If `False`, it returns actual Euclidean distances (which involves a square root operation).
-   **Output:**
    -   `nearest_seed_indices`: A PyTorch tensor of shape `(Q,)` containing integer indices. Each index refers to the row in the `seeds` tensor that is closest to the corresponding query point. If no seeds are provided, indices are -1.
    -   `distances`: A PyTorch tensor of shape `(Q,)` containing the distances (squared or Euclidean, based on `return_squared_distance`) to the nearest seed for each query point. If no seeds are provided, distances are `float('inf')`.
-   **Algorithm:**
    1.  Handles single query points by unsqueezing them to `(1, Dim)`.
    2.  Calculates all pairwise squared Euclidean distances between query points and seed points using broadcasting.
    3.  For each query point, finds the minimum squared distance and the index of the corresponding seed using `torch.min`.
    4.  If Euclidean distance is requested, computes the square root of the minimum squared distances.
-   **Error Handling:**
    -   Raises a `ValueError` if the dimensionality of query points and seed points do not match.
    -   Handles the case of an empty `seeds` tensor gracefully.
-   **Dependencies:** PyTorch.
