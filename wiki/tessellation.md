# Voronoi Tessellation (Rasterization)

This document describes the function for generating a rasterized Voronoi diagram.

## Compute Voronoi Tessellation

-   **Function:** `temp_voronoi_tessellation.compute_voronoi_tessellation(shape: tuple[int, ...], seeds: torch.Tensor, voxel_size: tuple[float, ...] | torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor`
    (Note: Currently in a temporary file `temp_voronoi_tessellation.py` due to environment issues.)
-   **Description:** Generates a rasterized Voronoi diagram on a grid. Each voxel (or pixel in 2D) in the grid is assigned to the nearest seed point.
-   **Input:**
    -   `shape`: A tuple of integers defining the dimensions of the output grid (e.g., `(H, W)` for 2D, `(D, H, W)` for 3D).
    -   `seeds`: A PyTorch tensor of shape `(S, Dim)` containing the coordinates of S seed points. `Dim` is the dimensionality (2 or 3) and must match the number of dimensions in `shape`.
    -   `voxel_size`: A tuple of floats or a PyTorch tensor defining the physical size of a voxel in each dimension (e.g., `(voxel_height, voxel_width)` for 2D, or `(voxel_depth, voxel_height, voxel_width)` for 3D). The number of elements must match `Dim`. It's assumed the order corresponds to the `shape` dimensions.
    -   `mask` (optional): A PyTorch boolean tensor of the same `shape` as the output grid. If provided, the tessellation is computed only for voxels where the mask is `True`. Voxels corresponding to `False` in the mask will be assigned a value of -1 in the output. If `None` (default), all voxels are tessellated.
-   **Output:**
    -   A PyTorch integer tensor of the specified `shape`. Each element in this tensor contains the index (from 0 to S-1, corresponding to the input `seeds` tensor) of the seed that the voxel is closest to. Voxels outside a provided mask or if no seeds are given are assigned -1.
-   **Algorithm:**
    1.  Validates input dimensions and shapes.
    2.  If no seeds are provided, returns a grid filled with -1.
    3.  Generates a grid of voxel center coordinates based on the `shape` and `voxel_size`.
    4.  If a `mask` is provided, filters these coordinates to include only those corresponding to `True` values in the mask.
    5.  If no query points remain after masking, returns the initial grid (-1 filled, or -1 in masked out areas).
    6.  Uses the `find_nearest_seed` function to determine the closest seed index for each selected voxel center coordinate.
    7.  Populates an output tensor of the target `shape`. For masked computations, values are placed at the appropriate locations corresponding to `True` mask elements; other elements remain -1. For non-masked computations, the output tensor is reshaped directly from the results of `find_nearest_seed`.
-   **Dependencies:** PyTorch, and the `find_nearest_seed` function (assumed to be available).
