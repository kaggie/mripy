# Seed Selection Utilities

This document outlines utility functions for selecting seed points, typically for initializing Voronoi diagrams or other spatial algorithms.

## Select Voronoi Seeds from Quality Map

-   **Function:** `temp_seed_selection.select_voronoi_seeds(quality_map: torch.Tensor, voxel_size: tuple[float, ...] | torch.Tensor, quality_threshold: float, min_seed_distance: float, mask: torch.Tensor | None = None) -> torch.Tensor`
    (Note: Currently implemented in a temporary file `temp_seed_selection.py` due to initial environment setup issues.)

-   **Description:** This function selects a set of seed points from a given N-dimensional quality map. The selection is based on a quality threshold (points must be above this threshold to be considered) and a minimum distance criterion (selected seeds must be at least this far apart from each other).

-   **Input Parameters:**
    -   `quality_map (torch.Tensor)`: An N-dimensional tensor (currently supports N=2 or N=3) where each element's value represents the "quality" or desirability of that point as a seed. Higher values are considered better.
    -   `voxel_size (tuple[float, ...] | torch.Tensor)`: The physical size of each voxel/pixel in the `quality_map` across its dimensions. For a 2D quality map of shape (H, W), `voxel_size` could be `(voxel_height, voxel_width)`. For 3D of shape (D, H, W), it could be `(voxel_depth, voxel_height, voxel_width)`. This is crucial for calculating physical distances.
    -   `quality_threshold (float)`: A scalar float value. Only points in the `quality_map` whose quality is strictly greater than this threshold will be considered as initial candidates for seed selection.
    -   `min_seed_distance (float)`: A scalar float value representing the minimum physical distance that must be maintained between any two selected seeds.
    -   `mask (torch.Tensor | None, optional)`: A boolean tensor of the same shape as `quality_map`. If provided, only points where the `mask` is `True` are eligible for consideration (in addition to meeting the quality threshold). If `None`, all points in the `quality_map` are considered.

-   **Algorithm Steps (Conceptual):**
    1.  **Initialization**: Validates inputs. Converts `voxel_size` to a tensor.
    2.  **Mask Application**: If a `mask` is provided, points where the mask is `False` are effectively ignored by setting their quality in a processed map to negative infinity.
    3.  **Candidate Identification**: Identifies all points in the `processed_quality_map` whose values exceed the `quality_threshold`. The N-dimensional indices of these points are recorded. If no candidates are found, an empty tensor is returned.
    4.  **Coordinate Conversion**: The voxel/pixel indices of the candidates are converted into physical coordinates using the `voxel_size` (typically assuming the coordinate represents the center of the voxel/pixel).
    5.  **Sorting**: The identified candidates (now with their physical coordinates and original quality values) are sorted in descending order based on their quality scores.
    6.  **Iterative Selection**:
        a.  Initialize an empty list to store the final selected seeds.
        b.  Maintain a list or mask of currently available candidates (initially all sorted candidates).
        c.  While there are available candidates:
            i.  Select the candidate with the highest quality from the currently available list. Add its physical coordinates to the `selected_seeds` list.
            ii. Mark this seed as no longer available.
            iii. Iterate through the remaining available candidates. If a candidate is closer than `min_seed_distance` (Euclidean distance in physical coordinates) to the *just selected seed*, mark that candidate as unavailable.
    7.  **Output**: The list of selected seed physical coordinates is converted to a PyTorch tensor of shape `(N_selected, Dim)` and returned.

-   **Return Value:**
    -   A PyTorch tensor of shape `(N_selected, Dim)`, where `N_selected` is the number of seeds that met all criteria, and `Dim` is the dimensionality of the input `quality_map`. The coordinates are physical coordinates.
    -   Returns an empty tensor of shape `(0, Dim)` if no seeds are selected.

-   **Dependencies:** PyTorch.
