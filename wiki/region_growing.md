# Voronoi Region Growing

This document describes utility functions related to Voronoi-based region growing.

## Voronoi Region Growing from Initial Seeds

-   **Function:** `temp_region_growing.voronoi_region_growing(initial_seeds_phys_coords: torch.Tensor, quality_map: torch.Tensor, voxel_size: tuple[float,...] | torch.Tensor, mask: torch.Tensor | None = None, max_iterations: int = 100, stop_threshold_fraction: float = 0.001) -> torch.Tensor`
    (Note: Currently implemented in a temporary file `temp_region_growing.py` due to initial environment setup issues.)

-   **Description:** This function performs a region growing process starting from a set of initial seed points. It iteratively expands regions by assigning unassigned voxels to the region of their nearest initial seed, based on physical distance. The growth can be constrained by a mask and is controlled by iteration limits and a stopping threshold. While named "Voronoi" region growing, this initial version directly uses proximity to the *original* seed locations rather than dynamically updating region characteristics.

-   **Input Parameters:**
    -   `initial_seeds_phys_coords (torch.Tensor)`: A PyTorch Tensor of shape `(S, Dim)` containing the physical coordinates of `S` initial seed points. `Dim` is the dimensionality (2 or 3).
    -   `quality_map (torch.Tensor)`: A PyTorch Tensor of shape `(D1, ..., Dk)` (where k=Dim). In the current implementation, this tensor primarily defines the shape, device, and dtype for the output `segmentation_map`. Its values are not directly used for determining growth affinity in the proximity-based logic but are part of the function signature for potential future extensions.
    -   `voxel_size (tuple[float,...] | torch.Tensor)`: The physical size of each voxel/pixel in each dimension (e.g., `(voxel_height, voxel_width)` for 2D). This is critical for converting between voxel indices and physical coordinates.
    -   `mask (torch.Tensor | None, optional)`: A boolean tensor of the same shape as `quality_map`. If provided, region growing is confined to areas where the mask is `True`. Voxels outside the mask (where `mask` is `False`) will not be assigned to any region and will remain -1 in the output. Defaults to `None` (no mask, all voxels in the `quality_map` shape are considered processable).
    -   `max_iterations (int, optional)`: The maximum number of iterations the region growing process will run. Defaults to `100`.
    -   `stop_threshold_fraction (float, optional)`: A fraction (e.g., 0.001 for 0.1%) relative to the total number of processable voxels. If the number of voxels that change their assignment in an iteration drops below `stop_threshold_fraction * num_processable_voxels`, the process stops. This helps terminate early if the segmentation stabilizes. Defaults to `0.001`.

-   **Algorithm Steps (Conceptual):**
    1.  **Initialization**:
        a.  Input parameters are validated. Dimensionality (Dim) is determined.
        b.  A `segmentation_map` tensor is created with the same shape as `quality_map`, initialized with -1 (representing unassigned voxels).
        c.  A `processable_mask` is derived from the input `mask` (or defaults to all `True` if no mask is given). The total number of processable voxels is calculated.
        d.  The physical coordinates of `initial_seeds_phys_coords` are mapped to their nearest voxel indices on the grid. These voxels in `segmentation_map` are assigned unique region IDs (0 to S-1). If a seed's nearest voxel is outside the `processable_mask`, it's ignored.
    2.  **Iterative Growth**:
        a.  The process iterates up to `max_iterations`.
        b.  In each iteration, "active front" voxels are identified. These are unassigned voxels (`segmentation_map == -1`) that are within the `processable_mask` and are adjacent (currently using 4-connectivity for 2D, 6-connectivity for 3D, implemented via a convolution with a specific kernel) to at least one voxel already assigned to a region.
        c.  If no active front voxels are found, the iteration stops.
        d.  For each voxel in the active front: Its physical coordinates (center of the voxel) are calculated. The distances from this voxel center to all *original physical seed locations* (`initial_seeds_phys_coords`) are computed. The voxel is assigned the region ID of the closest initial seed.
        e.  The number of voxels that changed their assignment in the current iteration is counted.
        f.  The process stops if no voxels changed assignment, or if the number of changed voxels is below the `stop_threshold_fraction` of the total processable voxels.
-   **Return Value:**
    -   An integer PyTorch tensor `segmentation_map` of the same shape as `quality_map`. Each element contains a region ID (0 to S-1) indicating which initial seed's region the voxel belongs to, or -1 if the voxel is unassigned (e.g., outside the mask or not reached by the growth process).

-   **Dependencies:** PyTorch. Relies on helper `map_physical_coords_to_voxel_indices`.

-   **Notes:**
    -   The current version's growth logic is based purely on proximity to the *initial* seed locations. It does not, for example, update region characteristics (like mean position) as they grow, nor does it heavily use the `quality_map` values to guide growth beyond initial masking (though the signature allows for future enhancements).
    -   The neighbor finding uses a convolution operation, which is a common technique for grid-based expansions.
