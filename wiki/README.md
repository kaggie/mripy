# Voronoi Utilities (PyTorch Implementation)

Welcome to the documentation for the Voronoi Utilities project. This project aims to provide PyTorch-based implementations of various Voronoi diagram-related algorithms and helper functions.

## Overview

The utilities developed cover aspects of:
- Core geometric operations (convex hulls, clipping - *initially planned for `voronoi_utils/geometry_core.py`*)
- Delaunay triangulations (2D and 3D - *initially planned for `voronoi_utils/delaunay_2d.py` and `voronoi_utils/delaunay_3d.py`*)
- Voronoi diagram construction from Delaunay duals (*initially planned for `voronoi_utils/voronoi_from_delaunay.py`*)
- Distance calculations, such as finding the nearest seed.
- Rasterized Voronoi tessellation generation.

**Important Note on File Locations:** Due to temporary environment instability during development, some modules were implemented in temporary files in the root directory (e.g., `temp_*.py`) rather than their planned locations within a `voronoi_utils` package. This documentation will refer to the functions by their intended module structure where appropriate, but be aware that the actual file paths might be different in the current state.

## Available Documentation Modules

Please refer to the following pages for detailed documentation on specific functionalities:

-   **[Delaunay Triangulation](./delaunay.md)**
    -   Covers 2D and 3D Delaunay triangulation implementations.
    -   Includes details on helper functions like `_orientation3d_pytorch` and `_in_circumsphere3d_pytorch`.
    -   Also includes documentation for **Circumcenter Calculations** (`compute_triangle_circumcenter_2d`, `compute_tetrahedron_circumcenter_3d`) which are fundamental for Voronoi construction from Delaunay. 
        *(These were initially planned for `geometry_algorithms.py` but documented in `delaunay.md` during development flow).*

-   **[Voronoi Diagram Construction](./voronoi_construction.md)**
    -   Details on constructing Voronoi cell structures (polygons in 2D, conceptual polyhedra in 3D) from their Delaunay duals.

-   **[Distance Utilities](./distance_utils.md)**
    -   Documentation for utility functions like `find_nearest_seed`.

-   **[Voronoi Tessellation (Rasterization)](./tessellation.md)**
    -   Explains the `compute_voronoi_tessellation` function, which generates a grid-based (rasterized) Voronoi diagram by assigning each grid cell to its nearest seed.

## Future Development

The project aims to continue implementing more Voronoi-related features as outlined in the initial issue, including but not limited to:
- `SelectVoronoiSeeds`
- `voronoi_region_growing`
- `MergeVoronoiCells_And_OptimizePaths`
- And other utility functions like `getCellNeighbors`, `computeCellCentroid`, etc.

The intention is to eventually consolidate all functionalities into a well-structured `voronoi_utils` Python package with robust testing and clear usage examples.
