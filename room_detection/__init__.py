"""
Room Detection Package

Automatically detect room geometry from OBJ files (BIM models).
"""

from .types import (
    Plane,
    VerticalHitResult,
    CandidatePoint,
    RoomDetectionConfig,
)
from .config import RoomDetectionConfig
from .mesh_loader import MeshData, load_building_mesh
from .grid_sampler import (
    generate_sample_grid,
    generate_sample_grid_at_height,
    get_floor_heights,
)
from .ray_caster import (
    cast_vertical_rays,
    cast_horizontal_rays,
    cast_single_ray,
)
from .point_filter import (
    filter_by_height,
    group_by_floor,
    create_candidate_points,
    filter_duplicate_candidates,
)
from .pipeline import (
    detect_rooms,
    detect_rooms_with_mesh,
    detect_rooms_by_floor,
    detect_rooms_simple,
)
from .visualizer import (
    visualize_results,
    create_point_cloud,
    create_ceiling_markers,
)
from .flood_fill import (
    Room,
    extract_rooms,
    flood_fill_for_candidate,
    group_candidates_by_level,
)

__all__ = [
    # Types
    'Plane',
    'VerticalHitResult',
    'CandidatePoint',
    'RoomDetectionConfig',
    # Mesh loading
    'MeshData',
    'load_building_mesh',
    # Grid sampling
    'generate_sample_grid',
    'generate_sample_grid_at_height',
    'get_floor_heights',
    # Ray casting
    'cast_vertical_rays',
    'cast_horizontal_rays',
    'cast_single_ray',
    # Filtering
    'filter_by_height',
    'group_by_floor',
    'create_candidate_points',
    'filter_duplicate_candidates',
    # Pipeline
    'detect_rooms',
    'detect_rooms_with_mesh',
    'detect_rooms_by_floor',
    'detect_rooms_simple',
    # Visualization
    'visualize_results',
    'create_point_cloud',
    'create_ceiling_markers',
    # Flood fill / Room extraction
    'Room',
    'extract_rooms',
    'flood_fill_for_candidate',
    'group_candidates_by_level',
]
