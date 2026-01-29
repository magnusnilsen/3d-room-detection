"""
Grid sampling for room detection.
"""

import numpy as np
from typing import List, Tuple, Optional
from .types import RoomDetectionConfig


def get_floor_heights(
    bounds: np.ndarray,
    config: RoomDetectionConfig
) -> np.ndarray:
    """
    Calculate Z heights at which to sample for each floor.
    
    Args:
        bounds: Mesh bounding box [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        config: Room detection configuration
        
    Returns:
        Array of Z coordinates to sample at
    """
    min_z = bounds[0][2]
    max_z = bounds[1][2]
    
    # Calculate number of potential floors
    height = max_z - min_z
    
    if height <= config.max_room_height_cm:
        # Single floor - sample at mid height
        return np.array([min_z + height / 2])
    
    # Multi-floor - sample at regular intervals
    # Start at sample_offset_from_floor_cm above the bottom
    # Then every floor_height_cm
    floor_heights = np.arange(
        min_z + config.sample_offset_from_floor_cm,
        max_z,
        config.floor_height_cm
    )
    
    return floor_heights


def generate_xy_grid(
    bounds: np.ndarray,
    spacing_cm: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate XY grid coordinates within bounds.
    
    Args:
        bounds: Mesh bounding box
        spacing_cm: Grid spacing in centimeters
        
    Returns:
        Tuple of (X coordinates, Y coordinates) as 1D arrays
    """
    min_x, min_y = bounds[0][0], bounds[0][1]
    max_x, max_y = bounds[1][0], bounds[1][1]
    
    # Start half a spacing inside the bounds
    xs = np.arange(min_x + spacing_cm / 2, max_x, spacing_cm)
    ys = np.arange(min_y + spacing_cm / 2, max_y, spacing_cm)
    
    return xs, ys


def generate_sample_grid(
    bounds: np.ndarray,
    config: Optional[RoomDetectionConfig] = None
) -> np.ndarray:
    """
    Generate a 3D grid of sample points for ray casting.
    
    Samples at multiple floor heights for multi-story buildings.
    
    Args:
        bounds: Mesh bounding box [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        config: Room detection configuration (uses defaults if None)
        
    Returns:
        Nx3 array of sample points
    """
    if config is None:
        config = RoomDetectionConfig()
    
    # Get floor heights
    floor_heights = get_floor_heights(bounds, config)
    
    # Generate XY grid
    xs, ys = generate_xy_grid(bounds, config.grid_spacing_cm)
    
    # Create meshgrid for XY
    xx, yy = np.meshgrid(xs, ys)
    xy_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Replicate for each floor height
    all_points = []
    for z in floor_heights:
        z_column = np.full((len(xy_points), 1), z)
        points_at_z = np.hstack([xy_points, z_column])
        all_points.append(points_at_z)
    
    result = np.vstack(all_points)
    
    print(f"Generated {len(result)} sample points "
          f"({len(xs)}x{len(ys)} grid, {len(floor_heights)} floor(s))")
    
    return result


def generate_sample_grid_at_height(
    bounds: np.ndarray,
    z_height: float,
    spacing_cm: float = 25.0
) -> np.ndarray:
    """
    Generate a grid of sample points at a specific Z height.
    
    Args:
        bounds: Mesh bounding box
        z_height: Z coordinate for all sample points
        spacing_cm: Grid spacing in centimeters
        
    Returns:
        Nx3 array of sample points
    """
    xs, ys = generate_xy_grid(bounds, spacing_cm)
    
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        np.full(xx.size, z_height)
    ])
    
    return points
