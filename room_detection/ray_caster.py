"""
Ray casting utilities for room detection.
"""

import numpy as np
import trimesh
from typing import List, Optional, Tuple
from .types import VerticalHitResult, RoomDetectionConfig
from .mesh_loader import MeshData


def cast_vertical_rays(
    mesh_data: MeshData,
    points: np.ndarray,
    config: Optional[RoomDetectionConfig] = None
) -> List[VerticalHitResult]:
    """
    Cast rays up and down from each sample point to find interior points.
    
    Args:
        mesh_data: Loaded mesh data with group information
        points: Nx3 array of sample points
        config: Room detection configuration
        
    Returns:
        List of VerticalHitResult for points that hit both up and down
    """
    if config is None:
        config = RoomDetectionConfig()
    
    mesh = mesh_data.mesh
    n_points = len(points)
    
    if n_points == 0:
        return []
    
    # Direction vectors
    dir_up = np.array([0.0, 0.0, 1.0])
    dir_down = np.array([0.0, 0.0, -1.0])
    
    results = []
    
    # Process in batches for memory efficiency
    batch_size = config.batch_size
    
    for batch_start in range(0, n_points, batch_size):
        batch_end = min(batch_start + batch_size, n_points)
        batch_points = points[batch_start:batch_end]
        batch_n = len(batch_points)
        
        # Create ray arrays
        ray_origins = batch_points.copy()
        ray_dirs_up = np.tile(dir_up, (batch_n, 1))
        ray_dirs_down = np.tile(dir_down, (batch_n, 1))
        
        # Cast rays up
        hits_up_locs, hits_up_ray_idx, hits_up_face_idx = mesh.ray.intersects_location(
            ray_origins, ray_dirs_up, multiple_hits=True
        )
        
        # Cast rays down
        hits_down_locs, hits_down_ray_idx, hits_down_face_idx = mesh.ray.intersects_location(
            ray_origins, ray_dirs_down, multiple_hits=True
        )
        
        # Process each point in batch
        batch_results = _process_vertical_hits(
            batch_points=batch_points,
            batch_start_idx=batch_start,
            hits_up_locs=hits_up_locs,
            hits_up_ray_idx=hits_up_ray_idx,
            hits_up_face_idx=hits_up_face_idx,
            hits_down_locs=hits_down_locs,
            hits_down_ray_idx=hits_down_ray_idx,
            hits_down_face_idx=hits_down_face_idx,
            mesh_data=mesh_data
        )
        
        results.extend(batch_results)
    
    print(f"Vertical ray casting: {len(results)}/{n_points} points have both up and down hits")
    
    return results


def _process_vertical_hits(
    batch_points: np.ndarray,
    batch_start_idx: int,
    hits_up_locs: np.ndarray,
    hits_up_ray_idx: np.ndarray,
    hits_up_face_idx: np.ndarray,
    hits_down_locs: np.ndarray,
    hits_down_ray_idx: np.ndarray,
    hits_down_face_idx: np.ndarray,
    mesh_data: MeshData
) -> List[VerticalHitResult]:
    """
    Process raw ray casting results to find closest hits in each direction.
    """
    results = []
    batch_n = len(batch_points)
    
    for i in range(batch_n):
        # Get hits for this ray (up direction)
        up_mask = hits_up_ray_idx == i
        if not np.any(up_mask):
            continue
        
        # Get hits for this ray (down direction)
        down_mask = hits_down_ray_idx == i
        if not np.any(down_mask):
            continue
        
        # Find closest hit in each direction
        origin = batch_points[i]
        
        # Up direction - find minimum distance (closest ceiling)
        up_locs = hits_up_locs[up_mask]
        up_faces = hits_up_face_idx[up_mask]
        up_distances = up_locs[:, 2] - origin[2]  # Z distance
        closest_up_idx = np.argmin(up_distances)
        top_hit = up_locs[closest_up_idx]
        top_face_idx = up_faces[closest_up_idx]
        
        # Down direction - find minimum distance (closest floor)
        down_locs = hits_down_locs[down_mask]
        down_faces = hits_down_face_idx[down_mask]
        down_distances = origin[2] - down_locs[:, 2]  # Z distance (positive)
        closest_down_idx = np.argmin(down_distances)
        bottom_hit = down_locs[closest_down_idx]
        
        # Calculate room height
        room_height = float(top_hit[2] - bottom_hit[2])
        
        # Get group name for the top hit
        group_name = mesh_data.get_group_name(top_face_idx)
        
        results.append(VerticalHitResult(
            sample_point=origin.copy(),
            top_hit=top_hit.copy(),
            bottom_hit=bottom_hit.copy(),
            room_height=room_height,
            top_face_idx=int(top_face_idx),
            top_mesh_name=group_name
        ))
    
    return results


def cast_single_ray(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    direction: np.ndarray
) -> Optional[Tuple[np.ndarray, float, int]]:
    """
    Cast a single ray and return the closest hit.
    
    Args:
        mesh: The mesh to cast against
        origin: Ray origin point
        direction: Ray direction (will be normalized)
        
    Returns:
        Tuple of (hit_point, distance, face_idx) or None if no hit
    """
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    
    origins = np.array([origin])
    directions = np.array([direction])
    
    locations, ray_indices, face_indices = mesh.ray.intersects_location(
        origins, directions, multiple_hits=True
    )
    
    if len(locations) == 0:
        return None
    
    # Find closest hit
    distances = np.linalg.norm(locations - origin, axis=1)
    closest_idx = np.argmin(distances)
    
    return (
        locations[closest_idx].copy(),
        float(distances[closest_idx]),
        int(face_indices[closest_idx])
    )


def cast_horizontal_rays(
    mesh_data: MeshData,
    vertical_results: List[VerticalHitResult],
    config: Optional[RoomDetectionConfig] = None
) -> List[VerticalHitResult]:
    """
    Cast horizontal rays to validate room size and filter out wall interior points.
    
    Args:
        mesh_data: Loaded mesh data
        vertical_results: Results from vertical ray casting
        config: Room detection configuration
        
    Returns:
        Filtered list of VerticalHitResult that pass horizontal validation
    """
    if config is None:
        config = RoomDetectionConfig()
    
    mesh = mesh_data.mesh
    
    # 6 horizontal directions (3 opposing pairs)
    # X-axis, Y-axis, and 45-degree diagonal
    sqrt2_2 = 0.7071067811865476
    direction_pairs = [
        (np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),    # X axis
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])),    # Y axis
        (np.array([sqrt2_2, sqrt2_2, 0.0]), np.array([-sqrt2_2, -sqrt2_2, 0.0])),  # 45 deg
    ]
    
    valid_results = []
    
    for result in vertical_results:
        # Cast from slightly below the ceiling to avoid hitting the ceiling itself
        origin = result.top_hit.copy()
        origin[2] -= config.horizontal_ray_offset_cm
        
        is_valid = True
        
        for dir_pos, dir_neg in direction_pairs:
            hit_pos = cast_single_ray(mesh, origin, dir_pos)
            hit_neg = cast_single_ray(mesh, origin, dir_neg)
            
            # Check if both directions hit something
            if hit_pos is None or hit_neg is None:
                # Ray escaped - point is near an opening or outside
                is_valid = False
                break
            
            dist_pos = hit_pos[1]
            dist_neg = hit_neg[1]
            
            # Check for wall interior (too close to walls on both sides)
            if dist_pos < config.wall_thickness_cm or dist_neg < config.wall_thickness_cm:
                # Inside a wall
                is_valid = False
                break
            
            # Check minimum room dimension
            total_span = dist_pos + dist_neg
            if total_span < config.min_room_dimension_cm:
                # Room too small
                is_valid = False
                break
        
        if is_valid:
            valid_results.append(result)
    
    print(f"Horizontal validation: {len(valid_results)}/{len(vertical_results)} points passed")
    
    return valid_results
