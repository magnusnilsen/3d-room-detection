"""
Filtering utilities for room detection candidate points.
"""

import numpy as np
from typing import List, Optional, Dict
from collections import defaultdict

from .types import VerticalHitResult, CandidatePoint, Plane, RoomDetectionConfig
from .mesh_loader import MeshData


def filter_by_height(
    results: List[VerticalHitResult],
    config: Optional[RoomDetectionConfig] = None
) -> List[VerticalHitResult]:
    """
    Filter out points with unrealistic room heights.
    
    Args:
        results: List of vertical hit results
        config: Room detection configuration
        
    Returns:
        Filtered list with only valid room heights
    """
    if config is None:
        config = RoomDetectionConfig()
    
    min_h = config.min_room_height_cm
    max_h = config.max_room_height_cm
    
    filtered = [
        r for r in results
        if min_h <= r.room_height <= max_h
    ]
    
    print(f"Height filtering: {len(filtered)}/{len(results)} points "
          f"have height in [{min_h}, {max_h}] cm")
    
    return filtered


def group_by_floor(
    results: List[VerticalHitResult],
    floor_tolerance_cm: float = 20.0
) -> Dict[float, List[VerticalHitResult]]:
    """
    Group results by floor elevation (bottom hit Z coordinate).
    
    Args:
        results: List of vertical hit results
        floor_tolerance_cm: Tolerance for grouping floor elevations
        
    Returns:
        Dictionary mapping floor Z to list of results on that floor
    """
    if not results:
        return {}
    
    # Get all floor Z values
    floor_zs = np.array([r.floor_z for r in results])
    
    # Find unique floor levels by clustering
    sorted_indices = np.argsort(floor_zs)
    
    floors = {}
    current_floor_z = None
    current_floor_results = []
    
    for idx in sorted_indices:
        z = floor_zs[idx]
        result = results[idx]
        
        if current_floor_z is None:
            current_floor_z = z
            current_floor_results = [result]
        elif abs(z - current_floor_z) <= floor_tolerance_cm:
            current_floor_results.append(result)
        else:
            # New floor level
            avg_z = np.mean([r.floor_z for r in current_floor_results])
            floors[avg_z] = current_floor_results
            current_floor_z = z
            current_floor_results = [result]
    
    # Don't forget the last floor
    if current_floor_results:
        avg_z = np.mean([r.floor_z for r in current_floor_results])
        floors[avg_z] = current_floor_results
    
    print(f"Grouped into {len(floors)} floor level(s): {sorted(floors.keys())}")
    
    return floors


def create_candidate_points(
    results: List[VerticalHitResult],
    mesh_data: MeshData
) -> List[CandidatePoint]:
    """
    Convert VerticalHitResult to CandidatePoint with plane and bounds info.
    
    Args:
        results: List of validated vertical hit results
        mesh_data: Loaded mesh data for extracting face normals and bounds
        
    Returns:
        List of CandidatePoint ready for flood fill
    """
    candidates = []
    
    for result in results:
        # Get face normal for the ceiling plane
        face_normal = mesh_data.get_face_normal(result.top_face_idx)
        
        # Create ceiling plane
        ceiling_plane = Plane(
            point=result.top_hit.copy(),
            normal=face_normal.copy()
        )
        
        # Get bounds for the hit mesh object
        mesh_bounds = mesh_data.get_group_bounds(result.top_mesh_name)
        if mesh_bounds is None:
            # Fallback to overall mesh bounds
            mesh_bounds = mesh_data.mesh.bounds
        
        candidates.append(CandidatePoint(
            sample_point=result.sample_point,
            top_hit_point=result.top_hit,
            bottom_hit_point=result.bottom_hit,
            room_height=result.room_height,
            ceiling_plane=ceiling_plane,
            ceiling_mesh_bounds=mesh_bounds.copy(),
            ceiling_mesh_name=result.top_mesh_name
        ))
    
    return candidates


def filter_duplicate_candidates(
    candidates: List[CandidatePoint],
    xy_tolerance_cm: float = 10.0,
    z_tolerance_cm: float = 20.0
) -> List[CandidatePoint]:
    """
    Remove duplicate candidate points that are very close together.
    
    Keeps the first occurrence in each spatial cluster.
    
    Args:
        candidates: List of candidate points
        xy_tolerance_cm: Tolerance for XY distance
        z_tolerance_cm: Tolerance for Z (floor level) distance
        
    Returns:
        Deduplicated list of candidates
    """
    if len(candidates) <= 1:
        return candidates
    
    # Simple greedy deduplication
    unique = []
    
    for candidate in candidates:
        is_duplicate = False
        
        for existing in unique:
            # Check XY distance
            xy_dist = np.linalg.norm(
                candidate.sample_point[:2] - existing.sample_point[:2]
            )
            z_dist = abs(candidate.floor_z - existing.floor_z)
            
            if xy_dist <= xy_tolerance_cm and z_dist <= z_tolerance_cm:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(candidate)
    
    if len(unique) < len(candidates):
        print(f"Deduplication: {len(unique)}/{len(candidates)} unique candidates")
    
    return unique
