"""
Flood fill algorithm for room polygon extraction.

This module implements the core flood fill algorithm that:
1. Takes ceiling hit points with their planes
2. Intersects mesh triangles with each plane to get boundary segments
3. Creates closed polygons from the segments
4. Unions overlapping polygons with a small buffer
5. Extracts separate room islands
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from shapely.geometry import Polygon, MultiPolygon, LineString, Point as ShapelyPoint
from shapely.ops import polygonize, unary_union
from shapely.validation import make_valid

from .types import CandidatePoint, Plane, RoomDetectionConfig
from .mesh_loader import MeshData


# Tolerance values
PLANE_TOLERANCE = 0.5  # Distance tolerance for plane intersection (cm)
SEGMENT_SNAP_PRECISION = 3  # Decimal places for coordinate snapping
BUFFER_AMOUNT = 0.5  # Buffer for wall thickness (cm)
SIMPLIFY_TOLERANCE = 2.0  # Tolerance for polygon simplification (cm)
MIN_ROOM_AREA = 10000  # Minimum room area in cm² (1 m²)


@dataclass
class Room:
    """Represents a detected room."""
    id: int
    polygon_2d: Polygon  # 2D polygon in plane coordinates
    polygon_3d: np.ndarray  # 3D vertices of the room boundary
    floor_z: float  # Z coordinate of the floor
    ceiling_z: float  # Z coordinate of the ceiling
    height: float  # Room height
    area: float  # Floor area in cm²
    centroid: np.ndarray  # 3D centroid of the room
    
    @property
    def area_m2(self) -> float:
        """Area in square meters."""
        return self.area / 10000  # cm² to m²


def create_plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create two orthonormal basis vectors on a plane given its normal.
    
    Args:
        normal: Unit normal vector of the plane
        
    Returns:
        Tuple of (u, v) basis vectors
    """
    normal = normal / np.linalg.norm(normal)
    
    # Choose a vector not parallel to normal
    if abs(normal[2]) > 0.9:
        # Normal is mostly vertical, use X axis
        ref = np.array([1.0, 0.0, 0.0])
    else:
        # Use Z axis
        ref = np.array([0.0, 0.0, 1.0])
    
    # Create orthonormal basis
    u = np.cross(normal, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    
    return u, v


def project_to_plane(points_3d: np.ndarray, plane: Plane) -> np.ndarray:
    """
    Project 3D points onto a 2D plane coordinate system.
    
    Args:
        points_3d: Nx3 array of 3D points
        plane: The plane to project onto
        
    Returns:
        Nx2 array of 2D coordinates
    """
    u, v = create_plane_basis(plane.normal)
    origin = plane.point
    
    # Translate to plane origin and project
    translated = points_3d - origin
    
    if translated.ndim == 1:
        return np.array([np.dot(translated, u), np.dot(translated, v)])
    
    return np.column_stack([
        np.dot(translated, u),
        np.dot(translated, v)
    ])


def unproject_from_plane(points_2d: np.ndarray, plane: Plane) -> np.ndarray:
    """
    Convert 2D plane coordinates back to 3D.
    
    Args:
        points_2d: Nx2 array of 2D coordinates
        plane: The plane to unproject from
        
    Returns:
        Nx3 array of 3D points
    """
    u, v = create_plane_basis(plane.normal)
    origin = plane.point
    
    if points_2d.ndim == 1:
        return origin + points_2d[0] * u + points_2d[1] * v
    
    return origin + np.outer(points_2d[:, 0], u) + np.outer(points_2d[:, 1], v)


def get_triangle_plane_intersection(
    triangle: np.ndarray,
    plane: Plane
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Get the line segment where a triangle intersects a plane.
    
    Args:
        triangle: 3x3 array of triangle vertices
        plane: The plane to intersect with
        
    Returns:
        Tuple of (point1, point2) as 2D coordinates, or None if no intersection
    """
    normal = plane.normal
    d = -np.dot(normal, plane.point)  # Plane equation: n·x + d = 0
    
    # Compute signed distances from vertices to plane
    distances = np.dot(triangle, normal) + d
    
    # Check if triangle intersects plane
    if np.all(distances >= -PLANE_TOLERANCE) or np.all(distances <= PLANE_TOLERANCE):
        # All vertices on same side - check if any edge is on the plane
        min_abs_dist = np.min(np.abs(distances))
        if min_abs_dist > PLANE_TOLERANCE:
            return None
    
    intersection_points = []
    
    # Check each edge
    for i in range(3):
        p0 = triangle[i]
        p1 = triangle[(i + 1) % 3]
        d0 = distances[i]
        d1 = distances[(i + 1) % 3]
        
        # Check if edge is on the plane (both endpoints close to plane)
        if abs(d0) < PLANE_TOLERANCE and abs(d1) < PLANE_TOLERANCE:
            # Edge lies on plane - add both endpoints
            pt0_2d = project_to_plane(p0, plane)
            pt1_2d = project_to_plane(p1, plane)
            return (pt0_2d, pt1_2d)
        
        # Check if edge crosses plane
        if (d0 > PLANE_TOLERANCE and d1 < -PLANE_TOLERANCE) or \
           (d0 < -PLANE_TOLERANCE and d1 > PLANE_TOLERANCE):
            # Compute intersection point
            t = d0 / (d0 - d1)
            intersection_pt = p0 + t * (p1 - p0)
            intersection_points.append(intersection_pt)
        elif abs(d0) < PLANE_TOLERANCE:
            # p0 is on the plane
            intersection_points.append(p0.copy())
        elif abs(d1) < PLANE_TOLERANCE:
            # p1 is on the plane
            intersection_points.append(p1.copy())
    
    # Remove duplicates
    if len(intersection_points) >= 2:
        unique_points = [intersection_points[0]]
        for pt in intersection_points[1:]:
            if np.linalg.norm(pt - unique_points[-1]) > PLANE_TOLERANCE:
                unique_points.append(pt)
        intersection_points = unique_points
    
    if len(intersection_points) == 2:
        pt0_2d = project_to_plane(intersection_points[0], plane)
        pt1_2d = project_to_plane(intersection_points[1], plane)
        
        # Skip degenerate segments
        if np.linalg.norm(pt0_2d - pt1_2d) < PLANE_TOLERANCE:
            return None
        
        return (pt0_2d, pt1_2d)
    
    return None


def get_intersection_segments(
    mesh_data: MeshData,
    plane: Plane,
    bounds: Optional[np.ndarray] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Get all line segments where mesh triangles intersect the plane.
    
    Args:
        mesh_data: The mesh data
        plane: The plane to intersect with
        bounds: Optional bounding box to limit search [min, max]
        
    Returns:
        List of (point1, point2) tuples in 2D plane coordinates
    """
    mesh = mesh_data.mesh
    vertices = mesh.vertices
    faces = mesh.faces
    
    segments = []
    
    # Get plane parameters
    normal = plane.normal
    d = -np.dot(normal, plane.point)
    
    # Compute distances from all vertices to plane
    vertex_distances = np.dot(vertices, normal) + d
    
    for face in faces:
        # Get triangle vertices
        triangle = vertices[face]
        dists = vertex_distances[face]
        
        # Quick rejection: if all vertices are on same side and far from plane
        if np.all(dists > PLANE_TOLERANCE) or np.all(dists < -PLANE_TOLERANCE):
            continue
        
        # Optional: check bounding box
        if bounds is not None:
            tri_min = triangle.min(axis=0)
            tri_max = triangle.max(axis=0)
            if np.any(tri_max < bounds[0]) or np.any(tri_min > bounds[1]):
                continue
        
        # Get intersection
        segment = get_triangle_plane_intersection(triangle, plane)
        if segment is not None:
            segments.append(segment)
    
    return segments


def snap_coordinate(value: float, precision: int = SEGMENT_SNAP_PRECISION) -> float:
    """Snap a coordinate to a fixed precision grid."""
    factor = 10 ** precision
    return round(value * factor) / factor


def deduplicate_segments(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    tolerance: float = 0.1
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Remove duplicate segments."""
    if not segments:
        return []
    
    unique = []
    
    for seg in segments:
        p0, p1 = seg
        is_duplicate = False
        
        for u_seg in unique:
            u0, u1 = u_seg
            # Check both orientations
            if (np.linalg.norm(p0 - u0) < tolerance and np.linalg.norm(p1 - u1) < tolerance) or \
               (np.linalg.norm(p0 - u1) < tolerance and np.linalg.norm(p1 - u0) < tolerance):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(seg)
    
    return unique


def segments_to_polygons(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    min_area: float = 10000  # 1m² minimum
) -> List[Polygon]:
    """
    Convert line segments to closed polygons using Shapely's polygonize.
    
    Uses noding (unary_union) to properly split segments at intersections
    before polygonizing, which is essential for BIM models where walls
    intersect at various points.
    
    Args:
        segments: List of (point1, point2) tuples
        min_area: Minimum polygon area in cm²
        
    Returns:
        List of Shapely Polygon objects
    """
    if len(segments) < 3:
        return []
    
    # Snap coordinates for better topology
    lines = []
    for p0, p1 in segments:
        snapped_p0 = (snap_coordinate(p0[0]), snap_coordinate(p0[1]))
        snapped_p1 = (snap_coordinate(p1[0]), snap_coordinate(p1[1]))
        
        # Skip degenerate segments
        if abs(snapped_p0[0] - snapped_p1[0]) < 1e-6 and abs(snapped_p0[1] - snapped_p1[1]) < 1e-6:
            continue
        
        lines.append(LineString([snapped_p0, snapped_p1]))
    
    if len(lines) < 3:
        return []
    
    # Node the lines - this splits them at intersections
    # Critical for BIM models where walls intersect
    try:
        noded = unary_union(lines)
    except Exception:
        return []
    
    # Use Shapely's polygonize to create polygons from noded lines
    try:
        polygons = list(polygonize(noded))
    except Exception:
        return []
    
    # Filter out invalid or tiny polygons
    valid_polygons = []
    for poly in polygons:
        if poly.is_valid and poly.area >= min_area:
            valid_polygons.append(poly)
    
    return valid_polygons


def get_all_floor_segments(
    mesh_data: MeshData,
    floor_z: float,
    ceiling_z: float,
    bounds_2d: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Get all wall intersection segments at ceiling level.
    
    We use the ceiling level because it has fewer openings (doors/windows).
    
    Args:
        mesh_data: The mesh data
        floor_z: Floor Z coordinate
        ceiling_z: Ceiling Z coordinate  
        bounds_2d: 2D XY bounds [[min_x, min_y], [max_x, max_y]]
        
    Returns:
        List of (point1, point2) tuples in 2D XY coordinates
    """
    # Create a horizontal plane at ceiling level (slightly below to avoid hitting ceiling itself)
    plane = Plane(
        point=np.array([0, 0, ceiling_z - 5]),  # 5cm below ceiling
        normal=np.array([0, 0, -1])
    )
    
    # Create 3D bounds for segment search
    bounds_3d = np.array([
        [bounds_2d[0][0], bounds_2d[0][1], floor_z],
        [bounds_2d[1][0], bounds_2d[1][1], ceiling_z]
    ])
    
    mesh = mesh_data.mesh
    vertices = mesh.vertices
    faces = mesh.faces
    
    segments = []
    
    # For horizontal plane, we just need to find where triangles cross the Z level
    target_z = ceiling_z - 5
    
    for face in faces:
        triangle = vertices[face]
        
        # Quick bounding box check
        tri_min = triangle.min(axis=0)
        tri_max = triangle.max(axis=0)
        
        # Skip if triangle doesn't cross our Z level
        if tri_max[2] < target_z - PLANE_TOLERANCE or tri_min[2] > target_z + PLANE_TOLERANCE:
            continue
        
        # Skip if outside XY bounds
        if tri_max[0] < bounds_2d[0][0] or tri_min[0] > bounds_2d[1][0]:
            continue
        if tri_max[1] < bounds_2d[0][1] or tri_min[1] > bounds_2d[1][1]:
            continue
        
        # Find intersection with Z plane
        intersection_points = []
        
        for i in range(3):
            p0 = triangle[i]
            p1 = triangle[(i + 1) % 3]
            
            z0 = p0[2] - target_z
            z1 = p1[2] - target_z
            
            # Both on plane
            if abs(z0) < PLANE_TOLERANCE and abs(z1) < PLANE_TOLERANCE:
                # Edge lies on plane
                segments.append((p0[:2].copy(), p1[:2].copy()))
                continue
            
            # Check if edge crosses plane
            if (z0 > PLANE_TOLERANCE and z1 < -PLANE_TOLERANCE) or \
               (z0 < -PLANE_TOLERANCE and z1 > PLANE_TOLERANCE):
                t = z0 / (z0 - z1)
                intersection_pt = p0 + t * (p1 - p0)
                intersection_points.append(intersection_pt[:2])
            elif abs(z0) < PLANE_TOLERANCE:
                intersection_points.append(p0[:2].copy())
            elif abs(z1) < PLANE_TOLERANCE:
                intersection_points.append(p1[:2].copy())
        
        # Remove duplicate points
        if len(intersection_points) >= 2:
            unique = [intersection_points[0]]
            for pt in intersection_points[1:]:
                if np.linalg.norm(pt - unique[-1]) > PLANE_TOLERANCE:
                    unique.append(pt)
            if len(unique) == 2:
                # Check segment isn't degenerate
                if np.linalg.norm(unique[0] - unique[1]) > PLANE_TOLERANCE:
                    segments.append((unique[0], unique[1]))
    
    return segments


def flood_fill_for_candidate(
    candidate: CandidatePoint,
    mesh_data: MeshData,
    search_radius: float = 500.0
) -> List[Polygon]:
    """
    Perform flood fill for a single candidate point.
    
    Args:
        candidate: The candidate point with ceiling plane info
        mesh_data: The mesh data
        search_radius: Radius around the hit point to search (cm)
        
    Returns:
        List of polygons found at this ceiling plane
    """
    plane = candidate.ceiling_plane
    hit_point = candidate.top_hit_point
    
    # Create bounding box around the hit point
    bounds = np.array([
        hit_point - search_radius,
        hit_point + search_radius
    ])
    
    # Get intersection segments
    segments = get_intersection_segments(mesh_data, plane, bounds)
    
    if len(segments) < 3:
        return []
    
    # Deduplicate
    segments = deduplicate_segments(segments)
    
    # Convert to polygons
    polygons = segments_to_polygons(segments)
    
    # Filter to polygons containing the hit point
    hit_point_2d = project_to_plane(hit_point, plane)
    hit_shapely = ShapelyPoint(hit_point_2d[0], hit_point_2d[1])
    
    containing_polygons = [p for p in polygons if p.contains(hit_shapely)]
    
    return containing_polygons


def group_candidates_by_level(
    candidates: List[CandidatePoint],
    tolerance: float = 30.0
) -> Dict[Tuple[float, float], List[CandidatePoint]]:
    """
    Group candidates by their (floor_z, ceiling_z) level using union-find.
    
    Candidates are grouped together if they share either:
    - The same floor Z level (within tolerance)
    - The same ceiling Z level (within tolerance)
    
    This ensures all rooms on the same physical floor are processed together,
    even if there are slight variations in floor or ceiling heights.
    
    Args:
        candidates: List of candidate points
        tolerance: Z-level tolerance in cm
        
    Returns:
        Dictionary mapping (floor_z, ceiling_z) to list of candidates
    """
    if not candidates:
        return {}
    
    n = len(candidates)
    
    # Union-Find data structure
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Group candidates that share floor OR ceiling levels
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = candidates[i], candidates[j]
            
            # Same floor level?
            if abs(ci.floor_z - cj.floor_z) < tolerance:
                union(i, j)
            # Same ceiling level?
            elif abs(ci.ceiling_z - cj.ceiling_z) < tolerance:
                union(i, j)
    
    # Collect groups
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    # Convert to (floor_z, ceiling_z) keyed dictionary
    # Use the median floor_z and ceiling_z for each group
    result: Dict[Tuple[float, float], List[CandidatePoint]] = {}
    
    for indices in groups.values():
        group_candidates = [candidates[i] for i in indices]
        
        # Use median values for the level key
        floor_z = float(np.median([c.floor_z for c in group_candidates]))
        ceiling_z = float(np.median([c.ceiling_z for c in group_candidates]))
        
        result[(floor_z, ceiling_z)] = group_candidates
    
    return result


def extract_rooms(
    candidates: List[CandidatePoint],
    mesh_data: MeshData,
    config: Optional[RoomDetectionConfig] = None,
    buffer_cm: float = BUFFER_AMOUNT,
    simplify_tolerance: float = SIMPLIFY_TOLERANCE,
    verbose: bool = True
) -> List[Room]:
    """
    Extract room polygons from candidate points.
    
    This function:
    1. Groups candidates by (floor, ceiling) level using union-find
    2. Gets all wall intersection segments at ceiling level for each floor
    3. Creates room polygons using Shapely's polygonize
    4. Filters to polygons containing candidate points (valid rooms)
    5. Ensures no overlapping rooms on the same floor
    
    Args:
        candidates: List of candidate points from ray casting
        mesh_data: The mesh data
        config: Room detection configuration
        buffer_cm: Buffer for wall thickness
        simplify_tolerance: Tolerance for simplifying polygons
        verbose: Print progress information
        
    Returns:
        List of Room objects
    """
    if not candidates:
        return []
    
    if verbose:
        print(f"\n{'='*60}")
        print("Room Extraction (Flood Fill)")
        print(f"{'='*60}")
        print(f"\nProcessing {len(candidates)} candidate points...")
    
    # Group candidates by (floor_z, ceiling_z) level
    level_groups = group_candidates_by_level(candidates, tolerance=30.0)
    
    if verbose:
        print(f"Found {len(level_groups)} floor level(s)")
    
    all_rooms = []
    room_id = 0
    
    # Track which XY areas have been assigned to rooms (to prevent overlaps)
    claimed_areas: List[Polygon] = []
    
    for (floor_z, ceiling_z), floor_candidates in sorted(level_groups.items()):
        height = ceiling_z - floor_z
        
        if verbose:
            print(f"\n  Level: floor={floor_z:.1f}, ceiling={ceiling_z:.1f}, height={height:.0f} cm")
            print(f"    {len(floor_candidates)} candidates")
        
        # Get bounds from candidates
        all_points = np.array([c.sample_point for c in floor_candidates])
        xy_min = all_points[:, :2].min(axis=0) - 200  # 2m buffer
        xy_max = all_points[:, :2].max(axis=0) + 200
        bounds_2d = np.array([xy_min, xy_max])
        
        # Get all wall segments at ceiling level
        if verbose:
            print(f"    Getting wall segments at Z={ceiling_z:.1f}...")
        
        segments = get_all_floor_segments(mesh_data, floor_z, ceiling_z, bounds_2d)
        
        if verbose:
            print(f"    Found {len(segments)} wall segments")
        
        if len(segments) < 3:
            continue
        
        # Deduplicate segments
        segments = deduplicate_segments(segments, tolerance=1.0)
        
        if verbose:
            print(f"    After dedup: {len(segments)} segments")
        
        # Create polygons from segments
        polygons = segments_to_polygons(segments)
        
        if verbose:
            print(f"    Created {len(polygons)} polygons from segments")
        
        if not polygons:
            continue
        
        # Find which polygons contain candidate points (these are valid rooms)
        candidate_points_2d = [ShapelyPoint(c.sample_point[0], c.sample_point[1]) 
                              for c in floor_candidates]
        
        room_polygons = []
        used_candidates = set()
        
        for poly in polygons:
            if poly.area < MIN_ROOM_AREA:
                continue
            
            # Check if this polygon overlaps with already claimed areas
            is_overlapping = False
            for claimed in claimed_areas:
                intersection = poly.intersection(claimed)
                # If more than 10% of the polygon overlaps, skip it
                if intersection.area > poly.area * 0.1:
                    is_overlapping = True
                    break
            
            if is_overlapping:
                continue
            
            # Check if any candidate is inside this polygon
            has_candidate = False
            for i, pt in enumerate(candidate_points_2d):
                if i not in used_candidates and poly.contains(pt):
                    has_candidate = True
                    used_candidates.add(i)
                    break
            
            if has_candidate:
                # Simplify the polygon
                simplified = poly.simplify(simplify_tolerance, preserve_topology=True)
                if simplified.is_valid and simplified.area > MIN_ROOM_AREA:
                    room_polygons.append(simplified)
                    claimed_areas.append(simplified)
        
        if verbose:
            print(f"    Validated {len(room_polygons)} room(s) with candidates")
        
        # Create Room objects
        for poly in room_polygons:
            room_id += 1
            
            # Get 3D boundary at floor level
            coords_2d = np.array(poly.exterior.coords)
            coords_3d = np.column_stack([
                coords_2d[:, 0],
                coords_2d[:, 1],
                np.full(len(coords_2d), floor_z)
            ])
            
            centroid_2d = np.array([poly.centroid.x, poly.centroid.y])
            centroid_3d = np.array([centroid_2d[0], centroid_2d[1], floor_z + height / 2])
            
            room = Room(
                id=room_id,
                polygon_2d=poly,
                polygon_3d=coords_3d,
                floor_z=floor_z,
                ceiling_z=ceiling_z,
                height=height,
                area=poly.area,
                centroid=centroid_3d
            )
            all_rooms.append(room)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Extracted {len(all_rooms)} room(s) total")
        for room in all_rooms:
            print(f"  Room {room.id}: {room.area_m2:.2f} m², height {room.height:.0f} cm")
        print(f"{'='*60}\n")
    
    return all_rooms
