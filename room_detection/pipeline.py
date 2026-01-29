"""
Room detection pipeline - main algorithm entry point.

This module provides the high-level detect_rooms() function that orchestrates
the entire room detection pipeline from OBJ loading through candidate point
generation ready for flood fill.
"""

from typing import List, Optional, Dict
import time

from .types import RoomDetectionConfig, CandidatePoint
from .mesh_loader import MeshData, load_building_mesh
from .grid_sampler import generate_sample_grid
from .ray_caster import cast_vertical_rays, cast_horizontal_rays
from .point_filter import filter_by_height, create_candidate_points, group_by_floor


def detect_rooms(
    obj_path: str,
    config: Optional[RoomDetectionConfig] = None,
    verbose: bool = True
) -> List[CandidatePoint]:
    """
    Detect room candidate points from an OBJ file.
    
    This function runs the complete preprocessing pipeline:
    1. Load mesh with group preservation
    2. Generate sample grid (multi-floor aware)
    3. Cast vertical rays to find interior points
    4. Filter by room height
    5. Cast horizontal rays to validate room dimensions
    6. Create candidate points ready for flood fill
    
    Args:
        obj_path: Path to the OBJ file
        config: Room detection configuration (uses defaults if None)
        verbose: Print progress information
        
    Returns:
        List of CandidatePoint objects ready for flood fill processing
    """
    if config is None:
        config = RoomDetectionConfig()
    
    start_time = time.time()
    
    # Step 1: Load mesh
    if verbose:
        print(f"\n{'='*60}")
        print(f"Room Detection Pipeline")
        print(f"{'='*60}")
        print(f"\nStep 1: Loading mesh...")
    
    mesh_data = load_building_mesh(obj_path)
    
    # Step 2: Generate sample grid
    if verbose:
        print(f"\nStep 2: Generating sample grid (spacing: {config.grid_spacing_cm}cm)...")
    
    sample_points = generate_sample_grid(mesh_data.mesh.bounds, config)
    
    # Step 3: Vertical ray casting
    if verbose:
        print(f"\nStep 3: Casting vertical rays...")
    
    vertical_results = cast_vertical_rays(mesh_data, sample_points, config)
    
    if not vertical_results:
        if verbose:
            print("No interior points found!")
        return []
    
    # Step 4: Height filtering
    if verbose:
        print(f"\nStep 4: Filtering by height ({config.min_room_height_cm}-{config.max_room_height_cm}cm)...")
    
    height_filtered = filter_by_height(vertical_results, config)
    
    if not height_filtered:
        if verbose:
            print("No points passed height filtering!")
        return []
    
    # Step 5: Horizontal ray casting
    if verbose:
        print(f"\nStep 5: Validating with horizontal rays (min room: {config.min_room_dimension_cm}cm)...")
    
    horizontal_validated = cast_horizontal_rays(mesh_data, height_filtered, config)
    
    if not horizontal_validated:
        if verbose:
            print("No points passed horizontal validation!")
        return []
    
    # Step 6: Create candidate points
    if verbose:
        print(f"\nStep 6: Creating candidate points for flood fill...")
    
    candidates = create_candidate_points(horizontal_validated, mesh_data)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Pipeline complete in {elapsed:.2f}s")
        print(f"Result: {len(candidates)} candidate points ready for flood fill")
        
        # Show floor breakdown
        floors = group_by_floor(horizontal_validated)
        print(f"\nFloor breakdown:")
        for floor_z in sorted(floors.keys()):
            print(f"  Floor at Z={floor_z:.1f}cm: {len(floors[floor_z])} points")
        print(f"{'='*60}\n")
    
    return candidates


def detect_rooms_by_floor(
    obj_path: str,
    config: Optional[RoomDetectionConfig] = None,
    verbose: bool = True
) -> Dict[float, List[CandidatePoint]]:
    """
    Detect room candidate points grouped by floor.
    
    Same as detect_rooms() but returns results organized by floor elevation.
    
    Args:
        obj_path: Path to the OBJ file
        config: Room detection configuration
        verbose: Print progress information
        
    Returns:
        Dictionary mapping floor Z coordinate to list of CandidatePoint
    """
    if config is None:
        config = RoomDetectionConfig()
    
    # Run the pipeline
    mesh_data = load_building_mesh(obj_path)
    sample_points = generate_sample_grid(mesh_data.mesh.bounds, config)
    vertical_results = cast_vertical_rays(mesh_data, sample_points, config)
    height_filtered = filter_by_height(vertical_results, config)
    horizontal_validated = cast_horizontal_rays(mesh_data, height_filtered, config)
    
    # Group by floor
    floors = group_by_floor(horizontal_validated)
    
    # Convert each floor's results to candidates
    result = {}
    for floor_z, floor_results in floors.items():
        result[floor_z] = create_candidate_points(floor_results, mesh_data)
    
    if verbose:
        print(f"\nDetected {len(result)} floor(s) with candidates:")
        for floor_z in sorted(result.keys()):
            print(f"  Floor Z={floor_z:.1f}cm: {len(result[floor_z])} candidates")
    
    return result


# For backward compatibility with the original simple interface
def detect_rooms_simple(obj_path: str) -> List[dict]:
    """
    Simple interface returning basic room information as dictionaries.
    
    This is the legacy interface for compatibility.
    """
    candidates = detect_rooms(obj_path, verbose=False)
    
    return [{
        "id": i + 1,
        "sample_point": c.sample_point.tolist(),
        "ceiling_point": c.top_hit_point.tolist(),
        "floor_point": c.bottom_hit_point.tolist(),
        "height": c.room_height,
        "ceiling_mesh": c.ceiling_mesh_name
    } for i, c in enumerate(candidates)]
