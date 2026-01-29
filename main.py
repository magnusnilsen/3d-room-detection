#!/usr/bin/env python3
"""
Main entry point for 3D room detection from OBJ files.

Usage:
    python main.py <path_to_obj_file_or_directory>
    python main.py sample_models/building.obj
    python main.py sample_models/
    
Options can be configured by modifying the RoomDetectionConfig parameters.
"""

import sys
import argparse
from pathlib import Path

from room_detection import (
    detect_rooms,
    detect_rooms_by_floor,
    RoomDetectionConfig,
    CandidatePoint,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect rooms from OBJ files (BIM models)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py sample_models/building.obj
  python main.py sample_models/
  python main.py --grid-spacing 50 sample_models/building.obj
        """
    )
    
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to OBJ file or directory containing OBJ files"
    )
    
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=25.0,
        help="Grid sampling spacing in cm (default: 25.0)"
    )
    
    parser.add_argument(
        "--min-height",
        type=float,
        default=200.0,
        help="Minimum room height in cm (default: 200.0)"
    )
    
    parser.add_argument(
        "--max-height",
        type=float,
        default=500.0,
        help="Maximum room height in cm (default: 500.0)"
    )
    
    parser.add_argument(
        "--min-room-size",
        type=float,
        default=100.0,
        help="Minimum room dimension in cm (default: 100.0)"
    )
    
    parser.add_argument(
        "--by-floor",
        action="store_true",
        help="Group results by floor level"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    return parser.parse_args()


def process_obj_file(
    obj_path: Path,
    config: RoomDetectionConfig,
    by_floor: bool = False,
    verbose: bool = True
) -> None:
    """Process a single OBJ file."""
    print(f"\nProcessing: {obj_path}")
    print("-" * 60)
    
    try:
        if by_floor:
            floors = detect_rooms_by_floor(str(obj_path), config, verbose=verbose)
            
            if not floors:
                print("No rooms detected.")
                return
            
            print(f"\nResults by floor:")
            for floor_z in sorted(floors.keys()):
                candidates = floors[floor_z]
                print(f"\n  Floor at Z = {floor_z:.1f} cm:")
                print(f"    {len(candidates)} candidate point(s)")
                
                if candidates and verbose:
                    # Show sample candidate info
                    for i, c in enumerate(candidates[:3]):
                        print(f"      [{i+1}] Height: {c.room_height:.1f}cm, "
                              f"Mesh: {c.ceiling_mesh_name}")
                    if len(candidates) > 3:
                        print(f"      ... and {len(candidates) - 3} more")
        else:
            candidates = detect_rooms(str(obj_path), config, verbose=verbose)
            
            if not candidates:
                print("No rooms detected.")
                return
            
            print(f"\nDetected {len(candidates)} room candidate(s)")
            
            if verbose:
                # Show details for first few candidates
                print("\nSample candidates:")
                for i, c in enumerate(candidates[:5]):
                    print(f"  [{i+1}] Position: ({c.sample_point[0]:.1f}, "
                          f"{c.sample_point[1]:.1f}, {c.sample_point[2]:.1f})")
                    print(f"       Height: {c.room_height:.1f}cm, "
                          f"Floor Z: {c.floor_z:.1f}cm, "
                          f"Ceiling: {c.ceiling_mesh_name}")
                
                if len(candidates) > 5:
                    print(f"  ... and {len(candidates) - 5} more candidates")
    
    except Exception as e:
        print(f"Error processing {obj_path}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    args = parse_args()
    
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"Error: Path '{input_path}' does not exist.")
        sys.exit(1)
    
    # Build configuration from arguments
    config = RoomDetectionConfig(
        grid_spacing_cm=args.grid_spacing,
        min_room_height_cm=args.min_height,
        max_room_height_cm=args.max_height,
        min_room_dimension_cm=args.min_room_size,
    )
    
    verbose = not args.quiet
    
    # Collect OBJ files
    obj_files = []
    if input_path.is_file():
        if input_path.suffix.lower() == '.obj':
            obj_files.append(input_path)
        else:
            print(f"Error: '{input_path}' is not an OBJ file.")
            sys.exit(1)
    elif input_path.is_dir():
        obj_files = list(input_path.glob("*.obj"))
        if not obj_files:
            print(f"Warning: No OBJ files found in '{input_path}'.")
            sys.exit(1)
    
    # Process each file
    for obj_file in obj_files:
        process_obj_file(obj_file, config, by_floor=args.by_floor, verbose=verbose)


if __name__ == "__main__":
    main()
