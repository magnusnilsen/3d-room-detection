#!/usr/bin/env python3
"""
Main entry point for 3D room detection from OBJ files.
"""

import sys
import os
from pathlib import Path
from room_detection import detect_rooms


def main():
    """Main function to process OBJ files and detect rooms."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_obj_file_or_directory>")
        print("\nExample:")
        print("  python main.py sample_models/building.obj")
        print("  python main.py sample_models/")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"Error: Path '{input_path}' does not exist.")
        sys.exit(1)
    
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
    
    # Process each OBJ file
    for obj_file in obj_files:
        print(f"\nProcessing: {obj_file}")
        print("-" * 60)
        try:
            rooms = detect_rooms(str(obj_file))
            print(f"Detected {len(rooms)} room(s)")
            for i, room in enumerate(rooms, 1):
                print(f"  Room {i}: {room}")
        except Exception as e:
            print(f"Error processing {obj_file}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
