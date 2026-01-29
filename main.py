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
from typing import List, Optional

from room_detection import (
    detect_rooms,
    detect_rooms_with_mesh,
    detect_rooms_by_floor,
    RoomDetectionConfig,
    CandidatePoint,
    MeshData,
    visualize_results,
    extract_rooms,
    Room,
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
  python main.py --visualize sample_models/building.obj
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
        default=100.0,
        help="Grid sampling spacing in cm (default: 100.0)"
    )
    
    parser.add_argument(
        "--min-height",
        type=float,
        default=100.0,
        help="Minimum room height in cm (default: 200.0)"
    )
    
    parser.add_argument(
        "--max-height",
        type=float,
        default=1000.0,
        help="Maximum room height in cm (default: 1000.0)"
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
        "--visualize",
        action="store_true",
        help="Open interactive 3D viewer to visualize results"
    )
    
    parser.add_argument(
        "--marker-size",
        type=float,
        default=20.0,
        help="Size of ceiling markers in cm (default: 20.0)"
    )
    
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.3,
        help="Building mesh opacity for visualization (0-1, default: 0.3)"
    )
    
    parser.add_argument(
        "--solid",
        action="store_true",
        help="Show building as solid mesh instead of wireframe (default: wireframe)"
    )
    
    parser.add_argument(
        "--extract-rooms",
        action="store_true",
        help="Run flood fill to extract room polygons"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    return parser.parse_args()


def visualize_rooms(
    mesh_data: MeshData,
    rooms: List[Room],
    candidates: List[CandidatePoint],
    wireframe: bool = True,
    opacity: float = 0.3
) -> None:
    """Visualize extracted rooms with Open3D.
    
    Shows rooms as extruded 3D polygons with floor, ceiling, and walls.
    """
    try:
        import open3d as o3d
        import numpy as np
        from shapely.geometry import Polygon as ShapelyPolygon
    except ImportError:
        print("Open3D not available for visualization")
        return
    
    geometries = []
    
    # Add building wireframe
    building_mesh = mesh_data.mesh
    o3d_building = o3d.geometry.TriangleMesh()
    o3d_building.vertices = o3d.utility.Vector3dVector(building_mesh.vertices.astype(np.float64))
    o3d_building.triangles = o3d.utility.Vector3iVector(building_mesh.faces.astype(np.int32))
    o3d_building.compute_vertex_normals()
    
    if wireframe:
        wireframe_lines = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_building)
        wireframe_lines.paint_uniform_color([0.3, 0.3, 0.35])
        geometries.append(wireframe_lines)
    else:
        o3d_building.paint_uniform_color([0.6, 0.6, 0.6])
        geometries.append(o3d_building)
    
    # Color palette for rooms (bright, distinct colors)
    colors = [
        [1.0, 0.3, 0.3],  # Red
        [0.3, 1.0, 0.3],  # Green
        [0.3, 0.3, 1.0],  # Blue
        [1.0, 1.0, 0.3],  # Yellow
        [1.0, 0.3, 1.0],  # Magenta
        [0.3, 1.0, 1.0],  # Cyan
        [1.0, 0.6, 0.2],  # Orange
        [0.6, 0.3, 1.0],  # Purple
        [0.3, 1.0, 0.6],  # Mint
        [1.0, 0.3, 0.6],  # Pink
    ]
    
    # Add room polygons as extruded 3D shapes
    for i, room in enumerate(rooms):
        color = colors[i % len(colors)]
        
        # Get floor polygon coordinates
        coords = np.array(room.polygon_2d.exterior.coords[:-1])  # Remove duplicate last point
        
        if len(coords) < 3:
            continue
        
        n = len(coords)
        floor_z = room.floor_z
        ceiling_z = room.floor_z + room.height
        
        # Create floor and ceiling vertices
        floor_verts = np.column_stack([coords[:, 0], coords[:, 1], np.full(n, floor_z)])
        ceiling_verts = np.column_stack([coords[:, 0], coords[:, 1], np.full(n, ceiling_z)])
        all_verts = np.vstack([floor_verts, ceiling_verts])
        
        # Create triangles for the extruded shape
        triangles = []
        
        # Floor triangles (fan triangulation, facing down)
        for j in range(1, n - 1):
            triangles.append([0, j + 1, j])  # Reversed winding for downward normal
        
        # Ceiling triangles (fan triangulation, facing up)
        for j in range(1, n - 1):
            triangles.append([n, n + j, n + j + 1])
        
        # Wall triangles (quads split into two triangles each)
        for j in range(n):
            next_j = (j + 1) % n
            # Wall quad: floor[j], floor[next_j], ceiling[next_j], ceiling[j]
            # Triangle 1: floor[j], floor[next_j], ceiling[j]
            triangles.append([j, next_j, n + j])
            # Triangle 2: floor[next_j], ceiling[next_j], ceiling[j]
            triangles.append([next_j, n + next_j, n + j])
        
        triangles = np.array(triangles, dtype=np.int32)
        
        # Create the extruded room mesh
        room_mesh = o3d.geometry.TriangleMesh()
        room_mesh.vertices = o3d.utility.Vector3dVector(all_verts)
        room_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        room_mesh.compute_vertex_normals()
        room_mesh.paint_uniform_color(color)
        geometries.append(room_mesh)
        
        # Add edge outline for better visibility
        edge_lines = []
        edge_points = all_verts.tolist()
        
        # Floor edges
        for j in range(n):
            edge_lines.append([j, (j + 1) % n])
        # Ceiling edges
        for j in range(n):
            edge_lines.append([n + j, n + (j + 1) % n])
        # Vertical edges
        for j in range(n):
            edge_lines.append([j, n + j])
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(edge_points)
        line_set.lines = o3d.utility.Vector2iVector(edge_lines)
        # Darker outline
        outline_color = [c * 0.5 for c in color]
        line_set.paint_uniform_color(outline_color)
        geometries.append(line_set)
    
    print(f"\nVisualizing {len(rooms)} extruded room(s)")
    for i, room in enumerate(rooms):
        color_name = ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'Orange', 'Purple', 'Mint', 'Pink'][i % 10]
        print(f"  Room {room.id} ({color_name}): {room.area_m2:.1f} m², height {room.height:.0f} cm")
    print("\nControls: Left-drag=Rotate, Scroll=Zoom, Middle-drag=Pan, Q=Quit")
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Room Detection Results", width=1400, height=900)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.15])
    opt.mesh_show_back_face = True
    opt.line_width = 2.0
    
    # Set camera - isometric view to see 3D extrusion
    ctr = vis.get_view_control()
    bounds = building_mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    ctr.set_lookat(center)
    ctr.set_zoom(0.5)
    ctr.set_front([0.5, -0.5, 0.7])  # Isometric view
    ctr.set_up([0, 0, 1])
    
    vis.run()
    vis.destroy_window()


def process_obj_file(
    obj_path: Path,
    config: RoomDetectionConfig,
    by_floor: bool = False,
    visualize: bool = False,
    extract_rooms_flag: bool = False,
    marker_size: float = 20.0,
    opacity: float = 0.3,
    wireframe: bool = True,
    verbose: bool = True
) -> None:
    """Process a single OBJ file."""
    print(f"\nProcessing: {obj_path}")
    print("-" * 60)
    
    try:
        if visualize or extract_rooms_flag:
            # Use the version that returns mesh data for visualization/extraction
            candidates, mesh_data = detect_rooms_with_mesh(
                str(obj_path), config, verbose=verbose
            )
            
            if not candidates:
                print("No rooms detected.")
                return
            
            print(f"\nDetected {len(candidates)} room candidate(s)")
            
            if extract_rooms_flag:
                # Run flood fill to extract room polygons
                rooms = extract_rooms(candidates, mesh_data, config, verbose=verbose)
                
                if rooms and visualize:
                    # Show visualization with rooms
                    visualize_rooms(mesh_data, rooms, candidates, wireframe, opacity)
            elif visualize:
                # Show visualization without room extraction
                visualize_results(
                    mesh_data=mesh_data,
                    candidates=candidates,
                    building_opacity=opacity,
                    marker_size_cm=marker_size,
                    show_points=True,
                    show_markers=True,
                    wireframe=wireframe
                )
        
        elif by_floor:
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
        process_obj_file(
            obj_file, 
            config, 
            by_floor=args.by_floor,
            visualize=args.visualize,
            extract_rooms_flag=args.extract_rooms,
            marker_size=args.marker_size,
            opacity=args.opacity,
            wireframe=not args.solid,
            verbose=verbose
        )


if __name__ == "__main__":
    main()
